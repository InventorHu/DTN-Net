import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .components import TextEncoder, PointNetPlusPlus, PoseNet
import math

# Global text encoder (CLIP, frozen weights)
text_encoder = TextEncoder(device=torch.device('cuda'))

# ========== DenseFusion Module for Affordance Prediction Only ==========
class DenseFusionOnly(nn.Module):
    """
    Lightweight DenseFusion for point-text feature alignment
    Only used for affordance prediction, no impact on PoseNet
    No Transformer, minimal computation
    """
    def __init__(self, point_dim=512, text_dim=512, hidden_dim=256, drop_prob=0.1):
        super().__init__()
        self.fuse_mlp = nn.Sequential(
            nn.Linear(point_dim + text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, point_dim)
        )
    
    def forward(self, point_features, text_features):
        """
        Args:
            point_features: [B, N, 512] - per-point point cloud features
            text_features: [B, 512] - global text features
        Returns:
            fused_point_features: [B, N, 512]
        """
        B, N, _ = point_features.shape
        # Repeat text features to match point cloud sequence length
        text_repeat = text_features.unsqueeze(1).repeat(1, N, 1)
        concat = torch.cat([point_features, text_repeat], dim=-1)
        return self.fuse_mlp(concat)

# ========== Linear Diffusion Scheduler ==========
def linear_diffusion_schedule(betas, T):
    """
    Linear noise schedule for diffusion model
    Compute alpha, beta, cumulative products and related coefficients
    """
    beta_t = (betas[1] - betas[0]) * torch.arange(0, T + 1, dtype=torch.float32) / T + betas[0]
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    
    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

# ========== Unified Detection-Diffusion Model ==========
class DetectionDiffusion(nn.Module):
    def __init__(self, betas, n_T, device, background_text, drop_prob=0.01):
        super().__init__()
        # Core networks
        self.posenet = PoseNet()  # Dual-branch with NoiseCutMix
        self.pointnetplusplus = PointNetPlusPlus()
        
        # DenseFusion exclusively for affordance prediction
        self.affordance_fusion = DenseFusionOnly(
            point_dim=512,
            text_dim=512,
            hidden_dim=256,
            drop_prob=drop_prob
        )
        
        # Learnable temperature scaling for similarity
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Register diffusion schedule buffers
        schedule = linear_diffusion_schedule(betas, n_T)
        for key, val in schedule.items():
            self.register_buffer(key, val)
        
        self.n_T = n_T
        self.device = device
        self.background_text = background_text
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, xyz, text, affordance_label, g):
        """
        Training forward pass
        Compute affordance loss + pose diffusion loss
        Args:
            xyz: [B, N, 3] - input point cloud
            text: text prompts
            affordance_label: [B, N] - affordance ground truth
            g: [B, 7] - ground truth 6DoF pose (quat + trans)
        Returns:
            affordance_loss, pose_loss
        """
        B = xyz.shape[0]
        
        # Step 1: Extract point cloud features
        point_features, global_cloud_feat = self.pointnetplusplus(xyz)
        # Reshape for affordance branch: [B, N, 512]
        point_feat_afford = point_features.permute(0, 2, 1)
        
        # Step 2: Extract frozen CLIP text features
        with torch.no_grad():
            text_feat_fore = text_encoder(text)
            text_feat_back = text_encoder([self.background_text] * B)
        
        # Step 3: DenseFusion for affordance prediction
        fused_afford_feat = self.affordance_fusion(point_feat_afford, text_feat_fore)
        fused_afford_feat = fused_afford_feat.permute(0, 2, 1)  # [B, 512, N]
        
        # Step 4: Affordance prediction & loss
        text_feat_all = torch.cat([
            text_feat_back.unsqueeze(1),
            text_feat_fore.unsqueeze(1)
        ], dim=1)  # [B, 2, 512]
        
        # Cosine similarity for affordance classification
        norm_text = torch.norm(text_feat_all, dim=2, keepdim=True)
        norm_point = torch.norm(fused_afford_feat, dim=1, keepdim=True)
        similarity = torch.einsum('bij,bjk->bik', text_feat_all, fused_afford_feat)
        normalized_sim = similarity / (torch.einsum('bij,bjk->bik', norm_text, norm_point))
        affordance_pred = self.logit_scale * normalized_sim
        affordance_pred = F.log_softmax(affordance_pred, dim=1)
        affordance_loss = F.nll_loss(affordance_pred, affordance_label)
        
        # Step 5: Pose diffusion training loss
        # Sample timesteps and noise
        t_step = torch.randint(1, self.n_T + 1, (B,)).to(self.device)
        noise = torch.randn_like(g)
        
        # Noisy pose input
        g_noisy = (
            self.sqrtab[t_step - 1, None] * g
            + self.sqrtmab[t_step - 1, None] * noise
        )
        
        # Context dropout mask
        context_mask = torch.bernoulli(torch.zeros(B, 1) + 1 - self.drop_prob).to(self.device)
        
        # Predict noise and compute MSE loss
        noise_pred = self.posenet(
            g_noisy, global_cloud_feat, text_feat_fore, context_mask, t_step / self.n_T
        )
        pose_loss = self.loss_mse(noise, noise_pred)
        
        return affordance_loss, pose_loss

    @torch.no_grad()
    def detect_and_sample(self, xyz, text, n_sample, guide_w):
        """
        Inference: predict affordance + sample 6DoF pose via diffusion
        Args:
            xyz: [1, N, 3] - single point cloud
            text: single text prompt
            n_sample: number of pose samples
            guide_w: classifier-free guidance weight
        Returns:
            affordance_result: predicted affordance label
            pose_samples: sampled 6DoF poses
        """
        # Initialize random pose
        g_i = torch.randn(n_sample, 7).to(self.device)
        
        # Step 1: Extract point cloud features
        point_features, global_cloud_feat = self.pointnetplusplus(xyz)
        point_feat_afford = point_features.permute(0, 2, 1)
        
        # Step 2: Extract text features
        text_feat_fore = text_encoder(text)
        text_feat_back = text_encoder([self.background_text])
        
        # Step 3: Affordance prediction
        fused_afford_feat = self.affordance_fusion(point_feat_afford, text_feat_fore)
        fused_afford_feat = fused_afford_feat.permute(0, 2, 1)
        
        text_feat_all = torch.cat([
            text_feat_back.unsqueeze(1),
            text_feat_fore.unsqueeze(1)
        ], dim=1)
        
        norm_text = torch.norm(text_feat_all, dim=2, keepdim=True)
        norm_point = torch.norm(fused_afford_feat, dim=1, keepdim=True)
        similarity = torch.einsum('bij,bjk->bik', text_feat_all, fused_afford_feat)
        normalized_sim = similarity / (torch.einsum('bij,bjk->bik', norm_text, norm_point))
        affordance_pred = self.logit_scale * normalized_sim
        affordance_pred = F.log_softmax(affordance_pred, dim=1)
        affordance_result = np.argmax(affordance_pred.cpu().numpy(), axis=1)
        
        # Step 4: Diffusion pose sampling
        # Expand features for batch sampling
        c_expand = global_cloud_feat.repeat(n_sample, 1)
        t_expand = text_feat_fore.repeat(n_sample, 1)
        context_mask = torch.ones(n_sample, 1, device=self.device)
        
        # Classifier-free guidance: duplicate with unconditional branch
        c_expand = c_expand.repeat(2, 1)
        t_expand = t_expand.repeat(2, 1)
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample:] = 0.0  # mask off text context for second half
        
        # Reverse diffusion loop
        for i in range(self.n_T, 0, -1):
            t_norm = torch.tensor([i / self.n_T], device=self.device).repeat(n_sample * 2)
            g_i = g_i.repeat(2, 1)
            
            # Noise for sampling
            z = torch.randn(n_sample, 7, device=self.device) if i > 1 else 0
            
            # Predict noise
            eps = self.posenet(g_i, c_expand, t_expand, context_mask, t_norm)
            eps_cond = eps[:n_sample]
            eps_uncond = eps[n_sample:]
            
            # Guidance
            eps = (1 + guide_w) * eps_cond - guide_w * eps_uncond
            g_i = g_i[:n_sample]
            
            # Update pose
            g_i = self.oneover_sqrta[i] * (g_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        
        return affordance_result, g_i.cpu().numpy()