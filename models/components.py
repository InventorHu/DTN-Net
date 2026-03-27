import torch
import torch.nn as nn
import open_clip
import math
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from einops.layers.torch import Rearrange

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional encoding for diffusion time steps.
    Generates frequency-based embeddings to inject temporal information.
    """
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        """
        Args:
            time: Diffusion timesteps, shape [B]
        Returns:
            Time embeddings, shape [B, dim]
        """
        time = time * self.scale
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1 + 1e-5)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings.squeeze(1)  # [B, dim]

    def __len__(self):
        return self.dim
    

class TimeNet(nn.Module):
    """
    MLP-based time embedding network for diffusion models.
    Maps raw timestep values to high-dimensional feature space.
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    def forward(self, t):
        return self.net(t)
   

class TextEncoder(nn.Module):
    """
    CLIP-based text encoder for text prompt feature extraction.
    Freezes pre-trained weights to avoid overfitting.
    """
    def __init__(self, device):
        super(TextEncoder, self).__init__()
        self.device = device
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=False, device=self.device
        )
        
        # Load local pre-trained CLIP weights
        local_model_path = "../pre-trained_model/open_clip_pytorch_model.bin"
        state_dict = torch.load(local_model_path, map_location=self.device)
        self.clip_model.load_state_dict(state_dict)

        # Freeze all CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

    def forward(self, texts):
        """
        Args:
            texts: Single string or list of text prompts
        Returns:
            CLIP text features, shape [B, 512]
        """
        with torch.no_grad():
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            tokens = tokenizer(texts).to(self.device)
            text_features = self.clip_model.encode_text(tokens).to(self.device)
        return text_features.detach()


class PointNetPlusPlus(nn.Module):
    """
    PointNet++ encoder for point cloud feature extraction.
    Combines set abstraction and feature propagation for hierarchical feature learning.
    """
    def __init__(self):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128])

        self.conv1 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, xyz):
        """
        Args:
            xyz: Input point cloud, shape [B, N, 3]
        Returns:
            point_features: Per-point features, shape [B, 512, N]
            global_feature: Global cloud feature, shape [B, 1024]
        """
        xyz = xyz.contiguous().transpose(1, 2)
        l0_xyz = xyz
        l0_points = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        global_feature = l3_points.squeeze()

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)
        point_features = self.bn1(self.conv1(l0_points))
        return point_features, global_feature
    

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention + cross-attention + MLP architecture.
    Used for multi-modal feature fusion in the pose estimation pipeline.
    """
    def __init__(self, d_model=128, nhead=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Cross-attention branch for multi-modal fusion
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # MLP feed-forward branch
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        
    def forward(self, x, cross_modes):
        """
        Args:
            x: Input sequence features, shape [B, Seq_len, D]
            cross_modes: Cross-modal context features, shape [B, Context_len, D]
        Returns:
            Fused features, shape [B, Seq_len, D]
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        x_self, _ = self.self_attn(x, x, x)
        x = residual + x_self
        
        # Cross-modal attention
        shortcut = x
        x_cross = self.norm_cross(x)
        x_cross, _ = self.cross_attn(x_cross, cross_modes, cross_modes)
        x = shortcut + x_cross
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class OutputFusion(nn.Module):
    """
    Final fusion head to map fused features to 7-DOF pose noise.
    Outputs 7-dimensional vector (quaternion + translation).
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.fc = nn.Linear(3 * d_model, 7)
        
    def forward(self, x):
        """
        Args:
            x: Fused sequence features, shape [B, 3, d_model]
        Returns:
            7-DOF pose noise, shape [B, 7]
        """
        return self.fc(x.flatten(1))


class NoiseMixModule(nn.Module):
    """
    NoiseCutMix module for diffusion training (follows paper formulation).
    Performs element-wise noise mixing using binary masks and Beta distribution.
    Designed for 7-DOF pose noise (quaternion + translation).
    """
    def __init__(self, noise_dim=7, alpha=1.0):
        super().__init__()
        self.noise_dim = noise_dim
        self.alpha = alpha

    def generate_mask_and_lambda(self, batch_size, device):
        """
        Generate binary mask and mixing ratio lambda from Beta distribution.
        Returns:
            mask: Binary mask, shape [B, noise_dim, 1]
            lam: Mixing ratio, shape [B]
        """
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size,)).to(device)
        mask = torch.ones((batch_size, self.noise_dim), device=device)
        
        for i in range(batch_size):
            num_zero_dims = int(self.noise_dim * (1 - lam[i]))
            zero_dims = torch.randperm(self.noise_dim, device=device)[:num_zero_dims]
            mask[i, zero_dims] = 0
        return mask.unsqueeze(-1), lam

    def forward(self, noise_A, noise_B):
        """
        Mix two noise tensors following paper formula: ε = M⊙ε_A + (1-M)⊙ε_B
        Args:
            noise_A: First noise branch output, shape [B, 7]
            noise_B: Second noise branch output, shape [B, 7]
        Returns:
            mixed_noise: Fused noise for diffusion loss
        """
        if self.training:
            batch_size = noise_A.shape[0]
            device = noise_A.device
            mask, lam = self.generate_mask_and_lambda(batch_size, device)
            mixed_noise = mask * noise_A.unsqueeze(-1) + (1 - mask) * noise_B.unsqueeze(-1)
            return mixed_noise.squeeze(-1)
        return noise_A


class PoseNet(nn.Module):
    """
    Dual-branch diffusion-based 6DoF pose estimation network.
    Fuses point cloud, text, pose, and time features via two independent transformer branches.
    Uses NoiseCutMix for robust diffusion training.
    """
    def __init__(self, d_model=128, num_layers=1):
        super().__init__()
        # Feature embedding layers
        self.cloud_embed = nn.Sequential(nn.Linear(1024, d_model), nn.LayerNorm(d_model))
        self.text_embed = nn.Sequential(nn.Linear(512, d_model), nn.LayerNorm(d_model))
        self.pose_embed = nn.Sequential(nn.Linear(7, d_model), nn.LayerNorm(d_model))
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model//2),
            nn.Linear(d_model//2, d_model)
        )
        
        # Dual independent fusion FC layers
        self.fc_branch1 = nn.Linear(3*d_model, 3*d_model)
        self.fc_branch2 = nn.Linear(3*d_model, 3*d_model)
        
        # Cross-modal projection layers for transformer compatibility
        self.cross_proj1 = nn.Linear(1024 + 7 + d_model, d_model)
        self.cross_proj2 = nn.Linear(512 + 7 + d_model, d_model)
        
        # Dual independent transformer encoders
        self.transformer_branch1 = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_layers)])
        self.transformer_branch2 = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_layers)])
        
        # Dual independent output heads
        self.output_branch1 = OutputFusion(d_model)
        self.output_branch2 = OutputFusion(d_model)
        
        # Noise fusion module
        self.noise_mix = NoiseMixModule(noise_dim=7)

    def forward(self, g, c, t, context_mask, _t):
        """
        Forward pass for dual-branch pose diffusion model.
        Args:
            g: Input pose, shape [B, 7]
            c: Point cloud global feature, shape [B, 1024]
            t: Text CLIP feature, shape [B, 512]
            context_mask: Multi-modal mask, shape [B, 1]
            _t: Diffusion timestep, shape [B]
        Returns:
            mixed_noise: Fused 7-DOF pose noise, shape [B, 7]
        """
        # Generate multi-modal embeddings
        c_emb = self.cloud_embed(c * context_mask).unsqueeze(1)  # [B, 1, d_model]
        t_emb = self.text_embed(t * context_mask).unsqueeze(1)  # [B, 1, d_model]
        g_emb = self.pose_embed(g.float()).unsqueeze(1)         # [B, 1, d_model]
        time_emb = self.time_embed(_t).unsqueeze(1)             # [B, 1, d_model]
        
        # Construct dual feature sequences
        seq1 = torch.cat([c_emb, g_emb, time_emb], dim=1)  # Cloud branch: [B, 3, d_model]
        seq2 = torch.cat([t_emb, g_emb, time_emb], dim=1)  # Text branch: [B, 3, d_model]
        
        # Dual FC fusion
        seq1_mixed = self.fc_branch1(seq1.flatten(1)).reshape(seq1.shape)
        seq2_mixed = self.fc_branch2(seq2.flatten(1)).reshape(seq2.shape)
        
        # Generate cross-modal context features
        cross1 = torch.cat([c, g, time_emb.squeeze(1)], dim=1)
        cross1 = self.cross_proj1(cross1).unsqueeze(1)  # [B, 1, d_model]
        
        cross2 = torch.cat([t, g, time_emb.squeeze(1)], dim=1)
        cross2 = self.cross_proj2(cross2).unsqueeze(1)  # [B, 1, d_model]
        
        # Dual transformer encoding
        x1 = seq1_mixed
        for layer in self.transformer_branch1:
            x1 = layer(x1, cross1)
        
        x2 = seq2_mixed
        for layer in self.transformer_branch2:
            x2 = layer(x2, cross2)
        
        # Dual branch noise prediction
        noise1 = self.output_branch1(x1)
        noise2 = self.output_branch2(x2)
        
        # Noise fusion via NoiseCutMix
        mixed_noise = self.noise_mix(noise1, noise2)
        
        return mixed_noise