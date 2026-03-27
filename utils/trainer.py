import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as opj
from utils import *
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

DEVICE = torch.device('cuda')

class Trainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.logger = running['logger']
        self.model = running["model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        
        # 初始化优化器（支持分层学习率）
        self._init_optimizer()
        
        # 学习率调度器
        self._init_scheduler()
        
        # 训练状态初始化
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        # BN动量配置
        self.bn_momentum = self.cfg.training_cfg.get('bn_momentum', None)
        
        # 梯度裁剪配置
        self.grad_clip = self.cfg.training_cfg.get('grad_clip', None)
        if self.grad_clip:
            self.logger.cprint(f"Gradient clipping enabled: {self.grad_clip}")
        
        # 学习率历史记录
        self.lr_history = []
        
        # 检查点路径
        self.checkpoint_path = opj(self.cfg.log_dir, 'best_model.pth')
    
    def _init_optimizer(self):
        """初始化分层优化器"""
        # 获取分层参数
        cross_attn_params = []
        cond_proj_params = []
        base_params = []
        for name, param in self.model.named_parameters():
            if 'cross_attn' in name:
                cross_attn_params.append(param)
            elif 'cond_proj' in name:
                cond_proj_params.append(param)
            else:
                base_params.append(param)
        
        # 从配置获取学习率设置
        optimizer_cfg = self.cfg.optimizer
        layer_wise_lr = optimizer_cfg.get('layer_wise_lr', {})
        
        # 构建参数组
        param_groups = [
            {'params': cross_attn_params, 
             'lr': layer_wise_lr.get('cross_attn', 1e-4)},
            {'params': cond_proj_params,
             'lr': layer_wise_lr.get('cond_proj', 5e-4)},
            {'params': base_params,
             'lr': layer_wise_lr.get('default', 3e-4)}
        ]
        
        # 初始化优化器
        if optimizer_cfg['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=optimizer_cfg['lr'],
                betas=optimizer_cfg['betas'],
                weight_decay=optimizer_cfg['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_cfg['type']}")
    
    def _init_scheduler(self):
        """初始化学习率调度器"""
        scheduler_cfg = self.cfg.get('scheduler', None)
        if scheduler_cfg is None:
            self.logger.cprint("No scheduler configured. Proceeding without scheduler.")
            self.scheduler = None
            return
        if scheduler_cfg['type'] == 'onecycle':
            total_steps = self.cfg.training_cfg['epoch'] * len(self.train_loader)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=scheduler_cfg['max_lr'],
                total_steps=total_steps,
                pct_start=scheduler_cfg.get('pct_start', 0.3)
            )
            self.step_per_batch = True
        elif scheduler_cfg['type']  == 'cosine':
            # 检查必要参数存在
            missing = [p for p in ['T_max', 'eta_min'] if p not in scheduler_cfg]
            if missing:
                raise ValueError(f"Missing required params {missing} for cosine scheduler")
        
            # 初始化基础余弦退火
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_cfg['T_max'],
                eta_min=scheduler_cfg['eta_min']
            )
            self.step_per_batch = False  # 按epoch更新
        else:
            self.scheduler = None
        
        if self.scheduler:
            self.logger.cprint(f"Initialized {scheduler_cfg['type'] } scheduler with params: {scheduler_cfg}")
   
    def train(self):
        self.model.train()
        self.logger.cprint(f"Epoch {self.epoch} start training...")
        pbar = tqdm(self.train_loader)
        
        total_afford_loss = 0.0
        total_pose_loss = 0.0
        total_pose_grad_norm = 0.0  # 新增：梯度统计变量
        max_pose_grad = 0.0         # 新增：梯度统计变量
        
        # 记录初始学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        for batch_idx, (_, _, xyz, text, afford_label, rot, trans) in enumerate(pbar):
            # 数据预处理
            xyz = xyz.float().to(DEVICE)
            rot = rot.float().to(DEVICE)
            trans = trans.float().to(DEVICE)
            afford_label = afford_label.squeeze().long().to(DEVICE)
            g = torch.cat((rot, trans), dim=1)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向计算
            afford_loss, pose_loss = self.model(xyz, text, afford_label, g)
            total_loss = afford_loss + pose_loss
            
            # 反向传播
            total_loss.backward()
             # ===================== 新增梯度监控代码 =====================
            pose_grad_norms = []
            pose_grad_max = []
            for name, param in self.model.posenet.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()   # L2范数
                    grad_max = param.grad.data.abs().max().item() # 最大绝对值
                    pose_grad_norms.append(grad_norm)
                    pose_grad_max.append(grad_max)
        
            # 计算统计量
            batch_grad_norm = sum(pose_grad_norms) if pose_grad_norms else 0.0
            batch_grad_max = max(pose_grad_max) if pose_grad_max else 0.0
            total_pose_grad_norm += batch_grad_norm
            max_pose_grad = max(max_pose_grad, batch_grad_max)
            # ============================================================
            # 梯度裁剪
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip['max_norm'],
                    norm_type=2
                )
            
            # 参数更新
            self.optimizer.step()
            
            # 按batch更新学习率
            if self.scheduler and self.step_per_batch:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.lr_history[-1] = current_lr  # 更新当前epoch的lr
            
            # 统计信息
            total_afford_loss += afford_loss.item()
            total_pose_loss += pose_loss.item()
            
            # 更新进度条 (增加梯度显示)
            pbar.set_description(
                f"Afford: {afford_loss.item():.4f} | "
                f"Pose: {pose_loss.item():.4f} | "
                f"GradN: {batch_grad_norm:.1e} | "  # 当前batch梯度范数
                f"GradM: {batch_grad_max:.1e} | "   # 当前batch最大梯度
                f"LR: {current_lr:.2e}"
            )
        
        # 按epoch更新学习率
        if self.scheduler and not self.step_per_batch:
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.lr_history.append(current_lr)
        
        # 计算平均损失
        avg_afford = total_afford_loss / len(self.train_loader)
        avg_pose = total_pose_loss / len(self.train_loader)
        
        # BN动量更新
        if self.bn_momentum:
            self.model.apply(lambda m: self.bn_momentum(m, self.epoch))
        
        # 保存检查点
        self._save_checkpoint(avg_afford + avg_pose)
        # 计算平均梯度统计 (epoch级别)
        avg_grad_norm = total_pose_grad_norm / len(self.train_loader)
        # 输出日志
        log_str = (f"Epoch {self.epoch} | "
                   f"Afford: {avg_afford:.4f} | Pose: {avg_pose:.4f} | "
                   f"LR: {current_lr:.2e} |"
                   f"Avg Norm: {avg_grad_norm:.1e} | "
                f"Max Grad: {max_pose_grad:.1e}")
        self.logger.cprint(log_str)
        
        self.epoch += 1
    
    def _save_checkpoint(self, current_loss):
        """保存模型检查点"""
        # 保存当前模型
        torch.save({
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
        }, opj(self.cfg.log_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳模型
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = self.epoch
            torch.save({
                'epoch': self.epoch,
                'model_state': self.model.state_dict(),
                'metrics': {'loss': current_loss}
            }, self.checkpoint_path)
            self.logger.cprint(f"Saved best model at epoch {self.epoch} with loss {current_loss:.4f}")
    
    def plot_learning_curve(self):
        """生成学习曲线图"""
        plt.figure(figsize=(12, 6))
        
        # 学习率曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.lr_history, 'b-o')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        # 损失曲线（需记录历史）
        # 注：需要在类中添加loss_history属性并记录
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_history, 'r-')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.savefig(opj(self.cfg.log_dir, 'training_curves.png'))
        plt.close()
    
    def run(self):
        """主训练循环"""
        try:
            total_epochs = self.cfg.training_cfg['epoch']
            while self.epoch < total_epochs:
                self.train()
        except KeyboardInterrupt:
            self.logger.cprint("Training interrupted, saving final model...")
        finally:
            #self.plot_learning_curve()
            self.logger.cprint(f"Best model at epoch {self.best_epoch} with loss {self.best_loss:.4f}")