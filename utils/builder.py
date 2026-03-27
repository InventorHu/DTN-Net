import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR, MultiStepLR
from dataset import *
from models import *
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW  # 添加AdamW导入

# Pools of models, optimizers, weights initialization methods, schedulers
model_pool = {
    'detectiondiffusion': DetectionDiffusion,
}

optimizer_pool = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW  # 新增AdamW支持
}

init_pool = {
    'default_init': weights_init
}

# 修改后的scheduler_pool
scheduler_pool = {
    'step': StepLR,
    'cos': CosineAnnealingLR,        # 保留原有键
    'cosine': CosineAnnealingLR,     # 新增别名
    'lr_lambda': LambdaLR,
    'multi_step': MultiStepLR
}


def build_model(cfg):
    """_summary_
    Function to build the model before training
    """
    if hasattr(cfg, 'model'):
        model_info = cfg.model
        weights_init = model_info.get('weights_init', None)
        background_text = model_info.get('background_text', 'none')
        device = model_info.get('device', torch.device('cuda'))
        model_name = model_info.type
        model_cls = model_pool[model_name]
        if model_name in ['detectiondiffusion']:
            betas = model_info.get('betas', [1e-4, 0.02])
            n_T = model_info.get('n_T', 1000)
            drop_prob = model_info.get('drop_prob', 0.1)
            model = model_cls(betas, n_T, device, background_text, drop_prob)
        else:
            raise ValueError("The model name does not exist!")
        if weights_init != None:
            init_fn = init_pool[weights_init]
            model.apply(init_fn)
        return model
    else:
        raise ValueError("Configuration does not have model config!")


def build_dataset(cfg):
    """_summary_
    Function to build the dataset
    """
    if hasattr(cfg, 'data'):
        data_info = cfg.data
        data_path = data_info.data_path
        train_set = ThreeDAPDataset(data_path, mode='train')
        val_set = ThreeDAPDataset(data_path, mode='val')
        test_set = ThreeDAPDataset(data_path, mode='test')
        dataset_dict = dict(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set
        )
        return dataset_dict
    else:
        raise ValueError("Configuration does not have data config!")


def build_loader(cfg, dataset_dict):
    """_summary_
    Function to build the loader
    """
    train_set = dataset_dict["train_set"]
    train_loader = DataLoader(train_set, batch_size=cfg.training_cfg.batch_size,
                              shuffle=True, drop_last=False, num_workers=8)
    loader_dict = dict(
        train_loader=train_loader,
    )

    return loader_dict


def build_optimizer(cfg, model):
    """构建优化器及调度器"""
    optimizer_info = cfg.optimizer.copy()  # 创建配置副本避免修改原始配置
    
    # 参数处理
    optimizer_type = optimizer_info.pop('type').lower()  # 统一转为小写
    
    try:
        optimizer_cls = optimizer_pool[optimizer_type]
    except KeyError:
        available = ', '.join(optimizer_pool.keys())
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Available: {available}")
    
    # 过滤无效参数
    valid_params = ['lr', 'betas', 'eps', 'weight_decay', 'momentum', 'amsgrad']
    filtered_params = {k: v for k, v in optimizer_info.items() if k in valid_params}
    
    # 特殊参数检查
    if optimizer_type == 'sgd' and 'momentum' not in filtered_params:
        filtered_params['momentum'] = 0.9  # 设置SGD默认动量
    
    # 初始化优化器
    try:
        optimizer = optimizer_cls(model.parameters(), **filtered_params)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {optimizer_type} optimizer: {str(e)}")
    
    # 构建学习率调度器
    scheduler_info = cfg.scheduler
    if scheduler_info:
        # 修改后（添加名称映射）
        scheduler_name = scheduler_info.type.lower().replace('cosine', 'cos')  # 统一cosine和cos
        try:
            scheduler_cls = scheduler_pool[scheduler_name]
        except KeyError:
            available = ', '.join(scheduler_pool.keys())
            raise ValueError(f"Unsupported scheduler: {scheduler_name}. Available: {available}")
        
        scheduler_params = scheduler_info.copy()
        scheduler_params.pop('type')
        scheduler = scheduler_cls(optimizer, **scheduler_params)
    else:
        scheduler = None
    
    return {
        'scheduler': scheduler,
        'optimizer': optimizer
    }