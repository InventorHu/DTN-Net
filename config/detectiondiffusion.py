import os
import torch
from os.path import join as opj
from utils import PN2_BNMomentum, PN2_Scheduler

exp_name = 'detectiondiffusion'
seed = 1
log_dir = opj("./log/", exp_name)
try:
    os.makedirs(log_dir)
except:
    print('Logging Dir is already existed!')

# # 正确配置1：使用基础余弦退火
# scheduler = dict(
#     type='cosine',
#     T_max=100,   
#     eta_min=1e-5
# )

scheduler = None

optimizer = dict(
    type='adamw',  # 改用AdamW
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=5e-4,  # 增强正则化
    layer_wise_lr={      # 分层学习率
        'cross_attn': 1e-4,
        'cond_proj': 5e-4,
        'default': 3e-4
    }
)

model = dict(
    type='detectiondiffusion',
    device=torch.device('cuda'),
    background_text='none',
    betas=[1e-4, 0.02],
    n_T=1000,
    drop_prob=0.01,
    weights_init='default_init',
)

training_cfg = dict(
    model=model,
    batch_size=30,  # 增大batch_size（需根据显存调整）
    epoch=200,      # 延长训练周期
    gpu='2',
    workflow=dict(train=1),
    bn_momentum=PN2_BNMomentum(origin_m=0.1, m_decay=0.4, step=30),  # 减缓BN衰减
    grad_clip=dict(max_norm=1.0),  # 新增梯度裁剪
)

data = dict(
    data_path="/home/coop/HuWei/original_model/30D_point_model/full_shape_release.pkl",
)