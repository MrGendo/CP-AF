import os
import torch
import torch.nn as nn

class Config:
    # ================== 数据与路径 ==================
    SINGLE_LABEL_CSV = '/mnt/data/home/ra/cpaf/datasets/tor1tab/tor1tab-dataset.csv'
    REAL_MULTI_LABEL_CSV = '/mnt/data/home/ra/cpaf/datasets/tor5tab/tor-5tab-finetuning100way5shot.csv'
    VAL_DATA_PATH = '/mnt/data/home/ra/cpaf/datasets/tor5tab/tor5tab-dataset-val0.csv'
    Test_DATA_PATH = '/mnt/data/home/ra/cpaf/datasets/tor5tab/tor5tab-dataset-test0.csv'
    
    # ================== 实验与模型保存 ==================
    EXPERIMENT_NAME = "tor-5tab-5shot" # 新实验名
    OUTPUT_DIR = f'/mnt/data/home/ra/cpaf/saved_models/{EXPERIMENT_NAME}'
    PRETRAINED_ENCODER_PATH = os.path.join(OUTPUT_DIR, 'best_encoder.pth')
    FINETUNED_MODEL_PATH = os.path.join(OUTPUT_DIR, 'finetuned_5tab_5shot.pth')
    
    # ================== 模型架构参数 ==================
    PRETRAIN_SEQ_LEN = 4096
    FINETUNE_FULL_SEQ_LEN = 15000
    SUB_SEQ_LEN = 4096
    SUB_SEQ_STRIDE = 512  #2048
    N_CLASSES = 100  # 微调时的多标签类别数
    LATENT_DIM = 256

    # ================== 预训练超参数 ==================
    N_SINGLE_CLASSES = 100       # 预训练时单标签数据的类别数
    LAMBDA_CLASSIFY = 0.5        # 多任务学习中，分类损失的权重
    
    PROJECTION_DIM = 128
    PRETRAIN_EPOCHS = 300  #100
    PRETRAIN_BATCH_SIZE = 32  #128
    PRETRAIN_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.2
    
    # 数据增强参数
    AUG_NOISE_STD = 0.05
    AUG_SCALE_RANGE = (0.8, 1.2)
    NUM_AUG_BLOCKS = 50
    MIN_MASK_BLOCKS_AUG = 2
    MAX_MASK_BLOCKS_AUG = 10

    # ================== 微调超参数 ==================
    FINETUNE_EPOCHS = 100
    FINETUNE_BATCH_SIZE = 16  #16
    FREEZE_EPOCHS = 10 #10
    FINETUNE_LR_FROZEN = 1e-3
    FINETUNE_LR_UNFROZEN = 2e-5

    FEATURE_START_COL = 106  #106
    FINETUNE_NUM_TRUE_LABELS = 5 #5

class ResidualBlock(nn.Module):
    """一个带残差连接的CNN块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.gelu(out)
        return out

class CnnEncoderOptimized(nn.Module):
    """强化的CNN编码器，使用残差块。"""
    def __init__(self, config: Config):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            ResidualBlock(1, 32, kernel_size=31, stride=4, padding=15),
            ResidualBlock(32, 64, kernel_size=15, stride=4, padding=7),
            ResidualBlock(64, 128, kernel_size=7, stride=5, padding=3),
            ResidualBlock(128, 256, kernel_size=5, stride=5, padding=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.encoder_fc = nn.Linear(256, config.LATENT_DIM)

    def forward(self, x):
        features = self.encoder_conv(x)
        z = self.encoder_fc(features)
        return z