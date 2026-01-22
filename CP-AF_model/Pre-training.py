import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings

# 导入所有需要的模块
from common_cnn_optimized import Config, CnnEncoderOptimized
from losses import SupervisedContrastiveLoss
from samplers import MPerClassSampler

# ======================================================================================
# 1. 预训练模型
# ======================================================================================
class PretrainSupConModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = CnnEncoderOptimized(config)
        self.projection_head = nn.Sequential(
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM),
            nn.GELU(),
            nn.Linear(config.LATENT_DIM, config.PROJECTION_DIM)
        )

    def forward(self, x):
        z = self.encoder(x)
        projection = self.projection_head(z)
        return projection

# ======================================================================================
# 2. 数据集与数据增强
# ======================================================================================
class PretrainDataset(Dataset):
    def __init__(self, filepath, config: Config):
        print(f"Loading single-label data for SupCon from {filepath}...")
        df = pd.read_csv(filepath, header=None)
        self.labels = df.iloc[:, 0].values
        features_raw = df.iloc[:, 2:].values
        self.features = np.zeros((features_raw.shape[0], config.PRETRAIN_SEQ_LEN), dtype=np.float32)
        copy_len = min(features_raw.shape[1], config.PRETRAIN_SEQ_LEN)
        self.features[:, :copy_len] = features_raw[:, :copy_len]
        print(f"Data loaded. Found {len(np.unique(self.labels))} unique labels.")

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return torch.tensor(self.features[idx]).unsqueeze(0), self.labels[idx]

def augment_data(sequence_batch, config):
    """
    Applies a series of augmentations to a batch of sequences.
    """
    augmented_batch = sequence_batch.clone()
    
    # Block masking
    block_size = sequence_batch.shape[2] // config.NUM_AUG_BLOCKS
    if block_size > 0:
        num_mask = random.randint(config.MIN_MASK_BLOCKS_AUG, config.MAX_MASK_BLOCKS_AUG)
        for _ in range(num_mask):
            block_idx = random.randint(0, config.NUM_AUG_BLOCKS - 1)
            start_idx, end_idx = block_idx * block_size, (block_idx + 1) * block_size
            augmented_batch[:, 0, start_idx:end_idx] = 0
    
    # Gaussian noise
    noise = torch.randn_like(augmented_batch) * config.AUG_NOISE_STD
    augmented_batch += noise
    
    # Random scaling
    scaling_factor = torch.rand(1, device=augmented_batch.device) * \
                     (config.AUG_SCALE_RANGE[1] - config.AUG_SCALE_RANGE[0]) + config.AUG_SCALE_RANGE[0]
    augmented_batch *= scaling_factor
    
    return augmented_batch

# ======================================================================================
# 3. 预训练主循环
# # ======================================================================================
def main_pretrain_supcon():
    warnings.filterwarnings('ignore')
    cfg = Config()
    
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n--- Ultimate Pre-training: SupCon with Custom Sampler (Fixed) ---")
    
    dataset = PretrainDataset(cfg.SINGLE_LABEL_CSV, cfg)
    
    m_per_class = 2
    sampler = MPerClassSampler(labels=dataset.labels, m=m_per_class, batch_size=cfg.PRETRAIN_BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    print(f"Using MPerClassSampler: batch_size={cfg.PRETRAIN_BATCH_SIZE}, m={m_per_class}, {sampler.n_classes_per_batch} classes per batch.")
    
    model = PretrainSupConModel(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.PRETRAIN_LR, weight_decay=cfg.WEIGHT_DECAY)
    
    num_training_steps = len(dataloader) * cfg.PRETRAIN_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)
    
    criterion = SupervisedContrastiveLoss(temperature=cfg.TEMPERATURE, device=device)

    best_loss = float('inf')
    for epoch in range(cfg.PRETRAIN_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"SupCon Pre-train Epoch {epoch+1}/{cfg.PRETRAIN_EPOCHS}")
        
        for data_batch, labels_batch in pbar:
            data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
            
            view1 = augment_data(data_batch, cfg)
            view2 = augment_data(data_batch, cfg)
            
            features = torch.cat([view1, view2], dim=0)
            labels = torch.cat([labels_batch, labels_batch], dim=0)
            
            optimizer.zero_grad()
            projections = model(features)
            projections = F.normalize(projections, dim=1)
            
            loss = criterion(projections, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch End: Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  ▲ New best model saved! Loss: {best_loss:.4f}")
            torch.save(model.encoder.state_dict(), cfg.PRETRAINED_ENCODER_PATH)

    print(f"\nSupervised Contrastive pre-training finished! Best Encoder saved to: {cfg.PRETRAINED_ENCODER_PATH}")

if __name__ == "__main__":
    def augment_data(sequence_batch, config):
        augmented_batch = sequence_batch.clone()
        block_size = sequence_batch.shape[2] // config.NUM_AUG_BLOCKS
        if block_size > 0:
            num_mask = random.randint(config.MIN_MASK_BLOCKS_AUG, config.MAX_MASK_BLOCKS_AUG)
            for _ in range(num_mask):
                block_idx = random.randint(0, config.NUM_AUG_BLOCKS - 1)
                start_idx, end_idx = block_idx * block_size, (block_idx + 1) * block_size
                augmented_batch[:, 0, start_idx:end_idx] = 0
        noise = torch.randn_like(augmented_batch) * config.AUG_NOISE_STD
        augmented_batch += noise
        scaling_factor = torch.rand(1, device=augmented_batch.device) * \
                        (config.AUG_SCALE_RANGE[1] - config.AUG_SCALE_RANGE[0]) + config.AUG_SCALE_RANGE[0]
        augmented_batch *= scaling_factor
        return augmented_batch
    main_pretrain_supcon()