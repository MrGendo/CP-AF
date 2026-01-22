import os
import random
import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings

# 导入共享的配置和模型
from common_cnn_optimized import Config, CnnEncoderOptimized
# 从本地文件导入 Asymmetric Loss
from asymmetric_loss import AsymmetricLossOptimized

# ======================================================================================
# 1. 对抗训练FGM
# ======================================================================================
class FGM():
    """
    Fast Gradient Method (FGM) for Adversarial Training.
    This class adds a small perturbation to the model's parameters (usually embeddings)
    in the direction of the gradient of the loss, making the model learn from "worst-case" scenarios.
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='shared_encoder.encoder_conv.0.conv1.weight'):
        """
        Adds adversarial perturbation to the specified parameter.

        Args:
            epsilon (float): The magnitude of the perturbation.
            emb_name (str): The name of the parameter to attack. We typically attack
                            the first convolutional layer's weights as it acts like an embedding layer
                            for the raw signal.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                # Calculate the norm of the gradient
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # Calculate the perturbation
                    r_at = epsilon * param.grad / norm
                    # Apply the perturbation
                    param.data.add_(r_at)

    def restore(self, emb_name='shared_encoder.encoder_conv.0.conv1.weight'):
        """
        Restores the original parameter values after the attack.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


# ======================================================================================
# 2. 微调模型与数据加载 
# ======================================================================================
class FineTuneMultiLabelModel(nn.Module):
    
    def __init__(self, config: Config, pretrained_encoder_state_dict):
        super().__init__()
        self.config = config
        self.shared_encoder = CnnEncoderOptimized(config)
        if pretrained_encoder_state_dict:
            self.shared_encoder.load_state_dict(pretrained_encoder_state_dict)
        self.aggregator_lstm = nn.LSTM(config.LATENT_DIM, config.LATENT_DIM, 
                                       bidirectional=True, batch_first=True, num_layers=2, dropout=0.2)
        self.attention = nn.Linear(config.LATENT_DIM * 2, 1)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(config.LATENT_DIM * 2),
            nn.Linear(config.LATENT_DIM * 2, 512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, config.N_CLASSES)
        )

    def forward(self, x):
        x_unfolded = x.unfold(2, self.config.SUB_SEQ_LEN, self.config.SUB_SEQ_STRIDE)
        N, C, num_sub_seqs, _ = x_unfolded.shape
        x_reshaped = x_unfolded.permute(0, 2, 1, 3).reshape(N * num_sub_seqs, C, self.config.SUB_SEQ_LEN)
        is_frozen = not self.shared_encoder.training
        with torch.no_grad() if is_frozen else torch.enable_grad():
            z_local = self.shared_encoder(x_reshaped)
        z_local_reshaped = z_local.view(N, num_sub_seqs, -1)
        lstm_out, _ = self.aggregator_lstm(z_local_reshaped)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        z_global = torch.sum(lstm_out * attn_weights, dim=1)
        logits = self.classifier_head(z_global)
        return logits

class MultiLabelDataset(Dataset):
    
    def __init__(self, filepath, config: Config):
        print(f"Loading multi-label classification data from: {filepath}")
        df = pd.read_csv(filepath, header=None)
        self.labels = torch.tensor(df.iloc[:, :config.N_CLASSES].values, dtype=torch.float32)

        feature_start_col = config.FEATURE_START_COL
        print(f"Features will be loaded from column {feature_start_col} onwards.")
        features_raw = df.iloc[:, feature_start_col:].values

        padded_features = np.zeros((features_raw.shape[0], config.FINETUNE_FULL_SEQ_LEN), dtype=np.float32)
        copy_len = min(features_raw.shape[1], config.FINETUNE_FULL_SEQ_LEN)
        padded_features[:, :copy_len] = features_raw[:, :copy_len]
        self.features = torch.tensor(padded_features).unsqueeze(1)
        print(f"Loaded {len(self.labels)} samples.")
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def calculate_topk_f1_metrics(outputs_probs, labels, k):
    
    preds = (outputs_probs > 0.5).astype(int)
    # Precision, Recall, F1-Score
    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)

    topk_indices = np.argsort(outputs_probs, axis=1)[:, -k:]
    correct_samples = 0
    for i in range(labels.shape[0]):
        true_labels_indices = np.where(labels[i] == 1)[0]
        if len(true_labels_indices) > 0 and set(true_labels_indices).issubset(set(topk_indices[i])):
            correct_samples += 1
        # if len(true_labels_indices) == k and set(true_labels_indices).issubset(set(topk_indices[i])):
        #     correct_samples += 1
    sample_acc_at_k = correct_samples / labels.shape[0]
    return {
        'precision_micro': precision_micro, 'recall_micro': recall_micro, 'f1_micro': f1_micro,
        'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro,
        f'sample_acc_@{k}': sample_acc_at_k
    }

# ======================================================================================
# 3. 微调主循环 (集成对抗训练)
# ======================================================================================
def main_finetune_adversarial():
    warnings.filterwarnings('ignore')
    cfg = Config()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n--- Ultimate Fine-tuning: Adversarial Training (FGM) ---")
    
    pretrained_encoder_dict = torch.load(cfg.PRETRAINED_ENCODER_PATH, map_location=device)
    model = FineTuneMultiLabelModel(cfg, pretrained_encoder_dict).to(device)
    
    # 初始化对抗训练模块
    fgm = FGM(model)
    print("Initialized FGM for adversarial training.")
    
    train_dataset = MultiLabelDataset(cfg.REAL_MULTI_LABEL_CSV, cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.FINETUNE_BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataset = MultiLabelDataset(cfg.VAL_DATA_PATH, cfg)
    val_loader = DataLoader(val_dataset, batch_size=cfg.FINETUNE_BATCH_SIZE, shuffle=False, num_workers=4)
    
    criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05)
    
    best_val_f1 = 0.0
    best_metrics = {}
    best_sample_acc = 0.0 
    k_val = cfg.FINETUNE_NUM_TRUE_LABELS
    optimizer = None

    for epoch in range(cfg.FINETUNE_EPOCHS):
        # 冻结/解冻与层衰减学习率逻辑
        if epoch < cfg.FREEZE_EPOCHS:
            if epoch == 0:
                print(f"\n--- Epochs 1-{cfg.FREEZE_EPOCHS}: 冻结Encoder，仅训练上层 ---")
                for param in model.shared_encoder.parameters():
                    param.requires_grad = False
                optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                                        lr=cfg.FINETUNE_LR_FROZEN, weight_decay=cfg.WEIGHT_DECAY)
        
        elif epoch == cfg.FREEZE_EPOCHS:
            print(f"\n--- Epochs {cfg.FREEZE_EPOCHS+1}+: 解冻Encoder，微调所有参数 ---")
            for param in model.shared_encoder.parameters():
                param.requires_grad = True
            base_lr = cfg.FINETUNE_LR_UNFROZEN
            encoder_groups = [{"params": model.shared_encoder.encoder_conv[i].parameters(), "lr": base_lr * (0.2 + 0.8 * (i/3))} for i in range(4)]
            encoder_groups.append({"params": model.shared_encoder.encoder_fc.parameters(), "lr": base_lr})
            head_groups = [
                {"params": model.aggregator_lstm.parameters(), "lr": cfg.FINETUNE_LR_FROZEN},
                {"params": model.attention.parameters(), "lr": cfg.FINETUNE_LR_FROZEN},
                {"params": model.classifier_head.parameters(), "lr": cfg.FINETUNE_LR_FROZEN},
            ]
            optimizer = optim.AdamW(encoder_groups + head_groups, weight_decay=cfg.WEIGHT_DECAY)

        # 训练循环集成对抗训练
        model.train()
        pbar = tqdm(train_loader, desc=f"Adversarial Fine-tuning Epoch {epoch+1}/{cfg.FINETUNE_EPOCHS}")
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 1. 正常的前向传播和损失计算
            logits = model(features)
            loss = criterion(logits, labels)
            
            # 2. 正常的反向传播，得到原始梯度
            loss.backward()

            # 3. 对抗训练：攻击
            # 我们攻击第一层卷积的权重，因为它最像Embedding层
            fgm.attack(epsilon=1.0, emb_name='encoder_conv.0.conv1.weight')
            
            # 4. 在受攻击的模型上计算对抗损失
            logits_adv = model(features)
            loss_adv = criterion(logits_adv, labels)
            
            # 5. 反向传播对抗损失，将梯度累加到原始梯度上
            loss_adv.backward()
            
            # 6. 恢复原始的、未受攻击的权重
            fgm.restore(emb_name='encoder_conv.0.conv1.weight')

            # 7. 使用累加后的梯度进行优化器更新
            optimizer.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", adv_loss=f"{loss_adv.item():.4f}")

        # 验证逻辑保持不变
        model.eval()
        all_outputs, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                logits = model(features.to(device))
                all_outputs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())
        
        k_value = cfg.FINETUNE_NUM_TRUE_LABELS
        val_metrics = calculate_topk_f1_metrics(np.concatenate(all_outputs), np.concatenate(all_labels), k=k_value)
        
        # print(f"Epoch End Summary: Val F1-Micro: {val_metrics['f1_micro']:.4f} | "
        #       f"Val F1-Macro: {val_metrics['f1_macro']:.4f} | Val Sample-A@{k_value}: {val_metrics[f'sample_acc_@{k_value}']:.4f}")
        
        # 修改此处的 print 语句以包含所有指标
        print(f"  - Val Precision (Micro/Macro): {val_metrics['precision_micro']:.4f} / {val_metrics['precision_macro']:.4f}")
        print(f"  - Val Recall (Micro/Macro):    {val_metrics['recall_micro']:.4f} / {val_metrics['recall_macro']:.4f}")
        print(f"  - Val F1-Score (Micro/Macro):  {val_metrics['f1_micro']:.4f} / {val_metrics['f1_macro']:.4f}")
        print(f"  - Val Sample-Acc@{cfg.FINETUNE_NUM_TRUE_LABELS}:          {val_metrics[f'sample_acc_@{cfg.FINETUNE_NUM_TRUE_LABELS}']:.4f}")
        
        current_f1 = val_metrics['f1_micro']
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_metrics = val_metrics
            print(f"  ▲ New best F1-Micro! Saving model to {cfg.FINETUNED_MODEL_PATH}")
            torch.save(model.state_dict(), cfg.FINETUNED_MODEL_PATH)
        
        if val_metrics[f'sample_acc_@{k_val}'] > best_sample_acc:
            best_sample_acc = val_metrics[f'sample_acc_@{k_val}']
            
    print(f"\n{'='*20}\nTraining Finished!\n{'='*20}")
    if best_metrics:
        print("\n--- Best Validation Results (based on F1-Micro) ---")
        print(f"  - Precision (Micro/Macro): {best_metrics['precision_micro']:.4f} / {best_metrics['precision_macro']:.4f}")
        print(f"  - Recall (Micro/Macro):    {best_metrics['recall_micro']:.4f} / {best_metrics['recall_macro']:.4f}")
        print(f"  - F1-Score (Micro/Macro):  {best_metrics['f1_micro']:.4f} / {best_metrics['f1_macro']:.4f}")
        print(f"  - Sample-Acc@{cfg.FINETUNE_NUM_TRUE_LABELS}:          {best_metrics[f'sample_acc_@{cfg.FINETUNE_NUM_TRUE_LABELS}']:.4f}")
    else:
        print("No best model was saved. Validation might not have run or improved.")

    print(f"\n--- Peak Sample-Acc@{k_val} across all epochs: {best_sample_acc:.4f} ---")

    # ======================================================================================
# 4. 测试/推理代码 (Evaluation on Test Set)
# ======================================================================================
def run_test_evaluation():
    """
    加载微调后的最佳模型，并在测试集上进行评估，输出与验证集一致的指标。
    """
    print(f"\n{'='*20}\nStarting Test Evaluation\n{'='*20}")
    
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    Test_DATA_PATH = cfg.Test_DATA_PATH
        
    if not os.path.exists(Test_DATA_PATH):
        print(f"Error: Test dataset not found at {Test_DATA_PATH}")
        return

    # 2. 初始化模型架构
    print("Initializing model architecture...")
    # 注意：这里 pretrained_encoder_state_dict 传 None 即可，
    # 因为我们马上会加载完整的 finetuned_state_dict 覆盖它。
    model = FineTuneMultiLabelModel(cfg, pretrained_encoder_state_dict=None).to(device)

    # 3. 加载微调后的最佳权重
    if os.path.exists(cfg.FINETUNED_MODEL_PATH):
        print(f"Loading best finetuned model from: {cfg.FINETUNED_MODEL_PATH}")
        checkpoint = torch.load(cfg.FINETUNED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"Error: Model file not found at {cfg.FINETUNED_MODEL_PATH}")
        return

    # 4. 加载测试数据
    print(f"Loading test data from: {Test_DATA_PATH}")
    test_dataset = MultiLabelDataset(Test_DATA_PATH, cfg)
    test_loader = DataLoader(test_dataset, batch_size=cfg.FINETUNE_BATCH_SIZE, shuffle=False, num_workers=4)

    # 5. 推理循环
    model.eval()
    all_outputs = []
    all_labels = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Testing"):
            features = features.to(device)
            # 前向传播
            logits = model(features)
            # 转为概率
            probs = torch.sigmoid(logits)
            
            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    # 6. 计算指标
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    
    # 使用与微调验证时相同的 k 值
    k_value = cfg.FINETUNE_NUM_TRUE_LABELS 
    
    print("\nCalculating metrics...")
    test_metrics = calculate_topk_f1_metrics(all_outputs, all_labels, k=k_value)

    # 7. 打印结果 (格式与微调输出一致)
    print(f"\n--- Test Set Evaluation Results ---")
    print(f"  - Test Precision (Micro/Macro): {test_metrics['precision_micro']:.4f} / {test_metrics['precision_macro']:.4f}")
    print(f"  - Test Recall (Micro/Macro):    {test_metrics['recall_micro']:.4f} / {test_metrics['recall_macro']:.4f}")
    print(f"  - Test F1-Score (Micro/Macro):  {test_metrics['f1_micro']:.4f} / {test_metrics['f1_macro']:.4f}")
    print(f"  - Test Sample-Acc@{k_value}:          {test_metrics[f'sample_acc_@{k_value}']:.4f}")
    
    return test_metrics


if __name__ == "__main__":
    # start_time = time.time()
    # main_finetune_adversarial()
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"  - Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    run_test_evaluation()