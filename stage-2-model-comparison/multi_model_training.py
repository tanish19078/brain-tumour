
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
import json

# ==========================================
# CONFIG & REPRODUCIBILITY
# ==========================================
DATA_CSV = 'dataset/data.csv'
IMAGE_ROOT = 'dataset'
OUT_DIR = 'comprehensive_results'
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ==========================================
# DATASET & TRANSFORMS
# ==========================================
class BrainTumorDataset(Dataset):
    def __init__(self, df, image_root, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = row['image_path']
        # Path correction for cross-platform
        p = p.replace('\\', '/')
        full = self.image_root / p
        img = cv2.imread(str(full))
        if img is None:
            raise FileNotFoundError(f"Image not found at: {full}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        label = int(row['label'])
        return img, label

def get_transforms(img_size=224, is_train=True, p_aug=0.33):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=p_aug),
            A.VerticalFlip(p=p_aug/2),
            A.RandomRotate90(p=p_aug),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=p_aug),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=p_aug),
            A.CoarseDropout(p=p_aug/2, min_holes=1, max_holes=8, min_height=1, max_height=img_size//10, min_width=1, max_width=img_size//10),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

# ==========================================
# TRAINING FUNCTIONS
# ==========================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds = []
    targets = []
    for imgs, labels in tqdm(loader, desc="  Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.extend(torch.argmax(out.detach(), dim=1).cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())
    acc = (np.array(preds) == np.array(targets)).mean()
    return np.mean(losses), acc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    probs = []
    targets = []
    for imgs, labels in tqdm(loader, desc="  Validating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        losses.append(loss.item())
        prob = torch.softmax(out, dim=1).cpu().numpy()
        p = torch.argmax(out, dim=1).cpu().numpy()
        probs.extend(prob.tolist())
        preds.extend(p.tolist())
        targets.extend(labels.cpu().numpy().tolist())
    acc = (np.array(preds) == np.array(targets)).mean()
    try:
        roc = roc_auc_score(targets, probs, multi_class='ovr')
    except:
        roc = 0.0
    return np.mean(losses), acc, roc, np.array(probs), np.array(preds), np.array(targets)

# ==========================================
# MAIN TRAINING LOGIC
# ==========================================
def train_model(model_name, df_train, df_val, cfg):
    print(f"\n🚀 Training Model: {model_name}")
    
    # Datasets & Loaders
    train_ds = BrainTumorDataset(df_train, IMAGE_ROOT, transforms=get_transforms(cfg['img_size'], is_train=True, p_aug=cfg['p_aug']))
    val_ds = BrainTumorDataset(df_val, IMAGE_ROOT, transforms=get_transforms(cfg['img_size'], is_train=False))
    
    # Weighted Sampler for Train
    class_counts = df_train.label.value_counts().sort_index().values
    weights = 1. / (class_counts + 1e-6)
    sample_weights = [weights[l] for l in df_train.label]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_ds, batch_size=cfg['bs'], sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg['bs'], shuffle=False, num_workers=0)
    
    # Model Setup
    model = timm.create_model(model_name, pretrained=True, num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2)
    
    best_roc = 0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_roc':[]}
    epochs_no_improve = 0
    
    model_dir = os.path.join(OUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    for e in range(1, cfg['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_roc, probs, preds, targets = validate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['val_roc'].append(val_roc)
        
        print(f"  Ep {e}/{cfg['epochs']} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | ROC: {val_roc:.4f}")
        
        scheduler.step(val_roc)
        
        if val_roc > best_roc:
            best_roc = val_roc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            # Save predictions for final evaluation
            np.savez(os.path.join(model_dir, 'best_preds.npz'), probs=probs, preds=preds, targets=targets)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg['patience']:
                print(f"  Early stopping at epoch {e}")
                break
                
    # Plotting
    plot_results(history, model_name, model_dir)
    return best_roc

def plot_results(history, model_name, save_dir):
    # Loss & Acc
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'curves.png'))
    plt.close()

def generate_final_report(results, class_names):
    report_path = os.path.join(OUT_DIR, 'summary_report.md')
    md = ["# 🧠 Brain Tumor Classification - Model Comparison\n"]
    md.append("| Model | Best ROC AUC | Status |")
    md.append("|---|---|---|")
    
    for name, score in results.items():
        md.append(f"| {name} | {score:.4f} | ✅ Complete |")
    
    md.append("\n## Detailed Metrics for Best Model")
    # Identify best overall
    best_model = max(results, key=results.get)
    md.append(f"Best Overall Model: **{best_model}** with ROC AUC of **{results[best_model]:.4f}**\n")
    
    # Load best model metrics
    data = np.load(os.path.join(OUT_DIR, best_model, 'best_preds.npz'))
    probs, preds, targets = data['probs'], data['preds'], data['targets']
    
    cls_report = classification_report(targets, preds, target_names=class_names)
    md.append("### Classification Report\n```\n" + cls_report + "\n```")
    
    # Confusion Matrix for Best
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {best_model}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(OUT_DIR, 'best_confusion_matrix.png'))
    plt.close()
    
    md.append(f"\n![Confusion Matrix](best_confusion_matrix.png)")
    
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    print(f"\nSummary report generated at {report_path}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_CSV):
        print(f"Error: {DATA_CSV} not found. Please ensure dataset is unzipped.")
    else:
        df = pd.read_csv(DATA_CSV)
        train_df_full = df[df['split'] == 'Training'].reset_index(drop=True)
        val_df = df[df['split'] == 'Testing'].reset_index(drop=True)
        
        models_to_train = ['resnet18', 'resnet34', 'efficientnet_b0', 'mobilenetv3_large_100', 'vit_tiny_patch16_224']
        
        # Best params based on previous tuning
        config = {
            'img_size': 224,
            'bs': 32,
            'lr': 5.97e-4,
            'p_aug': 0.33,
            'epochs': 30, # Increased per user request
            'patience': 10 # Increased for longer training
        }
        
        final_results = {}
        for m_name in models_to_train:
            try:
                # Subsample 3000 images per model (stratified)
                sample_size = min(3000, len(train_df_full))
                train_df = train_df_full.groupby('label', group_keys=False).apply(
                    lambda x: x.sample(n=int(len(x) * (sample_size / len(train_df_full))), random_state=SEED)
                ).reset_index(drop=True)
                
                print(f"Dataset: Using {len(train_df)} training images for {m_name}")
                
                best_score = train_model(m_name, train_df, val_df, config)
                final_results[m_name] = best_score
            except Exception as e:
                print(f"Failed to train {m_name}: {e}")
                
        # Final artifacts
        class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        generate_final_report(final_results, class_labels)
