"""
This is a single-file notebook (cells separated with # %%). Save as
`brain_tumour_full_notebook.py` and open in Jupyter (it will recognize cells),
or copy to a .ipynb if you prefer.
"""

# %%
# Basic imports
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Deep learning
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import optuna

# Repro
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)

# ... (imports) ...

# %%
# User-editable paths
DATA_CSV = 'dataset/data.csv'   # CSV with columns: image_path,label
IMAGE_ROOT = 'dataset'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# Helper: create basic features for classical ML (simple: mean, std, histogram)
import cv2

def compute_basic_features(row, image_root=IMAGE_ROOT, bins=32):
    p = row['image_path']
    if os.name != 'nt': # Fix paths for non-Windows (Colab)
        p = p.replace('\\', '/')
        
    full = str(Path(image_root) / p) if not os.path.isabs(p) else p
    img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(full)
    img = cv2.resize(img, (128,128))
    feats = []
    feats.append(img.mean())
    feats.append(img.std())
    # histogram
    hist = cv2.calcHist([img],[0],None,[bins],[0,256]).flatten()
    hist = hist / (hist.sum()+1e-9)
    feats.extend(hist.tolist())
    return np.array(feats, dtype=np.float32)

# Global execution code removed from top-level to prevent run-on-import
# ...

# %%
# Now build PyTorch dataset for DL & Transformer
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
        if os.name != 'nt':
            p = p.replace('\\', '/')
            
        full = str(self.image_root / p) if not os.path.isabs(p) else p
        img = cv2.imread(full)
        if img is None:
            raise FileNotFoundError(full)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        label = int(row['label'])
        return img, label

# %%
# Transforms
def get_transforms(img_size=224, is_train=True, aug_params=None):
    if is_train:
        # Defaults
        p_aug = 0.5
        if aug_params:
            p_aug = aug_params.get('p_aug', 0.5)
            
        return A.Compose([
            A.Resize(img_size,img_size),
            A.HorizontalFlip(p=p_aug),
            A.VerticalFlip(p=p_aug/2),
            A.RandomRotate90(p=p_aug),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=p_aug),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=p_aug),
            A.CoarseDropout(p=p_aug/2,  min_holes=1, max_holes=8, min_height=1, max_height=img_size//10, min_width=1, max_width=img_size//10), 
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size,img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

# ...

    # Optimizer selection
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2)
    
    early_stopping_patience = 5
    epochs_no_improve = 0

    best_roc = 0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_roc':[]}
    
    # Validation loop only variable
    probs, preds, targets = [], [], []

    for e in range(1, epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_roc, probs, preds, targets = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step(val_roc)
        
        print(f"Epoch {e}: tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_roc {val_roc:.4f}")
        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['val_roc'].append(val_roc)
        if val_roc > best_roc:
            epochs_no_improve = 0
            best_roc = val_roc
            torch.save({'model_state': model.state_dict(), 'model_name':model_name}, os.path.join(OUT_DIR,f'best_{model_name}.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping at epoch {e}!')
                break
    return model, history, probs, preds, targets, best_roc

if __name__ == "__main__":
    # Run a small example (ResNet18)
    # Check if df exists (it's global in notebook, but here we need to ensure it loads if run as script)
    if 'df' not in locals():
        df = pd.read_csv(DATA_CSV)

    model_resnet, hist_resnet, resnet_probs, resnet_preds, resnet_targets, best_roc = run_pytorch_training(df, model_name='resnet18', img_size=224, bs=32, epochs=6, lr=2e-4)

    # ... (plotting code) ...
    # Plotting code commented out for now to ensure clean import specific usage
    pass
