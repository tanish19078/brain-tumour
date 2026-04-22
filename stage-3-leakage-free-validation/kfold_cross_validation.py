# ==============================================================================
# 5-FOLD CROSS-VALIDATION - Patient-Level Splits
# Run this after your data is already extracted in Colab
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS = 5
EPOCHS = 15
BATCH_SIZE = 32
LR = 5.97e-4
LABEL_MAP = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary'}

print(f"Device: {DEVICE}")

# Load metadata
df = pd.read_csv('brain_tumor_data/brain_tumor_metadata.csv')
print(f"Total samples: {len(df)}, Total patients: {df['pid'].nunique()}")

# MAT parser
def parse_mat_file(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            image = np.array(f['cjdata']['image']).T
            return image.astype(np.float32)
    except:
        try:
            mat = loadmat(filepath)
            return mat['cjdata'][0, 0]['image'].astype(np.float32)
        except:
            return None

# Dataset
class BrainTumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = parse_mat_file(row['filepath'])
        if image is None:
            image = np.zeros((224, 224), dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, row['label']

# Transforms
train_tf = A.Compose([
    A.Resize(224, 224), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5), A.Normalize(), ToTensorV2()
])
val_tf = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])

# Training function
def train_fold(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    best_roc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                probs = torch.softmax(model(imgs), 1).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(labels.numpy())
        
        roc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
        if roc > best_roc:
            best_roc = roc
            best_probs = np.array(all_probs)
            best_targets = np.array(all_targets)
    
    return best_roc, best_probs, best_targets

# K-Fold Cross-Validation
print(f"\n{'='*60}")
print(f"🔬 {N_FOLDS}-FOLD CROSS-VALIDATION (Patient-Level)")
print(f"{'='*60}")

gkf = GroupKFold(n_splits=N_FOLDS)
fold_results = []
all_targets_cv = []
all_probs_cv = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['pid'])):
    print(f"\n📁 Fold {fold+1}/{N_FOLDS}")
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"   Train: {len(train_df)} imgs, {train_df['pid'].nunique()} patients")
    print(f"   Val:   {len(val_df)} imgs, {val_df['pid'].nunique()} patients")
    
    # Create loaders
    train_loader = DataLoader(BrainTumorDataset(train_df, train_tf), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(BrainTumorDataset(val_df, val_tf), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Fresh model each fold
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=3).to(DEVICE)
    
    # Train
    best_roc, probs, targets = train_fold(model, train_loader, val_loader, EPOCHS)
    fold_results.append(best_roc)
    all_targets_cv.extend(targets)
    all_probs_cv.extend(probs)
    
    print(f"   ✅ Fold {fold+1} Best ROC AUC: {best_roc:.4f}")

# Summary
print(f"\n{'='*60}")
print("📊 CROSS-VALIDATION RESULTS")
print(f"{'='*60}")
mean_roc = np.mean(fold_results)
std_roc = np.std(fold_results)
print(f"\nPer-Fold ROC AUC: {[f'{r:.4f}' for r in fold_results]}")
print(f"\n🏆 Mean ROC AUC: {mean_roc:.4f} ± {std_roc:.4f}")
print(f"   95% CI: [{mean_roc - 1.96*std_roc:.4f}, {mean_roc + 1.96*std_roc:.4f}]")

# Overall classification report
all_preds_cv = np.argmax(all_probs_cv, axis=1)
print(f"\n📋 Overall Classification Report (All Folds):")
print(classification_report(all_targets_cv, all_preds_cv, 
                           target_names=list(LABEL_MAP.values())))

# Save results
results_df = pd.DataFrame({
    'Fold': range(1, N_FOLDS+1),
    'ROC_AUC': fold_results
})
results_df.loc[len(results_df)] = ['Mean', mean_roc]
results_df.loc[len(results_df)] = ['Std', std_roc]
results_df.to_csv('kfold_results.csv', index=False)
print("\n💾 Saved: kfold_results.csv")

# Confusion matrix
cm = confusion_matrix(all_targets_cv, all_preds_cv)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_MAP.values(), yticklabels=LABEL_MAP.values())
plt.title(f'{N_FOLDS}-Fold CV Confusion Matrix\nMean ROC: {mean_roc:.4f} ± {std_roc:.4f}')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.savefig('kfold_confusion_matrix.png', dpi=150)
plt.show()
print("💾 Saved: kfold_confusion_matrix.png")
