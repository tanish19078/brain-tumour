# -*- coding: utf-8 -*-
"""
Brain Tumor Classification: Leakage-Free Training on Figshare Dataset
COLAB-READY VERSION with ROBUST MAT PARSING

Dataset: Figshare Brain Tumor Dataset (3064 slices, 233 patients)
Classes: Meningioma (1), Glioma (2), Pituitary Tumor (3)

INSTRUCTIONS:
1. Upload your Figshare zip (1512427.zip) to Google Drive
2. Mount Drive and run this script
"""

# ==============================================================================
# CELL 1: Install Dependencies (Run this cell first in Colab)
# ==============================================================================
# !pip install -q timm albumentations h5py scipy pandas scikit-learn matplotlib seaborn tqdm opencv-python-headless

# ==============================================================================
# CELL 2: Imports
# ==============================================================================
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==============================================================================
# CELL 3: Configuration
# ==============================================================================
CFG = {
    'seed': 42,
    'model_name': 'efficientnet_b0',
    'img_size': 224,
    'batch_size': 32,
    'epochs': 20,
    'lr': 5.97e-4,
    'weight_decay': 1e-4,
    'patience': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'data_dir': 'brain_tumor_data',
    'drive_zip_path': '/content/drive/MyDrive/1512427.zip',  # Change this to your path
}

LABEL_MAP = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary'}

def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG['seed'])
print(f"Device: {CFG['device']}")

# ==============================================================================
# CELL 4: ROBUST MAT FILE PARSER (Handles Both MATLAB v5 and v7.3)
# ==============================================================================
def parse_mat_file(filepath):
    """
    Robustly parse Figshare brain tumor .mat files.
    Handles both old scipy format and HDF5 format.
    
    Returns: (image, label, pid) or (None, None, None) on failure
    """
    import h5py
    from scipy.io import loadmat
    
    filepath = Path(filepath)
    
    # Try HDF5 format first (MATLAB v7.3+)
    try:
        with h5py.File(filepath, 'r') as f:
            cjdata = f['cjdata']
            
            # Extract image
            image = np.array(cjdata['image']).T  # Transpose for correct orientation
            
            # Extract label (formats vary)
            label_data = cjdata['label'][:]
            if label_data.ndim == 0:
                label = int(label_data)
            elif label_data.size == 1:
                label = int(label_data.flat[0])
            else:
                label = int(label_data[0][0])
            
            # Extract PID (Patient ID)
            pid_ref = cjdata['PID']
            if isinstance(pid_ref, h5py.Dataset):
                pid_data = pid_ref[:]
                if pid_data.dtype.kind == 'O' or pid_data.ndim > 1:
                    # String stored as array of chars
                    pid = "".join([chr(int(c)) if isinstance(c, (int, np.integer)) 
                                   else chr(int(c[0])) for c in pid_data.flat if c != 0])
                else:
                    pid = str(int(pid_data.flat[0]))
            else:
                pid = str(pid_ref)
            
            # Validate
            if label < 1 or label > 3:
                return None, None, None
                
            return image.astype(np.float32), label - 1, pid.strip()  # Zero-index labels
            
    except Exception as hdf_error:
        pass  # Fall through to scipy
    
    # Try scipy format (MATLAB v5)
    try:
        mat = loadmat(filepath)
        cjdata = mat['cjdata']
        
        # Handle structured array format
        if hasattr(cjdata, 'dtype') and cjdata.dtype.names:
            cjdata = cjdata[0, 0]
            image = cjdata['image'].astype(np.float32)
            label = int(cjdata['label'].flat[0])
            
            pid_field = cjdata['PID']
            if hasattr(pid_field, 'flat'):
                pid = "".join([str(c) for c in pid_field.flat]).strip()
            else:
                pid = str(pid_field)
        else:
            # Direct access format
            image = cjdata['image'][0, 0].astype(np.float32)
            label = int(cjdata['label'][0, 0][0, 0])
            pid = str(cjdata['PID'][0, 0][0])
        
        if label < 1 or label > 3:
            return None, None, None
            
        return image, label - 1, pid.strip()
        
    except Exception as scipy_error:
        print(f"Failed to parse {filepath.name}: HDF5 error, SciPy error")
        return None, None, None


def extract_and_parse_data(data_dir, zip_path=None):
    """
    Extract zip and parse all .mat files to create metadata.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / 'brain_tumor_metadata.csv'
    
    # Check for cached metadata
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        if len(df) > 0:
            print(f"✅ Loaded {len(df)} cached records from {metadata_path}")
            return df
    
    # Create directories
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Unzip if needed
    if zip_path and Path(zip_path).exists():
        print(f"📦 Extracting {zip_path}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        print("✅ Extraction complete")
    
    # Find all .mat files
    mat_files = list(data_dir.rglob('*.mat'))
    print(f"🔍 Found {len(mat_files)} .mat files")
    
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in {data_dir}. "
            "Please ensure your zip file is uploaded and path is correct."
        )
    
    # Parse files
    data_records = []
    failed = 0
    
    print("🧠 Parsing .mat files for images and patient IDs...")
    for mf in tqdm(mat_files):
        image, label, pid = parse_mat_file(mf)
        if image is not None:
            data_records.append({
                'filepath': str(mf),
                'label': label,
                'pid': pid,
                'class_name': LABEL_MAP[label]
            })
        else:
            failed += 1
    
    print(f"✅ Successfully parsed {len(data_records)} files ({failed} failed)")
    
    # Create and save metadata
    metadata_df = pd.DataFrame(data_records)
    metadata_df.to_csv(metadata_path, index=False)
    print(f"💾 Metadata saved to {metadata_path}")
    
    return metadata_df


# ==============================================================================
# CELL 5: Dataset Class
# ==============================================================================
class BrainTumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image from .mat file
        image, _, _ = parse_mat_file(row['filepath'])
        
        if image is None:
            # Fallback: return zeros (will be rare)
            image = np.zeros((CFG['img_size'], CFG['img_size']), dtype=np.float32)
        
        # Normalize to 0-255
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = image.astype(np.uint8)
        
        # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, row['label']


# ==============================================================================
# CELL 6: Transforms
# ==============================================================================
train_transform = A.Compose([
    A.Resize(CFG['img_size'], CFG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CFG['img_size'], CFG['img_size']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ==============================================================================
# CELL 7: Training Functions
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.1f}%'})
        
    return running_loss / total, 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return running_loss / total, 100. * correct / total, all_preds, all_targets, np.array(all_probs)


# ==============================================================================
# CELL 8: Main Training Pipeline
# ==============================================================================
def main():
    print("=" * 60)
    print("🧠 Brain Tumor Classification - Leakage-Free Training")
    print("=" * 60)
    
    # ---- Data Loading ----
    # Mount Google Drive (uncomment in Colab)
    # from google.colab import drive
    # drive.mount('/content/drive')
    
    df = extract_and_parse_data(
        data_dir=CFG['data_dir'],
        zip_path=CFG['drive_zip_path'] if os.path.exists(CFG['drive_zip_path']) else None
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Samples: {len(df)}")
    print(f"   Total Patients: {df['pid'].nunique()}")
    print(f"\n   Class Distribution:")
    for label, name in LABEL_MAP.items():
        count = (df['label'] == label).sum()
        print(f"   - {name}: {count} ({100*count/len(df):.1f}%)")
    
    # ---- Patient-Level Split (PREVENTS LEAKAGE) ----
    print("\n🔀 Splitting data at PATIENT level (prevents leakage)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG['seed'])
    train_idx, val_idx = next(gss.split(df, groups=df['pid']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"   TRAIN: {len(train_df)} images from {train_df['pid'].nunique()} patients")
    print(f"   VAL:   {len(val_df)} images from {val_df['pid'].nunique()} patients")
    
    # Safety check
    overlap = set(train_df['pid']) & set(val_df['pid'])
    print(f"   ✅ Patient overlap check: {len(overlap)} (should be 0)")
    
    if len(overlap) > 0:
        raise ValueError("DATA LEAKAGE DETECTED! Patients appear in both train and val sets.")
    
    # ---- Create DataLoaders ----
    train_ds = BrainTumorDataset(train_df, train_transform)
    val_ds = BrainTumorDataset(val_df, val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    # ---- Model Setup ----
    print(f"\n🏗️ Building model: {CFG['model_name']}")
    model = timm.create_model(CFG['model_name'], pretrained=True, num_classes=3)
    model.to(CFG['device'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, verbose=True)
    
    # ---- Training Loop ----
    print(f"\n🚀 Starting training for {CFG['epochs']} epochs...")
    best_val_acc = 0
    best_val_roc = 0
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_roc': []}
    
    for epoch in range(CFG['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{CFG['epochs']}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, CFG['device'])
        val_loss, val_acc, preds, targets, probs = validate(model, val_loader, criterion, CFG['device'])
        
        val_roc = roc_auc_score(targets, probs, multi_class='ovr')
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_roc'].append(val_roc)
        
        print(f"\n📈 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Val ROC AUC: {val_roc:.4f}")
        
        # Save best model
        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_roc': val_roc,
                'val_acc': val_acc,
            }, 'best_model_leakage_free.pth')
            print("   💾 Best model saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CFG['patience']:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # ---- Final Evaluation ----
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load('best_model_leakage_free.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, preds, targets, probs = validate(model, val_loader, criterion, CFG['device'])
    
    print("\n📋 Classification Report:")
    print(classification_report(targets, preds, target_names=list(LABEL_MAP.values())))
    
    final_roc = roc_auc_score(targets, probs, multi_class='ovr')
    print(f"\n🏆 Final ROC AUC (Leakage-Free): {final_roc:.4f}")
    
    # ---- Generate Visualizations ----
    print("\n🎨 Generating visualizations...")
    
    # Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_MAP.values(), yticklabels=LABEL_MAP.values())
    plt.title(f'Confusion Matrix - {CFG["model_name"]} (Leakage-Free)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_leakage_free.png', dpi=150)
    plt.show()
    
    # Training Curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy (%)')
    axes[1].legend()
    
    axes[2].plot(history['val_roc'])
    axes[2].set_title('Validation ROC AUC')
    axes[2].axhline(y=final_roc, color='r', linestyle='--', label=f'Best: {final_roc:.4f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('training_curves_leakage_free.png', dpi=150)
    plt.show()
    
    # ROC Curves
    plt.figure(figsize=(8, 6))
    for i, name in LABEL_MAP.items():
        binary_targets = [1 if t == i else 0 for t in targets]
        fpr, tpr, _ = roc_curve(binary_targets, probs[:, i])
        auc_score = roc_auc_score(binary_targets, probs[:, i])
        plt.plot(fpr, tpr, label=f'{name} (AUC: {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('roc_curves_leakage_free.png', dpi=150)
    plt.show()
    
    print("\n✅ Training complete! All artifacts saved.")
    print(f"🏆 Best Validation ROC AUC: {best_val_roc:.4f}")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    
    return model, history


# ==============================================================================
# CELL 9: Run Training
# ==============================================================================
if __name__ == "__main__":
    model, history = main()
