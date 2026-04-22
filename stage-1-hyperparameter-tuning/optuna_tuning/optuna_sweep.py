
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import optuna
import cv2

# ==========================================
# CONFIG & REPRODUCIBILITY
# ==========================================
DATA_CSV = 'dataset/data.csv'   
IMAGE_ROOT = 'dataset'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)

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
        # Fix paths for non-Windows (Colab compatibility)
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
            # Compatible CoarseDropout arguments
            A.CoarseDropout(p=p_aug/2, min_holes=1, max_holes=8, min_height=1, max_height=img_size//10, min_width=1, max_width=img_size//10), 
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size,img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

# ==========================================
# MODEL & TRAINING UTILS
# ==========================================
def create_cnn(num_classes=4, model_name='resnet18', pretrained=True, dropout=0.2):
    m = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
    return m

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds = []
    targets = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.extend(torch.argmax(out.detach(), dim=1).cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())
    acc = (np.array(preds)==np.array(targets)).mean()
    return np.mean(losses), acc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses=[]
    preds=[]
    probs=[]
    targets=[]
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        losses.append(loss.item())
        prob = torch.softmax(out, dim=1).cpu().numpy()
        p = torch.argmax(out, dim=1).cpu().numpy()
        probs.extend(prob.tolist())
        preds.extend(p.tolist())
        targets.extend(labels.cpu().numpy().tolist())
    acc = (np.array(preds)==np.array(targets)).mean()
    try:
        roc = roc_auc_score(targets, probs, multi_class='ovr')
    except Exception:
        roc = float('nan')
    return np.mean(losses), acc, roc, probs, preds, targets

def run_pytorch_training(df, model_name='resnet18', img_size=224, bs=32, epochs=10, lr=1e-4, 
                         optimizer_name='adamw', aug_params=None, device=DEVICE):
    # split
    # Stratified Block Split
    train_dfs = []
    val_dfs = []
    for label in df.label.unique():
        sub = df[df.label==label].sort_values('image_path')
        split_idx = int(len(sub)*0.8)
        train_dfs.append(sub.iloc[:split_idx])
        val_dfs.append(sub.iloc[split_idx:])
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    train_ds = BrainTumorDataset(train_df, IMAGE_ROOT, transforms=get_transforms(img_size, is_train=True, aug_params=aug_params))
    val_ds = BrainTumorDataset(val_df, IMAGE_ROOT, transforms=get_transforms(img_size, is_train=False))

    # Class balancing
    class_counts = train_df.label.value_counts().sort_index().values
    class_weights = 1. / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in train_df.label]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = create_cnn(num_classes=4, model_name=model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    
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
                # print(f'Early stopping at epoch {e}!')
                break
    return model, history, probs, preds, targets, best_roc

# ==========================================
# OPTUNA OBJECTIVE
# ==========================================
def objective(trial):
    # Load data
    df = pd.read_csv(DATA_CSV)
    
    # Define search space
    model_name = trial.suggest_categorical('model_name', [
        'resnet18', 'resnet34', 'efficientnet_b0', 'efficientnet_b1', 
        'mobilenetv3_large_100', 'vit_tiny_patch16_224'
    ])
    
    lr = trial.suggest_float('lr', 1e-5, 3e-3, log=True)
    bs = trial.suggest_categorical('bs', [16, 32])
    epochs = 8
    
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    p_aug = trial.suggest_float('p_aug', 0.3, 0.8)
    aug_params = {'p_aug': p_aug}
    
    print(f"Trial Params: {trial.params}")
    
    try:
        _, _, _, _, _, best_roc = run_pytorch_training(
            df, 
            model_name=model_name, 
            img_size=224, 
            bs=bs, 
            epochs=epochs, 
            lr=lr,
            optimizer_name=optimizer_name,
            aug_params=aug_params,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        return best_roc
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    study_name = "brain_tumor_study"
    storage_name = "sqlite:///{}.db".format(study_name)
    
    # Create Study
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize", 
        load_if_exists=True
    )
    
    print("Starting Optuna optimization...")
    study.optimize(objective, n_trials=20) 
    
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
    
    # Save results
    df_trials = study.trials_dataframe()
    df_trials.to_csv("optuna_results_comprehensive.csv", index=False)
    
    print("\n" + "="*30)
    print("RUNNING FINAL TRAINING WITH BEST PARAMS")
    print("="*30)
    
    best_params = study.best_params
    model_name = best_params['model_name']
    lr = best_params['lr']
    bs = best_params['bs']
    optimizer_name = best_params.get('optimizer', 'adamw')
    aug_params = {'p_aug': best_params.get('p_aug', 0.5)}
    
    final_epochs = 15
    df = pd.read_csv(DATA_CSV)
    
    model, history, probs, preds, targets, best_roc = run_pytorch_training(
        df, 
        model_name=model_name, 
        img_size=224, 
        bs=bs, 
        epochs=final_epochs, 
        lr=lr,
        optimizer_name=optimizer_name,
        aug_params=aug_params,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # --- Generate Artifacts ---
    print("\nGenerating Benchmarking Artifacts...")
    
    # 1. Classification Report
    report = classification_report(targets, preds, target_names=['glioma', 'meningioma', 'notumor', 'pituitary'])
    print("\nClassification Report:")
    print(report)
    with open("final_classification_report.txt", "w") as f:
        f.write(report)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['glioma', 'meningioma', 'notumor', 'pituitary'], yticklabels=['glioma', 'meningioma', 'notumor', 'pituitary'])
    plt.title(f'Confusion Matrix (Best Model: {model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("paper_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 3. ROC Curve
    plt.figure(figsize=(8,6))
    for i in range(4):
        y_true_binary = np.array(targets) == i
        y_score = np.array(probs)[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Best Model: {model_name})')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("paper_roc_curve.png", dpi=300)
    plt.close()
    
    # 4. Training Curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.tight_layout()
    plt.savefig("paper_training_curves.png", dpi=300)
    plt.close()
    
    print("DONE! All artifacts saved.")
