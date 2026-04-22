import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import from existing codebase
from run_all import (
    DATA_CSV, IMAGE_ROOT, DEVICE, SEED,
    BrainTumorDataset, get_transforms, create_cnn,
    train_epoch, validate
)

# Best Hyperparameters (from Optuna)
BEST_PARAMS = {
    'model_name': 'resnet18',
    'optimizer': 'adamw',
    'lr': 5.97e-4,
    'bs': 32,
    'p_aug': 0.33,
    'epochs': 12  # Slightly reduced for CV speed, usually enough for convergence
}

def run_cross_validation(n_splits=5):
    print(f"🚀 Starting {n_splits}-Fold Cross Validation")
    print(f"PARAMS: {BEST_PARAMS}")
    
    df = pd.read_csv(DATA_CSV)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # Split Data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Datasets
        train_ds = BrainTumorDataset(
            train_df, 
            IMAGE_ROOT, 
            transforms=get_transforms(img_size=224, is_train=True, aug_params={'p_aug': BEST_PARAMS['p_aug']})
        )
        val_ds = BrainTumorDataset(
            val_df, 
            IMAGE_ROOT, 
            transforms=get_transforms(img_size=224, is_train=False)
        )
        
        # Loaders (Weighted Sampling for Train)
        class_counts = train_df.label.value_counts().sort_index().values
        class_weights = 1. / (class_counts + 1e-6)
        sample_weights = [class_weights[l] for l in train_df.label]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_ds, batch_size=BEST_PARAMS['bs'], sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BEST_PARAMS['bs'], shuffle=False, num_workers=0)
        
        # Model Setup
        model = create_cnn(num_classes=4, model_name=BEST_PARAMS['model_name']).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if BEST_PARAMS['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=BEST_PARAMS['lr'])
        elif BEST_PARAMS['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=BEST_PARAMS['lr'], momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_PARAMS['lr'])
            
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2)
        
        # Training Loop
        best_fold_roc = 0.0
        epochs_no_improve = 0
        early_stopping = 4
        
        for e in range(1, BEST_PARAMS['epochs'] + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_roc, _, _, _ = validate(model, val_loader, criterion, DEVICE)
            
            scheduler.step(val_roc)
            
            if val_roc > best_fold_roc:
                best_fold_roc = val_roc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            print(f"  Ep {e}: ROC {val_roc:.4f} | Loss {val_loss:.4f} | Best {best_fold_roc:.4f}")
            
            if epochs_no_improve >= early_stopping:
                print(f"  Early stopping at epoch {e}")
                break
        
        fold_results.append(best_fold_roc)
        print(f"✅ Fold {fold+1} Result: {best_fold_roc:.5f}")

    # Summary
    mean_roc = np.mean(fold_results)
    std_roc = np.std(fold_results)
    
    print("\n" + "="*30)
    print("🏁 CROSS VALIDATION RESULTS")
    print("="*30)
    print(f"Folds: {fold_results}")
    print(f"Mean AUC: {mean_roc:.5f}")
    print(f"Std Dev:  {std_roc:.5f}")
    
    # Save to file
    with open("results/cv_results.txt", "w") as f:
        f.write(f"Mean AUC: {mean_roc:.5f}\n")
        f.write(f"Std Dev: {std_roc:.5f}\n")
        f.write(f"Folds: {fold_results}\n")

if __name__ == "__main__":
    run_cross_validation()
