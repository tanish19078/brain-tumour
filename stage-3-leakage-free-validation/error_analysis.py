# ==============================================================================
# ERROR ANALYSIS SCRIPT - Extract Misclassified Images
# Run this in Colab to see exactly which images the model failed on.
# ==============================================================================
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Config
MODEL_PATH = 'best_model_leakage_free.pth'
DATA_DIR = 'brain_tumor_data'
OUTPUT_DIR = 'analysis_results/misclassified_samples'
LABEL_MAP = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary'}

# Setup folders
for actual in LABEL_MAP.values():
    for pred in LABEL_MAP.values():
        if actual != pred:
            os.makedirs(f"{OUTPUT_DIR}/{actual}_as_{pred}", exist_ok=True)

# Dataset & Loader (Validation Split)
class BrainTumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # MAT parser
        try: # HDF5
            with h5py.File(row['filepath'], 'r') as f:
                image = np.array(f['cjdata']['image']).T
        except: # SciPy
            mat = loadmat(row['filepath'])
            image = mat['cjdata'][0, 0]['image']
        
        orig_image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, row['label'], str(row['filepath']), orig_image

# Load Metadata & Split (Use same seed as training)
df = pd.read_csv(f'{DATA_DIR}/brain_tumor_metadata.csv')
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, val_idx = next(gss.split(df, groups=df['pid']))
val_df = df.iloc[val_idx].reset_index(drop=True)

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_loader = DataLoader(BrainTumorDataset(val_df, val_transform), batch_size=1, shuffle=False)

# Load Model
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=3).to('cuda')
checkpoint = torch.load(MODEL_PATH, map_location='cuda', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"🔍 Analyzing {len(val_df)} validation samples...")
error_count = 0

with torch.no_grad():
    for img, label, path, orig_img in tqdm(val_loader):
        out = model(img.to('cuda'))
        pred = out.argmax(1).item()
        actual = label.item()
        
        if pred != actual:
            error_count += 1
            # Save the misclassified image
            folder = f"{OUTPUT_DIR}/{LABEL_MAP[actual]}_as_{LABEL_MAP[pred]}"
            filename = Path(path[0]).name.replace('.mat', '.png')
            
            # Normalize orig_img for saving
            save_img = orig_img[0].numpy()
            save_img = (save_img - save_img.min()) / (save_img.max() - save_img.min() + 1e-8) * 255
            cv2.imwrite(f"{folder}/{filename}", save_img.astype(np.uint8))

print(f"\n✅ Analysis complete. Found {error_count} errors.")
print(f"📁 Misclassified images saved to: {OUTPUT_DIR}")
print(f"📦 To download: !zip -r errors.zip {OUTPUT_DIR}")
