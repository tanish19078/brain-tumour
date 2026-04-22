import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import timm
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import zipfile
import pandas as pd

# ==============================================================================
# 🛡️ UNCERTAINTY ANALYSIS (MC DROPOUT) - COLAB COMPATIBLE
# ==============================================================================

# 1. Detect Environment
try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# 2. Configure Paths
if IS_COLAB:
    MODEL_PATH = '/content/best_model_leakage_free.pth'
    ZIP_PATH = '/content/errors.zip'
    BASE_DIR = '/content/error_analysis'
    ERROR_DIR = f'{BASE_DIR}/analysis_results/misclassified_samples'
    OUT_CSV = '/content/uncertainty_results.csv'
else:
    MODEL_PATH = 'best_model_leakage_free.pth'
    ZIP_PATH = 'stage3/errors.zip'
    BASE_DIR = 'stage3/error_analysis_results'
    ERROR_DIR = f'{BASE_DIR}/analysis_results/misclassified_samples'
    OUT_CSV = 'stage3/uncertainty_results.csv'

N_PASSES = 30  # Number of MC Dropout forward passes

# --- UTILITIES ---

def prepare_environment():
    """Extract zip if running on Colab"""
    if IS_COLAB and os.path.exists(ZIP_PATH):
        print(f"📦 Extracting {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
    
    if not os.path.exists(ERROR_DIR):
        print(f"❌ Error: Misclassified samples not found at {ERROR_DIR}")
        print("💡 Hint: Ensure 'errors.zip' is uploaded to Colab or folder exists locally.")
        return False
    return True

def enable_dropout(m):
    """ Force dropout layers to stay active during evaluation (MC Dropout) """
    for module in m.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

@torch.no_grad()
def get_uncertainty_stats(model, img_tensor):
    """ Perform multiple forward passes with Dropout enabled """
    model.eval()
    enable_dropout(model)
    
    all_probs = []
    for _ in range(N_PASSES):
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        all_probs.append(probs.cpu().numpy())
    
    all_probs = np.stack(all_probs)  # (N_PASSES, 1, 3)
    mean_probs = all_probs.mean(axis=0)[0]  
    
    # 📉 Calculate Predictive Entropy (Uncertainty Metric)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
    # 📉 Calculate Variation (Average standard deviation across classes)
    variation = all_probs.std(axis=0)[0].mean()
    
    return mean_probs, entropy, variation

def preprocess_image(img_path):
    img = np.array(Image.open(img_path).convert('RGB'))
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor

# --- EXECUTION ---

def run_analysis():
    if not prepare_environment(): return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")

    # Load Model
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=3).to(device)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {MODEL_PATH}")
    else:
        print(f"❌ Error: Model weights not found at {MODEL_PATH}")
        return

    results = []
    print(f"🔍 Analyzing uncertainty on error samples...")

    # Process all images in the misclassified folder
    folders = [f for f in os.listdir(ERROR_DIR) if os.path.isdir(os.path.join(ERROR_DIR, f))]
    
    for folder in folders:
        folder_path = os.path.join(ERROR_DIR, folder)
        images = [i for i in os.listdir(folder_path) if i.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Processing {folder}"):
            img_path = os.path.join(folder_path, img_name)
            tensor = preprocess_image(img_path).to(device)
            
            mean_probs, entropy, var = get_uncertainty_stats(model, tensor)
            results.append({
                'category': folder,
                'image': img_name,
                'entropy': entropy,
                'variance': var,
                'confidence': np.max(mean_probs)
            })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUT_CSV, index=False)
        print(f"\n📊 FINAL RESULTS:")
        print(f"Total Errors Processed: {len(results)}")
        print(f"Mean Predictive Entropy (Uncertainty): {df['entropy'].mean():.4f}")
        print(f"💾 Results saved to: {OUT_CSV}")
    else:
        print("❌ No images were processed.")

if __name__ == "__main__":
    run_analysis()
