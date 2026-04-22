# 🧠 Brain Tumor MRI Classification

A multi-stage deep learning study for classifying brain tumors from MRI scans, progressing from initial hyperparameter tuning to leakage-free validation with rigorous cross-validation.

## Dataset

**Figshare Brain Tumor Dataset**
- **3,064** T1-weighted contrast-enhanced MRI slices
- **233** patients
- **3 classes**: Meningioma, Glioma, Pituitary Tumor
- Source: [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

## Project Structure

```
brain-tumour/
├── stage-1-hyperparameter-tuning/     # Optuna-based hyperparameter search
│   ├── optuna_tuning/                 # Tuning scripts (Optuna sweep, training, CV)
│   └── results/                       # Initial fine-tuning results & plots
│
├── stage-2-model-comparison/          # Head-to-head model comparison
│   ├── multi_model_training.py        # Train 5 architectures with same config
│   ├── summary_report.md              # Results table
│   └── <model_name>/                  # Per-model curves & predictions
│
├── stage-3-leakage-free-validation/   # Rigorous patient-level split training
│   ├── train_leakage_free.py          # EfficientNet-B0 with GroupShuffleSplit
│   ├── kfold_cross_validation.py      # 5-fold patient-level CV
│   ├── error_analysis.py              # Misclassification analysis
│   ├── uncertainty_analysis.py        # MC Dropout uncertainty quantification
│   └── ERROR_ANALYSIS.md              # Qualitative error taxonomy
│
└── stage-4-architecture-search/       # Broad architecture comparison (8 models)
    └── hypertuning_results/           # Per-model results, comparison report
```

## Key Results

### Stage 2 — Model Comparison (5 models)

| Model | ROC AUC |
|---|---|
| EfficientNet-B0 | **0.9997** |
| ResNet-34 | 0.9993 |
| ResNet-18 | 0.9992 |
| MobileNetV3-Large | 0.9990 |
| ViT-Tiny | 0.9921 |

### Stage 3 — Leakage-Free Validation (Patient-Level Split)

- **Model**: EfficientNet-B0
- **Validation Accuracy**: 97.4%
- **Error Rate**: ~2.6% (16 / 612 samples)
- Most common error: Glioma misclassified as Meningioma (10 / 16 errors)

### Stage 4 — Architecture Search (8 models, patient-level splits)

| Model | ROC AUC | Accuracy |
|---|---|---|
| Swin Transformer (Tiny) | **0.9980** | 97.55% |
| ConvNeXt (Tiny) | 0.9979 | 97.55% |
| ViT-Small | 0.9962 | 95.75% |
| EfficientNet-B4 | 0.9951 | 93.79% |

## Methodology Highlights

1. **Patient-level data splitting** throughout to prevent data leakage
2. **Optuna** for automated hyperparameter search (Stage 1)
3. **Weighted sampling** to handle class imbalance
4. **Albumentations** for training augmentation
5. **Early stopping** with learning rate scheduling
6. **5-fold cross-validation** for robust performance estimation
7. **MC Dropout** for uncertainty quantification
8. **Grad-CAM** for model interpretability

## Tech Stack

- PyTorch + [timm](https://github.com/huggingface/pytorch-image-models)
- scikit-learn, Optuna, Albumentations
- matplotlib, seaborn

## Reproducing

1. Download the [Figshare dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
2. Install dependencies: `pip install torch timm albumentations scikit-learn optuna h5py scipy opencv-python matplotlib seaborn`
3. Run the training script for the desired stage

## License

This project is for educational and research purposes.
