# 🧠 Brain Tumor Classification - Hyperparameter Tuning Report

**Generated**: 2026-02-09 06:16:44
**Dataset**: Figshare Brain Tumor MRI (Patient-Level Splits)

## 📊 SOTA Benchmark Comparison

| Study | Year | Method | Result |
|-------|------|--------|--------|
| Cheng et al. | 2016 | Bag of Words | 91.2% Acc |
| Abiwinanda et al. | 2019 | Custom CNN | 84.1% Acc |
| Gumaei et al. | 2019 | Hybrid Feature | 94.2% Acc |
| Bodapati et al. | 2021 | ResNet-50 + Gating | 98.0% Acc |

---

## 🏆 Model Comparison Results

| Rank | Model | ROC AUC | Accuracy | Params (M) | Img Size | Time (s) |
|------|-------|---------|----------|------------|----------|----------|
| 1 | swin_tiny_patch4_window7_224 | 0.9980 | 97.55% | 27.9M | 224 | 631 |
| 2 | convnext_tiny | 0.9979 | 97.55% | 28.2M | 224 | 471 |
| 3 | vit_small_patch16_224 | 0.9962 | 95.75% | 21.9M | 224 | 548 |
| 4 | efficientnet_b4 | 0.9951 | 93.79% | 18.5M | 224 | 726 |
| 5 | mobilenetv3_large_100 | 0.9947 | 94.77% | 4.9M | 224 | 764 |
| 6 | efficientnet_b0 | 0.9936 | 95.75% | 4.7M | 224 | 479 |
| 7 | densenet121 | 0.9930 | 95.59% | 7.5M | 224 | 597 |
| 8 | resnet50 | 0.9882 | 92.65% | 24.6M | 224 | 711 |

---

## 🥇 Best Model: swin_tiny_patch4_window7_224

- **ROC AUC**: 0.9980
- **Accuracy**: 97.55%
- **Parameters**: 27.92M
- **Image Size**: 224
- **Training Time**: 10.5 minutes

### Classification Report
```
Meningioma: P=0.929 R=0.939 F1=0.934
Glioma: P=0.986 R=0.969 F1=0.977
Pituitary: P=0.983 R=1.000 F1=0.991
```

---

## ⚡ Efficiency Analysis

| Model | ROC AUC / Param (M) | Inference Friendly |
|-------|---------------------|-------------------|
| swin_tiny_patch4_window7_224 | 0.0357 | ✅ |
| convnext_tiny | 0.0354 | ✅ |
| vit_small_patch16_224 | 0.0456 | ✅ |
| efficientnet_b4 | 0.0539 | ✅ |
| mobilenetv3_large_100 | 0.2047 | ✅ |
| efficientnet_b0 | 0.2130 | ✅ |
| densenet121 | 0.1327 | ✅ |
| resnet50 | 0.0402 | ✅ |

---

## 📈 Key Findings

1. **Patient-Level Splits**: All results use patient-level splitting to prevent data leakage
2. **Fair Comparison**: All models trained with same hyperparameters (LR, epochs, patience)
3. **Modern Architectures**: ConvNeXt, Swin, MaxViT represent 2021-2022 SOTA

## 🔄 Next Steps

1. Run K-Fold cross-validation on top 3 models
2. Add uncertainty quantification (MC Dropout)
3. Generate Grad-CAM visualizations for interpretability
