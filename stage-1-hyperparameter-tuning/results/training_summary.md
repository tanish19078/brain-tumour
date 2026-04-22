# 🏆 Hyperparameter Tuning Results Summary

**Total Trials:** 20
**Best ROC AUC:** 0.999178 (Trial 1)

## 🥇 Best Model Configuration
- **Model Architecture:** `resnet18`
- **Optimizer:** `adamw`
- **Learning Rate:** `5.97e-04`
- **Batch Size:** `32`
- **Augmentation Strength (p_aug):** `0.33`

## 📊 Top 10 Models
| Rank | Trial | Score (ROC) | Model | Optimizer | LR | BS |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.99918 | resnet18 | adamw | 5.97e-04 | 32 |
| 2 | 18 | 0.99863 | resnet18 | adamw | 4.32e-04 | 16 |
| 3 | 11 | 0.99862 | resnet18 | adam | 7.12e-04 | 32 |
| 4 | 15 | 0.99855 | resnet18 | adam | 3.80e-04 | 32 |
| 5 | 6 | 0.99850 | vit_tiny_patch16_224 | adam | 1.37e-04 | 32 |
| 6 | 12 | 0.99843 | resnet18 | adam | 7.37e-04 | 32 |
| 7 | 0 | 0.99841 | mobilenetv3_large_100 | adam | 4.72e-04 | 16 |
| 8 | 10 | 0.99833 | resnet18 | adamw | 2.58e-03 | 32 |
| 9 | 16 | 0.99825 | efficientnet_b1 | adamw | 1.19e-03 | 32 |
| 10 | 13 | 0.99800 | resnet18 | adamw | 7.30e-04 | 32 |