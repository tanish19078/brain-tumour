import pandas as pd
import os

def generate_summary():
    csv_path = r"c:\Users\Tanish Singla\Desktop\brain tumour\results\optuna_results_comprehensive.csv"
    out_path = r"c:\Users\Tanish Singla\Desktop\brain tumour\results\training_summary.md"
    
    if not os.path.exists(csv_path):
        print("CSV not found!")
        return

    df = pd.read_csv(csv_path)
    # Sort by value (ROC AUC) descending
    df = df.sort_values(by='value', ascending=False).reset_index(drop=True)
    
    best = df.iloc[0]
    
    md = []
    md.append("# 🏆 Hyperparameter Tuning Results Summary")
    md.append(f"\n**Total Trials:** {len(df)}")
    md.append(f"**Best ROC AUC:** {best['value']:.6f} (Trial {best['number']})")
    
    md.append("\n## 🥇 Best Model Configuration")
    md.append(f"- **Model Architecture:** `{best['params_model_name']}`")
    md.append(f"- **Optimizer:** `{best['params_optimizer']}`")
    md.append(f"- **Learning Rate:** `{best['params_lr']:.2e}`")
    md.append(f"- **Batch Size:** `{best['params_bs']}`")
    md.append(f"- **Augmentation Strength (p_aug):** `{best['params_p_aug']:.2f}`")
    
    md.append("\n## 📊 Top 10 Models")
    md.append("| Rank | Trial | Score (ROC) | Model | Optimizer | LR | BS |")
    md.append("|---|---|---|---|---|---|---|")
    
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        md.append(f"| {i+1} | {row['number']} | {row['value']:.5f} | {row['params_model_name']} | {row['params_optimizer']} | {row['params_lr']:.2e} | {row['params_bs']} |")
        
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Summary written to {out_path}")

if __name__ == "__main__":
    generate_summary()
