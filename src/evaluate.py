"""
Evaluation metrics, confusion matrix, ROC curves, and financial impact analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, roc_curve
)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes standard classification evaluation metrics.
    """
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Log-Loss": round(log_loss(y_true, y_prob), 4)
    }

def generate_evaluation_report(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "reports"
) -> pd.DataFrame:
    """
    Evaluates all trained models on the held-out test set and produces a comparative table and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_rows = []
    plt.figure(figsize=(9, 6))
    
    for name, item in models_dict.items():
        model = item["model"]
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
            
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        metrics["Model"] = name
        report_rows.append(metrics)
        
        # Plot ROC curve for model
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['ROC-AUC']:.3f})", linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, "roc_curve_comparison.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    
    df_report = pd.DataFrame(report_rows)[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Log-Loss"]]
    df_report = df_report.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    
    # Save CSV summary
    df_report.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)
    print(f"[+] Saved evaluation report summary and ROC plot to: {output_dir}")
    
    return df_report

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: str = "reports/confusion_matrix.png"):
    """Plots and saves confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Good Risk (Non-Default)', 'High Risk (Default)'],
                yticklabels=['Good Risk (Non-Default)', 'High Risk (Default)'])
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('Actual True Label', fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {output_path}")
