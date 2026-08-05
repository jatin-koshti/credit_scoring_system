"""
Model Explainability and Feature Importance Module using SHAP & Scikit-Learn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def get_feature_importances(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extracts feature importances or coefficient weights from trained model.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
        
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    return df_imp

def generate_shap_analysis(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_dir: str = "reports"
) -> Tuple[Any, np.ndarray]:
    """
    Generates SHAP values and outputs summary plot figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            
            # Plot Summary Bar Plot
            plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title("SHAP Feature Importance & Contribution Breakdown", fontsize=12, fontweight='bold')
            plt.tight_layout()
            shap_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
            plt.savefig(shap_plot_path, dpi=300)
            plt.close()
            print(f"[+] Saved SHAP summary plot to: {shap_plot_path}")
            return explainer, shap_values
        except Exception as e:
            print(f"[-] SHAP calculation notice: {e}. Falling back to standard feature importances.")
            
    # Fallback Feature Importance Plot
    df_imp = get_feature_importances(model, feature_names)
    plt.figure(figsize=(9, 6))
    top_10 = df_imp.head(10).sort_values(by="Importance", ascending=True)
    plt.barh(top_10["Feature"], top_10["Importance"], color='#2b5c8f')
    plt.xlabel("Model Feature Importance Weight", fontweight='bold')
    plt.title("Top 10 Feature Importances", fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved feature importance plot to: {plot_path}")
    
    return None, None
