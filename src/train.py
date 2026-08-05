"""
Model training, hyperparameter tuning, and model comparison module.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

# Try importing XGBoost optional dependency
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

def get_candidate_models() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Returns candidate machine learning models along with their hyperparameter search grids.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            }
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [4, 6, 8, 12, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [6, 10, 15, None],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"]
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0]
            }
        )
    }
    
    if XGB_AVAILABLE:
        models["XGBoost"] = (
            xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
            {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        )
    else:
        models["Extra Trees"] = (
            ExtraTreesClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [6, 10, 15, None],
                "min_samples_split": [2, 5]
            }
        )
        
    return models

def train_and_tune_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    n_iter: int = 10,
    scoring: str = "roc_auc"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains and tunes all candidate models using Stratified Cross-Validation and RandomizedSearchCV.
    """
    models_dict = get_candidate_models()
    trained_results = {}
    comparison_list = []
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    print("\n[=== Starting Model Training & Hyperparameter Search ===]")
    for name, (model, param_grid) in models_dict.items():
        print(f"[*] Training & Tuning: {name}...")
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=min(n_iter, np.prod([len(v) for v in param_grid.values()])),
            scoring=scoring,
            cv=cv,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        cv_score = search.best_score_
        
        trained_results[name] = {
            "model": best_model,
            "best_params": search.best_params_,
            "cv_roc_auc": cv_score
        }
        
        comparison_list.append({
            "Model": name,
            "CV ROC-AUC": round(cv_score, 4),
            "Best Parameters": str(search.best_params_)
        })
        
    comparison_df = pd.DataFrame(comparison_list).sort_values(by="CV ROC-AUC", ascending=False).reset_index(drop=True)
    return trained_results, comparison_df

def save_best_model(
    model: Any,
    model_name: str,
    filepath: str = "models/best_model.joblib"
):
    """Saves the winning model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        "model_name": model_name,
        "model": model
    }
    joblib.dump(payload, filepath)
    print(f"[+] Best Model ({model_name}) saved successfully to: {filepath}")

def load_best_model(filepath: str = "models/best_model.joblib") -> Tuple[Any, str]:
    """Loads the winning model from disk."""
    payload = joblib.load(filepath)
    return payload["model"], payload["model_name"]
