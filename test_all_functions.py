"""
Tester script to verify all modules and functions in the Credit Scoring System.
Tests data generation, preprocessing, model training, evaluation, explainability, and saving/loading.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.generate_dataset import generate_credit_dataset, save_default_dataset
from src.data_loader import load_credit_data, split_features_target
from src.preprocessing import prepare_data_and_preprocessor, save_preprocessor, load_preprocessor
from src.train import get_candidate_models, train_and_tune_models, save_best_model, load_best_model
from src.evaluate import evaluate_predictions, generate_evaluation_report, plot_confusion_matrix
from src.explainability import get_feature_importances, generate_shap_analysis

def test_pipeline():
    print("[TEST] Starting Verification of Credit Scoring System Functions...")
    
    # 1. Test Dataset Generation
    print("[TEST] 1. Testing data/generate_dataset.py...")
    df = generate_credit_dataset(num_samples=100, random_state=42)
    assert isinstance(df, pd.DataFrame), "Generated data should be a pandas DataFrame"
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    assert 'default_status' in df.columns, "Target column default_status missing"
    print("[PASS] Dataset generation verified successfully.")
    
    # Save dummy dataset
    dummy_csv = "data/raw/test_credit_data.csv"
    os.makedirs(os.path.dirname(dummy_csv), exist_ok=True)
    df.to_csv(dummy_csv, index=False)
    print(f"[TEST] Saved test dataset to {dummy_csv}")
    
    # 2. Test Data Loader
    print("[TEST] 2. Testing src/data_loader.py...")
    df_loaded = load_credit_data(dummy_csv)
    assert len(df_loaded) == 100, "Loaded data size mismatch"
    X, y = split_features_target(df_loaded, target_col='default_status')
    assert 'default_status' not in X.columns, "Target column still in X features"
    assert len(y) == 100, "Target vector length mismatch"
    print("[PASS] Data loader verified successfully.")
    
    # 3. Test Preprocessing & Feature Engineering
    print("[TEST] 3. Testing src/preprocessing.py...")
    X_train, X_test, y_train, y_test, pipeline, feature_names = prepare_data_and_preprocessor(
        df_loaded, target_col='default_status', test_size=0.2, random_state=42
    )
    assert X_train.shape[0] == 80, f"Expected 80 train rows, got {X_train.shape[0]}"
    assert X_test.shape[0] == 20, f"Expected 20 test rows, got {X_test.shape[0]}"
    assert len(feature_names) > 0, "No features after preprocessing"
    
    # Save and reload preprocessor
    test_prep_path = "models/test_preprocessor.joblib"
    save_preprocessor(pipeline, feature_names, test_prep_path)
    assert os.path.exists(test_prep_path), "Preprocessor joblib file not written"
    loaded_pipe, loaded_fnames = load_preprocessor(test_prep_path)
    assert len(loaded_fnames) == len(feature_names), "Reloaded features list size mismatch"
    print("[PASS] Preprocessing pipeline and feature engineering verified successfully.")
    
    # 4. Test Model Training
    print("[TEST] 4. Testing src/train.py...")
    candidates = get_candidate_models()
    assert "Logistic Regression" in candidates, "Logistic Regression should be in candidate models"
    
    # Train using simple/fast parameters
    trained_results, comparison_df = train_and_tune_models(
        X_train, y_train, cv_folds=3, n_iter=2, scoring="roc_auc"
    )
    assert len(trained_results) > 0, "No trained models returned"
    assert isinstance(comparison_df, pd.DataFrame), "Comparison table should be a pandas DataFrame"
    
    best_name = comparison_df.iloc[0]['Model']
    best_model = trained_results[best_name]['model']
    test_model_path = "models/test_best_model.joblib"
    save_best_model(best_model, best_name, test_model_path)
    assert os.path.exists(test_model_path), "Best model joblib file not written"
    
    loaded_model, loaded_name = load_best_model(test_model_path)
    assert loaded_name == best_name, "Reloaded model name mismatch"
    print("[PASS] Model training, tuning, and saving/loading verified successfully.")
    
    # 5. Test Evaluation
    print("[TEST] 5. Testing src/evaluate.py...")
    y_pred = loaded_model.predict(X_test)
    y_prob = loaded_model.predict_proba(X_test)[:, 1] if hasattr(loaded_model, "predict_proba") else np.zeros(len(y_test))
    
    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    assert "Accuracy" in metrics, "Missing Accuracy metric"
    assert "ROC-AUC" in metrics, "Missing ROC-AUC metric"
    
    test_reports_dir = "reports_test"
    generate_evaluation_report(trained_results, X_test, y_test, output_dir=test_reports_dir)
    assert os.path.exists(os.path.join(test_reports_dir, "model_performance_summary.csv")), "Summary CSV not saved"
    assert os.path.exists(os.path.join(test_reports_dir, "roc_curve_comparison.png")), "ROC plot image not saved"
    
    plot_confusion_matrix(y_test, y_pred, best_name, os.path.join(test_reports_dir, "test_confusion_matrix.png"))
    assert os.path.exists(os.path.join(test_reports_dir, "test_confusion_matrix.png")), "Confusion matrix image not saved"
    print("[PASS] Evaluation and metric calculations verified successfully.")
    
    # 6. Test Explainability
    print("[TEST] 6. Testing src/explainability.py...")
    df_imp = get_feature_importances(loaded_model, feature_names)
    assert isinstance(df_imp, pd.DataFrame), "Feature importances should be a DataFrame"
    assert len(df_imp) == len(feature_names), "Feature count mismatch in importances"
    
    generate_shap_analysis(loaded_model, X_test, feature_names, output_dir=test_reports_dir)
    assert os.path.exists(os.path.join(test_reports_dir, "feature_importance.png")) or os.path.exists(os.path.join(test_reports_dir, "shap_summary_plot.png")), "Importance plot not saved"
    print("[PASS] Model explainability verified successfully.")
    
    # Clean up test output files
    print("[TEST] Cleaning up test files...")
    for f in [dummy_csv, test_prep_path, test_model_path]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(test_reports_dir):
        import shutil
        shutil.rmtree(test_reports_dir)
        
    print("\n[SUCCESS] All system modules and functions passed validation checks!")

if __name__ == "__main__":
    test_pipeline()
