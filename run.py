"""
Single Entry Point Orchestration Script for Credit Scoring & Loan Risk Prediction System.
"""

import sys
import os
import argparse
import subprocess

def run_dataset_generation():
    print("[*] Generating credit scoring dataset...")
    from data.generate_dataset import save_default_dataset
    save_default_dataset()

def run_model_training():
    print("\n[=== Starting Credit Risk Model Training Pipeline ===]")
    from data.generate_dataset import save_default_dataset
    from src.data_loader import load_credit_data
    from src.preprocessing import prepare_data_and_preprocessor, save_preprocessor
    from src.train import train_and_tune_models, save_best_model
    from src.evaluate import generate_evaluation_report, plot_confusion_matrix
    from src.explainability import generate_shap_analysis
    
    # 1. Load Data
    raw_csv = "data/raw/credit_data.csv"
    if not os.path.exists(raw_csv):
        save_default_dataset()
    df = load_credit_data(raw_csv)
    
    # 2. Preprocess & Feature Engineer
    X_train, X_test, y_train, y_test, pipeline, feature_names = prepare_data_and_preprocessor(df)
    save_preprocessor(pipeline, feature_names, "models/preprocessor.joblib")
    
    # 3. Train & Tune Candidate Models
    trained_results, comparison_df = train_and_tune_models(X_train, y_train, cv_folds=5, n_iter=10)
    print("\n[=== Model CV Results ===]")
    print(comparison_df.to_string(index=False))
    
    # 4. Evaluate Test Set Performance
    eval_df = generate_evaluation_report(trained_results, X_test, y_test, output_dir="reports")
    print("\n[=== Test Set Evaluation Summary ===]")
    print(eval_df.to_string(index=False))
    
    # 5. Save Winner Model & Plots
    winning_model_name = eval_df.iloc[0]['Model']
    winning_model = trained_results[winning_model_name]['model']
    
    plot_confusion_matrix(y_test, winning_model.predict(X_test), winning_model_name, "reports/confusion_matrix.png")
    generate_shap_analysis(winning_model, X_test[:100], feature_names, "reports")
    save_best_model(winning_model, winning_model_name, "models/best_model.joblib")
    
    print("\n[+] Training & Evaluation pipeline completed successfully!")

def run_streamlit_app():
    print("[*] Launching Streamlit Web Dashboard...")
    app_path = os.path.join("app", "streamlit_app.py")
    
    # Check if running in a cloud environment with a dynamic PORT (like Render)
    port = os.environ.get("PORT")
    if port:
        print(f"[*] Cloud environment detected. Binding to port {port}...")
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", port,
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]
        
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Credit Scoring & Loan Risk Prediction CLI")
    parser.add_argument("--train", action="store_true", help="Train models, perform tuning, and save evaluation reports")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit Web Application")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic credit scoring dataset")
    
    args = parser.parse_args()
    
    if args.generate_data:
        run_dataset_generation()
    elif args.train:
        run_model_training()
    elif args.app:
        run_streamlit_app()
    else:
        # Default action: check if model exists, if not, train, then run Streamlit app
        model_exists = os.path.exists(os.path.join("models", "best_model.joblib"))
        preprocessor_exists = os.path.exists(os.path.join("models", "preprocessor.joblib"))
        
        if not model_exists or not preprocessor_exists:
            print("[-] Model or preprocessor not found. Running training pipeline first...")
            run_model_training()
            
        run_streamlit_app()

if __name__ == "__main__":
    main()
