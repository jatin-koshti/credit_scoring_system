"""
Verification script for testing a single applicant's loan risk prediction.
Acts as a simple CLI check tool using test data.
"""

import os
import sys
import pandas as pd

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_preprocessor
from src.train import load_best_model

def run_single_prediction_test():
    print("[TESTER] Loading preprocessor and model...")
    try:
        model, model_name = load_best_model("models/best_model.joblib")
        preprocessor, feature_names = load_preprocessor("models/preprocessor.joblib")
        print(f"[TESTER] Loaded model: {model_name}")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return
        
    # Define simple test applicant data (representing "add data")
    test_applicant = {
        'age': 35,
        'annual_income': 75000.0,
        'credit_score': 720,
        'debt_to_income': 0.15,
        'revolving_utilization': 0.22,
        'num_open_credit_lines': 5,
        'delinquencies_2yrs': 0,
        'num_dependents': 0,
        'employment_length': 6.5,
        'loan_amount': 12000.0,
        'home_ownership': 'MORTGAGE',
        'loan_purpose': 'CREDIT_CARD'
    }
    
    print("\n[TESTER] Input Applicant Data:")
    for k, v in test_applicant.items():
        print(f"  - {k}: {v}")
        
    # Convert to DataFrame
    df_input = pd.DataFrame([test_applicant])
    
    # Run preprocess
    try:
        X_proc = preprocessor.transform(df_input)
        print(f"\n[TESTER] Preprocessed shape: {X_proc.shape}")
        
        # Predict probability
        prob = model.predict_proba(X_proc)[0, 1]
        prediction = model.predict(X_proc)[0]
        
        print("\n[TESTER] Prediction Results:")
        print(f"  - Default Probability: {prob * 100:.2f}%")
        print(f"  - Risk Class Decision: {prediction} ({'High Risk' if prediction == 1 else 'Good Risk'})")
        
        if prob < 0.25:
            print("  - Risk Assessment Status: APPROVED (Low Risk)")
        elif prob < 0.55:
            print("  - Risk Assessment Status: REVIEW REQUIRED (Moderate Risk)")
        else:
            print("  - Risk Assessment Status: REJECTED (High Risk)")
            
        print("\n[PASS] Single prediction test executed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed during prediction pipeline: {e}")

if __name__ == "__main__":
    run_single_prediction_test()
