"""
Data loading and validation module for Credit Scoring System.
"""

import os
import pandas as pd
from typing import Tuple

REQUIRED_COLUMNS = [
    'age', 'annual_income', 'credit_score', 'debt_to_income',
    'revolving_utilization', 'num_open_credit_lines', 'delinquencies_2yrs',
    'num_dependents', 'employment_length', 'loan_amount',
    'home_ownership', 'loan_purpose', 'default_status'
]

def load_credit_data(filepath: str = "data/raw/credit_data.csv") -> pd.DataFrame:
    """
    Loads raw credit scoring dataset from CSV file.
    
    Parameters:
        filepath (str): Path to CSV data file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        from data.generate_dataset import save_default_dataset
        print(f"[-] Data file not found at {filepath}. Auto-generating default dataset...")
        save_default_dataset(os.path.dirname(filepath))
        
    df = pd.read_csv(filepath)
    
    # Schema validation
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
        
    return df

def split_features_target(df: pd.DataFrame, target_col: str = 'default_status') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits dataset into feature matrix X and target vector y.
    
    Parameters:
        df (pd.DataFrame): Dataframe.
        target_col (str): Target column name.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: X features and y target.
    """
    ignore_cols = ['applicant_id', target_col]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y
