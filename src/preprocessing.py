"""
Data preprocessing, feature engineering, and transformation pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Tuple, List

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn Transformer for Credit Risk Feature Engineering."""
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # Payment to Monthly Income Ratio (Estimated monthly loan payment = loan_amount / 36 months)
        est_monthly_payment = (X_out['loan_amount'] / 36.0)
        monthly_income = (X_out['annual_income'] / 12.0).replace(0, np.nan)
        X_out['payment_to_income'] = (est_monthly_payment / monthly_income).fillna(0).round(4)
        
        # Credit Score Risk Factor Index
        X_out['score_tier_idx'] = pd.cut(
            X_out['credit_score'],
            bins=[0, 580, 670, 740, 800, 900],
            labels=[5, 4, 3, 2, 1]
        ).astype(float).fillna(3)
        
        # Combined Risk Interaction Terms
        X_out['debt_utilization_ratio'] = (X_out['debt_to_income'] * X_out['revolving_utilization']).round(4)
        X_out['loan_to_income_ratio'] = (X_out['loan_amount'] / (X_out['annual_income'] + 1)).round(4)
        
        return X_out

def build_preprocessing_pipeline(
    num_cols: List[str],
    cat_cols: List[str]
) -> Pipeline:
    """
    Constructs a full Scikit-Learn ColumnTransformer preprocessing pipeline.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    
    full_pipeline = Pipeline([
        ('engineer', CreditFeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

def prepare_data_and_preprocessor(
    df: pd.DataFrame,
    target_col: str = 'default_status',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline, List[str]]:
    """
    Applies feature engineering, train-test split, fits pipeline, and returns preprocessed matrices.
    """
    from sklearn.model_selection import train_test_split
    from src.data_loader import split_features_target
    
    X, y = split_features_target(df, target_col=target_col)
    
    # Identify initial numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Feature engineering adds new numerical columns
    engineered_num = num_cols + ['payment_to_income', 'score_tier_idx', 'debt_utilization_ratio', 'loan_to_income_ratio']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    pipeline = build_preprocessing_pipeline(num_cols=engineered_num, cat_cols=cat_cols)
    
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)
    
    # Generate feature names after OneHotEncoding
    ohe_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    feature_names = list(engineered_num) + list(ohe_cols)
    
    return X_train_proc, X_test_proc, y_train.values, y_test.values, pipeline, feature_names

def save_preprocessor(pipeline: Pipeline, feature_names: List[str], filepath: str = "models/preprocessor.joblib"):
    """Serializes fitted preprocessor pipeline and feature names."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        'pipeline': pipeline,
        'feature_names': feature_names
    }
    joblib.dump(payload, filepath)
    print(f"[+] Saved preprocessor pipeline to: {filepath}")

def load_preprocessor(filepath: str = "models/preprocessor.joblib") -> Tuple[Pipeline, List[str]]:
    """Loads saved preprocessor pipeline and feature names."""
    payload = joblib.load(filepath)
    return payload['pipeline'], payload['feature_names']
