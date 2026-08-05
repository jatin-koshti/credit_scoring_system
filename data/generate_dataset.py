"""
Dataset Generation and Fetching Utility for Credit Scoring System.

Generates a realistic credit scoring and loan risk dataset with realistic
financial features, correlations, non-linear relationships, and missing values
matching industry credit bureau characteristics.
"""

import os
import numpy as np
import pandas as pd

def generate_credit_dataset(num_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic credit dataset with realistic distributions and correlations.
    
    Parameters:
        num_samples (int): Number of applicant records to generate.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Synthetic credit dataset dataframe.
    """
    np.random.seed(random_state)
    
    # Applicant Demographics & Financial Attributes
    age = np.random.randint(21, 75, size=num_samples)
    annual_income = np.random.lognormal(mean=10.8, sigma=0.6, size=num_samples).round(2) # ~$25k to $200k+
    annual_income = np.clip(annual_income, 18000, 350000)
    
    credit_score = np.random.normal(loc=670, scale=80, size=num_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    debt_to_income = np.random.beta(a=2, b=5, size=num_samples) * 0.70 # DTI 0.0 to 0.70
    debt_to_income = np.round(debt_to_income, 3)
    
    revolving_utilization = np.random.beta(a=2, b=3, size=num_samples) # 0 to 1
    revolving_utilization = np.round(revolving_utilization, 3)
    
    num_open_credit_lines = np.random.poisson(lam=7, size=num_samples)
    num_open_credit_lines = np.clip(num_open_credit_lines, 1, 30)
    
    delinquencies_2yrs = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.75, 0.15, 0.06, 0.03, 0.01])
    
    num_dependents = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.45, 0.25, 0.18, 0.08, 0.04])
    
    employment_length = np.random.exponential(scale=6, size=num_samples).astype(float).round(1)
    employment_length = np.clip(employment_length, 0, 40)
    
    loan_amount = (annual_income * np.random.uniform(0.1, 0.45, size=num_samples)).round(-2)
    loan_amount = np.clip(loan_amount, 2000, 80000)
    
    home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN'], size=num_samples, p=[0.40, 0.48, 0.12])
    
    loan_purpose = np.random.choice(
        ['DEBT_CONSOLIDATION', 'HOME_IMPROVEMENT', 'CREDIT_CARD', 'SMALL_BUSINESS', 'PERSONAL'],
        size=num_samples,
        p=[0.50, 0.18, 0.17, 0.08, 0.07]
    )
    
    # Calculate Latent Default Risk Score based on domain financial rules
    # Higher score = Higher probability of default
    risk_score = (
        - 0.010 * (credit_score - 600)
        + 4.5 * debt_to_income
        + 3.5 * revolving_utilization
        + 0.8 * delinquencies_2yrs
        - 0.04 * employment_length
        + 0.000015 * (loan_amount - annual_income * 0.2)
        + np.random.normal(loc=0, scale=0.8, size=num_samples)
    )
    
    # Sigmoid function to convert to probability of default
    prob_default = 1 / (1 + np.exp(-risk_score))
    
    # Binary target: 1 = Default (High Risk), 0 = Non-Default (Good Risk)
    default_status = (prob_default > 0.65).astype(int)
    
    # Introduce small realistic missing values (~2-5%) in income, employment_length, revolving_utilization
    income_missing = np.random.choice([True, False], size=num_samples, p=[0.03, 0.97])
    annual_income[income_missing] = np.nan
    
    emp_missing = np.random.choice([True, False], size=num_samples, p=[0.04, 0.96])
    employment_length[emp_missing] = np.nan
    
    util_missing = np.random.choice([True, False], size=num_samples, p=[0.02, 0.98])
    revolving_utilization[util_missing] = np.nan
    
    df = pd.DataFrame({
        'applicant_id': [f"APP-{10000 + i}" for i in range(num_samples)],
        'age': age,
        'annual_income': annual_income,
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'revolving_utilization': revolving_utilization,
        'num_open_credit_lines': num_open_credit_lines,
        'delinquencies_2yrs': delinquencies_2yrs,
        'num_dependents': num_dependents,
        'employment_length': employment_length,
        'loan_amount': loan_amount,
        'home_ownership': home_ownership,
        'loan_purpose': loan_purpose,
        'default_status': default_status
    })
    
    return df

def save_default_dataset(output_dir: str = "data/raw") -> str:
    """Generates and saves the raw credit dataset to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    df = generate_credit_dataset()
    filepath = os.path.join(output_dir, "credit_data.csv")
    df.to_csv(filepath, index=False)
    print(f"[+] Dataset saved successfully to: {filepath} ({len(df)} rows)")
    return filepath

if __name__ == "__main__":
    save_default_dataset()
