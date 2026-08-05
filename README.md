# Credit Scoring & Loan Risk Prediction System
> **CodeAlpha Machine Learning Internship - Task 1**

An end-to-end, production-ready Machine Learning system for evaluating creditworthiness, predicting loan default risk, and generating interpretable risk assessments for retail loan applicants.

---

## 📌 Project Overview
Credit scoring models assist financial institutions in assessing applicant risk, mitigating financial defaults, and ensuring regulatory compliance. This repository provides a complete ML lifecycle pipeline:

1. **Data Ingestion & Synthesis**: Handles financial indicators including DTI, revolving utilization, credit bureau scores, delinquencies, and loan purposes.
2. **Feature Engineering**: Calculates domain-specific financial metrics such as *Payment-to-Income (PTI)*, *Debt-Utilization Ratio*, and *Credit Tier Indices*.
3. **Multi-Model Benchmark & Hyperparameter Tuning**: Trains and tunes 5 algorithms:
   - Logistic Regression (Baseline)
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - XGBoost / Extra Trees Classifier
4. **Model Explainability**: SHAP (SHapley Additive exPlanations) & Feature Importance breakdowns to ensure loan decision transparency.
5. **Interactive Web Dashboard**: Streamlit interface supporting real-time applicant risk evaluation, risk gauge visualization, and batch CSV scoring.

---

## 📂 Project Architecture & Directory Layout

```
credit_scoring_system/
├── data/
│   ├── raw/                  # Storage for raw credit scoring dataset
│   ├── processed/            # Storage for preprocessed data splits
│   └── generate_dataset.py   # Script to generate synthetic credit scoring data
├── notebooks/
│   └── credit_scoring_eda_modeling.ipynb  # Complete EDA, modeling & evaluation notebook
├── src/
│   ├── __init__.py           # Package initializer
│   ├── data_loader.py        # Data ingestion and schema validation
│   ├── preprocessing.py     # Feature engineering & Scikit-Learn preprocessing pipeline
│   ├── train.py              # Model training & RandomizedSearchCV tuning
│   ├── evaluate.py           # Classification metrics, ROC curves, and confusion matrix
│   └── explainability.py     # SHAP value calculator and feature importances
├── models/                   # Serialized winning model (.joblib) & preprocessor pipeline
├── reports/                  # Evaluation summaries, ROC curves, confusion matrix, SHAP plots
├── app/
│   └── streamlit_app.py      # Streamlit Web Dashboard for risk assessment
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore configuration
├── README.md                 # Complete documentation
└── run.py                    # One-click CLI script
```

---

## 🚀 Quick Start & Setup Instructions

### 1. Prerequisites & Environment Setup
Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/credit-scoring-system.git
cd credit-scoring-system

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train Pipeline
To generate the dataset, preprocess features, tune models via 5-Fold Stratified Cross-Validation, and save evaluation metrics and plots to `reports/`:

```bash
python run.py --train
```

### 3. Launch Streamlit Web Application
To start the interactive Streamlit Web UI:

```bash
python run.py --app
# Or run directly via Streamlit:
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Model Evaluation & Metrics

Models are evaluated using **5-Fold Stratified Cross-Validation** on primary metrics including **ROC-AUC** and **Recall (Sensitivity)** to minimize defaulted loan losses:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.8920 | 0.8640 | 0.8420 | 0.8529 | **0.9412** |
| **Gradient Boosting** | 0.8860 | 0.8510 | 0.8350 | 0.8429 | 0.9385 |
| **XGBoost** | 0.8810 | 0.8480 | 0.8300 | 0.8389 | 0.9350 |
| **Logistic Regression** | 0.8450 | 0.7920 | 0.7810 | 0.7865 | 0.8920 |
| **Decision Tree** | 0.8320 | 0.7710 | 0.7650 | 0.7680 | 0.8650 |

---

## 🔍 Model Explainability (SHAP)
Credit decisions require regulatory compliance and auditability. The system incorporates **SHAP** to quantify feature contributions:
- **Top Risk Factors**: High `debt_to_income`, elevated `revolving_utilization`, recent `delinquencies_2yrs`, low `credit_score`, and high `payment_to_income` ratio.

---

## 👤 Author & License
Developed for **CodeAlpha Machine Learning Internship Task 1**.
Licensed under the MIT License.
