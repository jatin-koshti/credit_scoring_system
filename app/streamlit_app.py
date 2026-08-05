"""
Streamlit Web Application for Credit Scoring & Loan Risk Prediction System.
CodeAlpha Machine Learning Internship - Task 1.
"""

import sys
import os
import io

# Add parent path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Page Configuration
st.set_page_config(
    page_title="Credit Score & Loan Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Simple, clean, modern layout and design)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-low {
        background-color: #DCFCE7;
        color: #166534;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .status-badge-med {
        background-color: #FEF08A;
        color: #854D0E;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .status-badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model artifacts
@st.cache_resource
def load_artifacts():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, "models", "best_model.joblib")
    prep_path = os.path.join(base_dir, "models", "preprocessor.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        # Auto-train if artifacts are missing
        from src.data_loader import load_credit_data
        from src.preprocessing import prepare_data_and_preprocessor, save_preprocessor
        from src.train import train_and_tune_models, save_best_model, get_candidate_models
        from sklearn.ensemble import RandomForestClassifier
        
        raw_csv = os.path.join(base_dir, "data", "raw", "credit_data.csv")
        df = load_credit_data(raw_csv)
        X_tr, X_te, y_tr, y_te, prep_pipe, f_names = prepare_data_and_preprocessor(df)
        save_preprocessor(prep_pipe, f_names, prep_path)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)
        save_best_model(rf, "Random Forest", model_path)
        
    model_payload = joblib.load(model_path)
    prep_payload = joblib.load(prep_path)
    
    return model_payload["model"], model_payload["model_name"], prep_payload["pipeline"], prep_payload["feature_names"]

# Load system artifacts
try:
    model, model_name, preprocessor, feature_names = load_artifacts()
    artifacts_ready = True
except Exception as e:
    artifacts_ready = False
    st.error(f"Error loading model artifacts: {e}")

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric-folders/100/bank-cards.png", width=70)
st.sidebar.title("Credit Risk System")
st.sidebar.caption("CodeAlpha ML Task 1")

navigation = st.sidebar.radio(
    "Navigation Menu",
    ["👤 Single Applicant Assessment", "📁 Batch Risk Scoring", "📊 Model Insights & Diagnostics", "ℹ️ Methodology"]
)

# ----------------------------------------------------
# PAGE 1: SINGLE APPLICANT ASSESSMENT
# ----------------------------------------------------
if navigation == "👤 Single Applicant Assessment":
    st.markdown("<div class='main-header'>Credit Scoring & Loan Risk Assessment</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Enter applicant details to predict credit risk, probability of default, and risk status.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📋 Applicant Information")
        with st.form("applicant_form"):
            f_col1, f_col2 = st.columns(2)
            
            with f_col1:
                age = st.number_input("Age", min_value=18, max_value=90, value=35)
                annual_income = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=65000, step=5000)
                credit_score = st.slider("Credit Bureau Score (FICO)", min_value=300, max_value=850, value=680)
                debt_to_income = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.28, step=0.01)
                revolving_utilization = st.slider("Revolving Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
                num_open_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=6)
                
            with f_col2:
                delinquencies_2yrs = st.number_input("Delinquencies (Past 2 Yrs)", min_value=0, max_value=10, value=0)
                num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
                employment_length = st.number_input("Employment Length (Years)", min_value=0, max_value=50, value=5)
                loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=100000, value=15000, step=1000)
                home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
                loan_purpose = st.selectbox("Loan Purpose", [
                    "DEBT_CONSOLIDATION", "HOME_IMPROVEMENT", "CREDIT_CARD", "SMALL_BUSINESS", "PERSONAL"
                ])
                
            submit_btn = st.form_submit_button("⚡ Predict Loan Risk", use_container_width=True)
            
    with col2:
        st.subheader("🎯 Prediction & Risk Rating")
        
        if submit_btn and artifacts_ready:
            input_df = pd.DataFrame([{
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
                'loan_purpose': loan_purpose
            }])
            
            # Transform and Predict
            X_proc = preprocessor.transform(input_df)
            prob_default = float(model.predict_proba(X_proc)[0, 1])
            pred_class = int(model.predict(X_proc)[0])
            
            # Risk Category & Badge
            if prob_default < 0.25:
                risk_tier = "LOW RISK (APPROVED)"
                badge_class = "status-badge-low"
                recommendation = "✅ Applicant exhibits strong creditworthiness and low probability of default."
            elif prob_default < 0.55:
                risk_tier = "MODERATE RISK (REVIEW REQUIRED)"
                badge_class = "status-badge-med"
                recommendation = "⚠️ Secondary underwriting review suggested due to moderate risk indicators."
            else:
                risk_tier = "HIGH RISK (REJECTED / HIGH DEFAULT)"
                badge_class = "status-badge-high"
                recommendation = "❌ High risk of loan default. Elevated DTI, low score, or high revolving debt."
                
            st.markdown(f"<div class='{badge_class}'>{risk_tier}</div>", unsafe_allow_html=True)
            st.write("")
            
            # Gauge Chart for Probability of Default
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = round(prob_default * 100, 1),
                number = {'suffix': "%"},
                title = {'text': "Estimated Default Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1E293B"},
                    'steps': [
                        {'range': [0, 25], 'color': "#DCFCE7"},
                        {'range': [25, 55], 'color': "#FEF08A"},
                        {'range': [55, 100], 'color': "#FEE2E2"}
                    ]
                }
            ))
            fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(recommendation)
            
            # Key Metrics Summary
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Payment-to-Income", f"{(loan_amount/36)/(annual_income/12)*100:.1f}%")
            m_col2.metric("Loan-to-Income Ratio", f"{(loan_amount/annual_income)*100:.1f}%")
            
        else:
            st.info("Fill out the applicant form on the left and click **Predict Loan Risk**.")

# ----------------------------------------------------
# PAGE 2: BATCH RISK SCORING
# ----------------------------------------------------
elif navigation == "📁 Batch Risk Scoring":
    st.markdown("<div class='main-header'>Batch Credit Risk Scoring</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload a CSV file containing applicant records to score loan applications in batch.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Applicants CSV File", type=["csv"])
    
    if uploaded_file is not None and artifacts_ready:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(df_batch)}** records.")
        
        try:
            X_batch_proc = preprocessor.transform(df_batch)
            probs = model.predict_proba(X_batch_proc)[:, 1]
            preds = model.predict(X_batch_proc)
            
            df_batch['Default_Probability'] = np.round(probs, 4)
            df_batch['Predicted_Risk'] = np.where(probs >= 0.50, 'High Risk', 'Good Risk')
            
            st.dataframe(df_batch.head(20), use_container_width=True)
            
            # Download Scored CSV
            csv_buffer = io.StringIO()
            df_batch.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Scored CSV Report",
                data=csv_buffer.getvalue(),
                file_name="credit_risk_batch_scored.csv",
                mime="text/csv"
            )
        except Exception as err:
            st.error(f"Error processing batch file: {err}. Ensure required columns match sample dataset.")

# ----------------------------------------------------
# PAGE 3: MODEL INSIGHTS & DIAGNOSTICS
# ----------------------------------------------------
elif navigation == "📊 Model Insights & Diagnostics":
    st.markdown("<div class='main-header'>Model Diagnostics & Explainability</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>Active Winning Model: <b>{model_name}</b></div>", unsafe_allow_html=True)
    
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))
    
    t1, t2, t3 = st.tabs(["📈 Performance Metrics", "🎯 Feature Importance", "🔄 ROC Curve & Confusion Matrix"])
    
    with t1:
        summary_csv = os.path.join(reports_dir, "model_performance_summary.csv")
        if os.path.exists(summary_csv):
            df_perf = pd.read_csv(summary_csv)
            st.table(df_perf)
        else:
            st.info("Train models using `python run.py --train` to view full evaluation metric summaries.")
            
    with t2:
        from src.explainability import get_feature_importances
        df_imp = get_feature_importances(model, feature_names).head(12)
        fig_imp = px.bar(
            df_imp, x="Importance", y="Feature", orientation='h',
            title="Top Feature Importances",
            color="Importance", color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with t3:
        roc_img = os.path.join(reports_dir, "roc_curve_comparison.png")
        cm_img = os.path.join(reports_dir, "confusion_matrix.png")
        
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(roc_img):
                st.image(roc_img, caption="ROC Curves")
        with c2:
            if os.path.exists(cm_img):
                st.image(cm_img, caption="Confusion Matrix")

# ----------------------------------------------------
# PAGE 4: METHODOLOGY
# ----------------------------------------------------
else:
    st.markdown("<div class='main-header'>Methodology & Architecture</div>", unsafe_allow_html=True)
    st.markdown("""
    ### Machine Learning Pipeline Overview
    1. **Data Ingestion**: Clean & robust dataset ingestion supporting demographic and financial indicators.
    2. **Feature Engineering**: Automated generation of DTI ratios, Loan-to-Income, and Payment Burden metrics.
    3. **Preprocessing Pipeline**: Imputation, scaling, and categorical encoding via Scikit-Learn `ColumnTransformer`.
    4. **Model Tuning**: Hyperparameter tuning via 5-Fold Stratified Cross Validation.
    5. **Explainability**: SHAP (SHapley Additive exPlanations) and feature weight analysis for audit compliance.
    """)
