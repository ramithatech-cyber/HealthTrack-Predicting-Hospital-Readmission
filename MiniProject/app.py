"""
app.py  –  HealthTrack: Hospital Readmission Prediction
Main entry point. Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Make sure utils/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthTrack – Readmission Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Slightly tighten the main container */
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

    /* Section headers */
    .section-header {
        background-color: #f0f4f8;
        border-left: 4px solid #4a7fa5;
        padding: 6px 12px;
        margin: 1.2rem 0 0.8rem 0;
        font-size: 1rem;
        font-weight: 600;
        color: #2c3e50;
        border-radius: 2px;
    }

    /* Result box */
    .result-box {
        background: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 14px 18px;
        margin-top: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        margin-top: 3rem;
        border-top: 1px solid #eee;
        padding-top: 0.8rem;
    }

    /* Tab styling – keep it subtle */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.88rem;
        padding: 6px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model once (cached) ──────────────────────────────────────────────────
from utils.helpers import load_model, load_data

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model()

@st.cache_data(show_spinner=False)
def get_data():
    return load_data()

clf, encoders, meta = get_model()
df                  = get_data()


# ── Main title ────────────────────────────────────────────────────────────────
st.markdown("# 🏥 HealthTrack: Hospital Readmission Prediction")
st.write(
    "Fill in the patient details below and click **Predict** to estimate "
    "the risk of readmission within 30 days of discharge."
)
st.markdown("---")

# ── Tabs for multi-page feel without a multi-page setup ───────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Predict", "📊 Dataset Insights", "📈 Model Report"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Section 1: Patient Information ────────────────────────────────────────
    st.markdown('<div class="section-header">Section 1 — Patient Information</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=0, max_value=120,
                              value=None, placeholder="e.g. 65")
    with c2:
        gender = st.selectbox("Gender",
                              ["— Select —", "Male", "Female", "Other"])
    with c3:
        insurance = st.selectbox("Insurance Type",
                                 ["— Select —", "Medicare", "Medicaid",
                                  "Private", "Uninsured", "Other"])

    c4, c5, c6 = st.columns(3)
    with c4:
        num_visits = st.number_input("Prior Hospital Visits (last 1 yr)",
                                     min_value=0, max_value=100,
                                     value=None, placeholder="e.g. 3")
    with c5:
        length_of_stay = st.number_input("Length of Stay (days)",
                                         min_value=0, max_value=365,
                                         value=None, placeholder="e.g. 7")
    with c6:
        num_medications = st.number_input("Medications at Discharge",
                                          min_value=0, max_value=50,
                                          value=None, placeholder="e.g. 5")

    c7, c8 = st.columns(2)
    with c7:
        diagnosis = st.selectbox(
            "Primary Diagnosis",
            ["— Select —", "Diabetes Mellitus", "Heart Failure",
             "Chronic Kidney Disease", "COPD", "Pneumonia",
             "Hypertension", "Sepsis", "Other"]
        )
    with c8:
        discharge_type = st.selectbox(
            "Discharge Disposition",
            ["— Select —", "Home", "Home with Home Care",
             "Skilled Nursing Facility", "Inpatient Rehab",
             "Against Medical Advice"]
        )

    comorbid_list = st.multiselect(
        "Comorbid Conditions (select all that apply)",
        ["Hypertension", "Diabetes", "Obesity", "Heart Disease",
         "COPD", "Depression", "Chronic Kidney Disease", "None"],
        help="Select every condition that applies. Leave blank if none."
    )
    num_comorbidities = len([c for c in comorbid_list if c != "None"])

    st.markdown("---")

    # ── Section 2: Prediction ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Section 2 — Prediction</div>',
                unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict Readmission Risk", type="primary")

    if predict_btn:
        # Validation
        errors = []
        if age         is None:               errors.append("Age")
        if gender      == "— Select —":        errors.append("Gender")
        if num_visits  is None:               errors.append("Prior Hospital Visits")
        if length_of_stay is None:            errors.append("Length of Stay")
        if num_medications is None:           errors.append("Medications at Discharge")
        if diagnosis   == "— Select —":        errors.append("Primary Diagnosis")
        if discharge_type == "— Select —":     errors.append("Discharge Disposition")
        if insurance   == "— Select —":        errors.append("Insurance Type")

        if errors:
            st.warning(f"Please fill in the following required field(s): **{', '.join(errors)}**")
        else:
            from utils.helpers import predict

            patient = {
                "age":              age,
                "gender":           gender,
                "num_prior_visits": num_visits,
                "length_of_stay":   length_of_stay,
                "num_medications":  num_medications,
                "num_comorbidities": num_comorbidities,
                "diagnosis":        diagnosis,
                "discharge_type":   discharge_type,
                "insurance":        insurance,
            }

            risk, prob = predict(clf, encoders, patient)

            # Display result
            st.markdown("**Prediction Result**")

            if risk == "Low":
                st.success(f"✅ Risk Level: **{risk}**  |  Probability: **{prob*100:.1f}%**")
            elif risk == "Medium":
                st.warning(f"⚠️ Risk Level: **{risk}**  |  Probability: **{prob*100:.1f}%**")
            else:
                st.error(f"🚨 Risk Level: **{risk}**  |  Probability: **{prob*100:.1f}%**")

            st.progress(float(prob))

            # Breakdown table
            st.markdown("**Input Summary**")
            summary = {
                "Age": age, "Gender": gender, "Insurance": insurance,
                "Prior Visits": num_visits, "Length of Stay (days)": length_of_stay,
                "Medications": num_medications, "Diagnosis": diagnosis,
                "Discharge Type": discharge_type,
                "Comorbidities": ", ".join(comorbid_list) if comorbid_list else "None",
            }
            import pandas as _pd
            st.dataframe(
                _pd.DataFrame(summary.items(), columns=["Field", "Value"]),
                use_container_width=True, hide_index=True
            )

            st.caption(
                "ℹ️ This prediction uses a Random Forest model trained on synthetic "
                "patient data. Do not use for real clinical decisions."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – DATASET INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    from utils.helpers import (
        chart_readmission_distribution,
        chart_age_readmission,
        chart_readmission_by_diagnosis,
        chart_visits_vs_readmission,
    )

    st.markdown('<div class="section-header">Section 3 — Dataset Insights</div>',
                unsafe_allow_html=True)
    st.caption(
        "Charts below are derived from the 2,000-patient training dataset. "
        "They illustrate patterns that inform the model's predictions."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Readmission Distribution**")
        st.pyplot(chart_readmission_distribution(df))

    with col_b:
        st.write("**Age Distribution by Readmission**")
        st.pyplot(chart_age_readmission(df))

    col_c, col_d = st.columns(2)
    with col_c:
        st.write("**Readmission Rate by Diagnosis**")
        st.pyplot(chart_readmission_by_diagnosis(df))

    with col_d:
        st.write("**Prior Visits vs Readmission**")
        st.pyplot(chart_visits_vs_readmission(df))

    # Raw data preview (collapsed)
    with st.expander("🗂️ View Raw Dataset (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Total rows: {len(df)} | Columns: {list(df.columns)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    from utils.helpers import chart_feature_importance, chart_confusion_matrix

    st.markdown('<div class="section-header">Model Evaluation Report</div>',
                unsafe_allow_html=True)
    st.caption("Performance metrics from the hold-out test set (20% of data).")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy",  f"{meta['accuracy']*100:.1f}%")
    m2.metric("ROC-AUC Score",  f"{meta['roc_auc']:.3f}")
    m3.metric("5-Fold CV Acc.", f"{meta['cv_accuracy']*100:.1f}%")
    m4.metric("Training Size",  f"{meta['train_size']} patients")

    st.markdown("---")

    col_e, col_f = st.columns(2)
    with col_e:
        st.write("**Feature Importance**")
        st.pyplot(chart_feature_importance(meta))

    with col_f:
        st.write("**Confusion Matrix**")
        st.pyplot(chart_confusion_matrix(meta))

    # Model details expander
    with st.expander("🔧 Model Configuration"):
        st.write("**Algorithm:** Random Forest Classifier (scikit-learn)")
        st.write("**Hyperparameters:**")
        st.code(
            "n_estimators  = 150\n"
            "max_depth     = 8\n"
            "min_samples_leaf = 10\n"
            "class_weight  = 'balanced'\n"
            "random_state  = 42",
            language="python"
        )
        st.write("**Features used:**")
        st.write(", ".join(meta["features"]))
        st.write("**Target variable:** `readmitted_30days` (0 = No, 1 = Yes)")

    with st.expander("📐 How predictions are made"):
        st.write(
            "1. Patient inputs are collected from the form (Section 1).\n"
            "2. Categorical fields (gender, diagnosis, discharge type, insurance) "
            "are label-encoded using the same encoders fitted on training data.\n"
            "3. The 9-feature vector is passed to `clf.predict_proba()`, which "
            "returns the probability of the positive class (readmitted).\n"
            "4. Probability < 0.35 → **Low**, 0.35–0.65 → **Medium**, > 0.65 → **High**."
        )



