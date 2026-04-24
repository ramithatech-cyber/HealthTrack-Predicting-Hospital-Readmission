# HealthTrack: Predicting Hospital Readmission

**Mini Project – Data Science**
---

## Abstract

Hospital readmission within 30 days of discharge is a major challenge in healthcare, leading to increased costs and poor patient outcomes. HealthTrack is a machine learning–based system designed to predict the likelihood of a patient being readmitted within 30 days of discharge using clinical intake data. The system uses a Random Forest classifier trained on 2,000 synthetic patient records, achieving 82% accuracy and a ROC-AUC of 0.899. The project includes a complete data pipeline, model training module, and an interactive Streamlit web interface that allows healthcare workers to input patient details and receive an instant risk assessment (Low, Medium, or High). This tool aims to assist care teams in flagging high-risk patients early so that preventive steps can be taken before discharge.

---

## Problem Statement

Unplanned hospital readmissions within 30 days of discharge are costly and often preventable. According to healthcare studies, nearly 20% of Medicare patients are readmitted within 30 days. The goal of this project is to build a predictive model that uses patient clinical data collected at the time of discharge to estimate the probability of readmission. This allows hospitals to intervene early for high-risk patients by arranging follow-up care, medication reviews, or extended monitoring.

---

## Dataset Source

The dataset used in this project is **synthetically generated** using Python's NumPy and Pandas libraries to simulate realistic patient records. It contains 2,000 patient records with the following features:

| Feature | Description |
|---|---|
| age | Patient age in years |
| gender | Male / Female / Other |
| num_prior_visits | Number of hospital visits in the past 1 year |
| length_of_stay | Duration of hospital stay in days |
| diagnosis | Primary diagnosis at admission |
| discharge_type | How the patient was discharged |
| insurance | Insurance type |
| num_medications | Number of medications at discharge |
| num_comorbidities | Number of other existing conditions |
| readmitted_30days | Target: 1 = Readmitted, 0 = Not Readmitted |

The data generation script is located at: `dataset/generate_dataset.py`
The raw dataset is located at: `dataset/raw_data/patient_data.csv`

---

## Methodology / Workflow

```
Raw Data Generation
       ↓
Exploratory Data Analysis (EDA)
       ↓
Data Preprocessing (encoding, cleaning)
       ↓
Model Training (Random Forest)
       ↓
Model Evaluation (Accuracy, ROC-AUC, Confusion Matrix)
       ↓
Deployment via Streamlit Web App
```

**Step-by-step:**

1. **Data Generation** — Synthetic patient data is created using controlled probability distributions to simulate real-world clinical patterns.
2. **EDA** — Distributions, correlations, and readmission rates are explored using Pandas and Matplotlib/Seaborn (see `notebooks/data_understanding.ipynb`).
3. **Preprocessing** — Categorical variables (gender, diagnosis, discharge type, insurance) are label-encoded. No missing values exist in the synthetic dataset.
4. **Model Training** — A Random Forest Classifier with 150 trees is trained on 80% of data. `class_weight='balanced'` handles any class imbalance.
5. **Evaluation** — Model is evaluated on 20% hold-out test set and using 5-fold cross-validation.
6. **Deployment** — A Streamlit app (`app.py`) provides an interactive UI with three tabs: Predict, Dataset Insights, and Model Report.

---

## Tools Used

| Tool | Purpose |
|---|---|
| Python 3.11+ | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning (Random Forest, encoders, metrics) |
| Matplotlib | Plotting and visualization |
| Seaborn | Statistical data visualization |
| Streamlit | Web application framework |
| Joblib | Model serialization (saving/loading .pkl files) |
| Jupyter Notebook | Exploratory analysis and preprocessing |

---

## Results / Findings

| Metric | Value |
|---|---|
| Test Accuracy | 82.0% |
| ROC-AUC Score | 0.899 |
| 5-Fold CV Accuracy | 82.8% |
| Precision (Readmitted) | 0.83 |
| Recall (Readmitted) | 0.81 |
| F1-Score (Readmitted) | 0.82 |

**Key Findings:**
- Prior hospital visits and age are the strongest predictors of readmission.
- Length of stay and number of medications also strongly influence risk.
- Patients discharged "Against Medical Advice" have significantly higher readmission risk.
- Diagnoses like Heart Failure and Sepsis are associated with the highest readmission rates.

---

## Project Structure

```
MiniProject/
│
├── README.md
├── requirements.txt
├── app.py                        ← Streamlit web application
│
├── docs/
│   ├── abstract.pdf
│   ├── problem_statement.pdf
│   └── presentation.pptx
│
├── dataset/
│   ├── generate_dataset.py       ← Script to create synthetic data
│   ├── raw_data/
│   │   └── patient_data.csv      ← Original 2000-row dataset
│   └── processed_data/
│       └── patient_data_encoded.csv ← Label-encoded dataset
│
├── notebooks/
│   ├── data_understanding.ipynb  ← EDA and visualizations
│   ├── preprocessing.ipynb       ← Data cleaning and encoding
│   └── visualization.ipynb       ← Charts and insight plots
│
├── src/
│   ├── preprocessing.py          ← Encoding and data prep functions
│   ├── analysis.py               ← Statistical analysis functions
│   └── model.py                  ← Model training and evaluation
│
├── models/
│   ├── train_model.py            ← Training script
│   ├── rf_model.pkl              ← Saved trained model
│   ├── encoders.pkl              ← Saved label encoders
│   └── model_meta.json           ← Metrics and feature importances
│
├── utils/
│   └── helpers.py                ← Shared helper functions
│
├── outputs/
│   ├── graphs/                   ← Saved chart images
│   └── results/
│       └── classification_report.txt
│
└── report/
    └── mini_project_report.pdf
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate dataset
python dataset/generate_dataset.py

# 3. (Optional) Retrain model
python models/train_model.py

# 4. Launch web app
streamlit run app.py
# OR
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Team Members

| Name | Roll No. |
MAHARAJAN E RA2311026050022
RAMITHA A RA23110026050013
---
