"""
generate_dataset.py
Generates a synthetic patient dataset for HealthTrack.
Run once: python data/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000

diagnoses   = ["Diabetes Mellitus","Heart Failure","Chronic Kidney Disease",
                "COPD","Pneumonia","Hypertension","Sepsis","Other"]
discharges  = ["Home","Home with Home Care","Skilled Nursing Facility",
                "Inpatient Rehab","Against Medical Advice"]
insurances  = ["Medicare","Medicaid","Private","Uninsured","Other"]
genders     = ["Male","Female","Other"]

diag        = np.random.choice(diagnoses,  N, p=[.18,.15,.12,.10,.12,.16,.07,.10])
discharge   = np.random.choice(discharges, N, p=[.50,.20,.15,.10,.05])
insurance   = np.random.choice(insurances, N, p=[.35,.20,.30,.10,.05])
gender      = np.random.choice(genders,    N, p=[.48,.49,.03])

age             = np.random.randint(18, 90, N)
num_visits      = np.random.poisson(2.5, N)
length_of_stay  = np.random.randint(1, 21, N)
num_medications = np.random.randint(1, 20, N)
num_comorbid    = np.random.randint(0, 5,  N)

# Derive readmission label from risk factors
score = (
    (age >= 65).astype(float) * 0.20 +
    (num_visits >= 3).astype(float) * 0.20 +
    (length_of_stay >= 7).astype(float) * 0.15 +
    (num_medications >= 10).astype(float) * 0.10 +
    (np.isin(diag, ["Heart Failure","Sepsis","COPD"])).astype(float) * 0.15 +
    (discharge == "Against Medical Advice").astype(float) * 0.20 +
    num_comorbid * 0.04 +
    np.random.normal(0, 0.08, N)
)
readmitted = (score > 0.45).astype(int)

df = pd.DataFrame({
    "age": age, "gender": gender,
    "num_prior_visits": num_visits,
    "length_of_stay": length_of_stay,
    "diagnosis": diag,
    "discharge_type": discharge,
    "insurance": insurance,
    "num_medications": num_medications,
    "num_comorbidities": num_comorbid,
    "readmitted_30days": readmitted,
})

out = Path(__file__).parent / "patient_data.csv"
df.to_csv(out, index=False)
print(f"Saved {N} rows → {out}")
print(df["readmitted_30days"].value_counts())
