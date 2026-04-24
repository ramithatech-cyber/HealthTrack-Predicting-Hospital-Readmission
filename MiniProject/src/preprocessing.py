"""
src/preprocessing.py
Data preprocessing functions for HealthTrack.
Handles label encoding, feature selection, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
CAT_COLS = ["gender", "diagnosis", "discharge_type", "insurance"]

NUM_COLS = [
    "age", "num_prior_visits", "length_of_stay",
    "num_medications", "num_comorbidities"
]

FEATURES = [
    "age", "num_prior_visits", "length_of_stay",
    "num_medications", "num_comorbidities",
    "gender_enc", "diagnosis_enc", "discharge_type_enc", "insurance_enc"
]

TARGET = "readmitted_30days"


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """Load raw CSV dataset."""
    if filepath is None:
        filepath = Path(__file__).parent.parent / "dataset" / "raw_data" / "patient_data.csv"
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def check_missing(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column."""
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing[missing > 0] if missing.sum() > 0 else "  None found.")
    return missing


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all categorical columns.
    Returns (encoded_dataframe, encoders_dict).
    """
    df = df.copy()
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  Encoded '{col}': {list(le.classes_)}")
    return df, encoders


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target)."""
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42):
    """Split into train and test sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, filepath: str = None):
    """Save the encoded dataframe to processed_data folder."""
    if filepath is None:
        filepath = (Path(__file__).parent.parent /
                    "dataset" / "processed_data" / "patient_data_encoded.csv")
    df.to_csv(filepath, index=False)
    print(f"Processed data saved → {filepath}")


def run_preprocessing_pipeline():
    """Run the full preprocessing pipeline end-to-end."""
    print("=" * 50)
    print("HealthTrack Preprocessing Pipeline")
    print("=" * 50)

    # 1. Load
    df = load_raw_data()

    # 2. Inspect
    print(f"\nShape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    check_missing(df)

    # 3. Encode
    print("\nEncoding categorical columns...")
    df_enc, encoders = encode_categoricals(df)

    # 4. Save processed
    save_processed_data(df_enc)

    # 5. Feature matrix
    X, y = get_feature_matrix(df_enc)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # 6. Split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nPreprocessing complete.")
    return df_enc, encoders, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing_pipeline()
