#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import sys
import warnings

warnings.filterwarnings("ignore")

DEFAULT_INVALID_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data(input_path: str):
    path = Path(input_path)
    if path.exists():
        df = pd.read_csv(path)
        print(f"[info] Loaded dataset: {path} -> shape {df.shape}")
        return df
    else:
        print(f"[error] Dataset not found at: {path}")
        sys.exit(1)

def replace_invalid_zeros(df: pd.DataFrame, cols=None):
    if cols is None:
        cols = DEFAULT_INVALID_ZERO_COLS
    
    df = df.copy()
    valid_cols = [c for c in cols if c in df.columns]
    
    for col in valid_cols:
        zeros = (df[col] == 0).sum()
        if zeros > 0:
            print(f"[info] Converting {zeros} zero values in '{col}' to NaN")
            df.loc[df[col] == 0, col] = np.nan
    
    return df

def build_pipeline(numeric_cols, n_neighbors=5):
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', KNNImputer(n_neighbors=n_neighbors)),
            ('scaler', MinMaxScaler())
        ]), numeric_cols)
    ], remainder='passthrough')

def main():
    input_path = "../diabetes.csv"
    output_path = "diabetes_preprocessing.csv"
    pipeline_out_path = "preprocessor_pipeline.joblib"
    target = "Outcome"

    print("[info] Loading dataset...")
    df = load_data(input_path)

    if target not in df.columns:
        print(f"[error] Target column '{target}' not found in dataset.")
        sys.exit(1)

    df_clean = replace_invalid_zeros(df)

    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    print(f"[info] Numeric columns detected: {numeric_cols}")

    preproc = build_pipeline(numeric_cols)

    X_pre = preproc.fit_transform(X)
    X_pre_df = pd.DataFrame(X_pre, columns=numeric_cols)
    X_pre_df[target] = y.values

    X_pre_df.to_csv(output_path, index=False)
    print(f"[info] Saved preprocessed dataset to: {output_path}")

    joblib.dump(preproc, pipeline_out_path)
    print(f"[info] Saved preprocessor to: {pipeline_out_path}")

    print("\n[done] Preprocessing Pipeline completed successfully.")

if __name__ == "__main__":
    main()
