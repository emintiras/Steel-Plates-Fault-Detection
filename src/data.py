"""
Data loading and preparation for Steel Plates Fault Detection.

Responsibilities:
  - Read CSV and validate expected columns
  - Convert one-hot encoded labels to a single Fault_Type column
  - Split into train / test with stratification
  - Scale features (StandardScaler fit on train only)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DataConfig


@dataclass
class SteelDataset:
    """Container for all data artefacts produced by load()."""
    df:           pd.DataFrame       
    features:     List[str]          
    fault_types:  List[str]          
    X_scaled:     np.ndarray         
    y:            np.ndarray         
    X_train:      np.ndarray
    X_test:       np.ndarray
    y_train:      np.ndarray
    y_test:       np.ndarray
    label_encoder: LabelEncoder
    scaler:       StandardScaler


def load(cfg: DataConfig) -> SteelDataset:
    """
    Load the UCI Steel Plates Faults CSV and return a SteelDataset.

    The CSV uses one-hot encoded labels; this function converts them to a
    single `Fault_Type` string column before splitting.
    """
    df_raw = pd.read_csv(cfg.path, index_col=cfg.index_col)

    missing = [c for c in cfg.fault_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing fault columns in CSV: {missing}")

    features    = [c for c in df_raw.columns if c not in cfg.fault_cols]
    fault_types = cfg.fault_cols         

    df_raw["Fault_Type"] = df_raw[cfg.fault_cols].idxmax(axis=1)
    df = df_raw[features + ["Fault_Type"]].copy()

    le     = LabelEncoder()
    scaler = StandardScaler()

    y        = le.fit_transform(df["Fault_Type"])
    X_scaled = scaler.fit_transform(df[features])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    print(f"Dataset : {df.shape[0]} samples  ×  {len(features)} features  ×  {len(fault_types)} classes")
    print(f"Train   : {len(X_train)}  |  Test: {len(X_test)}")
    print(df["Fault_Type"].value_counts().to_string())

    return SteelDataset(
        df=df,
        features=features,
        fault_types=fault_types,
        X_scaled=X_scaled,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=le,
        scaler=scaler,
    )
