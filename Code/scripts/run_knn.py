from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ..models.knn import KNNModel, KNNConfig

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "Results" / "BRFSS_2024_visualization.csv"

TARGET = "DIABETE4"
VALID_TARGET_CLASSES = {1, 3, 4}


def get_data_and_preprocess():
    """
    Minimal placeholder data/preprocess builder for a KNN smoke test.
    Replace with your real cleaning + ColumnTransformer when ready.
    """
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    if TARGET not in df.columns:
        raise KeyError(f"{TARGET} not found in {CSV_PATH}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin(VALID_TARGET_CLASSES)].copy()
    df.dropna(subset=[TARGET], inplace=True)

    candidate_cols = [c for c in df.columns if c != TARGET][:20]
    X = df[candidate_cols].copy()
    y = df[TARGET].astype(int)

    cat_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 15]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocess


def main():
    X_train, X_test, y_train, y_test, preprocess = get_data_and_preprocess()

    cfg = KNNConfig(
        n_neighbors=15,
        weights="distance",
        metric="minkowski",
        p=2,
    )

    model = KNNModel(preprocess=preprocess, cfg=cfg)

    model.fit(X_train, y_train, tune=False)
    metrics = model.evaluate(X_test, y_test)
    print("== KNN (no tuning) ==")
    print("Macro F1:", metrics["f1_macro"])
    print("ROC-AUC (macro OvR):", metrics["roc_auc_macro_ovr"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()

