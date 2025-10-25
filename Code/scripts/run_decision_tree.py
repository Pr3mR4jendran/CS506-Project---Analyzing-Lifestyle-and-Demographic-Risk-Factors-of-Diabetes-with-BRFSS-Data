from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

from models.decision_tree import DecisionTreeModel, DTConfig

CLEAN_CSV_PATH = "../Results/BRFSS_2024_visualization.csv"
TARGET = "DIABETE4"
VALID_TARGET_CLASSES = {1, 3, 4}


def get_data_and_preprocess():
    """
    Replace this placeholder with your *real* cleaning + ColumnTransformer.
    This is just enough to smoke-test the skeleton. Use light columns at first.
    """
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(CLEAN_CSV_PATH, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    if TARGET not in df.columns:
        raise KeyError(f"{TARGET} not found in {CLEAN_CSV_PATH}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin(VALID_TARGET_CLASSES)].copy()
    df.dropna(subset=[TARGET], inplace=True)

    feature_cols = [c for c in df.columns if c != TARGET][:25]
    X = df[feature_cols].copy()
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
                        (
                            "ohe",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
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
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocess


def main():
    X_train, X_test, y_train, y_test, preprocess = get_data_and_preprocess()

    cfg = DTConfig(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )

    model = DecisionTreeModel(preprocess=preprocess, cfg=cfg)

    model.fit(X_train, y_train, tune=False)
    metrics = model.evaluate(X_test, y_test)
    print("== Decision Tree (no tuning) ==")
    print("Macro F1:", metrics["f1_macro"])
    print("ROC-AUC (macro OvR):", metrics["roc_auc_macro_ovr"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    if metrics.get("feature_importances") is not None:
        print("Feature importances shape:", metrics["feature_importances"].shape)


if __name__ == "__main__":
    main()
