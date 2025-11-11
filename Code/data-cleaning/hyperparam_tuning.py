import re
import csv
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def load_brfss(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def is_sentinel(x) -> bool:
    if pd.isna(x): 
        return True
    s = str(x).strip().lower()
    if s in {"", "nan", "none", "refused", "don’t know", "don't know"}:
        return True
    return any(p.fullmatch(s) for p in [re.compile(r"^7+$"), re.compile(r"^9+$")])


def effective_missing_rate(series: pd.Series) -> float:
    return series.map(is_sentinel).mean()


def detect_feature_types(df: pd.DataFrame, target: str) -> tuple[list, list, list]:
    codebook_ordinals = {
        "GENHLTH", "EDUCA", "_EDUCAG", "INCOME3", "_INCOMG1",
        "_BMI5CAT", "SLEPTIM1", "AGE5YR", "_AGEG5YR", "_AGE80",
        "_AGE65YR", "_AGE_G", "CHECKUP1", "POORHLTH",
        "PHYSHLTH", "MENTHLTH", "EMPLOY1", "MARITAL", "RENTHOM1"
    }

    dynamic_ordinals, categorical_cols, continuous_cols = [], [], []

    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            categorical_cols.append(col)
            continue
        nunique = series.nunique(dropna=True)
        if nunique < 2:
            continue
        unique_vals = np.sort(series.dropna().unique())
        if (
            2 <= nunique <= 15
            and np.issubdtype(series.dtype, np.number)
            and np.all(np.mod(unique_vals, 1) == 0)
        ):
            minv, maxv = int(unique_vals.min()), int(unique_vals.max())
            contiguous = set(range(minv, maxv + 1))
            actual = set(unique_vals.astype(int))
            if len(contiguous - actual) <= 2:
                dynamic_ordinals.append(col)
            else:
                categorical_cols.append(col)
        elif nunique <= 15:
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)

    ordinal_cols = sorted(set(codebook_ordinals).union(dynamic_ordinals).intersection(df.columns))
    categorical_cols = [c for c in categorical_cols if c not in ordinal_cols]
    continuous_cols = [c for c in continuous_cols if c not in ordinal_cols]

    return ordinal_cols, categorical_cols, continuous_cols


def build_preprocessor(df, ordinal_cols, categorical_cols, continuous_cols):
    ordinal_categories = [sorted(df[c].dropna().unique()) for c in ordinal_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("ord", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ordenc", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1))
            ]), ordinal_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols),
            ("num", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), continuous_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocess

def run_experiments(
    csv_path="../Data/BRFSS_2024.csv",
    mrt_values=[0.2, 0.3, 0.4],
    tau_values=[0.5, 0.6, 0.7],
    C_values=np.logspace(-3, 2, 10),
    target="DIABETE4",
    valid_classes={1, 3, 4},
    n_min_features=50,
    output_csv="../../Results/experiment_log.csv"
):
    df = load_brfss(csv_path)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].isin(valid_classes)].copy()
    df.dropna(subset=[target], inplace=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "mrt", "tau", "best_C", "val_f1", "features_kept"])

    for mrt, tau in product(mrt_values, tau_values):
        print(f"\n=== Running Experiment (mrt={mrt}, tau={tau}) ===")

        missing_rates = df.drop(columns=[target]).apply(effective_missing_rate)
        to_keep = missing_rates[missing_rates <= mrt].index.tolist()
        df_sub = df[to_keep + [target]].copy()

        categorical_cols, continuous_cols = [], []
        for col in to_keep:
            nunique = df_sub[col].nunique(dropna=True)
            if pd.api.types.is_numeric_dtype(df_sub[col]):
                if nunique <= 15:
                    categorical_cols.append(col)
                else:
                    continuous_cols.append(col)
            else:
                categorical_cols.append(col)

        y = df_sub[target].astype(int)
        cat_results, cont_results = [], []

        for col in categorical_cols:
            sub = df_sub[[col, target]].dropna()
            if sub.empty:
                continue
            tbl = pd.crosstab(sub[col], sub[target])
            if tbl.shape[0] < 2 or tbl.shape[1] < 2:
                continue
            try:
                chi2_stat, p, dof, exp = stats.chi2_contingency(tbl, correction=False)
                cat_results.append({"feature": col, "p_value": p})
            except Exception:
                pass

        for col in continuous_cols:
            sub = df_sub[[col, target]].dropna()
            if sub.empty:
                continue
            groups = [sub.loc[sub[target]==cls, col] for cls in sorted(valid_classes)]
            try:
                fstat, p = stats.f_oneway(*groups)
            except Exception:
                try:
                    stat, p = stats.kruskal(*groups, nan_policy="omit")
                except Exception:
                    continue
            cont_results.append({"feature": col, "p_value": p})

        all_results = pd.concat([pd.DataFrame(cat_results), pd.DataFrame(cont_results)], ignore_index=True)
        if all_results.empty:
            print(f"No valid features found for mrt={mrt}, tau={tau}. Skipping.")
            continue

        all_results.dropna(inplace=True)
        all_results.sort_values("p_value", inplace=True)
        all_results["importance_score"] = -np.log10(all_results["p_value"].clip(lower=1e-300))
        all_results["importance_norm"] = (
            all_results["importance_score"] - all_results["importance_score"].min()
        ) / (all_results["importance_score"].max() - all_results["importance_score"].min())

        selected_by_tau = all_results[all_results["importance_norm"] >= tau]["feature"].tolist()
        if len(selected_by_tau) < n_min_features:
            selected_by_tau = all_results.head(n_min_features)["feature"].tolist()

        final_features = selected_by_tau + [target]
        df_cleaned = df_sub[final_features].copy()

        ordinal_cols, categorical_cols, continuous_cols = detect_feature_types(df_cleaned, target=target)
        preprocess = build_preprocessor(df_cleaned, ordinal_cols, categorical_cols, continuous_cols)

        X = df_cleaned.drop(columns=[target])
        y = df_cleaned[target].astype(int)
        X_proc = preprocess.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_proc, y, test_size=0.2, random_state=42, stratify=y)

        val_scores, nonzero_counts = [], []
        for C in C_values:
            model = LogisticRegression(
                penalty='l1',
                solver='saga',
                C=C,
                max_iter=5000,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            f1 = f1_score(y_val, preds, average="macro")
            val_scores.append(f1)
            n_nonzero = np.sum(model.coef_ != 0)
            nonzero_counts.append(n_nonzero)

        best_idx = np.argmax(val_scores)
        best_C = C_values[best_idx]
        best_f1 = val_scores[best_idx]
        best_features = nonzero_counts[best_idx]

        print(f"mrt={mrt}, tau={tau} → Best C={best_C:.4f} | Val macro-F1={best_f1:.4f} | Features={best_features}")

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                mrt, tau, best_C, round(best_f1, 4), best_features
            ])

    print(f"\nAll experiments completed. Results saved to '{output_csv}'.")
    df_results = pd.read_csv(output_csv)
    best_row = df_results.loc[df_results["val_f1"].idxmax()]
    print("\nBest Configuration Found:")
    print(best_row)

    with open("../../Results/best_config.txt", "w") as f:
        f.write(best_row.to_string(index=False))

    print("\nBest config saved to 'best_config.txt'")
    return best_row

if __name__ == "__main__":
    run_experiments(
        csv_path="../../Data/BRFSS_2024.csv",
        mrt_values=[0.20, 0.30, 0.40], #tunable
        tau_values=[0.50, 0.60, 0.75], #tunable
        C_values=[0.05, 0.20, 1.00], #tunable
        target="DIABETE4"
    )