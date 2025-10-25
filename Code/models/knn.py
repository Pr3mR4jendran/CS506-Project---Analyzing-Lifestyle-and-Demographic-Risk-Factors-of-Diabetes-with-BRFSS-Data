from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class KNNConfig:
    n_neighbors: int = 15
    weights: str = "distance"
    metric: str = "minkowski"
    p: int = 2
    n_jobs: Optional[int] = -1

    param_grid: Dict[str, list] = field(
        default_factory=lambda: {
            "clf__n_neighbors": [3, 5, 11, 15, 31],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        }
    )
    cv_splits: int = 3


class KNNModel:
    """
    Wraps a Pipeline = [preprocess] -> [KNN].
    """

    def __init__(self, preprocess: BaseEstimator, cfg: Optional[KNNConfig] = None):
        self.cfg = cfg or KNNConfig()
        self.preprocess = preprocess
        self.pipeline: Optional[Pipeline] = None
        self.grid_: Optional[GridSearchCV] = None
        self.classes_: Optional[np.ndarray] = None

    def build_pipeline(self) -> Pipeline:
        clf = KNeighborsClassifier(
            n_neighbors=self.cfg.n_neighbors,
            weights=self.cfg.weights,
            metric=self.cfg.metric,
            p=self.cfg.p,
            n_jobs=self.cfg.n_jobs,
        )
        return Pipeline([("prep", self.preprocess), ("clf", clf)])

    def fit(self, X: pd.DataFrame, y: pd.Series, tune: bool = False) -> "KNNModel":
        self.pipeline = self.build_pipeline()
        if tune:
            cv = StratifiedKFold(
                n_splits=self.cfg.cv_splits, shuffle=True, random_state=42
            )
            self.grid_ = GridSearchCV(
                estimator=self.pipeline,
                param_grid=self.cfg.param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
            )
            self.grid_.fit(X, y)
            self.pipeline = self.grid_.best_estimator_
        else:
            self.pipeline.fit(X, y)

        self.classes_ = np.unique(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        y_pred = self.predict(X)
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=self.classes_)

        auc = None
        try:
            y_proba = self.predict_proba(X)
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            pass

        return {
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc_macro_ovr": auc,
            "best_params": getattr(self.grid_, "best_params_", None),
        }
