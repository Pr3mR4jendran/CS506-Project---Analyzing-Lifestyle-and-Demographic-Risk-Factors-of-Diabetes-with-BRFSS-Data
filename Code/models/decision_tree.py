from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class DTConfig:
    criterion: str = "gini"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: Optional[str] = "balanced"
    random_state: int = 42

    param_grid: Dict[str, list] = field(
        default_factory=lambda: {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [None, 6, 12, 20],
            "clf__min_samples_split": [2, 10, 50],
            "clf__min_samples_leaf": [1, 5, 20],
        }
    )
    cv_splits: int = 3


class DecisionTreeModel:
    """
    Pipeline = [preprocess] -> [DecisionTreeClassifier]
    """

    def __init__(self, preprocess: BaseEstimator, cfg: Optional[DTConfig] = None):
        self.cfg = cfg or DTConfig()
        self.preprocess = preprocess
        self.pipeline: Optional[Pipeline] = None
        self.grid_: Optional[GridSearchCV] = None
        self.classes_: Optional[np.ndarray] = None

    def build_pipeline(self) -> Pipeline:
        clf = DecisionTreeClassifier(
            criterion=self.cfg.criterion,
            max_depth=self.cfg.max_depth,
            min_samples_split=self.cfg.min_samples_split,
            min_samples_leaf=self.cfg.min_samples_leaf,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
        )
        return Pipeline(
            [
                ("prep", self.preprocess),
                ("clf", clf),
            ]
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, tune: bool = False
    ) -> "DecisionTreeModel":
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

        auc_macro = None
        try:
            y_proba = self.predict_proba(X)
            auc_macro = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            pass

        importances = None
        try:
            clf = self.pipeline.named_steps["clf"]
            importances = getattr(clf, "feature_importances_", None)
        except Exception:
            pass

        return {
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc_macro_ovr": auc_macro,
            "best_params": getattr(self.grid_, "best_params_", None),
            "feature_importances": importances,
        }
