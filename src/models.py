"""
Model training and evaluation for Steel Plates Fault Detection.

Supports: Random Forest, Gradient Boosting, Logistic Regression, CatBoost.
CatBoost is optional — all other models run if it is not installed.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

from .config import ModelsConfig
from .data import SteelDataset

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


@dataclass
class ModelResult:
    name:        str
    model:       Any
    y_pred:      np.ndarray
    y_prob:      np.ndarray
    accuracy:    float
    f1:          float
    roc_auc:     float
    cv_mean:     float
    cv_std:      float
    importances: Optional[np.ndarray] = None  

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "f1":       self.f1,
            "roc_auc":  self.roc_auc,
            "cv_mean":  self.cv_mean,
            "cv_std":   self.cv_std,
        }


class FaultClassifierTrainer:
    """
    Builds and evaluates all classifiers.

    Usage
    -----
    trainer = FaultClassifierTrainer(cfg.models)
    results = trainer.fit_evaluate(dataset)
    best    = trainer.best(results)
    """

    def __init__(self, cfg: ModelsConfig):
        self.cfg = cfg

    def _build_models(self) -> Dict[str, Any]:
        cfg = self.cfg
        models: Dict[str, Any] = {
            "Random Forest": RandomForestClassifier(
                n_estimators=cfg.random_forest.n_estimators,
                random_state=cfg.random_forest.random_state,
                n_jobs=cfg.random_forest.n_jobs,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=cfg.gradient_boosting.n_estimators,
                random_state=cfg.gradient_boosting.random_state,
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=cfg.logistic_regression.max_iter,
                random_state=cfg.logistic_regression.random_state,
            ),
        }
        if CATBOOST_AVAILABLE:
            cb = cfg.catboost
            models["CatBoost"] = CatBoostClassifier(
                iterations=cb.iterations,
                learning_rate=cb.learning_rate,
                depth=cb.depth,
                random_seed=cb.random_seed,
                verbose=cb.verbose,
                loss_function=cb.loss_function,
                eval_metric=cb.eval_metric,
            )
        else:
            print("⚠ CatBoost not installed — skipping. Run: pip install catboost")
        return models

    def fit_evaluate(self, dataset: SteelDataset) -> Dict[str, ModelResult]:
        models = self._build_models()
        results: Dict[str, ModelResult] = {}

        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(dataset.X_train, dataset.y_train)

            y_pred = model.predict(dataset.X_test)
            y_prob = model.predict_proba(dataset.X_test)

            # CatBoost is not thread-safe with n_jobs=-1 in cross_val_score
            n_jobs_cv = 1 if name == "CatBoost" else -1
            cv = cross_val_score(
                model, dataset.X_scaled, dataset.y,
                cv=self.cfg.cv_folds,
                scoring="f1_weighted",
                n_jobs=n_jobs_cv,
            )

            acc     = float(accuracy_score(dataset.y_test, y_pred))
            f1      = float(f1_score(dataset.y_test, y_pred, average="weighted"))
            roc_auc = float(roc_auc_score(
                dataset.y_test, y_prob, multi_class="ovr", average="weighted"
            ))

            print(f"    Acc={acc:.3f}  F1={f1:.3f}  AUC={roc_auc:.3f}  "
                  f"CV={cv.mean():.3f}±{cv.std():.3f}")

            results[name] = ModelResult(
                name=name, model=model,
                y_pred=y_pred, y_prob=y_prob,
                accuracy=acc, f1=f1, roc_auc=roc_auc,
                cv_mean=float(cv.mean()), cv_std=float(cv.std()),
            )

        return results

    def compute_permutation_importance(
        self,
        result: ModelResult,
        dataset: SteelDataset,
    ) -> np.ndarray:
        perm = permutation_importance(
            result.model,
            dataset.X_test,
            dataset.y_test,
            n_repeats=self.cfg.permutation_repeats,
            random_state=42,
            n_jobs=-1,
        )
        result.importances = perm.importances_mean
        return perm.importances_mean

    @staticmethod
    def best(results: Dict[str, ModelResult]) -> ModelResult:
        return max(results.values(), key=lambda r: r.f1)
