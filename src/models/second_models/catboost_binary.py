# -*- coding: utf-8 -*-
"""
Cost-Sensitive / Regularized CatBoost Binary Classifier (Format A: 161D)
"""
import os
import sys
import time
import numpy as np
from typing import Dict, Any, Optional
from catboost import CatBoostClassifier

class RegularizedCatBoostBinaryClassifier:
    def __init__(
        self,
        depth: int = 6,
        learning_rate: float = 0.08,
        l2_leaf_reg: float = 5.0,
        iterations: int = 200,
        scale_pos_weight: float = 17.0,
        random_seed: int = 42,
        verbose: bool = False
    ):
        self.model = CatBoostClassifier(
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            iterations=iterations,
            scale_pos_weight=scale_pos_weight,
            random_seed=random_seed,
            verbose=verbose,
            thread_count=-1
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: Optional[list] = None):
        self.model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=20 if eval_set else None, verbose=False)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
