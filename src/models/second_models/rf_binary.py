import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

class BalancedRandomForestBinaryClassifier:
    """2단계: Balanced Random Forest 정상(1)/오류(0) 이진 분류기"""

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = 15, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced_subsample'
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BalancedRandomForestBinaryClassifier":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        inference_time_ms = ((time.time() - t0) / len(X_test)) * 1000.0

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        pos_f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        pos_prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
        pos_rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
        except Exception:
            roc_auc, pr_auc = 0.5, 0.0

        return {
            'model_name': 'Balanced Random Forest',
            'split_name': split_name,
            'num_samples': len(y_test),
            'accuracy': acc,
            'macro_f1': macro_f1,
            'pos_f1': pos_f1,
            'pos_precision': pos_prec,
            'pos_recall': pos_rec,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'inference_time_ms': inference_time_ms,
            'fps': 1000.0 / max(1e-4, inference_time_ms),
            'predictions': preds,
            'probabilities': probs
        }
