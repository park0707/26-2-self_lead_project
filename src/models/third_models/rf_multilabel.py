import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score

class MultiOutputRandomForestClassifier:
    """3단계: Multi-Output Random Forest 세부 결함 다중 라벨 분류기"""

    def __init__(self, n_estimators: int = 150, max_depth: int = 12, random_state: int = 42):
        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced'
        )
        self.model = MultiOutputClassifier(base_rf, n_jobs=-1)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "MultiOutputRandomForestClassifier":
        self.model.fit(X_train, Y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        preds = self.predict(X_test)
        inference_time_ms = ((time.time() - t0) / max(1, len(X_test))) * 1000.0

        subset_acc = accuracy_score(Y_test, preds)
        h_loss = hamming_loss(Y_test, preds)
        micro_f1 = f1_score(Y_test, preds, average='micro', zero_division=0)
        macro_f1 = f1_score(Y_test, preds, average='macro', zero_division=0)

        return {
            'model_name': 'Multi-Output Random Forest',
            'split_name': split_name,
            'num_samples': len(Y_test),
            'subset_accuracy': subset_acc,
            'hamming_loss': h_loss,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'inference_time_ms': inference_time_ms,
            'fps': 1000.0 / max(1e-4, inference_time_ms),
            'predictions': preds
        }
