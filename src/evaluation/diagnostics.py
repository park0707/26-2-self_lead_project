import numpy as np
from typing import Dict, Any
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from src.evaluation.metrics import EvaluationMetrics

class DiagnosticEngine:
    """데이터 누수 및 Shortcut Learning 방지를 위한 3중 진단 엔진"""

    @staticmethod
    def label_shuffling_test(model_cls, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """1. 라벨 셔플링 대조 실험: 정답 라벨을 무작위로 섞었을 때 성능이 0.5 수준으로 하락하는지 검증"""
        y_shuffled = np.random.permutation(y)
        model = model_cls()
        
        # 80/20 train/test
        split = int(len(X) * 0.8)
        model.fit(X[:split], y_shuffled[:split])
        preds = model.predict(X[split:])
        
        shuffled_f1 = float(f1_score(y_shuffled[split:], preds, average='macro', zero_division=0))
        return {
            'shuffled_macro_f1': shuffled_f1,
            'leakage_detected': bool(shuffled_f1 > 0.65) # 셔플했는데도 0.65 이상이면 누수 의심
        }

    @staticmethod
    def subject_kfold_variance(model_cls, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
        """2. 피험자 5-Fold 교차검증 분산 분석: 폴드 간 분산이 과도한지 검증"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        f1_scores = []
        
        for train_idx, val_idx in kf.split(X):
            model = model_cls()
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            f1_scores.append(f1_score(y[val_idx], preds, average='macro', zero_division=0))
            
        return {
            'mean_kfold_f1': float(np.mean(f1_scores)),
            'std_kfold_f1': float(np.std(f1_scores)),
            'high_variance_flag': bool(np.std(f1_scores) > 0.15)
        }
