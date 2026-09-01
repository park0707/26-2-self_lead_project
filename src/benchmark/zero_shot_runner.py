import numpy as np
from typing import Dict, Any, List
from src.core.types import KeypointSequence
from src.evaluation.metrics import EvaluationMetrics

class ZeroShotBenchmarkRunner:
    """Frozen Threshold 기반 UI-PRMD 및 REHAB24-6 제로샷 외부 일반화 평가기"""
    
    def __init__(self, trained_model: Any, frozen_threshold: float = 0.5):
        self.model = trained_model
        self.frozen_threshold = frozen_threshold

    def evaluate_external_sequences(self, sequences: List[KeypointSequence], dataset_name: str) -> Dict[str, Any]:
        """외부 벤치마크 시퀀스 집합에 대해 추가 튜닝 없이 제로샷 평가 수행"""
        from src.features.temporal_aggregator import FeatureAggregator
        
        X_list = []
        y_list = []
        for seq in sequences:
            fmt_a, _, _ = FeatureAggregator.build_features(seq)
            X_list.append(fmt_a)
            y_list.append(seq.label)
            
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=int)
        
        # 모델 예측 (Frozen Threshold 적용)
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[:, 1]
            preds = (probs >= self.frozen_threshold).astype(int)
        else:
            preds = self.model.predict(X)
            probs = preds.astype(float)
            
        metrics = EvaluationMetrics.compute_all_metrics(y, preds, probs)
        metrics['dataset_name'] = dataset_name
        metrics['frozen_threshold'] = self.frozen_threshold
        metrics['total_samples'] = len(sequences)
        return metrics
