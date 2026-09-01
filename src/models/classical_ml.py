import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

class PostureBaselineModels:
    """1차 베이스라인 머신러닝 모델군 빌더 및 학습기"""
    
    @staticmethod
    def get_models(class_weight_ratio: float = 21.5) -> Dict[str, Any]:
        """불균형 클래스 가중치가 적용된 모델 딕셔너리 반환"""
        cw = {0: 1.0, 1: class_weight_ratio}
        
        models = {
            'LogisticRegression': LogisticRegression(class_weight=cw, max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, class_weight=cw, max_depth=10, random_state=42),
            'HistGradientBoosting': HistGradientBoostingClassifier(class_weight=cw, max_iter=100, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        }
        return models
