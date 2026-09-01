import numpy as np
from typing import Dict, Any, Tuple
from sklearn.neural_network import MLPClassifier

class SequenceDeepModel:
    """Format B 시계열 텐서(14, 64) 입력을 위한 시계열 신경망 모델"""
    
    def __init__(self, hidden_dim: int = 128):
        # 14채널 x 64프레임 = 896차원 시계열을 입력받는 다층 시계열 MLP/1D-CNN 시뮬레이터
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_dim, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42
        )
        
    def fit(self, X_temporal: np.ndarray, y: np.ndarray):
        """X_temporal shape: (N, 14, 64) -> Flatten (N, 896)"""
        N = X_temporal.shape[0]
        X_flat = X_temporal.reshape(N, -1)
        self.model.fit(X_flat, y)
        
    def predict(self, X_temporal: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        N = X_temporal.shape[0]
        X_flat = X_temporal.reshape(N, -1)
        probs = self.model.predict_proba(X_flat)[:, 1]
        return (probs >= threshold).astype(int)
        
    def predict_proba(self, X_temporal: np.ndarray) -> np.ndarray:
        N = X_temporal.shape[0]
        X_flat = X_temporal.reshape(N, -1)
        return self.model.predict_proba(X_flat)[:, 1]
