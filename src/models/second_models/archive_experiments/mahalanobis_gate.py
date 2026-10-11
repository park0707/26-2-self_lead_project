# -*- coding: utf-8 -*-
"""
NovelFlow 정상 프로토타입 다양체 마할라노비스 거리 듀얼 게이트 (Solution 4)
- MBEC 2024 / PR 2026 UI-PRMD 재활 운동 이상 탐지 SOTA 기반
- 정상(Normal) 클래스가 형성하는 생체역학적 콤팩트 다양체(Compact Manifold)를 통계적 타원체로 모델링
- 정규화된 마할라노비스 거리(Mahalanobis Distance)를 통해 경계선 외곽 오탐(False Positive)을 원천 차단
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from typing import Optional

class NormalPrototypeMahalanobisGate:
    """
    정상 자세 프로토타입 마할라노비스 거리 기반 이상 판별 게이트
    """
    def __init__(self, n_components: int = 8, quantile: float = 0.95):
        self.n_components = n_components
        self.quantile = quantile
        self.pca: Optional[PCA] = None
        self.mean_vector: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.threshold: float = 1e9

    def fit(self, X_norm_tr: np.ndarray):
        """
        오직 Train 폴드의 '정상(y=1)' 샘플들만으로 정상 다양체 학습
        """
        N, D = X_norm_tr.shape
        n_comp = min(self.n_components, N - 1, D)
        
        # 1. PCA 차원 축소로 안정적인 직교 기저 형성
        self.pca = PCA(n_components=n_comp, random_state=42)
        Z_norm = self.pca.fit_transform(X_norm_tr)

        # 2. Ledoit-Wolf 수축 추정으로 안정적인 역공분산 행렬 산출
        lw = LedoitWolf()
        lw.fit(Z_norm)
        self.mean_vector = lw.location_
        cov = lw.covariance_
        # 대각선 지터 추가로 수치적 안정성 극대화
        cov_reg = cov + np.eye(n_comp) * 1e-4
        self.inv_cov = np.linalg.pinv(cov_reg)

        # 3. Train 정상 샘플들의 마할라노비스 거리 분포 계산 및 임계값(분위수) 설정
        diff = Z_norm - self.mean_vector
        # d^2 = sum( (diff @ inv_cov) * diff, axis=-1 )
        d2 = np.sum((diff @ self.inv_cov) * diff, axis=-1)
        d_vals = np.sqrt(np.maximum(d2, 0.0))
        self.threshold = float(np.percentile(d_vals, self.quantile * 100.0))
        return self

    def compute_distance(self, X: np.ndarray) -> np.ndarray:
        """
        입력 샘플들과 정상 프로토타입 간의 마할라노비스 거리 계산 -> Shape: (N,)
        """
        Z = self.pca.transform(X)
        diff = Z - self.mean_vector
        d2 = np.sum((diff @ self.inv_cov) * diff, axis=-1)
        return np.sqrt(np.maximum(d2, 0.0))

    def is_in_manifold(self, X: np.ndarray, custom_thresh: Optional[float] = None) -> np.ndarray:
        """
        정상 다양체 내부 포함 여부 반환 (True: 정상 다양체 내부, False: 이상치/외곽 기각)
        """
        th = custom_thresh if custom_thresh is not None else self.threshold
        dist = self.compute_distance(X)
        return dist <= th
