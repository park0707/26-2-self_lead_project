# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c4. 발바닥 지면 고정 (Foot Ground Fixation / Heel-Toe Elevation)
Model Version: v1.0 (c4_model_v1.py)
Status: 레거시 15관절 체계 프로토콜 공식 v1.0 모델 (16D)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 64프레임 표준 3차원 골격 시계열 텐서 Format B (N, 64, 92) 또는 단일 시퀀스 (64, 92)
2. 피처 엔지니어링 (16D 레거시 발목 생체역학 피처셋):
   - 발목 굴곡각 변위, 발목 수직 높이 변동, 스탠스 드리프트, 각속도 피크 등
3. 실측 성능 (Level 1~3 실무군 1,824건, 5-Fold StratifiedGroupKFold 완전 격리):
   - 단일 LightGBM: F1 = 59.84%, ROC-AUC = 67.72%
   - 단일 CatBoost: F1 = 59.07%, ROC-AUC = 67.55%
   - 2계층 캐스케이드: F1 = 63.41% (최적 임계값 F1 = 70.33%), ROC-AUC = 74.11%
   - 원천 1,906건 전체 기준: LightGBM F1 = 56.49%, CatBoost F1 = 57.39%
"""

import os
import sys
import pickle
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.second_models.babel_squat.run_c4_extended_pipeline import FEATURE_NAMES_16_C4, extract_c4_16d_features

class BabelSquatC4ModelV1:
    """
    바벨 스쿼트 'c4. 발바닥 지면 고정' 오류 분류 프로토콜 공식 v1.0 모델 (16D)
    """

    DEFECT_CODE = "c4"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 16
    FEATURE_NAMES = FEATURE_NAMES_16_C4

    def __init__(
        self,
        mode: str = "cascade",
        tier1_threshold_low: float = 0.25,
        tier1_threshold_high: float = 0.75,
        decision_threshold: float = 0.50
    ):
        self.mode = mode
        self.tier1_threshold_low = tier1_threshold_low
        self.tier1_threshold_high = tier1_threshold_high
        self.decision_threshold = decision_threshold
        self.feature_dim = 16
        self.feature_names = FEATURE_NAMES_16_C4

        self.model_tier1 = None
        self.model_tier2 = None
        self.model_champion = None
        self.is_fitted = False

        self._init_default_models()

    def _init_default_models(self):
        """기본 하이퍼파라미터 모델 초기화"""
        if lgb is not None:
            self.model_tier1 = lgb.LGBMClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.04, num_leaves=15,
                colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=-1
            )
            self.model_champion = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31,
                colsample_bytree=0.85, subsample=0.85, reg_lambda=2.0,
                random_state=42, verbose=-1, n_jobs=-1
            )
        if CatBoostClassifier is not None:
            self.model_tier2 = CatBoostClassifier(
                iterations=250, depth=6, learning_rate=0.04, l2_leaf_reg=4.0,
                random_seed=42, thread_count=-1, verbose=False,
                allow_writing_files=False
            )

    @classmethod
    def extract_features(cls, X_b: np.ndarray) -> np.ndarray:
        """Format B 시계열 텐서로부터 16D c4 피처 추출"""
        if X_b.ndim == 2:
            X_b = np.expand_dims(X_b, axis=0)
        return extract_c4_16d_features(X_b)

    def fit(self, X: np.ndarray, y: np.ndarray, levels: Optional[np.ndarray] = None):
        """모델 학습"""
        if levels is None:
            levels = np.full(len(y), 2)

        l1_mask = (levels == 1)
        if np.sum(l1_mask) > 10 and self.model_tier1 is not None:
            self.model_tier1.fit(X[l1_mask], y[l1_mask])

        l123_mask = (levels <= 3)
        X_train = X[l123_mask] if np.sum(l123_mask) > 0 else X
        y_train = y[l123_mask] if np.sum(l123_mask) > 0 else y

        if self.model_tier2 is not None:
            self.model_tier2.fit(X_train, y_train)

        if self.model_champion is not None:
            self.model_champion.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """2계층 캐스케이드 사후확률 추론"""
        if not self.is_fitted:
            raise RuntimeError("모델이 아직 학습되지 않았습니다.")

        p_t2 = self.model_tier2.predict_proba(X)[:, 1] if self.model_tier2 is not None else np.full(len(X), 0.5)

        if self.mode == "champion" or self.model_tier1 is None:
            probs = p_t2
        else:
            p_t1 = self.model_tier1.predict_proba(X)[:, 1]
            probs = p_t2.copy()
            for i in range(len(X)):
                if p_t1[i] <= self.tier1_threshold_low:
                    probs[i] = p_t1[i]
                elif p_t1[i] >= self.tier1_threshold_high:
                    probs[i] = p_t1[i]
                else:
                    probs[i] = p_t2[i]

        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.decision_threshold).astype(int)

    def save_weights(self, filepath: str):
        """가중치 직렬화 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'model_tier1': self.model_tier1,
            'model_tier2': self.model_tier2,
            'model_champion': self.model_champion,
            'mode': self.mode,
            'decision_threshold': self.decision_threshold,
            'tier1_threshold_low': self.tier1_threshold_low,
            'tier1_threshold_high': self.tier1_threshold_high,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[*] c4 v1.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model_tier1 = data.get('model_tier1')
        self.model_tier2 = data.get('model_tier2')
        self.model_champion = data.get('model_champion')
        self.mode = data.get('mode', 'cascade')
        self.decision_threshold = data.get('decision_threshold', 0.50)
        self.tier1_threshold_low = data.get('tier1_threshold_low', 0.25)
        self.tier1_threshold_high = data.get('tier1_threshold_high', 0.75)
        self.is_fitted = True
        return self

BabelSquatC4Model = BabelSquatC4ModelV1
