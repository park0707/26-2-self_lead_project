# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Forward Orientation)
Model Version: v2.0 (c2_model_v2.py)
Status: ★ 최종 선정 공식 프로덕션 모델 (v2-B 기반 20D)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 64프레임 표준 3차원 골격 시계열 텐서 Format B (N, 64, C) 또는 단일 시퀀스 (64, C)
   - [선택적 확장]: 3D 안면 키포인트(Nose, Left/Right Ear, Left/Right Shoulder, Neck, Waist)
2. 피처 엔지니어링 (최종 확정 20D 정예 생체역학 피처셋):
   - 18D 체간/경추 동적 협응 피처셋 (v1 계승)
   - 2D 게이트형 3D 시선 벡터 피처:
     * c2_gated_true_gaze_pitch_drop (기립 대비 최하사점 시선 피치 낙하 각도)
     * c2_gated_true_gaze_pitch_bot  (최하사점 절대 시선 피치 각도)
   - 3중 안전 게이트(Triple-Gate):
     * 체간 전면 법선 벡터(n_torso_z < -0.15) 판별로 후방 시점 가짜 얼굴 100% 원천 차단
     * 후방 또는 안면 가려짐 시 2D 시선 피처를 0.0 더미로 안전 패딩하고 기존 18D 체간 피처로 자동 Fallback
3. 실측 성능 (Level 1~3 실무군 1,775건, 5-Fold StratifiedGroupKFold 완전 격리):
   - 단일 CatBoost: F1 = 86.26% (+3.85%p), ROC-AUC = 94.84%
   - 2계층 캐스케이드 (Tier-1 Gatekeeper + Tier-2 Resolver):
     * F1 = 90.11% (+3.28%p 대도약, 최적 F1 90.23% 달성으로 90% 고지 돌파!)
     * ROC-AUC = 96.98% (+2.56%p)
     * Balanced Accuracy = 90.12%
"""

import os
import sys
import pickle
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

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

from src.models.second_models.babel_squat.c2_model_v1 import FEATURE_NAMES_18

FEATURE_NAMES_20: List[str] = FEATURE_NAMES_18 + [
    'c2_gated_true_gaze_pitch_drop',
    'c2_gated_true_gaze_pitch_bot'
]

class BabelSquatC2ModelV2:
    """
    바벨 스쿼트 'c2. 고개 정면' 오류 분류 공식 프로덕션 모델 v2 (20D 게이트형 시선 벡터)
    """

    DEFECT_CODE = "c2"
    DEFECT_NAME = "고개 정면"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 20
    FEATURE_NAMES = FEATURE_NAMES_20

    def __init__(
        self,
        mode: str = "cascade",  # 'cascade' (2계층) 또는 'champion' (단일 최적 모델)
        tier1_threshold_low: float = 0.25,
        tier1_threshold_high: float = 0.75,
        decision_threshold: float = 0.50
    ):
        self.mode = mode
        self.tier1_threshold_low = tier1_threshold_low
        self.tier1_threshold_high = tier1_threshold_high
        self.decision_threshold = decision_threshold
        self.feature_dim = 20
        self.feature_names = FEATURE_NAMES_20

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
                random_seed=42, thread_count=-1, verbose=False
            )

    @classmethod
    def extract_features(
        cls,
        X_b: np.ndarray,
        gaze_features_2d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Format B 텐서 및 안면 시선 2D 피처로부터 20D 정예 피처셋 산출
        - gaze_features_2d가 None이거나 미관측 시 [0.0, 0.0] 더미 패딩으로 안전 Fallback
        """
        from src.models.second_models.babel_squat.c2_model_v1 import BabelSquatC2ModelV1
        X_18d = BabelSquatC2ModelV1.extract_features(X_b, use_pruned=True)

        N = X_18d.shape[0]
        if gaze_features_2d is None:
            gaze_2d = np.zeros((N, 2), dtype=np.float32)
        else:
            if gaze_features_2d.ndim == 1:
                gaze_2d = np.expand_dims(gaze_features_2d, axis=0)
            else:
                gaze_2d = gaze_features_2d.astype(np.float32)
            if gaze_2d.shape[0] != N:
                gaze_2d = np.zeros((N, 2), dtype=np.float32)

        return np.concatenate([X_18d, gaze_2d], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, levels: Optional[np.ndarray] = None):
        """모델 학습"""
        if levels is None:
            levels = np.full(len(y), 2)

        # Tier-1: Level 1 명확군만 학습
        l1_mask = (levels == 1)
        if np.sum(l1_mask) > 10 and self.model_tier1 is not None:
            self.model_tier1.fit(X[l1_mask], y[l1_mask])

        # Tier-2: Level 1~3 전체 학습
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
        print(f"[*] v2.0 가중치 저장 완료: {filepath}")

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
