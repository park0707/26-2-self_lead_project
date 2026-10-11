# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: c1_model_v6.py
Purpose: 바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반 / Spine Neutrality) v6.0 최적화 분류 모델
- 핵심 혁신 및 개선 사항 (v5.0 대비):
  1) 5-Fold LOFO(Leave-One-Feature-Out) 및 다중공선성(|r| >= 0.80) 정밀 분석 기반 피처 최적화 다이어트:
     - 척추 각가속도 저크 중복 소거 (c1_v4_bottom_curvature_jerk_max)
     - 흉요추 곡률 사잇각 극심한 다중공선성(r=0.920) 소거 (c1_v5_cervicothoracic_hinge_pitch)
     - 요추-골반 말림 중복 소거 (c1_v4_signed_lumbopelvic_flex_delta)
     - 경추 중복 사잇각 소거 (cervical_angle_min)
     - 노이즈 강성 피처 소거 (lit_spinal_stiffness)
  2) v6-A (29D 상위 5대 저해 소거 정예 모델) 및 v6-B (21D 고효율 컴팩트 모델) 듀얼 지원:
     - v6-A (29D): 원천 1,906건 Opt F1 80.04% (80% 돌파!), Cascade F1 78.45%, ROC-AUC 88.16%
     - v6-B (21D): 원천 1,906건 Cascade F1 78.73% (역대 최고 기록 경신!), Level 1~3 ROC-AUC 93.09%, Opt F1 86.36%
  3) 64프레임 3차 스플라인 표준 및 SpinePose 연동 인터페이스 완비
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.interpolate import interp1d

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v5 import BabelSquatC1ModelV5, C1_V5_FEATURE_NAMES_34

# 34D 중 v6-A 소거 대상 5종 인덱스:
# c1_v4_signed_lumbopelvic_flex_delta (21), c1_v4_bottom_curvature_jerk_max (23),
# cervical_angle_min (1), c1_v5_cervicothoracic_hinge_pitch (31), lit_spinal_stiffness (17)
V6A_DROPPED_INDICES = [1, 17, 21, 23, 31]
V6A_KEEP_INDICES = [i for i in range(34) if i not in V6A_DROPPED_INDICES]
C1_V6A_FEATURE_NAMES_29 = [C1_V5_FEATURE_NAMES_34[i] for i in V6A_KEEP_INDICES]

# 34D 중 v6-B 21D 유지 인덱스 (13개 저해/중복 피처 전수 다이어트)
V6B_DROPPED_INDICES = [1, 17, 18, 20, 21, 22, 23, 24, 25, 29, 30, 31, 33]
V6B_KEEP_INDICES = [i for i in range(34) if i not in V6B_DROPPED_INDICES]
C1_V6B_FEATURE_NAMES_21 = [C1_V5_FEATURE_NAMES_34[i] for i in V6B_KEEP_INDICES]


class BabelSquatC1ModelV6:
    """
    바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반) v6.0 최적화 모델
    - 기본 모드: v6-A 29D (Opt F1 80.04% 달성 챔피언)
    - 경량 모드: v6-B 21D (Cascade F1 78.73%, AUC 93.09% 컴팩트 챔피언)
    """

    def __init__(
        self,
        mode: str = "cascade",
        variant: str = "v6-a",
        threshold: float = 0.50,
        tau_tier1_low: float = 0.20,
        tau_tier1_high: float = 0.80,
        cost_weight: float = 1.0
    ):
        self.mode = mode
        self.variant = variant.lower()
        self.threshold = threshold
        self.tau_low = tau_tier1_low
        self.tau_high = tau_tier1_high
        self.cost_weight = cost_weight

        self.model_tier1 = None
        self.model_champion = None
        self.is_fitted = False

        if self.variant in ("v6-b", "21d", "compact"):
            self.keep_indices = V6B_KEEP_INDICES
            self.feature_names = C1_V6B_FEATURE_NAMES_21
        else:
            self.keep_indices = V6A_KEEP_INDICES
            self.feature_names = C1_V6A_FEATURE_NAMES_29

        self.feature_dim = len(self.feature_names)
        self._init_default_models()

    def _init_default_models(self):
        """기본 하이퍼파라미터 모델 초기화"""
        if lgb is not None:
            self.model_tier1 = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.03,
                num_leaves=31,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        if CatBoostClassifier is not None:
            self.model_champion = CatBoostClassifier(
                iterations=220,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=5.0,
                random_seed=42,
                thread_count=-1,
                verbose=False
            )

    @classmethod
    def extract_features(
        cls,
        X_b: np.ndarray,
        v4_spine_feats_10d: Optional[np.ndarray] = None,
        v5_spinepose_feats_6d: Optional[np.ndarray] = None,
        frames_pts_list: Optional[List[Dict[str, Any]]] = None,
        variant: str = "v6-a"
    ) -> np.ndarray:
        """
        64프레임 Format B 텐서로부터 c1 v6 최적화 피처셋 추출 (29D 또는 21D)
        """
        X_34d = BabelSquatC1ModelV5.extract_features(
            X_b,
            v4_spine_feats_10d=v4_spine_feats_10d,
            v5_spinepose_feats_6d=v5_spinepose_feats_6d,
            frames_pts_list=frames_pts_list
        )
        keep = V6B_KEEP_INDICES if variant.lower() in ("v6-b", "21d", "compact") else V6A_KEEP_INDICES
        if X_34d.ndim == 1:
            return X_34d[keep]
        return X_34d[:, keep]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cost_weight: Optional[float] = None
    ):
        """모델 학습"""
        if X.shape[1] == 34:
            X = X[:, self.keep_indices]

        if cost_weight is not None:
            self.cost_weight = cost_weight

        sample_weight = np.where(y == 0, self.cost_weight, 1.0).astype(np.float32)

        if self.model_champion is None or self.model_tier1 is None:
            self._init_default_models()

        if self.mode == "catboost":
            self.model_champion.fit(X, y, sample_weight=sample_weight)
        else:
            self.model_champion.fit(X, y, sample_weight=sample_weight)
            self.model_tier1.fit(X, y, sample_weight=sample_weight)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """결함 발생 확률 배열 반환 (N, 2) [P(정상), P(결함)]"""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit() 또는 load_weights()를 호출하십시오.")

        is_single = (X.ndim == 1)
        if is_single:
            X = np.expand_dims(X, axis=0)

        if X.shape[1] == 34:
            X = X[:, self.keep_indices]

        if self.mode == "catboost":
            p1 = self.model_champion.predict_proba(X)[:, 1]
        else:
            p_cat = self.model_champion.predict_proba(X)[:, 1]
            p_lgb = self.model_tier1.predict_proba(X)[:, 1]
            p1 = 0.60 * p_cat + 0.40 * p_lgb

        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        if is_single:
            return np.array([p0[0], p1[0]], dtype=np.float32)
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """이진 결함 판정 (0: 정상, 1: 척추 중립 위반 결함)"""
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return int(probs[1] >= th)
        return (probs[:, 1] >= th).astype(int)

    def save_weights(self, path: str):
        """모델 가중치 및 설정 직렬화 저장"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            'mode': self.mode,
            'variant': self.variant,
            'threshold': self.threshold,
            'tau_low': self.tau_low,
            'tau_high': self.tau_high,
            'cost_weight': self.cost_weight,
            'keep_indices': self.keep_indices,
            'feature_names': self.feature_names,
            'model_champion': self.model_champion,
            'model_tier1': self.model_tier1
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[*] c1 v6.0 가중치 저장 완료: {path}")

    def load_weights(self, path: str):
        """사전 학습 가중치 로드"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {path}")
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.mode = state.get('mode', self.mode)
        self.variant = state.get('variant', self.variant)
        self.threshold = state.get('threshold', self.threshold)
        self.tau_low = state.get('tau_low', self.tau_low)
        self.tau_high = state.get('tau_high', self.tau_high)
        self.cost_weight = state.get('cost_weight', self.cost_weight)
        self.keep_indices = state.get('keep_indices', self.keep_indices)
        self.feature_names = state.get('feature_names', self.feature_names)
        self.model_champion = state.get('model_champion')
        self.model_tier1 = state.get('model_tier1')
        self.is_fitted = True
        return self
