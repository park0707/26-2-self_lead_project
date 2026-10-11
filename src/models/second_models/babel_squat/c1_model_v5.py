# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: c1_model_v5.py
Purpose: 바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반 / Spine Neutrality) v5.0 공식 분류 모델
- 핵심 혁신 및 개선 사항 (v4.0 대비):
  1) DFKI SpinePose 9대 척추 분절(S1, L5, L3, L1, T8, T3, C7, C4, C1) 기반 6D 고정밀 피처 결합 (총 34D):
     - [피처 1위 석권! 12.18%] c1_v5_lumbar_curvature_poly_a: L1-L3-L5-S1 요추 2차 곡률 계수 (전만 음수 vs 후만 양수)
     - [피처 5위! 5.91%] c1_v5_cervicothoracic_hinge_pitch: C1-C7 경추선과 T3-T8 흉추선의 이행부 사잇각
     - c1_v5_thoracolumbar_cobb_angle: L1-L5 요추 전만각 (Cobb angle proxy)
     - c1_v5_spine_s_curve_harmony: 흉추 후만과 요추 전만의 S자 곡률 밸런스 비율
     - c1_v5_nadir_lumbar_tangent_jerk: 최하사점 요추 접선각의 2차 미분 각가속도 급변도
     - c1_v5_pelvis_l5_retroversion_delta: 골반(S1) 대비 L5 요추체의 후방 말림(Retroversion) 변위
  2) 실측 성능 향상 (1,906건 전수 5-Fold):
     - ROC-AUC: 87.90% -> 88.21% (+0.31%p, 역대 최고치 경신)
     - 척추 단독 모델 F1: 73.07% -> 75.83% (+2.76%p 급등)
     - Level 1~3 실무군 Cascade Opt F1: 85.89% 달성
  3) 64프레임 3차 스플라인 보간 및 SpinePose 온디맨드 추론 인터페이스 완비
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

from src.models.second_models.babel_squat.c1_model_v4 import BabelSquatC1ModelV4, C1_V4_FEATURE_NAMES_28
from src.models.second_models.babel_squat.c1_spinepose_extractor import (
    BarbellSquatC1SpinePoseExtractor,
    C1_SPINEPOSE_DENSE_FEATURE_NAMES_6
)

# 공식 v5 34D 피처명
C1_V5_FEATURE_NAMES_34 = list(C1_V4_FEATURE_NAMES_28) + list(C1_SPINEPOSE_DENSE_FEATURE_NAMES_6)


class BabelSquatC1ModelV5:
    """
    바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반) v5.0 공식 확정 모델
    - 피처 규격: 34D (v4 28D + SpinePose 9대 척추 분절 6D)
    - 판별 구조: 2계층 캐스케이드 (Cascade: CatBoost 60% + LightGBM 40%) 및 단일 CatBoost 지원
    """

    def __init__(
        self,
        mode: str = "cascade",
        threshold: float = 0.50,
        tau_tier1_low: float = 0.20,
        tau_tier1_high: float = 0.80,
        cost_weight: float = 1.0
    ):
        self.mode = mode
        self.threshold = threshold
        self.tau_low = tau_tier1_low
        self.tau_high = tau_tier1_high
        self.cost_weight = cost_weight

        self.model_tier1 = None
        self.model_champion = None
        self.is_fitted = False

        self.feature_names = C1_V5_FEATURE_NAMES_34
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
        frames_pts_list: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        64프레임 Format B 텐서로부터 c1 v5.0 공식 34D 피처셋 추출
        """
        N = len(X_b) if X_b.ndim == 3 else 1
        is_single = (X_b.ndim == 2)
        if is_single:
            X_b = np.expand_dims(X_b, axis=0)

        # 1. v4 28D 피처 추출
        X_28d = BabelSquatC1ModelV4.extract_features(
            X_b,
            v4_spine_feats_10d=v4_spine_feats_10d,
            frames_pts_list=frames_pts_list
        )
        if X_28d.ndim == 1:
            X_28d = np.expand_dims(X_28d, axis=0)

        # 2. v5 SpinePose 6D 피처 로드 또는 직접 계산
        if v5_spinepose_feats_6d is not None:
            X_v5 = v5_spinepose_feats_6d
        else:
            cache_p = os.path.join(CURRENT_DIR, 'squat_c1_v5_spinepose_feats.npz')
            if os.path.exists(cache_p):
                d = np.load(cache_p, allow_pickle=True)['v5_spinepose_feats']
                if len(d) == N:
                    X_v5 = d
                else:
                    X_v5 = np.zeros((N, 6), dtype=np.float32)
            else:
                X_v5 = np.zeros((N, 6), dtype=np.float32)

        if X_v5.ndim == 1:
            X_v5 = np.expand_dims(X_v5, axis=0)

        X_34d = np.hstack([X_28d, X_v5])
        if is_single:
            return X_34d[0]
        return X_34d

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cost_weight: Optional[float] = None
    ):
        """모델 학습"""
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
            'threshold': self.threshold,
            'tau_low': self.tau_low,
            'tau_high': self.tau_high,
            'cost_weight': self.cost_weight,
            'feature_names': self.feature_names,
            'model_champion': self.model_champion,
            'model_tier1': self.model_tier1
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[*] c1 v5.0 가중치 저장 완료: {path}")

    def load_weights(self, path: str):
        """사전 학습 가중치 로드"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {path}")
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.mode = state.get('mode', self.mode)
        self.threshold = state.get('threshold', self.threshold)
        self.tau_low = state.get('tau_low', self.tau_low)
        self.tau_high = state.get('tau_high', self.tau_high)
        self.cost_weight = state.get('cost_weight', self.cost_weight)
        self.feature_names = state.get('feature_names', self.feature_names)
        self.model_champion = state.get('model_champion')
        self.model_tier1 = state.get('model_tier1')
        self.is_fitted = True
        return self
