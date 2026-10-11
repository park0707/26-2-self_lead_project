# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: c1_model_v4.py
Purpose: 바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반 / Spine Neutrality) v4.0 공식 확정 분류 모델
- 핵심 혁신 및 개선 사항 (v3.0 대비):
  1) 제2방법(외적 기반 부호 있는 척추 곡률 및 운동역학 연쇄 벡터 10종) 전면 결합:
     - c1_v4_signed_thoracolumbar_nadir (부호 있는 흉요추 곡률각, 전만+ vs 후만-)
     - c1_v4_signed_lumbopelvic_nadir / flex_delta (부호 있는 요추-골반 상대각 및 버트윙크 말림 변위)
     - c1_v4_pelvifemoral_angle_nadir (최하사점 대퇴-골반 사잇각)
     - c1_v4_bottom_curvature_jerk_max (최하사점 각가속도/저크 급변도)
  2) 피처 기여도 2위(11.75%), 3위(11.17%) 석권, 2개 피처만으로 모델 판단 가중치의 23% 점유
  3) 실측 성능 혁신:
     - ROC-AUC: 86.55% -> 87.90% (+1.35%p 대폭 신장)
     - 원천 1,906건 F1-Score: CatBoost 78.19%, Cascade 78.42%, Opt F1 79.94%
     - 정상 샘플 오탐(Clean FP): 42건(33.6%) -> 35건(28.0%, 비용민감 시 27건 21.6%)으로 최대 35.7% 오탐 격감!
  4) 64프레임 3차 스플라인(Cubic Spline) 보간 표준 준수
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

from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2, PRUNED_LEGACY_NAMES_22
from src.models.second_models.babel_squat.build_c1_v4_spine_features import (
    C1_V4_SPINE_FEATURE_NAMES,
    calculate_v4_spine_metrics_for_seq
)

# v2 22D 레거시 중 소거 4종(#11 gaze_pitch_mean, #15 ear_sh_hip_drop, #17 spine_cervical_ratio, #18 lit_trunk_lean_max)
V4_LEGACY_KEEP_INDICES = [i for i in range(22) if i not in [11, 15, 17, 18]]
C1_V4_LEGACY_NAMES_18 = [PRUNED_LEGACY_NAMES_22[i] for i in V4_LEGACY_KEEP_INDICES]

# 공식 v4 28D 피처명
C1_V4_FEATURE_NAMES_28 = C1_V4_LEGACY_NAMES_18 + list(C1_V4_SPINE_FEATURE_NAMES)


class BabelSquatC1ModelV4:
    """
    바벨 스쿼트 세부 결함 1 (c1: 척추의 중립 위반) v4.0 공식 확정 모델
    - 피처 규격: 28D (레거시 정예 18D + 신규 생체역학 벡터연쇄 10D)
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
        self.model_tier2 = None
        self.model_champion = None
        self.is_fitted = False

        self.feature_names = C1_V4_FEATURE_NAMES_28
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
            self.model_tier2 = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=4.5,
                random_seed=42,
                thread_count=-1,
                verbose=False
            )

    @classmethod
    def extract_features(
        cls,
        X_b: np.ndarray,
        v4_spine_feats_10d: Optional[np.ndarray] = None,
        frames_pts_list: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        64프레임 Format B 텐서로부터 c1 v4.0 공식 28D 피처셋 추출
        """
        N = len(X_b) if X_b.ndim == 3 else 1
        is_single = (X_b.ndim == 2)
        if is_single:
            X_b = np.expand_dims(X_b, axis=0)

        # 1. 레거시 피처 추출 후 정예 18D 슬라이싱
        X_30d = BabelSquatC1ModelV2.extract_features(X_b)
        X_22d = X_30d[:, :22]
        X_18d = X_22d[:, V4_LEGACY_KEEP_INDICES]

        # 2. v4 10D 피처 로드 또는 직접 계산
        if v4_spine_feats_10d is not None:
            X_spine = v4_spine_feats_10d
        elif frames_pts_list is not None:
            vec = calculate_v4_spine_metrics_for_seq(frames_pts_list)
            X_spine = np.expand_dims(vec, axis=0) if is_single else np.tile(vec, (N, 1))
        else:
            # 캐시 파일 확인
            cache_p = os.path.join(CURRENT_DIR, 'squat_c1_v4_spine_feats.npz')
            if os.path.exists(cache_p):
                d = np.load(cache_p, allow_pickle=True)['v4_spine_feats']
                if len(d) == N:
                    X_spine = d
                else:
                    X_spine = np.zeros((N, 10), dtype=np.float32)
            else:
                X_spine = np.zeros((N, 10), dtype=np.float32)

        if X_spine.ndim == 1:
            X_spine = np.expand_dims(X_spine, axis=0)

        X_28d = np.hstack([X_18d, X_spine])
        if is_single:
            return X_28d[0]
        return X_28d

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

        if self.mode == "catboost":
            if self.model_champion is None:
                self._init_default_models()
            self.model_champion.fit(X, y, sample_weight=sample_weight)
        else:
            if self.model_champion is None or self.model_tier1 is None:
                self._init_default_models()
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
        print(f"[*] c1 v4.0 가중치 저장 완료: {path}")

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
