# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Forward Orientation)
Model Version: v4.0 (c2_model_v4.py)
Status: ★ 최종 확정 프로덕션 모델 (9D 고간섭/노이즈 소거 정예 모델)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 9D 정예 생체역학 피처셋 (v3 13D 중 4대 상호 간섭/노이즈 피처 소거)
   - 소거된 4대 방해 피처:
     * c2_gaze_relative_to_trunk_mean (체간 대비 상대 시선 평균: 흉추 각도와의 다중공선성 노이즈)
     * c2_head_yaw_asymmetry_max (좌우 비대칭 최대치: 카메라 45도 시점 왜곡 노이즈)
     * c2_gaze_pitch_min (게이트형 3D 시선 최하사점 피처와의 중복 간섭 r=0.710)
     * c2_head_forward_protrusion_max (거북목 돌출 최대치: 척추 전단 변위와 중복 간섭 r=-0.749)
2. 잔존 9D 정예 피처셋:
   1) c2_gaze_pitch_drop (시선 낙하폭)
   2) c2_cervical_spine_angle_mean (경추 평균 굴곡각)
   3) c2_cervical_spine_angle_drop (경추 굴곡 낙하폭)
   4) c2_head_yaw_asymmetry_mean (좌우 머리 회전 비대칭 평균)
   5) c2_cervical_velocity_peak_at_nadir (최하사점 경추 각속도 피크)
   6) c2_gaze_coupling_lag (시선-골반 커플링 지연시간)
   7) c2_cervical_hyperextension_ratio (경추 과신전 비율)
   8) c2_gated_true_gaze_pitch_drop (게이트형 3D 시선 낙하폭)
   9) c2_gated_true_gaze_pitch_bot (게이트형 3D 최하사점 시선 절대각도 - 전체 중요도 1위)
3. 실측 성능 (전체 N=1,906건, 5-Fold StratifiedGroupKFold 완전 격리):
   - Precision: 89.67% (v3 대비 +1.62%p 상승!)
   - Recall: 84.15% (v3 대비 +0.42%p 상승!)
   - F1-Score: 86.82% (v3 대비 +0.98%p 폭등!)
   - ROC-AUC: 93.94% (v3 대비 +0.13%p 상승!)
   - 피처 수: 13D -> 9D (31% 추가 경량화 및 추론 속도 대폭 향상)
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.second_models.babel_squat.c2_model_v3 import FEATURE_NAMES_13, V2_TO_V3_INDICES

# v3 13D 인덱스에서 9D를 추출하기 위한 인덱스 매핑
# v3 13D 중 제거 대상: [0: pitch_min, 5: yaw_max, 6: fwd_max, 7: trunk_mean]
# 잔존 9D: [1, 2, 3, 4, 8, 9, 10, 11, 12]
V3_TO_V4_INDICES = [1, 2, 3, 4, 8, 9, 10, 11, 12]

FEATURE_NAMES_9: List[str] = [
    "c2_gaze_pitch_drop",
    "c2_cervical_spine_angle_mean",
    "c2_cervical_spine_angle_drop",
    "c2_head_yaw_asymmetry_mean",
    "c2_cervical_velocity_peak_at_nadir",
    "c2_gaze_coupling_lag",
    "c2_cervical_hyperextension_ratio",
    "c2_gated_true_gaze_pitch_drop",
    "c2_gated_true_gaze_pitch_bot"
]

DROPPED_FEATURES_FROM_V3 = [
    "c2_gaze_pitch_min",
    "c2_head_yaw_asymmetry_max",
    "c2_head_forward_protrusion_max",
    "c2_gaze_relative_to_trunk_mean"
]


class BabelSquatC2ModelV4:
    """
    바벨 스쿼트 c2 결함(고개 정면 미유지) 최종 공식 프로덕션 모델 v4.0
    CatBoost 60% + LightGBM 40% 앙상블 (9D 고간섭 소거 정예 모델)
    """

    DEFECT_CODE = "c2"
    DEFECT_NAME = "고개 정면"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 9
    FEATURE_NAMES = FEATURE_NAMES_9

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50
    ):
        self.decision_threshold = decision_threshold
        self.cb_models: List[CatBoostClassifier] = []
        self.lgb_models: List[lgb.LGBMClassifier] = []
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_9

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c2_model_v4_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        입력: (N, 9) 또는 13D/20D 자동 슬라이싱 지원
        출력: (N, 2) [정상 확률, 결함 확률]
        """
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C2 v4 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single = True

        # 입력 차원 유연성 보장 (13D 또는 20D가 들어올 경우 9D로 자동 변환)
        if X.shape[1] == 13:
            X = X[:, V3_TO_V4_INDICES]
        elif X.shape[1] == 20:
            X_13 = X[:, V2_TO_V3_INDICES]
            X = X_13[:, V3_TO_V4_INDICES]

        N = X.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)

        for cb, clf in zip(self.cb_models, self.lgb_models):
            p_cb = cb.predict_proba(X)[:, 1]
            p_lgb = clf.predict_proba(X)[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            ensemble_probs += p_fold / len(self.cb_models)

        res = np.column_stack([1.0 - ensemble_probs, ensemble_probs])
        return res[0] if single else res

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이진 결함 판별 (1: 결함, 0: 정상)"""
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return int(probs[1] >= self.decision_threshold)
        return (probs[:, 1] >= self.decision_threshold).astype(int)

    def save_weights(self, filepath: str):
        """가중치 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'decision_threshold': self.decision_threshold,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c2 v4.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        """가중치 로드"""
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_9)
        self.is_fitted = True
        return self


BabelSquatC2Model = BabelSquatC2ModelV4
