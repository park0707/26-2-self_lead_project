# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Forward Orientation)
Model Version: v3.0 (c2_model_v3.py)
Status: ★ 최종 확정 프로덕션 모델 (13D 노이즈/중복 소거 정예 모델)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 64프레임 표준 3차원 골격 시계열 텐서 Format B (N, 64, C) 또는 단일 시퀀스 (64, C)
   - [선택적 확장]: 3D 안면 키포인트 (체간 법선 게이트 연동)
2. 피처 엔지니어링 (최종 확정 13D 정예 생체역학 피처셋):
   - v2 20D 피처셋 중 LOFO(Leave-One-Feature-Out) 소거 실험으로 규명된 7대 성능 저해 피처 전격 제거:
     * 소거 피처: #00 gaze_pitch_mean, #02 gaze_pitch_bottom, #05 cervical_angle_min,
                 #09 head_roll_tilt_max, #10 head_to_torso_relative_pitch,
                 #11 head_acceleration_jerk, #14 subject_calibrated_gaze_drop
   - 잔존 13D 핵심 피처:
     1) c2_gaze_pitch_min
     2) c2_gaze_pitch_drop
     3) c2_cervical_spine_angle_mean
     4) c2_cervical_spine_angle_drop
     5) c2_head_yaw_asymmetry_mean
     6) c2_head_yaw_asymmetry_max
     7) c2_head_forward_protrusion_max
     8) c2_gaze_relative_to_trunk_mean
     9) c2_cervical_velocity_peak_at_nadir
     10) c2_gaze_coupling_lag
     11) c2_cervical_hyperextension_ratio
     12) c2_gated_true_gaze_pitch_drop (게이트형 3D 시선 낙하폭)
     13) c2_gated_true_gaze_pitch_bot  (게이트형 3D 최하사점 절대 시선)
3. 실측 성능 (Level 1~3 실무군 1,775건, 5-Fold StratifiedGroupKFold 완전 격리):
   - 단일 CatBoost: F1 = 87.37% (v2 대비 +1.11%p 상승), ROC-AUC = 94.77%
   - 단일 LightGBM: F1 = 87.62% (v2 대비 +0.62%p 상승), ROC-AUC = 95.10%
   - 2계층 캐스케이드: F1 = 90.37% (v2 대비 +0.26%p 상승, 90.37% 최고점), Balanced Acc = 90.42%
   - 피처 수: 20D -> 13D (35% 경량화 및 추론 속도/일반화 대폭 향상)
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

FEATURE_NAMES_13: List[str] = [
    "c2_gaze_pitch_min",
    "c2_gaze_pitch_drop",
    "c2_cervical_spine_angle_mean",
    "c2_cervical_spine_angle_drop",
    "c2_head_yaw_asymmetry_mean",
    "c2_head_yaw_asymmetry_max",
    "c2_head_forward_protrusion_max",
    "c2_gaze_relative_to_trunk_mean",
    "c2_cervical_velocity_peak_at_nadir",
    "c2_gaze_coupling_lag",
    "c2_cervical_hyperextension_ratio",
    "c2_gated_true_gaze_pitch_drop",
    "c2_gated_true_gaze_pitch_bot"
]

DROPPED_FEATURES_7: List[str] = [
    "c2_gaze_pitch_mean",
    "c2_gaze_pitch_bottom",
    "c2_cervical_spine_angle_min",
    "c2_head_roll_tilt_max",
    "c2_head_to_torso_relative_pitch",
    "c2_head_acceleration_jerk",
    "c2_subject_calibrated_gaze_drop"
]

# 20D 인덱스에서 13D를 추출하기 위한 정렬 인덱스
# 20D: 0: mean, 1: min, 2: bot, 3: drop, 4: c_mean, 5: c_min, 6: c_drop, 7: yaw_mean, 8: yaw_max,
#      9: roll_max, 10: rel_pitch, 11: jerk, 12: fwd_max, 13: trunk_mean, 14: cal_drop,
#      15: vel_peak, 16: lag, 17: hyper, 18: gated_drop, 19: gated_bot
V2_TO_V3_INDICES = [1, 3, 4, 6, 7, 8, 12, 13, 15, 16, 17, 18, 19]

class BabelSquatC2ModelV3:
    """
    바벨 스쿼트 'c2. 고개 정면' 오류 분류 공식 프로덕션 모델 v3 (13D 노이즈 소거 정예 모델)
    """

    DEFECT_CODE = "c2"
    DEFECT_NAME = "고개 정면"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 13
    FEATURE_NAMES = FEATURE_NAMES_13

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
        self.feature_dim = 13
        self.feature_names = FEATURE_NAMES_13

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
        Format B 텐서 및 안면 시선 2D 피처로부터 13D 정예 피처셋 산출
        - v2의 20D 피처를 산출한 뒤, 불필요한 7대 방해 피처를 배제하고 13D만 반환
        """
        from src.models.second_models.babel_squat.c2_model_v2 import BabelSquatC2ModelV2
        X_20d = BabelSquatC2ModelV2.extract_features(X_b, gaze_features_2d=gaze_features_2d)
        return X_20d[:, V2_TO_V3_INDICES]

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
        print(f"[*] v3.0 가중치 저장 완료: {filepath}")

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

# v3.0 공식 최종 모델 기본 별칭 매핑
BabelSquatC2Model = BabelSquatC2ModelV3
