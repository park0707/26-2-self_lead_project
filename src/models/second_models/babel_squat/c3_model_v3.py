# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c3. 발과 무릎의 방향 일치 (Knee-Toe Alignment / Knee Valgus/Varus)
Model Version: v3.0 (c3_model_v3.py)
Status: ★ 최종 확정 프로덕션 모델 (v3-A 20D: 4대 저해 피처 소거 정예 모델)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 64프레임 표준 3차원 골격 시계열 텐서 Format B (N, 64, 92) 또는 단일 시퀀스 (64, 92)
   - [확장 입력]: 3D 발 관절 (Left/Right Foot, Ankle, Knee, Hip 3차원 좌표계)
2. 피처 엔지니어링 (최종 확정 20D 정예 생체역학 피처셋):
   - v2 24D 중 LOFO(Leave-One-Feature-Out) 소거 실험으로 규명된 4대 성능 저해 피처 전격 제거:
     * 소거 피처: #00 knee_ankle_ratio_at_nadir, #04 knee_width_jerk,
                 #11 shank_verticality_at_nadir, #12 shank_verticality_wobble
   - 잔존 20D 핵심 피처:
     1) c3_knee_ankle_ratio_mean
     2) c3_knee_ankle_ratio_drop (핵심 필수 피처)
     3) c3_knee_inward_vel_max
     4) c3_net_valgus_at_nadir
     5) c3_net_valgus_max
     6) c3_valgus_increase_from_standing
     7) c3_bilateral_valgus_asymmetry
     8) c3_knee_collapse_accel_at_nadir (핵심 필수 피처)
     9) c3_knee_flex_valgus_phase_lag
     10) c3_knee_asymmetry_mean
     11) c3_knee_angle_flex_min
     12) c3_ascent_valgus_recollapse_ratio
     13) c3_true_knee_toe_angle_nadir
     14) c3_true_knee_toe_angle_drop (핵심 필수 피처)
     15) c3_true_knee_toe_angle_max
     16) c3_knee_inward_deviation_nadir (핵심 필수 피처: 이탈거리 cm)
     17) c3_knee_inward_deviation_drop
     18) c3_bilateral_knee_toe_asym (핵심 필수 피처: 비대칭도)
     19) c3_tibia_torsion_nadir
     20) c3_foot_roll_pronation_nadir
3. 실측 성능 (Level 1~3 실무군 1,777건, 5-Fold StratifiedGroupKFold 완전 격리):
   - 단일 CatBoost: F1 = 92.52%, ROC-AUC = 97.60%
   - 단일 LightGBM: F1 = 91.58%, ROC-AUC = 97.45%
   - 2계층 캐스케이드: F1 = 93.72% (v1 대비 +2.70%p, v2 대비 +0.50%p 상승, 최고치 경신!)
     * Precision = 94.76%
     * Recall = 92.71% (+1.14%p)
     * Balanced Accuracy = 93.85% (+0.46%p)
     * ROC-AUC = 98.19% (+0.15%p)
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

FEATURE_NAMES_20: List[str] = [
    "c3_knee_ankle_ratio_mean",
    "c3_knee_ankle_ratio_drop",
    "c3_knee_inward_vel_max",
    "c3_net_valgus_at_nadir",
    "c3_net_valgus_max",
    "c3_valgus_increase_from_standing",
    "c3_bilateral_valgus_asymmetry",
    "c3_knee_collapse_accel_at_nadir",
    "c3_knee_flex_valgus_phase_lag",
    "c3_knee_asymmetry_mean",
    "c3_knee_angle_flex_min",
    "c3_ascent_valgus_recollapse_ratio",
    "c3_true_knee_toe_angle_nadir",
    "c3_true_knee_toe_angle_drop",
    "c3_true_knee_toe_angle_max",
    "c3_knee_inward_deviation_nadir",
    "c3_knee_inward_deviation_drop",
    "c3_bilateral_knee_toe_asym",
    "c3_tibia_torsion_nadir",
    "c3_foot_roll_pronation_nadir"
]

DROPPED_FEATURES_4: List[str] = [
    "c3_knee_ankle_ratio_at_nadir",
    "c3_knee_width_jerk",
    "c3_shank_verticality_at_nadir",
    "c3_shank_verticality_wobble"
]

V2_TO_V3_INDICES = [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

class BabelSquatC3ModelV3:
    """
    바벨 스쿼트 'c3. 발과 무릎의 방향 일치' 오류 분류 공식 프로덕션 모델 v3 (20D 노이즈 소거 정예 모델)
    """

    DEFECT_CODE = "c3"
    DEFECT_NAME = "발과 무릎의 방향 일치"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 20
    FEATURE_NAMES = FEATURE_NAMES_20

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
                random_seed=42, thread_count=-1, verbose=False,
                allow_writing_files=False
            )

    @classmethod
    def extract_features(
        cls,
        X_b: np.ndarray,
        foot_features_8d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Format B 텐서 및 3D 발 관절 8D 피처로부터 20D 정예 피처셋 산출"""
        from src.models.second_models.babel_squat.c3_model_v2 import BabelSquatC3ModelV2
        X_24d = BabelSquatC3ModelV2.extract_features(X_b, foot_features_8d=foot_features_8d)
        return X_24d[:, V2_TO_V3_INDICES]

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
        print(f"[*] c3 v3.0 가중치 저장 완료: {filepath}")

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

BabelSquatC3Model = BabelSquatC3ModelV3
