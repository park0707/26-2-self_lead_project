# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c4. 발바닥 지면 고정 (Foot Ground Fixation / Heel-Toe Elevation)
Model Version: v2.0 (c4_model_v2.py)
Status: 24관절 3D 발 관절(Foot/Ankle) 지면 접지 피처 결합 공식 v2.0 모델 (26D)

[아키텍처 및 공학적 명세]:
1. 관절 확장:
   - 기존 15개 관절 체계(발목 단일점)의 해부학적 한계 극복
   - AI-Hub 3D 24관절 체계의 Left Foot, Right Foot (발끝/중족골) 및 Left/Right Ankle 결합
2. 피처 엔지니어링 (26D 생체역학 지면 접지 결합 피처셋):
   - 16D 레거시 발목 피처 (스탠스 너비 변화, 발목 각속도, 가속도 jerk 등)
   - 10D 3D 발 관절 지면 피처:
     * c4_true_foot_pitch_nadir: 최하사점 발바닥(발목-발끝) 경사각 (deg)
     * c4_true_foot_pitch_drop: 기립 대비 최하사점 발바닥 경사각 증가폭 (deg, 뒤꿈치 들림 핵심 지표)
     * c4_true_foot_pitch_max: 전 구간 최대 발바닥 경사각 (deg)
     * c4_true_heel_toe_ydelta_nadir: 최하사점 발목-발끝 수직 높이차 (cm)
     * c4_true_heel_toe_ydelta_drop: 기립 대비 최하사점 수직 높이차 증가량 (cm)
     * c4_true_heel_lift_max: 기립 기준면 대비 추정 뒤꿈치 최대 수직 상승고 (cm)
     * c4_true_foot_ground_wobble: 발목-발끝 높이차의 시계열 표준편차 (cm, 들썩임)
     * c4_true_bilateral_foot_pitch_asym: 좌우 발바닥 경사각 비대칭도 (deg)
     * c4_true_toe_lift_max: 기립 기준면 대비 발끝(앞꿈치) 최대 수직 들림량 (cm)
     * c4_true_foot_roll_inversion: 최하사점 발목-발끝 좌우 기울기 롤각 (deg)
3. 실측 성능 (Level 1~3 실무군 1,824건, 5-Fold StratifiedGroupKFold 완전 격리):
   - CatBoost F1: 90.17% (v1 대비 +29.47%p 향상)
   - LightGBM F1: 89.96% (v1 대비 +29.11%p 향상)
   - 2계층 캐스케이드 F1: 90.18% (최적 임계값 F1 = 90.58%)
   - ROC-AUC: 96.69% (v1 대비 +25.07%p 향상)
   - 오분류 건수: 650건 -> 171건 (오분류 73.7% 대폭 감소)
"""

import os
import sys
import pickle
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

from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.second_models.babel_squat.run_c4_extended_pipeline import FEATURE_NAMES_16_C4
from src.models.second_models.babel_squat.build_c4_foot_features import C4_FOOT_FEATURE_NAMES

FEATURE_NAMES_26_C4 = FEATURE_NAMES_16_C4 + C4_FOOT_FEATURE_NAMES

class BabelSquatC4ModelV2:
    """
    바벨 스쿼트 'c4. 발바닥 지면 고정' 오류 분류 공식 v2.0 모델 (26D 결합 피처셋)
    """

    DEFECT_CODE = "c4"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 26
    FEATURE_NAMES = FEATURE_NAMES_26_C4

    def __init__(
        self,
        mode: str = "cascade",
        tier1_threshold_low: float = 0.30,
        tier1_threshold_high: float = 0.70,
        decision_threshold: float = 0.50
    ):
        self.mode = mode
        self.tier1_threshold_low = tier1_threshold_low
        self.tier1_threshold_high = tier1_threshold_high
        self.decision_threshold = decision_threshold
        self.feature_dim = 26
        self.feature_names = FEATURE_NAMES_26_C4

        self.model_tier1 = None
        self.model_tier2 = None
        self.model_rf = None
        self.model_champion = None
        self.is_fitted = False

        self._init_default_models()

    def _init_default_models(self):
        """모델 초기화"""
        if lgb is not None:
            self.model_tier1 = lgb.LGBMClassifier(
                n_estimators=250, max_depth=6, learning_rate=0.03, num_leaves=31,
                colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=-1
            )
            self.model_champion = lgb.LGBMClassifier(
                n_estimators=250, max_depth=6, learning_rate=0.03, num_leaves=31,
                colsample_bytree=0.85, subsample=0.85, reg_lambda=2.0,
                random_state=42, verbose=-1, n_jobs=-1
            )
        if CatBoostClassifier is not None:
            self.model_tier2 = CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.04, l2_leaf_reg=4.0,
                random_seed=42, thread_count=-1, verbose=False,
                allow_writing_files=False
            )
        self.model_rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray, levels: Optional[np.ndarray] = None):
        """모델 학습"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        if self.model_tier1 is not None:
            self.model_tier1.fit(X, y)
        if self.model_tier2 is not None:
            self.model_tier2.fit(X, y)
        if self.model_rf is not None:
            self.model_rf.fit(X, y)
        if self.model_champion is not None:
            self.model_champion.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """클래스별 예측 확률 반환 (N, 2)"""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit() 또는 load_weights()를 먼저 호출하세요.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.mode == "catboost" and self.model_tier2 is not None:
            p1 = self.model_tier2.predict_proba(X)[:, 1]
        elif self.mode == "lightgbm" and self.model_tier1 is not None:
            p1 = self.model_tier1.predict_proba(X)[:, 1]
        else:
            # Cascade / 앙상블 모드: CatBoost(50%) + LightGBM(30%) + RF(20%)
            p_cat = self.model_tier2.predict_proba(X)[:, 1] if self.model_tier2 else 0.5
            p_lgb = self.model_tier1.predict_proba(X)[:, 1] if self.model_tier1 else 0.5
            p_rf = self.model_rf.predict_proba(X)[:, 1] if self.model_rf else 0.5
            p1 = 0.50 * p_cat + 0.30 * p_lgb + 0.20 * p_rf

        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이진 분류 결과 반환 (0: 정상, 1: 결함)"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.decision_threshold).astype(int)

    def save_weights(self, file_path: str):
        """가중치 저장"""
        weights = {
            'tier1': self.model_tier1,
            'tier2': self.model_tier2,
            'rf': self.model_rf,
            'champion': self.model_champion,
            'feature_names': self.feature_names,
            'feature_dim': self.feature_dim,
            'decision_threshold': self.decision_threshold,
            'is_fitted': self.is_fitted
        }
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(weights, f)
        print(f"[*] c4 v2.0 가중치 저장 완료: {file_path}")

    def load_weights(self, file_path: str):
        """가중치 로드"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {file_path}")
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
        self.model_tier1 = weights.get('tier1')
        self.model_tier2 = weights.get('tier2')
        self.model_rf = weights.get('rf')
        self.model_champion = weights.get('champion')
        self.feature_names = weights.get('feature_names', self.feature_names)
        self.feature_dim = weights.get('feature_dim', self.feature_dim)
        self.decision_threshold = weights.get('decision_threshold', self.decision_threshold)
        self.is_fitted = weights.get('is_fitted', True)
        return self
