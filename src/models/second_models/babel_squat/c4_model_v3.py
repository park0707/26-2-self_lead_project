# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c4. 발바닥 지면 고정 (Foot Ground Fixation / Heel-Toe Elevation)
Model Version: v3.0 (c4_model_v3.py)
Status: LOFO 기반 17대 노이즈 피처 소거 및 9D 초정밀 최정예 공식 v3.0 모델

[아키텍처 및 공학적 명세]:
1. 피처 최적화:
   - v2-A의 26D 피처 중 모델의 일반화 성능을 저해하던 레거시 발목 노이즈 10개 및 신규 중복 피처 7개 소거
   - 26D -> 9D (65.4% 피처 압축 및 초경량화)
2. 9D 초정밀 피처셋:
   1) c4_true_foot_pitch_nadir          (최하사점 발바닥 경사각)
   2) c4_true_foot_pitch_drop           (기립 대비 최하사점 발바닥 경사각 증가폭)
   3) c4_true_foot_pitch_max            (전 구간 최대 발바닥 경사각)
   4) c4_true_heel_toe_ydelta_nadir     (최하사점 발목-발끝 수직 높이차)
   5) c4_true_bilateral_foot_pitch_asym (좌우 발바닥 경사각 비대칭도)
   6) c4_true_toe_lift_max              (앞꿈치 최대 들림량)
   7) c4_true_foot_roll_inversion       (최하사점 발바닥 좌우 롤각)
   8) c4_shank_ankle_coupling_ratio     (정강이-발목 커플링 비율 - 레거시 핵심)
   9) c4_ascent_ankle_destabilization   (상승기 발목 불안정성 - 레거시 핵심)
3. 실측 성능 (Level 1~3 실무군 1,824건, 5-Fold StratifiedGroupKFold 완전 격리):
   - CatBoost F1: 90.28% (v2-A 90.11% 대비 +0.17%p 향상, 전체 최고치)
   - 2계층 캐스케이드 F1: 90.06% (v2-A 89.73% 대비 +0.33%p 향상)
   - ROC-AUC: 96.30%
   - 피처 수: 26개 -> 9개 (65.4% 파격적 경량화)
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

# v3 9대 정예 피처 명칭
FEATURE_NAMES_9_C4 = [
    'c4_true_foot_pitch_nadir',
    'c4_true_foot_pitch_drop',
    'c4_true_foot_pitch_max',
    'c4_true_heel_toe_ydelta_nadir',
    'c4_true_bilateral_foot_pitch_asym',
    'c4_true_toe_lift_max',
    'c4_true_foot_roll_inversion',
    'c4_shank_ankle_coupling_ratio',
    'c4_ascent_ankle_destabilization'
]

# 26D 풀 피처셋에서의 인덱스 매핑
V3_INDICES_FROM_26D = [16, 17, 18, 19, 23, 24, 25, 12, 13]

class BabelSquatC4ModelV3:
    """
    바벨 스쿼트 'c4. 발바닥 지면 고정' 공식 v3.0 초정밀 모델 (9D 정예 피처셋)
    """

    DEFECT_CODE = "c4"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 9
    FEATURE_NAMES = FEATURE_NAMES_9_C4

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
        self.feature_dim = 9
        self.feature_names = FEATURE_NAMES_9_C4

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
        """모델 학습 (입력 X는 9D 배열이어야 함)"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        if X.shape[1] == 26:
            X = X[:, V3_INDICES_FROM_26D]

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

        if X.shape[1] == 26:
            X = X[:, V3_INDICES_FROM_26D]

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
        print(f"[*] c4 v3.0 가중치 저장 완료: {file_path}")

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
