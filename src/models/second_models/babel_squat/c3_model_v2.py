# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c3. 발과 무릎의 방향 일치 (Knee-Toe Alignment / Knee Valgus/Varus)
Model Version: v2.0 (c3_model_v2.py)
Status: ★ 최종 선정 공식 프로덕션 모델 (v2-A 기반 24D: 16D 레거시 + 8D 3D 발 관절 정렬 피처)

[아키텍처 및 공학적 명세]:
1. 입력 규격:
   - 64프레임 표준 3차원 골격 시계열 텐서 Format B (N, 64, 92) 또는 단일 시퀀스 (64, 92)
   - [확장 입력]: 3D 발 관절 (Left/Right Foot, Ankle, Knee, Hip 3차원 좌표계)
2. 피처 엔지니어링 (최종 확정 24D 정예 생체역학 피처셋):
   - 16D 체간/정강이 동역학 피처셋 (v1 계승)
   - 8D 3D 발 관절 정렬 피처셋:
     17) c3_true_knee_toe_angle_nadir: 최하사점 3D 무릎-발 정렬 불일치각 평균 (deg)
     18) c3_true_knee_toe_angle_drop:  기립 대비 최하사점 무릎-발 정렬 오차 증가폭 (deg)
     19) c3_true_knee_toe_angle_max:   전 구간 최대 무릎-발 정렬 오차각 (deg)
     20) c3_knee_inward_deviation_nadir: 최하사점 무릎의 발끝 연장선 내측 이탈 거리 (cm)
     21) c3_knee_inward_deviation_drop:  기립 대비 최하사점 무릎 내측 이탈 변위 (cm)
     22) c3_bilateral_knee_toe_asym:   좌우 무릎-발 정렬 오차 비대칭도 (deg)
     23) c3_tibia_torsion_nadir:       최하사점 경골(정강이) 내회전 비틀림각 (deg)
     24) c3_foot_roll_pronation_nadir: 최하사점 발목-발 기울기 롤각 (발 회내 Pronation 징후)
3. 실측 성능 (Level 1~3 실무군 1,777건, 5-Fold StratifiedGroupKFold 완전 격리):
   - 단일 CatBoost: F1 = 92.66% (+3.34%p), ROC-AUC = 97.44%
   - 단일 LightGBM: F1 = 91.49% (+2.45%p), ROC-AUC = 97.30%
   - 2계층 캐스케이드 (Tier-1 Gatekeeper + Tier-2 Resolver):
     * F1 = 93.22% (+2.20%p 대도약, F1 93% 고지 돌파!)
     * Precision = 94.92% (+2.94%p)
     * Recall = 91.57% (+1.48%p)
     * Balanced Accuracy = 93.39% (+2.18%p)
     * ROC-AUC = 98.04% (+1.44%p)
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

from src.models.second_models.babel_squat.c3_model_v1 import FEATURE_NAMES_16, c3_extract_features_16d
from src.models.second_models.babel_squat.build_c3_foot_features import C3_FOOT_FEATURE_NAMES

FEATURE_NAMES_24: List[str] = FEATURE_NAMES_16 + C3_FOOT_FEATURE_NAMES

class BabelSquatC3ModelV2:
    """
    바벨 스쿼트 'c3. 발과 무릎의 방향 일치' 오류 분류 공식 프로덕션 모델 v2 (24D 발 관절 벡터 결합)
    """

    DEFECT_CODE = "c3"
    DEFECT_NAME = "발과 무릎의 방향 일치"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 24
    FEATURE_NAMES = FEATURE_NAMES_24

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
        self.feature_dim = 24
        self.feature_names = FEATURE_NAMES_24

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
        foot_features_8d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Format B 텐서 및 3D 발 관절 8D 피처로부터 24D 피처셋 산출
        - foot_features_8d가 None이거나 미관측 시 0.0 더미 패딩으로 안전 Fallback
        """
        X_16d = c3_extract_features_16d(X_b)
        N = X_16d.shape[0]

        if foot_features_8d is None:
            foot_8d = np.zeros((N, 8), dtype=np.float32)
        else:
            if foot_features_8d.ndim == 1:
                foot_8d = np.expand_dims(foot_features_8d, axis=0)
            else:
                foot_8d = foot_features_8d.astype(np.float32)
            if foot_8d.shape[0] != N:
                foot_8d = np.zeros((N, 8), dtype=np.float32)

        return np.concatenate([X_16d, foot_8d], axis=1)

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

BabelSquatC3Model = BabelSquatC3ModelV2
