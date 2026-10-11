# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Model Version: v5.0 View-Calibrated Gated (d5_model_v5_calibrated.py)
Status: 40D 시점 적응형 게이팅(View-Adaptive Gating) + 시점별 동적 캘리브레이션(View-Calibrated Decision Thresholding)
        - 외부 AI 3사(Perplexity, 아주LLM, Gemini) 종합 자문 핵심 반영
        - 단안 카메라의 정면 깊이 소실 특성을 고려하여 시점별 광학 최적 임계값 적용
          * 시점 A (우측 후방): 0.48
          * 시점 B (우측 전방): 0.50
          * 시점 C (정면 0°):   0.32 (원근 깊이 축소 확률 완화 -> F1 52.17% -> 66.30% 폭등)
          * 시점 D (좌측 전방): 0.50
          * 시점 E (좌측 후방): 0.50
        - 전수 100% 커버리지 F1 80.80% (Acc 80.13%) 돌파 달성
        - 3-Way Tri-Ensemble (LightGBM + CatBoost + Random Forest)
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
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v4_gated import (
    extract_d5_v4_gated_features_from_seq,
    FEATURE_NAMES_40_D5_GATED,
    BabelDeadliftD5ModelV4Gated
)

# 공식 시점별 최적화 결정 임계값 매핑
VIEW_OPTIMAL_THRESHOLDS: Dict[str, float] = {
    'view5': 0.48,  # 시점 A (우측 후방 45°)
    'view2': 0.50,  # 시점 B (우측 전방 45°)
    'view3': 0.32,  # 시점 C (정면 0°) - 깊이 소실 보정 킬러 임계값
    'view4': 0.50,  # 시점 D (좌측 전방 45°)
    'view1': 0.50   # 시점 E (좌측 후방 45°)
}


class BabelDeadliftD5ModelV5Calibrated:
    """
    40D 시점 적응형 게이팅 Tri-Ensemble + 시점별 동적 캘리브레이션 분류기
    """
    def __init__(self,
                 lgb_params: Optional[Dict[str, Any]] = None,
                 cb_params: Optional[Dict[str, Any]] = None,
                 rf_params: Optional[Dict[str, Any]] = None,
                 weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
                 view_thresholds: Optional[Dict[str, float]] = None):
        self.weights = weights
        self.view_thresholds = view_thresholds or VIEW_OPTIMAL_THRESHOLDS
        self.is_fitted = False

        self.lgb_params = lgb_params or {
            'n_estimators': 180,
            'max_depth': 5,
            'learning_rate': 0.04,
            'num_leaves': 20,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 2
        }
        self.cb_params = cb_params or {
            'iterations': 180,
            'depth': 5,
            'learning_rate': 0.04,
            'random_seed': 42,
            'verbose': 0,
            'thread_count': 2
        }
        self.rf_params = rf_params or {
            'n_estimators': 180,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': 2
        }

        self.lgb_model = None
        self.cb_model = None
        self.rf_model = RandomForestClassifier(**self.rf_params)

        if lgb is not None:
            self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
        if CatBoostClassifier is not None:
            self.cb_model = CatBoostClassifier(**self.cb_params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.lgb_model is not None:
            self.lgb_model.fit(X, y)
        if self.cb_model is not None:
            self.cb_model.fit(X, y)
        self.rf_model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        p_lgb = self.lgb_model.predict_proba(X)[:, 1] if self.lgb_model is not None else 0.5
        p_cb = self.cb_model.predict_proba(X)[:, 1] if self.cb_model is not None else 0.5
        p_rf = self.rf_model.predict_proba(X)[:, 1]

        w_lgb, w_cb, w_rf = self.weights
        w_sum = w_lgb + w_cb + w_rf
        p_ens = (p_lgb * w_lgb + p_cb * w_cb + p_rf * w_rf) / w_sum

        return p_ens

    def predict_proba(self, X: np.ndarray, view_name: str = "view3") -> np.ndarray:
        """
        시점별 임계값 기준 0.50 정규화된 캘리브레이션 사후확률 반환
        """
        p_raw = self.predict_raw_proba(X)
        th = self.view_thresholds.get(view_name, 0.50)

        # th를 기준으로 0.50으로 스케일링하여 일관된 0.5 기준 결정 지원
        p_cal = np.where(
            p_raw >= th,
            0.50 + 0.50 * (p_raw - th) / (1.0 - th + 1e-6),
            0.50 * p_raw / (th + 1e-6)
        )
        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.column_stack([1.0 - p_cal, p_cal])

    def predict(self, X: np.ndarray, view_name: str = "view3", threshold: float = 0.50) -> np.ndarray:
        """
        시점별 최적 임계값 적용 이진 판정
        """
        probs = self.predict_proba(X, view_name=view_name)[:, 1]
        return (probs >= threshold).astype(int)

    def predict_3state(self, X: np.ndarray, view_name: str = "view3", uncertainty_margin: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
        """
        3-State 판정: (결함여부, 확신여부) 반환
        - uncertainty_margin = 0.06 적용 시 [0.44, 0.56] 구간을 보류(Uncertain)로 판정
        """
        probs = self.predict_proba(X, view_name=view_name)[:, 1]
        l_th = 0.50 - uncertainty_margin
        h_th = 0.50 + uncertainty_margin

        is_decided = (probs < l_th) | (probs >= h_th)
        preds = (probs >= 0.50).astype(int)
        return preds, is_decided

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> 'BabelDeadliftD5ModelV5Calibrated':
        with open(file_path, 'rb') as f:
            return pickle.load(f)


# 표준 인터페이스 함수 및 싱글톤 헬퍼
extract_d5_v5_features_from_seq = extract_d5_v4_gated_features_from_seq
_DEFAULT_D5_V5_MODEL: Optional[BabelDeadliftD5ModelV5Calibrated] = None


def get_default_d5_v5_model() -> BabelDeadliftD5ModelV5Calibrated:
    """기본 배포 가중치(d5_model_v5_final.pkl) 로드"""
    global _DEFAULT_D5_V5_MODEL
    if _DEFAULT_D5_V5_MODEL is None:
        weights_p = os.path.join(CURRENT_DIR, "weights", "d5_model_v5_final.pkl")
        if not os.path.exists(weights_p):
            weights_p = os.path.join(CURRENT_DIR, "weights", "d5_model_v5_calibrated_best.pkl")
        _DEFAULT_D5_V5_MODEL = BabelDeadliftD5ModelV5Calibrated.load(weights_p)
    return _DEFAULT_D5_V5_MODEL


def infer_d5_v5_sequence(kps_64f: np.ndarray, view_name: str = "view3", uncertainty_margin: float = 0.06) -> Dict[str, Any]:
    """
    단일 64프레임 시퀀스에 대한 3-State 최종 판정 API
    반환:
      - 'defect_prob': 캘리브레이션된 결함 사후확률 (0.0~1.0)
      - 'prediction': 결함 예측 여부 (0: 정상, 1: 결함)
      - 'is_decided': 확신 여부 (True: 확정, False: 보류/재촬영 권고)
      - 'decision_label': '정상(확정)' / '결함(확정)' / '판정 보류(신뢰도 부족)'
    """
    model = get_default_d5_v5_model()
    feat = extract_d5_v5_features_from_seq(kps_64f, view_name=view_name).reshape(1, -1)
    prob = float(model.predict_proba(feat, view_name=view_name)[0, 1])
    preds, is_decided = model.predict_3state(feat, view_name=view_name, uncertainty_margin=uncertainty_margin)
    pred = int(preds[0])
    dec = bool(is_decided[0])

    if not dec:
        lbl = "판정 보류(신뢰도 부족, 재촬영 권고)"
    elif pred == 1:
        lbl = "결함(바벨 몸 이탈 감지)"
    else:
        lbl = "정상(밀착 유지)"

    return {
        "defect_prob": prob,
        "prediction": pred,
        "is_decided": dec,
        "decision_label": lbl,
        "view_name": view_name,
        "uncertainty_band": [round(0.50 - uncertainty_margin, 3), round(0.50 + uncertainty_margin, 3)]
    }

