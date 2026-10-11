# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Model Version: v6.0 Zero-Calibrated (d5_model_v6.py)
Status: 37D 동적 어드레스 영점화 및 체형 정규화 Tri-Ensemble 분류기
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

from src.models.second_models.babel_deadlift.d5_model_v6_features import (
    extract_d5_v6_zero_calibrated_features,
    FEATURE_NAMES_37_D5_V6
)

# v6 공식 최종 확정 시점별 결정 임계값
VIEW_OPTIMAL_THRESHOLDS_V6: Dict[str, float] = {
    'view5': 0.35,  # 시점 A (우측 후방 45°)
    'view2': 0.49,  # 시점 B (우측 전방 45°)
    'view3': 0.20,  # 시점 C (정면 0°)
    'view4': 0.50,  # 시점 D (좌측 전방 45°)
    'view1': 0.49   # 시점 E (좌측 후방 45°)
}


class BabelDeadliftD5ModelV6:
    """
    37D 동적 영점화/체형보정 Tri-Ensemble (LightGBM + CatBoost + Random Forest) 분류기
    """
    def __init__(self,
                 lgb_params: Optional[Dict[str, Any]] = None,
                 cb_params: Optional[Dict[str, Any]] = None,
                 rf_params: Optional[Dict[str, Any]] = None,
                 weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
                 view_thresholds: Optional[Dict[str, float]] = None):
        self.weights = weights
        self.view_thresholds = view_thresholds or VIEW_OPTIMAL_THRESHOLDS_V6
        self.is_fitted = False

        self.lgb_params = lgb_params or {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.035,
            'num_leaves': 22,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 2
        }
        self.cb_params = cb_params or {
            'iterations': 200,
            'depth': 5,
            'learning_rate': 0.035,
            'random_seed': 42,
            'verbose': 0,
            'thread_count': 2
        }
        self.rf_params = rf_params or {
            'n_estimators': 200,
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
        """3개 서브모델 독립 학습"""
        if self.lgb_model is not None:
            self.lgb_model.fit(X, y)
        if self.cb_model is not None:
            self.cb_model.fit(X, y)
        self.rf_model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        """가중 앙상블 원시 확률(0.0~1.0) 산출"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        p_rf = self.rf_model.predict_proba(X)[:, 1]

        if self.lgb_model is not None and self.cb_model is not None:
            p_lgb = self.lgb_model.predict_proba(X)[:, 1]
            p_cb = self.cb_model.predict_proba(X)[:, 1]
            w_lgb, w_cb, w_rf = self.weights
            return w_lgb * p_lgb + w_cb * p_cb + w_rf * p_rf
        elif self.lgb_model is not None:
            p_lgb = self.lgb_model.predict_proba(X)[:, 1]
            return 0.65 * p_lgb + 0.35 * p_rf
        elif self.cb_model is not None:
            p_cb = self.cb_model.predict_proba(X)[:, 1]
            return 0.65 * p_cb + 0.35 * p_rf
        else:
            return p_rf

    def predict_calibrated_proba(self, X: np.ndarray, view_name: str = "view3") -> np.ndarray:
        """시점별 임계값 기준 0.50 정렬 캘리브레이션 확률 반환"""
        raw_p = self.predict_raw_proba(X)
        th = self.view_thresholds.get(view_name, 0.50)
        cal_p = np.where(
            raw_p >= th,
            0.50 + 0.50 * (raw_p - th) / (1.0 - th + 1e-6),
            0.50 * raw_p / (th + 1e-6)
        )
        return np.clip(cal_p, 0.0, 1.0)

    def predict_with_3state(self,
                           X: np.ndarray,
                           view_name: str = "view3",
                           margin: float = 0.06) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        3-State 판정:
        - raw_p: 원시 확률
        - preds: 판정 (0: 정상, 1: 결함, -1: 불확실 보류)
        - is_decided: 확정 여부 boolean
        """
        cal_p = self.predict_calibrated_proba(X, view_name=view_name)
        l_th = 0.50 - margin
        h_th = 0.50 + margin

        is_decided = (cal_p < l_th) | (cal_p >= h_th)
        preds = np.where(is_decided, (cal_p >= 0.50).astype(int), -1)
        return cal_p, preds, is_decided

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'view_thresholds': self.view_thresholds,
                'lgb_model': self.lgb_model,
                'cb_model': self.cb_model,
                'rf_model': self.rf_model,
                'is_fitted': self.is_fitted
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'BabelDeadliftD5ModelV6':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        obj = cls(
            weights=data['weights'],
            view_thresholds=data['view_thresholds']
        )
        obj.lgb_model = data['lgb_model']
        obj.cb_model = data['cb_model']
        obj.rf_model = data['rf_model']
        obj.is_fitted = data['is_fitted']
        return obj


def infer_d5_v6_sequence(kps_64f: np.ndarray,
                         view_name: str = "view3",
                         model_path: Optional[str] = None,
                         uncertainty_margin: float = 0.06) -> Dict[str, Any]:
    """
    단일 64프레임 시퀀스에 대한 프로덕션 실서비스 추론 표준 함수
    """
    if model_path is None:
        model_path = os.path.join(CURRENT_DIR, "weights", "d5_model_v6_best.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = BabelDeadliftD5ModelV6.load(model_path)
    feats = extract_d5_v6_zero_calibrated_features(kps_64f, view_name=view_name)
    feats_2d = feats.reshape(1, -1)

    raw_p = float(model.predict_raw_proba(feats_2d)[0])
    cal_p, preds, is_dec = model.predict_with_3state(feats_2d, view_name=view_name, margin=uncertainty_margin)

    pred_val = int(preds[0])
    dec_bool = bool(is_dec[0])

    if not dec_bool:
        label_str = "판정 보류 (자세 모호, 재확인 권장)"
    elif pred_val == 1:
        label_str = "결함 (바벨이 몸에서 과도하게 멀어짐)"
    else:
        label_str = "정상 (바벨 밀착 궤적 양호)"

    return {
        'defect_prob': raw_p,
        'calibrated_prob': float(cal_p[0]),
        'prediction': pred_val,
        'is_decided': dec_bool,
        'decision_label': label_str,
        'view_name': view_name,
        'features_dim': len(feats)
    }
