# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Model Version: v7.0 Two-Tier Cascade (d5_model_v7_two_tier.py)
Status: ★ 4단계 모호성 분할 기반 2계층 구조 (Level 4 폐기, Tier 1: Level 1 평가, Tier 2: Level 1~3 평가)
        - 최신 v6.0 (37D Zero-Calibrated Tri-Ensemble) 전면 계승
        - Tier 1 Gatekeeper: Level 1 고순도 명확군 기반 초고정밀 게이트키퍼
        - Tier 2 Deep Resolver: Level 1~3 실무 정제 데이터 기반 심층 리졸버
        - Two-Tier Cascade: 고확신 영역 Fast-path + 경계선 Deep-path 결합
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

# v7.0 공식 최종 확정 시점별 결정 임계값 (Level 1~3 실무군 최적화)
VIEW_OPTIMAL_THRESHOLDS_V7: Dict[str, float] = {
    'view5': 0.26,  # 시점 A (우측 후방 45°)
    'view2': 0.33,  # 시점 B (우측 전방 45°)
    'view3': 0.27,  # 시점 C (정면 0°)
    'view4': 0.36,  # 시점 D (좌측 전방 45°)
    'view1': 0.64   # 시점 E (좌측 후방 45°)
}

# v7.0 공식 최종 확정 시점별 3-State 적응형 신뢰도 마진 (공식 권장 7.1% 보류 사양)
VIEW_ADAPTIVE_MARGINS_V7: Dict[str, float] = {
    'view5': 0.14,  # 시점 A
    'view2': 0.12,  # 시점 B
    'view3': 0.17,  # 시점 C (정면 깊이 축 불확실성 흡수)
    'view4': 0.01,  # 시점 D (F1 99.3%로 사실상 전수 판정)
    'view1': 0.14   # 시점 E
}


class SubEnsembleModel:
    """단일 계층용 Tri-Ensemble (LightGBM + CatBoost + Random Forest)"""
    def __init__(self,
                 lgb_params: Optional[Dict[str, Any]] = None,
                 cb_params: Optional[Dict[str, Any]] = None,
                 rf_params: Optional[Dict[str, Any]] = None,
                 weights: Tuple[float, float, float] = (0.50, 0.30, 0.20)):
        self.weights = weights
        self.lgb_params = lgb_params or {
            'n_estimators': 180,
            'max_depth': 5,
            'learning_rate': 0.035,
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
            'learning_rate': 0.035,
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

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.lgb_model is not None:
            self.lgb_model.fit(X, y)
        if self.cb_model is not None:
            self.cb_model.fit(X, y)
        self.rf_model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SubEnsembleModel is not fitted yet.")

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


class BabelDeadliftD5ModelV7TwoTier:
    """
    바벨 데드리프트 d5 2계층 구조 (Two-Tier Cascade) 공식 모델
    - Tier 1: Level 1 명확군 전문 게이트키퍼
    - Tier 2: Level 1~3 실무 정제군 전문 심층 리졸버
    - Level 4: 오염/역전 노이즈 원천 폐기
    """
    def __init__(self,
                 tier1_weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
                 tier2_weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
                 tau_low: float = 0.25,
                 tau_high: float = 0.75,
                 view_thresholds: Optional[Dict[str, float]] = None):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.view_thresholds = view_thresholds or VIEW_OPTIMAL_THRESHOLDS_V7

        # Tier 1 모델 (Level 1 명확군)
        self.tier1_model = SubEnsembleModel(weights=tier1_weights)
        # Tier 2 모델 (Level 1~3 실무군)
        self.tier2_model = SubEnsembleModel(weights=tier2_weights)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, ambiguity_levels: np.ndarray):
        """
        2계층 학습 수행:
        - Level 4는 원천 배제/폐기
        - Tier 1: Level 1만으로 학습
        - Tier 2: Level 1~3 전체로 학습
        """
        # Level 4 폐기 마스크
        valid_mask = (ambiguity_levels != 4)
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        levels_clean = ambiguity_levels[valid_mask]

        # Tier 1: Level 1 고순도 데이터
        mask_t1 = (levels_clean == 1)
        X_t1, y_t1 = X_clean[mask_t1], y_clean[mask_t1]
        self.tier1_model.fit(X_t1, y_t1)

        # Tier 2: Level 1~3 실무 정제 데이터
        self.tier2_model.fit(X_clean, y_clean)

        self.is_fitted = True
        return self

    def predict_tier1_proba(self, X: np.ndarray) -> np.ndarray:
        """Tier 1 원시 결함 확률"""
        return self.tier1_model.predict_proba(X)

    def predict_tier2_proba(self, X: np.ndarray) -> np.ndarray:
        """Tier 2 원시 결함 확률"""
        return self.tier2_model.predict_proba(X)

    def predict_cascade_proba(self, X: np.ndarray) -> np.ndarray:
        """
        2계층 캐스케이드 확률 융합:
        - p_t1 <= tau_low (확실한 정상): Tier 1 확정 (Fast-path)
        - p_t1 >= tau_high (확실한 결함): Tier 1 확정 (Fast-path)
        - tau_low < p_t1 < tau_high (경계선 모호 구간): 0.3 * p_t1 + 0.7 * p_t2 (Deep-path)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        p_t1 = self.predict_tier1_proba(X)
        p_t2 = self.predict_tier2_proba(X)

        p_cascade = np.zeros(len(X), dtype=np.float32)
        for i in range(len(X)):
            if p_t1[i] <= self.tau_low:
                p_cascade[i] = p_t1[i]
            elif p_t1[i] >= self.tau_high:
                p_cascade[i] = p_t1[i]
            else:
                p_cascade[i] = 0.30 * p_t1[i] + 0.70 * p_t2[i]

        return np.clip(p_cascade, 0.0, 1.0)

    def predict_calibrated_proba(self, X: np.ndarray, view_name: str = "view3") -> np.ndarray:
        """시점별 임계값 기준 0.50 정렬 캘리브레이션 확률 반환"""
        raw_p = self.predict_cascade_proba(X)
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
                            margin: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3-State 판정 (cal_p, preds, is_decided): 공식 권장 시점 적응형 마진 기본 적용"""
        cal_p = self.predict_calibrated_proba(X, view_name=view_name)
        m_val = margin if margin is not None else VIEW_ADAPTIVE_MARGINS_V7.get(view_name, 0.12)
        l_th = 0.50 - m_val
        h_th = 0.50 + m_val

        is_decided = (cal_p < l_th) | (cal_p >= h_th)
        preds = np.where(is_decided, (cal_p >= 0.50).astype(int), -1)
        return cal_p, preds, is_decided

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tier1_model': self.tier1_model,
                'tier2_model': self.tier2_model,
                'tau_low': self.tau_low,
                'tau_high': self.tau_high,
                'view_thresholds': self.view_thresholds,
                'is_fitted': self.is_fitted
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'BabelDeadliftD5ModelV7TwoTier':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        obj = cls(
            tau_low=data['tau_low'],
            tau_high=data['tau_high'],
            view_thresholds=data['view_thresholds']
        )
        obj.tier1_model = data['tier1_model']
        obj.tier2_model = data['tier2_model']
        obj.is_fitted = data['is_fitted']
        return obj


def infer_d5_v7_sequence(kps_64f: np.ndarray,
                          view_name: str = "view3",
                          model_path: Optional[str] = None,
                          uncertainty_margin: Optional[float] = None) -> Dict[str, Any]:
    """단일 시퀀스 2계층 프로덕션 실서비스 추론 (공식 최종 확정 사양)"""
    if model_path is None:
        model_path = os.path.join(CURRENT_DIR, "weights", "d5_model_final.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(CURRENT_DIR, "weights", "d5_model_v7_two_tier_best.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = BabelDeadliftD5ModelV7TwoTier.load(model_path)
    feats = extract_d5_v6_zero_calibrated_features(kps_64f, view_name=view_name)
    feats_2d = feats.reshape(1, -1)

    raw_p = float(model.predict_cascade_proba(feats_2d)[0])
    p_t1 = float(model.predict_tier1_proba(feats_2d)[0])
    p_t2 = float(model.predict_tier2_proba(feats_2d)[0])
    cal_p, preds, is_dec = model.predict_with_3state(feats_2d, view_name=view_name, margin=uncertainty_margin)

    pred_val = int(preds[0])
    dec_bool = bool(is_dec[0])

    path_used = "Fast-path (Tier 1)" if (p_t1 <= model.tau_low or p_t1 >= model.tau_high) else "Deep-path (Tier 1+2 Cascade)"

    if not dec_bool:
        label_str = "판정 보류 (자세 모호, 재확인 권장)"
    elif pred_val == 1:
        label_str = "결함 (바벨이 몸에서 과도하게 멀어짐)"
    else:
        label_str = "정상 (바벨 밀착 궤적 양호)"

    return {
        'defect_prob': raw_p,
        'tier1_prob': p_t1,
        'tier2_prob': p_t2,
        'calibrated_prob': float(cal_p[0]),
        'prediction': pred_val,
        'is_decided': dec_bool,
        'decision_label': label_str,
        'inference_path': path_used,
        'view_name': view_name,
        'features_dim': len(feats)
    }
