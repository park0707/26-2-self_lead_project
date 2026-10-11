# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v3.0 (d3_model_v3.py)
Status: 스쿼트 v11.0 MoE 아키텍처 계승: 33D View-Conditioned 지면 접지 앙상블 & 시점 적응형 3-State 안전 게이트
        - [기반 계승] d3 v2.0 28D 생체역학 지면 접지 피처셋 100% 완전 계승
        - [혁신 1] 5D 카메라 시점 원핫 조건부 주입 (33D View-Conditioned Feature Set)
                  -> 전체 385개 데이터를 공유하여 과적합을 차단하면서도, 시점(A~E)별 고유 가림/깊이 특성을 내부 학습
        - [혁신 2] 시점별 사후 적응형 3-State 최적 임계치 분리 (View-Adaptive 3-State Gate)
                  -> 전방 뷰(B, C, D)와 후방 뷰(A, E)의 관측 신뢰도에 따른 차등적 tau_norm / tau_def 자동 적용
        - [혁신 3] CatBoost + LightGBM + Random Forest 3-Way Tri-Ensemble 결합
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

from src.models.second_models.babel_deadlift.d3_model_v2 import (
    extract_d3_v2_features_from_seq,
    FEATURE_NAMES_28_D3
)

VIEW_CODE_LIST = ["view5", "view2", "view3", "view4", "view1"]

VIEW_TO_ONENOT: Dict[str, List[float]] = {
    "view5": [1.0, 0.0, 0.0, 0.0, 0.0],  # 시점 A: 우측 후방 45°
    "view2": [0.0, 1.0, 0.0, 0.0, 0.0],  # 시점 B: 우측 전방 45°
    "view3": [0.0, 0.0, 1.0, 0.0, 0.0],  # 시점 C: 정면 0°
    "view4": [0.0, 0.0, 0.0, 1.0, 0.0],  # 시점 D: 좌측 전방 45°
    "view1": [0.0, 0.0, 0.0, 0.0, 1.0]   # 시점 E: 좌측 후방 45°
}

VIEW_ONEHOT_NAMES: List[str] = [
    "d3_view_A_right_back",   # view5
    "d3_view_B_right_front",  # view2
    "d3_view_C_front_center", # view3
    "d3_view_D_left_front",   # view4
    "d3_view_E_left_back"     # view1
]

FEATURE_NAMES_33_D3: List[str] = FEATURE_NAMES_28_D3 + VIEW_ONEHOT_NAMES

# 시점별 기본 3-State 임계치 사전 (사후 그리드 최적화로 갱신됨)
DEFAULT_VIEW_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "view5": (0.420, 0.540), # 시점 A
    "view2": (0.400, 0.520), # 시점 B
    "view3": (0.420, 0.530), # 시점 C
    "view4": (0.400, 0.500), # 시점 D
    "view1": (0.420, 0.540)  # 시점 E
}


def extract_d3_v3_features_from_seq(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 시퀀스 및 시점 코드로부터 33D View-Conditioned 지면 접지 피처 추출
    """
    f28 = extract_d3_v2_features_from_seq(kps_33_64f)
    v_onehot = VIEW_TO_ONENOT.get(view_code, [0.0, 0.0, 1.0, 0.0, 0.0]) # 기본값: 정면
    return np.hstack([f28, np.array(v_onehot, dtype=np.float32)])


def extract_d3_v3_features(X_kps: np.ndarray, views: Union[List[str], np.ndarray]) -> np.ndarray:
    """배치 (N, 64, 33, 3) 및 시점 리스트로부터 33D 피처 행렬 추출"""
    N = X_kps.shape[0]
    out = np.zeros((N, 33), dtype=np.float32)
    for i in range(N):
        v = str(views[i]) if i < len(views) else "view3"
        out[i] = extract_d3_v3_features_from_seq(X_kps[i], v)
    return out


class BabelDeadliftD3ModelV3:
    """
    바벨 데드리프트 d3 결함(발바닥 지면 고정) 공식 프로덕션 모델 v3.0
    - 33D View-Conditioned 지면 접지 피처셋 (28D 생체역학 + 5D 시점 원핫)
    - 3-Way Tri-Ensemble (CatBoost 50% + LightGBM 30% + Random Forest 20%)
    - 시점별 사후 적응형 3-State 최적 임계치 분리 (View-Adaptive 3-State Gate)
    """

    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    FEATURE_DIM = 33
    FEATURE_NAMES = FEATURE_NAMES_33_D3

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.decision_threshold = decision_threshold
        self.view_thresholds = view_thresholds or DEFAULT_VIEW_THRESHOLDS.copy()

        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.rf_models: List[Any] = []
        self.weights: Tuple[float, float, float] = (0.50, 0.30, 0.20)
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_33_D3

        if weights_path is not None and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def fit_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cb_params: Optional[Dict[str, Any]] = None,
        lgb_params: Optional[Dict[str, Any]] = None,
        rf_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """단일 Fold 또는 전체 데이터셋 기반 33D View-Conditioned Tri-Ensemble 학습"""
        # 1. CatBoost
        if cb_params is None:
            cb_params = {
                'iterations': 320,
                'learning_rate': 0.03,
                'depth': 4,
                'l2_leaf_reg': 3.0,
                'auto_class_weights': 'Balanced',
                'verbose': False,
                'random_seed': random_state
            }
        cb = CatBoostClassifier(**cb_params)
        cb.fit(X_train, y_train)

        # 2. LightGBM
        if lgb_params is None:
            pos_weight = float(np.sum(y_train == 0)) / max(1, float(np.sum(y_train == 1)))
            lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': 0.03,
                'num_leaves': 15,
                'max_depth': 4,
                'scale_pos_weight': pos_weight,
                'verbosity': -1,
                'random_state': random_state
            }
        ds_tr = lgb.Dataset(X_train, label=y_train)
        lgb_m = lgb.train(lgb_params, ds_tr, num_boost_round=250)

        # 3. Random Forest
        if rf_params is None:
            rf_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'class_weight': 'balanced',
                'random_state': random_state,
                'n_jobs': -1
            }
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)

        self.cb_models = [cb]
        self.lgb_models = [lgb_m]
        self.rf_models = [rf]
        self.is_fitted = True

    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        """33D 피처 행렬로부터 결함 확률 추론"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        cb_preds = np.zeros(len(X), dtype=np.float32)
        for m in self.cb_models:
            cb_preds += m.predict_proba(X)[:, 1]
        cb_preds /= max(1, len(self.cb_models))

        lgb_preds = np.zeros(len(X), dtype=np.float32)
        for m in self.lgb_models:
            lgb_preds += m.predict(X)
        lgb_preds /= max(1, len(self.lgb_models))

        rf_preds = np.zeros(len(X), dtype=np.float32)
        for m in self.rf_models:
            rf_preds += m.predict_proba(X)[:, 1]
        rf_preds /= max(1, len(self.rf_models))

        w_cb, w_lgb, w_rf = self.weights
        final_probs = w_cb * cb_preds + w_lgb * lgb_preds + w_rf * rf_preds
        return final_probs

    def predict_proba(self, kps_seq: np.ndarray, view_code: str = "view3") -> float:
        """단일 (64, 33, 3) 시퀀스 및 시점 코드로부터 결함 확률 반환"""
        feat = extract_d3_v3_features_from_seq(kps_seq, view_code).reshape(1, -1)
        return float(self.predict_proba_features(feat)[0])

    def predict_3state(self, prob: float, view_code: str = "view3") -> Tuple[int, str]:
        """시점별 사후 적응형 3-State 판정 (0: NORMAL, 1: DEFECT, 2: RETAKE)"""
        tau_norm, tau_def = self.view_thresholds.get(view_code, (0.420, 0.540))
        if prob <= tau_norm:
            return 0, "NORMAL"
        elif prob >= tau_def:
            return 1, "DEFECT"
        else:
            return 2, "RETAKE"

    def predict_single_3state(self, kps_seq: np.ndarray, view_code: str = "view3") -> Dict[str, Any]:
        prob = self.predict_proba(kps_seq, view_code)
        state_code, state_name = self.predict_3state(prob, view_code)
        tau_norm, tau_def = self.view_thresholds.get(view_code, (0.420, 0.540))
        return {
            "defect_code": self.DEFECT_CODE,
            "defect_name": self.DEFECT_NAME,
            "view_code": view_code,
            "probability": prob,
            "applied_thresholds": {"tau_normal": tau_norm, "tau_defect": tau_def},
            "state_code": state_code,
            "state_name": state_name,
            "is_clean": (state_code == 0),
            "is_defect": (state_code == 1),
            "is_retake": (state_code == 2)
        }

    def save_weights(self, out_path: str):
        data = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'rf_models': self.rf_models,
            'weights': self.weights,
            'decision_threshold': self.decision_threshold,
            'view_thresholds': self.view_thresholds,
            'feature_names': self.feature_names
        }
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)

    def load_weights(self, in_path: str):
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        self.cb_models = data['cb_models']
        self.lgb_models = data['lgb_models']
        self.rf_models = data['rf_models']
        self.weights = data['weights']
        self.decision_threshold = data.get('decision_threshold', 0.50)
        self.view_thresholds = data.get('view_thresholds', DEFAULT_VIEW_THRESHOLDS.copy())
        self.feature_names = data.get('feature_names', FEATURE_NAMES_33_D3)
        self.is_fitted = True
