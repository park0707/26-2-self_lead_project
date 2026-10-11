# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Model Version: v3.0 (d5_model_v3.py)
Status: 32D 기하학적 단축 보정(Fore-shortening Compensation) + 
        시점 적응형 Dynamic Gating(View-Adaptive Weighting) 독립 결함 분류 엔진
        - 전략 1: 정면 뷰(시점 C)의 깊이 축소 극복을 위해 팔의 2D 겉보기 단축률(Fore-shortening) 역삼각 측량 (4D 신규 피처)
        - 전략 2: 시점 카테고리 임베딩 및 시점 조건부 트리 분기를 통한 시점별 최적화
        - 3-Way Tri-Ensemble (LightGBM + CatBoost + Random Forest)
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

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

from src.models.second_models.babel_deadlift.d5_model_v1 import (
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_ELBOW, MP_RIGHT_ELBOW,
    MP_LEFT_WRIST, MP_RIGHT_WRIST, MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE, MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
    MP_LEFT_HEEL, MP_RIGHT_HEEL, MP_LEFT_FOOT_INDEX, MP_RIGHT_FOOT_INDEX,
    extract_d5_v1_features_from_seq, FEATURE_NAMES_28_D5
)

FEATURE_NAMES_32_D5: List[str] = FEATURE_NAMES_28_D5 + [
    # Group 6: 기하학적 단축 보정 및 정면 시상면 복원 (4D)
    'd5_arm_foreshortening_min_ratio',  # [28] 팔 2D 겉보기 길이 / 최대 길이 최소 단축비 (cos theta)
    'd5_virtual_sagittal_angle_max_deg',# [29] 단축비로부터 역산된 가상 시상면 전방 경사각 (deg)
    'd5_virtual_sagittal_angle_mean_deg',# [30] 리프팅 구간 가상 시상면 전방 경사각 평균 (deg)
    'd5_frontal_bar_knee_clearance_y'   # [31] 정면 뷰 바벨-무릎 수직 상대 변위 정규화 오차
]

VIEW_CODE_MAP = {
    'view5': 0,  # 시점 A
    'view2': 1,  # 시점 B
    'view3': 2,  # 시점 C (정면)
    'view4': 3,  # 시점 D
    'view1': 4   # 시점 E
}


def extract_d5_v3_features_from_seq(kps_arr: np.ndarray, view_name: str = "view3") -> np.ndarray:
    """
    기존 28D 피처에 기하학적 단축 보정 4D 피처를 결합하여 총 32D 피처 추출
    """
    # 1. 베이스라인 28D 피처 추출
    f28 = extract_d5_v1_features_from_seq(kps_arr, view_name=view_name)

    T = kps_arr.shape[0]
    pts = kps_arr[:, :, :3].astype(np.float32)

    shoulder_l = pts[:, MP_LEFT_SHOULDER]
    shoulder_r = pts[:, MP_RIGHT_SHOULDER]
    wrist_l = pts[:, MP_LEFT_WRIST]
    wrist_r = pts[:, MP_RIGHT_WRIST]
    knee_l = pts[:, MP_LEFT_KNEE]
    knee_r = pts[:, MP_RIGHT_KNEE]

    shoulder = (shoulder_l + shoulder_r) / 2.0
    bar = (wrist_l + wrist_r) / 2.0
    knee = (knee_l + knee_r) / 2.0

    # 2D 평면(X, Y) 투영 길이 계산 (정면 카메라 투영 기하학)
    arm_2d_len = np.sqrt((bar[:, 0] - shoulder[:, 0])**2 + (bar[:, 1] - shoulder[:, 1])**2)
    arm_3d_len = np.linalg.norm(bar - shoulder, axis=-1)

    # 팔이 수직으로 바닥에 늘어졌을 때의 최대 길이 추정 (상위 90% 백분위수)
    max_arm_len = float(np.percentile(arm_3d_len, 90) + 1e-6)

    # 팔 단축률: cos(theta) = L_apparent_2D / L_max
    cos_theta = np.clip(arm_2d_len / max_arm_len, 0.1, 1.0)
    virtual_angle_deg = np.arccos(cos_theta) * 180.0 / np.pi  # 가상 전방 경사각

    # Y 스무딩
    bar_y_smooth = gaussian_filter1d(bar[:, 1], sigma=2.0)
    nadir_idx = int(np.argmax(bar_y_smooth))
    apex_idx = int(np.argmin(bar_y_smooth))
    concentric_slice = slice(nadir_idx, apex_idx + 1) if nadir_idx < apex_idx else slice(nadir_idx, T)

    f28_min_ratio = float(np.min(cos_theta))
    f29_virtual_max_deg = float(np.max(virtual_angle_deg))
    f30_virtual_mean_deg = float(np.mean(virtual_angle_deg[concentric_slice]))

    # 정면 뷰 바벨-무릎 수직 상대 오차
    ref_len = float(np.mean(np.linalg.norm(knee - shoulder, axis=-1)) + 1e-6)
    f31_frontal_clearance = float(np.mean(np.abs(bar[:, 1] - knee[:, 1])) / ref_len)

    new_feats = np.array([
        f28_min_ratio,
        f29_virtual_max_deg,
        f30_virtual_mean_deg,
        f31_frontal_clearance
    ], dtype=np.float32)
    new_feats = np.nan_to_num(new_feats, nan=0.0, posinf=1.0, neginf=-1.0)

    f32 = np.concatenate([f28, new_feats], axis=0)
    return f32


class BabelDeadliftD5ModelV3:
    """
    32D 기하학적 단축 보정 + 시점 적응형 Dynamic Gating 3-Way Tri-Ensemble 분류기
    """
    def __init__(self,
                 lgb_params: Optional[Dict[str, Any]] = None,
                 cb_params: Optional[Dict[str, Any]] = None,
                 rf_params: Optional[Dict[str, Any]] = None,
                 weights: Tuple[float, float, float] = (0.50, 0.30, 0.20)):
        self.weights = weights
        self.is_fitted = False

        self.lgb_params = lgb_params or {
            'n_estimators': 180,
            'max_depth': 5,
            'learning_rate': 0.04,
            'num_leaves': 20,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'verbose': -1
        }
        self.cb_params = cb_params or {
            'iterations': 180,
            'depth': 5,
            'learning_rate': 0.04,
            'random_seed': 42,
            'verbose': 0
        }
        self.rf_params = rf_params or {
            'n_estimators': 180,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': -1
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        preds = []
        w_eff = []

        if self.lgb_model is not None:
            preds.append(self.lgb_model.predict_proba(X)[:, 1])
            w_eff.append(self.weights[0])
        if self.cb_model is not None:
            preds.append(self.cb_model.predict_proba(X)[:, 1])
            w_eff.append(self.weights[1])
        preds.append(self.rf_model.predict_proba(X)[:, 1])
        w_eff.append(self.weights[2])

        w_sum = sum(w_eff)
        norm_weights = [w / w_sum for w in w_eff]

        ensemble_prob = np.zeros(X.shape[0], dtype=np.float32)
        for p, w in zip(preds, norm_weights):
            ensemble_prob += p * w

        return np.column_stack([1.0 - ensemble_prob, ensemble_prob])

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)

    def predict_3state(self, X: np.ndarray, tau_norm: float = 0.405, tau_def: float = 0.525) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        preds = np.full(X.shape[0], -1, dtype=int)
        preds[prob < tau_norm] = 0
        preds[prob >= tau_def] = 1
        return preds

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "BabelDeadliftD5ModelV3":
        with open(file_path, 'rb') as f:
            return pickle.load(f)
