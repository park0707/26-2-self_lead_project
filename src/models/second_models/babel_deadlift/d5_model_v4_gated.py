# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Model Version: v4.0 Gated (d5_model_v4_gated.py)
Status: 40D 시점 적응형 피처 게이팅(View-Adaptive Feature Gating) 엔진
        - 방식 1: 스쿼트 c2 모델의 '안면 게이트' 설계를 계승
        - 사각/측면 뷰(시점 B, D, A, E): 사용자가 발견한 '무릎-팔꿈치 상대 기하(SEK 각도, 체간 전방 돌출도)' 피처 100% 활성화
        - 정면 뷰(시점 C): 시상면 깊이 축소(Depth Flattening) 노이즈 피처(전방 돌출도, 가상 Z축 차이)를 0.0으로 완벽 차단(Zero-Masking)
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
from src.models.second_models.babel_deadlift.d5_model_v3 import (
    extract_d5_v3_features_from_seq, FEATURE_NAMES_32_D5, VIEW_CODE_MAP
)

FEATURE_NAMES_40_D5_GATED: List[str] = FEATURE_NAMES_32_D5 + [
    # Group 7: 시점 적응형 무릎-팔꿈치 상대 기하학 피처 (8D)
    'd5_gate_is_oblique',                # [32] 사각/측면 뷰 지시자 (시점 B, D, A, E = 1.0, 정면 C = 0.0)
    'd5_gated_ke_angle_sek_deg',         # [33] 최하사점 어깨-팔꿈치-무릎 사잇각 (SEK Angle, deg)
    'd5_gated_ke_elbow_knee_ant_diff',   # [34] 체간 기준 시상면 전방 돌출도 차이 (정면 C일 때 0.0 마스킹)
    'd5_gated_ke_width_ratio_nadir',     # [35] 최하사점 무릎/팔꿈치 너비비율 (X축 관상면 비율)
    'd5_gated_ke_dist_nadir',            # [36] 최하사점 무릎-팔꿈치 3D 유클리드 거리
    'd5_gated_ke_dist_pull_mean',        # [37] 퍼스트풀 12프레임 구간 무릎-팔꿈치 평균거리
    'd5_gated_ke_z_diff_nadir',          # [38] 최하사점 무릎-팔꿈치 Z축 상대거리 (정면 C일 때 0.0 마스킹)
    'd5_gated_ke_signed_x_nadir'         # [39] 최하사점 무릎-팔꿈치 부호화 수평(X) 오프셋
]


def extract_d5_v4_gated_features_from_seq(kps_arr: np.ndarray, view_name: str = "view3") -> np.ndarray:
    """
    32D 기하 단축 보정 피처 + 8D 시점 적응형 게이팅 무릎-팔꿈치 피처 = 총 40D 피처 추출
    """
    # 1. 기존 v3 32D 피처 추출
    f32 = extract_d5_v3_features_from_seq(kps_arr, view_name=view_name)

    pts = kps_arr[:, :, :3].astype(np.float32)
    T = pts.shape[0]

    shoulder_l = pts[:, MP_LEFT_SHOULDER]
    shoulder_r = pts[:, MP_RIGHT_SHOULDER]
    elbow_l = pts[:, MP_LEFT_ELBOW]
    elbow_r = pts[:, MP_RIGHT_ELBOW]
    wrist_l = pts[:, MP_LEFT_WRIST]
    wrist_r = pts[:, MP_RIGHT_WRIST]
    hip_l = pts[:, MP_LEFT_HIP]
    hip_r = pts[:, MP_RIGHT_HIP]
    knee_l = pts[:, MP_LEFT_KNEE]
    knee_r = pts[:, MP_RIGHT_KNEE]

    shoulder = (shoulder_l + shoulder_r) / 2.0
    elbow = (elbow_l + elbow_r) / 2.0
    bar = (wrist_l + wrist_r) / 2.0
    hip = (hip_l + hip_r) / 2.0
    knee = (knee_l + knee_r) / 2.0

    torso_len = np.linalg.norm(shoulder - hip, axis=-1).mean() + 1e-6
    femur_len = np.linalg.norm(hip - knee, axis=-1).mean() + 1e-6
    body_scale = torso_len + femur_len

    bar_y_smooth = gaussian_filter1d(bar[:, 1], sigma=2.0)
    nadir_idx = int(np.argmax(bar_y_smooth))

    pull_end = min(T - 1, nadir_idx + 12)
    if pull_end <= nadir_idx:
        pull_range = slice(max(0, nadir_idx - 5), min(T, nadir_idx + 6))
    else:
        pull_range = slice(nadir_idx, pull_end)

    k_nadir = knee[nadir_idx]
    e_nadir = elbow[nadir_idx]
    s_nadir = shoulder[nadir_idx]
    h_nadir = hip[nadir_idx]

    # [시점 판별 및 게이트 플래그]
    # view3(시점 C: 정면) 여부
    is_frontal = (view_name == "view3")
    is_oblique = 0.0 if is_frontal else 1.0

    # 1. 어깨-팔꿈치-무릎 사잇각 (SEK 각도, 정면/사각 모두 유효)
    v1 = s_nadir - e_nadir
    v2 = k_nadir - e_nadir
    cos_sek = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle_sek_deg = float(np.arccos(np.clip(cos_sek, -1.0, 1.0)) * 180.0 / np.pi)

    # 2. 체간 기준 시상면 전방 돌출도 차이 (방식 1 게이팅: 정면 C일 때 0.0 차단)
    if is_frontal:
        elbow_knee_anterior_diff = 0.0
    else:
        torso_vec = (h_nadir - s_nadir)[:2]
        torso_norm = np.linalg.norm(torso_vec) + 1e-6
        torso_unit = torso_vec / torso_norm
        n_ant = np.array([-torso_unit[1], torso_unit[0]], dtype=np.float32)
        knee_ant_proj = np.dot((k_nadir[:2] - h_nadir[:2]), n_ant)
        elbow_ant_proj = np.dot((e_nadir[:2] - s_nadir[:2]), n_ant)
        elbow_knee_anterior_diff = float((elbow_ant_proj - knee_ant_proj) / body_scale)

    # 3. 무릎/팔꿈치 너비비율
    knee_width_nadir = np.abs(knee_l[nadir_idx, 0] - knee_r[nadir_idx, 0])
    elbow_width_nadir = np.abs(elbow_l[nadir_idx, 0] - elbow_r[nadir_idx, 0])
    knee_elbow_width_ratio = float(knee_width_nadir / (elbow_width_nadir + 1e-6))

    # 4. 무릎-팔꿈치 거리
    ke_dist_nadir = float(np.linalg.norm(k_nadir - e_nadir) / body_scale)
    ke_dist_seq = np.linalg.norm(knee[pull_range] - elbow[pull_range], axis=-1) / body_scale
    ke_dist_pull_mean = float(np.mean(ke_dist_seq))

    # 5. Z축 상대거리 (방식 1 게이팅: 정면 C일 때 0.0 차단)
    if is_frontal:
        knee_elbow_z_diff = 0.0
    else:
        knee_elbow_z_diff = float((k_nadir[2] - e_nadir[2]) / body_scale)

    # 6. X축 상대 오프셋
    ke_signed_x = float((e_nadir[0] - k_nadir[0]) / body_scale)

    gated_8d = np.array([
        is_oblique,
        angle_sek_deg,
        elbow_knee_anterior_diff,
        knee_elbow_width_ratio,
        ke_dist_nadir,
        ke_dist_pull_mean,
        knee_elbow_z_diff,
        ke_signed_x
    ], dtype=np.float32)

    return np.concatenate([f32, gated_8d])


class BabelDeadliftD5ModelV4Gated:
    """
    40D 시점 적응형 게이팅 Tri-Ensemble 분류기
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        p_lgb = self.lgb_model.predict_proba(X)[:, 1] if self.lgb_model is not None else 0.5
        p_cb = self.cb_model.predict_proba(X)[:, 1] if self.cb_model is not None else 0.5
        p_rf = self.rf_model.predict_proba(X)[:, 1]

        w_lgb, w_cb, w_rf = self.weights
        w_sum = w_lgb + w_cb + w_rf
        p_ens = (p_lgb * w_lgb + p_cb * w_cb + p_rf * w_rf) / w_sum

        return np.column_stack([1.0 - p_ens, p_ens])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> 'BabelDeadliftD5ModelV4Gated':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
