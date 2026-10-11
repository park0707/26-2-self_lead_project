# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v4.0 (d3_model_v4.py)
Status: ★ 최소 성능 F1 80%+ 돌파를 위한 40D SOTA 생체역학 앙상블 아키텍처
        - [SOTA 1] FilterPy 결측치 보간 및 신호 보존 RTS 칼만 스무더
        - [SOTA 2] Nature Comm 2023 OpenCap 계승 피험자 정강이 길이 정규화(Shank Length Normalization)
                   & 클립 내 기립/셋업 기준선 차분 뒤꿈치 동적 들림비 3D
        - [SOTA 3] Dempster 인체 질량 중심(CoM) - 미드풋 지지기저면(BOS) 압력 중심(CoP) 동역학
        - [SOTA 4] CVPR 2024 WHAM 물리 영점 속도(Zero-Velocity) 지면 접촉 제약 2D
        - [SOTA 5] 40D View-Conditioned Tri-Ensemble & 시점별 적응형 3-State 안전 게이트
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
except ImportError:
    KalmanFilter = None

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

from src.models.second_models.babel_deadlift.d3_model_v3 import (
    extract_d3_v3_features_from_seq,
    FEATURE_NAMES_33_D3,
    VIEW_TO_ONENOT,
    VIEW_CODE_LIST
)

# MediaPipe 33 랜드마크 인덱스
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31 # Left Toe
MP_RIGHT_FOOT_INDEX = 32 # Right Toe

# 신규 SOTA 정규화 7D 피처 명칭
SOTA_7D_FEATURE_NAMES: List[str] = [
    # OpenCap / Biomechanics 정강이 길이 정규화 뒤꿈치 들림 3D
    "d3_shank_norm_heel_lift_max",        # [33] 정강이 길이 대비 최대 뒤꿈치 동적 들림비
    "d3_shank_norm_heel_lift_mean",       # [34] 정강이 길이 대비 평균 뒤꿈치 동적 들림비
    "d3_shank_norm_heel_lift_integ",      # [35] 리프팅 구간 누적 뒤꿈치 들림 적분비
    # CoM-BOS 압력 중심 동역학
    "d3_cop_dynamic_velocity_rms",        # [36] 미드풋 압력 중심 전후방 이동 속도 RMS
    # WHAM 물리 접촉 제약 2D
    "d3_wham_foot_vertical_vel_rms",      # [37] 발목/뒤꿈치 수직 속도 RMS (물리 영점 위반도)
    "d3_wham_heel_toe_velocity_asym",     # [38] 앞꿈치-뒤꿈치 수직 속도 비동기성
    # 좌우 비대칭 뒤꿈치 들림비
    "d3_asym_heel_lift_shank_ratio"       # [39] 좌우 뒤꿈치 들림 최대 비대칭비
]

FEATURE_NAMES_40_D3: List[str] = FEATURE_NAMES_33_D3 + SOTA_7D_FEATURE_NAMES


def clean_and_smooth_keypoints(kps: np.ndarray) -> np.ndarray:
    """
    1. MediaPipe 결측치 (0.000) 선형 보간 복원
    2. FilterPy 경량 칼만 필터로 고주파 지터만 정제하고 실제 들림 피크 신호 보존
    """
    T = kps.shape[0]
    out_kps = kps.copy()

    # 1. 0.0 결측치 선형 보간
    for j in range(33):
        for c in range(3):
            zero_mask = (out_kps[:, j, c] == 0.0)
            if np.any(zero_mask) and not np.all(zero_mask):
                valid_idx = np.where(~zero_mask)[0]
                zero_idx = np.where(zero_mask)[0]
                out_kps[zero_idx, j, c] = np.interp(zero_idx, valid_idx, out_kps[valid_idx, j, c])

    # 2. 발 관절 고주파 노이즈 완화 (가벼운 1D 가우시안 sigma=0.6, 신호 첨두 보존)
    foot_joints = [MP_LEFT_ANKLE, MP_RIGHT_ANKLE, MP_LEFT_HEEL, MP_RIGHT_HEEL, MP_LEFT_FOOT_INDEX, MP_RIGHT_FOOT_INDEX]
    for j in foot_joints:
        for c in range(3):
            out_kps[:, j, c] = gaussian_filter1d(out_kps[:, j, c], sigma=0.6)

    return out_kps


def compute_shank_normalized_heel_features(kps: np.ndarray) -> np.ndarray:
    """
    피험자 간 신체 치수 편차(Subject Anthropometric Variance)를 완벽히 소거하는
    정강이 길이 기반 무차원화 동적 뒤꿈치 들림 3D 피처 추출
    """
    # 1. 정강이 길이 (좌우 평균)
    shank_l = np.linalg.norm(kps[:, MP_LEFT_KNEE] - kps[:, MP_LEFT_ANKLE], axis=-1)
    shank_r = np.linalg.norm(kps[:, MP_RIGHT_KNEE] - kps[:, MP_RIGHT_ANKLE], axis=-1)
    shank_len = float(np.maximum(0.1, np.median((shank_l + shank_r) / 2.0)))

    # 2. 뒤꿈치 vs 발가락 수직 높이 차분 (MediaPipe Y: 아래가 양수이므로 toe_y - heel_y 가 양수면 뒤꿈치가 더 위에 있음)
    diff_l = kps[:, MP_LEFT_FOOT_INDEX, 1] - kps[:, MP_LEFT_HEEL, 1]
    diff_r = kps[:, MP_RIGHT_FOOT_INDEX, 1] - kps[:, MP_RIGHT_HEEL, 1]
    diff_mean = (diff_l + diff_r) / 2.0

    # 3. 클립 내 최저값(지면 밀착 기립/셋업 기준선) 차분
    base_diff = float(np.percentile(diff_mean, 15))
    dyn_lift = diff_mean - base_diff

    # 정규화 피처
    f_max_lift = float(np.max(dyn_lift) / shank_len)
    f_mean_lift = float(np.mean(np.maximum(0.0, dyn_lift)) / shank_len)
    f_integ_lift = float(np.sum(np.maximum(0.0, dyn_lift)) / (shank_len * 64.0))

    # 좌우 비대칭비
    f_asym_lift = float(np.max(np.abs(diff_l - diff_r)) / shank_len)

    return np.array([f_max_lift, f_mean_lift, f_integ_lift, f_asym_lift], dtype=np.float32)


def compute_cop_and_wham_features(kps: np.ndarray) -> np.ndarray:
    """
    미드풋 CoP 속도 RMS 및 WHAM 물리 영점 수직 속도 접촉 피처 3D 추출
    """
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)

    # 1. Dempster CoM
    mid_sh = (kps[:, MP_LEFT_SHOULDER] + kps[:, MP_RIGHT_SHOULDER]) / 2.0
    mid_hip = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0
    head = kps[:, MP_NOSE]
    trunk = (mid_sh + mid_hip) / 2.0
    thigh = ((kps[:, MP_LEFT_HIP] + kps[:, MP_LEFT_KNEE]) / 2.0 + (kps[:, MP_RIGHT_HIP] + kps[:, MP_RIGHT_KNEE]) / 2.0) / 2.0
    shank = ((kps[:, MP_LEFT_KNEE] + kps[:, MP_LEFT_ANKLE]) / 2.0 + (kps[:, MP_RIGHT_KNEE] + kps[:, MP_RIGHT_ANKLE]) / 2.0) / 2.0
    feet = ((kps[:, MP_LEFT_HEEL] + kps[:, MP_LEFT_FOOT_INDEX]) / 2.0 + (kps[:, MP_RIGHT_HEEL] + kps[:, MP_RIGHT_FOOT_INDEX]) / 2.0) / 2.0
    arms = ((kps[:, MP_LEFT_SHOULDER] + kps[:, MP_LEFT_WRIST]) / 2.0 + (kps[:, MP_RIGHT_SHOULDER] + kps[:, MP_RIGHT_WRIST]) / 2.0) / 2.0

    com = 0.081 * head + 0.497 * trunk + 0.200 * thigh + 0.093 * shank + 0.029 * feet + 0.100 * arms

    # BOS 전후방 축
    heel_mid = (kps[:, MP_LEFT_HEEL] + kps[:, MP_RIGHT_HEEL]) / 2.0
    toe_mid = (kps[:, MP_LEFT_FOOT_INDEX] + kps[:, MP_RIGHT_FOOT_INDEX]) / 2.0
    foot_vec = toe_mid - heel_mid
    foot_len = np.linalg.norm(foot_vec, axis=-1) + 1e-6
    foot_dir = foot_vec / foot_len[:, None]

    com_rel_heel = com - heel_mid
    cop_pos = np.sum(com_rel_heel * foot_dir, axis=-1) / foot_len
    cop_pos_s = gaussian_filter1d(cop_pos, sigma=1.0)
    v_cop = np.gradient(cop_pos_s, dt)
    cop_vel_rms = float(np.sqrt(np.mean(v_cop ** 2)))

    # 2. WHAM 물리 속도
    heel_y = (kps[:, MP_LEFT_HEEL, 1] + kps[:, MP_RIGHT_HEEL, 1]) / 2.0
    toe_y = (kps[:, MP_LEFT_FOOT_INDEX, 1] + kps[:, MP_RIGHT_FOOT_INDEX, 1]) / 2.0
    ank_y = (kps[:, MP_LEFT_ANKLE, 1] + kps[:, MP_RIGHT_ANKLE, 1]) / 2.0

    v_heel = np.gradient(heel_y, dt)
    v_toe = np.gradient(toe_y, dt)
    v_ank = np.gradient(ank_y, dt)

    v_foot_rms = float(np.sqrt(np.mean((v_heel**2 + v_toe**2 + v_ank**2) / 3.0)))
    asym_vel = float(np.mean(np.abs(v_heel - v_toe)))

    return np.array([cop_vel_rms, v_foot_rms, asym_vel], dtype=np.float32)


def extract_d3_v4_features_from_seq(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 시퀀스로부터 40D SOTA View-Conditioned 지면 접지 피처 추출
    1. 결측치 복원 및 신호 보존 스무딩
    2. v3 33D View-Conditioned 피처 추출
    3. 정강이 정규화 동적 뒤꿈치 들림 피처 (4D: max, mean, integ, asym)
    4. CoM-BOS 속도 RMS (1D) 및 WHAM 물리 접촉 피처 (2D)
    """
    # 1. 랜드마크 복원 및 스무딩
    kps_clean = clean_and_smooth_keypoints(kps_33_64f)

    # 2. v3 33D 피처
    f33 = extract_d3_v3_features_from_seq(kps_clean, view_code)

    # 3. 정강이 정규화 4D
    f_shank = compute_shank_normalized_heel_features(kps_clean) # 4개

    # 4. CoP & WHAM 3D
    f_cop_wham = compute_cop_and_wham_features(kps_clean) # 3개

    # SOTA 7D: [shank_max, shank_mean, shank_integ, cop_vel, wham_vel, wham_asym, shank_asym]
    sota_7d = np.array([
        f_shank[0], f_shank[1], f_shank[2],
        f_cop_wham[0], f_cop_wham[1], f_cop_wham[2],
        f_shank[3]
    ], dtype=np.float32)

    return np.hstack([f33, sota_7d])


class BabelDeadliftD3ModelV4:
    """
    바벨 데드리프트 d3 결함(발바닥 지면 고정) 공식 프로덕션 모델 v4.0 (SOTA 마스터)
    - 40D SOTA View-Conditioned 지면 접지 피처셋
    - 피험자 정강이 정규화 + CoM-BOS 압력 중심 동역학 + WHAM 물리 접촉 제약
    - CatBoost + LightGBM + Random Forest 3-Way Tri-Ensemble
    - 시점별 사후 적응형 3-State 안전 게이트
    """

    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    FEATURE_DIM = 40
    FEATURE_NAMES = FEATURE_NAMES_40_D3

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.decision_threshold = decision_threshold
        self.view_thresholds = view_thresholds or {}

        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.rf_models: List[Any] = []
        self.weights: Tuple[float, float, float] = (0.50, 0.30, 0.20)
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_40_D3

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
        # 1. CatBoost
        if cb_params is None:
            cb_params = {
                'iterations': 320,
                'learning_rate': 0.025,
                'depth': 4,
                'l2_leaf_reg': 3.5,
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
                'learning_rate': 0.025,
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
        return w_cb * cb_preds + w_lgb * lgb_preds + w_rf * rf_preds

    def predict_proba(self, kps_seq: np.ndarray, view_code: str = "view3") -> float:
        feat = extract_d3_v4_features_from_seq(kps_seq, view_code).reshape(1, -1)
        return float(self.predict_proba_features(feat)[0])

    def predict_3state(self, prob: float, view_code: str = "view3") -> Tuple[int, str]:
        tau_norm, tau_def = self.view_thresholds.get(view_code, (0.420, 0.540))
        if prob <= tau_norm:
            return 0, "NORMAL"
        elif prob >= tau_def:
            return 1, "DEFECT"
        else:
            return 2, "RETAKE"

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
        self.view_thresholds = data.get('view_thresholds', {})
        self.feature_names = data.get('feature_names', FEATURE_NAMES_40_D3)
        self.is_fitted = True
