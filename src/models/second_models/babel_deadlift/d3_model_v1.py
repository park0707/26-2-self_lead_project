# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation / Heel-Toe Elevation)
Model Version: v1.0 (d3_model_v1.py)
Status: 스쿼트 c4 v5.1 생체역학 25D 지면 접지 피처셋 및 Tri-Ensemble 아키텍처 100% 완전 계승 모델
        - 1) 기립 구간(Stand Slice: 0 ~ T//8) 지면 기준선 중앙값 추정 (Median Baseline)
        - 2) 발바닥 롤각(Roll) 아크탄젠트(arctan2) 축 반전 특이점(Singularity) 클리핑 (±35도 생체역학 한계)
        - 3) 25D 생체역학 지면 접지 피처셋 (22D 물리 접지 + 3D 가시성/전이 빈도)
        - 4) CatBoost + LightGBM + Random Forest 3-Way Tri-Ensemble 하이브리드 결합
        - 5) 3-State 선택적 판정(Selective Classification) 메서드 탑재
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

# MediaPipe 33 랜드마크 인덱스
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

FEATURE_NAMES_25_D3: List[str] = [
    # [1~22] 22D 핵심 지면 접지 물리 피처
    'd3_foot_pitch_max_deg',               # [0] 전 구간 최대 발바닥 경사각 (deg)
    'd3_toe_lift_max_norm',                # [1] 스케일 정규화 앞꿈치 최대 들림량
    'd3_foot_pitch_nadir_deg',             # [2] 최하사점 발바닥 경사각 (deg)
    'd3_delta_wobble_active',              # [3] 상승기 뒤꿈치-앞꿈치 수직차 흔들림 (std)
    'd3_roll_max_active',                  # [4] 능동 구간 최대 발바닥 롤각 (deg)
    'd3_delta_max_active',                 # [5] 능동 구간 최대 수직 높이차
    'd3_heel_toe_ydelta_nadir_cm',         # [6] 최하사점 뒤꿈치-앞꿈치 수직차 (cm)
    'd3_coupling_ratio_shank_pitch',       # [7] 정강이 경사 대비 발목 가동성 커플링비
    'd3_bilateral_foot_pitch_asym_deg',    # [8] 좌우 발바닥 경사각 비대칭도 (deg)
    'd3_pitch_max_active_deg',             # [9] 능동 구간 최대 피치각 (deg)
    'd3_shank_ankle_coupling_ratio_legacy',# [10] 정강이-발목 시계열 상관계수
    'd3_foot_pitch_drop_deg',              # [11] 기립 대비 최하사점 경사각 변화폭 (deg)
    'd3_foot_roll_inversion_deg',          # [12] 최하사점 발목 롤 / 안쪽날 들림 (deg)
    'd3_asym_pitch_active',                # [13] 능동 구간 최대 피치 비대칭도
    'd3_roll_nadir_deg',                   # [14] 최하사점 발바닥 롤각 (deg)
    'd3_delta_jerk_active',                # [15] 상승기 지면 가속도 저크
    'd3_ank_min_angle',                    # [16] 능동 구간 최소 발목 관절각 (deg)
    'd3_ascent_ankle_destabilization',     # [17] 상승기 발목 불안정성 (cm)
    'd3_toe_lift_max_cm',                  # [18] 기립 기준 앞꿈치 최대 들림량 (cm)
    'd3_asym_delta_active',                # [19] 능동 구간 좌우 높이차 비대칭도
    'd3_norm_delta_nadir',                 # [20] 발 길이 정규화 최하사점 수직차
    'd3_pitch_drop_active_deg',            # [21] 능동 구간 피치 드롭각 (deg)
    # [23~25] F0 3대 가시성 및 동역학 피처 (3D)
    'd3_f0_heel_view_visibility',          # [22] 시점별 뒤꿈치 가시성 계수 (정면 가림 vs 측면 노출)
    'd3_f0_shank_lift_corr',               # [23] 최하사점 정강이 경사-뒤꿈치 들림 연동 역학 상관계수
    'd3_f0_ascent_transition_rate'         # [24] 상승기 접촉 상태 머신 전이 빈도 (발 흔들림 정량화)
]

TOP_22_INDICES_FROM_29D = [2, 17, 0, 23, 19, 11, 3, 22, 4, 14, 7, 1, 6, 15, 18, 24, 20, 8, 5, 16, 27, 13]


def compute_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """세 3D 점 사이의 사잇각(도, deg) 계산"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_ang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def extract_d3_elite_9d_features(kps: np.ndarray) -> np.ndarray:
    """9D 핵심 지면 접지 피처 추출 (스쿼트 c4 v5.1 계승)"""
    hip_l, hip_r = kps[:, MP_LEFT_HIP], kps[:, MP_RIGHT_HIP]
    ank_l, ank_r = kps[:, MP_LEFT_ANKLE], kps[:, MP_RIGHT_ANKLE]
    heel_l, heel_r = kps[:, MP_LEFT_HEEL], kps[:, MP_RIGHT_HEEL]
    toe_l, toe_r = kps[:, MP_LEFT_FOOT_INDEX], kps[:, MP_RIGHT_FOOT_INDEX]

    hip_y = (hip_l[:, 1] + hip_r[:, 1]) / 2.0
    hip_y_s = gaussian_filter1d(hip_y, sigma=1.2)
    b = int(np.clip(np.argmax(hip_y_s), 4, len(kps) - 5))

    vec_foot_l = toe_l - heel_l
    vec_foot_r = toe_r - heel_r
    pitch_l = np.degrees(np.arctan2(vec_foot_l[:, 1], np.hypot(vec_foot_l[:, 0], vec_foot_l[:, 2]) + 1e-7))
    pitch_r = np.degrees(np.arctan2(vec_foot_r[:, 1], np.hypot(vec_foot_r[:, 0], vec_foot_r[:, 2]) + 1e-7))
    pitch_mean = (pitch_l + pitch_r) / 2.0

    delta_l = toe_l[:, 1] - heel_l[:, 1]
    delta_r = toe_r[:, 1] - heel_r[:, 1]
    delta_mean = (delta_l + delta_r) / 2.0

    # 지면 기준선 중앙값 추정 (Median Baseline)
    stand_slice = slice(0, max(2, len(kps) // 8))
    p_stand = float(np.median(pitch_mean[stand_slice]))
    delta_stand = float(np.median(delta_mean[stand_slice]))

    f1 = float(pitch_mean[b])
    f2 = float(pitch_mean[b] - p_stand)
    f3 = float(np.max(pitch_mean))
    f4 = float(delta_mean[b])
    f5 = float(np.max(np.abs(pitch_l - pitch_r)))
    f6 = float(max(np.max(np.maximum(0, heel_l[:, 1] - toe_l[:, 1])), np.max(np.maximum(0, heel_r[:, 1] - toe_r[:, 1]))))

    # 롤 특이점 보정: 생체역학 한계 (±35도)
    roll_l = np.clip(np.degrees(np.arctan2(vec_foot_l[:, 0], vec_foot_l[:, 1] + 1e-7)), -35.0, 35.0)
    roll_r = np.clip(np.degrees(np.arctan2(vec_foot_r[:, 0], vec_foot_r[:, 1] + 1e-7)), -35.0, 35.0)
    f7 = float((abs(roll_l[b]) + abs(roll_r[b])) / 2.0)

    knee_l, knee_r = kps[:, MP_LEFT_KNEE], kps[:, MP_RIGHT_KNEE]
    ang_ank_l = compute_angle_3d(knee_l, ank_l, toe_l)
    ang_ank_r = compute_angle_3d(knee_r, ank_r, toe_r)
    ang_ank = (ang_ank_l + ang_ank_r) / 2.0
    vec_shank = (ank_l + ank_r)/2.0 - (knee_l + knee_r)/2.0
    shank_pitch = np.degrees(np.arctan2(vec_shank[:, 2], vec_shank[:, 1] + 1e-7))
    f8 = float((np.max(ang_ank) - np.min(ang_ank)) / (np.max(shank_pitch) - np.min(shank_pitch) + 1e-6))
    f9 = float(np.std(delta_mean[b:]))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=np.float32)


def _extract_d3_20d_features(kps: np.ndarray) -> np.ndarray:
    """20D 확장 물리 피처 추출 (스쿼트 c4 v5.1 계승)"""
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)

    hip_l, hip_r = kps[:, MP_LEFT_HIP], kps[:, MP_RIGHT_HIP]
    knee_l, knee_r = kps[:, MP_LEFT_KNEE], kps[:, MP_RIGHT_KNEE]
    ank_l, ank_r = kps[:, MP_LEFT_ANKLE], kps[:, MP_RIGHT_ANKLE]
    heel_l, heel_r = kps[:, MP_LEFT_HEEL], kps[:, MP_RIGHT_HEEL]
    toe_l, toe_r = kps[:, MP_LEFT_FOOT_INDEX], kps[:, MP_RIGHT_FOOT_INDEX]

    hip_y = (hip_l[:, 1] + hip_r[:, 1]) / 2.0
    hip_y_s = gaussian_filter1d(hip_y, sigma=1.2)
    b = int(np.clip(np.argmax(hip_y_s), 4, T - 5))

    y_min, y_max = np.min(hip_y_s), np.max(hip_y_s)
    rom = y_max - y_min
    if rom > 0.0004:
        active_mask = (hip_y_s - y_min) >= (rom * 0.15)
    else:
        active_mask = np.ones(T, dtype=bool)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    stand_slice = slice(0, max(2, T // 8))

    delta_l = toe_l[:, 1] - heel_l[:, 1]
    delta_r = toe_r[:, 1] - heel_r[:, 1]
    delta_mean = (delta_l + delta_r) / 2.0
    delta_s = gaussian_filter1d(delta_mean, sigma=1.0)

    delta_stand = float(np.median(delta_s[stand_slice]))
    delta_nadir = float(delta_s[b])
    delta_drop = float(delta_nadir - delta_stand)
    delta_max_active = float(np.max(delta_s[active_mask]))

    vec_foot_l = toe_l - heel_l
    vec_foot_r = toe_r - heel_r
    pitch_l = np.degrees(np.arctan2(vec_foot_l[:, 1], np.hypot(vec_foot_l[:, 0], vec_foot_l[:, 2]) + 1e-7))
    pitch_r = np.degrees(np.arctan2(vec_foot_r[:, 1], np.hypot(vec_foot_r[:, 0], vec_foot_r[:, 2]) + 1e-7))
    pitch_mean = (pitch_l + pitch_r) / 2.0
    pitch_s = gaussian_filter1d(pitch_mean, sigma=1.0)

    pitch_stand = float(np.median(pitch_s[stand_slice]))
    pitch_nadir = float(pitch_s[b])
    pitch_drop = float(pitch_nadir - pitch_stand)
    pitch_max = float(np.max(pitch_s[active_mask]))

    asym_pitch = float(np.max(np.abs(pitch_l - pitch_r)[active_mask]))
    asym_delta = float(np.max(np.abs(delta_l - delta_r)[active_mask]))

    toe_lift_l = np.maximum(0.0, heel_l[:, 1] - toe_l[:, 1])
    toe_lift_r = np.maximum(0.0, heel_r[:, 1] - toe_r[:, 1])
    toe_lift_max = float(max(np.max(toe_lift_l[active_mask]), np.max(toe_lift_r[active_mask])))

    roll_l = np.clip(np.degrees(np.arctan2(vec_foot_l[:, 0], vec_foot_l[:, 1] + 1e-7)), -35.0, 35.0)
    roll_r = np.clip(np.degrees(np.arctan2(vec_foot_r[:, 0], vec_foot_r[:, 1] + 1e-7)), -35.0, 35.0)
    roll_nadir = float((abs(roll_l[b]) + abs(roll_r[b])) / 2.0)
    roll_max = float(max(np.max(np.abs(roll_l)[active_mask]), np.max(np.abs(roll_r)[active_mask])))

    ang_ank_l = compute_angle_3d(knee_l, ank_l, toe_l)
    ang_ank_r = compute_angle_3d(knee_r, ank_r, toe_r)
    ang_ank_mean = (ang_ank_l + ang_ank_r) / 2.0
    ang_ank_s = gaussian_filter1d(ang_ank_mean, sigma=1.0)
    ank_min = float(np.min(ang_ank_s[active_mask]))
    ank_stand = float(np.median(ang_ank_s[stand_slice]))
    ank_drop = float(ank_stand - ank_min)

    vec_shank_l = ank_l - knee_l
    vec_shank_r = ank_r - knee_r
    shank_pitch_l = np.degrees(np.arctan2(vec_shank_l[:, 2], vec_shank_l[:, 1] + 1e-7))
    shank_pitch_r = np.degrees(np.arctan2(vec_shank_r[:, 2], vec_shank_r[:, 1] + 1e-7))
    shank_pitch = (shank_pitch_l + shank_pitch_r) / 2.0
    shank_diff = float(np.max(shank_pitch[active_mask]) - np.min(shank_pitch[active_mask]) + 1e-6)
    ank_diff = float(np.max(ang_ank_s[active_mask]) - np.min(ang_ank_s[active_mask]))
    coupling_ratio = float(ank_diff / shank_diff)

    ascent_mask = np.zeros(T, dtype=bool)
    ascent_mask[b:] = active_mask[b:]
    if np.sum(ascent_mask) < 3:
        ascent_mask = active_mask
    delta_wobble = float(np.std(delta_s[ascent_mask]))

    v_delta = np.gradient(delta_s, dt)
    a_delta = np.gradient(v_delta, dt)
    j_delta = np.gradient(a_delta, dt)
    delta_jerk = float(np.mean(np.abs(j_delta[ascent_mask])))

    lh_stand = float(np.median(heel_l[stand_slice, 1]))
    rh_stand = float(np.median(heel_r[stand_slice, 1]))
    lh_lift = np.maximum(0.0, lh_stand - heel_l[:, 1])
    rh_lift = np.maximum(0.0, rh_stand - heel_r[:, 1])
    heel_lift_max = float(max(np.max(lh_lift[active_mask]), np.max(rh_lift[active_mask])))
    heel_lift_nadir = float((lh_lift[b] + rh_lift[b]) / 2.0)

    foot_len_l = np.linalg.norm(vec_foot_l, axis=-1)
    foot_len_r = np.linalg.norm(vec_foot_r, axis=-1)
    norm_delta_nadir = float(delta_nadir / (np.mean(foot_len_l) + 1e-7))
    norm_delta_drop = float(delta_drop / (np.mean(foot_len_l) + 1e-7))

    return np.array([
        delta_nadir, delta_drop, delta_max_active,
        pitch_nadir, pitch_drop, pitch_max,
        asym_pitch, asym_delta,
        toe_lift_max,
        roll_nadir, roll_max,
        ank_min, ank_drop,
        coupling_ratio,
        delta_wobble, delta_jerk,
        heel_lift_max, heel_lift_nadir,
        norm_delta_nadir, norm_delta_drop
    ], dtype=np.float32)


def extract_d3_v1_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    (64, 33, 3) 시퀀스로부터 d3 v1.0 생체역학 지면 접지 25D 피처 추출
    """
    kps = np.asarray(kps_33_64f, dtype=np.float32)
    T = kps.shape[0]

    # 1. 22D 베이스 피처 추출 (중앙값 기준선 및 롤 클리핑)
    f9 = extract_d3_elite_9d_features(kps)
    f20 = _extract_d3_20d_features(kps)
    full_29d = np.hstack([f9, f20])
    v4_22d = full_29d[TOP_22_INDICES_FROM_29D]

    # 2. F0 3대 핵심 피처 추출 (시점 가시성, 정강이 연동, 접촉 상태 전이 빈도)
    hip_l, hip_r = kps[:, MP_LEFT_HIP], kps[:, MP_RIGHT_HIP]
    knee_l, knee_r = kps[:, MP_LEFT_KNEE], kps[:, MP_RIGHT_KNEE]
    ank_l, ank_r = kps[:, MP_LEFT_ANKLE], kps[:, MP_RIGHT_ANKLE]
    heel_l, heel_r = kps[:, MP_LEFT_HEEL], kps[:, MP_RIGHT_HEEL]
    toe_l, toe_r = kps[:, MP_LEFT_FOOT_INDEX], kps[:, MP_RIGHT_FOOT_INDEX]

    hip_y = (hip_l[:, 1] + hip_r[:, 1]) / 2.0
    hip_y_s = gaussian_filter1d(hip_y, sigma=1.2)
    b = int(np.clip(np.argmax(hip_y_s), 4, T - 5))

    stand_slice = slice(0, max(2, T // 8))
    vec_foot_l = toe_l - heel_l
    vec_foot_r = toe_r - heel_r
    foot_len = (np.mean(np.linalg.norm(vec_foot_l, axis=-1)) + np.mean(np.linalg.norm(vec_foot_r, axis=-1))) / 2.0 + 1e-6

    # (F0-1) 시점별 뒤꿈치 가시성 점수 (heel_view_visibility)
    v_foot_x = (vec_foot_l[:, 0] + vec_foot_r[:, 0]) / 2.0
    v_foot_z = (vec_foot_l[:, 2] + vec_foot_r[:, 2]) / 2.0
    foot_yaw = np.degrees(np.arctan2(v_foot_x, v_foot_z + 1e-7))
    mean_foot_yaw = float(np.mean(foot_yaw))
    f22_heel_view_visibility = float(np.sin(np.radians(abs(mean_foot_yaw))))

    # (F0-2) 정강이-뒤꿈치 들림 연동 상관계수 (shank_lift_corr)
    delta_l = (toe_l[:, 1] - heel_l[:, 1]) / foot_len
    delta_r = (toe_r[:, 1] - heel_r[:, 1]) / foot_len
    delta_s = gaussian_filter1d((delta_l + delta_r) / 2.0, sigma=0.8)
    norm_lift = delta_s - np.median(delta_s[stand_slice])

    vec_shank = (ank_l + ank_r)/2.0 - (knee_l + knee_r)/2.0
    shank_tilt = np.degrees(np.arctan2(vec_shank[:, 2], -vec_shank[:, 1] + 1e-7))
    shank_tilt_s = gaussian_filter1d(shank_tilt, sigma=1.0)

    w_start, w_end = max(0, b - 3), min(T, b + 4)
    win_shank = shank_tilt_s[w_start:w_end]
    win_lift = norm_lift[w_start:w_end]
    if np.std(win_shank) > 1e-5 and np.std(win_lift) > 1e-5:
        f23_shank_lift_corr = float(np.corrcoef(win_shank, win_lift)[0, 1])
    else:
        f23_shank_lift_corr = 0.0

    # (F0-3) 상승기 접촉 상태 전이 빈도 (ascent_transition_rate)
    rom = np.max(hip_y_s) - np.min(hip_y_s)
    active_mask = (hip_y_s - np.min(hip_y_s)) >= (rom * 0.15) if rom > 0.0004 else np.ones(T, dtype=bool)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    is_heel_lift = (norm_lift > 0.03) & active_mask
    is_toe_lift = (norm_lift < -0.03) & active_mask
    states = np.zeros(T, dtype=int)
    states[is_heel_lift] = 1
    states[is_toe_lift] = 2

    ascent_mask = (np.arange(T) >= b) & active_mask
    if np.sum(ascent_mask) > 1:
        ascent_states = states[ascent_mask]
        f24_ascent_transition_rate = float(np.sum(ascent_states[1:] != ascent_states[:-1]) / len(ascent_states))
    else:
        f24_ascent_transition_rate = 0.0

    f0_3d = np.array([f22_heel_view_visibility, f23_shank_lift_corr, f24_ascent_transition_rate], dtype=np.float32)
    return np.hstack([v4_22d, f0_3d])


def extract_d3_v1_features(X_kps: np.ndarray) -> np.ndarray:
    """배치 (N, 64, 33, 3) 또는 단일 시퀀스로부터 25D 피처 추출"""
    if X_kps.ndim == 3:
        return extract_d3_v1_features_from_seq(X_kps)
    N = X_kps.shape[0]
    out = np.zeros((N, 25), dtype=np.float32)
    for i in range(N):
        out[i] = extract_d3_v1_features_from_seq(X_kps[i])
    return out


class BabelDeadliftD3ModelV1:
    """
    바벨 데드리프트 d3 결함(발바닥 지면 고정) 공식 프로덕션 모델 v1.0
    - 스쿼트 c4 v5.1 25D 물리 접지 아키텍처 100% 완전 계승
    - CatBoost(50%) + LightGBM(30%) + Random Forest(20%) 3-Way Tri-Ensemble
    - 3-State 선택적 판정(Selective Classification) 메서드 탑재
    """

    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    FEATURE_DIM = 25
    FEATURE_NAMES = FEATURE_NAMES_25_D3

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_normal: float = 0.420,
        tau_defect: float = 0.540
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect

        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.rf_models: List[Any] = []
        self.weights: Tuple[float, float, float] = (0.50, 0.30, 0.20)
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_25_D3

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
        """단일 Fold 또는 전체 데이터셋 기반 Tri-Ensemble 학습"""
        # 1. CatBoost
        if cb_params is None:
            cb_params = {
                'iterations': 300,
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
        """25D 피처 행렬로부터 결함 확률 추론"""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        # CatBoost 예측 평균
        cb_preds = np.zeros(len(X), dtype=np.float32)
        for m in self.cb_models:
            cb_preds += m.predict_proba(X)[:, 1]
        cb_preds /= max(1, len(self.cb_models))

        # LightGBM 예측 평균
        lgb_preds = np.zeros(len(X), dtype=np.float32)
        for m in self.lgb_models:
            lgb_preds += m.predict(X)
        lgb_preds /= max(1, len(self.lgb_models))

        # Random Forest 예측 평균
        rf_preds = np.zeros(len(X), dtype=np.float32)
        for m in self.rf_models:
            rf_preds += m.predict_proba(X)[:, 1]
        rf_preds /= max(1, len(self.rf_models))

        w_cb, w_lgb, w_rf = self.weights
        final_probs = w_cb * cb_preds + w_lgb * lgb_preds + w_rf * rf_preds
        return final_probs

    def predict_proba(self, kps_seq: np.ndarray) -> float:
        """단일 (64, 33, 3) 시퀀스로부터 결함 확률 반환"""
        feat = extract_d3_v1_features_from_seq(kps_seq).reshape(1, -1)
        return float(self.predict_proba_features(feat)[0])

    def predict_3state(self, prob: float) -> Tuple[int, str]:
        """
        3-State 안전 게이트 판정
        반환: (상태코드, 상태명)
        - 0: NORMAL (정상 확정)
        - 1: DEFECT (결함 확정)
        - 2: RETAKE (안전 보류 / 재촬영 유도)
        """
        if prob <= self.tau_normal:
            return 0, "NORMAL"
        elif prob >= self.tau_defect:
            return 1, "DEFECT"
        else:
            return 2, "RETAKE"

    def predict_single_3state(self, kps_seq: np.ndarray) -> Dict[str, Any]:
        """단일 시퀀스 종합 진단 결과 반환"""
        prob = self.predict_proba(kps_seq)
        state_code, state_name = self.predict_3state(prob)
        return {
            "defect_code": self.DEFECT_CODE,
            "defect_name": self.DEFECT_NAME,
            "probability": prob,
            "state_code": state_code,
            "state_name": state_name,
            "is_clean": (state_code == 0),
            "is_defect": (state_code == 1),
            "is_retake": (state_code == 2)
        }

    def save_weights(self, out_path: str):
        """가중치 및 메타데이터 덤프"""
        data = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'rf_models': self.rf_models,
            'weights': self.weights,
            'decision_threshold': self.decision_threshold,
            'tau_normal': self.tau_normal,
            'tau_defect': self.tau_defect,
            'feature_names': self.feature_names
        }
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)

    def load_weights(self, in_path: str):
        """가중치 및 메타데이터 로드"""
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        self.cb_models = data['cb_models']
        self.lgb_models = data['lgb_models']
        self.rf_models = data['rf_models']
        self.weights = data['weights']
        self.decision_threshold = data.get('decision_threshold', 0.50)
        self.tau_normal = data.get('tau_normal', 0.35)
        self.tau_defect = data.get('tau_defect', 0.65)
        self.feature_names = data.get('feature_names', FEATURE_NAMES_25_D3)
        self.is_fitted = True
