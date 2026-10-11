# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c4. 발바닥 지면 고정 (Foot Ground Fixation / Heel-Toe Elevation)
Model Version: v5.0 (c4_model_v5.py)
Status: ★ 순수 비전 100% MediaPipe 33관절 64프레임 기반 공식 프로덕션 모델 v5.0
        - 외부 AI F0(Contact State & View Observability) 제안 100% 반영
        - [v4 계승] 22D 정예 지면 접지 피처셋 100% 온전 계승
        - [F0 3대 핵심 피처 결합 (총 25D)]:
          1) c4_f0_heel_view_visibility: 발-카메라 Z축 기하학에 따른 시점별 뒤꿈치 가시성 계수
          2) c4_f0_shank_lift_corr: 최하사점 정강이 전방 경사각-뒤꿈치 들림 연동 역학 상관계수
          3) c4_f0_ascent_transition_rate: 상승기 접촉 상태 머신(Flat-Lift) 전이 빈도 (발 흔들림 정량화)
        - [최적 앙상블]: CatBoost(60%) + LightGBM(30%) + Random Forest(10%)
        - [실측 성적]: 2,605개 마스터 데이터셋 5-Fold 피험자 완전 격리
          - 이진 판정: F1 79.46%, Acc 79.42%, ROC-AUC 87.73% (정면 view3 F1 85.05%)
          - 선택적 판정([0.40, 0.60]): Coverage 85.6%, Selective F1 82.81%, Selective Acc 82.78%
          - 외부 AI의 "단안 키포인트 4~8px SNR 1:1 한계" 가설 실측 입증 및 3-State 안정화 달성
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

FEATURE_NAMES_25_C4 = [
    # [1~22] v4 계승 정예 지면 접지 피처
    'c4_foot_pitch_max_deg',              # [0] 전 구간 최대 발바닥 경사각 (deg)
    'c4_toe_lift_max_norm',               # [1] 스케일 정규화 앞꿈치 최대 들림량
    'c4_foot_pitch_nadir_deg',            # [2] 최하사점 발바닥 경사각 (deg)
    'c4_delta_wobble_active',             # [3] 상승기 뒤꿈치-앞꿈치 수직차 흔들림 (std)
    'c4_roll_max_active',                 # [4] 능동 구간 최대 발바닥 롤각 (deg)
    'c4_delta_max_active',                # [5] 능동 구간 최대 수직 높이차
    'c4_heel_toe_ydelta_nadir_cm',        # [6] 최하사점 뒤꿈치-앞꿈치 수직차 (cm)
    'c4_coupling_ratio_shank_pitch',      # [7] 정강이 경사 대비 발목 가동성 커플링비
    'c4_bilateral_foot_pitch_asym_deg',   # [8] 좌우 발바닥 경사각 비대칭도 (deg)
    'c4_pitch_max_active_deg',            # [9] 능동 구간 최대 피치각 (deg)
    'c4_shank_ankle_coupling_ratio_legacy',# [10] 정강이-발목 시계열 상관계수
    'c4_foot_pitch_drop_deg',             # [11] 기립 대비 최하사점 경사각 변화폭 (deg)
    'c4_foot_roll_inversion_deg',         # [12] 최하사점 발목 롤 / 안쪽날 들림 (deg)
    'c4_asym_pitch_active',               # [13] 능동 구간 최대 피치 비대칭도
    'c4_roll_nadir_deg',                  # [14] 최하사점 발바닥 롤각 (deg)
    'c4_delta_jerk_active',               # [15] 상승기 지면 가속도 저크
    'c4_ank_min_angle',                   # [16] 능동 구간 최소 발목 관절각 (deg)
    'c4_ascent_ankle_destabilization',    # [17] 상승기 발목 불안정성 (cm)
    'c4_toe_lift_max_cm',                 # [18] 기립 기준 앞꿈치 최대 들림량 (cm)
    'c4_asym_delta_active',               # [19] 능동 구간 좌우 높이차 비대칭도
    'c4_norm_delta_nadir',                # [20] 발 길이 정규화 최하사점 수직차
    'c4_pitch_drop_active_deg',           # [21] 능동 구간 피치 드롭각 (deg)
    # [23~25] F0 신규 핵심 피처 (3D)
    'c4_f0_heel_view_visibility',         # [22] 시점별 뒤꿈치 가시성 계수 (정면 가림 vs 측면 노출)
    'c4_f0_shank_lift_corr',              # [23] 최하사점 정강이 경사-뒤꿈치 들림 연동 역학 상관계수
    'c4_f0_ascent_transition_rate'        # [24] 상승기 접촉 상태 머신 전이 빈도 (발 흔들림 정량화)
]

TOP_22_INDICES_FROM_29D = [2, 17, 0, 23, 19, 11, 3, 22, 4, 14, 7, 1, 6, 15, 18, 24, 20, 8, 5, 16, 27, 13]


def compute_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    v1 = p1 - p2
    v2 = p3 - p2
    cos_ang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def extract_c4_elite_9d_features(kps: np.ndarray) -> np.ndarray:
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

    stand_slice = slice(0, max(2, len(kps) // 8))
    p_stand = np.mean(pitch_mean[stand_slice])
    d_stand = np.mean(delta_mean[stand_slice])

    f1 = float(pitch_mean[b])
    f2 = float(pitch_mean[b] - p_stand)
    f3 = float(np.max(pitch_mean))
    f4 = float(delta_mean[b])
    f5 = float(np.max(np.abs(pitch_l - pitch_r)))
    f6 = float(max(np.max(np.maximum(0, heel_l[:, 1] - toe_l[:, 1])), np.max(np.maximum(0, heel_r[:, 1] - toe_r[:, 1]))))
    
    roll_l = np.degrees(np.arctan2(vec_foot_l[:, 0], vec_foot_l[:, 1] + 1e-7))
    roll_r = np.degrees(np.arctan2(vec_foot_r[:, 0], vec_foot_r[:, 1] + 1e-7))
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


def _extract_c4_20d_features(kps: np.ndarray) -> np.ndarray:
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

    delta_stand = float(np.mean(delta_s[stand_slice]))
    delta_nadir = float(delta_s[b])
    delta_drop = float(delta_nadir - delta_stand)
    delta_max_active = float(np.max(delta_s[active_mask]))

    vec_foot_l = toe_l - heel_l
    vec_foot_r = toe_r - heel_r
    pitch_l = np.degrees(np.arctan2(vec_foot_l[:, 1], np.hypot(vec_foot_l[:, 0], vec_foot_l[:, 2]) + 1e-7))
    pitch_r = np.degrees(np.arctan2(vec_foot_r[:, 1], np.hypot(vec_foot_r[:, 0], vec_foot_r[:, 2]) + 1e-7))
    pitch_mean = (pitch_l + pitch_r) / 2.0
    pitch_s = gaussian_filter1d(pitch_mean, sigma=1.0)

    pitch_stand = float(np.mean(pitch_s[stand_slice]))
    pitch_nadir = float(pitch_s[b])
    pitch_drop = float(pitch_nadir - pitch_stand)
    pitch_max = float(np.max(pitch_s[active_mask]))

    asym_pitch = float(np.max(np.abs(pitch_l - pitch_r)[active_mask]))
    asym_delta = float(np.max(np.abs(delta_l - delta_r)[active_mask]))

    toe_lift_l = np.maximum(0.0, heel_l[:, 1] - toe_l[:, 1])
    toe_lift_r = np.maximum(0.0, heel_r[:, 1] - toe_r[:, 1])
    toe_lift_max = float(max(np.max(toe_lift_l[active_mask]), np.max(toe_lift_r[active_mask])))

    roll_l = np.degrees(np.arctan2(vec_foot_l[:, 0], vec_foot_l[:, 1] + 1e-7))
    roll_r = np.degrees(np.arctan2(vec_foot_r[:, 0], vec_foot_r[:, 1] + 1e-7))
    roll_nadir = float((abs(roll_l[b]) + abs(roll_r[b])) / 2.0)
    roll_max = float(max(np.max(np.abs(roll_l)[active_mask]), np.max(np.abs(roll_r)[active_mask])))

    ang_ank_l = compute_angle_3d(knee_l, ank_l, toe_l)
    ang_ank_r = compute_angle_3d(knee_r, ank_r, toe_r)
    ang_ank_mean = (ang_ank_l + ang_ank_r) / 2.0
    ang_ank_s = gaussian_filter1d(ang_ank_mean, sigma=1.0)
    ank_min = float(np.min(ang_ank_s[active_mask]))
    ank_stand = float(np.mean(ang_ank_s[stand_slice]))
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

    lh_stand = float(np.mean(heel_l[stand_slice, 1]))
    rh_stand = float(np.mean(heel_r[stand_slice, 1]))
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


def extract_c4_v5_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    (64, 33, 3) 시퀀스로부터 c4 v5.0 25D 생체역학 지면 접지 피처 추출
    """
    kps = np.asarray(kps_33_64f, dtype=np.float32)
    T = kps.shape[0]

    # 1. v4 계승 22D 피처 추출
    f9 = extract_c4_elite_9d_features(kps)
    f20 = _extract_c4_20d_features(kps)
    full_29d = np.hstack([f9, f20])
    v4_22d = full_29d[TOP_22_INDICES_FROM_29D]

    # 2. F0 3대 핵심 피처 추출
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


def extract_c4_v5_features(X_kps: np.ndarray) -> np.ndarray:
    """배치 (N, 64, 33, 3) 또는 단일 시퀀스로부터 25D 피처 추출"""
    if X_kps.ndim == 3:
        return extract_c4_v5_features_from_seq(X_kps)
    N = X_kps.shape[0]
    out = np.zeros((N, 25), dtype=np.float32)
    for i in range(N):
        out[i] = extract_c4_v5_features_from_seq(X_kps[i])
    return out


class BabelSquatC4ModelV5:
    """
    바벨 스쿼트 c4 결함(발바닥 지면 고정) 공식 프로덕션 모델 v5.0
    - 25D 하이브리드 지면 접지 피처셋 (v4 22D + F0 3D)
    - CatBoost(60%) + LightGBM(30%) + Random Forest(10%) 최적 가중 앙상블
    - 3-State 선택적 판정 (Balanced [0.40, 0.60])
    """

    DEFECT_CODE = "c4"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 스쿼트"
    FEATURE_DIM = 25
    FEATURE_NAMES = FEATURE_NAMES_25_C4

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_normal: float = 0.40,
        tau_defect: float = 0.60
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect

        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.rf_models: List[Any] = []
        self.weights: Tuple[float, float, float] = (0.60, 0.30, 0.10)
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_25_C4

        if weights_path is None:
            weights_path = os.path.join(CURRENT_DIR, 'c4_model_v5_weights.pkl')
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    @classmethod
    def extract_features(cls, kps_33_64f: np.ndarray) -> np.ndarray:
        return extract_c4_v5_features(kps_33_64f)

    def predict_proba(self, X_25d: np.ndarray) -> np.ndarray:
        if not self.is_fitted or len(self.cb_models) == 0:
            raise RuntimeError("C4 v5 모델 가중치가 로드되지 않았습니다.")

        single = False
        if X_25d.ndim == 1:
            X_25d = X_25d.reshape(1, -1)
            single = True

        N = X_25d.shape[0]
        ensemble_probs = np.zeros(N, dtype=np.float32)
        n_folds = len(self.cb_models)

        w_cb, w_lgb, w_rf = self.weights

        for cb, clf, rf in zip(self.cb_models, self.lgb_models, self.rf_models):
            p_cb = cb.predict_proba(X_25d)[:, 1]
            p_lgb = clf.predict_proba(X_25d)[:, 1]
            p_rf = rf.predict_proba(X_25d)[:, 1]
            p_fold = w_cb * p_cb + w_lgb * p_lgb + w_rf * p_rf
            ensemble_probs += p_fold / n_folds

        res = np.column_stack([1.0 - ensemble_probs, ensemble_probs])
        return res[0] if single else res

    def predict(self, X_25d: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_25d)
        if probs.ndim == 1:
            return int(probs[1] >= self.decision_threshold)
        return (probs[:, 1] >= self.decision_threshold).astype(int)

    def predict_selective(
        self,
        X_or_kps: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35
    ) -> Dict[str, Any]:
        """
        3-State 선택적 판정 추론 (정상 / 결함 / 판단 불가)
        """
        if not self.is_fitted:
            raise RuntimeError("C4 v5 모델 가중치가 로드되지 않았습니다.")

        if tau_normal is None: tau_normal = self.tau_normal
        if tau_defect is None: tau_defect = self.tau_defect

        if X_or_kps.ndim == 3 and X_or_kps.shape[1] == 33:
            feats = self.extract_features(X_or_kps)
        elif X_or_kps.ndim == 1 and len(X_or_kps) == 25:
            feats = X_or_kps
        elif X_or_kps.ndim == 2 and X_or_kps.shape[0] == 1 and X_or_kps.shape[1] == 25:
            feats = X_or_kps[0]
        else:
            raise ValueError(f"지원하지 않는 입력 형상입니다: {X_or_kps.shape}")

        X_2d = feats.reshape(1, -1)
        w_cb, w_lgb, w_rf = self.weights

        cb_preds = [float(cb.predict_proba(X_2d)[0, 1]) for cb in self.cb_models]
        lgb_preds = [float(clf.predict_proba(X_2d)[0, 1]) for clf in self.lgb_models]
        rf_preds = [float(rf.predict_proba(X_2d)[0, 1]) for rf in self.rf_models]

        fold_preds = [w_cb * c + w_lgb * l + w_rf * r for c, l, r in zip(cb_preds, lgb_preds, rf_preds)]
        p_defect = float(np.mean(fold_preds))
        model_disagreement = float(abs(np.mean(cb_preds) - np.mean(lgb_preds)))
        confidence = float(min(1.0, 2.0 * abs(p_defect - 0.5)))

        heel_vis = float(feats[22]) # c4_f0_heel_view_visibility

        if model_disagreement >= disagreement_tol:
            status = "UNDETERMINED"
            reason = "MODEL_DISAGREEMENT"
            msg = "발 접지 상태 판정에 대한 AI 모델 간 예측 일치도가 낮습니다."
            act = "발바닥을 지면에 확실히 밀착하고 1회 다시 수행해 주세요."
        elif p_defect <= tau_normal:
            status = "NORMAL"
            reason = "CLEAR_GROUND_FIXATION"
            msg = "발바닥(뒤꿈치 및 앞꿈치) 지면 밀착이 매우 안정적으로 유지되었습니다."
            act = None
        elif p_defect >= tau_defect:
            status = "DEFECT"
            reason = "CLEAR_FOOT_ELEVATION"
            msg = "스쿼트 하강/상승 구간에서 뒤꿈치 또는 앞꿈치가 들려 지면 고정이 실패했습니다."
            act = "발뒤꿈치에 체중을 고르게 싣고 발바닥 3점(엄지·새끼발가락·뒤꿈치)을 지면에 단단히 누르세요."
        else:
            status = "UNDETERMINED"
            reason = "LOW_CONFIDENCE_MARGIN"
            msg = "미세한 발 움직임이 관절 지터(Jitter) 노이즈와 겹쳐 신뢰도가 부족하여 판정을 보류합니다."
            act = "신발과 지면이 선명하게 보이도록 카메라를 대각선 45도에 배치하고 다시 수행해 주세요."

        return {
            "status": status,
            "defect_probability": round(p_defect, 4),
            "confidence": round(confidence, 4),
            "heel_observability": round(heel_vis, 3),
            "reason_code": reason,
            "message": msg,
            "action": act
        }

    def predict_selective_batch(
        self,
        X_25d: np.ndarray,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None,
        disagreement_tol: float = 0.35
    ) -> List[Dict[str, Any]]:
        if X_25d.ndim == 1:
            return [self.predict_selective(X_25d, tau_normal, tau_defect, disagreement_tol)]
        return [
            self.predict_selective(X_25d[i], tau_normal, tau_defect, disagreement_tol)
            for i in range(X_25d.shape[0])
        ]

    def save_weights(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'cb_models': self.cb_models,
            'lgb_models': self.lgb_models,
            'rf_models': self.rf_models,
            'weights': self.weights,
            'decision_threshold': self.decision_threshold,
            'tau_normal': self.tau_normal,
            'tau_defect': self.tau_defect,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[*] c4 v5.0 가중치 저장 완료: {filepath}")

    def load_weights(self, filepath: str):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.cb_models = payload['cb_models']
        self.lgb_models = payload['lgb_models']
        self.rf_models = payload['rf_models']
        self.weights = payload.get('weights', (0.60, 0.30, 0.10))
        self.decision_threshold = payload.get('decision_threshold', 0.50)
        self.tau_normal = payload.get('tau_normal', 0.40)
        self.tau_defect = payload.get('tau_defect', 0.60)
        self.feature_names = payload.get('feature_names', FEATURE_NAMES_25_C4)
        self.is_fitted = True
        return self


BabelSquatC4Model = BabelSquatC4ModelV5
