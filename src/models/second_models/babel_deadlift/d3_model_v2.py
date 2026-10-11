# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation / Heel-Toe Elevation)
Model Version: v2.0 (d3_model_v2.py)
Status: 데드리프트 위상 불변 적응형 지면 기준선 및 28D 생체역학 지면 접지 고도화 모델
        - [혁신 1] 위상 불변 적응형 기립 기준선 (Adaptive True Lockout/Stand Baseline Finder)
                  -> 바닥 셋업 시작(56.9%)과 기립 시작(43.1%) 모두에서 골반 최고점(hip_y 최소)을 동적 포착하여 지면 기준선 설정
        - [혁신 2] 정강이-발바닥 3D 로컬 상대각(Shank-Foot Relative Local Angle) 및 관절각 잔차화
                  -> 카메라 각도 및 셋업 상체 기울기에 불변하는 순수 발목 안정성 지표 확보
        - [혁신 3] 바벨-발목 가림 인식 및 동역학 필터링 (Occlusion Proximity)
        - [혁신 4] 28D 생체역학 지면 접지 피처셋 + CatBoost / LightGBM / RF Tri-Ensemble
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

FEATURE_NAMES_28_D3: List[str] = [
    # [1~22] 적응형 기립 기준선 기반 정밀 22D 물리 접지 피처
    'd3_foot_pitch_max_deg',               # [0] 전 구간 최대 발바닥 경사각 (deg)
    'd3_toe_lift_max_norm',                # [1] 적응형 스케일 정규화 앞꿈치 최대 들림량
    'd3_foot_pitch_nadir_deg',             # [2] 최하사점 발바닥 경사각 (deg)
    'd3_delta_wobble_active',              # [3] 능동 구간 뒤꿈치-앞꿈치 수직차 흔들림 (std)
    'd3_roll_max_active',                  # [4] 능동 구간 최대 발바닥 롤각 (deg)
    'd3_delta_max_active',                 # [5] 능동 구간 최대 수직 높이차
    'd3_heel_toe_ydelta_nadir_cm',         # [6] 최하사점 뒤꿈치-앞꿈치 수직차 (cm)
    'd3_coupling_ratio_shank_pitch',       # [7] 정강이 경사 대비 발목 가동성 커플링비
    'd3_bilateral_foot_pitch_asym_deg',    # [8] 좌우 발바닥 경사각 비대칭도 (deg)
    'd3_pitch_max_active_deg',             # [9] 능동 구간 최대 피치각 (deg)
    'd3_shank_ankle_coupling_ratio_legacy',# [10] 정강이-발목 시계열 상관계수
    'd3_foot_pitch_drop_deg',              # [11] '진짜 기립' 대비 최하사점 경사각 변화폭 (deg)
    'd3_foot_roll_inversion_deg',          # [12] 최하사점 발목 롤 / 안쪽날 들림 (deg)
    'd3_asym_pitch_active',                # [13] 능동 구간 최대 피치 비대칭도
    'd3_roll_nadir_deg',                   # [14] 최하사점 발바닥 롤각 (deg)
    'd3_delta_jerk_active',                # [15] 리프팅 구간 지면 가속도 저크
    'd3_ank_min_angle',                    # [16] 능동 구간 최소 발목 관절각 (deg)
    'd3_ascent_ankle_destabilization',     # [17] 리프팅 구간 발목 불안정성 (cm)
    'd3_toe_lift_max_cm',                  # [18] '진짜 기립' 기준 앞꿈치 최대 들림량 (cm)
    'd3_asym_delta_active',                # [19] 능동 구간 좌우 높이차 비대칭도
    'd3_norm_delta_nadir',                 # [20] 발 길이 정규화 최하사점 수직차
    'd3_pitch_drop_active_deg',            # [21] 능동 구간 피치 드롭각 (deg)
    # [23~25] F0 3대 시점 가시성 및 동역학 피처 (3D)
    'd3_f0_heel_view_visibility',          # [22] 시점별 뒤꿈치 가시성 계수 (정면 가림 vs 측면 노출)
    'd3_f0_shank_lift_corr',               # [23] 최하사점 정강이 경사-뒤꿈치 들림 연동 역학 상관계수
    'd3_f0_ascent_transition_rate',        # [24] 리프팅 구간 접촉 상태 머신 전이 빈도
    # [26~28] v2 신규 데드리프트 특화 3대 혁신 피처 (3D)
    'd3_shank_foot_angle_stand_deg',       # [25] 진짜 기립 상태 정강이-발바닥 3D 로컬 직각도 (deg)
    'd3_shank_foot_angle_drop_deg',        # [26] 기립 대비 최하사점 정강이-발바닥 각도 변화폭 (deg)
    'd3_bar_ankle_min_dist_norm'           # [27] 바벨(손목)-발목 최단 접근 거리 (원판 가림 노이즈 보정)
]

TOP_22_INDICES = [2, 17, 0, 23, 19, 11, 3, 22, 4, 14, 7, 1, 6, 15, 18, 24, 20, 8, 5, 16, 27, 13]


def compute_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """세 3D 점 사이의 사잇각(도, deg) 계산"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_ang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def extract_d3_v2_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    (64, 33, 3) 시퀀스로부터 d3 v2.0 위상 불변 28D 생체역학 지면 접지 피처 추출
    """
    kps = np.asarray(kps_33_64f, dtype=np.float32)
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)

    # 주요 랜드마크 추출
    l_wrist, r_wrist = kps[:, MP_LEFT_WRIST], kps[:, MP_RIGHT_WRIST]
    hip_l, hip_r = kps[:, MP_LEFT_HIP], kps[:, MP_RIGHT_HIP]
    knee_l, knee_r = kps[:, MP_LEFT_KNEE], kps[:, MP_RIGHT_KNEE]
    ank_l, ank_r = kps[:, MP_LEFT_ANKLE], kps[:, MP_RIGHT_ANKLE]
    heel_l, heel_r = kps[:, MP_LEFT_HEEL], kps[:, MP_RIGHT_HEEL]
    toe_l, toe_r = kps[:, MP_LEFT_FOOT_INDEX], kps[:, MP_RIGHT_FOOT_INDEX]

    # 1. [혁신 1] 위상 불변 적응형 기립(Lockout) 및 최하사점(Nadir) 윈도우 탐색
    hip_y = (hip_l[:, 1] + hip_r[:, 1]) / 2.0
    hip_y_s = gaussian_filter1d(hip_y, sigma=1.5)

    # MediaPipe y축: 0이 상단(최고점), 1이 하단(최저점)
    # 진짜 기립(True Lockout): hip_y_s가 최소인 프레임
    stand_idx = int(np.argmin(hip_y_s))
    stand_window = slice(max(0, stand_idx - 3), min(T, stand_idx + 4))

    # 최하사점(Bottom Setup): hip_y_s가 최대인 프레임
    nadir_idx = int(np.argmax(hip_y_s))
    nadir_window = slice(max(0, nadir_idx - 3), min(T, nadir_idx + 4))
    b = nadir_idx

    # 능동 가동 범위 (Active Mask)
    rom = float(np.max(hip_y_s) - np.min(hip_y_s))
    if rom > 0.0004:
        active_mask = (hip_y_s - np.min(hip_y_s)) >= (rom * 0.15)
    else:
        active_mask = np.ones(T, dtype=bool)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    # 발 기하 벡터
    vec_foot_l = toe_l - heel_l
    vec_foot_r = toe_r - heel_r
    foot_len_l = np.linalg.norm(vec_foot_l, axis=-1)
    foot_len_r = np.linalg.norm(vec_foot_r, axis=-1)
    foot_len = (np.mean(foot_len_l) + np.mean(foot_len_r)) / 2.0 + 1e-6

    pitch_l = np.degrees(np.arctan2(vec_foot_l[:, 1], np.hypot(vec_foot_l[:, 0], vec_foot_l[:, 2]) + 1e-7))
    pitch_r = np.degrees(np.arctan2(vec_foot_r[:, 1], np.hypot(vec_foot_r[:, 0], vec_foot_r[:, 2]) + 1e-7))
    pitch_mean = (pitch_l + pitch_r) / 2.0
    pitch_s = gaussian_filter1d(pitch_mean, sigma=1.0)

    delta_l = toe_l[:, 1] - heel_l[:, 1]
    delta_r = toe_r[:, 1] - heel_r[:, 1]
    delta_mean = (delta_l + delta_r) / 2.0
    delta_s = gaussian_filter1d(delta_mean, sigma=1.0)

    # 2. [혁신 1 적용] 진짜 기립 기준선 (True Standing Baseline)
    pitch_stand = float(np.median(pitch_s[stand_window]))
    pitch_nadir = float(pitch_s[b])
    pitch_drop = float(pitch_nadir - pitch_stand)
    pitch_max = float(np.max(pitch_s[active_mask]))

    delta_stand = float(np.median(delta_s[stand_window]))
    delta_nadir = float(delta_s[b])
    delta_drop = float(delta_nadir - delta_stand)
    delta_max_active = float(np.max(delta_s[active_mask]))

    # 앞꿈치 들림 및 뒤꿈치 들림 (진짜 기립 기준선 대비)
    lh_stand = float(np.median(heel_l[stand_window, 1]))
    rh_stand = float(np.median(heel_r[stand_window, 1]))
    lh_lift = np.maximum(0.0, lh_stand - heel_l[:, 1])
    rh_lift = np.maximum(0.0, rh_stand - heel_r[:, 1])
    heel_lift_max = float(max(np.max(lh_lift[active_mask]), np.max(rh_lift[active_mask])))
    heel_lift_nadir = float((lh_lift[b] + rh_lift[b]) / 2.0)

    toe_lift_l = np.maximum(0.0, heel_l[:, 1] - toe_l[:, 1])
    toe_lift_r = np.maximum(0.0, heel_r[:, 1] - toe_r[:, 1])
    toe_lift_max = float(max(np.max(toe_lift_l[active_mask]), np.max(toe_lift_r[active_mask])))

    # 롤 특이점 보정 (±35도)
    roll_l = np.clip(np.degrees(np.arctan2(vec_foot_l[:, 0], vec_foot_l[:, 1] + 1e-7)), -35.0, 35.0)
    roll_r = np.clip(np.degrees(np.arctan2(vec_foot_r[:, 0], vec_foot_r[:, 1] + 1e-7)), -35.0, 35.0)
    roll_nadir = float((abs(roll_l[b]) + abs(roll_r[b])) / 2.0)
    roll_max = float(max(np.max(np.abs(roll_l)[active_mask]), np.max(np.abs(roll_r)[active_mask])))

    # 관절각 및 커플링비
    ang_ank_l = compute_angle_3d(knee_l, ank_l, toe_l)
    ang_ank_r = compute_angle_3d(knee_r, ank_r, toe_r)
    ang_ank_mean = (ang_ank_l + ang_ank_r) / 2.0
    ang_ank_s = gaussian_filter1d(ang_ank_mean, sigma=1.0)
    ank_min = float(np.min(ang_ank_s[active_mask]))
    ank_stand = float(np.median(ang_ank_s[stand_window]))
    ank_drop = float(ank_stand - ank_min)

    vec_shank_l = ank_l - knee_l
    vec_shank_r = ank_r - knee_r
    shank_pitch_l = np.degrees(np.arctan2(vec_shank_l[:, 2], vec_shank_l[:, 1] + 1e-7))
    shank_pitch_r = np.degrees(np.arctan2(vec_shank_r[:, 2], vec_shank_r[:, 1] + 1e-7))
    shank_pitch = (shank_pitch_l + shank_pitch_r) / 2.0
    shank_diff = float(np.max(shank_pitch[active_mask]) - np.min(shank_pitch[active_mask]) + 1e-6)
    ank_diff = float(np.max(ang_ank_s[active_mask]) - np.min(ang_ank_s[active_mask]))
    coupling_ratio = float(ank_diff / shank_diff)

    asym_pitch = float(np.max(np.abs(pitch_l - pitch_r)[active_mask]))
    asym_delta = float(np.max(np.abs(delta_l - delta_r)[active_mask]))

    # 리프팅/변위 구간 저크 및 불안정성
    v_delta = np.gradient(delta_s, dt)
    a_delta = np.gradient(v_delta, dt)
    j_delta = np.gradient(a_delta, dt)
    delta_jerk = float(np.mean(np.abs(j_delta[active_mask])))
    delta_wobble = float(np.std(delta_s[active_mask]))

    norm_delta_nadir = float(delta_nadir / foot_len)
    norm_delta_drop = float(delta_drop / foot_len)

    # 9D 엘리트 적응형 피처
    f1 = float(pitch_mean[b])
    f2 = float(pitch_mean[b] - pitch_stand)
    f3 = float(np.max(pitch_mean))
    f4 = float(delta_mean[b])
    f5 = float(np.max(np.abs(pitch_l - pitch_r)))
    f6 = float(max(np.max(np.maximum(0, heel_l[:, 1] - toe_l[:, 1])), np.max(np.maximum(0, heel_r[:, 1] - toe_r[:, 1]))))
    f7 = float((abs(roll_l[b]) + abs(roll_r[b])) / 2.0)
    f8 = coupling_ratio
    f9 = delta_wobble
    f_elite_9d = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=np.float32)

    # 20D 확장 적응형 피처
    f_20d = np.array([
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

    full_29d = np.hstack([f_elite_9d, f_20d])
    v4_22d = full_29d[TOP_22_INDICES]

    # 3. F0 3대 피처 (적응형 기립 반영)
    v_foot_x = (vec_foot_l[:, 0] + vec_foot_r[:, 0]) / 2.0
    v_foot_z = (vec_foot_l[:, 2] + vec_foot_r[:, 2]) / 2.0
    foot_yaw = np.degrees(np.arctan2(v_foot_x, v_foot_z + 1e-7))
    f22_heel_view_visibility = float(np.sin(np.radians(abs(float(np.mean(foot_yaw))))))

    norm_lift = delta_s - delta_stand
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

    is_heel_lift = (norm_lift > 0.03 * foot_len) & active_mask
    is_toe_lift = (norm_lift < -0.03 * foot_len) & active_mask
    states = np.zeros(T, dtype=int)
    states[is_heel_lift] = 1
    states[is_toe_lift] = 2
    if np.sum(active_mask) > 1:
        act_states = states[active_mask]
        f24_ascent_transition_rate = float(np.sum(act_states[1:] != act_states[:-1]) / len(act_states))
    else:
        f24_ascent_transition_rate = 0.0

    f0_3d = np.array([f22_heel_view_visibility, f23_shank_lift_corr, f24_ascent_transition_rate], dtype=np.float32)

    # 4. [혁신 2 & 3] 신규 데드리프트 특화 3대 피처
    # (1) 진짜 기립 상태 정강이-발바닥 3D 로컬 상대각 (d3_shank_foot_angle_stand_deg)
    shank_foot_ang_stand = float(np.median(ang_ank_s[stand_window]))
    # (2) 기립 대비 최하사점 정강이-발바닥 상대각 변화폭 (d3_shank_foot_angle_drop_deg)
    shank_foot_ang_drop = float(shank_foot_ang_stand - ang_ank_s[b])
    # (3) 바벨(손목)-발목 최단 접근 거리 정규화 (d3_bar_ankle_min_dist_norm)
    wrist_mid = (l_wrist + r_wrist) / 2.0
    ank_mid = (ank_l + ank_r) / 2.0
    bar_ank_dist = np.linalg.norm(wrist_mid - ank_mid, axis=-1)
    bar_ank_min_dist_norm = float(np.min(bar_ank_dist) / foot_len)

    new_3d = np.array([shank_foot_ang_stand, shank_foot_ang_drop, bar_ank_min_dist_norm], dtype=np.float32)

    return np.hstack([v4_22d, f0_3d, new_3d])


def extract_d3_v2_features(X_kps: np.ndarray) -> np.ndarray:
    """배치 (N, 64, 33, 3) 또는 단일 시퀀스로부터 28D 피처 추출"""
    if X_kps.ndim == 3:
        return extract_d3_v2_features_from_seq(X_kps)
    N = X_kps.shape[0]
    out = np.zeros((N, 28), dtype=np.float32)
    for i in range(N):
        out[i] = extract_d3_v2_features_from_seq(X_kps[i])
    return out


class BabelDeadliftD3ModelV2:
    """
    바벨 데드리프트 d3 결함(발바닥 지면 고정) 공식 프로덕션 모델 v2.0
    - 위상 불변 적응형 기립 기준선 (Adaptive Standing Baseline) 탑재
    - 정강이-발바닥 3D 로컬 상대각 및 바벨 원판 가림 인식 28D 피처셋
    - CatBoost(50%) + LightGBM(30%) + Random Forest(20%) 3-Way Tri-Ensemble
    - 3-State 선택적 판정(Selective Classification) 메서드 탑재
    """

    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    FEATURE_DIM = 28
    FEATURE_NAMES = FEATURE_NAMES_28_D3

    def __init__(
        self,
        weights_path: Optional[str] = None,
        decision_threshold: float = 0.50,
        tau_normal: float = 0.40,
        tau_defect: float = 0.55
    ):
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect

        self.cb_models: List[Any] = []
        self.lgb_models: List[Any] = []
        self.rf_models: List[Any] = []
        self.weights: Tuple[float, float, float] = (0.50, 0.30, 0.20)
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES_28_D3

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
        """28D 피처 행렬로부터 결함 확률 추론"""
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

    def predict_proba(self, kps_seq: np.ndarray) -> float:
        """단일 (64, 33, 3) 시퀀스로부터 결함 확률 반환"""
        feat = extract_d3_v2_features_from_seq(kps_seq).reshape(1, -1)
        return float(self.predict_proba_features(feat)[0])

    def predict_3state(self, prob: float) -> Tuple[int, str]:
        """3-State 판정 (0: NORMAL, 1: DEFECT, 2: RETAKE)"""
        if prob <= self.tau_normal:
            return 0, "NORMAL"
        elif prob >= self.tau_defect:
            return 1, "DEFECT"
        else:
            return 2, "RETAKE"

    def predict_single_3state(self, kps_seq: np.ndarray) -> Dict[str, Any]:
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
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        self.cb_models = data['cb_models']
        self.lgb_models = data['lgb_models']
        self.rf_models = data['rf_models']
        self.weights = data['weights']
        self.decision_threshold = data.get('decision_threshold', 0.50)
        self.tau_normal = data.get('tau_normal', 0.40)
        self.tau_defect = data.get('tau_defect', 0.55)
        self.feature_names = data.get('feature_names', FEATURE_NAMES_28_D3)
        self.is_fitted = True
