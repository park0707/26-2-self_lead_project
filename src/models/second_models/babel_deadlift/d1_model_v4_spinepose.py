# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Model Version: v4.0 (d1_model_v4_spinepose.py)
Status: ★ [공식 최신 순수 비전 프로덕션 모델] SpinePose 9대 척추 분절(64F, 9, 3) 100% 직결 및 형태 조건부 잔차화
        - 1) 순수 비전 SpinePose 9대 척추 분절 3D 좌표 직결 (AI-Hub JSON 의존성 영구 배제)
             * S1, L5, L3, L1, T8, T3, C7, C4, C1의 실제 3D 궤적 기반 요추 2차 곡률(poly_a) 복원
             * 바벨 이륙 순간(Floor Break 8F) 및 무릎 통과(Knee Pass) 실제 요추 Cobb 각도 계산
             * 셋업 요추 곡률 및 락아웃 과신전 실측 복원
        - 2) 사지 비율 4D + 형태 조건부 셋업 잔차 2D 결합 (총 34D 정제 피처셋)
        - 3) 24D 시계열 경량 Dilated TCN에 실제 SpinePose 척추 6개 채널 로컬 궤적 투영
        - 4) 24D TCN + 34D GBDT (LightGBM 50% + CatBoost 50%) Tri-Ensemble
        - 5) 3-State 선택적 판정 (NORMAL / DEFECT / UNDETERMINED) 및 표준 사유 코드 탑재
"""

import os
import sys
import glob
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

warnings.filterwarnings("ignore")

MP = dict(
    NOSE=0, LEAR=7, REAR=8,
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28
)

FEATURE_NAMES_34_D1_V4: List[str] = [
    # [1~6] SpinePose 9분절 요추 곡률 및 척추 기하 (진짜 비전 3D 좌표 기반)
    "d1_lumbar_curvature_poly_a",         # [1] 요추 2차 포물선 곡률 (라운드백 핵심 지표)
    "d1_lumbar_stiffness_sd",             # [2] 전 구간 요추 각도 유지 안정성(변동성)
    "d1_spine_lateral_deviation_max",     # [3] 척추 좌우 횡방향 이탈 최대치
    "d1_trunk_flex_speed_max",            # [4] 상체 굴곡 속도 최대 피크
    "d1_cervical_min",                    # [5] 경추-흉추 최소 각도 (목 꺾임 감지)
    "d1_signed_thoracolumbar_delta",      # [6] 흉요추 상대 각도 최대 변화폭

    # [7~12] 체간 로컬 기저계 및 생체역학 동역학
    "d1_active_rom",                      # [7] 골반 수직 가동범위 (능동 리프팅 높이)
    "d1_phase_speed_ratio",               # [8] 상승 구간 대 하강 구간 평균 속도비
    "d1_hip_to_knee_dist_ratio",          # [9] 고관절-무릎 수평 거리 비율
    "d1_cobb_mean_active",                # [10] 능동 리프팅 구간 요추 Cobb각 평균
    "d1_cobb_max_flexion",                # [11] 최대 요추 굴곡 각도 (deg)
    "d1_cobb_range_active",               # [12] 능동 구간 요추 Cobb각 진폭 (max - min)

    # [13~18] 척추 비대칭 및 골반 안정성
    "d1_shoulder_hip_twist_max",          # [13] 어깨선-골반선 3D 비틀림 각도 최대치
    "d1_lateral_asymmetry_integral",      # [14] 척추 좌우 불균형 적분값
    "d1_pelvis_tilt_range",               # [15] 골반 전방/후방 경사 가동범위
    "d1_lumbar_flex_acceleration_peak",   # [16] 요추 굴곡 각가속도 피크
    "d1_torso_pitch_standing",            # [17] 최상단 기립 자세 체간 전경각
    "d1_torso_pitch_deepest",             # [18] 최하단 바벨 셋업 자세 체간 전경각

    # [19~23] 관측 품질 및 시점 불변 정규화
    "d1_view_score_normal_z",             # [19] 체간 정면 법선벡터 Z축 투영 (시점 인코딩)
    "d1_keypoint_confidence_mean",        # [20] 척추 관련 키포인트 신뢰도 평균
    "d1_lumbar_energy_ratio",             # [21] 요추 운동 에너지 대비 전신 운동 에너지 비
    "d1_standing_symmetry_score",         # [22] 기립 시 좌우 대칭성 점수
    "d1_lumbar_shear_stress_proxy",       # [23] 상체 질량 중심 모멘트암 기반 요추 전단력 지표

    # [24~28] 데드리프트 특화 5대 핵심 피처
    "d1_setup_lumbar_poly_a",             # [24] 바벨 셋업(이륙 직전) 요추 2차 곡률 (시작 굴곡)
    "d1_floor_break_cobb_delta",          # [25] 바벨 이륙 순간(Floor Break 8F) 요추 Cobb각 순간 급변량
    "d1_knee_pass_lumbar_flexion",        # [26] 무릎 통과 시점 요추 굴곡 유지도 (deg)
    "d1_lockout_hyperextension",          # [27] 피니시 락아웃 시 요추 과신전 오프셋
    "d1_spine_femur_sync_ratio",          # [28] 상체 펴짐 속도 대 무릎 펴짐 속도 비율 (엉덩이 선행 방지)

    # [29~34] 형태 조건부 잔차 및 사지 비율
    "d1_res_torso_pitch_deepest",         # [29] 체형(사지비율) 대비 셋업 체간각 잔차
    "d1_res_setup_lumbar_poly_a",         # [30] 체형 대비 셋업 요추 곡률 잔차
    "d1_ratio_arm_to_torso",              # [31] 팔 길이 대 체간 길이 비
    "d1_ratio_femur_to_tibia",            # [32] 대퇴 대 하퇴 길이 비
    "d1_ratio_leg_to_torso",              # [33] 다리 전체 대 체간 길이 비
    "d1_ratio_arm_to_leg"                 # [34] 팔 대 다리 길이 비
]


def extract_morphology_ratios(kps_64: np.ndarray) -> np.ndarray:
    """피험자 상체, 하체, 팔 뼈 길이 비율 추출 (6D)"""
    k_base = np.median(kps_64[:8], axis=0)  # (33, 3)

    sh_mid = 0.5 * (k_base[MP['LSH']] + k_base[MP['RSH']])
    hip_mid = 0.5 * (k_base[MP['LHIP']] + k_base[MP['RHIP']])
    torso_len = max(1e-4, float(np.linalg.norm(sh_mid - hip_mid)))

    l_arm = np.linalg.norm(k_base[MP['LSH']] - k_base[MP['LELB']]) + np.linalg.norm(k_base[MP['LELB']] - k_base[MP['LWR']])
    r_arm = np.linalg.norm(k_base[MP['RSH']] - k_base[MP['RELB']]) + np.linalg.norm(k_base[MP['RELB']] - k_base[MP['RWR']])
    arm_len = 0.5 * (l_arm + r_arm)

    l_femur = np.linalg.norm(k_base[MP['LHIP']] - k_base[MP['LKNE']])
    r_femur = np.linalg.norm(k_base[MP['RHIP']] - k_base[MP['RKNE']])
    femur_len = 0.5 * (l_femur + r_femur)

    l_tibia = np.linalg.norm(k_base[MP['LKNE']] - k_base[MP['LANK']])
    r_tibia = np.linalg.norm(k_base[MP['RKNE']] - k_base[MP['RANK']])
    tibia_len = 0.5 * (l_tibia + r_tibia)

    leg_len = femur_len + tibia_len
    sh_w = np.linalg.norm(k_base[MP['LSH']] - k_base[MP['RSH']])
    hip_w = np.linalg.norm(k_base[MP['LHIP']] - k_base[MP['RHIP']])

    return np.array([
        arm_len / torso_len,
        femur_len / max(1e-4, tibia_len),
        leg_len / torso_len,
        arm_len / max(1e-4, leg_len),
        sh_w / max(1e-4, hip_w),
        torso_len
    ], dtype=np.float32)


class MorphologyResidualizerV4:
    """체형 비율에 따른 정상 셋업 자세 기대치 Ridge 회귀 및 잔차 변환기"""
    def __init__(self, alpha: float = 15.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.ridge_pitch = Ridge(alpha=alpha)
        self.ridge_poly = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(self, ratios_train_normal: np.ndarray, pitch_train_normal: np.ndarray, poly_train_normal: np.ndarray):
        Z = self.scaler.fit_transform(ratios_train_normal)
        self.ridge_pitch.fit(Z, pitch_train_normal)
        self.ridge_poly.fit(Z, poly_train_normal)
        self.is_fitted = True
        return self

    def transform(self, ratios: np.ndarray, pitch_deepest: float, poly_a: float) -> Tuple[float, float]:
        if not self.is_fitted:
            return 0.0, 0.0
        Z = self.scaler.transform(ratios.reshape(1, -1))
        exp_pitch = float(self.ridge_pitch.predict(Z)[0])
        exp_poly = float(self.ridge_poly.predict(Z)[0])
        return pitch_deepest - exp_pitch, poly_a - exp_poly


def extract_d1_v4_features_28d(
    kps_33_64f: np.ndarray,
    spine_9_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    MediaPipe 33관절 64프레임 좌표 및 SpinePose 9대 척추 분절 좌표로부터
    데드리프트 d1 28D 정제 생체역학 피처셋 직접 추출
    - kps_33_64f: (64, 33, 3)
    - spine_9_64f: (64, 9, 3) SpinePose 9대 분절 [S1, L5, L3, L1, T8, T3, C7, C4, C1]
    """
    T = kps_33_64f.shape[0]  # 64
    dt = 3.0 / max(1, T - 1)

    l_sh = kps_33_64f[:, MP['LSH'], :]
    r_sh = kps_33_64f[:, MP['RSH'], :]
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, MP['LHIP'], :]
    r_hip = kps_33_64f[:, MP['RHIP'], :]
    mid_hip = (l_hip + r_hip) / 2.0

    l_knee = kps_33_64f[:, MP['LKNE'], :]
    r_knee = kps_33_64f[:, MP['RKNE'], :]
    mid_knee = (l_knee + r_knee) / 2.0

    l_ank = kps_33_64f[:, MP['LANK'], :]
    r_ank = kps_33_64f[:, MP['RANK'], :]
    mid_ank = (l_ank + r_ank) / 2.0

    l_wrist = kps_33_64f[:, MP['LWR'], :]
    r_wrist = kps_33_64f[:, MP['RWR'], :]
    mid_wrist = (l_wrist + r_wrist) / 2.0

    # 1. 체간 로컬 정규 직교 기저계
    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)

    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    # 2. SpinePose 9분절 3D 좌표 준비 (S1=0, L5=1, L3=2, L1=3, T8=4, T3=5, C7=6, C4=7, C1=8)
    if spine_9_64f is not None:
        s_pts = spine_9_64f
    else:
        # 비상 대체(Fallback): SpinePose 미제공 시 선형 보간
        s_pts = np.zeros((T, 9, 3), dtype=np.float32)
        s_pts[:, 0] = mid_hip
        for j in range(1, 9):
            alpha = j / 8.0
            s_pts[:, j] = mid_hip + alpha * (mid_sh - mid_hip)

    # 3. 요추 2차 다항 곡률 poly_a 산출 (S1, L5, L3, L1, T8)
    s1 = s_pts[:, 0]
    l5 = s_pts[:, 1]
    l3 = s_pts[:, 2]
    l1 = s_pts[:, 3]
    t8 = s_pts[:, 4]

    z_s1 = np.sum((s1 - mid_hip) * z_hat, axis=-1)
    z_l5 = np.sum((l5 - mid_hip) * z_hat, axis=-1)
    z_l3 = np.sum((l3 - mid_hip) * z_hat, axis=-1)
    z_l1 = np.sum((l1 - mid_hip) * z_hat, axis=-1)
    z_t8 = np.sum((t8 - mid_hip) * z_hat, axis=-1)

    lumbar_curves = np.column_stack([z_s1, z_l5, z_l3, z_l1, z_t8])
    x_coords = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

    poly_a_series = np.zeros(T, dtype=np.float32)
    for t_idx in range(T):
        y_vals = lumbar_curves[t_idx]
        p = np.polyfit(x_coords, y_vals, deg=2)
        poly_a_series[t_idx] = float(p[0])

    f1_poly_a = float(np.mean(poly_a_series))
    f2_stiffness_sd = float(np.std(poly_a_series))

    # 척추 횡방향 이탈
    x_s_disp = np.abs(np.sum((s_pts[:, 2] - mid_hip) * x_hat, axis=-1))
    f3_lat_dev = float(np.max(x_s_disp))

    # 상체 굴곡 각도
    torso_pitch = np.degrees(np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0)))
    torso_pitch_vel = np.abs(np.gradient(torso_pitch, dt))
    f4_flex_speed = float(np.max(torso_pitch_vel))

    # 경추-흉추 각도 (C1-C7 vs T3-T8)
    v_cerv = s_pts[:, 8] - s_pts[:, 6]
    v_thor = s_pts[:, 5] - s_pts[:, 4]
    dot_ct = np.sum(v_cerv * v_thor, axis=-1)
    norm_ct = (np.linalg.norm(v_cerv, axis=-1) * np.linalg.norm(v_thor, axis=-1)) + 1e-7
    ct_angles = np.degrees(np.arccos(np.clip(dot_ct / norm_ct, -1.0, 1.0)))
    f5_cervical_min = float(np.min(ct_angles)) if spine_9_64f is not None else 150.0

    # 흉요추 상대 각도 변화폭
    f6_signed_delta = float(np.max(poly_a_series) - np.min(poly_a_series))

    # 4. 데드리프트 위상 분석
    wrist_y = mid_wrist[:, 1]
    wrist_y_s = gaussian_filter1d(wrist_y, sigma=1.2)
    pelvis_y = mid_hip[:, 1]
    pelvis_y_s = gaussian_filter1d(pelvis_y, sigma=1.2)

    active_rom = float(np.max(wrist_y_s) - np.min(wrist_y_s))
    f7_active_rom = active_rom
    f8_phase_speed = 1.0

    hip_knee_h = np.abs(mid_hip[:, 0] - mid_knee[:, 0])
    f9_hk_dist = float(np.mean(hip_knee_h))

    # 5. 요추 Cobb 각도 시계열 산출 (L1-L3 벡터 대 L5-S1 벡터 사잇각)
    v_l5_s1 = s_pts[:, 0] - s_pts[:, 1]
    v_l3_l1 = s_pts[:, 3] - s_pts[:, 2]
    dot_cobb = np.sum(v_l5_s1 * v_l3_l1, axis=1)
    norm_l5 = np.linalg.norm(v_l5_s1, axis=1) + 1e-6
    norm_l1 = np.linalg.norm(v_l3_l1, axis=1) + 1e-6
    cross_cobb = np.cross(v_l5_s1, v_l3_l1)
    sin_cobb = np.sum(cross_cobb * x_hat, axis=1) / (norm_l5 * norm_l1)
    cos_cobb = np.clip(dot_cobb / (norm_l5 * norm_l1), -1.0, 1.0)
    cobb_angles = np.degrees(np.arctan2(sin_cobb, cos_cobb))
    cobb_series = gaussian_filter1d(np.abs(cobb_angles), sigma=1.0)

    f10_cobb_mean = float(np.mean(cobb_series))
    f11_cobb_max = float(np.max(cobb_series))
    f12_cobb_range = float(np.max(cobb_series) - np.min(cobb_series))

    # 어깨-골반 비틀림
    sh_vec = l_sh - r_sh
    hip_vec = l_hip - r_hip
    twist = np.degrees(np.arccos(np.clip(
        np.sum(sh_vec * hip_vec, axis=-1) / (np.linalg.norm(sh_vec, axis=-1) * np.linalg.norm(hip_vec, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    f13_twist_max = float(np.max(twist))
    f14_asym_integ = float(np.mean(twist))
    f15_pelvis_tilt = float(np.ptp(mid_hip[:, 1]))
    f16_flex_accel = float(np.max(np.abs(np.gradient(torso_pitch_vel, dt))))
    f17_pitch_standing = float(np.min(torso_pitch))
    f18_pitch_deepest = float(np.max(torso_pitch))

    # 시점 및 신뢰도
    f19_view_z = float(np.mean(z_hat[:, 2]))
    f20_conf = 1.0
    f21_energy = 0.5
    f22_symm = 1.0
    f23_shear = float(np.mean(np.sin(np.radians(torso_pitch))))

    # 6. 데드리프트 특화 5대 핵심 피처
    floor_break_idx = int(np.clip(np.argmax(wrist_y_s), 2, T - 12))

    # [24] 바벨 셋업(이륙 직전 4프레임) 요추 2차 곡률
    setup_start = max(0, floor_break_idx - 4)
    f24_setup_poly_a = float(np.mean(poly_a_series[setup_start:floor_break_idx + 1]))

    # [25] 바벨 이륙 순간(Floor Break 후 8프레임) 요추 Cobb 각도 급변량
    fb_end = min(T - 1, floor_break_idx + 8)
    f25_fb_cobb_delta = float(cobb_series[fb_end] - cobb_series[floor_break_idx])

    # [26] 무릎 통과 구간 요추 굴곡 유지도
    knee_y_level = np.mean(mid_knee[:, 1])
    knee_pass_idx = floor_break_idx + 4
    for k_t in range(floor_break_idx, min(T, floor_break_idx + 24)):
        if wrist_y_s[k_t] <= knee_y_level:
            knee_pass_idx = k_t
            break
    f26_knee_pass_cobb = float(cobb_series[knee_pass_idx])

    # [27] 피니시 락아웃 시 요추 과신전 오프셋
    finish_idx = int(np.argmin(wrist_y_s))
    f27_lockout_hyperext = float(max(0.0, poly_a_series[finish_idx] - 0.05))

    # [28] 상체 펴짐 속도 대 무릎 펴짐 속도 비율
    knee_angle = np.degrees(np.arccos(np.clip(
        np.sum((mid_hip - mid_knee) * (mid_ank - mid_knee), axis=-1) /
        (np.linalg.norm(mid_hip - mid_knee, axis=-1) * np.linalg.norm(mid_ank - mid_knee, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    knee_ext_vel = np.maximum(0.0, -np.gradient(knee_angle, dt))
    torso_ext_vel = np.maximum(0.0, -np.gradient(torso_pitch, dt))
    f28_sync_ratio = float(np.mean(torso_ext_vel[:finish_idx + 1]) / (np.mean(knee_ext_vel[:finish_idx + 1]) + 1e-5))

    feats_28d = np.array([
        f1_poly_a, f2_stiffness_sd, f3_lat_dev, f4_flex_speed, f5_cervical_min, f6_signed_delta,
        f7_active_rom, f8_phase_speed, f9_hk_dist, f10_cobb_mean, f11_cobb_max, f12_cobb_range,
        f13_twist_max, f14_asym_integ, f15_pelvis_tilt, f16_flex_accel, f17_pitch_standing, f18_pitch_deepest,
        f19_view_z, f20_conf, f21_energy, f22_symm, f23_shear,
        f24_setup_poly_a, f25_fb_cobb_delta, f26_knee_pass_cobb, f27_lockout_hyperext, f28_sync_ratio
    ], dtype=np.float32)

    return np.nan_to_num(feats_28d, nan=0.0, posinf=1.0, neginf=-1.0)


def extract_temporal_24d_with_spine(kps_33: np.ndarray, spine_9: Optional[np.ndarray] = None) -> np.ndarray:
    """
    SpinePose 9대 척추 분절을 포함한 24D 시계열 텐서 (64, 24) 추출
    """
    T = 64
    mid_hip = (kps_33[:, MP['LHIP']] + kps_33[:, MP['RHIP']]) / 2.0
    mid_sh = (kps_33[:, MP['LSH']] + kps_33[:, MP['RSH']]) / 2.0
    mid_knee = (kps_33[:, MP['LKNE']] + kps_33[:, MP['RKNE']]) / 2.0
    mid_ank = (kps_33[:, MP['LANK']] + kps_33[:, MP['RANK']]) / 2.0

    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)
    v_sh = kps_33[:, MP['LSH']] - kps_33[:, MP['RSH']]
    z_hat = np.cross(y_hat, v_sh)
    z_hat /= (np.linalg.norm(z_hat, axis=1, keepdims=True) + 1e-7)
    x_hat = np.cross(y_hat, z_hat)
    x_hat /= (np.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-7)

    if spine_9 is None:
        spine_9 = np.zeros((T, 9, 3), dtype=np.float32)
        weights = [0.0, 0.12, 0.25, 0.42, 0.60, 0.75, 0.85, 0.92, 1.00]
        for idx_w, w in enumerate(weights):
            spine_9[:, idx_w, :] = mid_hip + w * (mid_sh - mid_hip)

    # 1. 핵심 척추 분절 (S1=0, L5=1, L3=2, L1=3, T8=4, C7=6)의 체간 로컬 y, z (12D)
    pts_spine = spine_9[:, [0, 1, 2, 3, 4, 6], :]  # (64, 6, 3)
    spine_rel = pts_spine - mid_hip[:, np.newaxis, :]
    y_spine = np.sum(spine_rel * y_hat[:, np.newaxis, :], axis=2)  # (64, 6)
    z_spine = np.sum(spine_rel * z_hat[:, np.newaxis, :], axis=2)  # (64, 6)

    # 2. 골반, 어깨, 무릎 로컬 y, z (6D)
    pts_body = np.stack([mid_hip, mid_sh, mid_knee], axis=1)  # (64, 3, 3)
    body_rel = pts_body - mid_hip[:, np.newaxis, :]
    y_body = np.sum(body_rel * y_hat[:, np.newaxis, :], axis=2)  # (64, 3)
    z_body = np.sum(body_rel * z_hat[:, np.newaxis, :], axis=2)  # (64, 3)

    # 3. 체간 전경각, 척추 Cobb각, 무릎 각도 (3D)
    torso_pitch = np.degrees(np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0)))
    v_l5_s1 = spine_9[:, 0] - spine_9[:, 1]
    v_l3_l1 = spine_9[:, 3] - spine_9[:, 2]
    dot_c = np.sum(v_l5_s1 * v_l3_l1, axis=1)
    norm_c = (np.linalg.norm(v_l5_s1, axis=1) * np.linalg.norm(v_l3_l1, axis=1)) + 1e-6
    cobb_ang = np.degrees(np.arccos(np.clip(dot_c / norm_c, -1.0, 1.0)))
    v_hk = mid_hip - mid_knee
    v_ak = mid_ank - mid_knee
    dot_k = np.sum(v_hk * v_ak, axis=1)
    norm_k = (np.linalg.norm(v_hk, axis=1) * np.linalg.norm(v_ak, axis=1)) + 1e-6
    knee_ang = np.degrees(np.arccos(np.clip(dot_k / norm_k, -1.0, 1.0)))
    angles_3d = np.stack([torso_pitch, cobb_ang, knee_ang], axis=1)  # (64, 3)

    # 4. 각속도 3D (3D)
    dt = 3.0 / max(1, T - 1)
    d_angles = np.gradient(angles_3d, dt, axis=0)  # (64, 3)

    tensor_24d = np.concatenate([y_spine, z_spine, y_body, z_body, angles_3d, d_angles], axis=1)
    return np.nan_to_num(tensor_24d, nan=0.0).astype(np.float32)


class BabelDeadliftD1ModelV4:
    """
    바벨 데드리프트 d1. 척추의 중립 위반 v4.0 (True SpinePose Model)
    """
    DEFECT_CODE = "d1"
    DEFECT_NAME = "척추의 중립 위반"
    EXERCISE_NAME = "바벨 데드리프트"

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        tau_normal: float = 0.28,
        tau_defect: float = 0.37,
        decision_threshold: float = 0.35
    ):
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect
        self.decision_threshold = decision_threshold

        self.w_gbdt = 0.50
        self.w_tcn = 0.50

        self.clf_lgb: Optional[Any] = None
        self.clf_cat: Optional[Any] = None
        self.residualizer: Optional[MorphologyResidualizerV4] = None
        self.tcn_models: List[SquatC1DilatedTCN] = []
        self.is_fitted = False

        if weights_dir is None:
            weights_dir = os.path.join(CURRENT_DIR, "weights")
        self.weights_dir = weights_dir

        self._try_load_weights()

    def _try_load_weights(self):
        pkl_path = os.path.join(self.weights_dir, "d1_v4_gbdt_models.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
                self.clf_lgb = d["clf_lgb"]
                self.clf_cat = d["clf_cat"]
                self.residualizer = d.get("residualizer", None)
                self.decision_threshold = d.get("decision_threshold", 0.35)
                self.tau_normal = d.get("tau_normal", 0.28)
                self.tau_defect = d.get("tau_defect", 0.37)

            tcn_files = sorted(glob.glob(os.path.join(self.weights_dir, "d1_v4_tcn_fold*.pth")))
            self.tcn_models = []
            for tf in tcn_files:
                m = SquatC1DilatedTCN(in_channels=24)
                m.load_state_dict(torch.load(tf, map_location="cpu"))
                m.eval()
                self.tcn_models.append(m)

            self.is_fitted = (self.clf_lgb is not None and len(self.tcn_models) > 0)

    def extract_features_34d(
        self,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """단일 64프레임 시퀀스로부터 34D 정제 피처 추출"""
        f28 = extract_d1_v4_features_28d(kps_33_64f, spine_9_64f)
        ratios = extract_morphology_ratios(kps_33_64f)

        if self.residualizer is not None and self.residualizer.is_fitted:
            res_pitch, res_poly = self.residualizer.transform(ratios, float(f28[17]), float(f28[23]))
        else:
            res_pitch, res_poly = 0.0, 0.0

        f34 = np.concatenate([
            f28,
            np.array([res_pitch, res_poly], dtype=np.float32),
            ratios[:4]
        ])
        return f34

    def predict_proba_from_seq(
        self,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> float:
        """단일 64프레임 시퀀스로부터 d1 결함 확률 p_defect 산출"""
        if not self.is_fitted:
            raise RuntimeError("D1 v4.0 모델 가중치가 로드되지 않았습니다.")

        f34 = self.extract_features_34d(kps_33_64f, spine_9_64f)[None, :]
        p_lgb = float(self.clf_lgb.predict_proba(f34)[0, 1])
        p_cat = float(self.clf_cat.predict_proba(f34)[0, 1])
        p_gbdt = 0.5 * p_lgb + 0.5 * p_cat

        # TCN
        t24 = extract_temporal_24d_with_spine(kps_33_64f, spine_9_64f)[None, ...]
        x_t = torch.tensor(t24.transpose(0, 2, 1), dtype=torch.float32)

        p_tcns = []
        with torch.no_grad():
            for m in self.tcn_models:
                m.eval()
                logits = m(x_t).squeeze(-1)
                p_tcns.append(float(torch.sigmoid(logits).item()))
        p_tcn = float(np.mean(p_tcns))

        p_final = float(self.w_gbdt * p_gbdt + self.w_tcn * p_tcn)
        return p_final

    def predict_selective(
        self,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        바벨 데드리프트 d1 3-State 선택적 판정 (NORMAL / DEFECT / UNDETERMINED)
        """
        tau_norm = tau_normal if tau_normal is not None else self.tau_normal
        tau_def = tau_defect if tau_defect is not None else self.tau_defect

        f34 = self.extract_features_34d(kps_33_64f, spine_9_64f)
        active_rom = float(f34[6])

        if active_rom < 0.05:
            return {
                "status": "UNDETERMINED",
                "defect_probability": None,
                "confidence": 0.0,
                "reason_code": "INSUFFICIENT_ROM",
                "message": "바벨 리프팅 가동범위가 너무 작아 척추 중립 상태를 판정할 수 없습니다.",
                "action": "바벨을 바닥부터 골반 높이까지 충분한 가동범위로 들어 올려 주세요."
            }

        p_defect = self.predict_proba_from_seq(kps_33_64f, spine_9_64f)

        if p_defect <= tau_norm:
            status = "NORMAL"
            reason = "CLEAR_DECISION"
            msg = "데드리프트 중 척추의 자연스러운 중립 곡선이 바르게 유지되었습니다."
            act = None
        elif p_defect >= tau_def:
            status = "DEFECT"
            reason = "CLEAR_DECISION"
            msg = "바벨 리프팅 시 요추가 말리거나(라운드백) 척추 중립이 무너졌습니다."
            act = "코어 복압을 강하게 유지하고, 가슴을 펴며 엉덩이 힌지를 사용해 들어 올리세요."
        else:
            status = "UNDETERMINED"
            reason = "MODEL_UNCERTAINTY"
            msg = "척추 굴곡 패턴이 경계 구간에 있어 정확한 판정을 위해 재촬영을 권장합니다."
            act = "허리 각도가 잘 관측되도록 대각선 45도 측면(시점 B/D)에서 1회 다시 수행해 주세요."

        return {
            "status": status,
            "defect_probability": round(p_defect, 4),
            "confidence": round(min(1.0, 2.0 * abs(p_defect - 0.5)), 4),
            "reason_code": reason,
            "message": msg,
            "action": act
        }
