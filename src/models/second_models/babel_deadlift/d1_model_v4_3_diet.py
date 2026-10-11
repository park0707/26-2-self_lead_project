# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Model Version: v4.3 Diet (d1_model_v4_3_diet.py)
Status: ★ [공식 최신 피처 다이어트 최적화 모델]
         - 기존 40D 중 분산 0 상수 4개 및 다중공선성/성능 저해 피처 11개 전수 제거
         - 25D 정예 생체역학 피처셋 구축 (Drop-one Ablation & 5-Fold 실측 검증)
         - 24D Dilated TCN + 25D LightGBM + 25D CatBoost Tri-Ensemble
         - 3-State 선택적 판정 (INSUFFICIENT_ROM 안전망 + 최적 임계치)
"""

import os
import sys
import glob
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
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

# 25D 정예 피처 명칭 (40D에서 15개 저해/상수/고상관 피처 제거)
FEATURE_NAMES_25_D1_V4_3: List[str] = [
    "d1_lumbar_curvature_poly_a",         # [ 1] 요추 2차 포물선 곡률
    "d1_lumbar_stiffness_sd",             # [ 2] 전 구간 요추 각도 변동성(안정성)
    "d1_spine_lateral_deviation_max",     # [ 3] 척추 좌우 횡방향 이탈 최대치
    "d1_trunk_flex_speed_max",            # [ 4] 상체 굴곡 속도 최대 피크
    "d1_cervical_min",                    # [ 5] 경추-흉추 최소 각도
    "d1_signed_thoracolumbar_delta",      # [ 6] 흉요추 상대 각도 최대 변화폭
    "d1_active_rom",                      # [ 7] 바벨/손목 수직 가동범위
    "d1_hip_to_knee_dist_ratio",          # [ 8] 고관절-무릎 수평 거리 비율
    "d1_cobb_mean_active",                # [ 9] 능동 리프팅 구간 요추 Cobb각 평균
    "d1_shoulder_hip_twist_max",          # [10] 어깨선-골반선 3D 비틀림 각도
    "d1_lateral_asymmetry_integral",      # [11] 척추 좌우 불균형 적분값
    "d1_lumbar_flex_acceleration_peak",   # [12] 요추 굴곡 각가속도 피크
    "d1_torso_pitch_standing",            # [13] 기립 자세 체간 전경각
    "d1_view_score_normal_z",             # [14] 체간 정면 법선벡터 Z축 투영
    "d1_lumbar_shear_stress_proxy",       # [15] 상체 질량 중심 모멘트암 기반 요추 전단력 지표
    "d1_lockout_hyperextension",          # [16] 피니시 락아웃 시 요추 과신전 오프셋
    "d1_spine_femur_sync_ratio",          # [17] 상체 펴짐 대 무릎 펴짐 속도비
    "d1_torso_pitch_rom",                 # [18] 상체 굴곡 각도 변화 진폭 (Max - Min)
    "d1_knee_valgus_ratio",               # [19] 무릎 간 거리 대 발목 간 거리 비 (d4 외반 지표)
    "d1_view_occlusion_factor",           # [20] 체간 시선축 투영 가림 지표 (후방 시점 깊이 왜곡)
    "d1_view_compensated_poly_a",         # [21] ★ 핵심: 후방 시점 가림 보정 3D 공간 만곡
    "d1_res_lumbar_poly_decoupled",       # [22] ★ 핵심: 골반/무릎 회전 제거 잔차 척추 곡률
    "d1_ratio_arm_to_torso",              # [23] 팔 길이 대 체간 길이 비
    "d1_ratio_femur_to_tibia",            # [24] 대퇴 대 하퇴 길이 비
    "d1_ratio_leg_to_torso"               # [25] 다리 전체 대 체간 길이 비
]

# 원본 40D 인덱스 매핑
DIET_25_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 15, 16, 18, 22, 26, 27, 28, 31, 32, 33, 36, 37, 38, 39]


def extract_morphology_and_crosstalk_ratios(kps_64: np.ndarray) -> np.ndarray:
    """피험자 사지 비율(6D) + 무릎 외반 비율 + 골반 회전도 (총 8D)"""
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

    knee_w = np.linalg.norm(kps_64[:, MP['LKNE']] - kps_64[:, MP['RKNE']], axis=-1)
    ank_w = np.linalg.norm(kps_64[:, MP['LANK']] - kps_64[:, MP['RANK']], axis=-1)
    valgus_ratio = float(np.mean(knee_w / np.maximum(1e-4, ank_w)))

    sh_vec = kps_64[:, MP['LSH']] - kps_64[:, MP['RSH']]
    hip_vec = kps_64[:, MP['LHIP']] - kps_64[:, MP['RHIP']]
    twist = np.degrees(np.arccos(np.clip(
        np.sum(sh_vec * hip_vec, axis=-1) / (np.linalg.norm(sh_vec, axis=-1) * np.linalg.norm(hip_vec, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    twist_mean = float(np.mean(twist))

    return np.array([
        arm_len / torso_len,
        femur_len / max(1e-4, tibia_len),
        leg_len / torso_len,
        arm_len / max(1e-4, leg_len),
        sh_w / max(1e-4, hip_w),
        torso_len,
        valgus_ratio,
        twist_mean
    ], dtype=np.float32)


class MorphologyResidualizerV4_3:
    """골반/무릎 회전 제거 잔차 척추 곡률 전용 경량 잔차화기"""
    def __init__(self, alpha: float = 15.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.ridge_curv = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(self, cond_train_normal: np.ndarray, curv_normal: np.ndarray):
        Z = self.scaler.fit_transform(cond_train_normal)
        self.ridge_curv.fit(Z, curv_normal)
        self.is_fitted = True
        return self

    def transform(self, cond: np.ndarray, curv_mean: float) -> float:
        if not self.is_fitted:
            return 0.0
        Z = self.scaler.transform(cond.reshape(1, -1))
        exp_curv = float(self.ridge_curv.predict(Z)[0])
        return curv_mean - exp_curv


def extract_d1_v4_3_diet_features_25d(
    kps_33_64f: np.ndarray,
    spine_9_64f: Optional[np.ndarray] = None,
    residualizer: Optional[MorphologyResidualizerV4_3] = None
) -> np.ndarray:
    """
    단일 64프레임 시퀀스로부터 25D 정예 다이어트 피처 직접 추출
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

    # 2. SpinePose 9분절 3D 좌표
    if spine_9_64f is not None:
        s_pts = spine_9_64f
    else:
        s_pts = np.zeros((T, 9, 3), dtype=np.float32)
        s_pts[:, 0] = mid_hip
        for j in range(1, 9):
            s_pts[:, j] = mid_hip + (j / 8.0) * (mid_sh - mid_hip)

    # 3. 요추 2차 다항 곡률 poly_a
    z_s1 = np.sum((s_pts[:, 0] - mid_hip) * z_hat, axis=-1)
    z_l5 = np.sum((s_pts[:, 1] - mid_hip) * z_hat, axis=-1)
    z_l3 = np.sum((s_pts[:, 2] - mid_hip) * z_hat, axis=-1)
    z_l1 = np.sum((s_pts[:, 3] - mid_hip) * z_hat, axis=-1)
    z_t8 = np.sum((s_pts[:, 4] - mid_hip) * z_hat, axis=-1)

    lumbar_curves = np.column_stack([z_s1, z_l5, z_l3, z_l1, z_t8])
    x_coords = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

    poly_a_series = np.zeros(T, dtype=np.float32)
    for t_idx in range(T):
        p = np.polyfit(x_coords, lumbar_curves[t_idx], deg=2)
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

    # 경추-흉추 각도
    v_cerv = s_pts[:, 8] - s_pts[:, 6]
    v_thor = s_pts[:, 5] - s_pts[:, 4]
    dot_ct = np.sum(v_cerv * v_thor, axis=-1)
    norm_ct = (np.linalg.norm(v_cerv, axis=-1) * np.linalg.norm(v_thor, axis=-1)) + 1e-7
    ct_angles = np.degrees(np.arccos(np.clip(dot_ct / norm_ct, -1.0, 1.0)))
    f5_cervical_min = float(np.min(ct_angles)) if spine_9_64f is not None else 150.0

    # 흉요추 상대 각도 변화폭
    f6_signed_delta = float(np.max(poly_a_series) - np.min(poly_a_series))

    # 4. 위상 및 가동범위
    wrist_y = mid_wrist[:, 1]
    wrist_y_s = gaussian_filter1d(wrist_y, sigma=1.2)
    f7_active_rom = float(np.ptp(wrist_y_s))

    hip_knee_h = np.abs(mid_hip[:, 0] - mid_knee[:, 0])
    f8_hk_dist = float(np.mean(hip_knee_h))

    # 5. 요추 Cobb 각도 평균 (단일 대표치)
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
    f9_cobb_mean = float(np.mean(cobb_series))

    # 어깨-골반 비틀림
    sh_vec = l_sh - r_sh
    hip_vec = l_hip - r_hip
    twist = np.degrees(np.arccos(np.clip(
        np.sum(sh_vec * hip_vec, axis=-1) / (np.linalg.norm(sh_vec, axis=-1) * np.linalg.norm(hip_vec, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    f10_twist_max = float(np.max(twist))
    f11_asym_integ = float(np.mean(twist))

    f12_flex_accel = float(np.max(np.abs(np.gradient(torso_pitch_vel, dt))))
    f13_pitch_standing = float(np.min(torso_pitch))

    f14_view_z = float(np.mean(z_hat[:, 2]))
    f15_shear = float(np.mean(np.sin(np.radians(torso_pitch))))

    # 락아웃 과신전
    finish_idx = int(np.argmin(wrist_y_s))
    f16_lockout = float(max(0.0, poly_a_series[finish_idx] - 0.05))

    # 상체-무릎 협응 속도비
    knee_angle = np.degrees(np.arccos(np.clip(
        np.sum((mid_hip - mid_knee) * (mid_ank - mid_knee), axis=-1) /
        (np.linalg.norm(mid_hip - mid_knee, axis=-1) * np.linalg.norm(mid_ank - mid_knee, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    knee_ext_vel = np.maximum(0.0, -np.gradient(knee_angle, dt))
    torso_ext_vel = np.maximum(0.0, -np.gradient(torso_pitch, dt))
    f17_sync_ratio = float(np.mean(torso_ext_vel[:finish_idx + 1]) / (np.mean(knee_ext_vel[:finish_idx + 1]) + 1e-5))

    f18_pitch_rom = float(np.ptp(torso_pitch))

    # 무릎 외반 비율 (d4 디커플링)
    knee_dist = np.linalg.norm(l_knee - r_knee, axis=-1)
    ank_dist = np.linalg.norm(l_ank - r_ank, axis=-1)
    valgus_series = knee_dist / np.maximum(1e-4, ank_dist)
    floor_break_idx = int(np.clip(np.argmax(wrist_y_s), 2, T - 12))
    setup_start = max(0, floor_break_idx - 4)
    f19_knee_valgus = float(np.mean(valgus_series[setup_start:finish_idx + 1]))

    # 후방 시점 가림 지표
    f20_view_occlusion = float(np.mean(np.abs(y_hat[:, 2])))

    # 3D 공간 만곡 보정 곡률 (★ 핵심)
    v_c7_s1 = s_pts[:, 6] - s_pts[:, 0]
    chord_len = np.linalg.norm(v_c7_s1, axis=-1) + 1e-6
    l3_disp = np.linalg.norm(np.cross(v_c7_s1, s_pts[:, 2] - s_pts[:, 0]), axis=-1) / chord_len
    f21_view_comp_curv = float(np.mean(l3_disp))

    # 잔차화 척추 곡률 (★ 핵심)
    cond_8d = extract_morphology_and_crosstalk_ratios(kps_33_64f)
    if residualizer is not None and residualizer.is_fitted:
        f22_res_curv = float(residualizer.transform(cond_8d, f1_poly_a))
    else:
        f22_res_curv = 0.0

    # 사지 비율 3D
    f23_arm_torso = float(cond_8d[0])
    f24_femur_tibia = float(cond_8d[1])
    f25_leg_torso = float(cond_8d[2])

    feats_25d = np.array([
        f1_poly_a, f2_stiffness_sd, f3_lat_dev, f4_flex_speed, f5_cervical_min, f6_signed_delta,
        f7_active_rom, f8_hk_dist, f9_cobb_mean, f10_twist_max, f11_asym_integ, f12_flex_accel,
        f13_pitch_standing, f14_view_z, f15_shear, f16_lockout, f17_sync_ratio, f18_pitch_rom,
        f19_knee_valgus, f20_view_occlusion, f21_view_comp_curv, f22_res_curv,
        f23_arm_torso, f24_femur_tibia, f25_leg_torso
    ], dtype=np.float32)

    return np.nan_to_num(feats_25d, nan=0.0, posinf=1.0, neginf=-1.0)


def extract_temporal_24d_with_spine(kps_33: np.ndarray, spine_9: Optional[np.ndarray] = None) -> np.ndarray:
    """SpinePose 9대 척추 분절을 포함한 24D 시계열 텐서 (64, 24)"""
    T = 64
    mid_hip = (kps_33[:, MP['LHIP']] + kps_33[:, MP['RHIP']]) / 2.0
    mid_sh = (kps_33[:, MP['LSH']] + kps_33[:, MP['RSH']]) / 2.0
    mid_knee = (kps_33[:, MP['LKNE']] + kps_33[:, MP['RKNE']]) / 2.0
    mid_ank = (kps_33[:, MP['LANK']] + kps_33[:, MP['RANK']]) / 2.0

    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_sh = kps_33[:, MP['LSH']] - kps_33[:, MP['RSH']]
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)
    x_hat = np.cross(y_hat, z_hat)

    if spine_9 is not None:
        s_pts = spine_9
    else:
        s_pts = np.zeros((T, 9, 3), dtype=np.float32)
        s_pts[:, 0] = mid_hip
        for j in range(1, 9):
            s_pts[:, j] = mid_hip + (j / 8.0) * (mid_sh - mid_hip)

    channels = []
    key_spine_indices = [0, 1, 2, 3, 4, 6]
    for sp_idx in key_spine_indices:
        disp_z = np.sum((s_pts[:, sp_idx] - mid_hip) * z_hat, axis=-1)
        channels.append(disp_z)

    channels.append(np.sum((s_pts[:, 2] - mid_hip) * x_hat, axis=-1))
    channels.append(np.sum((s_pts[:, 4] - mid_hip) * x_hat, axis=-1))

    torso_pitch = np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0))
    channels.append(torso_pitch)
    channels.append(np.gradient(torso_pitch))

    knee_v1 = mid_hip - mid_knee
    knee_v2 = mid_ank - mid_knee
    knee_ang = np.arccos(np.clip(np.sum(knee_v1 * knee_v2, axis=-1) / (np.linalg.norm(knee_v1, axis=-1) * np.linalg.norm(knee_v2, axis=-1) + 1e-7), -1.0, 1.0))
    channels.append(knee_ang)
    channels.append(np.gradient(knee_ang))

    mid_wrist = (kps_33[:, MP['LWR']] + kps_33[:, MP['RWR']]) / 2.0
    w_y = mid_wrist[:, 1]
    channels.append(w_y)
    channels.append(np.gradient(w_y))
    channels.append(np.gradient(np.gradient(w_y)))

    knee_w = np.linalg.norm(kps_33[:, MP['LKNE']] - kps_33[:, MP['RKNE']], axis=-1)
    channels.append(knee_w)
    channels.append(np.gradient(knee_w))
    channels.append(np.sum(v_torso * z_hat, axis=-1))

    channels.append(np.linalg.norm(s_pts[:, 6] - s_pts[:, 0], axis=-1))
    channels.append(np.linalg.norm(s_pts[:, 3] - s_pts[:, 1], axis=-1))
    channels.append(y_hat[:, 2])
    channels.append(z_hat[:, 2])
    channels.append(disp_z - np.mean(disp_z))
    channels.append(gaussian_filter1d(torso_pitch, sigma=1.0))

    tensor_24d = np.column_stack(channels[:24])
    return np.nan_to_num(tensor_24d, nan=0.0).astype(np.float32)


class BabelDeadliftD1ModelV4_3:
    """
    vidio-health-trainer
    공식 최신 피처 다이어트 프로덕션 바벨 데드리프트 d1 모델 (v4.3 Diet)
    - 25D 정예 생체역학 피처셋 (다중공선성 및 저해 피처 15개 제거 최적화)
    - Tri-Ensemble (24D TCN + 25D LightGBM + 25D CatBoost)
    - 3-State 선택적 판정 [tau_norm=0.26, tau_def=0.45]
    """
    def __init__(
        self,
        weights_dir: Optional[str] = None,
        w_gbdt: float = 0.5,
        w_tcn: float = 0.5,
        decision_threshold: float = 0.35,
        tau_normal: float = 0.26,
        tau_defect: float = 0.45
    ):
        self.w_gbdt = w_gbdt
        self.w_tcn = w_tcn
        self.decision_threshold = decision_threshold
        self.tau_normal = tau_normal
        self.tau_defect = tau_defect

        self.clf_lgb = None
        self.clf_cat = None
        self.residualizer: Optional[MorphologyResidualizerV4_3] = None
        self.tcn_models: List[nn.Module] = []
        self.is_fitted = False

        if weights_dir is None:
            weights_dir = os.path.join(CURRENT_DIR, "weights")
        self.weights_dir = weights_dir

        self._try_load_weights()

    def _try_load_weights(self):
        pkl_path = os.path.join(self.weights_dir, "d1_v4_3_gbdt_models.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
                self.clf_lgb = d["clf_lgb"]
                self.clf_cat = d["clf_cat"]
                self.residualizer = d.get("residualizer", None)
                self.decision_threshold = d.get("decision_threshold", 0.35)

            tcn_files = sorted(glob.glob(os.path.join(self.weights_dir, "d1_v4_3_tcn_fold*.pth")))
            self.tcn_models = []
            for tf in tcn_files:
                m = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
                m.load_state_dict(torch.load(tf, map_location="cpu"))
                m.eval()
                self.tcn_models.append(m)

            self.is_fitted = (self.clf_lgb is not None and len(self.tcn_models) > 0)
            if self.is_fitted:
                print(f"[*] BabelDeadliftD1ModelV4_3 (Diet 25D) 공식 가중치 로드 완료")

    def extract_features_25d(
        self,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return extract_d1_v4_3_diet_features_25d(kps_33_64f, spine_9_64f, self.residualizer)

    def predict_proba_from_seq(
        self,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> float:
        if not self.is_fitted:
            raise RuntimeError("D1 v4.3 모델 가중치가 로드되지 않았습니다. train_and_eval_d1_v4_3_diet.py를 먼저 실행하세요.")

        f25 = self.extract_features_25d(kps_33_64f, spine_9_64f)[None, :]
        p_lgb = float(self.clf_lgb.predict_proba(f25)[0, 1])
        p_cat = float(self.clf_cat.predict_proba(f25)[0, 1])
        p_gbdt = 0.5 * p_lgb + 0.5 * p_cat

        t24 = extract_temporal_24d_with_spine(kps_33_64f, spine_9_64f)[None, ...]
        x_t = torch.tensor(t24.transpose(0, 2, 1), dtype=torch.float32)

        p_tcns = []
        with torch.no_grad():
            for m in self.tcn_models:
                logits = m(x_t).squeeze(-1)
                p_tcns.append(float(torch.sigmoid(logits).item()))
        p_tcn = float(np.mean(p_tcns))

        return float(self.w_gbdt * p_gbdt + self.w_tcn * p_tcn)

    def predict_selective(
        self,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None,
        tau_normal: Optional[float] = None,
        tau_defect: Optional[float] = None
    ) -> Dict[str, Any]:
        tau_norm = tau_normal if tau_normal is not None else self.tau_normal
        tau_def = tau_defect if tau_defect is not None else self.tau_defect

        f25 = self.extract_features_25d(kps_33_64f, spine_9_64f)
        active_rom = float(f25[6])
        pitch_rom = float(f25[17])

        if pitch_rom < 40.0 or active_rom < 0.25:
            return {
                "status": "UNDETERMINED",
                "defect_probability": None,
                "confidence": 0.0,
                "reason_code": "INSUFFICIENT_ROM",
                "message": "상체 숙임 각도나 바벨 이동폭이 너무 얕아(가동범위 부족) 척추 중립을 정확히 평가할 수 없습니다.",
                "action": "바벨을 정강이 아래까지 충분히 내린 뒤 골반 힌지를 사용하여 온전한 가동범위로 다시 수행해 주세요."
            }

        p_defect = self.predict_proba_from_seq(kps_33_64f, spine_9_64f)

        if p_defect <= tau_norm:
            status = "NORMAL"
            reason = "CLEAR_DECISION"
            msg = "데드리프트 중 요추와 흉추의 자연스러운 중립 곡선이 바르게 유지되었습니다."
            act = None
        elif p_defect >= tau_def:
            status = "DEFECT"
            reason = "CLEAR_DECISION"
            msg = "바벨 리프팅 시 요추가 말리거나(라운드백) 척추 중립이 위반되었습니다."
            act = "코어 복압을 강하게 유지하고, 가슴을 펴며 엉덩이 힌지를 사용해 들어 올리세요."
        else:
            status = "UNDETERMINED"
            reason = "MODEL_UNCERTAINTY"
            msg = "척추 굴곡 패턴이 경계 구간에 위치하여 안전을 위해 재측정을 권장합니다."
            act = "허리 각도가 잘 관측되도록 대각선 45도 측면(시점 B/D)에서 1회 다시 수행해 주세요."

        return {
            "status": status,
            "defect_probability": round(p_defect, 4),
            "confidence": round(min(1.0, 2.0 * abs(p_defect - 0.5)), 4),
            "reason_code": reason,
            "message": msg,
            "action": act
        }


BabelDeadliftD1Model = BabelDeadliftD1ModelV4_3
