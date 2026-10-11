# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Model Version: v4.2 (d1_model_v4_2_spinepose.py)
Status: ★ [v4.1 계승 및 3대 오분류 개선 전략 통합 모델]
         1) [전략 1] 가동범위(ROM) 적격성 모델링 & 스케일링 (피험자 4·5번 불완전 동작 라벨 노이즈 방어)
         2) [전략 2] 다중 결함 디커플링 (Cross-talk Decoupling: d4 무릎 모임 및 골반 비틀림 제거 잔차화)
         3) [전략 3] 후방 시점(시점 A, E) 투영 왜곡 보정 (View-Z Occlusion Compensation)
         - 40D 고순도 생체역학 정제 피처셋 구축
         - 24D 시계열 경량 Dilated TCN + 40D GBDT (LightGBM + CatBoost) 앙상블
         - 3-State 선택적 판정(Selective Classification) 최적화
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

FEATURE_NAMES_40_D1_V4_2: List[str] = [
    # [1~6] SpinePose 9분절 요추 곡률 및 척추 기하
    "d1_lumbar_curvature_poly_a",         # [1] 요추 2차 포물선 곡률 (라운드백 핵심 지표)
    "d1_lumbar_stiffness_sd",             # [2] 전 구간 요추 각도 유지 안정성(변동성)
    "d1_spine_lateral_deviation_max",     # [3] 척추 좌우 횡방향 이탈 최대치
    "d1_trunk_flex_speed_max",            # [4] 상체 굴곡 속도 최대 피크
    "d1_cervical_min",                    # [5] 경추-흉추 최소 각도 (목 꺾임 감지)
    "d1_signed_thoracolumbar_delta",      # [6] 흉요추 상대 각도 최대 변화폭

    # [7~12] 체간 로컬 기저계 및 생체역학 동역학
    "d1_active_rom",                      # [7] 골반 수직 가동범위 (손목 이동폭)
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
    "d1_torso_pitch_standing",            # [17] 기립 자세 체간 전경각
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

    # [29~34] 전략 1: 가동범위(ROM) 적격성 및 전략 3: 후방 시점 보정 피처
    "d1_torso_pitch_rom",                 # [29] 상체 굴곡 각도 변화 진폭 (Max - Min)
    "d1_rom_competence_score",            # [30] 종합 가동범위 적격성 점수 [0.0 ~ 1.0]
    "d1_rom_scaled_lumbar_poly_a",        # [31] ROM 적격성으로 스케일링된 요추 곡률
    "d1_knee_valgus_ratio",               # [32] 무릎 간 거리 대 발목 간 거리 비 (d4 외반 지표)
    "d1_view_occlusion_factor",           # [33] 체간 시선축 투영 가림 지표 (후방 시점 깊이 왜곡)
    "d1_view_compensated_poly_a",         # [34] 후방 시점 가림 보정 3D 척추 곡률

    # [35~40] 전략 2: 디커플링 잔차 및 사지 비율
    "d1_res_torso_pitch_deepest",         # [35] 체형+무릎모임 대비 셋업 체간각 잔차
    "d1_res_setup_lumbar_poly_a",         # [36] 체형+무릎모임 대비 셋업 요추 곡률 잔차
    "d1_res_lumbar_poly_decoupled",       # [37] 골반/무릎 회전 제거 잔차 척추 곡률
    "d1_ratio_arm_to_torso",              # [38] 팔 길이 대 체간 길이 비
    "d1_ratio_femur_to_tibia",            # [39] 대퇴 대 하퇴 길이 비
    "d1_ratio_leg_to_torso"               # [40] 다리 전체 대 체간 길이 비
]


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

    # 무릎 외반 지표 (동작 전 구간 평균)
    knee_w = np.linalg.norm(kps_64[:, MP['LKNE']] - kps_64[:, MP['RKNE']], axis=-1)
    ank_w = np.linalg.norm(kps_64[:, MP['LANK']] - kps_64[:, MP['RANK']], axis=-1)
    valgus_ratio = float(np.mean(knee_w / np.maximum(1e-4, ank_w)))

    # 골반 비틀림 각도
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


class MorphologyResidualizerV4_2:
    """체형 비율 및 다중 결함(무릎 모임, 골반 비틀림) 조건부 잔차화 변환기"""
    def __init__(self, alpha: float = 15.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.ridge_pitch = Ridge(alpha=alpha)
        self.ridge_poly = Ridge(alpha=alpha)
        self.ridge_curv = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(self, cond_train_normal: np.ndarray, pitch_normal: np.ndarray, poly_normal: np.ndarray, curv_normal: np.ndarray):
        Z = self.scaler.fit_transform(cond_train_normal)
        self.ridge_pitch.fit(Z, pitch_normal)
        self.ridge_poly.fit(Z, poly_normal)
        self.ridge_curv.fit(Z, curv_normal)
        self.is_fitted = True
        return self

    def transform(self, cond: np.ndarray, pitch_deepest: float, poly_a: float, curv_mean: float) -> Tuple[float, float, float]:
        if not self.is_fitted:
            return 0.0, 0.0, 0.0
        Z = self.scaler.transform(cond.reshape(1, -1))
        exp_pitch = float(self.ridge_pitch.predict(Z)[0])
        exp_poly = float(self.ridge_poly.predict(Z)[0])
        exp_curv = float(self.ridge_curv.predict(Z)[0])
        return pitch_deepest - exp_pitch, poly_a - exp_poly, curv_mean - exp_curv


def extract_d1_v4_2_features_34d_base(
    kps_33_64f: np.ndarray,
    spine_9_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    SpinePose 9분절 3D 좌표 기반 v4.2 34D 기본 생체역학 피처 추출
    (전략 1 ROM 적격성, 전략 2 무릎 외반, 전략 3 후방 시점 보정 피처 결합)
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

    # 2. SpinePose 9분절 3D 좌표 (S1=0, L5=1, L3=2, L1=3, T8=4, T3=5, C7=6, C4=7, C1=8)
    if spine_9_64f is not None:
        s_pts = spine_9_64f
    else:
        s_pts = np.zeros((T, 9, 3), dtype=np.float32)
        s_pts[:, 0] = mid_hip
        for j in range(1, 9):
            alpha = j / 8.0
            s_pts[:, j] = mid_hip + alpha * (mid_sh - mid_hip)

    # 3. 요추 2차 다항 곡률 poly_a 산출
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

    # 4. 데드리프트 위상 분석
    wrist_y = mid_wrist[:, 1]
    wrist_y_s = gaussian_filter1d(wrist_y, sigma=1.2)
    pelvis_y = mid_hip[:, 1]
    pelvis_y_s = gaussian_filter1d(pelvis_y, sigma=1.2)

    active_rom = float(np.ptp(wrist_y_s))
    f7_active_rom = active_rom
    f8_phase_speed = 1.0

    hip_knee_h = np.abs(mid_hip[:, 0] - mid_knee[:, 0])
    f9_hk_dist = float(np.mean(hip_knee_h))

    # 5. 요추 Cobb 각도 시계열 산출
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
    setup_start = max(0, floor_break_idx - 4)
    f24_setup_poly_a = float(np.mean(poly_a_series[setup_start:floor_break_idx + 1]))

    fb_end = min(T - 1, floor_break_idx + 8)
    f25_fb_cobb_delta = float(cobb_series[fb_end] - cobb_series[floor_break_idx])

    knee_y_level = np.mean(mid_knee[:, 1])
    knee_pass_idx = floor_break_idx + 4
    for k_t in range(floor_break_idx, min(T, floor_break_idx + 24)):
        if wrist_y_s[k_t] <= knee_y_level:
            knee_pass_idx = k_t
            break
    f26_knee_pass_cobb = float(cobb_series[knee_pass_idx])

    finish_idx = int(np.argmin(wrist_y_s))
    f27_lockout_hyperext = float(max(0.0, poly_a_series[finish_idx] - 0.05))

    knee_angle = np.degrees(np.arccos(np.clip(
        np.sum((mid_hip - mid_knee) * (mid_ank - mid_knee), axis=-1) /
        (np.linalg.norm(mid_hip - mid_knee, axis=-1) * np.linalg.norm(mid_ank - mid_knee, axis=-1) + 1e-7),
        -1.0, 1.0
    )))
    knee_ext_vel = np.maximum(0.0, -np.gradient(knee_angle, dt))
    torso_ext_vel = np.maximum(0.0, -np.gradient(torso_pitch, dt))
    f28_sync_ratio = float(np.mean(torso_ext_vel[:finish_idx + 1]) / (np.mean(knee_ext_vel[:finish_idx + 1]) + 1e-5))

    # -------------------------------------------------------------
    # [전략 1] 가동범위(ROM) 적격성 모델링
    # -------------------------------------------------------------
    pitch_rom = float(np.ptp(torso_pitch))
    f29_pitch_rom = pitch_rom

    # ROM 적격성 점수 [0.0 ~ 1.0]: 얕은 숙임(pitch_rom < 60° 또는 active_rom < 0.40m)일 때 0에 수렴
    score_pitch = np.clip((pitch_rom - 45.0) / 45.0, 0.0, 1.0)
    score_wrist = np.clip((active_rom - 0.30) / 0.25, 0.0, 1.0)
    f30_rom_score = float(score_pitch * score_wrist)

    # ROM 적격성 스케일링된 척추 곡률 (불완전 동작 시 가짜 결함 억제)
    f31_rom_scaled_curv = float(f1_poly_a * f30_rom_score)

    # -------------------------------------------------------------
    # [전략 2] 무릎 외반(Knee Valgus, d4) 디커플링 피처
    # -------------------------------------------------------------
    knee_dist = np.linalg.norm(l_knee - r_knee, axis=-1)
    ank_dist = np.linalg.norm(l_ank - r_ank, axis=-1)
    valgus_series = knee_dist / np.maximum(1e-4, ank_dist)
    f32_knee_valgus = float(np.mean(valgus_series[setup_start:finish_idx + 1]))

    # -------------------------------------------------------------
    # [전략 3] 후방 시점(시점 A, E) 투영 가림 보정
    # -------------------------------------------------------------
    # 상체가 카메라 시선(Z축)으로 겹치는 가림 지표
    f33_view_occlusion = float(np.mean(np.abs(y_hat[:, 2])))

    # 3D 공간 만곡 보정 곡률 (Z축 투영 왜곡을 억제하는 3D 스플라인 곡률)
    v_c7_s1 = s_pts[:, 6] - s_pts[:, 0]
    chord_len = np.linalg.norm(v_c7_s1, axis=-1) + 1e-6
    # 중간 지점(L3)의 3D 이탈 거리
    l3_disp = np.linalg.norm(np.cross(v_c7_s1, s_pts[:, 2] - s_pts[:, 0]), axis=-1) / chord_len
    f34_view_comp_curv = float(np.mean(l3_disp))

    feats_34d = np.array([
        f1_poly_a, f2_stiffness_sd, f3_lat_dev, f4_flex_speed, f5_cervical_min, f6_signed_delta,
        f7_active_rom, f8_phase_speed, f9_hk_dist, f10_cobb_mean, f11_cobb_max, f12_cobb_range,
        f13_twist_max, f14_asym_integ, f15_pelvis_tilt, f16_flex_accel, f17_pitch_standing, f18_pitch_deepest,
        f19_view_z, f20_conf, f21_energy, f22_symm, f23_shear,
        f24_setup_poly_a, f25_fb_cobb_delta, f26_knee_pass_cobb, f27_lockout_hyperext, f28_sync_ratio,
        f29_pitch_rom, f30_rom_score, f31_rom_scaled_curv, f32_knee_valgus, f33_view_occlusion, f34_view_comp_curv
    ], dtype=np.float32)

    return np.nan_to_num(feats_34d, nan=0.0, posinf=1.0, neginf=-1.0)


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

    # 24개 채널 구성
    channels = []
    # 1~6: SpinePose 6개 분절(S1, L5, L3, L1, T8, C7)의 체간 Z축 로컬 변위
    key_spine_indices = [0, 1, 2, 3, 4, 6]
    for sp_idx in key_spine_indices:
        disp_z = np.sum((s_pts[:, sp_idx] - mid_hip) * z_hat, axis=-1)
        channels.append(disp_z)

    # 7~10: L3, T8의 X축(횡방향) 변위 및 C7-C1 각도
    channels.append(np.sum((s_pts[:, 2] - mid_hip) * x_hat, axis=-1))
    channels.append(np.sum((s_pts[:, 4] - mid_hip) * x_hat, axis=-1))

    # 9~12: 고관절, 무릎, 발목 각도 및 체간 피치각
    torso_pitch = np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0))
    channels.append(torso_pitch)
    channels.append(np.gradient(torso_pitch))

    knee_v1 = mid_hip - mid_knee
    knee_v2 = mid_ank - mid_knee
    knee_ang = np.arccos(np.clip(np.sum(knee_v1 * knee_v2, axis=-1) / (np.linalg.norm(knee_v1, axis=-1) * np.linalg.norm(knee_v2, axis=-1) + 1e-7), -1.0, 1.0))
    channels.append(knee_ang)
    channels.append(np.gradient(knee_ang))

    # 13~18: 손목 Y 좌표 및 속도, 가속도, 무릎 거리
    mid_wrist = (kps_33[:, MP['LWR']] + kps_33[:, MP['RWR']]) / 2.0
    w_y = mid_wrist[:, 1]
    channels.append(w_y)
    channels.append(np.gradient(w_y))
    channels.append(np.gradient(np.gradient(w_y)))

    knee_w = np.linalg.norm(kps_33[:, MP['LKNE']] - kps_33[:, MP['RKNE']], axis=-1)
    channels.append(knee_w)
    channels.append(np.gradient(knee_w))
    channels.append(np.sum(v_torso * z_hat, axis=-1))

    # 19~24: SpinePose 분절 간 상대 거리 및 회전
    channels.append(np.linalg.norm(s_pts[:, 6] - s_pts[:, 0], axis=-1))
    channels.append(np.linalg.norm(s_pts[:, 3] - s_pts[:, 1], axis=-1))
    channels.append(y_hat[:, 2])  # 시점 Z 성분
    channels.append(z_hat[:, 2])
    channels.append(disp_z - np.mean(disp_z))
    channels.append(gaussian_filter1d(torso_pitch, sigma=1.0))

    tensor_24d = np.column_stack(channels[:24])  # (64, 24)
    return np.nan_to_num(tensor_24d, nan=0.0).astype(np.float32)



class BabelDeadliftD1ModelV4_2:
    """
    vidio-health-trainer
    공식 프로덕션 바벨 데드리프트 d1 (척추 중립 위반) v4.2 모델
    - 40D 고순도 생체역학 정제 피처 (ROM 적격성 + 다중결함 디커플링 + 후방시점 보정)
    - 24D 시계열 경량 Dilated TCN + GBDT (LightGBM 50% + CatBoost 50%) Tri-Ensemble
    - 3-State 선택적 판정 (NORMAL / DEFECT / UNDETERMINED)
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
        self.residualizer: Optional[MorphologyResidualizerV4_2] = None
        self.tcn_models: List[nn.Module] = []
        self.is_fitted = False

        if weights_dir is None:
            weights_dir = os.path.join(CURRENT_DIR, "weights")
        self.weights_dir = weights_dir

        self._try_load_weights()

    def _try_load_weights(self):
        pkl_path = os.path.join(self.weights_dir, "d1_v4_2_gbdt_models.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
                self.clf_lgb = d["clf_lgb"]
                self.clf_cat = d["clf_cat"]
                self.residualizer = d.get("residualizer", None)
                self.decision_threshold = d.get("decision_threshold", 0.35)

            tcn_files = sorted(glob.glob(os.path.join(self.weights_dir, "d1_v4_2_tcn_fold*.pth")))
            self.tcn_models = []
            for tf in tcn_files:
                m = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
                m.load_state_dict(torch.load(tf, map_location="cpu"))
                m.eval()
                self.tcn_models.append(m)

            self.is_fitted = (self.clf_lgb is not None and len(self.tcn_models) > 0)
            if self.is_fitted:
                print(f"[*] BabelDeadliftD1ModelV4_2 공식 가중치 로드 완료 (GBDT + {len(self.tcn_models)} TCN Folds)")

    def extract_features_40d(
        self,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """단일 64프레임 시퀀스로부터 40D 정제 피처 추출"""
        f34 = extract_d1_v4_2_features_34d_base(kps_33_64f, spine_9_64f)
        cond = extract_morphology_and_crosstalk_ratios(kps_33_64f)

        if self.residualizer is not None and self.residualizer.is_fitted:
            res_pitch, res_poly, res_curv = self.residualizer.transform(
                cond, float(f34[17]), float(f34[23]), float(f34[0])
            )
        else:
            res_pitch, res_poly, res_curv = 0.0, 0.0, 0.0

        f40 = np.concatenate([
            f34,
            np.array([res_pitch, res_poly, res_curv], dtype=np.float32),
            cond[:3]
        ])
        return f40

    def predict_proba_from_seq(
        self,
        kps_33_64f: np.ndarray,
        spine_9_64f: Optional[np.ndarray] = None
    ) -> float:
        """단일 64프레임 시퀀스로부터 d1 결함 확률 p_defect 산출"""
        if not self.is_fitted:
            raise RuntimeError("D1 v4.2 모델 가중치가 로드되지 않았습니다. train_and_eval_d1_v4_2_spinepose.py를 먼저 실행하세요.")

        f40 = self.extract_features_40d(kps_33_64f, spine_9_64f)[None, :]
        p_lgb = float(self.clf_lgb.predict_proba(f40)[0, 1])
        p_cat = float(self.clf_cat.predict_proba(f40)[0, 1])
        p_gbdt = 0.5 * p_lgb + 0.5 * p_cat

        # TCN
        t24 = extract_temporal_24d_with_spine(kps_33_64f, spine_9_64f)[None, ...]
        x_t = torch.tensor(t24.transpose(0, 2, 1), dtype=torch.float32)

        p_tcns = []
        with torch.no_grad():
            for m in self.tcn_models:
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
        - 가동범위(ROM) 부족 시 안전 게이팅
        - 최적화된 3-State 안전 임계치 [tau_norm=0.26, tau_def=0.45]
        """
        tau_norm = tau_normal if tau_normal is not None else self.tau_normal
        tau_def = tau_defect if tau_defect is not None else self.tau_defect

        f40 = self.extract_features_40d(kps_33_64f, spine_9_64f)
        active_rom = float(f40[6])
        pitch_rom = float(f40[28])

        # 가동범위 적격성 사전 안전망
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


# 최신 프로덕션 공식 별칭 Alias
BabelDeadliftD1Model = BabelDeadliftD1ModelV4_2
