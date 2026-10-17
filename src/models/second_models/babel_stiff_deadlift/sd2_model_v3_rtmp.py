# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 2: sd2 (척추의 중립 위반 / Spine Neutrality Violation)
Version: v3.0 RTMPose Optimized (sd2_model_v3_rtmp.py)
Purpose: RTMPose-m (Halpe26) 2D 고정밀 관절 및 체간/골반 해부학적 랜드마크에 최적화된
         Two-Tier Tri-Ensemble & 3-State 선택적 진단 분류기
Rule: AGENTS.md 규정 및 공식 5대 시점 체계(시점 A~E), 순수 비전 64프레임 입력 준수
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

# CURRENT_DIR 및 프로젝트 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "sd2_model_v3_rtmp_weights.pkl")


class TemporalBlock(nn.Module):
    """Dilated Residual Convolutional Block"""
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.bn2(self.conv2(out)))
        return self.relu2(out + res)


class SD2DilatedTCN(nn.Module):
    """sd2 전용 경량 Dilated TCN 신경망 (Attention Pooling)"""
    def __init__(self, in_channels: int = 24, hidden_dim: int = 48, dropout: float = 0.25):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block1 = TemporalBlock(hidden_dim, hidden_dim, dilation=1, dropout=dropout)
        self.block2 = TemporalBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout)
        self.block3 = TemporalBlock(hidden_dim, hidden_dim, dilation=4, dropout=dropout)

        # Attention Pooling
        self.att_conv = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        att = F.softmax(self.att_conv(h), dim=-1)
        pooled = torch.sum(h * att, dim=-1)
        logits = self.head(pooled)
        return logits

# Halpe26 키포인트 인덱스
HALPE = dict(
    NOSE=0, LEYE=1, REYE=2, LEAR=3, REAR=4,
    LSH=5, RSH=6, LELB=7, RELB=8, LWR=9, RWR=10,
    LHIP=11, RHIP=12, LKNE=13, RKNE=14, LANK=15, RANK=16,
    HEAD=17, NECK=18, HIP=19,
    LBIG_TOE=20, RBIG_TOE=21, LHEEL=24, RHEEL=25
)

# MediaPipe 33 키포인트 인덱스 (호환용)
MP = dict(
    NOSE=0, LEAR=7, REAR=8,
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28
)

FEATURE_NAMES_30_SD2_RTMP: List[str] = [
    # [1~6] 2D 시상면 척추 만곡 및 요추 동역학
    "stiff_lumbar_poly_a_2d",              # [ 1] 요추 2차 포물선 곡률 계수 평균 (굴곡 정도)
    "stiff_lumbar_stiffness_sd_2d",        # [ 2] 전 구간 요추 각도 변동성 (척추 강성 안정도)
    "stiff_spine_lateral_dev_max_2d",      # [ 3] 척추 중심선 횡방향 이탈 최대치 (좌우 비대칭)
    "stiff_trunk_flex_speed_max_2d",       # [ 4] 상체 굴곡 속도 최대 피크
    "stiff_cervical_min_2d",               # [ 5] 경추-흉추 최소 각도 (목-체간 정렬)
    "stiff_active_rom_2d",                 # [ 6] 손목 기준 수직 가동범위 (ROM)
    "stiff_cobb_mean_2d",                  # [ 7] 요추 Cobb각 프록시 평균
    "stiff_shoulder_hip_twist_max_2d",     # [ 8] 어깨선-골반선 2D 비틀림 각도
    "stiff_shear_stress_proxy_2d",         # [ 9] 상체 모멘트암 기반 요추 전단력 프록시
    "stiff_pitch_rom_2d",                  # [10] 상체 굴곡 각도 진폭 (Max - Min)
    "stiff_neck_hip_alignment_dev",        # [11] 목(Neck)-골반(Hip) 중심축 정렬 이탈도
    "stiff_spine_chord_disp_mean",         # [12] 척추 시현 변위(Chord-Displacement) 평균
    "stiff_ratio_arm_to_torso_2d",         # [13] 팔 길이 대 체간 길이 비
    "stiff_ratio_femur_to_tibia_2d",       # [14] 대퇴 대 하퇴 길이 비
    "stiff_ratio_leg_to_torso_2d",         # [15] 다리 전체 대 체간 길이 비
    "stiff_knee_fixation_sd_2d",           # [16] 무릎 관절 각도 고정 안정도
    "stiff_torso_pelvis_offset_2d",        # [17] 체간 중심과 골반 중심의 횡방향 상대 편차
    "stiff_sagittal_stiffness_bottom_2d",  # [18] 최하사점 전환 시 척추 형태 보존 강성
    # [19~23] ★ 바벨 및 체간 좌우 기울임 특화 (Key 211, 212, 213, 214 대응)
    "stiff_barbell_roll_max_2d",           # [19] 양 손목(바벨) 수평 기울기 최대 각도
    "stiff_barbell_roll_mean_2d",          # [20] 양 손목(바벨) 수평 기울기 평균 각도
    "stiff_barbell_lateral_drift_2d",      # [21] 골반 중심 대비 바벨(손목 중점) 횡방향 비대칭 편차
    "stiff_shoulder_roll_max_2d",          # [22] 어깨선 수평 기울기 최대 편차
    "stiff_hip_roll_max_2d",               # [23] 골반선 수평 기울기 최대 편차
    # [24~27] ★ 허리 과신전(Hyperextension / Lockout Lean) 특화 (Key 215 대응)
    "stiff_lockout_posterior_lean_2d",     # [24] 기립 락아웃 시 상체 후방 과신전 경사각
    "stiff_lockout_pelvic_sway_2d",        # [25] 기립 락아웃 시 발목 대비 골반 전방 스웨이 변위
    "stiff_lumbar_lordosis_peak_2d",       # [26] 요추 전만 과신전 피크
    "stiff_lockout_reversal_accel_2d",     # [27] 락아웃 정점 도달 시 반대 방향 역감속 충격량
    # [28~30] ★ 등 굽힘(Round Back / Thoracic Kyphosis) 특화 (Key 216 대응)
    "stiff_thoracic_kyphosis_peak_2d",     # [28] 상부 흉추 곡률 및 어깨 전방 말림 피크
    "stiff_c_curve_depth_2d",              # [29] Neck-Hip 직선 대비 흉요추 만곡 최대 함몰 깊이
    "stiff_roundback_inflexion_integral_2d"# [30] 하강 최저 변곡점 부근의 흉추 굴곡 시간 적분값
]


def extract_sd2_rtmp_features(kps_seq: np.ndarray) -> np.ndarray:
    """
    RTMPose Halpe26 (또는 호환 MP33) 시퀀스로부터 sd2 전용 30D 생체역학 피처 추출
    - kps_seq: (64, 26, 3) 또는 (64, 33, 3)
    """
    T = kps_seq.shape[0]
    dt = 3.0 / max(1, T - 1)

    if kps_seq.shape[1] == 26:
        # Halpe26
        l_sh = kps_seq[:, HALPE['LSH'], :2]
        r_sh = kps_seq[:, HALPE['RSH'], :2]
        l_hip = kps_seq[:, HALPE['LHIP'], :2]
        r_hip = kps_seq[:, HALPE['RHIP'], :2]
        l_kne = kps_seq[:, HALPE['LKNE'], :2]
        r_kne = kps_seq[:, HALPE['RKNE'], :2]
        l_ank = kps_seq[:, HALPE['LANK'], :2]
        r_ank = kps_seq[:, HALPE['RANK'], :2]
        l_wr = kps_seq[:, HALPE['LWR'], :2]
        r_wr = kps_seq[:, HALPE['RWR'], :2]
        neck = kps_seq[:, HALPE['NECK'], :2]
        pelvis = kps_seq[:, HALPE['HIP'], :2]
    else:
        # MP33 호환 모드
        l_sh = kps_seq[:, MP['LSH'], :2]
        r_sh = kps_seq[:, MP['RSH'], :2]
        l_hip = kps_seq[:, MP['LHIP'], :2]
        r_hip = kps_seq[:, MP['RHIP'], :2]
        l_kne = kps_seq[:, MP['LKNE'], :2]
        r_kne = kps_seq[:, MP['RKNE'], :2]
        l_ank = kps_seq[:, MP['LANK'], :2]
        r_ank = kps_seq[:, MP['RANK'], :2]
        l_wr = kps_seq[:, MP['LWR'], :2]
        r_wr = kps_seq[:, MP['RWR'], :2]
        neck = (l_sh + r_sh) / 2.0
        pelvis = (l_hip + r_hip) / 2.0

    mid_sh = (l_sh + r_sh) / 2.0
    mid_hip = pelvis if np.linalg.norm(pelvis) > 1e-4 else (l_hip + r_hip) / 2.0
    mid_kne = (l_kne + r_kne) / 2.0
    mid_ank = (l_ank + r_ank) / 2.0
    mid_wr = (l_wr + r_wr) / 2.0

    # 1. 2D 체간 기저계 및 길이
    v_torso = neck - mid_hip
    torso_len = np.linalg.norm(v_torso, axis=-1) + 1e-6
    u_torso = v_torso / torso_len[:, None]          # 체간 축 (Y)
    n_torso = np.column_stack([-u_torso[:, 1], u_torso[:, 0]])  # 수직 법선 (X)

    # 2. 척추 중심선 5개 분절 프록시 (Pelvis, L5, L3, T8, Neck)
    # 2D 상에서 곡률 산출
    s_pts = np.zeros((T, 5, 2), dtype=np.float32)
    s_pts[:, 0] = mid_hip
    s_pts[:, 1] = mid_hip + 0.25 * v_torso
    s_pts[:, 2] = mid_hip + 0.50 * v_torso
    s_pts[:, 3] = mid_hip + 0.75 * v_torso
    s_pts[:, 4] = neck

    # 3. 요추 2차 곡률 poly_a
    # 체간 축 대비 법선 방향 변위 계산
    disp_l5 = np.sum((s_pts[:, 1] - mid_hip) * n_torso, axis=-1)
    disp_l3 = np.sum((s_pts[:, 2] - mid_hip) * n_torso, axis=-1)
    disp_t8 = np.sum((s_pts[:, 3] - mid_hip) * n_torso, axis=-1)
    disp_neck = np.sum((s_pts[:, 4] - mid_hip) * n_torso, axis=-1)

    x_norm = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    curves_matrix = np.column_stack([np.zeros(T), disp_l5, disp_l3, disp_t8, disp_neck])

    poly_a_series = np.zeros(T, dtype=np.float32)
    for t in range(T):
        p = np.polyfit(x_norm, curves_matrix[t], deg=2)
        poly_a_series[t] = float(p[0])

    f1_poly_a = float(np.mean(poly_a_series))
    f2_stiffness_sd = float(np.std(poly_a_series))

    # 4. 척추 횡방향 이탈
    f3_lat_dev = float(np.max(np.abs(disp_l3)))

    # 5. 상체 굴곡 각도 및 속도
    # 이미지 평면에서 아래 방향이 +Y이므로 수직선은 (0, -1) (상승)
    torso_pitch = np.degrees(np.arctan2(u_torso[:, 0], -u_torso[:, 1]))
    torso_pitch_vel = np.abs(np.gradient(torso_pitch, dt))
    f4_flex_speed = float(np.max(torso_pitch_vel))

    # 6. 경추-체간 최소 각도
    f5_cervical_min = float(180.0 - np.max(np.abs(torso_pitch)))

    # 7. 가동범위 (Active ROM)
    wrist_y = mid_wr[:, 1]
    wrist_y_s = gaussian_filter1d(wrist_y, sigma=1.2)
    f6_active_rom = float(np.ptp(wrist_y_s))

    # 8. 요추 Cobb 각 프록시
    f7_cobb_mean = float(np.mean(np.abs(poly_a_series)) * 25.0)

    # 9. 어깨-골반 비틀림
    v_sh = l_sh - r_sh
    v_hip = l_hip - r_hip
    sh_angle = np.degrees(np.arctan2(v_sh[:, 1], v_sh[:, 0] + 1e-7))
    hip_angle = np.degrees(np.arctan2(v_hip[:, 1], v_hip[:, 0] + 1e-7))
    twist = np.abs(sh_angle - hip_angle)
    twist = np.minimum(twist, 360.0 - twist)
    f8_twist_max = float(np.max(twist))

    # 10. 요추 전단력 프록시
    f9_shear = float(np.mean(np.sin(np.radians(np.abs(torso_pitch)))))

    # 11. 체간 각도 진폭
    f10_pitch_rom = float(np.ptp(torso_pitch))

    # 12. 목-골반 중심축 정렬 이탈도
    f11_align_dev = float(np.mean(np.abs(disp_l3 / (torso_len + 1e-6))))

    # 13. 척추 시현 변위
    f12_chord_disp = float(np.mean(np.abs(disp_t8)))

    # 14~16. 신체 비율
    mean_torso = float(np.mean(torso_len))
    arm_len = float(np.mean(np.linalg.norm(mid_sh - mid_wr, axis=-1)))
    femur_len = float(np.mean(np.linalg.norm(mid_hip - mid_kne, axis=-1)))
    tibia_len = float(np.mean(np.linalg.norm(mid_kne - mid_ank, axis=-1)))
    f13_arm_torso = arm_len / max(1e-4, mean_torso)
    f14_femur_tibia = femur_len / max(1e-4, tibia_len)
    f15_leg_torso = (femur_len + tibia_len) / max(1e-4, mean_torso)

    # 17. 무릎 고정 안정도
    kne_v1 = mid_hip - mid_kne
    kne_v2 = mid_ank - mid_kne
    dot_k = np.sum(kne_v1 * kne_v2, axis=-1)
    norm_k = np.linalg.norm(kne_v1, axis=-1) * np.linalg.norm(kne_v2, axis=-1) + 1e-7
    knee_angle = np.degrees(np.arccos(np.clip(dot_k / norm_k, -1.0, 1.0)))
    f16_knee_fixation_sd = float(np.std(knee_angle))

    # 18. 체간-골반 횡방향 상대 오프셋
    f17_torso_pelvis_offset = float(np.mean(np.abs(neck[:, 0] - mid_hip[:, 0])))

    # 19. 최하사점 척추 강성 유지 지표
    bottom_idx = int(np.argmax(wrist_y_s))
    curv_at_bottom = poly_a_series[bottom_idx]
    f18_sagittal_stiffness = float(curv_at_bottom - f1_poly_a)

    # -------------------------------------------------------------
    # ★ [20~24] 바벨 및 체간 좌우 기울임 특화 (Key 211~214)
    # -------------------------------------------------------------
    v_wrist = l_wr - r_wr
    barbell_roll = np.degrees(np.arctan2(v_wrist[:, 1], np.abs(v_wrist[:, 0]) + 1e-7))
    f19_barbell_roll_max = float(np.max(np.abs(barbell_roll)))
    f20_barbell_roll_mean = float(np.mean(np.abs(barbell_roll)))

    f21_barbell_lateral_drift = float(np.max(np.abs(mid_wr[:, 0] - mid_hip[:, 0])))

    sh_roll = np.degrees(np.arctan2(v_sh[:, 1], np.abs(v_sh[:, 0]) + 1e-7))
    f22_shoulder_roll_max = float(np.max(np.abs(sh_roll)))

    hip_roll = np.degrees(np.arctan2(v_hip[:, 1], np.abs(v_hip[:, 0]) + 1e-7))
    f23_hip_roll_max = float(np.max(np.abs(hip_roll)))

    # -------------------------------------------------------------
    # ★ [25~28] 허리 과신전(Hyperextension / Lockout Lean) 특화 (Key 215)
    # -------------------------------------------------------------
    ptp_y = np.ptp(wrist_y_s)
    if ptp_y > 1e-4:
        standing_frames = np.where(wrist_y_s < (np.min(wrist_y_s) + 0.15 * ptp_y))[0]
    else:
        standing_frames = np.arange(T)
    if len(standing_frames) == 0:
        standing_frames = np.array([0, T - 1])

    lockout_torso_pitch = torso_pitch[standing_frames]
    f24_lockout_posterior_lean = float(np.max(np.maximum(0.0, -lockout_torso_pitch)))

    pelvic_sway_2d = mid_hip[:, 0] - mid_ank[:, 0]
    f25_lockout_pelvic_sway = float(np.max(np.abs(pelvic_sway_2d[standing_frames])))

    f26_lumbar_lordosis_peak = float(np.min(poly_a_series))

    accel_full = np.gradient(torso_pitch_vel, dt)
    f27_lockout_reversal_accel = float(np.max(np.abs(accel_full[standing_frames])))

    # -------------------------------------------------------------
    # ★ [29~30] 등 굽힘(Round Back / Thoracic Kyphosis) 특화 (Key 216)
    # -------------------------------------------------------------
    kyphosis_series = np.maximum(0.0, poly_a_series)
    f28_thoracic_kyphosis_peak = float(np.max(kyphosis_series))

    c_curve_depth = np.max(np.maximum(0.0, disp_t8))
    f29_c_curve_depth = float(c_curve_depth)

    # 하강 최저 변곡점 부근의 흉추 굴곡 시간 적분값
    win_start = max(0, bottom_idx - 6)
    win_end = min(T, bottom_idx + 7)
    try:
        trapz_func = np.trapezoid
    except AttributeError:
        trapz_func = np.trapz
    f30_roundback_integral = float(trapz_func(kyphosis_series[win_start:win_end], dx=dt))

    feats = np.array([
        f1_poly_a, f2_stiffness_sd, f3_lat_dev, f4_flex_speed, f5_cervical_min,
        f6_active_rom, f7_cobb_mean, f8_twist_max, f9_shear, f10_pitch_rom,
        f11_align_dev, f12_chord_disp, f13_arm_torso, f14_femur_tibia, f15_leg_torso,
        f16_knee_fixation_sd, f17_torso_pelvis_offset, f18_sagittal_stiffness,
        f19_barbell_roll_max, f20_barbell_roll_mean, f21_barbell_lateral_drift,
        f22_shoulder_roll_max, f23_hip_roll_max, f24_lockout_posterior_lean,
        f25_lockout_pelvic_sway, f26_lumbar_lordosis_peak, f27_lockout_reversal_accel,
        f28_thoracic_kyphosis_peak, f29_c_curve_depth, f30_roundback_integral
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_sd2_rtmp_temporal_24d(kps_seq: np.ndarray) -> np.ndarray:
    """(64, 24) RTMPose 2D 척추 및 주요 관절 시계열 텐서 추출"""
    T = kps_seq.shape[0]
    if kps_seq.shape[1] == 26:
        # Halpe26: 12 key joints x 2 coordinates (X, Y) = 24D
        joints_idx = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19]
        temp = kps_seq[:, joints_idx, :2].reshape(T, 24)
    else:
        # MP33
        joints_idx = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        temp = kps_seq[:, joints_idx, :2].reshape(T, 24)
    return np.nan_to_num(temp.astype(np.float32), nan=0.0)


class SD2RTMPClassifier:
    """
    바벨 스티프 데드리프트 2번 오류(sd2: 척추 중립 위반) RTMPose 전용 Two-Tier Tri-Ensemble 분류기
    - Halpe26 및 MP33 64프레임 관절 좌표 입력 완전 지원
    - 2D 고정밀 30D 생체역학 피처 + 24D 시계열 TCN + LightGBM + CatBoost Tri-Ensemble
    - 3-State 선택적 진단 (CLEAN, DEFECT, RETAKE)
    """

    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"sd2_rtmp 가중치 번들 부재: {bundle_path}")

        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.model_name = bundle.get('model_name', 'sd2_model_v3_rtmp')
        self.defect_code = 'sd2'
        self.defect_name = 'sd2 (척추의 중립 위반 / Spine Neutrality)'
        self.feature_names = bundle.get('feature_names', FEATURE_NAMES_30_SD2_RTMP)

        self.tcn_weights = bundle['tcn_weights']
        self.lgb_models = bundle['lgb_models']
        self.cat_models = bundle['cat_models']
        self.scalers = bundle.get('scalers', None)
        self.weights = bundle.get('ensemble_weights', {'tcn': 0.20, 'lgb': 0.30, 'cat': 0.50})

        self.tau_clean = bundle['thresholds']['tau_clean']
        self.tau_defect = bundle['thresholds']['tau_defect']
        self.binary_threshold = bundle['thresholds'].get('binary_threshold', 0.50)

        # PyTorch TCN 앙상블 복원
        self.device = torch.device("cpu")
        self.tcn_models = []
        for state in self.tcn_weights:
            m = SD2DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25).to(self.device)
            m.load_state_dict(state)
            m.eval()
            self.tcn_models.append(m)

    @classmethod
    def extract_features(cls, kps_seq: np.ndarray) -> Dict[str, np.ndarray]:
        f30 = extract_sd2_rtmp_features(kps_seq)
        t24 = extract_sd2_rtmp_temporal_24d(kps_seq)
        return {'features_30d': f30, 'temporal_24d': t24}

    def predict_proba(self, kps_seq: np.ndarray) -> float:
        extracted = self.extract_features(kps_seq)
        f30 = extracted['features_30d'].reshape(1, -1)
        t24 = extracted['temporal_24d']

        # 1. TCN 추론
        tcn_x = torch.tensor(t24.transpose(1, 0), dtype=torch.float32).unsqueeze(0).to(self.device)
        tcn_probs = []
        with torch.no_grad():
            for m in self.tcn_models:
                p = torch.sigmoid(m(tcn_x)).item()
                tcn_probs.append(p)
        mean_tcn_p = float(np.mean(tcn_probs))

        # 2. GBDT 추론
        lgb_probs = []
        cat_probs = []
        for i in range(len(self.lgb_models)):
            if self.scalers is not None and i < len(self.scalers):
                f_sc = self.scalers[i].transform(f30)
            else:
                f_sc = f30
            lgb_probs.append(float(self.lgb_models[i].predict_proba(f_sc)[0, 1]))
            cat_probs.append(float(self.cat_models[i].predict_proba(f_sc)[0, 1]))

        mean_lgb_p = float(np.mean(lgb_probs))
        mean_cat_p = float(np.mean(cat_probs))

        p_defect = (
            self.weights['tcn'] * mean_tcn_p +
            self.weights['lgb'] * mean_lgb_p +
            self.weights['cat'] * mean_cat_p
        )
        return float(np.clip(p_defect, 0.0, 1.0))

    def diagnose(self, kps_seq: np.ndarray) -> Dict[str, Any]:
        p_def = self.predict_proba(kps_seq)
        extracted = self.extract_features(kps_seq)
        f30 = extracted['features_30d']

        sub_defects = []
        if f30[18] >= 4.0:  # barbell_roll_max
            sub_defects.append("바벨 좌우 기울임(Barbell Tilt Asymmetry)")
        if f30[23] >= 3.0:  # lockout_posterior_lean
            sub_defects.append("기립 락아웃 시 허리 과신전(Lumbar Hyperextension)")
        if f30[27] >= 0.04: # thoracic_kyphosis_peak
            sub_defects.append("상체 하강 시 등/어깨 굽힘(Round Back / Thoracic Kyphosis)")

        if p_def < self.tau_clean:
            state = 0
            state_label = "CLEAN"
            feedback = "척추의 중립이 전 구간에서 훌륭하게 유지되고 있습니다. 완벽한 자세입니다!"
        elif p_def >= self.tau_defect:
            state = 1
            state_label = "DEFECT"
            if len(sub_defects) > 0:
                feedback = f"척추 중립 위반 감지: {', '.join(sub_defects)}. 코어에 힘을 주고 척추를 일직선으로 곧게 펴주세요."
            else:
                feedback = "동작 중 척추가 굽거나 과도하게 젖혀졌습니다. 복압을 유지하고 척추 중립을 유지하세요."
        else:
            state = 2
            state_label = "RETAKE"
            feedback = "동작의 척추 정렬이 다소 모호하게 관측되었습니다. 카메라가 상체를 정면 또는 대각선 45도에서 잘 잡도록 다시 1회 수행해 주세요."

        return {
            'state': state,
            'state_label': state_label,
            'status': state_label,
            'probability': round(p_def, 4),
            'binary_prediction': int(p_def >= self.binary_threshold),
            'detected_sub_defects': sub_defects,
            'feedback': feedback
        }
