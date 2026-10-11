# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: side_lunge_c2_v8_front_features.py
Purpose: 시점 B (우측 전방 45°) & 시점 D (좌측 전방 45°) 전용 F1 70% 돌파 SOTA 58D 피처 추출기 (sl2_v8_front)
         - sl2_v7의 51D 피처 (3D 직교 기하 + 시계열 5D + 역투영 3D + 정강이 역학 복원 4D) 100% 완전 계승
         - 전방 45도 시점 특화 7대 혁신 생체역학 피처 신규 탑재 (총 58D):
           1. [FPA] 동적 발 진행각 (Dynamic Foot Progression Angle)
           2. [2D 스크린 정합] 카메라 화면 투영면 상의 체간-발끝 2D 코사인 정렬도
           3. [시점 비대칭 보정] 동측(Ipsilateral) vs 대측(Contralateral) 런지 정규화 발 편각
           4. [동적 외반슬] 45도 관측면 상의 무릎 내측 붕괴(Dynamic Knee Valgus) 비율
           5. [외번 델타] 직립 대비 최하사점 족부 관상면 외번/회내 기울기 변화량 (ΔEversion)
           6. [수평면 비틀림] 대퇴골 회전면 vs 정강이 진행면의 수평면 비틀림 각 (Tibiofemoral Twist)
           7. [스칼라 삼중적] 무릎 굴곡 평면 법선 vs 발 종축 삼중적 결함 지표 (κ_c2)
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.signal import savgol_filter

import sys
import os
PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features.side_lunge_spatial_leg_separator import detect_side_lunge_lead_leg_spatial
from side_lunge_c2_features_v3 import (
    FEATURE_NAMES_30D,
    safe_savgol_filter_kps,
    build_pacs_basis,
    vector_angle_3d,
    extract_side_lunge_c2_features_29d
)
from side_lunge_c2_v6_geometric_features import (
    FEATURE_NAMES_39D,
    compute_user_geometric_alignment,
    extract_side_lunge_c2_v6_geometric_features_39d
)
from side_lunge_c2_v7_features import (
    FEATURE_NAMES_51D,
    compute_v7_breakthrough_features_12d
)

V8_FRONT_NEW_FEATURE_NAMES_7D = [
    "fpa_dynamic_foot_progression_angle",               # 52. [전방특화] 동적 발 진행각 (FPA: 런지 진행선 대비 발 종축 회전각)
    "front_screen_space_2d_torso_foot_cosine",          # 53. [전방특화] 2D 화면 투영면 상의 체간-발끝 겉보기 코사인 정합도
    "front_view_relative_ipsi_contra_yaw",              # 54. [전방특화] 동측/대측(Ipsi/Contra) 시점 비대칭 보정 발 편각
    "dynamic_knee_valgus_medial_ratio",                 # 55. [전방특화] 전방 45도 투영면 상의 무릎 내측 붕괴(Valgus) 변위 비율
    "foot_eversion_coronal_tilt_delta",                 # 56. [전방특화] 하강 구간 족부 관상면 외번/회내 기울기 변화량 (ΔEversion)
    "front_tibiofemoral_transverse_torsion",            # 57. [전방특화] 대퇴골-정강이뼈 수평면 상대 비틀림 각도
    "knee_flexion_plane_foot_scalar_triple_product",    # 58. [전방특화] 무릎 굴곡 평면 법선 vs 발 종축 스칼라 삼중적 결함 지표 (κ_c2)
]

FEATURE_NAMES_58D = FEATURE_NAMES_51D + V8_FRONT_NEW_FEATURE_NAMES_7D


def compute_v8_front_breakthrough_features_7d(
    kps_clean: np.ndarray,
    view_code: str = "view2"
) -> np.ndarray:
    """
    (64, 33, 3) 스무딩된 3D 포즈 텐서로부터 전방 시점(B, D) 특화 7D 생체역학 피처 추출
    """
    T, V, C = kps_clean.shape
    is_left_lead, t_bot = detect_side_lunge_lead_leg_spatial(kps_clean)
    t_bot = max(5, min(T - 5, t_bot))

    if is_left_lead:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
        lat_sign = 1.0
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
        lat_sign = -1.0

    # 1. 동적 발 진행각 (Foot Progression Angle, FPA)
    v_prog = kps_clean[t_bot, ank_idx] - kps_clean[0, ank_idx]
    v_prog_h = np.array([v_prog[0], 0.0, v_prog[2]], dtype=np.float32)
    norm_prog = np.linalg.norm(v_prog_h) + 1e-6
    u_prog = v_prog_h / norm_prog

    v_foot = kps_clean[t_bot, toe_idx] - kps_clean[t_bot, heel_idx]
    v_foot_h = np.array([v_foot[0], 0.0, v_foot[2]], dtype=np.float32)
    norm_foot = np.linalg.norm(v_foot_h) + 1e-6
    u_foot = v_foot_h / norm_foot

    prog_normal = np.array([-u_prog[2], 0.0, u_prog[0]], dtype=np.float32) * lat_sign
    fpa_angle = float(np.degrees(np.arctan2(
        np.dot(u_foot, prog_normal),
        np.dot(u_foot, u_prog) + 1e-6
    )))

    # 2. 2D 스크린 공간 겉보기 체간-발끝 코사인 정합도
    pelvis_mid_2d = 0.5 * (kps_clean[t_bot, 23, :2] + kps_clean[t_bot, 24, :2])
    shoulder_mid_2d = 0.5 * (kps_clean[t_bot, 11, :2] + kps_clean[t_bot, 12, :2])
    v_torso_2d = shoulder_mid_2d - pelvis_mid_2d
    u_torso_2d = v_torso_2d / (np.linalg.norm(v_torso_2d) + 1e-6)

    v_foot_2d = kps_clean[t_bot, toe_idx, :2] - kps_clean[t_bot, heel_idx, :2]
    u_foot_2d = v_foot_2d / (np.linalg.norm(v_foot_2d) + 1e-6)
    front_screen_space_2d_torso_foot_cosine = float(np.dot(u_torso_2d, u_foot_2d))

    # 3. 동측/대측(Ipsilateral vs Contralateral) 시점 비대칭 보정 발 편각
    if view_code == 'view2':
        is_ipsi = 1.0 if not is_left_lead else -1.0
    else: # view4
        is_ipsi = 1.0 if is_left_lead else -1.0

    raw_yaw = math.atan2(v_foot[0] * lat_sign, v_foot[2] + 1e-6)
    front_view_relative_ipsi_contra_yaw = float(math.degrees(raw_yaw) * is_ipsi)

    # 4. 전방 45도 투영면 상의 무릎 내측 붕괴 (Dynamic Knee Valgus Index)
    p_hip = kps_clean[t_bot, hip_idx]
    p_knee = kps_clean[t_bot, knee_idx]
    p_ank = kps_clean[t_bot, ank_idx]

    line_hip_ank = p_ank - p_hip
    line_norm = np.linalg.norm(line_hip_ank) + 1e-6
    u_line = line_hip_ank / line_norm

    vec_hip_knee = p_knee - p_hip
    proj_dist = np.dot(vec_hip_knee, u_line)
    closest_p = p_hip + proj_dist * u_line
    valgus_vec = p_knee - closest_p
    mid_hip = 0.5 * (kps_clean[t_bot, 23] + kps_clean[t_bot, 24])
    dir_to_mid = mid_hip - p_knee
    dir_to_mid /= (np.linalg.norm(dir_to_mid) + 1e-6)
    dynamic_knee_valgus_medial_ratio = float(np.dot(valgus_vec, dir_to_mid) / line_norm)

    # 5. 하강 구간 족부 관상면 외번/회내 기울기 변화량 (ΔEversion)
    def get_coronal_tilt(t_idx):
        v_ah = kps_clean[t_idx, heel_idx] - kps_clean[t_idx, ank_idx]
        v_at = kps_clean[t_idx, toe_idx] - kps_clean[t_idx, ank_idx]
        n_foot = np.cross(v_ah, v_at)
        n_foot /= (np.linalg.norm(n_foot) + 1e-6)
        return float(np.degrees(np.arctan2(n_foot[0] * lat_sign, n_foot[1] + 1e-6)))

    tilt_0 = get_coronal_tilt(0)
    tilt_bot = get_coronal_tilt(t_bot)
    foot_eversion_coronal_tilt_delta = float(tilt_bot - tilt_0)

    # 6. 대퇴골-정강이뼈 수평면 상대 비틀림 각도 (Tibiofemoral Transverse Twist)
    v_thigh = kps_clean[t_bot, knee_idx] - kps_clean[t_bot, hip_idx]
    v_shank = kps_clean[t_bot, ank_idx] - kps_clean[t_bot, knee_idx]
    thigh_yaw = np.arctan2(v_thigh[0] * lat_sign, v_thigh[2] + 1e-6)
    shank_yaw = np.arctan2(v_shank[0] * lat_sign, v_shank[2] + 1e-6)
    front_tibiofemoral_transverse_torsion = float(np.degrees(shank_yaw - thigh_yaw))

    # 7. 무릎 굴곡 평면 법선 vs 발 종축 스칼라 삼중적 결함 지표 (κ_c2)
    # v_thigh x v_shank = 무릎 굴곡 평면 법선
    n_knee_flexion = np.cross(v_thigh, v_shank)
    norm_kf = np.linalg.norm(n_knee_flexion) + 1e-6
    u_n_knee = n_knee_flexion / norm_kf
    u_foot_3d = v_foot / (np.linalg.norm(v_foot) + 1e-6)
    # 정상 시 무릎 굽힘면과 발 방향이 평행하므로 법선과 발 벡터는 직교 (|내적| ≈ 0)
    # 결함 시 무릎과 발의 축이 어긋나 내적이 증가함
    kappa_c2 = float(abs(np.dot(u_foot_3d, u_n_knee)))

    return np.array([
        fpa_angle,
        front_screen_space_2d_torso_foot_cosine,
        front_view_relative_ipsi_contra_yaw,
        dynamic_knee_valgus_medial_ratio,
        foot_eversion_coronal_tilt_delta,
        front_tibiofemoral_transverse_torsion,
        kappa_c2
    ], dtype=np.float32)


def extract_side_lunge_c2_v8_front_features_58d(
    kps_seq_3d: np.ndarray,
    view_code: str = "view2"
) -> np.ndarray:
    """
    (64, 33, 3) 3D 포즈 시퀀스로부터 시점 B/D 전용 58D SOTA 피처 벡터 추출
    """
    if kps_seq_3d.shape[-1] == 4:
        kps_seq_3d = kps_seq_3d[:, :, :3]
    kps_clean = safe_savgol_filter_kps(kps_seq_3d, window=7, poly=2)

    # 1. v6 39D 피처 전수 계승
    feat_39d = extract_side_lunge_c2_v6_geometric_features_39d(kps_clean, view_code=view_code)

    # 2. v7 신규 12D 피처
    feat_12d = compute_v7_breakthrough_features_12d(kps_clean, view_code=view_code)
    feat_51d = np.concatenate([feat_39d, feat_12d])

    # 3. v8 전방 특화 7D 피처
    feat_7d = compute_v8_front_breakthrough_features_7d(kps_clean, view_code=view_code)
    feat_58d = np.concatenate([feat_51d, feat_7d])

    assert len(feat_58d) == 58, f"피처 차원 불일치: {len(feat_58d)} != 58"
    return feat_58d.astype(np.float32)
