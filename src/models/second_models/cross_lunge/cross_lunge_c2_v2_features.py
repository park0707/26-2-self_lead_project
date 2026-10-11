# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Module: cross_lunge_c2_v2_features.py
Purpose: cl2_v2 전용 크로스 런지 맞춤형 59차원 SOTA 피처 추출기
         - [v1 51D 베이스라인 전수 계승]:
           Group A: PACS 골반 직교계 기반 체간-무릎-발 3D 정렬 (8D)
           Group B: 후방 뷰 가림 극복 정강이 비틀림 & 궤적 불변 피처 (8D)
           Group C: 3단계 시간적 동역학 및 직립 상대 변화량 피처 (9D)
           Group D: 공식 5대 시점 One-Hot 인코딩 피처 (5D)
           Group E: 크로스 런지 삼방향 일치 기하 지표 (7D)
           Group F: 카메라 광축 폐색 및 단축 피처 (2D)
           Group G: v7 혁신 시계열, 광축 역투영, 역학적 복원 피처 (12D)
         - [v2 크로스 런지 특화 8대 혁신 피처 신설 (총 59D)]:
           52. lead_foot_outward_external_rotation_deg: 앞발 수평 외회전 절대 각도 (발 바깥쪽 꺾임 직격 포착)
           53. delta_lead_foot_external_rotation_deg: 기립 대비 최하사점 앞발 외회전 증가량
           54. torso_coronal_lateral_canting_angle: 체간 척추의 관상면 좌우 측방 기울기 (몸통 구부림 포착)
           55. torso_pelvis_yaw_torsion_bottom: 최하사점 어깨축 vs 골반축 수평 비틀림 각도
           56. cross_step_angle_to_torso: 뒷발-앞발 교차 진행선 vs 체간 법선 사잇각
           57. pelvic_cross_torque_coupling: 뒷다리 교차선과 골반 횡축 간의 3차원 토크 결합도
           58. lead_leg_fppa_coronal_collapse: 관상면 상 앞무릎 동적 외반 붕괴각
           59. lead_chain_4pt_planar_residual: 고관절-무릎-발목-발끝 4점 하지 사슬 동일평면성 잔차 RMSE
         100% 무누수(0.00% Leakage) 순수 비전 MediaPipe 33관절 3D 텐서 기반
"""

import os
import sys
import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.cross_lunge_spatial_leg_separator import detect_cross_lunge_lead_leg_spatial
from src.models.second_models.cross_lunge.cross_lunge_c2_v7_features import (
    FEATURE_NAMES_51D,
    extract_cross_lunge_c2_v7_features_51d,
    safe_savgol_filter_kps,
    vector_angle_3d,
    build_pacs_basis,
    compute_cross_lunge_geometric_alignment
)

V2_SPECIALIZED_FEATURE_NAMES_8D = [
    "lead_foot_outward_external_rotation_deg",  # 52. [v2 특화] 앞발 수평 외회전 절대 각도 (발 바깥쪽 꺾임 직격 포착)
    "delta_lead_foot_external_rotation_deg",    # 53. [v2 특화] 기립 대비 최하사점 앞발 외회전 증가량 (동적 외회전)
    "torso_coronal_lateral_canting_angle",      # 54. [v2 특화] 체간 척추의 관상면 좌우 측방 기울기 (몸통 갸우뚱/구부림 직격 포착)
    "torso_pelvis_yaw_torsion_bottom",          # 55. [v2 특화] 최하사점 어깨축 vs 골반축 수평 비틀림 각도
    "cross_step_angle_to_torso",                # 56. [v2 특화] 뒷발-앞발 교차 진행선 vs 체간 법선 사잇각 (크로스 깊이)
    "pelvic_cross_torque_coupling",             # 57. [v2 특화] 뒷다리 교차선과 골반 횡축 간의 3차원 토크 결합도
    "lead_leg_fppa_coronal_collapse",           # 58. [v2 특화] 관상면 상 앞무릎 동적 외반 붕괴각 (Knee Valgus)
    "lead_chain_4pt_planar_residual",           # 59. [v2 특화] 고관절-무릎-발목-발끝 4점 하지 사슬 동일평면성 잔차 RMSE
]

FEATURE_NAMES_59D = FEATURE_NAMES_51D + V2_SPECIALIZED_FEATURE_NAMES_8D


def compute_v2_cross_lunge_specialized_features_8d(kps_clean: np.ndarray, is_left_lead: bool, t_bot: int) -> np.ndarray:
    """
    크로스 런지 c2 결함(발방향 바깥쪽, 몸통 구부림/비틀림) 특화 8D 피처 산출
    """
    T, V, C = kps_clean.shape
    if is_left_lead:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
        rear_ank_idx = 28
        lat_sign = 1.0 # 왼발은 바깥쪽이 -X 방향 (MediaPipe 기준)
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
        rear_ank_idx = 27
        lat_sign = -1.0 # 오른발은 바깥쪽이 +X 방향

    kps_stand = np.mean(kps_clean[:4], axis=0)
    kps_bot = kps_clean[t_bot]

    R_pacs_bot, mid_hip_bot = build_pacs_basis(kps_bot)
    R_pacs_stand, _ = build_pacs_basis(kps_stand)

    # -------------------------------------------------------------------------
    # 52. lead_foot_outward_external_rotation_deg
    # -------------------------------------------------------------------------
    # 발 장축 벡터 (뒤꿈치 -> 앞꿈치)
    v_foot_bot = kps_bot[toe_idx] - kps_bot[heel_idx]
    v_foot_pacs = R_pacs_bot @ v_foot_bot
    # pacs 좌표계: X는 좌측(+), Y는 상향(+), Z는 전방(+)
    # 외회전(External Rotation):
    # 왼발(lat_sign=1.0)인 경우 X > 0이면 외측, 오른발(lat_sign=-1.0)인 경우 X < 0이면 외측
    foot_yaw_signed = float(np.degrees(np.arctan2(v_foot_pacs[0], max(abs(v_foot_pacs[2]), 1e-4))))
    ext_rot_deg = float(lat_sign * foot_yaw_signed)
    lead_foot_outward_external_rotation_deg = max(0.0, ext_rot_deg)

    # -------------------------------------------------------------------------
    # 53. delta_lead_foot_external_rotation_deg
    # -------------------------------------------------------------------------
    v_foot_stand = kps_stand[toe_idx] - kps_stand[heel_idx]
    v_foot_stand_pacs = R_pacs_stand @ v_foot_stand
    stand_yaw_signed = float(np.degrees(np.arctan2(v_foot_stand_pacs[0], max(abs(v_foot_stand_pacs[2]), 1e-4))))
    stand_ext_rot = float(lat_sign * stand_yaw_signed)
    delta_lead_foot_external_rotation_deg = float(ext_rot_deg - stand_ext_rot)

    # -------------------------------------------------------------------------
    # 54. torso_coronal_lateral_canting_angle
    # -------------------------------------------------------------------------
    # 척추 벡터 (골반 중심 -> 어깨 중심)
    mid_sh_bot = (kps_bot[11] + kps_bot[12]) * 0.5
    v_spine = mid_sh_bot - mid_hip_bot
    # 골반 관상면 상 기울기: 수직(Y) 축 대비 X축 편향 각도
    v_spine_pacs = R_pacs_bot @ v_spine
    torso_coronal_lateral_canting_angle = float(np.degrees(np.arctan2(abs(v_spine_pacs[0]), max(abs(v_spine_pacs[1]), 1e-4))))

    # -------------------------------------------------------------------------
    # 55. torso_pelvis_yaw_torsion_bottom
    # -------------------------------------------------------------------------
    v_shoulder = kps_bot[11] - kps_bot[12]
    v_shoulder_pacs = R_pacs_bot @ v_shoulder
    # PACS 상 골반 횡축은 X축 (1, 0, 0)
    torso_pelvis_yaw_torsion_bottom = float(np.degrees(np.arctan2(abs(v_shoulder_pacs[2]), max(abs(v_shoulder_pacs[0]), 1e-4))))

    # -------------------------------------------------------------------------
    # 56. cross_step_angle_to_torso
    # -------------------------------------------------------------------------
    # 뒷발목 -> 앞발목 교차선 벡터
    v_cross = kps_bot[ank_idx] - kps_bot[rear_ank_idx]
    v_cross_pacs = R_pacs_bot @ v_cross
    # 체간 전방축(Z축)과의 사잇각
    u_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cross_step_angle_to_torso = vector_angle_3d(v_cross_pacs, u_fwd)

    # -------------------------------------------------------------------------
    # 57. pelvic_cross_torque_coupling
    # -------------------------------------------------------------------------
    norm_cross = np.linalg.norm(v_cross_pacs) + 1e-6
    u_cross = v_cross_pacs / norm_cross
    # 골반 횡축(X축: 1, 0, 0)과의 내적 결합도
    pelvic_cross_torque_coupling = float(abs(u_cross[0]))

    # -------------------------------------------------------------------------
    # 58. lead_leg_fppa_coronal_collapse
    # -------------------------------------------------------------------------
    # 관상면(X-Y) 상 대퇴와 정강이의 투영 각도
    v_thigh_coronal = np.array([v_foot_pacs[0], v_foot_pacs[1]], dtype=np.float32)
    v_thigh_bot = kps_bot[knee_idx] - kps_bot[hip_idx]
    v_shank_bot = kps_bot[ank_idx] - kps_bot[knee_idx]
    v_thigh_p = R_pacs_bot @ v_thigh_bot
    v_shank_p = R_pacs_bot @ v_shank_bot
    v_t_cor = np.array([v_thigh_p[0], v_thigh_p[1]], dtype=np.float32)
    v_s_cor = np.array([v_shank_p[0], v_shank_p[1]], dtype=np.float32)
    norm_tc = np.linalg.norm(v_t_cor) + 1e-6
    norm_sc = np.linalg.norm(v_s_cor) + 1e-6
    cos_cor = np.clip(np.dot(v_t_cor, v_s_cor) / (norm_tc * norm_sc), -1.0, 1.0)
    lead_leg_fppa_coronal_collapse = float(180.0 - np.degrees(np.arccos(cos_cor)))

    # -------------------------------------------------------------------------
    # 59. lead_chain_4pt_planar_residual
    # -------------------------------------------------------------------------
    # 앞고관절, 앞무릎, 앞발목, 앞발끝 4점 평면 적합 잔차
    chain_pts = np.vstack([kps_bot[hip_idx], kps_bot[knee_idx], kps_bot[ank_idx], kps_bot[toe_idx]])
    pts_centered = chain_pts - np.mean(chain_pts, axis=0)
    try:
        _, _, vh = np.linalg.svd(pts_centered)
        normal = vh[2] # 최소 특이값 대응 법선 벡터
        residuals = np.dot(pts_centered, normal)
        lead_chain_4pt_planar_residual = float(np.sqrt(np.mean(residuals ** 2)))
    except Exception:
        lead_chain_4pt_planar_residual = 0.0

    feats_8d = [
        lead_foot_outward_external_rotation_deg,
        delta_lead_foot_external_rotation_deg,
        torso_coronal_lateral_canting_angle,
        torso_pelvis_yaw_torsion_bottom,
        cross_step_angle_to_torso,
        pelvic_cross_torque_coupling,
        lead_leg_fppa_coronal_collapse,
        lead_chain_4pt_planar_residual
    ]
    return np.array(feats_8d, dtype=np.float32)


def extract_cross_lunge_c2_v2_features_59d(kps_seq_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 관절 텐서로부터 cl2_v2 59차원 정예 피처 추출 (51D + 8D)
    """
    kps_clean = safe_savgol_filter_kps(kps_seq_64f[:, :, :3], window=7, poly=2)
    feats_51d = extract_cross_lunge_c2_v7_features_51d(kps_clean, view_code=view_code)

    is_left_lead, t_bot = detect_cross_lunge_lead_leg_spatial(kps_clean)
    feats_8d = compute_v2_cross_lunge_specialized_features_8d(kps_clean, is_left_lead, t_bot)

    feats_59d = np.concatenate([feats_51d, feats_8d])
    assert len(feats_59d) == 59, f"피처 차원이 59가 아닙니다: {len(feats_59d)}"
    return feats_59d.astype(np.float32)
