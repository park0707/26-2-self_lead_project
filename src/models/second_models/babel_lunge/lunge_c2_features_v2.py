# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: lunge_c2_features_v2.py
Version: v2.0 (SOTA Research-Driven Upgrade)
Purpose: 64프레임 순수 MediaPipe 33관절 3D 포즈 텐서로부터 View-Invariant 국소 좌표계 변환 기반 30차원 정예 생체역학 피처 추출
         - Local Anatomical Coordinate System (R_body) 강체 회전 변환으로 시점 편향(0°, 45°, 후방 45°) 원천 소거
         - Tri-planar Kinetic Chain 지표 (FPPA, Foot Progression Angle, Directional Misalignment, Hip Adduction, Trunk Lateral Lean)
         - 3위상(하강기-최하점-상승기) 동역학 전이 피처
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

FEATURE_NAMES_30D = [
    # Group A: 선행 발 지향 및 FPA 내회전 피처 (7D)
    "lead_foot_inward_angle_bottom",        # 1. 최하사점 선행 발목-발끝 벡터와 체간 정면 법선의 수평면 내각
    "fpa_inward_angle_body",                # 2. [SOTA] 신체 국소 직교계(X-Z) 투영 정밀 Foot Progression Angle (FPA)
    "lead_foot_yaw_rom",                    # 3. 전체 동작 중 선행 발 장축의 수평 회전 가동범위
    "lead_ankle_toe_proj_ml",               # 4. 골반 수평축(ML)에 투영된 발끝-발목 상대 변위비
    "lead_foot_axis_stability",             # 5. 하강/상승 구간 발끝 방향 벡터의 요동 분산
    "lead_foot_inward_ratio",               # 6. 발끝이 12° 이상 안쪽으로 꺾여 있는 프레임 비율
    "lead_heel_toe_skew",                   # 7. 발뒤꿈치 대비 발끝의 내측 편차 거리비

    # Group B: 선행 무릎 FPPA 및 동적 외반 피처 (8D)
    "lead_knee_fppa_bottom",                # 8. 관상면 상 대퇴골-경골 정사영 각도 (FPPA)
    "lead_knee_fppa_body_loc",              # 9. [SOTA] 신체 국소 전두면(X-Y) 상 정밀 FPPA 각도 (180° 편차)
    "lead_knee_medial_drift",               # 10. 골반-발목 연결선 대비 무릎 관절점의 내측 쏠림 변위량
    "lead_knee_valgus_peak",                # 11. 최하사점 부근 무릎 외반 최대 피크 각도
    "lead_knee_lateral_ratio",              # 12. 골반 너비 대비 선행 무릎의 횡방향 이탈 비율
    "lead_knee_velocity_ml",                # 13. 최하사점 진입 시 무릎 관절의 내측 횡방향 이동 속도
    "lead_knee_ankle_alignment",            # 14. 무릎 중심과 발목 중심의 수직 정렬 오차
    "fppa_eccentric_to_bottom_delta",       # 15. [SOTA] 하강기 시작 대비 최하사점 FPPA 동적 붕괴 편차량

    # Group C: 몸통-골반 정렬 및 경사 피처 (7D)
    "trunk_lateral_lean_bottom",            # 16. 체간 축(어깨중심-골반중심)의 전두면 롤 각도 (몸통 갸우뚱)
    "trunk_lateral_lean_deg_loc",           # 17. [SOTA] 국소 직교계 상 체간 측방 기울기(Trunk Lateral Lean, TLL)
    "pelvic_obliquity_bottom",              # 18. 전두면 상의 골반 수평 기울기각 (Trendelenburg Drop)
    "trunk_pelvis_yaw_torsion",             # 19. 흉곽 평면 법선과 골반 축 법선 간의 수평면 비틀림(Torsion) 각도
    "trunk_canting_rom",                    # 20. 전체 동작 중 상체 측방 기울기 가동범위
    "trunk_counterbalance_score",           # 21. 선행 발 반대 방향으로 상체가 보상 기울어짐을 계측한 지표
    "pelvic_rotation_bottom",               # 22. 최하사점에서 골반 축의 회전 편차각

    # Group D: 상하체 결합 및 비폐색 프록시 피처 (8D)
    "lead_hip_adduction_bottom",            # 23. 선행 대퇴골의 관상면 내전 각도
    "lead_hip_adduction_angle_loc",         # 24. [SOTA] 골반축 직교 정밀 대퇴골 내전각
    "trunk_foot_angle_deviation",           # 25. 체간 전방 벡터와 선행 발 장축 간의 3차원 입체 편차각
    "directional_misalignment_thigh_foot",  # 26. [SOTA] 대퇴 진행 벡터와 발 장축 간의 횡단면 사잇각 (Torsional Discrepancy)
    "knee_foot_torsion_angle",              # 27. 무릎 굴곡 방향과 발 장축 간의 비틀림 각도
    "inter_knee_dist_ratio",                # 28. 양 무릎 간 3D 이격거리 대비 골반 너비 비율
    "lead_femur_coronal_tilt",              # 29. 전두면 상에서 선행 허벅지의 수직 대비 경사각
    "kinetic_chain_triplanar_score"         # 30. [SOTA] 3면(전두/횡단/시상) 결합 운동사슬 파탄 복합지수
]


def detect_lead_leg_and_bottom(kps: np.ndarray) -> Tuple[bool, int, np.ndarray, int, int]:
    """
    (64, 33, 3) 시퀀스로부터 선행 다리, 최하사점, 위상 인덱스 계산
    Returns:
        is_left_lead (bool)
        bottom_idx (int)
        bottom_win_mask (np.ndarray)
        start_idx (int)
        end_idx (int)
    """
    T = kps.shape[0]

    hip_l = kps[:, 23, :]
    hip_r = kps[:, 24, :]
    knee_l = kps[:, 25, :]
    knee_r = kps[:, 26, :]
    ankle_l = kps[:, 27, :]
    ankle_r = kps[:, 28, :]

    pelvis_center = 0.5 * (hip_l + hip_r)
    hip_axis = hip_r - hip_l
    hip_axis_norm = hip_axis / (np.linalg.norm(hip_axis, axis=-1, keepdims=True) + 1e-7)

    vert_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    ap_axis = np.cross(vert_axis[None, :], hip_axis_norm)
    ap_axis = ap_axis / (np.linalg.norm(ap_axis, axis=-1, keepdims=True) + 1e-7)

    disp_ap_l = np.sum((ankle_l - pelvis_center) * ap_axis, axis=-1)
    disp_ap_r = np.sum((ankle_r - pelvis_center) * ap_axis, axis=-1)

    mean_disp_l = np.mean(disp_ap_l)
    mean_disp_r = np.mean(disp_ap_r)

    if np.abs(mean_disp_l - mean_disp_r) > 0.05:
        is_left_lead = bool(mean_disp_l > mean_disp_r)
    else:
        is_left_lead = bool(np.max(np.abs(ankle_l[:, 2] - pelvis_center[:, 2])) > 
                            np.max(np.abs(ankle_r[:, 2] - pelvis_center[:, 2])))

    if is_left_lead:
        v_th = hip_l - knee_l
        v_sh = ankle_l - knee_l
    else:
        v_th = hip_r - knee_r
        v_sh = ankle_r - knee_r

    cos_k = np.sum(v_th * v_sh, axis=-1) / (
        (np.linalg.norm(v_th, axis=-1) + 1e-7) * (np.linalg.norm(v_sh, axis=-1) + 1e-7)
    )
    knee_angles = np.degrees(np.arccos(np.clip(cos_k, -1.0, 1.0)))

    valid_range = np.arange(5, T - 5)
    bottom_idx = int(valid_range[np.argmin(knee_angles[valid_range])])
    bottom_win_mask = (np.arange(T) >= max(0, bottom_idx - 6)) & (np.arange(T) <= min(T - 1, bottom_idx + 6))
    
    start_idx = max(0, bottom_idx - 16)
    end_idx = min(T - 1, bottom_idx + 16)

    return is_left_lead, bottom_idx, bottom_win_mask, start_idx, end_idx


def extract_lunge_c2_features_v2(kps: np.ndarray) -> np.ndarray:
    """
    단일 (64, 33, 3) 텐서에서 30차원 SOTA 생체역학 정예 피처 추출
    """
    T = kps.shape[0]
    is_left_lead, bottom_idx, win_mask, start_idx, end_idx = detect_lead_leg_and_bottom(kps)
    win_cnt = float(np.sum(win_mask)) or 1.0

    sh_l = kps[:, 11, :]
    sh_r = kps[:, 12, :]
    hip_l = kps[:, 23, :]
    hip_r = kps[:, 24, :]
    knee_l = kps[:, 25, :]
    knee_r = kps[:, 26, :]
    ankle_l = kps[:, 27, :]
    ankle_r = kps[:, 28, :]
    heel_l = kps[:, 29, :]
    heel_r = kps[:, 30, :]
    toe_l = kps[:, 31, :]
    toe_r = kps[:, 32, :]

    pelvis_w = np.mean(np.linalg.norm(hip_r - hip_l, axis=-1)) + 1e-6
    sh_center = 0.5 * (sh_l + sh_r)
    pelvis_center = 0.5 * (hip_l + hip_r)

    # -------------------------------------------------------------------------
    # [SOTA 핵심 1] View-Invariant Local Anatomical Coordinate Frame (R_body)
    # -------------------------------------------------------------------------
    # u_pelvis (좌우 ML 축: 우골반 -> 좌골반)
    u_pelvis = hip_l - hip_r
    u_pelvis = u_pelvis / (np.linalg.norm(u_pelvis, axis=-1, keepdims=True) + 1e-7)

    # v_trunk (상하 VT 축: 골반중심 -> 어깨중심)
    v_trunk = sh_center - pelvis_center
    v_trunk = v_trunk / (np.linalg.norm(v_trunk, axis=-1, keepdims=True) + 1e-7)

    # w_ap (전후 AP 축: ML x VT)
    w_ap = np.cross(u_pelvis, v_trunk)
    w_ap = w_ap / (np.linalg.norm(w_ap, axis=-1, keepdims=True) + 1e-7)

    # 재직교화된 좌우 ML 축 (VT x AP)
    u_ml = np.cross(v_trunk, w_ap)
    u_ml = u_ml / (np.linalg.norm(u_ml, axis=-1, keepdims=True) + 1e-7)

    if is_left_lead:
        lead_hip, trail_hip = hip_l, hip_r
        lead_knee, trail_knee = knee_l, knee_r
        lead_ankle, trail_ankle = ankle_l, ankle_r
        lead_heel, trail_heel = heel_l, heel_r
        lead_toe, trail_toe = toe_l, toe_r
        inward_sign = 1.0
    else:
        lead_hip, trail_hip = hip_r, hip_l
        lead_knee, trail_knee = knee_r, knee_l
        lead_ankle, trail_ankle = ankle_r, ankle_l
        lead_heel, trail_heel = heel_r, heel_l
        lead_toe, trail_toe = toe_r, toe_l
        inward_sign = -1.0

    # 국소계 투영 함수
    def project_loc(pt, origin):
        diff = pt - origin
        x_loc = np.sum(diff * u_ml, axis=-1) * inward_sign # 항상 내측이 양수(+)
        y_loc = np.sum(diff * v_trunk, axis=-1)
        z_loc = np.sum(diff * w_ap, axis=-1)
        return x_loc, y_loc, z_loc

    # 국소 좌표 계산
    knee_x, knee_y, knee_z = project_loc(lead_knee, lead_hip)
    ankle_x, ankle_y, ankle_z = project_loc(lead_ankle, lead_hip)
    heel_x, heel_y, heel_z = project_loc(lead_heel, lead_ankle)
    toe_x, toe_y, toe_z = project_loc(lead_toe, lead_ankle)

    foot_vec = lead_toe - lead_heel
    foot_vec_norm = foot_vec / (np.linalg.norm(foot_vec, axis=-1, keepdims=True) + 1e-7)

    # -------------------------------------------------------------------------
    # Group A: 선행 발 지향 및 FPA 내회전 피처 (7D)
    # -------------------------------------------------------------------------
    # 1. 수평면 내각
    foot_proj_ap = np.sum(foot_vec_norm * w_ap, axis=-1)
    foot_proj_ml = np.sum(foot_vec_norm * u_ml, axis=-1)
    foot_angle_deg = np.degrees(np.arctan2(foot_proj_ml, foot_proj_ap)) * inward_sign
    lead_foot_inward_angle_bottom = float(np.sum(foot_angle_deg * win_mask) / win_cnt)

    # 2. [SOTA] 신체 국소계 상의 Foot Progression Angle (FPA)
    # 발끝-발뒤꿈치 벡터의 국소 X-Z 수평각
    fpa_loc = np.degrees(np.arctan2(toe_x - heel_x, toe_z - heel_z + 1e-7))
    fpa_inward_angle_body = float(np.sum(fpa_loc * win_mask) / win_cnt)

    # 3. 발 회전 가동범위
    lead_foot_yaw_rom = float(np.ptp(foot_angle_deg))

    # 4. 골반 수평축 투영 변위비
    toe_ankle_ml = np.sum((lead_toe - lead_ankle) * u_ml, axis=-1) * inward_sign
    lead_ankle_toe_proj_ml = float(np.sum((toe_ankle_ml / pelvis_w) * win_mask) / win_cnt)

    # 5. 발끝 안정성
    lead_foot_axis_stability = float(np.std(foot_angle_deg))

    # 6. 내회전 비율
    lead_foot_inward_ratio = float(np.mean(foot_angle_deg > 10.0))

    # 7. 뒤꿈치 대비 발끝 스큐
    heel_toe_diff = (lead_toe - lead_heel) * inward_sign
    lead_heel_toe_skew = float(np.sum(np.sum(heel_toe_diff * u_ml, axis=-1) * win_mask) / (win_cnt * pelvis_w))

    # -------------------------------------------------------------------------
    # Group B: 선행 무릎 FPPA 및 동적 외반 피처 (8D)
    # -------------------------------------------------------------------------
    # 8. 관상면 FPPA
    femur_ml = np.sum((lead_knee - lead_hip) * u_ml, axis=-1)
    femur_vt = np.sum((lead_knee - lead_hip) * v_trunk, axis=-1)
    tibia_ml = np.sum((lead_ankle - lead_knee) * u_ml, axis=-1)
    tibia_vt = np.sum((lead_ankle - lead_knee) * v_trunk, axis=-1)

    femur_ang = np.arctan2(femur_ml * inward_sign, np.abs(femur_vt) + 1e-7)
    tibia_ang = np.arctan2(tibia_ml * inward_sign, np.abs(tibia_vt) + 1e-7)
    fppa_deg = np.degrees(np.abs(femur_ang - tibia_ang))
    lead_knee_fppa_bottom = float(np.sum(fppa_deg * win_mask) / win_cnt)

    # 9. [SOTA] 신체 국소 전두면(X-Y) 상 정밀 FPPA 각도 (180° 편차)
    v_thigh_loc = np.stack([knee_x, knee_y], axis=-1)
    v_shank_loc = np.stack([ankle_x - knee_x, ankle_y - knee_y], axis=-1)
    cos_fppa_loc = np.sum(v_thigh_loc * v_shank_loc, axis=-1) / (
        np.linalg.norm(v_thigh_loc, axis=-1) * np.linalg.norm(v_shank_loc, axis=-1) + 1e-7
    )
    fppa_loc_deg = np.abs(180.0 - np.degrees(np.arccos(np.clip(cos_fppa_loc, -1.0, 1.0))))
    lead_knee_fppa_body_loc = float(np.sum(fppa_loc_deg * win_mask) / win_cnt)

    # 10. 무릎 내측 쏠림 변위량
    mid_hip_ankle = 0.5 * (lead_hip + lead_ankle)
    knee_drift_ml = np.sum((lead_knee - mid_hip_ankle) * u_ml, axis=-1) * inward_sign
    lead_knee_medial_drift = float(np.sum((knee_drift_ml / pelvis_w) * win_mask) / win_cnt)

    # 11. 무릎 외반 피크치
    lead_knee_valgus_peak = float(np.max(fppa_deg[win_mask]))

    # 12. 골반 너비 대비 횡방향 이탈비
    knee_hip_ml = np.sum((lead_knee - lead_hip) * u_ml, axis=-1) * inward_sign
    lead_knee_lateral_ratio = float(np.sum((knee_hip_ml / pelvis_w) * win_mask) / win_cnt)

    # 13. 횡방향 이동 속도
    knee_vel_ml = np.diff(knee_drift_ml) * 30.0
    lead_knee_velocity_ml = float(np.max(np.abs(knee_vel_ml[max(0, bottom_idx-6):min(T-1, bottom_idx+1)])))

    # 14. 무릎-발목 수직 정렬 오차
    lead_knee_ankle_alignment = float(np.sum(np.abs(femur_ml - tibia_ml) * win_mask) / (win_cnt * pelvis_w))

    # 15. [SOTA] 하강기(start_idx) 대비 최하사점 FPPA 동적 붕괴 편차량
    fppa_start = float(fppa_deg[start_idx])
    fppa_eccentric_to_bottom_delta = float(lead_knee_fppa_bottom - fppa_start)

    # -------------------------------------------------------------------------
    # Group C: 몸통-골반 정렬 및 경사 피처 (7D)
    # -------------------------------------------------------------------------
    # 16. 체간 롤 각도
    trunk_dy = sh_r[:, 1] - sh_l[:, 1]
    trunk_dx = sh_r[:, 0] - sh_l[:, 0]
    trunk_roll = np.degrees(np.arctan2(trunk_dy, np.abs(trunk_dx) + 1e-7))
    trunk_lateral_lean_bottom = float(np.sum(np.abs(trunk_roll) * win_mask) / win_cnt)

    # 17. [SOTA] 국소 직교계 상 체간 측방 기울기(TLL)
    v_trunk_ml = np.sum((sh_center - pelvis_center) * u_ml, axis=-1)
    v_trunk_vt = np.sum((sh_center - pelvis_center) * v_trunk, axis=-1)
    tll_loc = np.degrees(np.arctan2(v_trunk_ml * inward_sign, np.abs(v_trunk_vt) + 1e-7))
    trunk_lateral_lean_deg_loc = float(np.sum(np.abs(tll_loc) * win_mask) / win_cnt)

    # 18. 골반 수평 기울기각
    pelvic_dy = hip_r[:, 1] - hip_l[:, 1]
    pelvic_dx = hip_r[:, 0] - hip_l[:, 0]
    pelvic_roll = np.degrees(np.arctan2(pelvic_dy, np.abs(pelvic_dx) + 1e-7))
    pelvic_obliquity_bottom = float(np.sum(np.abs(pelvic_roll) * win_mask) / win_cnt)

    # 19. 흉곽-골반 비틀림 각도
    sh_axis = sh_l - sh_r
    sh_axis_norm = sh_axis / (np.linalg.norm(sh_axis, axis=-1, keepdims=True) + 1e-7)
    sh_ap = np.cross(sh_axis_norm, v_trunk)
    sh_ap = sh_ap / (np.linalg.norm(sh_ap, axis=-1, keepdims=True) + 1e-7)
    torsion_cos = np.sum(sh_ap * w_ap, axis=-1)
    torsion_deg = np.degrees(np.arccos(np.clip(torsion_cos, -1.0, 1.0)))
    trunk_pelvis_yaw_torsion = float(np.sum(torsion_deg * win_mask) / win_cnt)

    # 20. 상체 가동범위
    trunk_canting_rom = float(np.ptp(trunk_roll))

    # 21. 카운터밸런스 점수
    lead_side_roll = trunk_roll * inward_sign
    trunk_counterbalance_score = float(np.sum(np.maximum(0.0, lead_side_roll) * win_mask) / win_cnt)

    # 22. 최하사점 골반 회전
    pelvic_rotation_bottom = float(np.abs(pelvic_roll[bottom_idx]))

    # -------------------------------------------------------------------------
    # Group D: 상하체 결합 및 비폐색 프록시 피처 (8D)
    # -------------------------------------------------------------------------
    # 23. 선행 대퇴골 내전각
    femur_vec = lead_knee - lead_hip
    femur_proj_ml = np.sum(femur_vec * u_ml, axis=-1) * inward_sign
    femur_proj_vt = np.sum(femur_vec * (-v_trunk), axis=-1)
    adduction_deg = np.degrees(np.arctan2(femur_proj_ml, np.maximum(0.05, femur_proj_vt)))
    lead_hip_adduction_bottom = float(np.sum(np.maximum(0.0, adduction_deg) * win_mask) / win_cnt)

    # 24. [SOTA] 골반축 직교 정밀 대퇴골 내전각
    hip_add_loc = np.degrees(np.arctan2(knee_x, -knee_y + 1e-7))
    lead_hip_adduction_angle_loc = float(np.sum(np.maximum(0.0, hip_add_loc) * win_mask) / win_cnt)

    # 25. 체간-발 입체 편차각
    cos_tf = np.sum(w_ap * foot_vec_norm, axis=-1)
    ang_tf = np.degrees(np.arccos(np.clip(cos_tf, -1.0, 1.0)))
    trunk_foot_angle_deviation = float(np.sum(ang_tf * win_mask) / win_cnt)

    # 26. [SOTA] 대퇴 진행 벡터 vs 발 장축 수평면 사잇각 (Directional Misalignment)
    v_thigh_trans = np.stack([knee_x, knee_z], axis=-1)
    v_foot_trans = np.stack([toe_x - heel_x, toe_z - heel_z], axis=-1)
    cos_misalign = np.sum(v_thigh_trans * v_foot_trans, axis=-1) / (
        np.linalg.norm(v_thigh_trans, axis=-1) * np.linalg.norm(v_foot_trans, axis=-1) + 1e-7
    )
    misalignment_deg = np.degrees(np.arccos(np.clip(cos_misalign, -1.0, 1.0)))
    directional_misalignment_thigh_foot = float(np.sum(misalignment_deg * win_mask) / win_cnt)

    # 27. 무릎-발 비틀림
    knee_flex_vec = (lead_knee - lead_hip) + (lead_ankle - lead_knee)
    knee_flex_norm = knee_flex_vec / (np.linalg.norm(knee_flex_vec, axis=-1, keepdims=True) + 1e-7)
    cos_kf = np.sum(knee_flex_norm * foot_vec_norm, axis=-1)
    knee_foot_torsion_angle = float(np.degrees(np.arccos(np.clip(cos_kf, -1.0, 1.0)))[bottom_idx])

    # 28. 양 무릎 이격거리비
    knee_dist = np.linalg.norm(knee_l - knee_r, axis=-1)
    inter_knee_dist_ratio = float(np.sum((knee_dist / pelvis_w) * win_mask) / win_cnt)

    # 29. 선행 허벅지 관상면 경사각
    lead_femur_coronal_tilt = float(np.sum(np.abs(adduction_deg) * win_mask) / win_cnt)

    # 30. [SOTA] 3면(전두/횡단/시상) 결합 운동사슬 파탄 복합지수
    kinetic_chain_triplanar_score = float(
        (lead_foot_inward_angle_bottom / 12.0)**2 +
        (lead_knee_fppa_body_loc / 10.0)**2 +
        (directional_misalignment_thigh_foot / 12.0)**2 +
        (trunk_lateral_lean_deg_loc / 6.0)**2 +
        (lead_hip_adduction_angle_loc / 8.0)**2
    )

    feats = np.array([
        lead_foot_inward_angle_bottom,
        fpa_inward_angle_body,
        lead_foot_yaw_rom,
        lead_ankle_toe_proj_ml,
        lead_foot_axis_stability,
        lead_foot_inward_ratio,
        lead_heel_toe_skew,
        lead_knee_fppa_bottom,
        lead_knee_fppa_body_loc,
        lead_knee_medial_drift,
        lead_knee_valgus_peak,
        lead_knee_lateral_ratio,
        lead_knee_velocity_ml,
        lead_knee_ankle_alignment,
        fppa_eccentric_to_bottom_delta,
        trunk_lateral_lean_bottom,
        trunk_lateral_lean_deg_loc,
        pelvic_obliquity_bottom,
        trunk_pelvis_yaw_torsion,
        trunk_canting_rom,
        trunk_counterbalance_score,
        pelvic_rotation_bottom,
        lead_hip_adduction_bottom,
        lead_hip_adduction_angle_loc,
        trunk_foot_angle_deviation,
        directional_misalignment_thigh_foot,
        knee_foot_torsion_angle,
        inter_knee_dist_ratio,
        lead_femur_coronal_tilt,
        kinetic_chain_triplanar_score
    ], dtype=np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    return feats


def extract_lunge_c2_features_v2_batch(X_seqs: np.ndarray) -> np.ndarray:
    N = X_seqs.shape[0]
    feats = np.zeros((N, 30), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_lunge_c2_features_v2(X_seqs[i])
    return feats


if __name__ == '__main__':
    dummy = np.random.randn(64, 33, 3).astype(np.float32)
    f = extract_lunge_c2_features_v2(dummy)
    print(f"[+] Lunge C2 v2 Features extracted! Shape: {f.shape}")
    assert f.shape == (30,), f"Expected (30,), got {f.shape}"
    print("[+] All assertions passed for v2 features!")
