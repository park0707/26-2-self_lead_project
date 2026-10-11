# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: lunge_c2_features.py
Purpose: 64프레임 순수 MediaPipe 33관절 3D 포즈 텐서로부터 24차원 정예 생체역학 피처 추출
         - 선행 다리(Lead Leg) 동적 무오류 자동 감지 (is_left_lead)
         - 최하사점(Bottom Phase) 윈도우 스냅샷 앵커링 (t_bottom ± 6 프레임)
         - 4대 피처 그룹 (발 지향 내회전 6D, 무릎 FPPA/외반 6D, 몸통-골반 정렬 6D, 비폐색 상위 프록시 6D)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

FEATURE_NAMES_24D = [
    # Group A: 선행 발 지향 및 내회전 피처 (6D)
    "lead_foot_inward_angle_bottom", # 1. 최하사점 선행 발목-발끝 벡터와 체간 정면 법선의 수평면 내각 (내회전 직접 계측)
    "lead_foot_yaw_rom",             # 2. 전체 동작 중 선행 발 장축의 수평 회전 가동범위 (각도)
    "lead_ankle_toe_proj_ml",        # 3. 골반 수평축(ML)에 투영된 발끝-발목 상대 변위비 (골반 너비 정규화)
    "lead_foot_axis_stability",      # 4. 하강/상승 구간 발끝 방향 벡터의 요동 분산
    "lead_foot_inward_ratio",        # 5. 발끝이 12° 이상 안쪽으로 꺾여 있는 프레임 비율
    "lead_heel_toe_skew",            # 6. 발뒤꿈치 대비 발끝의 내측 편차 거리비

    # Group B: 선행 무릎 FPPA 및 동적 외반 피처 (6D)
    "lead_knee_fppa_bottom",         # 7. 관상면 상 대퇴골-경골 정사영 각도 (FPPA, 정상 180° 대비 내측 꺾임)
    "lead_knee_medial_drift",        # 8. 골반-발목 연결선 대비 무릎 관절점의 내측 쏠림 변위량
    "lead_knee_valgus_peak",         # 9. 최하사점 부근 무릎 외반 최대 피크 각도
    "lead_knee_lateral_ratio",       # 10. 골반 너비 대비 선행 무릎의 횡방향 이탈 비율
    "lead_knee_velocity_ml",         # 11. 최하사점 진입 시 무릎 관절의 내측 횡방향 이동 속도
    "lead_knee_ankle_alignment",     # 12. 무릎 중심과 발목 중심의 수직 정렬 오차

    # Group C: 몸통-골반 정렬 및 경사 피처 (6D)
    "trunk_lateral_lean_bottom",     # 13. 체간 축(어깨중심-골반중심)의 전두면 롤 각도 (몸통 갸우뚱 감지)
    "pelvic_obliquity_bottom",       # 14. 전두면 상의 골반 수평 기울기각 (Trendelenburg Drop)
    "trunk_pelvis_yaw_torsion",      # 15. 흉곽 평면 법선과 골반 축 법선 간의 수평면 비틀림(Torsion) 각도
    "trunk_canting_rom",             # 16. 전체 동작 중 상체 측방 기울기 가동범위
    "trunk_counterbalance_score",    # 17. 선행 발 반대 방향으로 상체가 보상 기울어짐을 계측한 지표
    "pelvic_rotation_bottom",        # 18. 최하사점에서 골반 축의 회전 편차각

    # Group D: 상하체 결합 및 비폐색 프록시 피처 (6D)
    "lead_hip_adduction_bottom",     # 19. 선행 대퇴골의 관상면 내전 각도 (발 가림 시에도 노출되는 핵심 비폐색 지표)
    "trunk_foot_angle_deviation",    # 20. 체간 전방 벡터와 선행 발 장축 간의 3차원 입체 편차각
    "knee_foot_torsion_angle",       # 21. 무릎 굴곡 방향과 발 장축 간의 비틀림 각도
    "inter_knee_dist_ratio",         # 22. 양 무릎 간 3D 이격거리 대비 골반 너비 비율
    "lead_femur_coronal_tilt",       # 23. 전두면 상에서 선행 허벅지의 수직 대비 경사각
    "kinetic_chain_alignment_score"  # 24. [몸통 - 대퇴 - 경골 - 발] 4개 분절의 3D 공간적 정렬 지수
]


def detect_lead_leg_and_bottom(kps: np.ndarray) -> Tuple[bool, int, np.ndarray]:
    """
    (64, 33, 3) 시퀀스로부터 선행 다리(Lead Leg)와 최하사점(Bottom Phase) 윈도우 마스크를 계산
    Returns:
        is_left_lead (bool): 왼발 선행 여부 (True: 왼발 앞, False: 오른발 앞)
        bottom_idx (int): 선행 무릎 최대 굴곡(최하사점) 프레임 인덱스
        bottom_win_mask (np.ndarray): 최하사점 윈도우 (t_bottom ± 6) boolean 마스크 (64,)
    """
    T = kps.shape[0]

    # 관절 인덱스
    hip_l = kps[:, 23, :]
    hip_r = kps[:, 24, :]
    knee_l = kps[:, 25, :]
    knee_r = kps[:, 26, :]
    ankle_l = kps[:, 27, :]
    ankle_r = kps[:, 28, :]

    pelvis_center = 0.5 * (hip_l + hip_r)
    hip_axis = hip_r - hip_l # (64, 3)
    hip_axis_norm = hip_axis / (np.linalg.norm(hip_axis, axis=-1, keepdims=True) + 1e-7)

    # 전방(AP) 축 정의: 골반 축과 수직 상향([0, 1, 0] or [0, -1, 0])의 외적
    vert_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    ap_axis = np.cross(vert_axis[None, :], hip_axis_norm)
    ap_axis = ap_axis / (np.linalg.norm(ap_axis, axis=-1, keepdims=True) + 1e-7)

    # 양 발목의 골반 중심 대비 전방(AP) 변위
    disp_ap_l = np.sum((ankle_l - pelvis_center) * ap_axis, axis=-1) # (64,)
    disp_ap_r = np.sum((ankle_r - pelvis_center) * ap_axis, axis=-1) # (64,)

    # 전체 시퀀스 동안 전방으로 더 멀리 나간 발을 선행 발로 결정
    mean_disp_l = np.mean(disp_ap_l)
    mean_disp_r = np.mean(disp_ap_r)

    # 보조 판별: 무릎 높이(Y축) 및 발목 높이 비교
    if np.abs(mean_disp_l - mean_disp_r) > 0.05:
        is_left_lead = bool(mean_disp_l > mean_disp_r)
    else:
        # Z축 변위 기반 2차 판별
        is_left_lead = bool(np.max(np.abs(ankle_l[:, 2] - pelvis_center[:, 2])) > 
                            np.max(np.abs(ankle_r[:, 2] - pelvis_center[:, 2])))

    # 선행 무릎 각도 계산 (Hip-Knee-Ankle 3D 각도)
    if is_left_lead:
        v_th = hip_l - knee_l
        v_sh = ankle_l - knee_l
    else:
        v_th = hip_r - knee_r
        v_sh = ankle_r - knee_r

    cos_k = np.sum(v_th * v_sh, axis=-1) / (
        (np.linalg.norm(v_th, axis=-1) + 1e-7) * (np.linalg.norm(v_sh, axis=-1) + 1e-7)
    )
    knee_angles = np.degrees(np.arccos(np.clip(cos_k, -1.0, 1.0))) # (64,)

    # 최하사점: 무릎 굴곡이 가장 깊은(각도가 가장 작은) 프레임
    # 시작/끝 5프레임 제외
    valid_range = np.arange(5, T - 5)
    bottom_idx = int(valid_range[np.argmin(knee_angles[valid_range])])

    # 최하사점 윈도우 (t_bottom ± 6 프레임)
    bottom_win_mask = (np.arange(T) >= max(0, bottom_idx - 6)) & (np.arange(T) <= min(T - 1, bottom_idx + 6))
    return is_left_lead, bottom_idx, bottom_win_mask


def extract_lunge_c2_features(kps: np.ndarray) -> np.ndarray:
    """
    단일 시퀀스 (64, 33, 3) 포즈로부터 24D 정예 생체역학 피처 추출
    Input:
        kps: np.ndarray (64, 33, 3) - 순수 MediaPipe 33관절 3D 좌표
    Output:
        feats: np.ndarray (24,) - 정예 24차원 피처 벡터
    """
    T = kps.shape[0]
    is_left_lead, bottom_idx, win_mask = detect_lead_leg_and_bottom(kps)
    win_cnt = float(np.sum(win_mask)) or 1.0

    # 주요 관절점 추출
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

    # 신체 로컬 정규직교계 구축
    # 1. 골반 ML 축 (우측 -> 좌측)
    ml_axis = hip_l - hip_r
    ml_axis = ml_axis / (np.linalg.norm(ml_axis, axis=-1, keepdims=True) + 1e-7)

    # 2. 체간 수직(VT) 축 (골반중심 -> 어깨중심)
    vt_axis = sh_center - pelvis_center
    vt_axis = vt_axis / (np.linalg.norm(vt_axis, axis=-1, keepdims=True) + 1e-7)

    # 3. 체간 전방(AP) 축 (ML x VT)
    ap_axis = np.cross(ml_axis, vt_axis)
    ap_axis = ap_axis / (np.linalg.norm(ap_axis, axis=-1, keepdims=True) + 1e-7)

    # 선행/후행 관절 분리
    if is_left_lead:
        lead_hip, trail_hip = hip_l, hip_r
        lead_knee, trail_knee = knee_l, knee_r
        lead_ankle, trail_ankle = ankle_l, ankle_r
        lead_heel, trail_heel = heel_l, heel_r
        lead_toe, trail_toe = toe_l, toe_r
        inward_sign = 1.0 # 왼발은 우측(내측)으로 돌아갈 때 내회전
    else:
        lead_hip, trail_hip = hip_r, hip_l
        lead_knee, trail_knee = knee_r, knee_l
        lead_ankle, trail_ankle = ankle_r, ankle_l
        lead_heel, trail_heel = heel_r, heel_l
        lead_toe, trail_toe = toe_r, toe_l
        inward_sign = -1.0 # 오른발은 좌측(내측)으로 돌아갈 때 내회전

    # 선행 발 장축 벡터 (heel -> toe)
    foot_vec = lead_toe - lead_heel # (64, 3)
    foot_vec_norm = foot_vec / (np.linalg.norm(foot_vec, axis=-1, keepdims=True) + 1e-7)

    # -------------------------------------------------------------------------
    # Group A: 선행 발 지향 및 내회전 피처 (6D)
    # -------------------------------------------------------------------------
    # 1. 수평면 상 발끝 내회전 각도 (체간 전방 벡터와의 각도, 내측 회전 시 양수)
    foot_proj_ap = np.sum(foot_vec_norm * ap_axis, axis=-1)
    foot_proj_ml = np.sum(foot_vec_norm * ml_axis, axis=-1)
    foot_angle_deg = np.degrees(np.arctan2(foot_proj_ml, foot_proj_ap)) * inward_sign # (64,)

    lead_foot_inward_angle_bottom = float(np.sum(foot_angle_deg * win_mask) / win_cnt)
    lead_foot_yaw_rom = float(np.ptp(foot_angle_deg))

    # 3. 골반 수평축(ML)에 투영된 발끝-발목 상대 변위비
    toe_ankle_ml = np.sum((lead_toe - lead_ankle) * ml_axis, axis=-1) * inward_sign
    lead_ankle_toe_proj_ml = float(np.sum((toe_ankle_ml / pelvis_w) * win_mask) / win_cnt)

    # 4. 발끝 방향 벡터의 요동 분산
    lead_foot_axis_stability = float(np.std(foot_angle_deg))

    # 5. 발끝이 12도 이상 안쪽으로 꺾여 있는 프레임 비율
    lead_foot_inward_ratio = float(np.mean(foot_angle_deg > 12.0))

    # 6. 발뒤꿈치 대비 발끝의 내측 편차 거리비
    heel_toe_diff = (lead_toe - lead_heel) * inward_sign
    lead_heel_toe_skew = float(np.sum(np.sum(heel_toe_diff * ml_axis, axis=-1) * win_mask) / (win_cnt * pelvis_w))

    # -------------------------------------------------------------------------
    # Group B: 선행 무릎 FPPA 및 동적 외반 피처 (6D)
    # -------------------------------------------------------------------------
    # 관상면(Coronal: ML-VT 평면) 투영 무릎 각도 (FPPA)
    femur_ml = np.sum((lead_knee - lead_hip) * ml_axis, axis=-1)
    femur_vt = np.sum((lead_knee - lead_hip) * vt_axis, axis=-1)
    tibia_ml = np.sum((lead_ankle - lead_knee) * ml_axis, axis=-1)
    tibia_vt = np.sum((lead_ankle - lead_knee) * vt_axis, axis=-1)

    femur_ang = np.arctan2(femur_ml * inward_sign, np.abs(femur_vt) + 1e-7)
    tibia_ang = np.arctan2(tibia_ml * inward_sign, np.abs(tibia_vt) + 1e-7)
    fppa_deg = np.degrees(np.abs(femur_ang - tibia_ang)) # 정상은 0도에 가까움, 외반 시 증가

    lead_knee_fppa_bottom = float(np.sum(fppa_deg * win_mask) / win_cnt)

    # 8. 무릎 관절점의 내측 쏠림 변위량 (발목-골반 중심선 대비)
    mid_hip_ankle = 0.5 * (lead_hip + lead_ankle)
    knee_drift_ml = np.sum((lead_knee - mid_hip_ankle) * ml_axis, axis=-1) * inward_sign
    lead_knee_medial_drift = float(np.sum((knee_drift_ml / pelvis_w) * win_mask) / win_cnt)

    # 9. 최하사점 부근 무릎 외반 최대 피크 각도
    lead_knee_valgus_peak = float(np.max(fppa_deg[win_mask]))

    # 10. 골반 너비 대비 선행 무릎의 횡방향 이탈 비율
    knee_hip_ml = np.sum((lead_knee - lead_hip) * ml_axis, axis=-1) * inward_sign
    lead_knee_lateral_ratio = float(np.sum((knee_hip_ml / pelvis_w) * win_mask) / win_cnt)

    # 11. 최하사점 진입 시 무릎 관절의 내측 횡방향 이동 속도
    knee_vel_ml = np.diff(knee_drift_ml) * 30.0 # fps=30
    lead_knee_velocity_ml = float(np.max(np.abs(knee_vel_ml[max(0, bottom_idx-6):min(T-1, bottom_idx+1)])))

    # 12. 무릎 중심과 발목 중심의 수직 정렬 오차
    lead_knee_ankle_alignment = float(np.sum(np.abs(femur_ml - tibia_ml) * win_mask) / (win_cnt * pelvis_w))

    # -------------------------------------------------------------------------
    # Group C: 몸통-골반 정렬 및 경사 피처 (6D)
    # -------------------------------------------------------------------------
    # 13. 체간 축의 전두면 롤 각도 (어깨선 기울기 및 체간 롤, '몸통 갸우뚱' 감지)
    trunk_dy = sh_r[:, 1] - sh_l[:, 1]
    trunk_dx = sh_r[:, 0] - sh_l[:, 0]
    trunk_roll = np.degrees(np.arctan2(trunk_dy, np.abs(trunk_dx) + 1e-7))
    trunk_lateral_lean_bottom = float(np.sum(np.abs(trunk_roll) * win_mask) / win_cnt)

    # 14. 전두면 상의 골반 수평 기울기각 (Trendelenburg Drop)
    pelvic_dy = hip_r[:, 1] - hip_l[:, 1]
    pelvic_dx = hip_r[:, 0] - hip_l[:, 0]
    pelvic_roll = np.degrees(np.arctan2(pelvic_dy, np.abs(pelvic_dx) + 1e-7))
    pelvic_obliquity_bottom = float(np.sum(np.abs(pelvic_roll) * win_mask) / win_cnt)

    # 15. 흉곽 평면 법선과 골반 축 법선 간의 수평면 비틀림(Torsion) 각도
    sh_axis = sh_l - sh_r
    sh_axis_norm = sh_axis / (np.linalg.norm(sh_axis, axis=-1, keepdims=True) + 1e-7)
    sh_ap = np.cross(sh_axis_norm, vt_axis)
    sh_ap = sh_ap / (np.linalg.norm(sh_ap, axis=-1, keepdims=True) + 1e-7)
    torsion_cos = np.sum(sh_ap * ap_axis, axis=-1)
    torsion_deg = np.degrees(np.arccos(np.clip(torsion_cos, -1.0, 1.0)))
    trunk_pelvis_yaw_torsion = float(np.sum(torsion_deg * win_mask) / win_cnt)

    # 16. 전체 동작 중 상체 측방 기울기 가동범위
    trunk_canting_rom = float(np.ptp(trunk_roll))

    # 17. 선행 발 반대 방향으로 상체가 보상 기울어짐을 계측한 점수
    lead_side_roll = trunk_roll * inward_sign
    trunk_counterbalance_score = float(np.sum(np.maximum(0.0, lead_side_roll) * win_mask) / win_cnt)

    # 18. 최하사점에서 골반 축의 회전 편차각
    pelvic_rotation_bottom = float(np.abs(pelvic_roll[bottom_idx]))

    # -------------------------------------------------------------------------
    # Group D: 상하체 결합 및 비폐색 프록시 피처 (6D)
    # -------------------------------------------------------------------------
    # 19. 선행 대퇴골의 관상면 내전 각도 (Hip Adduction, 비폐색 핵심 지표)
    femur_vec = lead_knee - lead_hip
    femur_proj_ml = np.sum(femur_vec * ml_axis, axis=-1) * inward_sign
    femur_proj_vt = np.sum(femur_vec * (-vt_axis), axis=-1) # 하향 방향
    adduction_deg = np.degrees(np.arctan2(femur_proj_ml, np.maximum(0.05, femur_proj_vt)))
    lead_hip_adduction_bottom = float(np.sum(np.maximum(0.0, adduction_deg) * win_mask) / win_cnt)

    # 20. 체간 전방 벡터와 선행 발 장축 간의 3차원 입체 편차각
    cos_tf = np.sum(ap_axis * foot_vec_norm, axis=-1)
    ang_tf = np.degrees(np.arccos(np.clip(cos_tf, -1.0, 1.0)))
    trunk_foot_angle_deviation = float(np.sum(ang_tf * win_mask) / win_cnt)

    # 21. 무릎 굴곡 방향과 발 장축 간의 비틀림 각도
    knee_flex_vec = (lead_knee - lead_hip) + (lead_ankle - lead_knee)
    knee_flex_norm = knee_flex_vec / (np.linalg.norm(knee_flex_vec, axis=-1, keepdims=True) + 1e-7)
    cos_kf = np.sum(knee_flex_norm * foot_vec_norm, axis=-1)
    knee_foot_torsion_angle = float(np.degrees(np.arccos(np.clip(cos_kf, -1.0, 1.0)))[bottom_idx])

    # 22. 양 무릎 간 3D 이격거리 대비 골반 너비 비율
    knee_dist = np.linalg.norm(knee_l - knee_r, axis=-1)
    inter_knee_dist_ratio = float(np.sum((knee_dist / pelvis_w) * win_mask) / win_cnt)

    # 23. 전두면 상에서 선행 허벅지의 수직 대비 경사각
    lead_femur_coronal_tilt = float(np.sum(np.abs(adduction_deg) * win_mask) / win_cnt)

    # 24. [몸통 - 대퇴 - 경골 - 발] 4개 분절의 3D 공간적 정렬 지수
    # 모든 분절의 내측 편차 제곱합 평균
    kinetic_chain_alignment_score = float(
        (lead_foot_inward_angle_bottom / 15.0)**2 +
        (lead_knee_fppa_bottom / 15.0)**2 +
        (trunk_lateral_lean_bottom / 10.0)**2 +
        (lead_hip_adduction_bottom / 12.0)**2
    )

    feats = np.array([
        lead_foot_inward_angle_bottom,
        lead_foot_yaw_rom,
        lead_ankle_toe_proj_ml,
        lead_foot_axis_stability,
        lead_foot_inward_ratio,
        lead_heel_toe_skew,
        lead_knee_fppa_bottom,
        lead_knee_medial_drift,
        lead_knee_valgus_peak,
        lead_knee_lateral_ratio,
        lead_knee_velocity_ml,
        lead_knee_ankle_alignment,
        trunk_lateral_lean_bottom,
        pelvic_obliquity_bottom,
        trunk_pelvis_yaw_torsion,
        trunk_canting_rom,
        trunk_counterbalance_score,
        pelvic_rotation_bottom,
        lead_hip_adduction_bottom,
        trunk_foot_angle_deviation,
        knee_foot_torsion_angle,
        inter_knee_dist_ratio,
        lead_femur_coronal_tilt,
        kinetic_chain_alignment_score
    ], dtype=np.float32)

    # NaN / Inf 안전 처리
    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    return feats


def extract_lunge_c2_features_batch(X_seqs: np.ndarray) -> np.ndarray:
    """
    (N, 64, 33, 3) 배치 시퀀스로부터 (N, 24) 피처 행렬 고속 추출
    """
    N = X_seqs.shape[0]
    feats_arr = np.zeros((N, 24), dtype=np.float32)
    for i in range(N):
        feats_arr[i] = extract_lunge_c2_features(X_seqs[i])
    return feats_arr


if __name__ == '__main__':
    # 간단한 더미 텐서 테스트
    dummy = np.random.randn(64, 33, 3).astype(np.float32)
    f = extract_lunge_c2_features(dummy)
    print(f"[+] Lunge C2 Features extracted successfully! Shape: {f.shape}")
    assert f.shape == (24,), f"Expected shape (24,), got {f.shape}"
    print("[+] All assertions passed!")
