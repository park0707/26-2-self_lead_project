# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: lunge_c2_features_v3.py
Purpose: 전수 피처 감사 및 다중공선성/잡음 제거를 통해 도출된 '17차원 정예 생체역학 피처' 추출기
         - R_body 강체 직교계 3D 정규화 기반
         - 100% 완전 중복 피처(fpa_inward_angle_body) 및 0분산 피처(trunk_lateral_lean_deg_loc) 원천 제거
         - 고상관 다중공선성 및 저변별력 잡음 피처 13개 배제
         - 판별력 최상위 17개 정예 피처 구성
"""

import numpy as np
from typing import Dict, List, Tuple, Any

FEATURE_NAMES_17D = [
    # Group A: 선행 발 지향 및 족부 안정성 (4D)
    "lead_foot_inward_angle_bottom",        # 1. 최하사점 선행 발목-발끝 벡터와 체간 정면 법선의 수평면 내각 (내회전 직접 계측)
    "lead_foot_axis_stability",             # 2. 하강/상승 구간 발끝 방향 벡터의 요동 분산
    "lead_foot_inward_ratio",               # 3. 발끝이 12° 이상 안쪽으로 꺾여 있는 프레임 비율
    "lead_heel_toe_skew",                   # 4. 발뒤꿈치 대비 발끝의 내측 편차 거리비

    # Group B: 슬관절 동적 정렬 및 외반 변위 (4D)
    "lead_knee_valgus_peak",                # 5. 최하사점 부근 무릎 외반 최대 피크 각도
    "lead_knee_lateral_ratio",              # 6. 골반 너비 대비 선행 무릎의 횡방향 이탈 비율 (F-stat 32.77)
    "lead_knee_velocity_ml",                # 7. 최하사점 진입 시 무릎 관절의 내측 횡방향 이동 속도
    "lead_knee_ankle_alignment",            # 8. 무릎 중심과 발목 중심의 수직 정렬 오차

    # Group C: 관상면 동적 변화량 및 하지 연쇄 (2D)
    "fppa_eccentric_to_bottom_delta",       # 9. 신전 상태 대비 최하사점에서의 FPPA 각도 붕괴량 (Dynamic Collapse)
    "directional_misalignment_thigh_foot",  # 10. 신체 직교계 상 대퇴 진행 벡터 vs 발 장축 사잇각 (Directional Torsion)

    # Group D: 몸통-골반 정렬 및 회전 (4D)
    "trunk_pelvis_yaw_torsion",             # 11. 흉곽 평면 법선과 골반 축 법선 간의 수평면 비틀림(Torsion) 각도
    "trunk_canting_rom",                    # 12. 전체 동작 중 상체 측방 기울기 가동범위
    "trunk_counterbalance_score",           # 13. 선행 발 반대 방향으로 상체가 보상 기울어짐을 계측한 지표
    "pelvic_rotation_bottom",               # 14. 최하사점에서 골반 축의 수평 회전 편차각 (F-stat 23.34)

    # Group E: 상하체 결합 및 비폐색 상위 프록시 (3D)
    "lead_hip_adduction_bottom",            # 15. 선행 대퇴골의 관상면 내전 각도 (F-stat 28.17)
    "trunk_foot_angle_deviation",           # 16. 체간 전방 벡터와 선행 발 장축 간의 3차원 입체 편차각
    "inter_knee_dist_ratio"                 # 17. 양 무릎 간 3D 이격거리 대비 골반 너비 비율
]


def build_body_orthonormal_basis(kps_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 프레임(33, 3)에서 양 어깨와 양 골반 4점을 이용하여 신체 고유 직교 좌표계(R_body, center) 구축
    """
    l_sh = kps_frame[11, :3]
    r_sh = kps_frame[12, :3]
    l_hip = kps_frame[23, :3]
    r_hip = kps_frame[24, :3]

    hip_center = (l_hip + r_hip) * 0.5
    sh_center = (l_sh + r_sh) * 0.5
    pelvis_vec = l_hip - r_hip

    v_trunk = sh_center - hip_center
    norm_v = np.linalg.norm(v_trunk)
    if norm_v < 1e-6:
        v_trunk = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        v_trunk /= norm_v

    norm_p = np.linalg.norm(pelvis_vec)
    if norm_p < 1e-6:
        pelvis_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        pelvis_vec /= norm_p

    w_ap = np.cross(pelvis_vec, v_trunk)
    norm_w = np.linalg.norm(w_ap)
    if norm_w < 1e-6:
        w_ap = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        w_ap /= norm_w

    u_ml = np.cross(v_trunk, w_ap)
    norm_u = np.linalg.norm(u_ml)
    if norm_u < 1e-6:
        u_ml = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        u_ml /= norm_u

    R_body = np.vstack([u_ml, v_trunk, w_ap])
    return R_body, hip_center


def transform_to_body_coordinates(kps_seq: np.ndarray) -> np.ndarray:
    """
    (64, 33, 3) 시퀀스를 각 프레임별 신체 직교 좌표계로 변환 (64, 33, 3)
    """
    T, V, C = kps_seq.shape
    kps_local = np.zeros_like(kps_seq)
    for t in range(T):
        R, center = build_body_orthonormal_basis(kps_seq[t])
        diff = kps_seq[t] - center[np.newaxis, :]
        kps_local[t] = diff @ R.T
    return kps_local


def detect_lead_leg(kps_seq: np.ndarray) -> Tuple[bool, int]:
    """
    런지 선행 다리 식별 (True: 왼다리 선행, False: 오른다리 선행) 및 최하사점 프레임 인덱스 반환
    """
    l_knee_y = kps_seq[:, 25, 1]
    r_knee_y = kps_seq[:, 26, 1]
    l_ankle_z = kps_seq[:, 27, 2]
    r_ankle_z = kps_seq[:, 28, 2]

    l_max_y_idx = int(np.argmax(l_knee_y))
    r_max_y_idx = int(np.argmax(r_knee_y))

    if l_knee_y[l_max_y_idx] >= r_knee_y[r_max_y_idx]:
        is_left_lead = True
        bottom_idx = l_max_y_idx
    else:
        is_left_lead = False
        bottom_idx = r_max_y_idx

    l_step_fwd = np.min(l_ankle_z) - np.mean(l_ankle_z[:5])
    r_step_fwd = np.min(r_ankle_z) - np.mean(r_ankle_z[:5])
    if abs(l_knee_y[l_max_y_idx] - r_knee_y[r_max_y_idx]) < 0.05:
        is_left_lead = bool(l_step_fwd < r_step_fwd)

    return is_left_lead, bottom_idx


def extract_lunge_c2_features_v3(kps_seq_3d: np.ndarray) -> np.ndarray:
    """
    (64, 33, 3) MediaPipe 3D 포즈 시퀀스로부터 최적화된 17차원 정예 피처 벡터 추출
    """
    if kps_seq_3d.shape[-1] == 4:
        kps_seq_3d = kps_seq_3d[:, :, :3]

    T, V, C = kps_seq_3d.shape
    is_left_lead, t_bot = detect_lead_leg(kps_seq_3d)

    kps_local = transform_to_body_coordinates(kps_seq_3d)

    bot_start = max(0, t_bot - 4)
    bot_end = min(T, t_bot + 5)

    if is_left_lead:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 23, 25, 27, 29, 31
        opp_knee_idx = 26
        lat_sign = 1.0
    else:
        hip_idx, knee_idx, ank_idx, heel_idx, toe_idx = 24, 26, 28, 30, 32
        opp_knee_idx = 25
        lat_sign = -1.0

    pelvis_widths = np.linalg.norm(kps_seq_3d[:, 23, :3] - kps_seq_3d[:, 24, :3], axis=1)
    avg_pelvis_w = float(np.mean(pelvis_widths)) + 1e-6

    # 1. lead_foot_inward_angle_bottom
    foot_vec_bot = np.mean(kps_seq_3d[bot_start:bot_end, toe_idx, :3] - kps_seq_3d[bot_start:bot_end, ank_idx, :3], axis=0)
    foot_horiz = np.array([foot_vec_bot[0], foot_vec_bot[2]], dtype=np.float32)
    norm_fh = np.linalg.norm(foot_horiz) + 1e-6
    foot_horiz /= norm_fh
    pelvis_bot = np.mean(kps_seq_3d[bot_start:bot_end, 23, :3] - kps_seq_3d[bot_start:bot_end, 24, :3], axis=0)
    body_fwd = np.array([-pelvis_bot[2], pelvis_bot[0]], dtype=np.float32)
    body_fwd /= (np.linalg.norm(body_fwd) + 1e-6)
    dot_fpa = np.clip(np.dot(body_fwd, foot_horiz), -1.0, 1.0)
    lead_foot_inward_angle_bottom = float(np.degrees(np.arccos(dot_fpa)))

    # 2. lead_foot_axis_stability
    foot_vecs = kps_seq_3d[:, toe_idx, :3] - kps_seq_3d[:, ank_idx, :3]
    foot_yaws = np.degrees(np.arctan2(foot_vecs[:, 0], foot_vecs[:, 2] + 1e-6))
    lead_foot_axis_stability = float(np.var(foot_yaws))

    # 3. lead_foot_inward_ratio
    inward_frames = np.sum(foot_yaws * lat_sign < -12.0)
    lead_foot_inward_ratio = float(inward_frames / T)

    # 4. lead_heel_toe_skew
    heel_toe_dx = (kps_seq_3d[bot_start:bot_end, toe_idx, 0] - kps_seq_3d[bot_start:bot_end, heel_idx, 0]) * lat_sign
    lead_heel_toe_skew = float(np.mean(heel_toe_dx) / avg_pelvis_w)

    # 5. lead_knee_valgus_peak
    thigh_v = kps_seq_3d[:, hip_idx, :3] - kps_seq_3d[:, knee_idx, :3]
    shank_v = kps_seq_3d[:, ank_idx, :3] - kps_seq_3d[:, knee_idx, :3]
    thigh_u = thigh_v / (np.linalg.norm(thigh_v, axis=1, keepdims=True) + 1e-6)
    shank_u = shank_v / (np.linalg.norm(shank_v, axis=1, keepdims=True) + 1e-6)
    knee_angles = np.degrees(np.arccos(np.clip(np.sum(thigh_u * shank_u, axis=1), -1.0, 1.0)))
    lead_knee_valgus_peak = float(np.max(np.abs(180.0 - knee_angles[bot_start:bot_end])))

    # 6. lead_knee_lateral_ratio
    pelvis_c_x = (kps_seq_3d[:, 23, 0] + kps_seq_3d[:, 24, 0]) * 0.5
    knee_dev_x = (kps_seq_3d[:, knee_idx, 0] - pelvis_c_x) * lat_sign
    lead_knee_lateral_ratio = float(np.mean(knee_dev_x[bot_start:bot_end]) / avg_pelvis_w)

    # 7. lead_knee_velocity_ml
    knee_v_ml = np.gradient(kps_seq_3d[:, knee_idx, 0])
    lead_knee_velocity_ml = float(np.mean(knee_v_ml[bot_start:bot_end]) * lat_sign)

    # 8. lead_knee_ankle_alignment
    knee_ank_dx = np.abs(kps_seq_3d[bot_start:bot_end, knee_idx, 0] - kps_seq_3d[bot_start:bot_end, ank_idx, 0])
    lead_knee_ankle_alignment = float(np.mean(knee_ank_dx) / avg_pelvis_w)

    # 9. fppa_eccentric_to_bottom_delta
    hip_loc = kps_local[:, hip_idx, :]
    knee_loc = kps_local[:, knee_idx, :]
    ank_loc = kps_local[:, ank_idx, :]
    v_femur_cor = hip_loc[:, :2] - knee_loc[:, :2]
    v_tibia_cor = ank_loc[:, :2] - knee_loc[:, :2]
    u_f = v_femur_cor / (np.linalg.norm(v_femur_cor, axis=1, keepdims=True) + 1e-6)
    u_t = v_tibia_cor / (np.linalg.norm(v_tibia_cor, axis=1, keepdims=True) + 1e-6)
    fppa_body_arr = np.degrees(np.arccos(np.clip(np.sum(u_f * u_t, axis=1), -1.0, 1.0)))
    fppa_eccentric = np.mean(fppa_body_arr[:10])
    fppa_bottom_mean = np.mean(fppa_body_arr[bot_start:bot_end])
    fppa_eccentric_to_bottom_delta = float(fppa_eccentric - fppa_bottom_mean)

    # 10. directional_misalignment_thigh_foot
    thigh_vec_body = knee_loc[bot_start:bot_end, :] - hip_loc[bot_start:bot_end, :]
    foot_vec_body = kps_local[bot_start:bot_end, toe_idx, :] - kps_local[bot_start:bot_end, ank_idx, :]
    thigh_plane = np.mean(thigh_vec_body[:, [0, 2]], axis=0)
    foot_plane = np.mean(foot_vec_body[:, [0, 2]], axis=0)
    thigh_p_u = thigh_plane / (np.linalg.norm(thigh_plane) + 1e-6)
    foot_p_u = foot_plane / (np.linalg.norm(foot_plane) + 1e-6)
    directional_misalignment_thigh_foot = float(np.degrees(np.arccos(np.clip(np.dot(thigh_p_u, foot_p_u), -1.0, 1.0))))

    # 11. trunk_pelvis_yaw_torsion
    sh_vec = kps_seq_3d[:, 11, :3] - kps_seq_3d[:, 12, :3]
    sh_yaw = np.degrees(np.arctan2(sh_vec[:, 2], sh_vec[:, 0] + 1e-6))
    pel_vec = kps_seq_3d[:, 23, :3] - kps_seq_3d[:, 24, :3]
    pel_yaw = np.degrees(np.arctan2(pel_vec[:, 2], pel_vec[:, 0] + 1e-6))
    trunk_pelvis_yaw_torsion = float(np.mean(np.abs(sh_yaw[bot_start:bot_end] - pel_yaw[bot_start:bot_end])))

    # 12. trunk_canting_rom
    trunk_dx = (kps_seq_3d[:, 11, 0] + kps_seq_3d[:, 12, 0]) * 0.5 - pelvis_c_x
    trunk_dy = (kps_seq_3d[:, 11, 1] + kps_seq_3d[:, 12, 1]) * 0.5 - (kps_seq_3d[:, 23, 1] + kps_seq_3d[:, 24, 1]) * 0.5
    trunk_angles = np.degrees(np.arctan2(trunk_dx, np.abs(trunk_dy) + 1e-6))
    trunk_canting_rom = float(np.ptp(trunk_angles))

    # 13. trunk_counterbalance_score
    trunk_roll_bot = np.mean(trunk_angles[bot_start:bot_end])
    trunk_counterbalance_score = float(trunk_roll_bot * (-lat_sign))

    # 14. pelvic_rotation_bottom
    pel_rot_bot = np.degrees(np.arctan2(np.mean(pel_vec[bot_start:bot_end, 2]), np.mean(pel_vec[bot_start:bot_end, 0]) + 1e-6))
    pelvic_rotation_bottom = float(abs(pel_rot_bot))

    # 15. lead_hip_adduction_bottom
    hip_knee_v = kps_seq_3d[:, knee_idx, :3] - kps_seq_3d[:, hip_idx, :3]
    hip_add_deg = np.degrees(np.arctan2(hip_knee_v[:, 0] * lat_sign, np.abs(hip_knee_v[:, 1]) + 1e-6))
    lead_hip_adduction_bottom = float(np.mean(hip_add_deg[bot_start:bot_end]))

    # 16. trunk_foot_angle_deviation
    trunk_fwd_3d = np.cross(sh_vec, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    norm_tf = np.linalg.norm(trunk_fwd_3d, axis=1, keepdims=True) + 1e-6
    trunk_fwd_3d /= norm_tf
    norm_fv = np.linalg.norm(foot_vecs, axis=1, keepdims=True) + 1e-6
    foot_u_3d = foot_vecs / norm_fv
    trunk_foot_dot = np.clip(np.sum(trunk_fwd_3d * foot_u_3d, axis=1), -1.0, 1.0)
    trunk_foot_angle_deviation = float(np.mean(np.degrees(np.arccos(trunk_foot_dot))[bot_start:bot_end]))

    # 17. inter_knee_dist_ratio
    inter_knee_d = np.linalg.norm(kps_seq_3d[:, 25, :3] - kps_seq_3d[:, 26, :3], axis=1)
    inter_knee_dist_ratio = float(np.mean(inter_knee_d[bot_start:bot_end]) / avg_pelvis_w)

    feats = np.array([
        lead_foot_inward_angle_bottom,
        lead_foot_axis_stability,
        lead_foot_inward_ratio,
        lead_heel_toe_skew,
        lead_knee_valgus_peak,
        lead_knee_lateral_ratio,
        lead_knee_velocity_ml,
        lead_knee_ankle_alignment,
        fppa_eccentric_to_bottom_delta,
        directional_misalignment_thigh_foot,
        trunk_pelvis_yaw_torsion,
        trunk_canting_rom,
        trunk_counterbalance_score,
        pelvic_rotation_bottom,
        lead_hip_adduction_bottom,
        trunk_foot_angle_deviation,
        inter_knee_dist_ratio
    ], dtype=np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    return feats


def extract_lunge_c2_features_v3_batch(X_seqs: np.ndarray) -> np.ndarray:
    """
    (N, 64, 33, 3) 시퀀스로부터 (N, 17) 피처 행렬 일괄 추출
    """
    N = X_seqs.shape[0]
    feats = np.zeros((N, 17), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_lunge_c2_features_v3(X_seqs[i])
    return feats


if __name__ == '__main__':
    dummy = np.random.randn(64, 33, 3).astype(np.float32)
    f = extract_lunge_c2_features_v3(dummy)
    print(f"[+] Lunge C2 v3 Features extracted! Shape: {f.shape}")
    assert f.shape == (17,), f"Expected (17,), got {f.shape}"
    print("[+] All assertions passed for v3 17D features!")
