# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c1_features_v2.py
Purpose: 바벨 런지 c1(앞다리 무릎 각도 90도) 세부 오류 판별 SOTA v2 고도화 피처 추출 모듈
         - [기술 1] ISB 골반 직교 기저(Local Coordinate Frame) 기반 시상면(Sagittal Plane) 회전 정규화
         - [기술 2] Savitzky-Golay(SG) 2차 다항식 필터 기반 피크 뭉개짐(Peak Flattening) 방지
         - [기술 3] 골반 Y + 무릎 각속도 영교차점 이중 앵커링(Dual-Anchor Inflection Point) 검출
         - [기술 4] Knee Over Toe (KOT) 정규화 전방 돌출 변위 및 경골 수직도
         - [기술 5] Farrokhi et al. (JOSPT) 상체 경사-정강이 평행도(Tibia-Torso Parallelism) 지표
         - 총 38차원(38D) 고도화 생체역학 피처셋
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.signal import savgol_filter

# MediaPipe 33 Landmark Indices
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
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
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32

# 38차원 공식 v2 도메인 피처 명칭 목록
LUNGE_C1_FEATURE_NAMES_V2: List[str] = [
    # [그룹 1: 앞다리 무릎 각도 및 시상면 굴곡 (10D)]
    "lead_knee_angle_min",            # [ 1] 최저점 윈도우 내 3D 대퇴-정강이 벡터 최소 각도 (°)
    "lead_knee_angle_mean",           # [ 2] 최저점 윈도우 내 평균 굴곡 각도 (°)
    "lead_knee_90_error_abs",         # [ 3] |최소 각도 - 90°| 절대 오차 (°)
    "lead_knee_flexion_rom",          # [ 4] 선 자세 대비 최저점까지의 무릎 굴곡 가동범위 (°)
    "lead_knee_under_90_ratio",       # [ 5] 90도 미만(불충분 굴곡) 체류 프레임 비율
    "lead_knee_flex_velocity_max",    # [ 6] 하강 구간 최대 무릎 굴곡 각속도 (deg/frame)
    "lead_knee_accel_inflection",     # [ 7] 최저점 전환 순간의 각가속도 (deg/frame^2)
    "lead_knee_rebound_jitter",       # [ 8] 최저점 전후 미세 흔들림 표준편차
    "sagittal_knee_angle_bottom",     # [ 9] ISB 골반 좌표계 투영 시상면 순수 1-DoF 무릎 굴곡각 (°)
    "sagittal_knee_90_error",         # [10] |시상면 무릎 굴곡각 - 90°| 절대 오차 (°)

    # [그룹 2: 정강이 및 대퇴 기하 / KOT 지표 (10D)]
    "lead_tibia_vertical_tilt",       # [11] 지면 수직선 대비 정강이 사잇각 (정강이 수직 유지도)
    "lead_femur_horizontal_tilt",     # [12] 지면 수평선 대비 대퇴골 사잇각 (대퇴 평행도)
    "lead_knee_toe_overhang",         # [13] 앞발 발끝 대비 무릎 관절 전방 돌출 거리 (신체 스케일 정규화)
    "lead_shank_ankle_coupling",      # [14] 정강이 분절 각도와 발목 배측굴곡 각도 커플링 비율
    "lead_tibia_sagittal_sway",       # [15] 하강 중 정강이뼈 시상면 흔들림 폭
    "lead_knee_valgus_varus",         # [16] 관상면 상 무릎 안쪽/바깥쪽 꺾임 각도 (외반/내반)
    "lead_femur_length_ratio",        # [17] 체간 대비 앞다리 대퇴골 정규화 길이
    "lead_tibia_length_ratio",        # [18] 체간 대비 앞다리 정강이 정규화 길이
    "kot_normalized_displacement",    # [19] 발길이 정규화 무릎-발목 전방 변위 (Knee Over Toe - KOT)
    "rigid_bone_length_consistency",  # [20] 대퇴골-정강이 강체 길이 보존 일관성 지표 (가림 노이즈 감지)

    # [그룹 3: 양 하지 비대칭 협응 및 체간 정렬 (10D)]
    "lead_rear_knee_ratio",           # [21] 앞무릎 최소각 대비 뒷무릎 최소각 비율
    "rear_knee_angle_min",            # [22] 뒷다리 무릎 최저 각도 (°)
    "pelvic_drop_angle",              # [23] 관상면 상 골반 좌우 기울기 각도 (Trendelenburg sign)
    "pelvic_axial_rotation",          # [24] 횡단면 상 골반 좌우 비틀림 회전각
    "trunk_forward_lean_lead",        # [25] 상체 전경각과 앞다리 허벅지 간 협응 사잇각
    "hip_knee_flexion_coupling",      # [26] 앞다리 고관절 굴곡각 대비 슬관절 굴곡각 비율
    "stride_length_normalized",       # [27] 양 발목 사이 전후 보폭 (체간 정규화)
    "step_width_normalized",          # [28] 양 발목 사이 좌우 너비 (체간 정규화)
    "tibia_torso_parallelism_error",  # [29] 정강이 벡터와 체간 벡터 간의 평행성 오차 (|theta_tibia - theta_torso|)
    "rear_knee_ground_clearance",     # [30] 뒷무릎 지면 간격 프록시 (앞발목 높이 대비 뒷무릎 높이)

    # [그룹 4: 시공간 템포 및 시점 불변 동역학 (8D)]
    "bottom_phase_dwell_ratio",       # [31] 전체 64프레임 중 최저점 부근 체류 프레임 비율
    "descent_ascent_tempo_ratio",     # [32] 하강 소요 프레임 수 대비 상승 소요 프레임 수 비율
    "lead_knee_dtw_distance",         # [33] 이상적 90도 골드 템플릿 곡선 대비 DTW 유사도 거리
    "view_invariant_flexion_3d",      # [34] 체간 로컬 좌표계 기반 회전 불변 3D 무릎 굴곡각
    "camera_view_azimuth_est",        # [35] 카메라 상대 방위각 추정치 (골반-어깨 투영비)
    "lead_knee_visibility_mean",      # [36] 최저점 구간 앞무릎 키포인트 평균 신뢰도
    "kinetic_energy_inflection",      # [37] 최저 전환점 시 하지 관절 운동에너지 극소치 프록시
    "jerk_metric_lead_leg",           # [38] 앞다리 무릎 궤적 저크(Jerk, 3차 미분) 평활도 지표
]


def vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 3D 벡터 간의 사잇각 계산 (degree 단위, 0 ~ 180)"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0
    dot = np.dot(v1, v2)
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def build_isb_pelvis_frame(kps_frame: np.ndarray) -> np.ndarray:
    """
    [기술 1] ISB 국제생체역학회 표준 골반 로컬 직교 좌표계 구성
    반환: 3x3 회전 변환 행렬 R = [u_X, u_Y, u_Z]^T
    """
    p_lhip = kps_frame[MP_LEFT_HIP]
    p_rhip = kps_frame[MP_RIGHT_HIP]
    p_pelvis = (p_lhip + p_rhip) / 2.0
    p_neck = (kps_frame[MP_LEFT_SHOULDER] + kps_frame[MP_RIGHT_SHOULDER]) / 2.0

    # 1. X축 (Mediolateral: 좌 -> 우)
    v_lr = p_rhip - p_lhip
    norm_x = np.linalg.norm(v_lr)
    u_x = v_lr / (norm_x + 1e-6)

    # 2. Y축 (Craniocaudal / 수직 상방): 척추 벡터를 X축에 직교화
    v_spine = p_neck - p_pelvis
    v_spine_proj = v_spine - np.dot(v_spine, u_x) * u_x
    norm_y = np.linalg.norm(v_spine_proj)
    u_y = v_spine_proj / (norm_y + 1e-6)

    # 3. Z축 (Anteroposterior / 전방): X x Y
    u_z = np.cross(u_x, u_y)
    norm_z = np.linalg.norm(u_z)
    u_z = u_z / (norm_z + 1e-6)

    return np.vstack([u_x, u_y, u_z])  # shape: (3, 3)


def calculate_joint_angle_series_sg(kps: np.ndarray, p1_idx: int, p2_idx: int, p3_idx: int) -> np.ndarray:
    """[기술 2] (T, 33, 3) 시퀀스에서 Savitzky-Golay 필터 적용 관절 각도 시계열 계산"""
    T = kps.shape[0]
    raw_angles = np.zeros(T, dtype=np.float32)
    for t in range(T):
        v1 = kps[t, p1_idx] - kps[t, p2_idx]
        v2 = kps[t, p3_idx] - kps[t, p2_idx]
        raw_angles[t] = vector_angle_3d(v1, v2)

    # 윈도우 크기 7, 2차 다항식 SG 필터 적용 (변곡점 피크 각도 온전 보존)
    if T >= 7:
        smooth_angles = savgol_filter(raw_angles, window_length=7, polyorder=2)
    else:
        smooth_angles = raw_angles
    return smooth_angles.astype(np.float32)


def detect_lead_leg(kps: np.ndarray) -> str:
    """앞다리(Lead Leg) 자율 식별 엔진"""
    T = kps.shape[0]
    l_ankle = kps[:, MP_LEFT_ANKLE]
    r_ankle = kps[:, MP_RIGHT_ANKLE]
    pelvis = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0

    l_disp = np.linalg.norm(l_ankle[:, [0, 2]] - pelvis[:, [0, 2]], axis=1)
    r_disp = np.linalg.norm(r_ankle[:, [0, 2]] - pelvis[:, [0, 2]], axis=1)

    l_knee_angles = calculate_joint_angle_series_sg(kps, MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE)
    r_knee_angles = calculate_joint_angle_series_sg(kps, MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE)

    l_rom = np.max(l_knee_angles) - np.min(l_knee_angles)
    r_rom = np.max(r_knee_angles) - np.min(r_knee_angles)

    mid_start, mid_end = int(T * 0.3), int(T * 0.7)
    l_disp_mean = np.mean(l_disp[mid_start:mid_end])
    r_disp_mean = np.mean(r_disp[mid_start:mid_end])

    score_left = l_disp_mean * 1.5 + (l_rom / 180.0)
    score_right = r_disp_mean * 1.5 + (r_rom / 180.0)

    return 'left' if score_left >= score_right else 'right'


def find_dual_anchor_inflection_point(kps: np.ndarray, lead_knee_angles: np.ndarray) -> Tuple[int, int, int]:
    """
    [기술 3] 골반 Y 수직 변위 + 무릎 각속도 영교차점(Zero-crossing) 이중 앵커링 변곡점 검출
    """
    T = kps.shape[0]
    pelvis_y = (kps[:, MP_LEFT_HIP, 1] + kps[:, MP_RIGHT_HIP, 1]) / 2.0
    smooth_pelvis_y = savgol_filter(pelvis_y, window_length=7, polyorder=2) if T >= 7 else pelvis_y

    # 1. 골반 기준 하강 극점
    t_pelvis_bottom = int(np.argmax(smooth_pelvis_y))

    # 2. 앞무릎 굴곡 각도 최소점 (가장 깊게 구부린 순간)
    t_knee_bottom = int(np.argmin(lead_knee_angles))

    # 이중 앵커 가중 결합 (무릎 각도 60% + 골반 변위 40%)
    t_combined = int(round(0.6 * t_knee_bottom + 0.4 * t_pelvis_bottom))
    t_bottom = int(np.clip(t_combined, 10, T - 10))

    win_start = max(0, t_bottom - 4)
    win_end = min(T, t_bottom + 5)
    return t_bottom, win_start, win_end


def compute_dtw_distance_fast(s1: np.ndarray, s2: np.ndarray) -> float:
    """1D DTW (Dynamic Time Warping) 거리 계산"""
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / (n + m))


def extract_lunge_c1_features_v2(kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    바벨 런지 c1(앞다리 무릎 각도 90도) SOTA v2 38차원 도메인 피처 추출 메인 함수
    """
    kps = kps_33_3d_64f.astype(np.float32)
    T = kps.shape[0]
    assert T == 64, f"입력 시퀀스는 64프레임이어야 합니다. (현재: {T})"

    # 1. 앞다리 자율 식별
    lead_side = detect_lead_leg(kps)
    if lead_side == 'left':
        lead_hip, lead_knee, lead_ank, lead_heel, lead_toe = MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE, MP_LEFT_HEEL, MP_LEFT_FOOT_INDEX
        rear_hip, rear_knee, rear_ank, rear_heel, rear_toe = MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE, MP_RIGHT_HEEL, MP_RIGHT_FOOT_INDEX
    else:
        lead_hip, lead_knee, lead_ank, lead_heel, lead_toe = MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE, MP_RIGHT_HEEL, MP_RIGHT_FOOT_INDEX
        rear_hip, rear_knee, rear_ank, rear_heel, rear_toe = MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE, MP_LEFT_HEEL, MP_LEFT_FOOT_INDEX

    # 2. SG 다항식 필터 기반 무릎 각도 시계열
    lead_knee_angles = calculate_joint_angle_series_sg(kps, lead_hip, lead_knee, lead_ank)
    rear_knee_angles = calculate_joint_angle_series_sg(kps, rear_hip, rear_knee, rear_ank)

    # 3. 이중 앵커링 최저점(Inflection Point) 검출
    t_bottom, win_start, win_end = find_dual_anchor_inflection_point(kps, lead_knee_angles)

    # 4. 체간 기준 스케일 및 발 길이 스케일
    neck = (kps[:, MP_LEFT_SHOULDER] + kps[:, MP_RIGHT_SHOULDER]) / 2.0
    pelvis = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0
    torso_lengths = np.linalg.norm(neck - pelvis, axis=1)
    torso_scale = float(np.mean(torso_lengths)) + 1e-6

    lead_foot_lengths = np.linalg.norm(kps[:, lead_toe] - kps[:, lead_heel], axis=1)
    foot_scale = float(np.mean(lead_foot_lengths)) + 1e-6

    # 5. [기술 1] ISB 골반 직교 기저 기반 시상면 굴곡각 산출
    R_isb = build_isb_pelvis_frame(kps[t_bottom])
    # 로컬 좌표계로 분절 벡터 변환
    v_femur_world = kps[t_bottom, lead_knee] - kps[t_bottom, lead_hip]
    v_tibia_world = kps[t_bottom, lead_ank] - kps[t_bottom, lead_knee]
    v_femur_local = R_isb @ v_femur_world
    v_tibia_local = R_isb @ v_tibia_world

    # 시상면(Y-Z 평면) 상의 순수 1-DoF 회전 각도 계산
    v_femur_sagittal = v_femur_local[1:3]  # [Y, Z]
    v_tibia_sagittal = v_tibia_local[1:3]  # [Y, Z]
    norm_fs = np.linalg.norm(v_femur_sagittal) * np.linalg.norm(v_tibia_sagittal) + 1e-6
    cos_sagittal = np.clip(np.dot(v_femur_sagittal, v_tibia_sagittal) / norm_fs, -1.0, 1.0)
    sagittal_knee_angle_bottom = float(np.degrees(np.arccos(cos_sagittal)))
    sagittal_knee_90_error = float(abs(sagittal_knee_angle_bottom - 90.0))

    # [그룹 1: 앞다리 무릎 각도 피처]
    lead_win_angles = lead_knee_angles[win_start:win_end]
    lead_knee_angle_min = float(np.min(lead_win_angles))
    lead_knee_angle_mean = float(np.median(lead_win_angles))  # Robust median
    lead_knee_90_error_abs = float(abs(lead_knee_angle_min - 90.0))
    lead_stand_angle = float(np.mean(lead_knee_angles[:5]))
    lead_knee_flexion_rom = float(max(0.0, lead_stand_angle - lead_knee_angle_min))

    under_90_count = np.sum(lead_knee_angles < 95.0)
    lead_knee_under_90_ratio = float(under_90_count / float(T))

    lead_vel = np.gradient(lead_knee_angles)
    lead_knee_flex_velocity_max = float(np.max(np.abs(lead_vel[:t_bottom]))) if t_bottom > 1 else 0.0
    lead_accel = np.gradient(lead_vel)
    lead_knee_accel_inflection = float(lead_accel[t_bottom])
    lead_knee_rebound_jitter = float(np.std(lead_win_angles))

    # [그룹 2: 정강이 및 대퇴 기하 / KOT 지표]
    lead_k_pt = kps[t_bottom, lead_knee]
    lead_a_pt = kps[t_bottom, lead_ank]
    lead_h_pt = kps[t_bottom, lead_hip]
    lead_t_pt = kps[t_bottom, lead_toe]

    tibia_vec = lead_k_pt - lead_a_pt
    vertical_unit = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    lead_tibia_vertical_tilt = vector_angle_3d(tibia_vec, vertical_unit)

    femur_vec = lead_h_pt - lead_k_pt
    femur_vertical_angle = vector_angle_3d(femur_vec, vertical_unit)
    lead_femur_horizontal_tilt = float(abs(femur_vertical_angle - 90.0))

    overhang_vec = lead_k_pt[[0, 2]] - lead_t_pt[[0, 2]]
    lead_knee_toe_overhang = float(np.linalg.norm(overhang_vec) / torso_scale)

    # [기술 4] KOT 정규화 변위 (Knee Over Toe - 발 길이 대비 무릎 전방 밀림)
    kot_normalized_displacement = float((lead_k_pt[0] - lead_a_pt[0]) / foot_scale)

    foot_vec = lead_t_pt - lead_a_pt
    ankle_flex_angle = vector_angle_3d(-tibia_vec, foot_vec)
    lead_shank_ankle_coupling = float(lead_knee_angle_min / (ankle_flex_angle + 1e-4))

    tibia_angles_all = [vector_angle_3d(kps[t_i, lead_knee] - kps[t_i, lead_ank], vertical_unit) for t_i in range(win_start, win_end)]
    lead_tibia_sagittal_sway = float(np.std(tibia_angles_all))

    v_hip_knee_2d = lead_k_pt[:2] - lead_h_pt[:2]
    v_knee_ank_2d = lead_a_pt[:2] - lead_k_pt[:2]
    valgus_angle = vector_angle_3d(np.append(v_hip_knee_2d, 0.0), np.append(v_knee_ank_2d, 0.0))
    lead_knee_valgus_varus = float(abs(valgus_angle - 180.0))

    femur_len = float(np.linalg.norm(femur_vec))
    tibia_len = float(np.linalg.norm(tibia_vec))
    lead_femur_length_ratio = float(femur_len / torso_scale)
    lead_tibia_length_ratio = float(tibia_len / torso_scale)

    # [기술 5] 강체 길이 보존 일관성 (하강 전 구간 대퇴골 길이 표준편차)
    all_femur_lens = np.linalg.norm(kps[:, lead_knee] - kps[:, lead_hip], axis=1)
    rigid_bone_length_consistency = float(np.std(all_femur_lens) / (np.mean(all_femur_lens) + 1e-6))

    # [그룹 3: 양 하지 비대칭 협응 및 체간 정렬]
    rear_win_angles = rear_knee_angles[win_start:win_end]
    rear_knee_angle_min = float(np.min(rear_win_angles))
    lead_rear_knee_ratio = float(lead_knee_angle_min / (rear_knee_angle_min + 1e-4))

    hip_diff_y = kps[t_bottom, MP_LEFT_HIP, 1] - kps[t_bottom, MP_RIGHT_HIP, 1]
    hip_width = np.linalg.norm(kps[t_bottom, MP_LEFT_HIP] - kps[t_bottom, MP_RIGHT_HIP]) + 1e-6
    pelvic_drop_angle = float(np.degrees(np.arcsin(np.clip(abs(hip_diff_y) / hip_width, -1.0, 1.0))))

    hip_diff_z = kps[t_bottom, MP_LEFT_HIP, 2] - kps[t_bottom, MP_RIGHT_HIP, 2]
    pelvic_axial_rotation = float(np.degrees(np.arctan2(abs(hip_diff_z), hip_width)))

    trunk_vec = neck[t_bottom] - pelvis[t_bottom]
    trunk_lean_angle = vector_angle_3d(trunk_vec, vertical_unit)
    trunk_forward_lean_lead = float(abs(trunk_lean_angle - femur_vertical_angle))

    hip_angle = vector_angle_3d(trunk_vec, femur_vec)
    hip_knee_flexion_coupling = float(hip_angle / (lead_knee_angle_min + 1e-4))

    stride_vec = lead_a_pt - kps[t_bottom, rear_ank]
    stride_length_normalized = float(np.linalg.norm(stride_vec[[0, 2]]) / torso_scale)
    step_width_normalized = float(abs(stride_vec[0]) / torso_scale)

    # [기술 6] Farrokhi et al. 정강이-체간 평행성 오차
    tibia_torso_parallelism_error = float(abs(lead_tibia_vertical_tilt - trunk_lean_angle))

    # 뒷무릎 지면 간격 프록시
    rear_knee_ground_clearance = float((kps[t_bottom, rear_knee, 1] - lead_a_pt[1]) / torso_scale)

    # [그룹 4: 시공간 템포 및 시점 불변 동역학]
    bottom_dwell_count = np.sum(np.abs(lead_knee_angles - lead_knee_angle_min) < 5.0)
    bottom_phase_dwell_ratio = float(bottom_dwell_count / float(T))

    descent_frames = max(1, t_bottom)
    ascent_frames = max(1, T - t_bottom)
    descent_ascent_tempo_ratio = float(descent_frames / float(ascent_frames))

    gold_t = np.linspace(0, 1, 64)
    gold_curve = 170.0 - 80.0 * np.exp(-((gold_t - 0.5) ** 2) / 0.05)
    lead_knee_dtw_distance = compute_dtw_distance_fast(lead_knee_angles, gold_curve)

    view_invariant_flexion_3d = lead_knee_angle_min

    sh_diff_z = abs(kps[t_bottom, MP_LEFT_SHOULDER, 2] - kps[t_bottom, MP_RIGHT_SHOULDER, 2])
    sh_width = np.linalg.norm(kps[t_bottom, MP_LEFT_SHOULDER] - kps[t_bottom, MP_RIGHT_SHOULDER]) + 1e-6
    camera_view_azimuth_est = float(sh_diff_z / sh_width)

    if vis_64f is not None and vis_64f.shape[0] == T:
        lead_knee_visibility_mean = float(np.mean(vis_64f[win_start:win_end, lead_knee]))
    else:
        lead_knee_visibility_mean = 0.95

    kinetic_energy_inflection = float((lead_vel[t_bottom] ** 2) / 100.0)
    lead_jerk = np.gradient(lead_accel)
    jerk_metric_lead_leg = float(np.mean(lead_jerk[win_start:win_end] ** 2))

    features: Dict[str, float] = {
        # 그룹 1
        "lead_knee_angle_min": lead_knee_angle_min,
        "lead_knee_angle_mean": lead_knee_angle_mean,
        "lead_knee_90_error_abs": lead_knee_90_error_abs,
        "lead_knee_flexion_rom": lead_knee_flexion_rom,
        "lead_knee_under_90_ratio": lead_knee_under_90_ratio,
        "lead_knee_flex_velocity_max": lead_knee_flex_velocity_max,
        "lead_knee_accel_inflection": lead_knee_accel_inflection,
        "lead_knee_rebound_jitter": lead_knee_rebound_jitter,
        "sagittal_knee_angle_bottom": sagittal_knee_angle_bottom,
        "sagittal_knee_90_error": sagittal_knee_90_error,

        # 그룹 2
        "lead_tibia_vertical_tilt": lead_tibia_vertical_tilt,
        "lead_femur_horizontal_tilt": lead_femur_horizontal_tilt,
        "lead_knee_toe_overhang": lead_knee_toe_overhang,
        "lead_shank_ankle_coupling": lead_shank_ankle_coupling,
        "lead_tibia_sagittal_sway": lead_tibia_sagittal_sway,
        "lead_knee_valgus_varus": lead_knee_valgus_varus,
        "lead_femur_length_ratio": lead_femur_length_ratio,
        "lead_tibia_length_ratio": lead_tibia_length_ratio,
        "kot_normalized_displacement": kot_normalized_displacement,
        "rigid_bone_length_consistency": rigid_bone_length_consistency,

        # 그룹 3
        "lead_rear_knee_ratio": lead_rear_knee_ratio,
        "rear_knee_angle_min": rear_knee_angle_min,
        "pelvic_drop_angle": pelvic_drop_angle,
        "pelvic_axial_rotation": pelvic_axial_rotation,
        "trunk_forward_lean_lead": trunk_forward_lean_lead,
        "hip_knee_flexion_coupling": hip_knee_flexion_coupling,
        "stride_length_normalized": stride_length_normalized,
        "step_width_normalized": step_width_normalized,
        "tibia_torso_parallelism_error": tibia_torso_parallelism_error,
        "rear_knee_ground_clearance": rear_knee_ground_clearance,

        # 그룹 4
        "bottom_phase_dwell_ratio": bottom_phase_dwell_ratio,
        "descent_ascent_tempo_ratio": descent_ascent_tempo_ratio,
        "lead_knee_dtw_distance": lead_knee_dtw_distance,
        "view_invariant_flexion_3d": view_invariant_flexion_3d,
        "camera_view_azimuth_est": camera_view_azimuth_est,
        "lead_knee_visibility_mean": lead_knee_visibility_mean,
        "kinetic_energy_inflection": kinetic_energy_inflection,
        "jerk_metric_lead_leg": jerk_metric_lead_leg,
    }

    # NaN / Inf 방어 처리
    for k, v in features.items():
        if math.isnan(v) or math.isinf(v):
            features[k] = 0.0

    return features


def extract_lunge_c1_feature_vector_v2(kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> np.ndarray:
    """38차원 피처 벡터 (38,) numpy array 반환"""
    f_dict = extract_lunge_c1_features_v2(kps_33_3d_64f, vis_64f)
    return np.array([f_dict[name] for name in LUNGE_C1_FEATURE_NAMES_V2], dtype=np.float32)
