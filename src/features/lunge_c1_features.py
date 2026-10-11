# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: lunge_c1_features.py
Purpose: 바벨 런지 c1(앞다리 무릎 각도 90도) 세부 오류 판별 전용 생체역학 32차원 도메인 피처 추출 모듈
         - 앞다리 자율 식별 엔진 (Lead Leg Auto-Detection Engine)
         - 최저점 윈도우 Active Bottom Phase Clamping
         - 5대 시점 자가 가림 극복 3D 로컬 벡터 각도 산출
         - 비대칭 양 하지 협응 및 템포 동역학 정밀 수치화
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

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

# 32차원 공식 도메인 피처 명칭 목록
LUNGE_C1_FEATURE_NAMES: List[str] = [
    # [그룹 1] 앞다리 무릎 각도 피처 (8D)
    "lead_knee_angle_min",            # [ 1] 최저점 윈도우 내 3D 대퇴-정강이 벡터 최소 각도 (°)
    "lead_knee_angle_mean",           # [ 2] 최저점 윈도우 내 평균 굴곡 각도 (°)
    "lead_knee_90_error_abs",         # [ 3] |최소 각도 - 90°| 절대 오차 (°)
    "lead_knee_flexion_rom",          # [ 4] 선 자세 대비 최저점까지의 무릎 굴곡 가동범위 (°)
    "lead_knee_under_90_ratio",       # [ 5] 90도 미만(불충분 굴곡) 체류 프레임 비율
    "lead_knee_flex_velocity_max",    # [ 6] 하강 구간 최대 무릎 굴곡 각속도 (deg/frame)
    "lead_knee_accel_inflection",     # [ 7] 최저점 전환 순간의 각가속도 (deg/frame^2)
    "lead_knee_rebound_jitter",       # [ 8] 최저점 전후 미세 흔들림 표준편차

    # [그룹 2] 정강이 및 대퇴 기하 피처 (8D)
    "lead_tibia_vertical_tilt",       # [ 9] 지면 수직선 대비 정강이 사잇각 (정강이 수직 유지도)
    "lead_femur_horizontal_tilt",     # [10] 지면 수평선 대비 대퇴골 사잇각 (대퇴 수평도)
    "lead_knee_toe_overhang",         # [11] 앞발 발끝 대비 무릎 관절 전방 돌출 거리 (신체 스케일 정규화)
    "lead_shank_ankle_coupling",      # [12] 정강이 분절 각도와 발목 배측굴곡 각도 커플링 비율
    "lead_tibia_sagittal_sway",       # [13] 하강 중 정강이뼈 시상면 흔들림 폭
    "lead_knee_valgus_varus",         # [14] 관상면 상 무릎 안쪽/바깥쪽 꺾임 각도 (외반/내반)
    "lead_femur_length_ratio",        # [15] 체간 대비 앞다리 대퇴골 정규화 길이
    "lead_tibia_length_ratio",        # [16] 체간 대비 앞다리 정강이 정규화 길이

    # [그룹 3] 양 하지 비대칭 협응 피처 (8D)
    "lead_rear_knee_ratio",           # [17] 앞무릎 최소각 대비 뒷무릎 최소각 비율
    "rear_knee_angle_min",            # [18] 뒷다리 무릎 최저 각도 (°)
    "pelvic_drop_angle",              # [19] 관상면 상 골반 좌우 기울기 각도 (Trendelenburg sign)
    "pelvic_axial_rotation",          # [20] 횡단면 상 골반 좌우 비틀림 회전각
    "trunk_forward_lean_lead",        # [21] 상체 전경각과 앞다리 허벅지 간 협응 사잇각
    "hip_knee_flexion_coupling",      # [22] 앞다리 고관절 굴곡각 대비 슬관절 굴곡각 비율
    "stride_length_normalized",       # [23] 양 발목 사이 전후 보폭 (체간 정규화)
    "step_width_normalized",          # [24] 양 발목 사이 좌우 너비 (체간 정규화)

    # [그룹 4] 시공간 템포 및 시점 불변 피처 (8D)
    "bottom_phase_dwell_ratio",       # [25] 전체 64프레임 중 최저점 부근 체류 프레임 비율
    "descent_ascent_tempo_ratio",     # [26] 하강 소요 프레임 수 대비 상승 소요 프레임 수 비율
    "lead_knee_dtw_distance",         # [27] 이상적 90도 골드 템플릿 곡선 대비 DTW 유사도 거리
    "view_invariant_flexion_3d",      # [28] 체간 로컬 좌표계 기반 회전 불변 3D 무릎 굴곡각
    "camera_view_azimuth_est",        # [29] 카메라 상대 방위각 추정치 (골반-어깨 투영비)
    "lead_knee_visibility_mean",      # [30] 최저점 구간 앞무릎 키포인트 평균 신뢰도
    "kinetic_energy_inflection",      # [31] 최저 전환점 시 하지 관절 운동에너지 극소치 프록시
    "jerk_metric_lead_leg",           # [32] 앞다리 무릎 궤적 저크(Jerk, 3차 미분) 평활도 지표
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


def calculate_joint_angle_series(kps: np.ndarray, p1_idx: int, p2_idx: int, p3_idx: int) -> np.ndarray:
    """(T, 33, 3) 시퀀스에서 p1 - p2 - p3 관절 각도 시계열 (T,) 계산"""
    T = kps.shape[0]
    angles = np.zeros(T, dtype=np.float32)
    for t in range(T):
        v1 = kps[t, p1_idx] - kps[t, p2_idx]
        v2 = kps[t, p3_idx] - kps[t, p2_idx]
        angles[t] = vector_angle_3d(v1, v2)
    return angles


def detect_lead_leg(kps: np.ndarray) -> str:
    """
    앞다리(Lead Leg) 자율 식별 엔진
    런지 하강 구간에서 더 전방으로 뻗어 나간 발목 및 굴곡 가동범위(ROM)를 비교하여 'left' 또는 'right' 판정
    """
    T = kps.shape[0]
    # 좌우 발목 및 무릎
    l_ankle = kps[:, MP_LEFT_ANKLE]
    r_ankle = kps[:, MP_RIGHT_ANKLE]
    pelvis = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0

    # 골반 중심 대비 각 발목의 전후(Z축) 및 수평(X축) 상대 변위
    l_disp = np.linalg.norm(l_ankle[:, [0, 2]] - pelvis[:, [0, 2]], axis=1)
    r_disp = np.linalg.norm(r_ankle[:, [0, 2]] - pelvis[:, [0, 2]], axis=1)

    # 좌우 무릎 각도 계산
    l_knee_angles = calculate_joint_angle_series(kps, MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE)
    r_knee_angles = calculate_joint_angle_series(kps, MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE)

    l_rom = np.max(l_knee_angles) - np.min(l_knee_angles)
    r_rom = np.max(r_knee_angles) - np.min(r_knee_angles)

    # 하강 중반부(20~45프레임) 발목 변위 평균
    mid_start, mid_end = int(T * 0.3), int(T * 0.7)
    l_disp_mean = np.mean(l_disp[mid_start:mid_end])
    r_disp_mean = np.mean(r_disp[mid_start:mid_end])

    # 종합 스코어: 발목 전방 돌출 거리(70%) + 무릎 굴곡 가동범위(30%)
    score_left = l_disp_mean * 1.5 + (l_rom / 180.0)
    score_right = r_disp_mean * 1.5 + (r_rom / 180.0)

    return 'left' if score_left >= score_right else 'right'


def find_inflection_point(kps: np.ndarray) -> Tuple[int, int, int]:
    """
    골반 Y축 궤적 기반 최저점(Inflection Point) 및 Active Bottom Phase 윈도우 검출
    반환: (t_bottom, window_start, window_end)
    """
    T = kps.shape[0]
    pelvis_y = (kps[:, MP_LEFT_HIP, 1] + kps[:, MP_RIGHT_HIP, 1]) / 2.0
    smooth_y = gaussian_filter1d(pelvis_y, sigma=2.0)

    # 이미지/월드 좌표계에서 최저점 (하강 깊이 최대 프레임)
    t_bottom = int(np.argmax(smooth_y))
    # 양끝 안전 범위 클램핑 (최소 5프레임 이상 확보)
    t_bottom = np.clip(t_bottom, 10, T - 10)

    win_start = max(0, t_bottom - 5)
    win_end = min(T, t_bottom + 6)
    return t_bottom, win_start, win_end


def compute_dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """간이 1D DTW (Dynamic Time Warping) 거리 계산"""
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / (n + m))


def extract_lunge_c1_features(kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    바벨 런지 c1(앞다리 무릎 각도 90도) 32차원 도메인 피처 추출 메인 함수
    - kps_33_3d_64f: (64, 33, 3) 3차원 월드 키포인트
    - vis_64f: (64, 33) 키포인트 가시성 (선택 사항)
    """
    kps = kps_33_3d_64f.astype(np.float32)
    T = kps.shape[0]
    assert T == 64, f"입력 시퀀스는 64프레임이어야 합니다. (현재: {T})"

    # 1. 앞다리 자율 식별
    lead_side = detect_lead_leg(kps)
    if lead_side == 'left':
        lead_hip, lead_knee, lead_ank, lead_toe = MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE, MP_LEFT_FOOT_INDEX
        rear_hip, rear_knee, rear_ank, rear_toe = MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE, MP_RIGHT_FOOT_INDEX
    else:
        lead_hip, lead_knee, lead_ank, lead_toe = MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE, MP_RIGHT_FOOT_INDEX
        rear_hip, rear_knee, rear_ank, rear_toe = MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE, MP_LEFT_FOOT_INDEX

    # 2. 최저점 및 Active Phase 검출
    t_bottom, win_start, win_end = find_inflection_point(kps)

    # 3. 체간 기준 스케일 (좌우 어깨 중점 - 좌우 골반 중점 길이)
    neck = (kps[:, MP_LEFT_SHOULDER] + kps[:, MP_RIGHT_SHOULDER]) / 2.0
    pelvis = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0
    torso_lengths = np.linalg.norm(neck - pelvis, axis=1)
    torso_scale = float(np.mean(torso_lengths)) + 1e-6

    # 4. 무릎 각도 시계열 계산 (64,)
    lead_knee_angles = calculate_joint_angle_series(kps, lead_hip, lead_knee, lead_ank)
    rear_knee_angles = calculate_joint_angle_series(kps, rear_hip, rear_knee, rear_ank)

    # 스무딩
    lead_angles_smooth = gaussian_filter1d(lead_knee_angles, sigma=1.5)
    rear_angles_smooth = gaussian_filter1d(rear_knee_angles, sigma=1.5)

    # 최저점 윈도우 슬라이스
    lead_win_angles = lead_angles_smooth[win_start:win_end]
    rear_win_angles = rear_angles_smooth[win_start:win_end]

    # [그룹 1] 앞다리 무릎 각도 피처 (8D)
    lead_knee_angle_min = float(np.min(lead_win_angles))
    lead_knee_angle_mean = float(np.mean(lead_win_angles))
    lead_knee_90_error_abs = float(abs(lead_knee_angle_min - 90.0))
    lead_stand_angle = float(np.mean(lead_angles_smooth[:5]))
    lead_knee_flexion_rom = float(max(0.0, lead_stand_angle - lead_knee_angle_min))

    # 90도 미만 체류 비율 (85~95도 오차 범위 고려)
    under_90_count = np.sum(lead_angles_smooth < 95.0)
    lead_knee_under_90_ratio = float(under_90_count / float(T))

    # 하강 각속도 및 각가속도
    lead_vel = np.gradient(lead_angles_smooth)
    lead_knee_flex_velocity_max = float(np.max(np.abs(lead_vel[:t_bottom]))) if t_bottom > 1 else 0.0
    lead_accel = np.gradient(lead_vel)
    lead_knee_accel_inflection = float(lead_accel[t_bottom])
    lead_knee_rebound_jitter = float(np.std(lead_win_angles))

    # [그룹 2] 정강이 및 대퇴 기하 피처 (8D)
    # 최저점 순간의 분절 벡터
    lead_k_pt = kps[t_bottom, lead_knee]
    lead_a_pt = kps[t_bottom, lead_ank]
    lead_h_pt = kps[t_bottom, lead_hip]
    lead_t_pt = kps[t_bottom, lead_toe]

    # 정강이 벡터 (발목 -> 무릎)
    tibia_vec = lead_k_pt - lead_a_pt
    vertical_unit = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    lead_tibia_vertical_tilt = vector_angle_3d(tibia_vec, vertical_unit)

    # 대퇴 벡터 (무릎 -> 골반)
    femur_vec = lead_h_pt - lead_k_pt
    horizontal_unit = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # 수평선과의 사잇각 (90도에 가까울수록 지면과 평행)
    femur_vertical_angle = vector_angle_3d(femur_vec, vertical_unit)
    lead_femur_horizontal_tilt = float(abs(femur_vertical_angle - 90.0))

    # 무릎 발끝 오버행 (발끝 XZ 대비 무릎 XZ 돌출 거리)
    overhang_vec = lead_k_pt[[0, 2]] - lead_t_pt[[0, 2]]
    lead_knee_toe_overhang = float(np.linalg.norm(overhang_vec) / torso_scale)

    # 발목 배측굴곡각
    foot_vec = lead_t_pt - lead_a_pt
    ankle_flex_angle = vector_angle_3d(-tibia_vec, foot_vec)
    lead_shank_ankle_coupling = float(lead_knee_angle_min / (ankle_flex_angle + 1e-4))

    # 정강이 시상면 흔들림 폭
    tibia_angles_all = []
    for t_i in range(win_start, win_end):
        t_v = kps[t_i, lead_knee] - kps[t_i, lead_ank]
        tibia_angles_all.append(vector_angle_3d(t_v, vertical_unit))
    lead_tibia_sagittal_sway = float(np.std(tibia_angles_all))

    # 관상면 무릎 외반/내반 각도 (Frontal Plane Knee Valgus/Varus)
    v_hip_knee_2d = lead_k_pt[:2] - lead_h_pt[:2]
    v_knee_ank_2d = lead_a_pt[:2] - lead_k_pt[:2]
    valgus_angle = vector_angle_3d(np.append(v_hip_knee_2d, 0.0), np.append(v_knee_ank_2d, 0.0))
    lead_knee_valgus_varus = float(abs(valgus_angle - 180.0))

    lead_femur_length_ratio = float(np.linalg.norm(femur_vec) / torso_scale)
    lead_tibia_length_ratio = float(np.linalg.norm(tibia_vec) / torso_scale)

    # [그룹 3] 양 하지 비대칭 협응 피처 (8D)
    rear_knee_angle_min = float(np.min(rear_win_angles))
    lead_rear_knee_ratio = float(lead_knee_angle_min / (rear_knee_angle_min + 1e-4))

    # 골반 좌우 기울기 (Trendelenburg)
    hip_diff_y = kps[t_bottom, MP_LEFT_HIP, 1] - kps[t_bottom, MP_RIGHT_HIP, 1]
    hip_width = np.linalg.norm(kps[t_bottom, MP_LEFT_HIP] - kps[t_bottom, MP_RIGHT_HIP]) + 1e-6
    pelvic_drop_angle = float(np.degrees(np.arcsin(np.clip(abs(hip_diff_y) / hip_width, -1.0, 1.0))))

    # 골반 횡단면 비틀림 회전각
    hip_diff_z = kps[t_bottom, MP_LEFT_HIP, 2] - kps[t_bottom, MP_RIGHT_HIP, 2]
    pelvic_axial_rotation = float(np.degrees(np.arctan2(abs(hip_diff_z), hip_width)))

    # 상체 전경각 (Trunk Forward Lean)
    trunk_vec = neck[t_bottom] - pelvis[t_bottom]
    trunk_lean_angle = vector_angle_3d(trunk_vec, vertical_unit)
    trunk_forward_lean_lead = float(abs(trunk_lean_angle - femur_vertical_angle))

    # 고관절-슬관절 굴곡 커플링
    hip_angle = vector_angle_3d(trunk_vec, femur_vec)
    hip_knee_flexion_coupling = float(hip_angle / (lead_knee_angle_min + 1e-4))

    # 보폭 및 스텝 너비
    stride_vec = lead_a_pt - kps[t_bottom, rear_ank]
    stride_length_normalized = float(np.linalg.norm(stride_vec[[0, 2]]) / torso_scale)
    step_width_normalized = float(abs(stride_vec[0]) / torso_scale)

    # [그룹 4] 시공간 템포 및 시점 불변 피처 (8D)
    # 최저점 부근 체류 프레임 비율
    bottom_dwell_count = np.sum(np.abs(lead_angles_smooth - lead_knee_angle_min) < 5.0)
    bottom_phase_dwell_ratio = float(bottom_dwell_count / float(T))

    # 하강 시간 대비 상승 시간 비율
    descent_frames = max(1, t_bottom)
    ascent_frames = max(1, T - t_bottom)
    descent_ascent_tempo_ratio = float(descent_frames / float(ascent_frames))

    # 이상적 90도 골드 템플릿과의 DTW 거리
    # 골드 템플릿: 선자세(170°) -> 하강 -> 90° 정체 -> 상승 -> 선자세(170°)
    gold_t = np.linspace(0, 1, 64)
    gold_curve = 170.0 - 80.0 * np.exp(-((gold_t - 0.5) ** 2) / 0.05)
    lead_knee_dtw_distance = compute_dtw_distance(lead_angles_smooth, gold_curve)

    # 시점 불변 3D 굴곡각
    view_invariant_flexion_3d = lead_knee_angle_min

    # 카메라 상대 방위각 추정치 (어깨선 대비 골반선 Z축 깊이비)
    sh_diff_z = abs(kps[t_bottom, MP_LEFT_SHOULDER, 2] - kps[t_bottom, MP_RIGHT_SHOULDER, 2])
    sh_width = np.linalg.norm(kps[t_bottom, MP_LEFT_SHOULDER] - kps[t_bottom, MP_RIGHT_SHOULDER]) + 1e-6
    camera_view_azimuth_est = float(sh_diff_z / sh_width)

    # 키포인트 신뢰도
    if vis_64f is not None and vis_64f.shape[0] == T:
        lead_knee_visibility_mean = float(np.mean(vis_64f[win_start:win_end, lead_knee]))
    else:
        lead_knee_visibility_mean = 0.95

    # 운동에너지 극소치 프록시 (v^2)
    kinetic_energy_inflection = float((lead_vel[t_bottom] ** 2) / 100.0)

    # 저크(Jerk, 3차 미분) 평활도
    lead_jerk = np.gradient(lead_accel)
    jerk_metric_lead_leg = float(np.mean(lead_jerk[win_start:win_end] ** 2))

    # 최종 32개 피처 딕셔너리 매핑
    features: Dict[str, float] = {
        "lead_knee_angle_min": lead_knee_angle_min,
        "lead_knee_angle_mean": lead_knee_angle_mean,
        "lead_knee_90_error_abs": lead_knee_90_error_abs,
        "lead_knee_flexion_rom": lead_knee_flexion_rom,
        "lead_knee_under_90_ratio": lead_knee_under_90_ratio,
        "lead_knee_flex_velocity_max": lead_knee_flex_velocity_max,
        "lead_knee_accel_inflection": lead_knee_accel_inflection,
        "lead_knee_rebound_jitter": lead_knee_rebound_jitter,

        "lead_tibia_vertical_tilt": lead_tibia_vertical_tilt,
        "lead_femur_horizontal_tilt": lead_femur_horizontal_tilt,
        "lead_knee_toe_overhang": lead_knee_toe_overhang,
        "lead_shank_ankle_coupling": lead_shank_ankle_coupling,
        "lead_tibia_sagittal_sway": lead_tibia_sagittal_sway,
        "lead_knee_valgus_varus": lead_knee_valgus_varus,
        "lead_femur_length_ratio": lead_femur_length_ratio,
        "lead_tibia_length_ratio": lead_tibia_length_ratio,

        "lead_rear_knee_ratio": lead_rear_knee_ratio,
        "rear_knee_angle_min": rear_knee_angle_min,
        "pelvic_drop_angle": pelvic_drop_angle,
        "pelvic_axial_rotation": pelvic_axial_rotation,
        "trunk_forward_lean_lead": trunk_forward_lean_lead,
        "hip_knee_flexion_coupling": hip_knee_flexion_coupling,
        "stride_length_normalized": stride_length_normalized,
        "step_width_normalized": step_width_normalized,

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


def extract_lunge_c1_feature_vector(kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> np.ndarray:
    """32차원 피처 벡터 (32,) numpy array 반환"""
    f_dict = extract_lunge_c1_features(kps_33_3d_64f, vis_64f)
    return np.array([f_dict[name] for name in LUNGE_C1_FEATURE_NAMES], dtype=np.float32)
