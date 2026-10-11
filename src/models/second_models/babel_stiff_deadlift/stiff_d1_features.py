# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Defect: d1. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Module: stiff_d1_features.py
Purpose: 바벨 데드리프트 v6.0 Zero-Calibrated(37D) 입증 베이스라인 계승
         + 스티프 고유 무릎 고정 고관절 힌지 연직선(Plumb Line) 이탈 피처 결합
         -> 총 38차원 Stiff Zero-Calibrated 도메인 피처 추출 엔진
Rule: AGENTS.md 및 공식 5대 시점 규격, 순수 비전 MediaPipe 33관절(64F) 입력 준수
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

# MediaPipe 33 관절 인덱스
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
MP_LEFT_FOOT_INDEX = 31  # Left Toe
MP_RIGHT_FOOT_INDEX = 32  # Right Toe

# 공식 5대 시점 명칭 매핑 (AGENTS.md 표준)
VIEW_ALIAS_MAP: Dict[str, str] = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}

VIEW_CODE_MAP: Dict[str, float] = {
    'view5': 0.0,
    'view2': 1.0,
    'view3': 2.0,
    'view4': 3.0,
    'view1': 4.0
}

# 38D 피처 명칭 목록
FEATURE_NAMES_38_STIFF_D1: List[str] = [
    # Group 1: 바벨-하지 수평 이격 거리 동역학 (8D)
    'stiff_d1_bar_shank_dist_max',            # [0] 정강이 구간 바벨-정강이 최대 수평 이격거리
    'stiff_d1_bar_shank_dist_mean',           # [1] 정강이 구간 바벨-정강이 평균 이격거리
    'stiff_d1_bar_knee_dist_pass',            # [2] 무릎 높이 통과 순간 바벨-무릎 거리
    'stiff_d1_bar_thigh_dist_mean',           # [3] 대퇴부 구간 바벨-허벅지 평균 이격거리
    'stiff_d1_bar_thigh_dist_max',            # [4] 대퇴부 구간 바벨-허벅지 최대 이격거리
    'stiff_d1_bar_midfoot_init_offset',       # [5] 셋업 시작점 바벨-미드풋 정렬 오차
    'stiff_d1_bar_midfoot_max_offset',        # [6] 전 구간 바벨-미드풋 최대 전후 이격거리
    'stiff_d1_bar_leg_contact_deficit_ratio', # [7] 정강이/허벅지 밀착 역치 초과 비율

    # Group 2: 바벨 수직 궤적 선형성 및 흔들림 (4D)
    'stiff_d1_bar_drift_horizontal_std',      # [8] 바벨 전후 표류 표준편차
    'stiff_d1_bar_path_curvature',            # [9] 바벨 3D 궤적 길이 / 수직 변위 비율
    'stiff_d1_bar_lateral_wobble_std',        # [10] 바벨 좌우(X축) 흔들림 표준편차
    'stiff_d1_bar_ascent_velocity_smoothness',# [11] 바벨 수직 속도 부드러움 (저크 억제)

    # Group 3: 상체 모멘트 암 & 광배 결합 (3D)
    'stiff_d1_shoulder_wrist_angle_max_deg',  # [12] 어깨-손목 벡터 최대 전방 경사각 (deg)
    'stiff_d1_shoulder_wrist_angle_mean_deg', # [13] 리프팅 구간 어깨-손목 평균 전방 경사각 (deg)
    'stiff_d1_hip_bar_moment_arm_at_knee',    # [14] 무릎 높이 고관절-바벨 모멘트 암

    # Group 4: 구간별 위상 동역학 (5D)
    'stiff_d1_p1_shank_clearance_rate',       # [15] Phase 1 바닥~무릎 바벨 이탈률
    'stiff_d1_p2_thigh_drag_contact_rate',    # [16] Phase 2 무릎~락아웃 허벅지 밀착 유지율
    'stiff_d1_eccentric_bar_drift_max',       # [17] 편심성 하강 구간 바벨 최대 전방 이탈량
    'stiff_d1_peak_drift_timing_norm',        # [18] 최대 이탈 발생 프레임 정규화 시점
    'stiff_d1_knee_pass_timing_norm',         # [19] 무릎 통과 발생 프레임 정규화 시점

    # Group 5: 멀티뷰 투영 & 왜곡 보정 (6D)
    'stiff_d1_sagittal_proj_drift',           # [20] 시점 회전 보정 시상면 유효 표류량
    'stiff_d1_coronal_overlap_deficit',       # [21] 정면 뷰(view3) 정강이-바벨 수직 투영 결손도
    'stiff_d1_bilateral_wrist_asym',          # [22] 좌우 손목 수평 비대칭도
    'stiff_d1_view_code',                     # [23] 카메라 시점 범주형 코드 (0~4)
    'stiff_d1_is_frontal',                    # [24] 정면 뷰 지시자 (view3=1.0)
    'stiff_d1_is_rear',                       # [25] 후방 뷰 지시자 (view1, view5=1.0)

    # Group 6: 시점 적응형 무릎-팔꿈치 상대 기하학 (6D)
    'stiff_d1_gate_is_oblique',                # [26] 사각/측면 뷰 지시자 (시점 B, D, A, E = 1.0)
    'stiff_d1_gated_ke_angle_sek_deg',         # [27] 최하사점 어깨-팔꿈치-무릎 사잇각 (deg)
    'stiff_d1_gated_ke_width_ratio_nadir',     # [28] 최하사점 무릎/팔꿈치 너비비율
    'stiff_d1_gated_ke_dist_nadir',            # [29] 최하사점 무릎-팔꿈치 3D 유클리드 거리
    'stiff_d1_gated_ke_z_diff_nadir',          # [30] 최하사점 무릎-팔꿈치 Z축 상대거리
    'stiff_d1_gated_ke_signed_x_nadir',        # [31] 최하사점 무릎-팔꿈치 수평 오프셋

    # Group 7: 어드레스 동적 영점화 및 체형 정규화 (5D) [바벨 데드 성공 요인]
    'stiff_d1_zero_delta_bar_shank_max',       # [32] 셋업 대비 리프팅 중 바벨-정강이 추가 이탈량 (Δ max)
    'stiff_d1_zero_delta_bar_shank_mean',      # [33] 셋업 대비 리프팅 중 바벨-정강이 평균 추가 이탈량 (Δ mean)
    'stiff_d1_zero_delta_shoulder_wrist_angle',# [34] 셋업 대비 어깨-손목 전방 경사각 추가 증가량 (Δ deg)
    'stiff_d1_anatomical_arm_to_torso_ratio',  # [35] 팔 길이 / 몸통 길이 비율 (체형 보정)
    'stiff_d1_anatomical_torso_to_body_ratio', # [36] 몸통 길이 / 신체 스케일 비율 (상체 보정)

    # Group 8: 스티프 데드 고유 무릎 고정 힌지-광배근 연직선 피처 (1D) [신규 특화]
    'stiff_d1_bar_to_midfoot_plumb_drift'      # [37] 고관절 힌지 동작 중 미드풋 수직선 대비 바벨 전방 드리프트 최대치
]


def extract_stiff_d1_features(kps_arr: np.ndarray, view_name: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 3D 키포인트 배열로부터 스티프 데드리프트 바벨 밀착 38차원 피처 벡터 추출
    """
    pts = kps_arr[:, :, :3].astype(np.float32)
    T = pts.shape[0]

    shoulder_l = pts[:, MP_LEFT_SHOULDER]
    shoulder_r = pts[:, MP_RIGHT_SHOULDER]
    elbow_l = pts[:, MP_LEFT_ELBOW]
    elbow_r = pts[:, MP_RIGHT_ELBOW]
    wrist_l = pts[:, MP_LEFT_WRIST]
    wrist_r = pts[:, MP_RIGHT_WRIST]
    hip_l = pts[:, MP_LEFT_HIP]
    hip_r = pts[:, MP_RIGHT_HIP]
    knee_l = pts[:, MP_LEFT_KNEE]
    knee_r = pts[:, MP_RIGHT_KNEE]
    ankle_l = pts[:, MP_LEFT_ANKLE]
    ankle_r = pts[:, MP_RIGHT_ANKLE]
    heel_l = pts[:, MP_LEFT_HEEL]
    heel_r = pts[:, MP_RIGHT_HEEL]
    toe_l = pts[:, MP_LEFT_FOOT_INDEX]
    toe_r = pts[:, MP_RIGHT_FOOT_INDEX]

    # 신체 중심점
    shoulder = (shoulder_l + shoulder_r) / 2.0
    elbow = (elbow_l + elbow_r) / 2.0
    bar = (wrist_l + wrist_r) / 2.0  # 가상 바벨 중심점
    hip = (hip_l + hip_r) / 2.0
    knee = (knee_l + knee_r) / 2.0
    ankle = (ankle_l + ankle_r) / 2.0
    midfoot = (heel_l + heel_r + toe_l + toe_r) / 4.0

    # 신체 스케일 정규화 기준 (대퇴골 + 정강이)
    thigh_len = np.mean(np.linalg.norm(hip - knee, axis=-1))
    shank_len = np.mean(np.linalg.norm(knee - ankle, axis=-1))
    torso_len = np.mean(np.linalg.norm(shoulder - hip, axis=-1)) + 1e-6
    arm_len = np.mean(np.linalg.norm(shoulder - wrist_l, axis=-1) + np.linalg.norm(shoulder - wrist_r, axis=-1)) / 2.0 + 1e-6
    ref_len = float(max(thigh_len + shank_len, 0.2))

    # 위상 탐색 (Nadir: 바벨 Y축 최하점)
    bar_y_smooth = gaussian_filter1d(bar[:, 1], sigma=2.0)
    nadir_idx = int(np.argmax(bar_y_smooth))

    # 정강이 중심점 및 대퇴부 중심점
    shank_center = (knee + ankle) / 2.0
    thigh_center = (hip + knee) / 2.0

    # [0, 1] 바벨-정강이 이격거리
    bar_shank_dist = np.linalg.norm(bar - shank_center, axis=-1) / ref_len
    f0 = float(np.max(bar_shank_dist))
    f1 = float(np.mean(bar_shank_dist))

    # [2] 무릎 통과 순간 거리
    knee_y = knee[:, 1]
    y_diff = np.abs(bar[:, 1] - knee_y)
    knee_pass_idx = int(np.argmin(y_diff))
    f2 = float(np.linalg.norm(bar[knee_pass_idx] - knee[knee_pass_idx]) / ref_len)

    # [3, 4] 대퇴부 이격거리
    bar_thigh_dist = np.linalg.norm(bar - thigh_center, axis=-1) / ref_len
    f3 = float(np.mean(bar_thigh_dist))
    f4 = float(np.max(bar_thigh_dist))

    # [5, 6] 미드풋 대비 수평(XZ) 오차
    bar_midfoot_xz = np.linalg.norm(bar[:, [0, 2]] - midfoot[:, [0, 2]], axis=-1) / ref_len
    f5 = float(bar_midfoot_xz[0])
    f6 = float(np.max(bar_midfoot_xz))

    # [7] 밀착 결손 비율
    f7 = float(np.mean(bar_shank_dist > 0.35))

    # [8] 수평 흔들림 std
    bar_xz_disp = np.linalg.norm(bar[:, [0, 2]] - bar[0, [0, 2]], axis=-1) / ref_len
    f8 = float(np.std(bar_xz_disp))

    # [9] 곡률
    step_diffs = np.linalg.norm(np.diff(bar, axis=0), axis=-1)
    total_path = np.sum(step_diffs)
    vert_disp = abs(bar[nadir_idx, 1] - bar[0, 1]) + 1e-5
    f9 = float(total_path / vert_disp)

    # [10] 좌우 흔들림 std
    f10 = float(np.std(bar[:, 0] - np.mean(bar[:, 0])) / ref_len)

    # [11] 수직 속도 부드러움
    bar_vy = np.diff(bar[:, 1])
    bar_ay = np.diff(bar_vy)
    f11 = float(np.std(bar_ay) / ref_len)

    # [12, 13] 어깨-손목 전방 경사각
    sw_vec = bar - shoulder
    sw_norm = np.linalg.norm(sw_vec, axis=-1) + 1e-6
    sw_cos = np.clip(sw_vec[:, 1] / sw_norm, -1.0, 1.0)
    sw_angle_deg = np.degrees(np.arccos(sw_cos))
    f12 = float(np.max(sw_angle_deg))
    f13 = float(np.mean(sw_angle_deg))

    # [14] 무릎 통과 순간 고관절-바벨 모멘트 암
    f14 = float(np.linalg.norm(bar[knee_pass_idx, [0, 2]] - hip[knee_pass_idx, [0, 2]]) / ref_len)

    # [15, 16] 위상별 지표
    p1_slice = slice(0, max(1, knee_pass_idx))
    p2_slice = slice(knee_pass_idx, T)
    f15 = float(np.mean(bar_shank_dist[p1_slice] > 0.30)) if knee_pass_idx > 0 else 0.0
    f16 = float(np.mean(bar_thigh_dist[p2_slice] <= 0.40)) if (T - knee_pass_idx) > 0 else 1.0

    # [17] 하강 구간 이탈
    ecc_slice = slice(0, max(1, nadir_idx))
    f17 = float(np.max(bar_shank_dist[ecc_slice])) if nadir_idx > 0 else f0

    # [18, 19] 타이밍
    f18 = float(np.argmax(bar_shank_dist) / max(T - 1, 1))
    f19 = float(knee_pass_idx / max(T - 1, 1))

    # [20, 21, 22] 멀티뷰 보정
    f20 = float(np.max(np.abs(bar[:, 2] - midfoot[:, 2])) / ref_len)
    f21 = float(np.max(np.abs(bar[:, 0] - shank_center[:, 0])) / ref_len)
    wrist_diff_y = np.abs(wrist_l[:, 1] - wrist_r[:, 1])
    f22 = float(np.mean(wrist_diff_y) / ref_len)

    # [23, 24, 25] 시점 인디케이터
    v_code = VIEW_CODE_MAP.get(view_name, 2.0)
    f23 = float(v_code)
    f24 = 1.0 if view_name == 'view3' else 0.0
    f25 = 1.0 if view_name in ['view1', 'view5'] else 0.0

    # [26~31] 게이팅 무릎-팔꿈치 기하 피처
    is_oblique = 0.0 if view_name == 'view3' else 1.0
    f26 = is_oblique

    # 최하사점 SEK 사잇각
    se = elbow[nadir_idx] - shoulder[nadir_idx]
    ek = knee[nadir_idx] - elbow[nadir_idx]
    denom = (np.linalg.norm(se) * np.linalg.norm(ek)) + 1e-6
    cos_sek = np.clip(np.dot(se, ek) / denom, -1.0, 1.0)
    f27 = float(np.degrees(np.arccos(cos_sek))) * is_oblique

    # 너비비율
    knee_w = abs(knee_l[nadir_idx, 0] - knee_r[nadir_idx, 0]) + 1e-5
    elbow_w = abs(elbow_l[nadir_idx, 0] - elbow_r[nadir_idx, 0]) + 1e-5
    f28 = float(knee_w / elbow_w)

    # 유클리드 거리
    ke_dist_nadir = np.linalg.norm(knee[nadir_idx] - elbow[nadir_idx]) / ref_len
    f29 = float(ke_dist_nadir)

    # Z 오프셋 (정면 마스킹)
    f30 = float(abs(knee[nadir_idx, 2] - elbow[nadir_idx, 2]) / ref_len) * is_oblique

    # X 오프셋
    f31 = float((knee[nadir_idx, 0] - elbow[nadir_idx, 0]) / ref_len)

    # [32~36] 어드레스 동적 영점화 및 체형 정규화
    setup_slice = slice(0, min(8, T))
    base_shank_dist = float(np.mean(bar_shank_dist[setup_slice]))
    base_sw_angle = float(np.mean(sw_angle_deg[setup_slice]))

    f32 = float(max(0.0, np.max(bar_shank_dist) - base_shank_dist))
    f33 = float(max(0.0, np.mean(bar_shank_dist) - base_shank_dist))
    f34 = float(max(0.0, np.max(sw_angle_deg) - base_sw_angle))
    f35 = float(arm_len / torso_len)
    f36 = float(torso_len / ref_len)

    # [37] 스티프 고유 피처: 고관절 힌지 시 미드풋 연직선 대비 바벨 전방 드리프트
    # 미드풋 X축 수직선 대비 바벨 전방 이동량
    bar_to_midfoot_ant = (bar[:, 0] - midfoot[:, 0]) if is_oblique > 0.5 else (bar[:, 2] - midfoot[:, 2])
    f37 = float(np.max(np.abs(bar_to_midfoot_ant)) / ref_len)

    feat_vector = np.array([
        f0, f1, f2, f3, f4, f5, f6, f7,
        f8, f9, f10, f11,
        f12, f13, f14,
        f15, f16, f17, f18, f19,
        f20, f21, f22, f23, f24, f25,
        f26, f27, f28, f29, f30, f31,
        f32, f33, f34, f35, f36,
        f37
    ], dtype=np.float32)

    # NaN 및 Inf 안전 방어
    feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=1.0, neginf=-1.0)
    return feat_vector


# ==============================================================================
# v2.0 정제 피처셋 (6대 간섭 피처 Pruning 후 32D)
# ==============================================================================
PRUNED_FEATURE_NAMES_6: List[str] = [
    'stiff_d1_bar_thigh_dist_max',            # [4] 대퇴부 최대 이격거리 (#36 영점화 거리와의 간섭)
    'stiff_d1_bar_midfoot_max_offset',        # [6] 미드풋 최대 이격거리 (#20과 r=0.992 중복 간섭)
    'stiff_d1_shoulder_wrist_angle_max_deg',  # [12] 어깨-손목 최대 각도 (#34와 r=0.942 중복 교란)
    'stiff_d1_eccentric_bar_drift_max',       # [17] 신장성 하강 최대 이탈 (미세 노이즈 저해)
    'stiff_d1_gated_ke_width_ratio_nadir',     # [28] 최하사점 너비비율 (정면 노이즈)
    'stiff_d1_gated_ke_dist_nadir'             # [29] 최하사점 3D 거리 (정면 노이즈)
]

PRUNED_INDICES_6: List[int] = [4, 6, 12, 17, 28, 29]
KEPT_INDICES_32: List[int] = [i for i in range(38) if i not in PRUNED_INDICES_6]

FEATURE_NAMES_32_STIFF_D1: List[str] = [FEATURE_NAMES_38_STIFF_D1[i] for i in KEPT_INDICES_32]
FEATURE_NAMES_SD3_V2: List[str] = FEATURE_NAMES_32_STIFF_D1


# ==============================================================================
# v3.0 전략 1: 생체역학 강체 복원 36D 피처셋 (정면 뷰 깊이 복원 + 디커플링 지표)
# ==============================================================================
FEATURE_NAMES_4D_RECOVERY: List[str] = [
    'stiff_d1_recovered_sagittal_bar_gap',     # [32] 정면 2D 압축비 기반 Closed-form Z축 복원 바벨-몸 간격
    'stiff_d1_hinge_bar_decoupling_index',    # [33] Nadir 직전 고관절 힌지 감속과 바벨 전방 가속 디커플링 지표
    'stiff_d1_lat_shoulder_torque_proxy',     # [34] 광배근 등척성 어깨 신전 토크 요구량 대리 지표 (모멘트 암 적분)
    'stiff_d1_coronal_wrist_ankle_proj_ratio' # [35] 정면 관상면 투영 손목-발목 상대 정렬 비율
]

FEATURE_NAMES_36_STIFF_D1: List[str] = FEATURE_NAMES_32_STIFF_D1 + FEATURE_NAMES_4D_RECOVERY
FEATURE_NAMES_SD3_V3: List[str] = FEATURE_NAMES_36_STIFF_D1


def compute_sagittal_recovery_features(pts: np.ndarray, view_name: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 3D 키포인트 배열로부터 생체역학 강체 복원 4대 피처 추출
    """
    T = pts.shape[0]
    shoulder_l = pts[:, MP_LEFT_SHOULDER]
    shoulder_r = pts[:, MP_RIGHT_SHOULDER]
    wrist_l = pts[:, MP_LEFT_WRIST]
    wrist_r = pts[:, MP_RIGHT_WRIST]
    hip_l = pts[:, MP_LEFT_HIP]
    hip_r = pts[:, MP_RIGHT_HIP]
    knee_l = pts[:, MP_LEFT_KNEE]
    knee_r = pts[:, MP_RIGHT_KNEE]
    ankle_l = pts[:, MP_LEFT_ANKLE]
    ankle_r = pts[:, MP_RIGHT_ANKLE]

    shoulder = (shoulder_l + shoulder_r) / 2.0
    bar = (wrist_l + wrist_r) / 2.0
    hip = (hip_l + hip_r) / 2.0
    knee = (knee_l + knee_r) / 2.0
    ankle = (ankle_l + ankle_r) / 2.0
    thigh_center = (hip + knee) / 2.0
    shank_center = (knee + ankle) / 2.0

    # 신체 스케일 기준
    thigh_len = np.mean(np.linalg.norm(hip - knee, axis=-1))
    shank_len = np.mean(np.linalg.norm(knee - ankle, axis=-1))
    ref_len = float(max(thigh_len + shank_len, 0.2))

    # 1. 해부학적 팔 뼈 길이 불변 기준 (셋업 초기 8프레임 평균)
    setup_arm_len = float(np.mean(np.linalg.norm(bar[:8] - shoulder[:8], axis=-1)) + 1e-5)

    # 2. 피타고라스 강체 구속식에 의한 시상면 전후 변위 Z 복원 (Closed-form Sagittal Recovery)
    # 정면 2D 평면에서의 겉보기 변위: ΔX, ΔY
    disp_xy = np.linalg.norm(bar[:, :2] - shoulder[:, :2], axis=-1)
    recovered_z_disp = np.sqrt(np.maximum(0.0, setup_arm_len**2 - disp_xy**2))

    # 대퇴/정강이의 복원 Z 위치 대비 바벨의 상대 이격 거리
    leg_z = (thigh_center[:, 2] + shank_center[:, 2]) / 2.0
    is_front = 1.0 if view_name == 'view3' else 0.0
    # 정면 뷰에서는 복원된 Z를 적극 사용하고, 사각/측면 뷰에서는 관측된 3D Z와 융합
    effective_bar_z = (recovered_z_disp * is_front) + (np.abs(bar[:, 2] - leg_z) * (1.0 - is_front))
    f_rec_gap = float(np.max(effective_bar_z) / ref_len)

    # 3. Nadir 직전 고관절 힌지 감속과 바벨 전방 가속 디커플링 지표
    bar_y_smooth = gaussian_filter1d(bar[:, 1], sigma=2.0)
    nadir_idx = int(np.argmax(bar_y_smooth))
    pre_nadir_start = max(0, nadir_idx - 10)
    pre_slice = slice(pre_nadir_start, max(pre_nadir_start + 1, nadir_idx))

    # 고관절 힌지 각도 변화량 (Hip Flexion)
    torso_vec = shoulder - hip
    thigh_vec = knee - hip
    hip_angles = np.degrees(np.arccos(np.clip(
        np.sum(torso_vec * thigh_vec, axis=-1) / (np.linalg.norm(torso_vec, axis=-1) * np.linalg.norm(thigh_vec, axis=-1) + 1e-6),
        -1.0, 1.0
    )))
    hip_vel = np.abs(np.diff(hip_angles))
    bar_ant_disp = np.abs(bar[:, 0] - ankle[:, 0]) if is_front < 0.5 else effective_bar_z
    bar_ant_vel = np.abs(np.diff(bar_ant_disp))

    mean_hip_vel = float(np.mean(hip_vel[pre_slice])) if nadir_idx > 0 else 0.1
    mean_bar_vel = float(np.mean(bar_ant_vel[pre_slice])) if nadir_idx > 0 else 0.0
    f_decoupling = float((mean_bar_vel / (mean_hip_vel + 1e-4)) / ref_len)

    # 4. 광배근 등척성 어깨 신전 토크 대리 지표 (모멘트 암 면적 적분)
    horiz_arm = np.abs(bar[:, 0] - shoulder[:, 0]) if is_front > 0.5 else np.abs(bar[:, 2] - shoulder[:, 2])
    vert_arm = np.abs(bar[:, 1] - shoulder[:, 1])
    torque_proxy = np.mean(horiz_arm * vert_arm) / (ref_len**2)
    f_lat_torque = float(torque_proxy)

    # 5. 정면 관상면 투영 손목-발목 상대 정렬 비율
    wrist_w = np.abs(wrist_l[:, 0] - wrist_r[:, 0])
    ankle_w = np.abs(ankle_l[:, 0] - ankle_r[:, 0]) + 1e-5
    f_coronal_align = float(np.mean(wrist_w / ankle_w))

    feats_4d = np.array([f_rec_gap, f_decoupling, f_lat_torque, f_coronal_align], dtype=np.float32)
    feats_4d = np.nan_to_num(feats_4d, nan=0.0, posinf=1.0, neginf=-1.0)
    return feats_4d


def extract_stiff_d1_v2_features(kps_arr: np.ndarray, view_name: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 3D 키포인트 배열로부터 스티프 데드리프트 바벨 밀착 32차원 정제 피처 벡터 추출 (sd3_v2)
    - 6대 상호 간섭 및 성능 저해 피처 Pruning 완료
    """
    feat_38 = extract_stiff_d1_features(kps_arr, view_name=view_name)
    feat_32 = feat_38[KEPT_INDICES_32]
    feat_32 = np.nan_to_num(feat_32, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    return feat_32


def extract_stiff_d1_v3_features(kps_arr: np.ndarray, view_name: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 3D 키포인트 배열로부터 스티프 데드리프트 바벨 밀착 36차원 최적화 피처 벡터 추출 (전략 1)
    - 32D 정제 피처 + 4대 생체역학 강체 복원 피처 결합
    """
    feat_32 = extract_stiff_d1_v2_features(kps_arr, view_name=view_name)
    pts = kps_arr[:, :, :3].astype(np.float32)
    feat_4d = compute_sagittal_recovery_features(pts, view_name=view_name)
    feat_36 = np.concatenate([feat_32, feat_4d], axis=0).astype(np.float32)
    feat_36 = np.nan_to_num(feat_36, nan=0.0, posinf=1.0, neginf=-1.0)
    return feat_36


if __name__ == "__main__":
    # 무결성 테스트
    dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
    feats_38 = extract_stiff_d1_features(dummy_kps, view_name="view4")
    feats_32 = extract_stiff_d1_v2_features(dummy_kps, view_name="view4")
    feats_36 = extract_stiff_d1_v3_features(dummy_kps, view_name="view3")
    print(f"38D Feature extraction passed! Shape: {feats_38.shape}")
    print(f"32D Feature extraction passed! Shape: {feats_32.shape}")
    print(f"36D Feature extraction passed! Shape: {feats_36.shape}")
    assert feats_38.shape[0] == 38, "38D Dimension mismatch!"
    assert feats_32.shape[0] == 32, "32D Dimension mismatch!"
    assert feats_36.shape[0] == 36, "36D Dimension mismatch!"
    assert len(FEATURE_NAMES_36_STIFF_D1) == 36, "36D Feature names length mismatch!"
    print("All assertions passed successfully.")
