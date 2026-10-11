# -*- coding: utf-8 -*-
"""
NovelFlow: 1D Temporal Shift Module (1D-TSM) Kinematic Feature Extractor
- AI 허브 가이드라인 2.6.1.1 Temporal Shift 원리를 1D 인체 관절 시계열 텐서(16F x 92D)에 경량 구현
- 연산량/파라미터 0개: 채널의 1/4을 t-1(과거), 1/4을 t+1(미래)로 Shifting하여 전후 프레임 간 속도·위상 지연 직접 포착
- 하강-상승 전환기(최하사점)에서의 템포 불일치 및 체간-무릎 위상차 특화 16D 피처 생성
"""

import numpy as np
from typing import List, Tuple

FEATURE_NAMES_TSM_16: List[str] = [
    # 1. 척추/체간 위상 지연 (t-1 vs t vs t+1 시프팅)
    'tsm_torso_pitch_lead_lag_corr',    # 체간 피치 선행-후행 위상차 상관계수
    'tsm_torso_pitch_jerk_forward',     # 전방 시프트 체간 피치 1차 차분 저크
    'tsm_spine_alignment_phase_shift',  # 척추 정렬도 시공간 시프트 편차 RMSE
    'tsm_torso_roll_lateral_sway',      # 좌우 롤링 전후 시프트 잔여 분산
    # 2. 고개 정면 동적 타이밍
    'tsm_neck_spine_timing_lead',       # 목-체간 선행 시프트 정렬 드리프트
    'tsm_neck_spine_timing_lag',        # 목-체간 후행 시프트 정렬 드리프트
    'tsm_head_gaze_velocity_shift',     # 시선 속도 시공간 텐서 시프트 분산
    # 3. 발-무릎 방향 일치 (Valgus 전후 위상차)
    'tsm_knee_valgus_l_lead_spread',    # 좌측 무릎 외반 선행 시프트 변위
    'tsm_knee_valgus_r_lead_spread',    # 우측 무릎 외반 선행 시프트 변위
    'tsm_knee_valgus_dynamic_hysteresis',# 무릎 외반 전후 시프트 궤적 면적
    'tsm_knee_hip_sync_shift_rmse',     # 무릎-고관절 동조성 시프트 오차
    # 4. 발바닥 지면 고정 (발목-정강이 전후 미세 흔들림)
    'tsm_shank_vert_left_shift_delta',  # 좌측 정강이 수직도 선행-후행 차이
    'tsm_shank_vert_right_shift_delta', # 우측 정강이 수직도 선행-후행 차이
    'tsm_ankle_z_timing_spread',        # 발목 전후 분리 거리 시프트 변화량
    'tsm_foot_ground_micro_impulse',    # 지면 접촉 반력 추정 임펄스 시프트 지표
    'tsm_bottom_turnaround_phase_gap'   # 최하사점 프레임 전후 3점 위상 갭
]

def apply_1d_temporal_shift(X_b: np.ndarray, shift_ratio: float = 0.25) -> np.ndarray:
    """
    Format B (N, T=16, C=92) 텐서에 1D-TSM Shifting 적용
    - 채널의 1/4: t - 1로 쉬프트 (과거 정보 주입, 맨 앞은 첫 프레임 복제)
    - 채널의 1/4: t + 1로 쉬프트 (미래 정보 주입, 맨 뒤는 마지막 프레임 복제)
    - 채널의 1/2: 원본 t 유지
    반환 Shape: (N, T=16, C=92)
    """
    N, T, C = X_b.shape
    C_shift = int(C * shift_ratio)

    X_tsm = X_b.copy()

    # 1. 과거 시프트 (t - 1): index 0 ~ C_shift
    # t시점에 t-1의 값을 가져옴
    part_left = np.zeros((N, T, C_shift), dtype=X_b.dtype)
    part_left[:, 1:, :] = X_b[:, :-1, :C_shift]
    part_left[:, 0, :] = X_b[:, 0, :C_shift] # boundary replicate

    # 2. 미래 시프트 (t + 1): index C_shift ~ 2*C_shift
    # t시점에 t+1의 값을 가져옴
    part_right = np.zeros((N, T, C_shift), dtype=X_b.dtype)
    part_right[:, :-1, :] = X_b[:, 1:, C_shift:2*C_shift]
    part_right[:, -1, :] = X_b[:, -1, C_shift:2*C_shift] # boundary replicate

    # 3. 채널 결합
    X_tsm[:, :, :C_shift] = part_left
    X_tsm[:, :, C_shift:2*C_shift] = part_right
    # 나머지 2*C_shift ~ C는 원본 유지

    return X_tsm

def extract_1d_tsm_features(X_b: np.ndarray) -> np.ndarray:
    """
    Format B (N, T=16, C=92) -> 1D-TSM 기반 16대 시공간 위상 피처 추출 -> Shape: (N, 16)
    """
    N, T, C = X_b.shape
    X_tsm = apply_1d_temporal_shift(X_b, shift_ratio=0.25)

    # 주요 시그널 인덱스 (kinematics_3d Stream 정의 참조):
    # s2: knee_l(56), knee_r(57), hip_l(58), hip_r(59), ankle_l(60), ankle_r(61)
    # s3: torso_pitch(64), torso_roll(65), valgus_l(66), valgus_r(67), sh_hip_twist(68), ankle_z(71)
    # s6: neck_spine(74), barbell_dist(75), knee_hip_sync(76)
    # s7: shank_vert_l(81), shank_vert_r(82), hip_knee_rom(83)

    pitch_orig = X_b[:, :, 64]
    pitch_shifted = X_tsm[:, :, 0] # 과거 시프트된 pitch
    roll_orig = X_b[:, :, 65]
    valgus_l = X_b[:, :, 66]
    valgus_r = X_b[:, :, 67]
    valgus_l_shift = X_tsm[:, :, 2]
    valgus_r_shift = X_tsm[:, :, 3]
    neck_spine = X_b[:, :, 74]
    neck_spine_shift = X_tsm[:, :, 10]
    knee_hip_sync = X_b[:, :, 76]
    shank_l = X_b[:, :, 81]
    shank_r = X_b[:, :, 82]
    ankle_z = X_b[:, :, 71]

    feats = np.zeros((N, 16), dtype=np.float32)

    for i in range(N):
        # 1. 체간 피치 선행-후행 상관계수 및 저크
        p_o = pitch_orig[i]
        p_s = pitch_shifted[i]
        std_po = np.std(p_o) + 1e-7
        std_ps = np.std(p_s) + 1e-7
        corr_pitch = float(np.mean((p_o - np.mean(p_o)) * (p_s - np.mean(p_s))) / (std_po * std_ps))
        feats[i, 0] = corr_pitch
        feats[i, 1] = float(np.mean(np.abs(np.diff(p_s, n=2)))) # 저크
        feats[i, 2] = float(np.sqrt(np.mean((p_o - p_s)**2))) # RMSE
        feats[i, 3] = float(np.std(roll_orig[i]))

        # 2. 고개 정면 타이밍
        ns_o = neck_spine[i]
        ns_s = neck_spine_shift[i]
        feats[i, 4] = float(np.max(np.abs(ns_o - ns_s)))
        feats[i, 5] = float(np.mean(np.abs(ns_o - ns_s)))
        feats[i, 6] = float(np.std(np.diff(ns_s)))

        # 3. 무릎 외반 전후 위상차
        vl = valgus_l[i]
        vl_s = valgus_l_shift[i]
        vr = valgus_r[i]
        vr_s = valgus_r_shift[i]
        feats[i, 7] = float(np.max(np.abs(vl - vl_s)))
        feats[i, 8] = float(np.max(np.abs(vr - vr_s)))
        feats[i, 9] = float(np.sum(np.abs(vl * vr_s - vr * vl_s))) # 외반 히스테리시스
        feats[i, 10] = float(np.mean(knee_hip_sync[i]))

        # 4. 발바닥 지면 고정 및 정강이
        feats[i, 11] = float(np.std(shank_l[i]))
        feats[i, 12] = float(np.std(shank_r[i]))
        feats[i, 13] = float(np.max(ankle_z[i]) - np.min(ankle_z[i]))
        # 임펄스 (가속도 차분)
        feats[i, 14] = float(np.mean(np.abs(np.diff(ankle_z[i], n=2))))
        # 최하사점(변곡점) 위상 갭: 무릎 각도 최저점 주변의 피치 변화
        knee_mean = (X_b[i, :, 56] + X_b[i, :, 57]) / 2.0
        bot_idx = int(np.argmin(knee_mean))
        if 1 <= bot_idx < T - 1:
            feats[i, 15] = float(np.abs(p_o[bot_idx+1] - p_o[bot_idx-1]))
        else:
            feats[i, 15] = 0.0

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
