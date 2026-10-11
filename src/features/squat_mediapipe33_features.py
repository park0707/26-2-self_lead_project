# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: squat_mediapipe33_features.py
Purpose: MediaPipe 33관절 64프레임 무삭제 스플라인 데이터 기반 순수 비전 생체역학 피처 추출기
         - AI-Hub 3D JSON 완전 배제 (100% MediaPipe 33개 관절 기반)
         - 횟수 분할 100% 배제 (64프레임 전 시퀀스 무삭제 스플라인 기반)
         - Active Motion Masking 기본 탑재 (기립 시 잉여 프레임/스텝 오탐 원천 차단)
         - 4대 결함 서브모델(c1, c2, c3, c4) 전용 피처 산출 지원
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.ndimage import gaussian_filter1d

# MediaPipe 33 관절 인덱스 정의
MP_NOSE = 0
MP_LEFT_EYE_INNER = 1
MP_LEFT_EYE = 2
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_INNER = 4
MP_RIGHT_EYE = 5
MP_RIGHT_EYE_OUTER = 6
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_MOUTH_LEFT = 9
MP_MOUTH_RIGHT = 10
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_PINKY = 17
MP_RIGHT_PINKY = 18
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_THUMB = 21
MP_RIGHT_THUMB = 22
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31  # Left Toe
MP_RIGHT_FOOT_INDEX = 32 # Right Toe

# Joint15 인덱스 정의
J15_PELVIS = 0
J15_TORSO = 1
J15_NECK = 2
J15_HEAD = 3
J15_LEFT_SHOULDER = 4
J15_RIGHT_SHOULDER = 5
J15_LEFT_HIP = 8
J15_RIGHT_HIP = 9
J15_LEFT_KNEE = 10
J15_RIGHT_KNEE = 11
J15_LEFT_ANKLE = 12
J15_RIGHT_ANKLE = 13


def compute_squat_active_motion_mask(
    kps_33_3d: np.ndarray,
    rom_threshold: float = 0.05,
    cutoff_ratio: float = 0.85
) -> Tuple[np.ndarray, int]:
    """
    스쿼트 능동 운동 구간 마스크 (Active Motion Mask) 및 최하사점(Nadir) 프레임 산출
    - 반환: (active_mask: bool(64,), bottom_idx: int)
    """
    T = kps_33_3d.shape[0]
    # 양 고관절/무릎의 y좌표(높이) 궤적 분석
    hip_y = (kps_33_3d[:, MP_LEFT_HIP, 1] + kps_33_3d[:, MP_RIGHT_HIP, 1]) / 2.0
    knee_y = (kps_33_3d[:, MP_LEFT_KNEE, 1] + kps_33_3d[:, MP_RIGHT_KNEE, 1]) / 2.0
    depth_traj = (hip_y + knee_y) / 2.0

    # 평활화
    depth_s = gaussian_filter1d(depth_traj, sigma=2.0)
    # MediaPipe y좌표는 아래로 갈수록 커지거나(월드좌표계는 음수), 깊이 변곡점 탐색
    # 월드좌표계에서 스쿼트 하강 시 골반/무릎 y는 최솟값(또는 부호에 따라 극값)
    # 가동범위 분석
    min_idx = int(np.argmin(depth_s))
    max_idx = int(np.argmax(depth_s))

    # 안전 클램핑 (초기 4프레임, 말기 4프레임 배제)
    b = int(np.clip(min_idx, 4, T - 5))

    rom = float(np.max(depth_s) - np.min(depth_s))
    if rom > rom_threshold:
        # 가동범위 하위 85% 구간을 유효 운동 구간으로 정의
        active_mask = (depth_s - np.min(depth_s)) <= (rom * cutoff_ratio)
    else:
        active_mask = np.ones(T, dtype=bool)

    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    return active_mask, b


def extract_c2_gated_gaze_features(kps_33_3d: np.ndarray) -> np.ndarray:
    """
    바벨 스쿼트 c2(고개 정면) 게이트형 3D 시선 피처 (3D)
    100% MediaPipe 33 관절 기반 (AI-Hub 3D JSON 완전 배제)
    - feat 0: c2_gated_true_gaze_pitch_drop (기립 대비 최하사점 시선 낙하폭, deg)
    - feat 1: c2_gated_true_gaze_pitch_bot  (최하사점 시선 절대각도, deg)
    - feat 2: is_face_visible_gate          (체간 정면/측방 게이트 활성화 여부: 1.0 or 0.0)
    """
    T = kps_33_3d.shape[0]
    ls = kps_33_3d[:, MP_LEFT_SHOULDER, :]
    rs = kps_33_3d[:, MP_RIGHT_SHOULDER, :]
    lh = kps_33_3d[:, MP_LEFT_HIP, :]
    rh = kps_33_3d[:, MP_RIGHT_HIP, :]

    neck = (ls + rs) / 2.0
    pelvis = (lh + rh) / 2.0

    # 1. 체간 법선 벡터 (Torso Normal Vector) 계산
    v_sh = ls - rs
    v_spine = neck - pelvis
    n_torso = np.cross(v_sh, v_spine)
    n_norms = np.linalg.norm(n_torso, axis=1, keepdims=True) + 1e-6
    n_torso_norm = n_torso / n_norms

    # 전방/측방 시점 게이트: n_torso_z >= -0.15 이면 정면/측면 관측 가능
    initial_nz = float(np.mean(n_torso_norm[:8, 2]))

    # 얼굴 관절 유효성 검사 (코 및 귀 관절 존재 여부)
    nose = kps_33_3d[:, MP_NOSE, :]
    lear = kps_33_3d[:, MP_LEFT_EAR, :]
    rear = kps_33_3d[:, MP_RIGHT_EAR, :]

    face_detected = (np.linalg.norm(nose, axis=1) > 1e-4) & (
        (np.linalg.norm(lear, axis=1) > 1e-4) | (np.linalg.norm(rear, axis=1) > 1e-4)
    )
    face_valid_ratio = float(np.mean(face_detected))

    # 후방(view1) 또는 얼굴 가려짐 시 즉각 Fallback (더미 0.0)
    if initial_nz < -0.15 or face_valid_ratio < 0.30:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # 2. 3D 시선 벡터 u_gaze = Nose - HeadMid
    head_mid = (lear + rear) / 2.0
    u_gaze = nose - head_mid

    # 시선 피치각 (상하 각도, deg): -y는 상향, +y는 하향
    horiz_dist = np.sqrt(u_gaze[:, 0]**2 + u_gaze[:, 2]**2) + 1e-6
    pitch_series = np.arctan2(-u_gaze[:, 1], horiz_dist) * (180.0 / np.pi)

    # 능동 구간 및 최하사점 프레임
    active_mask, b = compute_squat_active_motion_mask(kps_33_3d)

    # 기립 초기 기준 각도 (첫 8프레임 평균)
    standing_slice = max(2, T // 8)
    p_standing = float(np.mean(pitch_series[:standing_slice]))
    p_bot = float(pitch_series[b])
    p_drop = float(p_standing - p_bot)

    return np.array([p_drop, p_bot, 1.0], dtype=np.float32)


def extract_c4_foot_contact_features(kps_33_3d: np.ndarray) -> np.ndarray:
    """
    바벨 스쿼트 c4(발바닥 지면 고정) 순수 MediaPipe 33 기반 지면 접지 피처 (10D)
    100% MediaPipe 33 관절 기반 (뒤꿈치 29/30, 앞꿈치 31/32 직접 계측!)
    - c4_true_foot_pitch_nadir: 최하사점 발바닥(뒤꿈치-앞꿈치) 경사각 (deg)
    - c4_true_foot_pitch_drop: 기립 대비 최하사점 경사각 변화폭 (deg)
    - c4_true_foot_pitch_max: 전 구간 최대 경사각 (deg)
    - c4_true_heel_toe_ydelta_nadir: 최하사점 뒤꿈치-앞꿈치 수직 높이차 (cm)
    - c4_true_heel_toe_ydelta_drop: 기립 대비 높이차 변화량 (cm)
    - c4_true_heel_lift_max: 기립 기준 뒤꿈치 최대 수직 들림량 (cm)
    - c4_true_foot_ground_wobble: 높이차 시계열 표준편차 (들썩임, cm)
    - c4_true_bilateral_foot_pitch_asym: 좌우 발바닥 경사각 비대칭도 (deg)
    - c4_true_toe_lift_max: 기립 기준 앞꿈치 최대 수직 들림량 (cm)
    - c4_true_foot_roll_inversion: 최하사점 발목-발뒤꿈치 좌우 롤/안쪽날 들림 (deg)
    """
    T = kps_33_3d.shape[0]

    # 관절 추출 (미터 단위)
    la = kps_33_3d[:, MP_LEFT_ANKLE, :]
    ra = kps_33_3d[:, MP_RIGHT_ANKLE, :]
    lh = kps_33_3d[:, MP_LEFT_HEEL, :]
    rh = kps_33_3d[:, MP_RIGHT_HEEL, :]
    lt = kps_33_3d[:, MP_LEFT_FOOT_INDEX, :] # Left Toe
    rt = kps_33_3d[:, MP_RIGHT_FOOT_INDEX, :] # Right Toe
    lk = kps_33_3d[:, MP_LEFT_KNEE, :]
    rk = kps_33_3d[:, MP_RIGHT_KNEE, :]

    active_mask, b = compute_squat_active_motion_mask(kps_33_3d)

    # 1. 발바닥 경사각 (Foot Pitch, 뒤꿈치 -> 앞꿈치 벡터의 수직 경사각)
    # MediaPipe 월드좌표계: y는 상하 (미터 단위)
    # Left foot vector
    v_foot_l = lt - lh
    v_foot_r = rt - rh

    horiz_l = np.sqrt(v_foot_l[:, 0]**2 + v_foot_l[:, 2]**2) + 1e-6
    horiz_r = np.sqrt(v_foot_r[:, 0]**2 + v_foot_r[:, 2]**2) + 1e-6

    pitch_l = np.arctan2(v_foot_l[:, 1], horiz_l) * (180.0 / np.pi)
    pitch_r = np.arctan2(v_foot_r[:, 1], horiz_r) * (180.0 / np.pi)
    pitch_mean = (pitch_l + pitch_r) / 2.0

    # 2. 뒤꿈치-앞꿈치 수직 높이차 (Heel - Toe y, cm 단위 변환 * 100)
    dy_l = (lh[:, 1] - lt[:, 1]) * 100.0
    dy_r = (rh[:, 1] - rt[:, 1]) * 100.0
    dy_mean = (dy_l + dy_r) / 2.0

    # 3. 기립 초기 기준면 산출 (첫 8프레임 평균)
    standing_slice = max(2, T // 8)
    base_dy = float(np.mean(dy_mean[:standing_slice]))
    base_pitch = float(np.mean(pitch_mean[:standing_slice]))

    base_heel_y = float(np.mean((lh[:standing_slice, 1] + rh[:standing_slice, 1]) / 2.0)) * 100.0
    base_toe_y = float(np.mean((lt[:standing_slice, 1] + rt[:standing_slice, 1]) / 2.0)) * 100.0

    # 능동 구간 내 피처 계산
    dy_active = dy_mean[active_mask]
    pitch_active = pitch_mean[active_mask]
    heel_y_active = ((lh[:, 1] + rh[:, 1]) / 2.0)[active_mask] * 100.0
    toe_y_active = ((lt[:, 1] + rt[:, 1]) / 2.0)[active_mask] * 100.0

    # 1) 최하사점 발바닥 경사각
    p_nadir = float(pitch_mean[b])
    # 2) 기립 대비 경사각 증가폭
    p_drop = float(p_nadir - base_pitch)
    # 3) 전 구간 최대 경사각
    p_max = float(np.max(pitch_active)) if len(pitch_active) > 0 else p_nadir
    # 4) 최하사점 높이차
    dy_nadir = float(dy_mean[b])
    # 5) 높이차 증가량
    dy_drop = float(dy_nadir - base_dy)
    # 6) 뒤꿈치 최대 수직 들림량 (cm)
    heel_lift_max = float(np.max(np.abs(heel_y_active - base_heel_y))) if len(heel_y_active) > 0 else 0.0
    # 7) 지면 들썩임 (Wobble std)
    ground_wobble = float(np.std(dy_active)) if len(dy_active) > 0 else 0.0
    # 8) 좌우 비대칭도
    asym_nadir = float(np.abs(pitch_l[b] - pitch_r[b]))
    # 9) 앞꿈치 최대 들림량 (cm)
    toe_lift_max = float(np.max(np.abs(toe_y_active - base_toe_y))) if len(toe_y_active) > 0 else 0.0
    # 10) 발목 롤 / 안쪽날 들림 (deg)
    roll_l = np.arctan2(la[:, 0] - lh[:, 0], np.abs(la[:, 1] - lh[:, 1]) + 1e-6) * (180.0 / np.pi)
    roll_r = np.arctan2(ra[:, 0] - rh[:, 0], np.abs(ra[:, 1] - rh[:, 1]) + 1e-6) * (180.0 / np.pi)
    roll_inversion = float(np.abs(roll_l[b] - roll_r[b]))

    return np.array([
        p_nadir, p_drop, p_max, dy_nadir, dy_drop,
        heel_lift_max, ground_wobble, asym_nadir, toe_lift_max, roll_inversion
    ], dtype=np.float32)


def extract_c4_elite_9d_features(kps_33_3d: np.ndarray) -> np.ndarray:
    """
    바벨 스쿼트 c4_model_v3.py와 100% 호환되는 9D 정예 피처 산출
    (7D 지면 접지 피처 + 2D 정강이/발목 커플링 피처)
    """
    feats10 = extract_c4_foot_contact_features(kps_33_3d)

    # 정강이-발목 커플링 피처 계산 (MediaPipe 33 관절 기반)
    lk = kps_33_3d[:, MP_LEFT_KNEE, :]
    rk = kps_33_3d[:, MP_RIGHT_KNEE, :]
    la = kps_33_3d[:, MP_LEFT_ANKLE, :]
    ra = kps_33_3d[:, MP_RIGHT_ANKLE, :]

    # 정강이 기울기 (Shank Angle)
    v_shank_l = lk - la
    v_shank_r = rk - ra
    shank_angle_l = np.arctan2(v_shank_l[:, 0], np.abs(v_shank_l[:, 1]) + 1e-6) * (180.0 / np.pi)
    shank_angle_r = np.arctan2(v_shank_r[:, 0], np.abs(v_shank_r[:, 1]) + 1e-6) * (180.0 / np.pi)
    shank_mean = (shank_angle_l + shank_angle_r) / 2.0

    # 1) shank_ankle_coupling_ratio (피치와 정강이 각도의 상관계수)
    active_mask, b = compute_squat_active_motion_mask(kps_33_3d)
    lh = kps_33_3d[:, MP_LEFT_HEEL, :]
    rh = kps_33_3d[:, MP_RIGHT_HEEL, :]
    lt = kps_33_3d[:, MP_LEFT_FOOT_INDEX, :]
    rt = kps_33_3d[:, MP_RIGHT_FOOT_INDEX, :]
    v_foot_l = lt - lh
    v_foot_r = rt - rh
    horiz_l = np.sqrt(v_foot_l[:, 0]**2 + v_foot_l[:, 2]**2) + 1e-6
    horiz_r = np.sqrt(v_foot_r[:, 0]**2 + v_foot_r[:, 2]**2) + 1e-6
    pitch_mean = ((np.arctan2(v_foot_l[:, 1], horiz_l) + np.arctan2(v_foot_r[:, 1], horiz_r)) / 2.0) * (180.0 / np.pi)

    if np.sum(active_mask) > 6 and np.std(shank_mean[active_mask]) > 1e-3 and np.std(pitch_mean[active_mask]) > 1e-3:
        corr = float(np.corrcoef(shank_mean[active_mask], pitch_mean[active_mask])[0, 1])
        coupling_ratio = corr if not np.isnan(corr) else 0.5
    else:
        coupling_ratio = 0.5

    # 2) ascent_ankle_destabilization (상승기 발목 흔들림)
    ascent_slice = slice(b, 64)
    if (64 - b) > 4:
        ascent_destab = float(np.std(la[ascent_slice, 1] + ra[ascent_slice, 1])) * 100.0
    else:
        ascent_destab = 0.0

    # 9D 구성:
    # 0: p_nadir, 1: p_drop, 2: p_max, 3: dy_nadir, 4: asym_nadir, 5: toe_lift_max,
    # 6: roll_inversion, 7: coupling_ratio, 8: ascent_destab
    return np.array([
        feats10[0], # c4_true_foot_pitch_nadir
        feats10[1], # c4_true_foot_pitch_drop
        feats10[2], # c4_true_foot_pitch_max
        feats10[3], # c4_true_heel_toe_ydelta_nadir
        feats10[7], # c4_true_bilateral_foot_pitch_asym
        feats10[8], # c4_true_toe_lift_max
        feats10[9], # c4_true_foot_roll_inversion
        coupling_ratio, # c4_shank_ankle_coupling_ratio
        ascent_destab   # c4_ascent_ankle_destabilization
    ], dtype=np.float32)


def extract_c3_valgus_10d_features(kps_33_3d: np.ndarray) -> np.ndarray:
    """
    바벨 스쿼트 c3(무릎-발 방향 일치 / 무릎 외반) 10D 정예 피처 산출
    100% MediaPipe 33 관절 기반
    """
    T = kps_33_3d.shape[0]
    lk = kps_33_3d[:, MP_LEFT_KNEE, :]
    rk = kps_33_3d[:, MP_RIGHT_KNEE, :]
    la = kps_33_3d[:, MP_LEFT_ANKLE, :]
    ra = kps_33_3d[:, MP_RIGHT_ANKLE, :]
    lh = kps_33_3d[:, MP_LEFT_HIP, :]
    rh = kps_33_3d[:, MP_RIGHT_HIP, :]

    active_mask, b = compute_squat_active_motion_mask(kps_33_3d)

    # 1. 무릎-발목 너비 비율 (Knee-to-Ankle Ratio)
    knee_width = np.linalg.norm(lk - rk, axis=1)
    ankle_width = np.linalg.norm(la - ra, axis=1)
    ka_ratio = knee_width / (ankle_width + 1e-6)

    ka_nadir = float(ka_ratio[b])
    ka_min = float(np.min(ka_ratio[active_mask])) if np.sum(active_mask) > 0 else ka_nadir
    ka_standing = float(np.mean(ka_ratio[:max(2, T // 8)]))

    valgus_dur_ratio = float(np.mean(ka_ratio[active_mask] < 0.85)) if np.sum(active_mask) > 0 else 0.0
    deficit_integ = float(np.sum(np.maximum(0.0, 0.90 - ka_ratio[active_mask])))
    chronic_valgus = 1.0 if ka_standing < 0.85 else 0.0

    # 2. 좌우 독립 외반 변위 및 절댓값 합
    # 골반 대비 무릎의 내측 편위량 (x축 기준)
    vl = (lk[:, 0] - lh[:, 0]) # 양수일수록 외측, 음수일수록 내측
    vr = (rh[:, 0] - rk[:, 0]) # 양수일수록 외측, 음수일수록 내측

    valgus_abs_sum = np.abs(vl) + np.abs(vr)
    valgus_sum_nadir = float(valgus_abs_sum[b]) * 100.0
    valgus_sum_max = float(np.max(valgus_abs_sum[active_mask])) * 100.0 if np.sum(active_mask) > 0 else valgus_sum_nadir

    # 양 무릎 동시 내측 붕괴 강도
    concurrence = np.minimum(-vl, -vr) # 둘 다 내측으로 무너질 때 양수
    concurrence_max = float(np.max(concurrence[active_mask])) * 100.0 if np.sum(active_mask) > 0 else 0.0
    bilateral_min = float(min(vl[b], vr[b])) * 100.0

    return np.array([
        ka_nadir, ka_min, ka_standing, valgus_dur_ratio, deficit_integ,
        chronic_valgus, valgus_sum_nadir, valgus_sum_max, concurrence_max, bilateral_min
    ], dtype=np.float32)


def extract_c1_spine_features(kps_15_3d: np.ndarray) -> np.ndarray:
    """
    바벨 스쿼트 c1(척추의 중립 위반) 10D 기하학 생체역학 피처 산출
    100% MediaPipe 관절 기반 (Joint15 3D 좌표)
    """
    T = kps_15_3d.shape[0]
    pelvis = kps_15_3d[:, J15_PELVIS, :]
    torso = kps_15_3d[:, J15_TORSO, :]
    neck = kps_15_3d[:, J15_NECK, :]
    lk = kps_15_3d[:, J15_LEFT_KNEE, :]
    rk = kps_15_3d[:, J15_RIGHT_KNEE, :]
    knee = (lk + rk) / 2.0

    # 능동 구간 산출
    depth = (pelvis[:, 1] + knee[:, 1]) / 2.0
    b = int(np.clip(np.argmin(depth), 4, T - 5))

    # 1. 체간 기울기 (Trunk Angle, 수직선 기준 각도, deg)
    v_trunk = neck - pelvis
    horiz = np.sqrt(v_trunk[:, 0]**2 + v_trunk[:, 2]**2) + 1e-6
    trunk_angle = np.arctan2(horiz, -v_trunk[:, 1]) * (180.0 / np.pi)

    trunk_nadir = float(trunk_angle[b])
    trunk_standing = float(np.mean(trunk_angle[:max(2, T // 8)]))
    trunk_drop = float(trunk_nadir - trunk_standing)
    trunk_max = float(np.max(trunk_angle))

    # 2. 척추 곡률 / 요추 전만-후만 프록시 (Lumbar Curvature)
    # v_lower: pelvis -> torso, v_upper: torso -> neck
    v_lower = torso - pelvis
    v_upper = neck - torso
    norm_l = np.linalg.norm(v_lower, axis=1, keepdims=True) + 1e-6
    norm_u = np.linalg.norm(v_upper, axis=1, keepdims=True) + 1e-6
    cos_spine = np.sum((v_lower / norm_l) * (v_upper / norm_u), axis=1)
    spine_flexion = np.arccos(np.clip(cos_spine, -1.0, 1.0)) * (180.0 / np.pi)

    spine_flex_nadir = float(spine_flexion[b])
    spine_flex_max = float(np.max(spine_flexion))
    spine_flex_standing = float(np.mean(spine_flexion[:max(2, T // 8)]))
    spine_flex_delta = float(spine_flex_nadir - spine_flex_standing)

    # 3. 굿모닝 스쿼트 프록시 (엉덩이 선행 상승 비율)
    # 상승기(b ~ b+15)에서 골반과 어깨의 수직 속도 비
    ascent_slice = slice(b, min(T, b + 16))
    if (min(T, b + 16) - b) > 4:
        vel_pelvis = float(np.mean(np.gradient(pelvis[ascent_slice, 1])))
        vel_neck = float(np.mean(np.gradient(neck[ascent_slice, 1])))
        good_morning_ratio = float(vel_pelvis / (vel_neck + 1e-6))
    else:
        good_morning_ratio = 1.0

    # 4. 버트윙크 프록시 (최하사점 직전 요추 곡률 가속도)
    butt_wink_score = float(np.abs(spine_flexion[b] - spine_flexion[max(0, b - 3)]))

    return np.array([
        trunk_nadir, trunk_drop, trunk_max, spine_flex_nadir, spine_flex_max,
        spine_flex_delta, spine_flex_standing, good_morning_ratio, butt_wink_score,
        float(trunk_angle[b] - spine_flexion[b])
    ], dtype=np.float32)

