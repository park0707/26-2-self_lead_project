# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c3 (안 굽히는 다리 구부정 / 신전 결함 - Knee Extension Defect of Trail Leg)
Module: side_lunge_c3_features.py
Purpose:
  - 사이드 런지 3번 결함("안 굽히는 다리 구부정 / 신전 미흡") 판정을 위한 20차원 정예 생체역학 도메인 피처 추출기.
  - 100% 순수 비전 MediaPipe 33관절 3D (64, 33, 3) 및 15관절 호환.
  - 스포츠 생체역학(IEEE TBME, Kinesiology) 기반:
    1. 작동 다리(Lead Leg) 최하사점(Bottom Phase) 동기화 및 신전 다리(Trail Leg) 분리 식별
    2. 신전 다리 3D 무릎 각도, 신전 결손도(180도 기준 편차), 바닥 윈도우 평균
    3. 양다리 무릎 각도 절대차 및 상대 비율 (비대칭성 지표)
    4. 대퇴-하퇴 3D 직선성 지표 (Hip-Knee-Ankle 거리비 및 방향 코사인)
    5. 횡방향(X축) 외측 변위량 및 양 무릎 간 3D 거리
    6. 시점 편향 자가 보정을 위한 체간비 및 관상면 투영 정렬 피처
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional, Union


FEATURE_NAMES_20D: List[str] = [
    # [1] 신전 다리 1차 핵심 각도 지표 (5D)
    'trail_knee_angle_bottom',             # 1. 최하사점 신전 다리 3D 무릎 각도 (°) (정상: ~160-175, 결함: ~105-125)
    'trail_knee_angle_min',                # 2. 전 프레임 신전 다리 최소 3D 무릎 각도 (°)
    'trail_knee_angle_mean_bottom_win',    # 3. 최하사점 전후 5프레임 신전 다리 무릎 각도 평균 (°)
    'trail_knee_angle_active_mean',        # 4. 작동 다리 굴곡 활성 구간(Bottom 30%) 신전 다리 무릎 각도 평균 (°)
    'trail_knee_extension_defect_margin',  # 5. 완전 신전(180°) 대비 결손각 (180 - trail_knee_angle_bottom) (°)

    # [2] 양 무릎 상대 비교 및 비대칭성 지표 (4D)
    'inter_knee_angle_diff_bottom',        # 6. 최하사점 양 무릎 각도 절대차 (|trail - lead|) (정상: ~90-110, 결함: ~40-60)
    'lead_to_trail_knee_ratio_bottom',     # 7. 최하사점 작동/신전 무릎 각도비 (lead / trail)
    'inter_knee_angle_diff_min',           # 8. 활성 구간 양 무릎 각도 차이 최소값 (°)
    'trail_knee_rom',                      # 9. 신전 다리 운동 가동범위 (max - min) (정상은 작고 결함은 큼)

    # [3] 기구학적 3D 선형성 및 공간 거리 지표 (5D)
    'trail_leg_collinearity_ratio',        # 10. 신전 다리 3D 직선성 거리비 ||H-A|| / (||H-K|| + ||K-A||) (정상: >0.96, 결함: <0.88)
    'trail_femur_shank_cos_similarity',    # 11. 대퇴(K-H)와 정강이(A-K) 벡터 3D 방향 일치도 (180도 일직선 시 ~1.0)
    'inter_knee_distance_bottom',          # 12. 최하사점 양 무릎 간 3D 유클리드 거리 (체간길이 정규화)
    'trail_knee_lateral_disp_bottom',      # 13. 최하사점 신전 다리 무릎의 횡방향(X축) 외측 변위량 (체간길이 정규화)
    'inter_ankle_distance_bottom',         # 14. 최하사점 양 발목 간 3D 보폭 거리 (체간길이 정규화)

    # [4] 동역학 및 안정성 지표 (3D)
    'trail_knee_stability_std',            # 15. 최하사점 윈도우 신전 다리 각도 표준편차 (흔들림/미세 굴곡 불안정성)
    'trail_knee_velocity_bottom',          # 16. 최하사점 신전 다리 무릎 각속도 (deg/frame)
    'trail_knee_accel_bottom',             # 17. 최하사점 신전 다리 무릎 각가속도 (deg/frame^2)

    # [5] 시점 자가 보정 및 앵커 피처 (3D)
    'lead_knee_angle_bottom',              # 18. 최하사점 작동 다리 3D 무릎 각도 (운동 깊이 기준 앵커) (°)
    'pelvis_width_to_torso_ratio',         # 19. 골반폭 대비 체간길이 비율 (카메라 촬영 각도/방위각 추정용)
    'coronal_plane_proj_ratio'             # 20. 관상면(Frontal XY) 투영 신전 대퇴/하퇴 길이비 (시점 앙각 편향 보정)
]


def _calc_vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 3D 벡터 간 사이각 (0° ~ 180°) 계산"""
    norm1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-7
    norm2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-7
    dot = np.sum((v1 / norm1) * (v2 / norm2), axis=-1)
    cos_theta = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _calc_cosine_similarity_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 3D 벡터 간 코사인 유사도 (-1.0 ~ 1.0)"""
    norm1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-7
    norm2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-7
    dot = np.sum((v1 / norm1) * (v2 / norm2), axis=-1)
    return np.clip(dot, -1.0, 1.0)


def extract_side_lunge_c3_features_20d(kps_seq: np.ndarray) -> np.ndarray:
    """
    단일 64프레임 관절 좌표 배열로부터 20D 정예 도메인 생체역학 피처 추출
    Input:
        kps_seq: (64, 33, 3) MediaPipe 33관절 3D 또는 (64, 15, 3) 15관절
    Output:
        feat_vec: (20,) np.float32 배열
    """
    assert kps_seq.ndim == 3 and kps_seq.shape[0] == 64, f"입력 규격 불일치: {kps_seq.shape}"
    num_joints = kps_seq.shape[1]

    if num_joints == 33:
        # MediaPipe 33관절 매핑
        lh = kps_seq[:, 23, :3]  # Left Hip
        rh = kps_seq[:, 24, :3]  # Right Hip
        lk = kps_seq[:, 25, :3]  # Left Knee
        rk = kps_seq[:, 26, :3]  # Right Knee
        la = kps_seq[:, 27, :3]  # Left Ankle
        ra = kps_seq[:, 28, :3]  # Right Ankle
        ls = kps_seq[:, 11, :3]  # Left Shoulder
        rs = kps_seq[:, 12, :3]  # Right Shoulder
    elif num_joints >= 15:
        # 15관절 매핑
        lh = kps_seq[:, 8, :3]
        rh = kps_seq[:, 9, :3]
        lk = kps_seq[:, 10, :3]
        rk = kps_seq[:, 11, :3]
        la = kps_seq[:, 12, :3]
        ra = kps_seq[:, 13, :3]
        ls = kps_seq[:, 5, :3] if num_joints > 5 else kps_seq[:, 1, :3]
        rs = kps_seq[:, 6, :3] if num_joints > 6 else kps_seq[:, 1, :3]
    else:
        raise ValueError(f"지원하지 않는 관절 수: {num_joints}")

    # 골반 중심 및 체간 중심 계산
    mid_hip = (lh + rh) / 2.0
    mid_shoulder = (ls + rs) / 2.0
    torso_len_t = np.linalg.norm(mid_shoulder - mid_hip, axis=-1)
    ref_torso_len = float(np.median(torso_len_t)) + 1e-6
    ref_pelvis_w = float(np.median(np.linalg.norm(lh - rh, axis=-1))) + 1e-6

    # 1. 좌/우 3D 무릎 각도 시계열 계산 (0~180°)
    v_femur_l = lk - lh
    v_shank_l = la - lk
    ang_l = 180.0 - _calc_vector_angle_3d(v_femur_l, v_shank_l)

    v_femur_r = rk - rh
    v_shank_r = ra - rk
    ang_r = 180.0 - _calc_vector_angle_3d(v_femur_r, v_shank_r)

    # Savitzky-Golay 필터링 (윈도우 7, 2차 다항식)
    try:
        ang_l_s = savgol_filter(ang_l, 7, 2)
        ang_r_s = savgol_filter(ang_r, 7, 2)
    except Exception:
        ang_l_s = ang_l
        ang_r_s = ang_r

    # 2. 작동 다리(Lead)와 신전 다리(Trail) 자동 분리 식별
    min_l = np.min(ang_l_s)
    min_r = np.min(ang_r_s)

    if min_l < min_r:
        lead_ang = ang_l_s
        trail_ang = ang_r_s
        lead_h, lead_k, lead_a = lh, lk, la
        trail_h, trail_k, trail_a = rh, rk, ra
        v_fem_trail = v_femur_r
        v_shk_trail = v_shank_r
    else:
        lead_ang = ang_r_s
        trail_ang = ang_l_s
        lead_h, lead_k, lead_a = rh, rk, ra
        trail_h, trail_k, trail_a = lh, lk, la
        v_fem_trail = v_femur_l
        v_shk_trail = v_shank_l

    # 3. 최하사점(Bottom Phase) 동기화
    bottom_idx = int(np.argmin(lead_ang))
    w_start = max(0, bottom_idx - 2)
    w_end = min(64, bottom_idx + 3)

    # 활성 구간 (작동 다리가 바닥 30% 굴곡 구간에 머무는 프레임들)
    lead_rom = np.ptp(lead_ang) + 1e-6
    active_mask = (lead_ang <= (min_l if min_l < min_r else min_r) + 0.35 * lead_rom)
    if np.sum(active_mask) == 0:
        active_mask = np.zeros(64, dtype=bool)
        active_mask[w_start:w_end] = True

    # --- 피처 산출 시작 ---
    # [1] 신전 다리 1차 핵심 각도 지표 (5D)
    f1 = float(trail_ang[bottom_idx])
    f2 = float(np.min(trail_ang))
    f3 = float(np.mean(trail_ang[w_start:w_end]))
    f4 = float(np.mean(trail_ang[active_mask]))
    f5 = float(max(0.0, 180.0 - f1))

    # [2] 양 무릎 상대 비교 및 비대칭성 지표 (4D)
    f6 = float(abs(trail_ang[bottom_idx] - lead_ang[bottom_idx]))
    f7 = float(lead_ang[bottom_idx] / (trail_ang[bottom_idx] + 1e-5))
    diff_t = np.abs(trail_ang - lead_ang)
    f8 = float(np.min(diff_t[active_mask]))
    f9 = float(np.ptp(trail_ang))

    # [3] 기구학적 3D 선형성 및 공간 거리 지표 (5D)
    d_ha = np.linalg.norm(trail_h[bottom_idx] - trail_a[bottom_idx])
    d_hk = np.linalg.norm(trail_h[bottom_idx] - trail_k[bottom_idx])
    d_ka = np.linalg.norm(trail_k[bottom_idx] - trail_a[bottom_idx])
    f10 = float(d_ha / (d_hk + d_ka + 1e-7))

    # 대퇴와 정강이 방향 정렬 코사인
    cos_sim = _calc_cosine_similarity_3d(v_fem_trail[bottom_idx], v_shk_trail[bottom_idx])
    f11 = float(cos_sim)

    # 양 무릎 거리 및 발목 보폭 거리
    d_knees = np.linalg.norm(lead_k[bottom_idx] - trail_k[bottom_idx]) / ref_torso_len
    f12 = float(d_knees)

    # 신전 다리 무릎 횡방향 외측 변위
    lat_disp = abs(trail_k[bottom_idx, 0] - mid_hip[bottom_idx, 0]) / ref_torso_len
    f13 = float(lat_disp)

    d_ankles = np.linalg.norm(lead_a[bottom_idx] - trail_a[bottom_idx]) / ref_torso_len
    f14 = float(d_ankles)

    # [4] 동역학 및 안정성 지표 (3D)
    w_stab = trail_ang[max(0, bottom_idx - 3): min(64, bottom_idx + 4)]
    f15 = float(np.std(w_stab))

    vel_t = np.gradient(trail_ang)
    acc_t = np.gradient(vel_t)
    f16 = float(vel_t[bottom_idx])
    f17 = float(acc_t[bottom_idx])

    # [5] 시점 자가 보정 및 앵커 피처 (3D)
    f18 = float(lead_ang[bottom_idx])
    f19 = float(ref_pelvis_w / ref_torso_len)

    # 2D 관상면(XY) 투영 대퇴/하퇴 길이비
    fem_2d = np.linalg.norm(v_fem_trail[bottom_idx, :2])
    shk_2d = np.linalg.norm(v_shk_trail[bottom_idx, :2])
    f20 = float(fem_2d / (shk_2d + 1e-7))

    feat_vec = np.array([
        f1, f2, f3, f4, f5,
        f6, f7, f8, f9,
        f10, f11, f12, f13, f14,
        f15, f16, f17,
        f18, f19, f20
    ], dtype=np.float32)

    # 결측치 방어
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_vec


def extract_side_lunge_c3_features_batch(kps_batch: np.ndarray) -> np.ndarray:
    """배치 시퀀스 (N, 64, 33, 3)에 대해 20D 피처 일괄 추출"""
    N = kps_batch.shape[0]
    feats = np.zeros((N, 20), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_side_lunge_c3_features_20d(kps_batch[i])
    return feats
