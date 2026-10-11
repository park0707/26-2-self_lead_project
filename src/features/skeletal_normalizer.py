# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: skeletal_normalizer.py
Purpose: MediaPipe 3D 관절 좌표의 단안 깊이(z) 착시 및 고무줄 뼈 길이 왜곡(Z-Wobble)을 해소하는
         뼈 길이 불변 정규화(Skeletal Length Normalization) 및 Dual-Stream 보존 엔진
         - [스트림 A (Retargeted/FK)]: 직립 기준 뼈 길이(L_ref) 불변 정규화 좌표계 (척추/무릎 각도용)
         - [스트림 B (Raw)]: 원천 발 접지 변위 100% 보존 좌표계 (c4 발바닥 지면 고정용)
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np


def normalize_skeletal_lengths(
    kps_3d: np.ndarray,
    eps: float = 1e-7
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    MediaPipe 33관절 3D 시퀀스에 대해 피험자 고유 뼈 길이를 직립 구간에서 추정하고,
    계층적 Forward Kinematics(FK)를 통해 뼈 길이를 엄격히 고정(Stream A)함과 동시에
    원천 접지 좌표계(Stream B)를 보존하는 Dual-Stream 정규화기.

    Args:
        kps_3d: (T, 33, 3) 또는 (64, 33, 3) 3D 관절 좌표 배열 (미터 또는 정규화 단위)
        eps: 0 분모 방지 작은 값

    Returns:
        kps_stream_a: (T, 33, 3) 뼈 길이가 완벽히 고정된 정규화 좌표 (c1, c3, 운동사슬용)
        kps_stream_b: (T, 33, 3) 원천 좌표 보존 배열 (c4 발바닥 지면고정용)
        metrics: 정규화 전후 뼈 길이 변동성 및 스트림 잔차 통계 딕셔너리
    """
    assert kps_3d.ndim == 3 and kps_3d.shape[1] >= 33 and kps_3d.shape[2] == 3, \
        f"Expected kps_3d shape (T, 33, 3), got {kps_3d.shape}"

    T = kps_3d.shape[0]
    stream_b = kps_3d.copy()
    stream_a = kps_3d.copy()

    # 1. 스쿼트 직립 안정 구간 (Standing Phase) 감지
    mid_hip = (kps_3d[:, 23, :] + kps_3d[:, 24, :]) / 2.0
    hip_y = mid_hip[:, 1]
    rom_y = float(np.max(hip_y) - np.min(hip_y))

    # 골반 높이가 가장 높은(y값이 작은) 직립 구간 상위 15% 프레임 추출
    standing_mask = (hip_y <= np.min(hip_y) + 0.15 * rom_y)
    if np.sum(standing_mask) < 3:
        standing_mask = np.zeros(T, dtype=bool)
        standing_mask[:max(3, T // 8)] = True

    # 2. 직립 구간 중앙값 기반 피험자 고유 뼈 세그먼트 기준 길이 (L_ref) 산출
    def get_ref_len(idx_a, idx_b):
        dists = np.linalg.norm(kps_3d[:, idx_b, :] - kps_3d[:, idx_a, :], axis=1)
        ref_val = float(np.median(dists[standing_mask]))
        return max(ref_val, 0.05)

    # 주요 뼈 세그먼트 고유 길이
    l_pelvis = get_ref_len(23, 24)
    l_femur_l = get_ref_len(23, 25)
    l_femur_r = get_ref_len(24, 26)
    l_tibia_l = get_ref_len(25, 27)
    l_tibia_r = get_ref_len(26, 28)
    l_heel_l  = get_ref_len(27, 29)
    l_heel_r  = get_ref_len(28, 30)
    l_toe_l   = get_ref_len(27, 31)
    l_toe_r   = get_ref_len(28, 32)

    mid_sh = (kps_3d[:, 11, :] + kps_3d[:, 12, :]) / 2.0
    d_torso = np.linalg.norm(mid_sh - mid_hip, axis=1)
    l_torso = float(np.median(d_torso[standing_mask]))
    l_torso = max(l_torso, 0.15)

    l_shoulder = get_ref_len(11, 12)
    l_neck     = float(np.median(np.linalg.norm(kps_3d[:, 0, :] - mid_sh, axis=1)[standing_mask]))
    l_neck     = max(l_neck, 0.05)

    # 3. 계층적 Forward Kinematics (FK) - 골반 중심(Root) 기준 관절 재배치
    root = mid_hip  # 골반 중심 이동 궤적은 100% 보존

    # (1) 골반 좌우 (23: left hip, 24: right hip)
    v_pelvis = kps_3d[:, 24, :] - kps_3d[:, 23, :]
    u_pelvis = v_pelvis / (np.linalg.norm(v_pelvis, axis=1, keepdims=True) + eps)
    stream_a[:, 23, :] = root - 0.5 * l_pelvis * u_pelvis
    stream_a[:, 24, :] = root + 0.5 * l_pelvis * u_pelvis

    # (2) 몸통(체간) 및 어깨 (11: left shoulder, 12: right shoulder)
    v_torso = mid_sh - root
    u_torso = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + eps)
    mid_sh_norm = root + l_torso * u_torso

    v_sh = kps_3d[:, 12, :] - kps_3d[:, 11, :]
    u_sh = v_sh / (np.linalg.norm(v_sh, axis=1, keepdims=True) + eps)
    stream_a[:, 11, :] = mid_sh_norm - 0.5 * l_shoulder * u_sh
    stream_a[:, 12, :] = mid_sh_norm + 0.5 * l_shoulder * u_sh

    # (3) 목 및 코 (0: nose)
    v_neck = kps_3d[:, 0, :] - mid_sh
    u_neck = v_neck / (np.linalg.norm(v_neck, axis=1, keepdims=True) + eps)
    stream_a[:, 0, :] = mid_sh_norm + l_neck * u_neck

    # (4) 좌측 하지: 대퇴골(23->25) -> 경골(25->27) -> 발(27->29, 27->31)
    v_fem_l = kps_3d[:, 25, :] - kps_3d[:, 23, :]
    u_fem_l = v_fem_l / (np.linalg.norm(v_fem_l, axis=1, keepdims=True) + eps)
    stream_a[:, 25, :] = stream_a[:, 23, :] + l_femur_l * u_fem_l

    v_tib_l = kps_3d[:, 27, :] - kps_3d[:, 25, :]
    u_tib_l = v_tib_l / (np.linalg.norm(v_tib_l, axis=1, keepdims=True) + eps)
    stream_a[:, 27, :] = stream_a[:, 25, :] + l_tibia_l * u_tib_l

    v_heel_l = kps_3d[:, 29, :] - kps_3d[:, 27, :]
    u_heel_l = v_heel_l / (np.linalg.norm(v_heel_l, axis=1, keepdims=True) + eps)
    stream_a[:, 29, :] = stream_a[:, 27, :] + l_heel_l * u_heel_l

    v_toe_l = kps_3d[:, 31, :] - kps_3d[:, 27, :]
    u_toe_l = v_toe_l / (np.linalg.norm(v_toe_l, axis=1, keepdims=True) + eps)
    stream_a[:, 31, :] = stream_a[:, 27, :] + l_toe_l * u_toe_l

    # (5) 우측 하지: 대퇴골(24->26) -> 경골(26->28) -> 발(28->30, 28->32)
    v_fem_r = kps_3d[:, 26, :] - kps_3d[:, 24, :]
    u_fem_r = v_fem_r / (np.linalg.norm(v_fem_r, axis=1, keepdims=True) + eps)
    stream_a[:, 26, :] = stream_a[:, 24, :] + l_femur_r * u_fem_r

    v_tib_r = kps_3d[:, 28, :] - kps_3d[:, 26, :]
    u_tib_r = v_tib_r / (np.linalg.norm(v_tib_r, axis=1, keepdims=True) + eps)
    stream_a[:, 28, :] = stream_a[:, 26, :] + l_tibia_r * u_tib_r

    v_heel_r = kps_3d[:, 30, :] - kps_3d[:, 28, :]
    u_heel_r = v_heel_r / (np.linalg.norm(v_heel_r, axis=1, keepdims=True) + eps)
    stream_a[:, 30, :] = stream_a[:, 28, :] + l_heel_r * u_heel_r

    v_toe_r = kps_3d[:, 32, :] - kps_3d[:, 28, :]
    u_toe_r = v_toe_r / (np.linalg.norm(v_toe_r, axis=1, keepdims=True) + eps)
    stream_a[:, 32, :] = stream_a[:, 28, :] + l_toe_r * u_toe_r

    # 4. 정규화 전후 진단 통계 산출
    raw_femur_std = float((
        np.std(np.linalg.norm(kps_3d[:, 23, :] - kps_3d[:, 25, :], axis=1)) +
        np.std(np.linalg.norm(kps_3d[:, 24, :] - kps_3d[:, 26, :], axis=1))
    ) / 2.0)

    raw_tibia_std = float((
        np.std(np.linalg.norm(kps_3d[:, 25, :] - kps_3d[:, 27, :], axis=1)) +
        np.std(np.linalg.norm(kps_3d[:, 26, :] - kps_3d[:, 28, :], axis=1))
    ) / 2.0)

    residual_norm = float(np.mean(np.linalg.norm(stream_a - stream_b, axis=2)))

    metrics = {
        'raw_femur_wobble_std': raw_femur_std,
        'raw_tibia_wobble_std': raw_tibia_std,
        'stream_residual_norm': residual_norm,
        'ref_femur_length': float((l_femur_l + l_femur_r) / 2.0),
        'ref_tibia_length': float((l_tibia_l + l_tibia_r) / 2.0),
        'ref_torso_length': l_torso
    }

    return stream_a, stream_b, metrics
