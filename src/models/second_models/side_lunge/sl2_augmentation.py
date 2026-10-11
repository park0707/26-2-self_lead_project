# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: sl2_augmentation.py
Purpose: 사이드 런지 3D 포즈 텐서(64, 33, 3) 대상 생체역학 정합 데이터 증강 엔진
         1. 시간축 기반 분열 (Temporal Warping / Phase Resampling)
         2. 정방향 기구학 기반 뼈 길이 수정 (Kinematic Forward Skeletal Scaling)
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, List, Optional


def temporal_warp_pose_sequence(
    kps_64f: np.ndarray,
    speed_factor: float = 1.0,
    phase_shift: float = 0.0
) -> np.ndarray:
    """
    (64, 33, 3) 3D 포즈 시퀀스를 시간축 상에서 비선형 또는 선형 워핑하여 템포 변형
    speed_factor: 1.0보다 크면 빠름, 작으면 느림 (0.85 ~ 1.15)
    phase_shift: 위상 이동 (-0.05 ~ +0.05)
    """
    T, V, C = kps_64f.shape
    t_orig = np.linspace(0.0, 1.0, T)

    # 비선형 시간 왜곡 곡선 생성
    # speed_factor와 phase_shift를 반영한 monotonic time map
    t_new = t_orig ** speed_factor + phase_shift
    t_new = (t_new - t_new[0]) / (t_new[-1] - t_new[0] + 1e-9)
    t_new = np.clip(t_new, 0.0, 1.0)

    # 3차 스플라인 / 선형 보간으로 64프레임 복원
    f_interp = interp1d(t_orig, kps_64f, axis=0, kind='linear', fill_value="extrapolate")
    warped_kps = f_interp(t_new).astype(np.float32)
    return warped_kps


def kinematic_bone_scaling(
    kps_64f: np.ndarray,
    scale_femur: float = 1.0,
    scale_tibia: float = 1.0,
    scale_foot: float = 1.0,
    scale_torso: float = 1.0
) -> np.ndarray:
    """
    (64, 33, 3) 텐서에 대해 정방향 기구학(Forward Kinematics) 체인을 유지하며 뼈 길이 스케일링
    골반 중심을 루트(Root)로 하여 체간, 양 대퇴골, 양 정강이, 양 족부의 길이를 자연스럽게 변형
    """
    T, V, C = kps_64f.shape
    out = kps_64f.copy()

    for t in range(T):
        # 1. 체간 (Torso) 스케일링
        # 루트: 양 골반 중심
        pelvis_mid = 0.5 * (out[t, 23] + out[t, 24])
        for shoulder_idx in [11, 12]:
            v_spine = out[t, shoulder_idx] - pelvis_mid
            out[t, shoulder_idx] = pelvis_mid + v_spine * scale_torso

        # 어깨 이동에 따른 상지(팔, 머리) 전체 동반 평행 이동
        d_shoulder = 0.5 * (out[t, 11] + out[t, 12]) - 0.5 * (kps_64f[t, 11] + kps_64f[t, 12])
        upper_body_indices = list(range(0, 11)) + [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        out[t, upper_body_indices] += d_shoulder

        # 2. 하지 (Lower Extremity) 정방향 기구학 스케일링 (좌/우)
        for hip_idx, knee_idx, ank_idx, heel_idx, toe_idx in [
            (23, 25, 27, 29, 31), # Left leg
            (24, 26, 28, 30, 32)  # Right leg
        ]:
            # (A) 대퇴골 (Femur): Hip -> Knee
            v_femur = out[t, knee_idx] - out[t, hip_idx]
            new_knee = out[t, hip_idx] + v_femur * scale_femur
            d_knee = new_knee - out[t, knee_idx]
            out[t, knee_idx] = new_knee

            # 하위 관절 평행 이동
            out[t, [ank_idx, heel_idx, toe_idx]] += d_knee

            # (B) 정강이 (Tibia): Knee -> Ankle
            v_tibia = out[t, ank_idx] - out[t, knee_idx]
            new_ank = out[t, knee_idx] + v_tibia * scale_tibia
            d_ank = new_ank - out[t, ank_idx]
            out[t, ank_idx] = new_ank

            # 하위 족부 평행 이동
            out[t, [heel_idx, toe_idx]] += d_ank

            # (C) 족부 (Foot): Ankle -> Heel, Ankle -> Toe
            v_heel = out[t, heel_idx] - out[t, ank_idx]
            v_toe = out[t, toe_idx] - out[t, ank_idx]
            out[t, heel_idx] = out[t, ank_idx] + v_heel * scale_foot
            out[t, toe_idx] = out[t, ank_idx] + v_toe * scale_foot

    return out.astype(np.float32)


def rodrigues_rot(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    """
    v: (T, 3) 벡터 배열
    k: (T, 3) 정규화된 회전축 배열
    theta: 라디안 각도
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    k_cross_v = np.cross(k, v)
    k_dot_v = np.sum(k * v, axis=1, keepdims=True)
    return v * cos_theta + k_cross_v * sin_theta + k * k_dot_v * (1 - cos_theta)


def apply_angular_perturbation(
    kps_64f: np.ndarray,
    knee_valgus_deg: float = 0.0,
    foot_yaw_deg: float = 0.0,
    target_side: str = 'both'
) -> np.ndarray:
    """
    Knee Valgus: Hip-Ankle 벡터를 축으로 무릎 관절 회전 (내반/외반 유도)
    Foot Yaw: Knee-Ankle 벡터를 축으로 발(Heel, Toe) 회전 (발 비틀림 유도)
    """
    out = kps_64f.copy()
    
    sides = []
    if target_side in ['left', 'both']:
        sides.append((23, 25, 27, 29, 31)) # Left: Hip, Knee, Ankle, Heel, Toe
    if target_side in ['right', 'both']:
        sides.append((24, 26, 28, 30, 32)) # Right
        
    knee_valgus_rad = np.radians(knee_valgus_deg)
    foot_yaw_rad = np.radians(foot_yaw_deg)
    
    for hip_idx, knee_idx, ank_idx, heel_idx, toe_idx in sides:
        # 1. Knee Valgus Perturbation
        if knee_valgus_deg != 0.0:
            hip = out[:, hip_idx]
            ank = out[:, ank_idx]
            knee = out[:, knee_idx]
            
            # 회전축: Hip -> Ankle
            axis = ank - hip
            axis_norm = np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
            axis = axis / axis_norm
            
            # 회전할 벡터: Hip -> Knee
            v_knee = knee - hip
            
            # Rodrigues rotation
            v_knee_rot = rodrigues_rot(v_knee, axis, knee_valgus_rad)
            out[:, knee_idx] = hip + v_knee_rot
            
        # 2. Foot Yaw (Progression Angle) Perturbation
        if foot_yaw_deg != 0.0:
            knee = out[:, knee_idx]
            ank = out[:, ank_idx]
            heel = out[:, heel_idx]
            toe = out[:, toe_idx]
            
            # 회전축: Knee -> Ankle (Shank)
            axis = ank - knee
            axis_norm = np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
            axis = axis / axis_norm
            
            # 회전할 벡터: Ankle -> Heel, Ankle -> Toe
            v_heel = heel - ank
            v_toe = toe - ank
            
            v_heel_rot = rodrigues_rot(v_heel, axis, foot_yaw_rad)
            v_toe_rot = rodrigues_rot(v_toe, axis, foot_yaw_rad)
            
            out[:, heel_idx] = ank + v_heel_rot
            out[:, toe_idx] = ank + v_toe_rot
            
    return out


def generate_augmented_defect_variants(
    kps_64f: np.ndarray,
    n_variants: int = 4
) -> List[np.ndarray]:
    """
    단일 결함 시퀀스로부터 시간축 분열 + 뼈 길이 수정 + 각도 교란(Angular Perturbation)을 결합한 
    고품질 증강 시퀀스 리스트 생성.
    각도 교란을 통해 모델이 발 방향과 무릎 외반에 더 민감하게 반응하도록 유도.
    """
    variants = []
    configs = [
        # Variant 1: 템포 느림, 약간의 무릎 외반(+5도), 발 끝 외회전(+8도)
        {"speed": 0.90, "femur": 1.06, "tibia": 0.96, "foot": 1.02, "torso": 1.03, "knee_valgus": 5.0, "foot_yaw": 8.0},
        # Variant 2: 템포 빠름, 약간의 무릎 내반(-4도), 발 끝 내회전(-6도)
        {"speed": 1.10, "femur": 0.95, "tibia": 1.06, "foot": 0.98, "torso": 0.97, "knee_valgus": -4.0, "foot_yaw": -6.0},
        # Variant 3: 균형 확장, 극단적 무릎 외반(+10도)
        {"speed": 1.02, "femur": 1.05, "tibia": 1.05, "foot": 1.04, "torso": 1.05, "knee_valgus": 10.0, "foot_yaw": 2.0},
        # Variant 4: 골격 축소, 극단적 발 끝 회전(+15도)
        {"speed": 0.95, "femur": 0.90, "tibia": 0.95, "foot": 0.95, "torso": 0.95, "knee_valgus": 2.0, "foot_yaw": 15.0},
        # Variant 5: 무릎 내반(-10도) 극단
        {"speed": 1.05, "femur": 1.0, "tibia": 1.0, "foot": 1.0, "torso": 1.0, "knee_valgus": -10.0, "foot_yaw": 0.0},
    ]

    for cfg in configs[:n_variants]:
        # 1. 시간축 워핑
        warped = temporal_warp_pose_sequence(kps_64f, speed_factor=cfg["speed"])
        # 2. 정방향 기구학 뼈 길이 스케일링
        scaled = kinematic_bone_scaling(
            warped,
            scale_femur=cfg["femur"],
            scale_tibia=cfg["tibia"],
            scale_foot=cfg["foot"],
            scale_torso=cfg["torso"]
        )
        # 3. 각도 교란 (Angular Perturbation)
        augmented = apply_angular_perturbation(
            scaled,
            knee_valgus_deg=cfg["knee_valgus"],
            foot_yaw_deg=cfg["foot_yaw"],
            target_side='both'
        )
        variants.append(augmented)

    return variants
