# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: kinematics_v6.py
Location: src/features/kinematics_v6.py
Purpose: 1단계 운동 분류 v6.0을 위한 듀얼 스트림(Format B v2.0 + Model F Direct KCF) 피처 생성 엔진
- Stream 1 (for 딥러닝 Bi-GRU): 국소 분절 정규화(Part-Based Normalization) 및 로컬 프레임이 적용된 Format B v2.0 시계열 텐서
- Stream 2 (for 트리 GBDT): 3D 관절 좌표에서 직접 추출되는 Model F Direct KCF v2.0 (59D) 정형 벡터
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.core.skeleton2d import Joint15, SKELETON_EDGES
from src.features.kinematics_3d import KinematicsExtractor3D
from src.features.kinematic_coordination import KinematicCoordinationExtractor

class KinematicsEngineV6:
    """v6.0 차세대 듀얼 스트림 생체역학 키네마틱스 엔진"""

    @staticmethod
    def reconstruct_3d_keypoints_from_raw_b(X_b: np.ndarray) -> np.ndarray:
        """
        Format B 텐서 (N, T, C>=56)의 Stream 1 (14 뼈대 dx, dy, dz)로부터 
        15개 3D 관절 좌표 (N, T, 15, 3)를 오차 없이 복원
        """
        is_single = (X_b.ndim == 2)
        if is_single:
            X_b = np.expand_dims(X_b, axis=0)

        N, T, _ = X_b.shape
        kps_3d = np.zeros((N, T, 15, 3), dtype=np.float32)
        kps_3d[:, :, Joint15.PELVIS, :] = 0.0

        for edge_idx, (src, dst) in enumerate(SKELETON_EDGES):
            ch = edge_idx * 4
            vec = X_b[:, :, ch:ch+3]
            kps_3d[:, :, dst, :] = kps_3d[:, :, src, :] + vec

        return kps_3d[0] if is_single else kps_3d

    @classmethod
    def apply_part_based_normalization(cls, kps_3d: np.ndarray) -> np.ndarray:
        """
        [신기술 1] 계층적 국소 분절 정규화 (Part-Based Hierarchical Normalization)
        체간, 두경부, 상지, 하지를 각각 고유한 앵커와 기준 길이로 정규화하여 
        상체 숙임에 따른 스케일 왜곡(Scale Modulation Artifact)을 원천 차단
        """
        is_single = (kps_3d.ndim == 3)
        if is_single:
            kps_3d = np.expand_dims(kps_3d, axis=0)

        N, T, _, _ = kps_3d.shape
        eps = 1e-6
        kps_norm = kps_3d.copy()

        # 1. 체간 스케일: Pelvis -> Neck
        l_torso = np.linalg.norm(kps_3d[:, :, Joint15.NECK] - kps_3d[:, :, Joint15.PELVIS], axis=-1, keepdims=True) + eps

        # 2. 상지 스케일: Shoulder -> Elbow (좌우 평균)
        l_arm_l = np.linalg.norm(kps_3d[:, :, Joint15.LEFT_ELBOW] - kps_3d[:, :, Joint15.LEFT_SHOULDER], axis=-1, keepdims=True)
        l_arm_r = np.linalg.norm(kps_3d[:, :, Joint15.RIGHT_ELBOW] - kps_3d[:, :, Joint15.RIGHT_SHOULDER], axis=-1, keepdims=True)
        l_arm = (l_arm_l + l_arm_r) / 2.0 + eps

        # 3. 하지 스케일: Knee -> Ankle (정강이 길이, 좌우 평균)
        l_shank_l = np.linalg.norm(kps_3d[:, :, Joint15.LEFT_ANKLE] - kps_3d[:, :, Joint15.LEFT_KNEE], axis=-1, keepdims=True)
        l_shank_r = np.linalg.norm(kps_3d[:, :, Joint15.RIGHT_ANKLE] - kps_3d[:, :, Joint15.RIGHT_KNEE], axis=-1, keepdims=True)
        l_shank = (l_shank_l + l_shank_r) / 2.0 + eps

        # 국소 분절별 스케일링 적용
        kps_norm[:, :, Joint15.NECK] = (kps_3d[:, :, Joint15.NECK] - kps_3d[:, :, Joint15.PELVIS]) / l_torso
        kps_norm[:, :, Joint15.LEFT_ELBOW] = kps_3d[:, :, Joint15.LEFT_ELBOW] / l_arm
        kps_norm[:, :, Joint15.RIGHT_ELBOW] = kps_3d[:, :, Joint15.RIGHT_ELBOW] / l_arm
        kps_norm[:, :, Joint15.LEFT_WRIST] = kps_3d[:, :, Joint15.LEFT_WRIST] / l_arm
        kps_norm[:, :, Joint15.RIGHT_WRIST] = kps_3d[:, :, Joint15.RIGHT_WRIST] / l_arm
        kps_norm[:, :, Joint15.LEFT_ANKLE] = kps_3d[:, :, Joint15.LEFT_ANKLE] / l_shank
        kps_norm[:, :, Joint15.RIGHT_ANKLE] = kps_3d[:, :, Joint15.RIGHT_ANKLE] / l_shank

        return kps_norm[0] if is_single else kps_norm

    @classmethod
    def build_format_b_v2(cls, kps_3d: np.ndarray) -> np.ndarray:
        """
        [Stream 1]: 딥러닝(Bi-GRU)용 차세대 Format B v2.0 시계열 텐서 생성 -> Shape: (N, 64, 96)
        - 국소 분절 정규화가 적용되어 상체 숙임 시에도 왜곡 없는 고순도 시계열
        - 발목/목 기준 로컬 키네마틱 프레임 채널 결합 (+4D)
        """
        is_single = (kps_3d.ndim == 3)
        if is_single:
            kps_3d = np.expand_dims(kps_3d, axis=0)

        N, T, _, _ = kps_3d.shape
        kps_norm = cls.apply_part_based_normalization(kps_3d)

        # 92채널 표준 포맷 빌드
        fb_base = np.zeros((N, T, 92), dtype=np.float32)
        for i in range(N):
            fb_base[i] = KinematicsExtractor3D.build_format_b_timeseries_3d(kps_norm[i])

        # 로컬 키네마틱 프레임 4개 시계열 채널 추가 (+4D)
        # 1) 발목 기준 무릎 상대 3D 변위 (dx, dy, dz)
        rel_knee_l = kps_3d[:, :, Joint15.LEFT_KNEE] - kps_3d[:, :, Joint15.LEFT_ANKLE]
        rel_knee_r = kps_3d[:, :, Joint15.RIGHT_KNEE] - kps_3d[:, :, Joint15.RIGHT_ANKLE]
        local_dx = ((rel_knee_l[:, :, 0] + rel_knee_r[:, :, 0]) / 2.0)[:, :, None]
        local_dy = ((rel_knee_l[:, :, 1] + rel_knee_r[:, :, 1]) / 2.0)[:, :, None]
        local_dz = ((rel_knee_l[:, :, 2] + rel_knee_r[:, :, 2]) / 2.0)[:, :, None]

        # 2) 목 기준 어깨 비틀림 상대 Z 변위
        rel_sh_twist = (kps_3d[:, :, Joint15.RIGHT_SHOULDER, 2] - kps_3d[:, :, Joint15.LEFT_SHOULDER, 2])[:, :, None]

        fb_v2 = np.concatenate([fb_base, local_dx, local_dy, local_dz, rel_sh_twist], axis=-1) # (N, T, 96)

        return fb_v2[0] if is_single else fb_v2

    @classmethod
    def build_model_f_direct_kcf(cls, kps_3d: np.ndarray) -> np.ndarray:
        """
        [Stream 2]: 트리 모델(LightGBM + XGBoost)용 Model F Direct KCF v2.0 정형 피처 (59D)
        - 국소 분절 정규화 기반 55D KCF
        - Kinematic Tree 로컬 골격 프레임 4종 결합 = 총 59D
        """
        is_single = (kps_3d.ndim == 3)
        if is_single:
            kps_3d = np.expand_dims(kps_3d, axis=0)

        N, T, _, _ = kps_3d.shape
        kps_norm = cls.apply_part_based_normalization(kps_3d)

        # 1. 국소 정규화 55D KCF 산출
        streams_55 = np.zeros((N, T, 92), dtype=np.float32)
        for i in range(N):
            streams_55[i] = KinematicsExtractor3D.build_format_b_timeseries_3d(kps_norm[i])
        kcf_55 = KinematicCoordinationExtractor.extract_features(streams_55)

        # 2. 로컬 키네마틱 프레임 가동범위 피처 4종 산출
        rel_knee_l = kps_3d[:, :, Joint15.LEFT_KNEE] - kps_3d[:, :, Joint15.LEFT_ANKLE]
        rel_knee_r = kps_3d[:, :, Joint15.RIGHT_KNEE] - kps_3d[:, :, Joint15.RIGHT_ANKLE]
        local_shank_dx_rom = np.ptp(rel_knee_l[:, :, 0], axis=1) + np.ptp(rel_knee_r[:, :, 0], axis=1)
        local_shank_dy_rom = np.ptp(rel_knee_l[:, :, 1], axis=1) + np.ptp(rel_knee_r[:, :, 1], axis=1)
        local_shank_dz_rom = np.ptp(rel_knee_l[:, :, 2], axis=1) + np.ptp(rel_knee_r[:, :, 2], axis=1)

        rel_sh_l = kps_3d[:, :, Joint15.LEFT_SHOULDER] - kps_3d[:, :, Joint15.NECK]
        rel_sh_r = kps_3d[:, :, Joint15.RIGHT_SHOULDER] - kps_3d[:, :, Joint15.NECK]
        local_sh_twist_rom = np.ptp(rel_sh_r[:, :, 2] - rel_sh_l[:, :, 2], axis=1)

        local_feats_4d = np.stack([
            local_shank_dx_rom, local_shank_dy_rom, local_shank_dz_rom, local_sh_twist_rom
        ], axis=1).astype(np.float32)

        kcf_59d = np.concatenate([kcf_55, local_feats_4d], axis=-1)

        return kcf_59d[0] if is_single else kcf_59d
