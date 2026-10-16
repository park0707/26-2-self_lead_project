# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/core/topology_adapter.py
Purpose: 관절 토폴로지 상호 변환기 (MediaPipe 33J <-> RTMPose 26J <-> Format B 15J)
         및 64프레임 3차 스플라인(Cubic Spline) 보간 표준기
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional
from src.core.skeleton2d import Joint15


class SkeletalTopologyAdapter:
    """비전 모델 관절 좌표계 간 상호 변환 및 64F 스플라인 정규화 어댑터"""

    # MediaPipe 33 -> 15 표준 매핑 인덱스
    # Pelvis: 23(L_HIP)과 24(R_HIP)의 중점
    # Neck: 11(L_SHOULDER)과 12(R_SHOULDER)의 중점
    MP_JOINTS_MAP = {
        Joint15.LEFT_SHOULDER: 11,
        Joint15.RIGHT_SHOULDER: 12,
        Joint15.LEFT_ELBOW: 13,
        Joint15.RIGHT_ELBOW: 14,
        Joint15.LEFT_WRIST: 15,
        Joint15.RIGHT_WRIST: 16,
        Joint15.LEFT_HIP: 23,
        Joint15.RIGHT_HIP: 24,
        Joint15.LEFT_KNEE: 25,
        Joint15.RIGHT_KNEE: 26,
        Joint15.LEFT_ANKLE: 27,
        Joint15.RIGHT_ANKLE: 28,
    }

    # RTMPose 26 -> 15 표준 매핑 인덱스
    # RTMPose-26 Keypoints:
    # 0: nose, 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist,
    # 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle
    RTM_JOINTS_MAP = {
        Joint15.LEFT_SHOULDER: 5,
        Joint15.RIGHT_SHOULDER: 6,
        Joint15.LEFT_ELBOW: 7,
        Joint15.RIGHT_ELBOW: 8,
        Joint15.LEFT_WRIST: 9,
        Joint15.RIGHT_WRIST: 10,
        Joint15.LEFT_HIP: 11,
        Joint15.RIGHT_HIP: 12,
        Joint15.LEFT_KNEE: 13,
        Joint15.RIGHT_KNEE: 14,
        Joint15.LEFT_ANKLE: 15,
        Joint15.RIGHT_ANKLE: 16,
    }

    @classmethod
    def mediapipe_33_to_15(cls, kps_33: np.ndarray) -> np.ndarray:
        """(T, 33, C) MediaPipe 3D 관절 좌표를 표준 (T, 15, C)로 변환"""
        kps = np.asarray(kps_33, dtype=np.float32)
        T, J, C = kps.shape
        out = np.zeros((T, 15, C), dtype=np.float32)

        # 골반, 경추 및 체간 중점 계산
        pelvis = (kps[:, 23, :C] + kps[:, 24, :C]) / 2.0
        neck = (kps[:, 11, :C] + kps[:, 12, :C]) / 2.0
        torso = (pelvis + neck) / 2.0
        out[:, Joint15.PELVIS, :C] = pelvis
        out[:, Joint15.NECK, :C] = neck
        out[:, Joint15.TORSO, :C] = torso

        for dst_j, src_j in cls.MP_JOINTS_MAP.items():
            out[:, dst_j, :C] = kps[:, src_j, :C]

        return out

    mediapipe_33_to_joint15 = mediapipe_33_to_15


    @classmethod
    def rtmpose_26_to_15(cls, kps_26: np.ndarray) -> np.ndarray:
        """(T, 26, C) RTMPose 26 관절 좌표를 표준 (T, 15, C)로 변환"""
        kps = np.asarray(kps_26, dtype=np.float32)
        T, J, C = kps.shape
        out = np.zeros((T, 15, C), dtype=np.float32)

        pelvis = (kps[:, 11, :C] + kps[:, 12, :C]) / 2.0
        neck = (kps[:, 5, :C] + kps[:, 6, :C]) / 2.0
        torso = (pelvis + neck) / 2.0
        out[:, Joint15.PELVIS, :C] = pelvis
        out[:, Joint15.NECK, :C] = neck
        out[:, Joint15.TORSO, :C] = torso

        for dst_j, src_j in cls.RTM_JOINTS_MAP.items():
            out[:, dst_j, :C] = kps[:, src_j, :C]

        return out

    # MediaPipe 33 <-> RTMPose (Halpe 26) 상호 매핑 테이블
    HALPE26_TO_MP33_MAP = {
        0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
        5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
        20: 31, 21: 32, 24: 29, 25: 30
    }
    MP33_TO_HALPE26_MAP = {
        0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
        11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
        23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
        31: 20, 32: 21, 29: 24, 30: 25
    }

    @classmethod
    def rtmpose_26_to_mediapipe_33(cls, kps_26: np.ndarray) -> np.ndarray:
        """(T, 26, C) RTMPose 텐서를 MediaPipe (T, 33, C) 텐서로 변환"""
        kps = np.asarray(kps_26, dtype=np.float32)
        T, J, C = kps.shape
        out = np.zeros((T, 33, C), dtype=np.float32)

        for h_idx, m_idx in cls.HALPE26_TO_MP33_MAP.items():
            if h_idx < J:
                out[:, m_idx, :C] = kps[:, h_idx, :C]

        # 발끝 정밀 보정 (Halpe 20: Big Toe + Halpe 22: Small Toe 중점 -> MP 31: Left Foot Index)
        if J >= 24:
            out[:, 31, :C] = (kps[:, 20, :C] + kps[:, 22, :C]) / 2.0
            out[:, 32, :C] = (kps[:, 21, :C] + kps[:, 23, :C]) / 2.0

        # 파생 관절 추정
        # 코(0) -> 입(9, 10) 추정
        out[:, 9, :C] = out[:, 0, :C] * 0.7 + (out[:, 11, :C] + out[:, 12, :C]) * 0.15
        out[:, 10, :C] = out[:, 0, :C] * 0.7 + (out[:, 11, :C] + out[:, 12, :C]) * 0.15

        # 2D 좌표 기반인 경우 Z 채널을 0.0으로 보장하여 confidence 값이 가상 깊이로 오인되는 현상 방지
        if C >= 3:
            out[:, :, 2] = 0.0

        return out

    @classmethod
    def mediapipe_33_to_rtmpose_26(cls, kps_33: np.ndarray) -> np.ndarray:
        """(T, 33, C) MediaPipe 텐서를 RTMPose Halpe (T, 26, C) 텐서로 변환"""
        kps = np.asarray(kps_33, dtype=np.float32)
        T, J, C = kps.shape
        out = np.zeros((T, 26, C), dtype=np.float32)

        for m_idx, h_idx in cls.MP33_TO_HALPE26_MAP.items():
            if m_idx < J:
                out[:, h_idx, :C] = kps[:, m_idx, :C]

        # Halpe-26 Joint Topology 정합 (src.core.skeleton2d.JointHalpe26 표준 준수)
        # 17: Head (머리/얼굴 중심: 귀 중점 또는 코)
        out[:, 17, :C] = (kps[:, 7, :C] + kps[:, 8, :C]) / 2.0 if J > 8 else kps[:, 0, :C]
        # 18: Neck (목/상부체간: 양 어깨 중점)
        out[:, 18, :C] = (kps[:, 11, :C] + kps[:, 12, :C]) / 2.0
        # 19: Hip (골반 중심/Root: 양 고관절 중점)
        out[:, 19, :C] = (kps[:, 23, :C] + kps[:, 24, :C]) / 2.0
        return out

    @staticmethod
    def resample_to_64_frames(seq: np.ndarray) -> np.ndarray:
        """
        임의 길이 시퀀스 (T, J, C) 또는 (T, D)를 64프레임으로 3차 스플라인(Cubic Spline) 보간
        (AGENTS.md 제2조 표준 규정 준수 및 T<=1 엣지 케이스 완벽 방어)
        """
        seq = np.asarray(seq, dtype=np.float32)
        T = len(seq)
        if T == 64:
            return seq
        if T <= 0:
            target_shape = (64,) + seq.shape[1:]
            return np.zeros(target_shape, dtype=np.float32)
        if T == 1:
            target_shape = (64,) + seq.shape[1:]
            return np.repeat(seq, 64, axis=0).astype(np.float32)

        orig_shape = seq.shape
        flat_seq = seq.reshape(T, -1)  # (T, J*C)

        t_orig = np.linspace(0.0, 1.0, T)
        t_target = np.linspace(0.0, 1.0, 64)

        if T < 4:
            from scipy.interpolate import interp1d
            f = interp1d(t_orig, flat_seq, kind='linear', axis=0, fill_value='extrapolate')
            resampled_flat = f(t_target).astype(np.float32)
        else:
            cs = CubicSpline(t_orig, flat_seq, axis=0)
            resampled_flat = cs(t_target).astype(np.float32)

        target_shape = (64,) + orig_shape[1:]
        return resampled_flat.reshape(target_shape)

    resample_to_64f_spline = resample_to_64_frames


    @staticmethod
    def center_on_pelvis(kps_15: np.ndarray) -> np.ndarray:
        """(T, 15, 3) 관절을 골반(Pelvis) 원점 기준으로 정규화"""
        pelvis = kps_15[:, Joint15.PELVIS:Joint15.PELVIS+1, :]
        return kps_15 - pelvis
