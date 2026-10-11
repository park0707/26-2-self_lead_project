# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src.preprocessing.rtmpose_preprocessor
Purpose: RTMPose-m (Halpe26) 관절 전용 최적화 전처리 파이프라인
         - 신뢰도 게이팅 기반 3차 스플라인 결측치 보간
         - Halpe26 Neck(18) / Pelvis(19) 다이렉트 중심화 및 체간 척추 스케일 정규화
         - 3각 발바닥 접지면(Foot Triangle: 엄지/새끼/뒤꿈치) 기하 정규화
         - 26관절 편측 대칭 미러링 (런지류)
         - 표준 규정 준수 고정 64프레임 3차 스플라인 리샘플링
         - 1단계 모델 하위 호환 어댑터 (Halpe26 -> Joint15) 내장
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from typing import Dict, Any, Optional, Tuple, List
import math

from src.core.skeleton2d import JointHalpe26, SYMMETRIC_PAIRS_HALPE26, Joint15


class RTMPosePreprocessor:
    """RTMPose-m (Halpe26) 전용 7단계 고도화 전처리 엔진"""

    CONFIDENCE_THRESHOLD = 0.25

    @classmethod
    def filter_and_impute_confidence(
        cls,
        kps_seq: np.ndarray,
        conf_thresh: float = CONFIDENCE_THRESHOLD,
        min_valid_ratio: float = 0.3
    ) -> Optional[np.ndarray]:
        """
        kps_seq: (T, 26, 3) [x, y, conf]
        신뢰도가 conf_thresh 미만이거나 NaN/(0,0)인 프레임을 탐지하여 3차 스플라인으로 시간축 결측 보간
        """
        T, J, C = kps_seq.shape
        if T < 2:
            return None

        clean_seq = np.copy(kps_seq).astype(np.float32)
        xy = clean_seq[:, :, :2]
        conf = clean_seq[:, :, 2] if C >= 3 else np.ones((T, J), dtype=np.float32)

        # 결측치 플래그: NaN or (0,0) or Confidence < conf_thresh
        is_nan = np.isnan(xy).any(axis=-1)
        is_zero = (np.abs(xy[:, :, 0]) < 1e-6) & (np.abs(xy[:, :, 1]) < 1e-6)
        is_low_conf = conf < conf_thresh
        is_invalid = is_nan | is_zero | is_low_conf

        # 전체 시퀀스 유효 비율 검사
        valid_ratio = 1.0 - (float(np.sum(is_invalid)) / (T * J))
        if valid_ratio < min_valid_ratio:
            return None

        t_indices = np.arange(T)
        for j in range(J):
            valid_mask = ~is_invalid[:, j]
            valid_count = int(np.sum(valid_mask))

            if valid_count < 2:
                # 유효 프레임이 거의 없으면 Pelvis(19) 또는 첫 프레임으로 대체
                if j != JointHalpe26.HIP:
                    clean_seq[:, j, :2] = clean_seq[:, JointHalpe26.HIP, :2]
                    clean_seq[:, j, 2] = 0.1
                continue

            t_valid = t_indices[valid_mask]
            for c_idx in range(2): # x, y 보간
                vals_valid = clean_seq[valid_mask, j, c_idx]
                if valid_count >= 4:
                    try:
                        cs = CubicSpline(t_valid, vals_valid, bc_type='natural')
                        clean_seq[:, j, c_idx] = cs(t_indices)
                    except Exception:
                        f = interp1d(t_valid, vals_valid, kind='linear', bounds_error=False, fill_value='extrapolate')
                        clean_seq[:, j, c_idx] = f(t_indices)
                else:
                    f = interp1d(t_valid, vals_valid, kind='linear', bounds_error=False, fill_value='extrapolate')
                    clean_seq[:, j, c_idx] = f(t_indices)

            # 보간된 결측 프레임의 신뢰도는 부드럽게 조정
            clean_seq[is_invalid[:, j], j, 2] = np.maximum(0.2, conf[is_invalid[:, j], j])

        return clean_seq

    @classmethod
    def root_center_pelvis(cls, kps_seq: np.ndarray) -> np.ndarray:
        """
        kps_seq: (T, 26, C)
        Halpe26의 HIP(19번 골반 중심)을 매 프레임 원점 (0, 0)으로 평행 이동
        """
        out = np.copy(kps_seq)
        pelvis = out[:, JointHalpe26.HIP:JointHalpe26.HIP+1, :2] # (T, 1, 2)
        out[:, :, :2] -= pelvis
        return out

    @classmethod
    def in_plane_rotate(cls, kps_seq: np.ndarray, num_init_frames: int = 5) -> Tuple[np.ndarray, float]:
        """
        초기 프레임의 어깨선(5, 6) 및 골반선(11, 12)을 기반으로 수평 회전 정렬 (카메라 롤 보정)
        """
        T = kps_seq.shape[0]
        n_frames = min(T, max(1, num_init_frames))

        sh_vec = np.mean(
            kps_seq[:n_frames, JointHalpe26.RIGHT_SHOULDER, :2] - kps_seq[:n_frames, JointHalpe26.LEFT_SHOULDER, :2],
            axis=0
        )

        if np.linalg.norm(sh_vec) < 1e-4:
            sh_vec = np.mean(
                kps_seq[:n_frames, JointHalpe26.RIGHT_HIP, :2] - kps_seq[:n_frames, JointHalpe26.LEFT_HIP, :2],
                axis=0
            )

        norm = np.linalg.norm(sh_vec)
        if norm < 1e-4:
            return kps_seq, 0.0

        theta = float(np.arctan2(sh_vec[1], sh_vec[0]))
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

        out = np.copy(kps_seq)
        out[:, :, :2] = np.matmul(out[:, :, :2], R.T)
        return out, theta

    @classmethod
    def scale_normalize_torso(cls, kps_seq: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Halpe26의 Neck(18) - Hip(19) 유클리드 척추 거리를 1.0으로 스케일 정규화
        """
        neck_hip_vec = kps_seq[:, JointHalpe26.NECK, :2] - kps_seq[:, JointHalpe26.HIP, :2]
        torso_len = float(np.mean(np.linalg.norm(neck_hip_vec, axis=-1)))

        if torso_len < 1e-4:
            sh_mid = (kps_seq[:, JointHalpe26.LEFT_SHOULDER, :2] + kps_seq[:, JointHalpe26.RIGHT_SHOULDER, :2]) / 2.0
            hip_mid = (kps_seq[:, JointHalpe26.LEFT_HIP, :2] + kps_seq[:, JointHalpe26.RIGHT_HIP, :2]) / 2.0
            torso_len = float(np.mean(np.linalg.norm(sh_mid - hip_mid, axis=-1)))

        if torso_len < 1e-4:
            torso_len = 1.0

        out = np.copy(kps_seq)
        out[:, :, :2] = out[:, :, :2] / (torso_len + 1e-7)
        return out, torso_len

    @classmethod
    def extract_foot_triangle_base(cls, kps_seq: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Halpe26 발 6개 관절(엄지 20/21, 새끼 22/23, 뒤꿈치 24/25)로부터
        좌/우 발바닥 종축 벡터(Heel -> MidToe) 및 발 너비 시계열 산출
        """
        xy = kps_seq[:, :, :2]
        l_big = xy[:, JointHalpe26.LEFT_BIG_TOE]
        l_small = xy[:, JointHalpe26.LEFT_SMALL_TOE]
        l_heel = xy[:, JointHalpe26.LEFT_HEEL]

        r_big = xy[:, JointHalpe26.RIGHT_BIG_TOE]
        r_small = xy[:, JointHalpe26.RIGHT_SMALL_TOE]
        r_heel = xy[:, JointHalpe26.RIGHT_HEEL]

        l_mid_toe = (l_big + l_small) / 2.0
        r_mid_toe = (r_big + r_small) / 2.0

        l_foot_vec = l_mid_toe - l_heel # (T, 2)
        r_foot_vec = r_mid_toe - r_heel

        l_foot_width = np.linalg.norm(l_big - l_small, axis=-1) # (T,)
        r_foot_width = np.linalg.norm(r_big - r_small, axis=-1)

        return {
            'l_foot_axis': l_foot_vec,
            'r_foot_axis': r_foot_vec,
            'l_foot_width': l_foot_width,
            'r_foot_width': r_foot_width,
            'l_mid_toe': l_mid_toe,
            'r_mid_toe': r_mid_toe
        }

    @classmethod
    def lateral_canonicalize(cls, kps_seq: np.ndarray, exercise_name: str) -> Tuple[np.ndarray, bool]:
        """
        런지 등 편측 운동 시 앞선 발을 일관되게 우측 다리로 미러링 정렬
        """
        is_flipped = False
        out = np.copy(kps_seq)

        if any(keyword in exercise_name for keyword in ['런지', 'lunge', 'Lunge']):
            l_ankle_disp = np.max(np.abs(out[:, JointHalpe26.LEFT_ANKLE, 0]))
            r_ankle_disp = np.max(np.abs(out[:, JointHalpe26.RIGHT_ANKLE, 0]))

            if l_ankle_disp > r_ankle_disp:
                # X축 반전
                out[:, :, 0] = -out[:, :, 0]
                # Halpe26 대칭 쌍 스왑
                for left_idx, right_idx in SYMMETRIC_PAIRS_HALPE26:
                    tmp = np.copy(out[:, left_idx, :])
                    out[:, left_idx, :] = out[:, right_idx, :]
                    out[:, right_idx, :] = tmp
                is_flipped = True

        return out, is_flipped

    @classmethod
    def resample_to_64f(cls, kps_seq: np.ndarray, target_frames: int = 64) -> np.ndarray:
        """
        표준 규정(64프레임 3차 스플라인) 시간축 리샘플링
        """
        T, J, C = kps_seq.shape
        if T == target_frames:
            return kps_seq.astype(np.float32)

        t_orig = np.linspace(0.0, 1.0, T)
        t_target = np.linspace(0.0, 1.0, target_frames)

        res = np.zeros((target_frames, J, C), dtype=np.float32)
        for j in range(J):
            for c in range(C):
                vals = kps_seq[:, j, c]
                if T >= 4:
                    try:
                        cs = CubicSpline(t_orig, vals, bc_type='natural')
                        res[:, j, c] = cs(t_target)
                    except Exception:
                        f = interp1d(t_orig, vals, kind='linear', bounds_error=False, fill_value='extrapolate')
                        res[:, j, c] = f(t_target)
                else:
                    f = interp1d(t_orig, vals, kind='linear', bounds_error=False, fill_value='extrapolate')
                    res[:, j, c] = f(t_target)

        return res

    @classmethod
    def halpe26_to_joint15(cls, kps_halpe: np.ndarray) -> np.ndarray:
        """
        Halpe26 (T, 26, C) 텐서를 기존 1단계 모델용 Joint15 (T, 15, C)로 변환하는 하위 호환 어댑터
        """
        T, _, C = kps_halpe.shape
        out15 = np.zeros((T, 15, C), dtype=np.float32)

        # 0: PELVIS -> 19 HIP
        out15[:, Joint15.PELVIS] = kps_halpe[:, JointHalpe26.HIP]
        # 1: TORSO -> (18 NECK + 19 HIP) / 2
        out15[:, Joint15.TORSO] = (kps_halpe[:, JointHalpe26.NECK] + kps_halpe[:, JointHalpe26.HIP]) / 2.0
        # 2, 3: SHOULDERS
        out15[:, Joint15.LEFT_SHOULDER] = kps_halpe[:, JointHalpe26.LEFT_SHOULDER]
        out15[:, Joint15.RIGHT_SHOULDER] = kps_halpe[:, JointHalpe26.RIGHT_SHOULDER]
        # 4, 5: ELBOWS
        out15[:, Joint15.LEFT_ELBOW] = kps_halpe[:, JointHalpe26.LEFT_ELBOW]
        out15[:, Joint15.RIGHT_ELBOW] = kps_halpe[:, JointHalpe26.RIGHT_ELBOW]
        # 6, 7: WRISTS
        out15[:, Joint15.LEFT_WRIST] = kps_halpe[:, JointHalpe26.LEFT_WRIST]
        out15[:, Joint15.RIGHT_WRIST] = kps_halpe[:, JointHalpe26.RIGHT_WRIST]
        # 8, 9: HIPS
        out15[:, Joint15.LEFT_HIP] = kps_halpe[:, JointHalpe26.LEFT_HIP]
        out15[:, Joint15.RIGHT_HIP] = kps_halpe[:, JointHalpe26.RIGHT_HIP]
        # 10, 11: KNEES
        out15[:, Joint15.LEFT_KNEE] = kps_halpe[:, JointHalpe26.LEFT_KNEE]
        out15[:, Joint15.RIGHT_KNEE] = kps_halpe[:, JointHalpe26.RIGHT_KNEE]
        # 12, 13: ANKLES
        out15[:, Joint15.LEFT_ANKLE] = kps_halpe[:, JointHalpe26.LEFT_ANKLE]
        out15[:, Joint15.RIGHT_ANKLE] = kps_halpe[:, JointHalpe26.RIGHT_ANKLE]
        # 14: NECK -> 18 NECK
        out15[:, Joint15.NECK] = kps_halpe[:, JointHalpe26.NECK]

        return out15

    @classmethod
    def process_raw_sequence(
        cls,
        raw_kps: np.ndarray,
        exercise_name: str = "",
        target_frames: int = 64
    ) -> Optional[Dict[str, Any]]:
        """
        RTMPose-m (Halpe26) 관절 시퀀스 전체 7단계 전처리 일괄 실행
        raw_kps: (T, 26, 3) [x, y, conf]
        """
        # 1. 신뢰도 기반 결측 보간
        clean_kps = cls.filter_and_impute_confidence(raw_kps)
        if clean_kps is None:
            return None

        # 2. 골반 중심화 (Root-Centering)
        centered_kps = cls.root_center_pelvis(clean_kps)

        # 3. 카메라 롤 수평 회전 정렬
        rotated_kps, rot_angle = cls.in_plane_rotate(centered_kps)

        # 4. 체간 척추 스케일 정규화
        scaled_kps, torso_scale = cls.scale_normalize_torso(rotated_kps)

        # 5. 발바닥 3각 지지면 기하 특징 사전 추출
        foot_info = cls.extract_foot_triangle_base(scaled_kps)

        # 6. 편측 운동 좌우 미러링
        canonical_kps, is_flipped = cls.lateral_canonicalize(scaled_kps, exercise_name)

        # 7. 표준 64프레임 3차 스플라인 리샘플링
        resampled_64f = cls.resample_to_64f(canonical_kps, target_frames=target_frames)

        # 하위 호환 15-관절 64F 변환
        resampled_15_64f = cls.halpe26_to_joint15(resampled_64f)

        return {
            'kps_26_64f': resampled_64f,       # (64, 26, 3) 표준 RTMPose 전처리 텐서
            'kps_15_64f': resampled_15_64f,    # (64, 15, 3) 1단계 모델 호환 텐서
            'kps_clean': clean_kps,            # (T, 26, 3) 결측 보간된 원본 스케일 텐서
            'foot_info': foot_info,            # 발 3각 기하 정보
            'orig_frames': raw_kps.shape[0],
            'rot_angle_deg': float(np.degrees(rot_angle)),
            'torso_scale': torso_scale,
            'is_flipped': is_flipped
        }
