import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional
from src.core.skeleton import Joint15, SYMMETRIC_PAIRS
from src.core.types import KeypointSequence

class StandardPreprocessor:
    """5단계 생체역학 표준 전처리 및 기하/시간 정규화 파이프라인"""
    
    def __init__(self, target_frames: int = 64):
        self.target_frames = target_frames

    def process_sequence(self, seq: KeypointSequence, is_lateral: bool = False, target_side: str = 'left') -> KeypointSequence:
        """5단계 표준 전처리 순차 적용"""
        kps = seq.keypoints.copy() # (T, 15, 3)
        
        # 1. 골반 중심 원점 정렬 (Origin Centering)
        kps = self.origin_centering(kps)
        
        # 2. 신체 분절 스케일 정규화 (Scale Normalization)
        kps = self.scale_normalization(kps)
        
        # 3. 체간 축 기반 회전 정렬 (View Alignment)
        kps = self.view_alignment(kps)
        
        # 4. 좌우 대칭 통일 (Lateral Flip Normalization)
        if is_lateral:
            kps = self.lateral_flip(kps, target_side)
            
        # 5. 시간축 3차 스플라인 리샘플링 (Temporal Resampling, T=target_frames)
        kps = self.temporal_resampling(kps, self.target_frames)
        
        seq.keypoints = kps
        return seq

    @staticmethod
    def origin_centering(kps: np.ndarray) -> np.ndarray:
        """Step 1: Pelvis(골반 중심)을 원점 (0, 0, 0)으로 평행이동"""
        pelvis = kps[:, [Joint15.PELVIS], :] # (T, 1, 3)
        return kps - pelvis

    @staticmethod
    def scale_normalization(kps: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Step 2: ShoulderMidpoint to Pelvis 거리를 기준 길이 1.0으로 정규화"""
        l_shoulder = kps[:, Joint15.LEFT_SHOULDER, :] # (T, 3)
        r_shoulder = kps[:, Joint15.RIGHT_SHOULDER, :] # (T, 3)
        shoulder_mid = (l_shoulder + r_shoulder) / 2.0 # (T, 3)
        pelvis = kps[:, Joint15.PELVIS, :] # (T, 3)
        
        # 체간 길이 (T,)
        torso_lengths = np.linalg.norm(shoulder_mid - pelvis, axis=-1)
        mean_torso_length = np.mean(torso_lengths)
        
        if mean_torso_length < eps:
            scale_factor = 1.0
        else:
            scale_factor = 1.0 / mean_torso_length
            
        return kps * scale_factor

    @staticmethod
    def view_alignment(kps: np.ndarray) -> np.ndarray:
        """Step 3: 양 어깨 벡터를 X축, 척추 벡터를 Y축/Z축으로 정렬"""
        l_shoulder = np.mean(kps[:, Joint15.LEFT_SHOULDER, :], axis=0)
        r_shoulder = np.mean(kps[:, Joint15.RIGHT_SHOULDER, :], axis=0)
        
        x_axis = r_shoulder - l_shoulder
        norm_x = np.linalg.norm(x_axis)
        if norm_x > 1e-6:
            x_axis = x_axis / norm_x
        else:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            
        # Z축 근사 (위쪽 방향)
        torso = np.mean(kps[:, Joint15.TORSO, :], axis=0)
        pelvis = np.mean(kps[:, Joint15.PELVIS, :], axis=0)
        up_vector = torso - pelvis
        
        # 직교화 (Gram-Schmidt)
        z_axis = up_vector - np.dot(up_vector, x_axis) * x_axis
        norm_z = np.linalg.norm(z_axis)
        if norm_z > 1e-6:
            z_axis = z_axis / norm_z
        else:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=0) # (3, 3)
        
        # 회전 행렬 적용 (T, 15, 3) @ (3, 3).T
        aligned_kps = np.einsum('tjc,kc->tjk', kps, R)
        return aligned_kps

    @staticmethod
    def lateral_flip(kps: np.ndarray, target_side: str = 'left') -> np.ndarray:
        """Step 4: 편측 운동 대칭 반전 (X축 반전 및 좌우 관절 스왑)"""
        flipped = kps.copy()
        flipped[:, :, 0] = -flipped[:, :, 0] # Flip X
        
        for left_idx, right_idx in SYMMETRIC_PAIRS:
            temp = flipped[:, left_idx, :].copy()
            flipped[:, left_idx, :] = flipped[:, right_idx, :]
            flipped[:, right_idx, :] = temp
            
        return flipped

    @staticmethod
    def temporal_resampling(kps: np.ndarray, target_frames: int = 64) -> np.ndarray:
        """Step 5: 3차 스플라인(Cubic Spline) 보간을 통해 시간축 T=target_frames 고정"""
        T_orig, num_joints, dims = kps.shape
        if T_orig == target_frames:
            return kps
            
        orig_t = np.linspace(0.0, 1.0, T_orig)
        new_t = np.linspace(0.0, 1.0, target_frames)
        
        resampled = np.zeros((target_frames, num_joints, dims), dtype=np.float32)
        for j in range(num_joints):
            for d in range(dims):
                cs = CubicSpline(orig_t, kps[:, j, d], bc_type='natural')
                resampled[:, j, d] = cs(new_t)
                
        return resampled
