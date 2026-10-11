import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Dict, Any, Tuple
from src.core.skeleton2d import Joint15, SYMMETRIC_PAIRS

class GeometricNormalizer3D:
    """3D 15-Joint 5단계 기하 정규화 및 시간축 리샘플링 엔진"""

    @staticmethod
    def filter_and_interpolate_3d(kps_3d: np.ndarray, min_valid_ratio: float = 0.5) -> Optional[np.ndarray]:
        """(T, 15, 3) 결측치((0, 0, 0) 동시 0 또는 NaN) 탐지 및 3차 스플라인/선형 시간축 보간"""
        T, J, C = kps_3d.shape
        if T < 4:
            return None
            
        kps_clean = np.copy(kps_3d).astype(np.float32)
        
        # NaN 탐지
        is_nan = np.isnan(kps_clean).any(axis=-1) # (T, J)
        # X, Y, Z 모두 0에 수렴하는 경우 결측치 판단
        is_zero = (np.abs(kps_clean[:, :, 0]) < 1e-5) & (np.abs(kps_clean[:, :, 1]) < 1e-5) & (np.abs(kps_clean[:, :, 2]) < 1e-5)
        is_invalid_point = is_nan | is_zero # (T, J)
        
        valid_ratio = 1.0 - (float(np.sum(is_invalid_point)) / (T * J))
        if valid_ratio < min_valid_ratio:
            return None
            
        t_indices = np.arange(T)
        for j in range(J):
            valid_mask = ~is_invalid_point[:, j]
            valid_count = int(np.sum(valid_mask))
            if valid_count < 2:
                # 보간 불가 시 Pelvis 좌표로 대체
                kps_clean[:, j, :] = kps_clean[:, Joint15.PELVIS, :]
                continue
                
            t_valid = t_indices[valid_mask]
            for c_idx in range(C):
                vals_valid = kps_clean[valid_mask, j, c_idx]
                kind = 'cubic' if valid_count >= 4 else 'linear'
                try:
                    f = interp1d(t_valid, vals_valid, kind=kind, bounds_error=False, fill_value='extrapolate')
                    kps_clean[:, j, c_idx] = f(t_indices)
                except Exception:
                    f = interp1d(t_valid, vals_valid, kind='linear', bounds_error=False, fill_value='extrapolate')
                    kps_clean[:, j, c_idx] = f(t_indices)
                    
        return kps_clean

    @staticmethod
    def root_center_3d(kps_3d: np.ndarray) -> np.ndarray:
        """Pelvis(0번 관절)를 매 프레임 원점 (0, 0, 0)으로 평행 이동"""
        pelvis = kps_3d[:, Joint15.PELVIS:Joint15.PELVIS+1, :] # (T, 1, 3)
        return kps_3d - pelvis

    @staticmethod
    def spatial_rotate_3d(kps_3d: np.ndarray, num_init_frames: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        초기 선 자세(Stand pose)를 기준으로 3차원 공간 직교 좌표계 정렬
        - X축: 양 고관절 벡터 (Left Hip -> Right Hip)
        - Y축: 체간 척추 벡터 (Pelvis -> Neck)
        - Z축: X x Y 외적 법선 벡터 (Anteroposterior 전후 축)
        """
        T = kps_3d.shape[0]
        n_frames = min(T, num_init_frames)
        
        # 1. 초기 프레임의 고관절 벡터 및 척추 벡터 평균
        init_hip_vec = np.mean(
            kps_3d[:n_frames, Joint15.RIGHT_HIP, :] - kps_3d[:n_frames, Joint15.LEFT_HIP, :],
            axis=0
        ) # (3,)
        init_spine_vec = np.mean(
            kps_3d[:n_frames, Joint15.NECK, :] - kps_3d[:n_frames, Joint15.PELVIS, :],
            axis=0
        ) # (3,)

        # 2. 직교 기저 벡터 (Gram-Schmidt Orthogonalization)
        # Y축 기저: 척추 벡터 정규화
        y_axis = init_spine_vec / (np.linalg.norm(init_spine_vec) + 1e-7)
        
        # Z축 기저: 척추와 고관절의 외적 (앞뒤 방향)
        z_axis = np.cross(init_hip_vec, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-7)
        
        # X축 기저: Y축과 Z축의 외적 (좌우 방향)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-7)

        # 3. 회전 행렬 R (World -> Body Canonical Frame)
        R = np.stack([x_axis, y_axis, z_axis], axis=0) # (3, 3)

        # 4. 전체 시퀀스에 회전 변환 적용 (T, 15, 3) @ (3, 3)^T
        kps_rotated = np.einsum('tjc,kc->tjk', kps_3d, R)
        return kps_rotated, R

    @staticmethod
    def torso_scale_3d(kps_3d: np.ndarray, target_length: float = 1.0) -> Tuple[np.ndarray, float]:
        """Pelvis와 Torso 간의 3차원 유클리드 거리를 기준으로 인체 스케일 정규화"""
        pelvis = kps_3d[:, Joint15.PELVIS, :] # (T, 3)
        torso = kps_3d[:, Joint15.TORSO, :]   # (T, 3)
        
        torso_lengths = np.linalg.norm(torso - pelvis, axis=-1) # (T,)
        mean_torso_length = float(np.mean(torso_lengths))
        
        if mean_torso_length < 1e-4:
            scale_factor = 1.0
        else:
            scale_factor = target_length / mean_torso_length
            
        kps_scaled = kps_3d * scale_factor
        return kps_scaled, mean_torso_length

    @staticmethod
    def canonical_resample_time_3d(kps_3d: np.ndarray, target_frames: int = 64) -> np.ndarray:
        """시간축 균등 리샘플링 (T -> target_frames)"""
        T, J, C = kps_3d.shape
        if T == target_frames:
            return kps_3d.astype(np.float32)
            
        t_orig = np.linspace(0.0, 1.0, T)
        t_new = np.linspace(0.0, 1.0, target_frames)
        
        resampled = np.zeros((target_frames, J, C), dtype=np.float32)
        for j in range(J):
            for c in range(C):
                f = interp1d(t_orig, kps_3d[:, j, c], kind='linear', bounds_error=False, fill_value='extrapolate')
                resampled[:, j, c] = f(t_new)
                
        return resampled

    @classmethod
    def process_sequence_3d(
        cls,
        kps_3d_raw: np.ndarray,
        exercise_name: str = '',
        target_frames: int = 64
    ) -> Optional[Dict[str, Any]]:
        """3D 15-Joint 5단계 통합 기하 정규화 파이프라인"""
        # 1. 결측치 탐지 및 보간
        clean_kps = cls.filter_and_interpolate_3d(kps_3d_raw)
        if clean_kps is None:
            return None
            
        orig_frames = clean_kps.shape[0]

        # 2. 골반 중심 이동 (Pelvis -> (0,0,0))
        centered_kps = cls.root_center_3d(clean_kps)

        # 3. 3차원 공간 회전 정렬
        rotated_kps, R_mat = cls.spatial_rotate_3d(centered_kps)

        # 4. 체간 기준 3D 스케일 정규화
        scaled_kps, torso_scale = cls.torso_scale_3d(rotated_kps, target_length=1.0)

        # 5. 시간축 64프레임 리샘플링
        resampled_64 = cls.canonical_resample_time_3d(scaled_kps, target_frames=target_frames)

        return {
            'keypoints_3d_norm': resampled_64,         # (64, 15, 3)
            'keypoints_3d_canonical': scaled_kps,      # (T, 15, 3)
            'keypoints_3d_clean': clean_kps,           # (T, 15, 3)
            'orig_frames': orig_frames,
            'rotation_matrix': R_mat,
            'torso_scale': torso_scale
        }
