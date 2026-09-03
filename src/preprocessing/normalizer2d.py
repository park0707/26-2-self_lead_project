import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Dict, Any, Tuple
from src.core.skeleton2d import Joint15, SYMMETRIC_PAIRS

class GeometricNormalizer2D:
    """2D 15-Joint 5단계 기하 정규화 및 시간축 리샘플링 엔진"""

    @staticmethod
    def filter_and_interpolate_2d(kps_2d: np.ndarray, min_valid_ratio: float = 0.5) -> Optional[np.ndarray]:
        """(T, 15, 2) 결측치((0, 0) 동시 0 또는 NaN) 탐지 및 3차 스플라인/선형 시간축 보간"""
        T, J, C = kps_2d.shape
        if T < 4:
            return None
            
        kps_clean = np.copy(kps_2d).astype(np.float32)
        
        # NaN 탐지
        is_nan = np.isnan(kps_clean).any(axis=-1) # (T, J)
        # X, Y 둘 다 0인 경우만 결측 포인트로 판단
        is_zero = (np.abs(kps_clean[:, :, 0]) < 1e-6) & (np.abs(kps_clean[:, :, 1]) < 1e-6) # (T, J)
        is_invalid_point = is_nan | is_zero # (T, J)
        
        valid_ratio = 1.0 - (float(np.sum(is_invalid_point)) / (T * J))
        if valid_ratio < min_valid_ratio:
            return None
            
        t_indices = np.arange(T)
        for j in range(J):
            valid_mask = ~is_invalid_point[:, j]
            valid_count = int(np.sum(valid_mask))
            if valid_count < 2:
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
    def root_center_2d(kps_2d: np.ndarray) -> np.ndarray:
        """Pelvis(0번 관절)를 매 프레임 원점 (0, 0)으로 평행 이동"""
        pelvis = kps_2d[:, Joint15.PELVIS:Joint15.PELVIS+1, :] # (T, 1, 2)
        return kps_2d - pelvis

    @staticmethod
    def in_plane_rotate_2d(kps_2d: np.ndarray, num_init_frames: int = 5) -> Tuple[np.ndarray, float]:
        """초기 선 자세의 어깨선/골반선을 수평으로 2D 회전 정렬 (카메라 롤 오차 보정)"""
        T = kps_2d.shape[0]
        n_frames = min(T, num_init_frames)
        
        # 어깨 벡터
        sh_vec = np.mean(
            kps_2d[:n_frames, Joint15.RIGHT_SHOULDER, :] - kps_2d[:n_frames, Joint15.LEFT_SHOULDER, :],
            axis=0
        )
        
        # 어깨 폭이 너무 좁으면(측면 뷰 등) 골반 벡터 활용
        if np.linalg.norm(sh_vec) < 1e-4:
            sh_vec = np.mean(
                kps_2d[:n_frames, Joint15.RIGHT_HIP, :] - kps_2d[:n_frames, Joint15.LEFT_HIP, :],
                axis=0
            )
            
        norm = np.linalg.norm(sh_vec)
        if norm < 1e-4:
            return kps_2d, 0.0
            
        theta = float(np.arctan2(sh_vec[1], sh_vec[0]))
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        
        kps_rot = np.matmul(kps_2d, R.T)
        return kps_rot, theta

    @staticmethod
    def scale_normalize_2d(kps_2d: np.ndarray) -> Tuple[np.ndarray, float]:
        """초기 체간(Torso-Pelvis) 길이를 1.0으로 스케일 정규화 (신장/거리 불변성)"""
        torso_vec = kps_2d[:, Joint15.TORSO, :] - kps_2d[:, Joint15.PELVIS, :]
        torso_len = float(np.mean(np.linalg.norm(torso_vec, axis=-1)))
        
        if torso_len < 1e-4:
            neck_vec = kps_2d[:, Joint15.NECK, :] - kps_2d[:, Joint15.PELVIS, :]
            torso_len = float(np.mean(np.linalg.norm(neck_vec, axis=-1)))
            
        if torso_len < 1e-4:
            torso_len = 1.0
            
        kps_scaled = kps_2d / (torso_len + 1e-7)
        return kps_scaled, torso_len

    @staticmethod
    def lateral_canonicalize_2d(kps_2d: np.ndarray, exercise_name: str) -> Tuple[np.ndarray, bool]:
        """런지 등 편측 운동 시 항상 우측 다리 전방으로 대칭 미러링"""
        is_flipped = False
        kps_out = np.copy(kps_2d)
        
        if any(keyword in exercise_name for keyword in ['런지', 'lunge', 'Lunge']):
            l_ankle_disp = np.max(np.abs(kps_2d[:, Joint15.LEFT_ANKLE, :]))
            r_ankle_disp = np.max(np.abs(kps_2d[:, Joint15.RIGHT_ANKLE, :]))
            
            if l_ankle_disp > r_ankle_disp:
                kps_out[:, :, 0] = -kps_out[:, :, 0]
                for left_idx, right_idx in SYMMETRIC_PAIRS:
                    tmp = np.copy(kps_out[:, left_idx, :])
                    kps_out[:, left_idx, :] = kps_out[:, right_idx, :]
                    kps_out[:, right_idx, :] = tmp
                is_flipped = True
                
        return kps_out, is_flipped

    @staticmethod
    def temporal_resample_2d(kps_2d: np.ndarray, target_frames: int = 64) -> np.ndarray:
        """3차 스플라인 시간축 리샘플링 (고정 T=64 텐서 생성)"""
        T_orig, J, C = kps_2d.shape
        if T_orig == target_frames:
            return kps_2d.astype(np.float32)
            
        t_orig = np.linspace(0.0, 1.0, T_orig)
        t_target = np.linspace(0.0, 1.0, target_frames)
        
        kind = 'cubic' if T_orig >= 4 else 'linear'
        f = interp1d(t_orig, kps_2d, axis=0, kind=kind, bounds_error=False, fill_value='extrapolate')
        return f(t_target).astype(np.float32)

    @classmethod
    def process_sequence_2d(cls, kps_2d: np.ndarray, exercise_name: str, target_frames: int = 64) -> Optional[Dict[str, Any]]:
        """5단계 전처리 전체 파이프라인 일괄 실행"""
        clean_kps = cls.filter_and_interpolate_2d(kps_2d)
        if clean_kps is None:
            return None
            
        centered_kps = cls.root_center_2d(clean_kps)
        rotated_kps, rot_angle = cls.in_plane_rotate_2d(centered_kps)
        scaled_kps, torso_scale = cls.scale_normalize_2d(rotated_kps)
        canonical_kps, is_flipped = cls.lateral_canonicalize_2d(scaled_kps, exercise_name)
        resampled_64 = cls.temporal_resample_2d(canonical_kps, target_frames=target_frames)
        
        return {
            'keypoints_2d_norm': resampled_64,
            'keypoints_2d_canonical': canonical_kps,
            'keypoints_2d_clean': clean_kps,
            'orig_frames': kps_2d.shape[0],
            'rot_angle_deg': float(np.degrees(rot_angle)),
            'torso_scale': torso_scale,
            'is_flipped': is_flipped
        }
