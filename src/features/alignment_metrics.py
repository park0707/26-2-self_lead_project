import numpy as np
from typing import Dict
from src.core.skeleton import Joint15

class AlignmentMetricsExtractor:
    """Knee Valgus, 체간 전방 경사, 좌우 비대칭 등 역학적 자세 결함 지표 추출기"""

    @staticmethod
    def extract_torso_inclination(kps: np.ndarray) -> np.ndarray:
        """체간 전방 경사각 (수직 Z축 [0, 0, 1] 대비 Torso-Pelvis 벡터의 기울기, degree)"""
        torso_vec = kps[:, Joint15.TORSO, :] - kps[:, Joint15.PELVIS, :] # (T, 3)
        vertical_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32) # (3,)
        
        norm_torso = np.linalg.norm(torso_vec, axis=-1) + 1e-7
        dot = np.dot(torso_vec, vertical_vec) # (T,)
        cos_val = np.clip(dot / norm_torso, -1.0, 1.0)
        return np.degrees(np.arccos(cos_val))

    @staticmethod
    def extract_knee_valgus(kps: np.ndarray) -> Dict[str, np.ndarray]:
        """Knee Valgus (관상면 Frontal Plane에서 Hip-Knee-Ankle 2D 각도 편위, 안쪽 모임 검출)"""
        # 관상면 투영 (X, Z 좌표만 사용)
        hip_l_2d = kps[:, Joint15.LEFT_HIP, [0, 2]]
        knee_l_2d = kps[:, Joint15.LEFT_KNEE, [0, 2]]
        ankle_l_2d = kps[:, Joint15.LEFT_ANKLE, [0, 2]]
        
        hip_r_2d = kps[:, Joint15.RIGHT_HIP, [0, 2]]
        knee_r_2d = kps[:, Joint15.RIGHT_KNEE, [0, 2]]
        ankle_r_2d = kps[:, Joint15.RIGHT_ANKLE, [0, 2]]
        
        # 180도 기준 편위 (180도 = 완벽한 일직선, <170도 = Valgus/Varus)
        def get_valgus(h, k, a):
            v1 = h - k
            v2 = a - k
            dot = np.sum(v1 * v2, axis=-1)
            norm = (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)) + 1e-7
            return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
            
        return {
            'knee_valgus_l': get_valgus(hip_l_2d, knee_l_2d, ankle_l_2d),
            'knee_valgus_r': get_valgus(hip_r_2d, knee_r_2d, ankle_r_2d)
        }

    @staticmethod
    def extract_asymmetry_index(left_angles: np.ndarray, right_angles: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """좌우 대칭성 지표: |Left - Right| / (|Left| + |Right| + eps) * 100 (%)"""
        diff = np.abs(left_angles - right_angles)
        denom = np.abs(left_angles) + np.abs(right_angles) + eps
        return (diff / denom) * 100.0

    @classmethod
    def extract_all_alignments(cls, kps: np.ndarray, joint_angles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """모든 정렬 및 비대칭 피처 통합 산출"""
        res = {}
        res['torso_inclination'] = cls.extract_torso_inclination(kps)
        valgus = cls.extract_knee_valgus(kps)
        res.update(valgus)
        
        # 좌우 비대칭률
        if 'knee_angle_l' in joint_angles and 'knee_angle_r' in joint_angles:
            res['knee_asymmetry'] = cls.extract_asymmetry_index(
                joint_angles['knee_angle_l'], joint_angles['knee_angle_r']
            )
        if 'hip_angle_l' in joint_angles and 'hip_angle_r' in joint_angles:
            res['hip_asymmetry'] = cls.extract_asymmetry_index(
                joint_angles['hip_angle_l'], joint_angles['hip_angle_r']
            )
            
        return res
