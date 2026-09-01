import numpy as np
from typing import Dict, List, Tuple
from src.core.skeleton import Joint15

class JointAngleExtractor:
    """3D/2D 관절 굴곡/신전 각도 및 척추 각도 계산 엔진"""

    @staticmethod
    def calculate_angle_3points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """세 점 p1-p2-p3 사이의 각도(p2가 꼭짓점)를 라디안 및 도(degree)로 계산
        p1, p2, p3 shape: (T, 3) or (T, 2)
        Output shape: (T,) in degrees [0, 180]
        """
        v1 = p1 - p2 # Vector from p2 to p1
        v2 = p3 - p2 # Vector from p2 to p3
        
        # 내적 및 노름 계산
        dot_prod = np.sum(v1 * v2, axis=-1)
        norm_v1 = np.linalg.norm(v1, axis=-1)
        norm_v2 = np.linalg.norm(v2, axis=-1)
        
        cos_theta = dot_prod / (norm_v1 * norm_v2 + eps)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        rad = np.arccos(cos_theta)
        return np.degrees(rad)

    @classmethod
    def extract_all_joint_angles(cls, kps: np.ndarray) -> Dict[str, np.ndarray]:
        """(T, 15, 3) 스켈레톤에서 주요 생체역학 관절 각도 추출"""
        angles = {}
        
        # 1. 하지 관절 각도 (무릎, 고관절, 발목)
        # 좌/우 무릎 굴곡각 (Hip - Knee - Ankle)
        angles['knee_angle_l'] = cls.calculate_angle_3points(
            kps[:, Joint15.LEFT_HIP, :], kps[:, Joint15.LEFT_KNEE, :], kps[:, Joint15.LEFT_ANKLE, :]
        )
        angles['knee_angle_r'] = cls.calculate_angle_3points(
            kps[:, Joint15.RIGHT_HIP, :], kps[:, Joint15.RIGHT_KNEE, :], kps[:, Joint15.RIGHT_ANKLE, :]
        )
        
        # 좌/우 고관절 굴곡각 (Torso - Hip - Knee)
        angles['hip_angle_l'] = cls.calculate_angle_3points(
            kps[:, Joint15.TORSO, :], kps[:, Joint15.LEFT_HIP, :], kps[:, Joint15.LEFT_KNEE, :]
        )
        angles['hip_angle_r'] = cls.calculate_angle_3points(
            kps[:, Joint15.TORSO, :], kps[:, Joint15.RIGHT_HIP, :], kps[:, Joint15.RIGHT_KNEE, :]
        )
        
        # 2. 상체 관절 각도 (팔꿈치, 어깨, 체간)
        # 좌/우 팔꿈치 굴곡각 (Shoulder - Elbow - Wrist)
        angles['elbow_angle_l'] = cls.calculate_angle_3points(
            kps[:, Joint15.LEFT_SHOULDER, :], kps[:, Joint15.LEFT_ELBOW, :], kps[:, Joint15.LEFT_WRIST, :]
        )
        angles['elbow_angle_r'] = cls.calculate_angle_3points(
            kps[:, Joint15.RIGHT_SHOULDER, :], kps[:, Joint15.RIGHT_ELBOW, :], kps[:, Joint15.RIGHT_WRIST, :]
        )
        
        # 좌/우 어깨 굴곡각 (Torso - Shoulder - Elbow)
        angles['shoulder_angle_l'] = cls.calculate_angle_3points(
            kps[:, Joint15.TORSO, :], kps[:, Joint15.LEFT_SHOULDER, :], kps[:, Joint15.LEFT_ELBOW, :]
        )
        angles['shoulder_angle_r'] = cls.calculate_angle_3points(
            kps[:, Joint15.TORSO, :], kps[:, Joint15.RIGHT_SHOULDER, :], kps[:, Joint15.RIGHT_ELBOW, :]
        )
        
        # 체간/척추 정렬 각도 (Pelvis - Torso - Neck)
        angles['spine_angle'] = cls.calculate_angle_3points(
            kps[:, Joint15.PELVIS, :], kps[:, Joint15.TORSO, :], kps[:, Joint15.NECK, :]
        )
        
        return angles
