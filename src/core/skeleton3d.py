from enum import IntEnum
from typing import List, Tuple
import numpy as np
from src.core.skeleton2d import Joint15, SKELETON_EDGES, SYMMETRIC_PAIRS

# 3D 관절 위상은 2D와 100% 동일한 15-Joint 표준을 공유
Joint15_3D = Joint15
SKELETON_EDGES_3D = SKELETON_EDGES
SYMMETRIC_PAIRS_3D = SYMMETRIC_PAIRS

# 3D 해부학적 주요 분절 벡터 정의 (Start Joint -> End Joint)
ANATOMICAL_VECTORS_3D = {
    'spine_vector': (Joint15.PELVIS, Joint15.NECK),          # 척추 축 (Y축 정렬 기준)
    'shoulder_vector': (Joint15.LEFT_SHOULDER, Joint15.RIGHT_SHOULDER), # 어깨 축 (X축 정렬 기준)
    'hip_vector': (Joint15.LEFT_HIP, Joint15.RIGHT_HIP),                # 골반 축 (X축 정렬 기준)
    'left_thigh': (Joint15.LEFT_HIP, Joint15.LEFT_KNEE),     # 좌측 대퇴
    'right_thigh': (Joint15.RIGHT_HIP, Joint15.RIGHT_KNEE),  # 우측 대퇴
    'left_shank': (Joint15.LEFT_KNEE, Joint15.LEFT_ANKLE),   # 좌측 하퇴
    'right_shank': (Joint15.RIGHT_KNEE, Joint15.RIGHT_ANKLE),# 우측 하퇴
    'left_arm': (Joint15.LEFT_SHOULDER, Joint15.LEFT_ELBOW), # 좌측 상완
    'right_arm': (Joint15.RIGHT_SHOULDER, Joint15.RIGHT_ELBOW), # 우측 상완
    'left_forearm': (Joint15.LEFT_ELBOW, Joint15.LEFT_WRIST), # 좌측 전완
    'right_forearm': (Joint15.RIGHT_ELBOW, Joint15.RIGHT_WRIST) # 우측 전완
}

# 3D 8대 핵심 관절 각도 정의 (Point 1, Vertex Point 2, Point 3)
JOINT_ANGLE_DEFINITIONS_3D = {
    'knee_angle_l': (Joint15.LEFT_HIP, Joint15.LEFT_KNEE, Joint15.LEFT_ANKLE),
    'knee_angle_r': (Joint15.RIGHT_HIP, Joint15.RIGHT_KNEE, Joint15.RIGHT_ANKLE),
    'hip_angle_l': (Joint15.TORSO, Joint15.LEFT_HIP, Joint15.LEFT_KNEE),
    'hip_angle_r': (Joint15.TORSO, Joint15.RIGHT_HIP, Joint15.RIGHT_KNEE),
    'ankle_angle_l': (Joint15.LEFT_KNEE, Joint15.LEFT_ANKLE, Joint15.LEFT_ANKLE), # 발끝 방향 고려
    'ankle_angle_r': (Joint15.RIGHT_KNEE, Joint15.RIGHT_ANKLE, Joint15.RIGHT_ANKLE),
    'elbow_angle_l': (Joint15.LEFT_SHOULDER, Joint15.LEFT_ELBOW, Joint15.LEFT_WRIST),
    'elbow_angle_r': (Joint15.RIGHT_SHOULDER, Joint15.RIGHT_ELBOW, Joint15.RIGHT_WRIST)
}
