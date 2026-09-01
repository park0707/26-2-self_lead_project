from enum import IntEnum
from typing import Dict, List, Tuple
import numpy as np

class Joint15(IntEnum):
    """15-Joint Unified Common Skeleton Topology"""
    PELVIS = 0            # 골반 중심 (Origin)
    TORSO = 1             # 척추 / 체간 중심
    LEFT_SHOULDER = 2     # 좌측 어깨
    RIGHT_SHOULDER = 3    # 우측 어깨
    LEFT_ELBOW = 4        # 좌측 팔꿈치
    RIGHT_ELBOW = 5       # 우측 팔꿈치
    LEFT_WRIST = 6        # 좌측 손목
    RIGHT_WRIST = 7       # 우측 손목
    LEFT_HIP = 8          # 좌측 고관절
    RIGHT_HIP = 9         # 우측 고관절
    LEFT_KNEE = 10        # 좌측 무릎
    RIGHT_KNEE = 11       # 우측 무릎
    LEFT_ANKLE = 12       # 좌측 발목
    RIGHT_ANKLE = 13      # 우측 발목
    NECK = 14             # 목 / 상부 체간

# 신체 연결선 (Bones / Edges for visualization and GCN)
SKELETON_EDGES: List[Tuple[int, int]] = [
    (Joint15.PELVIS, Joint15.TORSO),
    (Joint15.TORSO, Joint15.NECK),
    (Joint15.TORSO, Joint15.LEFT_SHOULDER),
    (Joint15.TORSO, Joint15.RIGHT_SHOULDER),
    (Joint15.LEFT_SHOULDER, Joint15.LEFT_ELBOW),
    (Joint15.LEFT_ELBOW, Joint15.LEFT_WRIST),
    (Joint15.RIGHT_SHOULDER, Joint15.RIGHT_ELBOW),
    (Joint15.RIGHT_ELBOW, Joint15.RIGHT_WRIST),
    (Joint15.PELVIS, Joint15.LEFT_HIP),
    (Joint15.PELVIS, Joint15.RIGHT_HIP),
    (Joint15.LEFT_HIP, Joint15.LEFT_KNEE),
    (Joint15.LEFT_KNEE, Joint15.LEFT_ANKLE),
    (Joint15.RIGHT_HIP, Joint15.RIGHT_KNEE),
    (Joint15.RIGHT_KNEE, Joint15.RIGHT_ANKLE)
]

# 좌우 대칭 쌍 (Lateral Flip 용)
SYMMETRIC_PAIRS: List[Tuple[int, int]] = [
    (Joint15.LEFT_SHOULDER, Joint15.RIGHT_SHOULDER),
    (Joint15.LEFT_ELBOW, Joint15.RIGHT_ELBOW),
    (Joint15.LEFT_WRIST, Joint15.RIGHT_WRIST),
    (Joint15.LEFT_HIP, Joint15.RIGHT_HIP),
    (Joint15.LEFT_KNEE, Joint15.RIGHT_KNEE),
    (Joint15.LEFT_ANKLE, Joint15.RIGHT_ANKLE)
]
