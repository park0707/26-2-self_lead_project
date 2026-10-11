from enum import IntEnum
from typing import List, Tuple

class Joint15(IntEnum):
    """15-Joint Unified 2D Common Skeleton Topology"""
    PELVIS = 0            # 골반 중심 (Root Origin (0, 0))
    TORSO = 1             # 척추 / 체간 중심 (Scale Reference L=1.0)
    LEFT_SHOULDER = 2     # 좌측 어깨 (In-plane Rotation Reference)
    RIGHT_SHOULDER = 3    # 우측 어깨 (In-plane Rotation Reference)
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

# 2D 신체 연결선 (Bones / Edges)
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

# 좌우 대칭 쌍 (Lateral Flip 및 미러링용)
SYMMETRIC_PAIRS: List[Tuple[int, int]] = [
    (Joint15.LEFT_SHOULDER, Joint15.RIGHT_SHOULDER),
    (Joint15.LEFT_ELBOW, Joint15.RIGHT_ELBOW),
    (Joint15.LEFT_WRIST, Joint15.RIGHT_WRIST),
    (Joint15.LEFT_HIP, Joint15.RIGHT_HIP),
    (Joint15.LEFT_KNEE, Joint15.RIGHT_KNEE),
    (Joint15.LEFT_ANKLE, Joint15.RIGHT_ANKLE)
]


class JointHalpe26(IntEnum):
    """Halpe 26-Keypoint Topology for RTMPose-m BodyWithFeet"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    HEAD = 17
    NECK = 18
    HIP = 19              # Pelvis 중심
    LEFT_BIG_TOE = 20
    RIGHT_BIG_TOE = 21
    LEFT_SMALL_TOE = 22
    RIGHT_SMALL_TOE = 23
    LEFT_HEEL = 24
    RIGHT_HEEL = 25


SYMMETRIC_PAIRS_HALPE26: List[Tuple[int, int]] = [
    (JointHalpe26.LEFT_EYE, JointHalpe26.RIGHT_EYE),
    (JointHalpe26.LEFT_EAR, JointHalpe26.RIGHT_EAR),
    (JointHalpe26.LEFT_SHOULDER, JointHalpe26.RIGHT_SHOULDER),
    (JointHalpe26.LEFT_ELBOW, JointHalpe26.RIGHT_ELBOW),
    (JointHalpe26.LEFT_WRIST, JointHalpe26.RIGHT_WRIST),
    (JointHalpe26.LEFT_HIP, JointHalpe26.RIGHT_HIP),
    (JointHalpe26.LEFT_KNEE, JointHalpe26.RIGHT_KNEE),
    (JointHalpe26.LEFT_ANKLE, JointHalpe26.RIGHT_ANKLE),
    (JointHalpe26.LEFT_BIG_TOE, JointHalpe26.RIGHT_BIG_TOE),
    (JointHalpe26.LEFT_SMALL_TOE, JointHalpe26.RIGHT_SMALL_TOE),
    (JointHalpe26.LEFT_HEEL, JointHalpe26.RIGHT_HEEL)
]

