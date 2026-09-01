from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, List, Dict, Any
import numpy as np

class ExerciseCategory(str, Enum):
    WEIGHT = 'weight_loaded'
    BODYWEIGHT = 'bodyweight'

class LabelType(IntEnum):
    INCORRECT = 0
    CORRECT = 1

@dataclass
class KeypointSequence:
    """표준화된 단일 운동 시퀀스 데이터 클래스"""
    sequence_id: str
    exercise_name: str
    label: int                  # 1: Normal/Correct, 0: Incorrect/Non-optimal
    keypoints: np.ndarray       # Shape: (T, 15, 3) or (T, 15, 2)
    fps: float                  # 원본 프레임레이트
    conditions: Optional[List[bool]] = None
    subject_id: Optional[str] = None
    source_dataset: str = 'AIHub'
    metadata: Optional[Dict[str, Any]] = None
