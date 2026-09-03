from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class KeypointSequence2D:
    """표준화된 단일 2D 운동 시퀀스 데이터 클래스"""
    sequence_id: str
    exercise_name: str
    label: int                  # 1: Normal/Correct, 0: Incorrect/Non-optimal
    keypoints_2d: np.ndarray    # Shape: (T, 15, 2)
    fps: float = 30.0           # 원본 프레임레이트
    conditions: Optional[List[bool]] = None
    subject_id: Optional[str] = None
    source_dataset: str = 'AIHub'
    metadata: Optional[Dict[str, Any]] = None

    @property
    def keypoints(self) -> np.ndarray:
        return self.keypoints_2d
