from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class KeypointSequence:
    """표준화된 단일 2D/3D 운동 시퀀스 데이터 클래스"""
    sequence_id: str
    exercise_name: str
    label: int                              # 1: Normal/Correct, 0: Incorrect/Non-optimal
    keypoints_2d: Optional[np.ndarray] = None  # Shape: (T, 15, 2)
    keypoints_3d: Optional[np.ndarray] = None  # Shape: (T, 15, 3) (미터 단위 월드 좌표)
    visibility: Optional[np.ndarray] = None    # Shape: (T, 15)
    fps: float = 30.0                       # 원본 프레임레이트
    conditions: Optional[List[bool]] = None
    subject_id: Optional[str] = None
    source_dataset: str = 'AIHub'           # 'AIHub', 'MediaPipe', 'Fused'
    metadata: Optional[Dict[str, Any]] = None

    @property
    def keypoints(self) -> np.ndarray:
        """기본 키포인트 배열 반환 (3D가 존재하면 3D, 아니면 2D)"""
        if self.keypoints_3d is not None:
            return self.keypoints_3d
        if self.keypoints_2d is not None:
            return self.keypoints_2d
        raise ValueError("키포인트 데이터가 비어 있습니다.")

# 기존 코드와의 100% 하위 호환성을 위한 별칭(Alias)
KeypointSequence2D = KeypointSequence
