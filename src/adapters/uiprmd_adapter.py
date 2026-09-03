import os
from typing import Optional, List
import numpy as np
from src.core.skeleton2d import Joint15
from src.core.types import KeypointSequence2D

class UIPRMDAdapter:
    """UI-PRMD 물리치료 재활 2D 투영 파서 및 15-Joint 변환기"""

    KINECT_TO_15_MAP = {
        Joint15.PELVIS: 0,          # SpineBase (골반 중심)
        Joint15.TORSO: 1,           # SpineMid (체간/척추)
        Joint15.LEFT_SHOULDER: 4,   # Left shoulder
        Joint15.RIGHT_SHOULDER: 10, # Right shoulder
        Joint15.LEFT_ELBOW: 5,      # Left elbow
        Joint15.RIGHT_ELBOW: 11,    # Right elbow
        Joint15.LEFT_WRIST: 6,      # Left wrist
        Joint15.RIGHT_WRIST: 12,    # Right wrist
        Joint15.LEFT_HIP: 16,       # Left hip
        Joint15.RIGHT_HIP: 19,      # Right hip
        Joint15.LEFT_KNEE: 17,      # Left knee
        Joint15.RIGHT_KNEE: 20,     # Right knee
        Joint15.LEFT_ANKLE: 18,     # Left ankle
        Joint15.RIGHT_ANKLE: 21,    # Right ankle
        Joint15.NECK: 3             # Neck
    }

    @classmethod
    def parse_kinect_txt_2d(cls, txt_path: str, movement_name: str, is_correct: int, proj_plane: str = 'XY') -> Optional[KeypointSequence2D]:
        """Kinect 66D TXT를 파싱하여 지정 평면(XY: 정면/관상면, XZ: 수평, YZ: 측면) 2D로 투영 후 15-Joint 추출"""
        if not os.path.exists(txt_path):
            return None

        data = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 66:
                        vals = [float(p.strip()) for p in parts[:66]]
                        data.append(vals)
        except Exception:
            return None

        if not data or len(data) < 4:
            return None

        arr = np.array(data, dtype=np.float32) # (T, 66)
        T = arr.shape[0]
        arr_3d = arr.reshape(T, 22, 3) # (T, 22, 3)

        # 2D 투영 인덱스 선택 (기본: X, Y 정면 투영)
        if proj_plane.upper() == 'YZ':
            coords_2d = arr_3d[:, :, [2, 1]] # Z (깊이) -> X, Y (수직) -> Y (측면 시상면)
        else:
            coords_2d = arr_3d[:, :, [0, 1]] # X (좌우), Y (수직) (정면 관상면)

        kps_15_2d = np.zeros((T, 15, 2), dtype=np.float32)
        for j15_idx, kinect_idx in cls.KINECT_TO_15_MAP.items():
            if kinect_idx < 22:
                kps_15_2d[:, j15_idx, :] = coords_2d[:, kinect_idx, :]

        seq_id = os.path.splitext(os.path.basename(txt_path))[0]
        return KeypointSequence2D(
            sequence_id=seq_id,
            exercise_name=movement_name,
            label=is_correct,
            keypoints_2d=kps_15_2d,
            fps=30.0,
            source_dataset='UI-PRMD_Kinect'
        )
