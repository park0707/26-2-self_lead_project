import os
from typing import Optional, List
import numpy as np
from src.core.skeleton import Joint15
from src.core.types import KeypointSequence

class REHABAdapter:
    """REHAB24-6 모션캡처 NPY 및 15-Joint 변환기"""
    
    # REHAB24-6 26개 관절 토폴로지 (joints_names.txt 기준 0-indexed)
    # 0: Hips, 1: Spine, 2: Spine1, 3: Neck, 4: Head, 5: Head_end,
    # 6: LeftShoulder, 7: LeftArm, 8: LeftForeArm, 9: LeftHand, 10: LeftHand_end,
    # 11: RightShoulder, 12: RightArm, 13: RightForeArm, 14: RightHand, 15: RightHand_end,
    # 16: LeftUpLeg, 17: LeftLeg, 18: LeftFoot, 19: LeftToeBase, 20: LeftToeBase_end,
    # 21: RightUpLeg, 22: RightLeg, 23: RightFoot, 24: RightToeBase, 25: RightToeBase_end
    
    REHAB_TO_15_MAP = {
        Joint15.PELVIS: 0,          # Hips
        Joint15.TORSO: 1,           # Spine
        Joint15.LEFT_SHOULDER: 6,   # LeftShoulder
        Joint15.RIGHT_SHOULDER: 11, # RightShoulder
        Joint15.LEFT_ELBOW: 7,      # LeftArm (Elbow proxy)
        Joint15.RIGHT_ELBOW: 12,    # RightArm (Elbow proxy)
        Joint15.LEFT_WRIST: 8,      # LeftForeArm (Wrist proxy)
        Joint15.RIGHT_WRIST: 13,    # RightForeArm (Wrist proxy)
        Joint15.LEFT_HIP: 16,       # LeftUpLeg
        Joint15.RIGHT_HIP: 21,      # RightUpLeg
        Joint15.LEFT_KNEE: 17,      # LeftLeg
        Joint15.RIGHT_KNEE: 22,     # RightLeg
        Joint15.LEFT_ANKLE: 18,     # LeftFoot
        Joint15.RIGHT_ANKLE: 23,    # RightFoot
        Joint15.NECK: 3             # Neck
    }

    @classmethod
    def parse_3d_npy(cls, npy_path: str, exercise_name: str, label: int, fps: float = 30.0) -> Optional[KeypointSequence]:
        """REHAB24-6 3D joints NPY 로드 및 15-Joint 추출 (T, 26, 4 -> T, 15, 3)"""
        if not os.path.exists(npy_path):
            return None
            
        arr = np.load(npy_path) # Shape: (T, 26, 4) or (T, 26, 3)
        if arr.ndim != 3 or arr.shape[1] != 26:
            return None
            
        # 동차좌표 w=1.0 제거 (XYZ만 유지)
        xyz = arr[:, :, :3]
        T = xyz.shape[0]
        
        kps_15 = np.zeros((T, 15, 3), dtype=np.float32)
        for j15_idx, r_idx in cls.REHAB_TO_15_MAP.items():
            kps_15[:, j15_idx, :] = xyz[:, r_idx, :]
            
        seq_id = os.path.splitext(os.path.basename(npy_path))[0]
        return KeypointSequence(
            sequence_id=seq_id,
            exercise_name=exercise_name,
            label=label,
            keypoints=kps_15,
            fps=fps,
            source_dataset='REHAB24-6_OptiTrack'
        )
