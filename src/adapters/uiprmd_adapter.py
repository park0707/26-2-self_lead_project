import os
from typing import Optional, List
import numpy as np
from src.core.skeleton import Joint15
from src.core.types import KeypointSequence

class UIPRMDAdapter:
    """UI-PRMD 물리치료 재활 데이터셋 TXT 파서 및 15-Joint 변환기"""
    
    # Kinect v2 22개 관절 중 15-Joint 인덱스 매핑 (0-indexed)
    # 0: Waist, 1: Spine, 2: Chest, 3: Neck, 4: Head,
    # 5: Left shoulder, 6: Left elbow, 7: Left wrist, 8: Left hand,
    # 12: Right shoulder, 13: Right elbow, 14: Right wrist, 15: Right hand,
    # 18: Left hip, 19: Left knee, 20: Left ankle, 21: Left foot,
    # 22: Right hip, 23: Right knee, 24: Right ankle, 25: Right foot (총 22개 구조)
    
    KINECT_TO_15_MAP = {
        Joint15.PELVIS: 0,          # Waist
        Joint15.TORSO: 1,           # Spine
        Joint15.LEFT_SHOULDER: 4,   # Left shoulder (index 4 or 5)
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
    def parse_kinect_txt(cls, txt_path: str, movement_name: str, is_correct: int) -> Optional[KeypointSequence]:
        """Kinect 66D Positions TXT 파일 파싱 및 15-Joint 추출"""
        if not os.path.exists(txt_path):
            return None
            
        data = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 66:
                    vals = [float(p.strip()) for p in parts[:66]]
                    data.append(vals)
                    
        if not data:
            return None
            
        arr = np.array(data, dtype=np.float32) # (T, 66)
        T = arr.shape[0]
        arr_3d = arr.reshape(T, 22, 3) # (T, 22, 3)
        
        # 15개 관절 추출
        kps_15 = np.zeros((T, 15, 3), dtype=np.float32)
        for j15_idx, kinect_idx in cls.KINECT_TO_15_MAP.items():
            if kinect_idx < 22:
                kps_15[:, j15_idx, :] = arr_3d[:, kinect_idx, :]
                
        seq_id = os.path.splitext(os.path.basename(txt_path))[0]
        return KeypointSequence(
            sequence_id=seq_id,
            exercise_name=movement_name,
            label=is_correct,
            keypoints=kps_15,
            fps=30.0,
            source_dataset='UI-PRMD_Kinect'
        )
