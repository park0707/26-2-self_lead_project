import os
import csv
from typing import Optional, List, Dict, Any
import numpy as np
from src.core.skeleton2d import Joint15
from src.core.types import KeypointSequence2D

class REHABAdapter:
    """REHAB24-6 모션캡처 2D 투영 파서 및 15-Joint 변환기"""

    # REHAB24-6 26개 관절 토폴로지 (joints_names.txt 기준 0-indexed)
    REHAB_TO_15_MAP = {
        Joint15.PELVIS: 0,          # Hips (골반 중심)
        Joint15.TORSO: 1,           # Spine (척추/체간)
        Joint15.LEFT_SHOULDER: 6,   # LeftShoulder
        Joint15.RIGHT_SHOULDER: 11, # RightShoulder
        Joint15.LEFT_ELBOW: 7,      # LeftArm
        Joint15.RIGHT_ELBOW: 12,    # RightArm
        Joint15.LEFT_WRIST: 8,      # LeftForeArm
        Joint15.RIGHT_WRIST: 13,    # RightForeArm
        Joint15.LEFT_HIP: 16,       # LeftUpLeg
        Joint15.RIGHT_HIP: 21,      # RightUpLeg
        Joint15.LEFT_KNEE: 17,      # LeftLeg
        Joint15.RIGHT_KNEE: 22,     # RightLeg
        Joint15.LEFT_ANKLE: 18,     # LeftFoot
        Joint15.RIGHT_ANKLE: 23,    # RightFoot
        Joint15.NECK: 3             # Neck
    }

    @classmethod
    def load_segmentation_records(cls, csv_path: str) -> List[Dict[str, Any]]:
        """REHAB24-6 Segmentation.csv 파싱"""
        records = []
        if not os.path.exists(csv_path):
            return records

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    records.append({
                        'video_id': row['video_id'].strip(),
                        'repetition_number': int(row['repetition_number']),
                        'exercise_id': int(row['exercise_id']),
                        'person_id': int(row['person_id']),
                        'first_frame': int(row['first_frame']),
                        'last_frame': int(row['last_frame']),
                        'mocap_erroneous': int(row.get('mocap_erroneous', 0)),
                        'correctness': int(row['correctness']),
                        'cam17_orientation': row.get('cam17_orientation', 'front').strip()
                    })
                except Exception:
                    continue
        return records

    @classmethod
    def parse_repetition_2d(
        cls,
        npy_path: str,
        start_frame: int,
        end_frame: int,
        exercise_name: str,
        label: int,
        seq_id: str,
        subject_id: str,
        fps: float = 30.0,
        proj_plane: str = 'XY'
    ) -> Optional[KeypointSequence2D]:
        """세션 NPY 파일에서 특정 repetition 구간을 슬라이싱하여 2D 15-Joint KeypointSequence2D 생성"""
        if not os.path.exists(npy_path):
            return None

        try:
            arr = np.load(npy_path) # Shape: (Total_T, 26, 2) or (Total_T, 26, 3/4)
        except Exception:
            return None

        if arr.ndim != 3 or arr.shape[1] != 26:
            return None

        total_t = arr.shape[0]
        # 인덱스 범위 클리핑
        s_idx = max(0, start_frame)
        e_idx = min(total_t, end_frame + 1)
        if s_idx >= e_idx or (e_idx - s_idx) < 4:
            return None

        rep_slice = arr[s_idx:e_idx] # (T_rep, 26, C)
        T_rep = rep_slice.shape[0]

        # 2D 좌표 추출
        if rep_slice.shape[2] >= 3:
            if proj_plane.upper() == 'YZ':
                coords_2d = rep_slice[:, :, [2, 1]]
            else:
                coords_2d = rep_slice[:, :, [0, 1]]
        else:
            coords_2d = rep_slice[:, :, :2]

        kps_15_2d = np.zeros((T_rep, 15, 2), dtype=np.float32)
        for j15_idx, r_idx in cls.REHAB_TO_15_MAP.items():
            kps_15_2d[:, j15_idx, :] = coords_2d[:, r_idx, :]

        return KeypointSequence2D(
            sequence_id=seq_id,
            exercise_name=exercise_name,
            label=label,
            keypoints_2d=kps_15_2d,
            fps=fps,
            subject_id=subject_id,
            source_dataset='REHAB24-6_OptiTrack'
        )
