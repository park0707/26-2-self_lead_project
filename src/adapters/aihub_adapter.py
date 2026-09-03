import json
import os
from typing import Optional, List, Dict, Any
import numpy as np
from src.core.skeleton2d import Joint15
from src.core.types import KeypointSequence2D

class AIHubAdapter:
    """AI Hub 피트니스 2D JSON 파서 및 15-Joint 추출기"""

    AIHUB_NAME_TO_15_MAP = {
        'Waist': Joint15.PELVIS,
        'Back': Joint15.TORSO,
        'Left Shoulder': Joint15.LEFT_SHOULDER,
        'Right Shoulder': Joint15.RIGHT_SHOULDER,
        'Left Elbow': Joint15.LEFT_ELBOW,
        'Right Elbow': Joint15.RIGHT_ELBOW,
        'Left Wrist': Joint15.LEFT_WRIST,
        'Right Wrist': Joint15.RIGHT_WRIST,
        'Left Hip': Joint15.LEFT_HIP,
        'Right Hip': Joint15.RIGHT_HIP,
        'Left Knee': Joint15.LEFT_KNEE,
        'Right Knee': Joint15.RIGHT_KNEE,
        'Left Ankle': Joint15.LEFT_ANKLE,
        'Right Ankle': Joint15.RIGHT_ANKLE,
        'Neck': Joint15.NECK
    }

    AIHUB_INDEX_TO_15_MAP = {
        21: Joint15.PELVIS,         # Waist
        20: Joint15.TORSO,          # Back
        5: Joint15.LEFT_SHOULDER,   # Left Shoulder
        6: Joint15.RIGHT_SHOULDER,  # Right Shoulder
        7: Joint15.LEFT_ELBOW,      # Left Elbow
        8: Joint15.RIGHT_ELBOW,     # Right Elbow
        9: Joint15.LEFT_WRIST,      # Left Wrist
        10: Joint15.RIGHT_WRIST,    # Right Wrist
        11: Joint15.LEFT_HIP,       # Left Hip
        12: Joint15.RIGHT_HIP,      # Right Hip
        13: Joint15.LEFT_KNEE,      # Left Knee
        14: Joint15.RIGHT_KNEE,     # Right Knee
        15: Joint15.LEFT_ANKLE,     # Left Ankle
        16: Joint15.RIGHT_ANKLE,    # Right Ankle
        17: Joint15.NECK            # Neck
    }

    @classmethod
    def parse_2d_json(cls, json_path: str, min_frames: int = 4, view_key: str = 'view1') -> Optional[KeypointSequence2D]:
        """AI Hub 2D JSON 파일을 파싱하여 특정 시점(view1~view5)의 KeypointSequence2D 객체 생성"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return None

        type_info = data.get('type_info', {})
        exercise_name = type_info.get('exercise', '')
        key_code = str(type_info.get('key', ''))
        pose = type_info.get('pose', '')
        raw_conditions = type_info.get('conditions', [])

        cond_bools = []
        for c in raw_conditions:
            if isinstance(c, dict):
                cond_bools.append(bool(c.get('value', False)))
            elif isinstance(c, bool):
                cond_bools.append(c)

        if cond_bools:
            is_correct = int(all(cond_bools))
        else:
            is_correct = 1 if pose in ['Normal', '정자세'] else 0

        frames_list = data.get('frames', [])
        if not frames_list or len(frames_list) < min_frames:
            return None

        extracted_frames = []
        for frame_obj in frames_list:
            view_data = frame_obj.get(view_key, {})
            pts = view_data.get('pts', {})
            frame_kps = np.zeros((15, 2), dtype=np.float32)

            if isinstance(pts, dict):
                for pt_name, pt_val in pts.items():
                    if isinstance(pt_val, dict):
                        x = float(pt_val.get('x', 0.0))
                        y = float(pt_val.get('y', 0.0))
                        
                        # 1) 이름 기반 매핑
                        if pt_name in cls.AIHUB_NAME_TO_15_MAP:
                            j15_idx = cls.AIHUB_NAME_TO_15_MAP[pt_name]
                            frame_kps[j15_idx, 0] = x
                            frame_kps[j15_idx, 1] = y
                        # 2) 숫자 인덱스 기반 매핑
                        elif pt_name.isdigit():
                            idx_num = int(pt_name)
                            if idx_num in cls.AIHUB_INDEX_TO_15_MAP:
                                j15_idx = cls.AIHUB_INDEX_TO_15_MAP[idx_num]
                                frame_kps[j15_idx, 0] = x
                                frame_kps[j15_idx, 1] = y
                            elif (idx_num - 1) in cls.AIHUB_INDEX_TO_15_MAP:
                                j15_idx = cls.AIHUB_INDEX_TO_15_MAP[idx_num - 1]
                                frame_kps[j15_idx, 0] = x
                                frame_kps[j15_idx, 1] = y

            extracted_frames.append(frame_kps)

        if len(extracted_frames) < min_frames:
            return None

        kps_arr = np.stack(extracted_frames, axis=0) # (T, 15, 2)
        seq_id = os.path.splitext(os.path.basename(json_path))[0]

        parts = seq_id.split('-')
        subject_id = parts[3] if len(parts) > 3 else f"Subj_{parts[0] if parts else '01'}"

        return KeypointSequence2D(
            sequence_id=seq_id,
            exercise_name=exercise_name,
            label=is_correct,
            keypoints_2d=kps_arr,
            fps=30.0,
            conditions=cond_bools,
            subject_id=subject_id,
            source_dataset='AIHub',
            metadata={'key_code': key_code, 'raw_frames': len(extracted_frames), 'view': view_key}
        )
