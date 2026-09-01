import json
import os
import glob
from typing import Optional, List, Dict, Any
import numpy as np
from src.core.skeleton import Joint15
from src.core.types import KeypointSequence

class AIHubAdapter:
    """AI Hub 피트니스 데이터셋 2D/3D JSON 파서 및 15-Joint 변환기"""
    
    AIHUB_24_TO_15_MAP = {
        Joint15.PELVIS: 21,         # Waist (골반 중심)
        Joint15.TORSO: 20,          # Back (등/척추)
        Joint15.LEFT_SHOULDER: 5,   # Left Shoulder
        Joint15.RIGHT_SHOULDER: 6,  # Right Shoulder
        Joint15.LEFT_ELBOW: 7,      # Left Elbow
        Joint15.RIGHT_ELBOW: 8,     # Right Elbow
        Joint15.LEFT_WRIST: 9,      # Left Wrist
        Joint15.RIGHT_WRIST: 10,    # Right Wrist
        Joint15.LEFT_HIP: 11,       # Left Hip
        Joint15.RIGHT_HIP: 12,      # Right Hip
        Joint15.LEFT_KNEE: 13,      # Left Knee
        Joint15.RIGHT_KNEE: 14,     # Right Knee
        Joint15.LEFT_ANKLE: 15,     # Left Ankle
        Joint15.RIGHT_ANKLE: 16,    # Right Ankle
        Joint15.NECK: 17            # Neck
    }

    @classmethod
    def parse_2d_json(cls, json_path: str, min_active_frames: int = 8) -> Optional[KeypointSequence]:
        """2D JSON 파일 로드 및 유효 15-Joint 시퀀스 추출"""
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
        if not frames_list:
            return None
            
        extracted_frames = []
        for f_idx, frame_obj in enumerate(frames_list):
            view_data = frame_obj.get('view1', {})
            active = view_data.get('active', 'Yes')
            pts = view_data.get('pts', {})
            
            if active.lower() in ['yes', 'y', 'true', '1'] and pts:
                frame_kps = np.zeros((15, 3), dtype=np.float32)
                for j15_idx, aihub_idx in cls.AIHUB_24_TO_15_MAP.items():
                    pt_val = pts.get(str(aihub_idx + 1), pts.get(str(aihub_idx), {}))
                    if isinstance(pt_val, dict):
                        x = pt_val.get('x', 0.0)
                        y = pt_val.get('y', 0.0)
                        frame_kps[j15_idx, 0] = float(x)
                        frame_kps[j15_idx, 1] = float(y)
                        frame_kps[j15_idx, 2] = 0.0
                extracted_frames.append(frame_kps)
                
        if len(extracted_frames) < min_active_frames:
            return None
            
        kps_arr = np.stack(extracted_frames, axis=0)
        seq_id = os.path.splitext(os.path.basename(json_path))[0]
        
        parts = seq_id.split('-')
        subject_id = parts[3] if len(parts) > 3 else f"Subj_{parts[0] if parts else '01'}"
        
        return KeypointSequence(
            sequence_id=seq_id,
            exercise_name=exercise_name,
            label=is_correct,
            keypoints=kps_arr,
            fps=30.0,
            conditions=cond_bools,
            subject_id=subject_id,
            source_dataset='AIHub',
            metadata={'key_code': key_code, 'raw_frames': len(extracted_frames)}
        )
