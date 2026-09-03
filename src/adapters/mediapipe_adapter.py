import os
import json
import urllib.request
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.core.skeleton2d import Joint15
from src.core.types import KeypointSequence2D

DEFAULT_MODEL_PATH = "d:/configs/pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

class MediaPipeAdapter:
    """MediaPipe PoseLandmarker 기반 2D 이미지 프레임 파서 및 15-Joint 추출기"""

    # MediaPipe 33 Landmark Index to Joint15 Map (12 Direct Joints)
    DIRECT_MP_TO_15_MAP = {
        11: Joint15.LEFT_SHOULDER,
        12: Joint15.RIGHT_SHOULDER,
        13: Joint15.LEFT_ELBOW,
        14: Joint15.RIGHT_ELBOW,
        15: Joint15.LEFT_WRIST,
        16: Joint15.RIGHT_WRIST,
        23: Joint15.LEFT_HIP,
        24: Joint15.RIGHT_HIP,
        25: Joint15.LEFT_KNEE,
        26: Joint15.RIGHT_KNEE,
        27: Joint15.LEFT_ANKLE,
        28: Joint15.RIGHT_ANKLE,
    }

    _detector: Optional[vision.PoseLandmarker] = None
    _model_path: str = DEFAULT_MODEL_PATH

    @classmethod
    def get_detector(cls, model_path: Optional[str] = None) -> vision.PoseLandmarker:
        """PoseLandmarker 인스턴스 반환 (필요 시 모델 자동 다운로드)"""
        target_path = model_path or cls._model_path
        if cls._detector is None or cls._model_path != target_path:
            if not os.path.exists(target_path):
                # 루트 디렉토리에 이미 다운로드된 파일이 있는지 확인
                root_fallback = "d:/pose_landmarker_heavy.task"
                if os.path.exists(root_fallback):
                    target_path = root_fallback
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    print(f"[*] Downloading MediaPipe Pose model to {target_path}...")
                    urllib.request.urlretrieve(MODEL_URL, target_path)
            
            base_options = python.BaseOptions(model_asset_path=target_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            cls._detector = vision.PoseLandmarker.create_from_options(options)
            cls._model_path = target_path
        return cls._detector

    @classmethod
    def extract_frame_landmarks(cls, img_path: str, detector: Optional[vision.PoseLandmarker] = None) -> Tuple[np.ndarray, np.ndarray]:
        """단일 이미지 파일에서 (15, 2) 픽셀 좌표 및 (15,) 신뢰도(visibility) 추출"""
        if not os.path.exists(img_path):
            return np.zeros((15, 2), dtype=np.float32), np.zeros((15,), dtype=np.float32)

        # Windows 한글 경로 대응 이미지 로드
        img_bytes = np.fromfile(img_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return np.zeros((15, 2), dtype=np.float32), np.zeros((15,), dtype=np.float32)

        h, w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        det = detector or cls.get_detector()
        result = det.detect(mp_image)

        frame_kps = np.zeros((15, 2), dtype=np.float32)
        frame_vis = np.zeros((15,), dtype=np.float32)

        if not result.pose_landmarks:
            return frame_kps, frame_vis

        landmarks = result.pose_landmarks[0]
        pts_33 = np.zeros((33, 2), dtype=np.float32)
        vis_33 = np.zeros((33,), dtype=np.float32)

        for i, lm in enumerate(landmarks):
            pts_33[i, 0] = lm.x * float(w)
            pts_33[i, 1] = lm.y * float(h)
            vis_33[i] = float(lm.visibility)

        # 1. 12개 직접 관절 매핑
        for mp_idx, j15_idx in cls.DIRECT_MP_TO_15_MAP.items():
            frame_kps[j15_idx] = pts_33[mp_idx]
            frame_vis[j15_idx] = vis_33[mp_idx]

        # 2. 3개 파생 중점 관절 계산
        # 1) PELVIS: 양 고관절(23, 24) 중점
        pelvis_pt = (pts_33[23] + pts_33[24]) / 2.0
        pelvis_vis = min(vis_33[23], vis_33[24])
        frame_kps[Joint15.PELVIS] = pelvis_pt
        frame_vis[Joint15.PELVIS] = pelvis_vis

        # 2) NECK: 양 어깨(11, 12) 중점
        neck_pt = (pts_33[11] + pts_33[12]) / 2.0
        neck_vis = min(vis_33[11], vis_33[12])
        frame_kps[Joint15.NECK] = neck_pt
        frame_vis[Joint15.NECK] = neck_vis

        # 3) TORSO: 체간 중심 (PELVIS와 NECK의 중점)
        torso_pt = (pelvis_pt + neck_pt) / 2.0
        torso_vis = min(pelvis_vis, neck_vis)
        frame_kps[Joint15.TORSO] = torso_pt
        frame_vis[Joint15.TORSO] = torso_vis

        return frame_kps, frame_vis

    @classmethod
    def parse_sequence_from_images(
        cls,
        seq_dir: str,
        view_prefix: str = "시점A",
        min_frames: int = 4,
        json_path: Optional[str] = None,
        detector: Optional[vision.PoseLandmarker] = None
    ) -> Optional[KeypointSequence2D]:
        """시퀀스 이미지 폴더에서 특정 시점의 프레임들을 정렬 순회하여 KeypointSequence2D 객체 생성"""
        if not os.path.exists(seq_dir):
            return None

        # os.listdir로 안전하게 검색 및 정렬 (특수문자 대괄호 이슈 방지)
        all_files = os.listdir(seq_dir)
        filtered = [f for f in all_files if view_prefix in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        filtered.sort()
        img_paths = [os.path.join(seq_dir, f) for f in filtered]

        if len(img_paths) < min_frames:
            # 시점 접두사가 없는 경우 전체 이미지에서 탐색
            filtered = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            filtered.sort()
            img_paths = [os.path.join(seq_dir, f) for f in filtered]
            if len(img_paths) < min_frames:
                return None

        # 메타데이터 파싱
        seq_id = os.path.basename(os.path.normpath(seq_dir))
        exercise_name = os.path.basename(os.path.dirname(os.path.normpath(seq_dir)))
        
        is_correct = 1 if "[정상]" in seq_id or "정상" in seq_id else 0
        cond_bools = []
        subject_id = "Subj_Unknown"
        key_code = ""

        # 동일 시퀀스의 JSON 파일이 있으면 상세 라벨/조건 정보 상속
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    jdata = json.load(f)
                type_info = jdata.get('type_info', {})
                if 'exercise' in type_info:
                    exercise_name = type_info['exercise']
                key_code = str(type_info.get('key', ''))
                pose = type_info.get('pose', '')
                raw_conditions = type_info.get('conditions', [])
                for c in raw_conditions:
                    if isinstance(c, dict):
                        cond_bools.append(bool(c.get('value', False)))
                    elif isinstance(c, bool):
                        cond_bools.append(c)
                if cond_bools:
                    is_correct = int(all(cond_bools))
                else:
                    is_correct = 1 if pose in ['Normal', '정자세'] else 0
            except Exception:
                pass

        det = detector or cls.get_detector()
        extracted_frames = []
        vis_frames = []

        for p in img_paths:
            kps, vis = cls.extract_frame_landmarks(p, detector=det)
            extracted_frames.append(kps)
            vis_frames.append(vis)

        if len(extracted_frames) < min_frames:
            return None

        kps_arr = np.stack(extracted_frames, axis=0) # (T, 15, 2)
        vis_arr = np.stack(vis_frames, axis=0)       # (T, 15)

        # 피험자 ID 파싱
        parts = seq_id.split('_')
        for part in parts:
            if '피험자' in part:
                subject_id = part
                break

        return KeypointSequence2D(
            sequence_id=seq_id,
            exercise_name=exercise_name,
            label=is_correct,
            keypoints_2d=kps_arr,
            fps=30.0,
            conditions=cond_bools,
            subject_id=subject_id,
            source_dataset="MediaPipe",
            metadata={
                'key_code': key_code,
                'raw_frames': len(extracted_frames),
                'view': view_prefix,
                'visibility_mean': float(np.mean(vis_arr))
            }
        )
