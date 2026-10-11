# -*- coding: utf-8 -*-
"""
MediaPipe 3D Pose Heavy 기반 비디오 프레임 단위 15-Joint 월드 좌표 및 1-Euro 스무더
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from typing import Tuple, Dict, Any, Optional

from src.adapters.mediapipe_adapter import MediaPipeAdapter
from src.core.skeleton2d import Joint15

class OneEuroFilter:
    """프레임 지터 억제용 1-Euro Filter"""
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.005, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.copy(x0).astype(np.float32)
        self.dx_prev = np.zeros_like(x0, dtype=np.float32) + dx0
        self.t_prev = float(t0)

    def alpha(self, rate, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 1e-5:
            return self.x_prev
        rate = 1.0 / t_e

        # 속도 추정 및 필터링
        dx = (x - self.x_prev) / t_e
        edx = self.alpha(rate, self.d_cutoff)
        dx_hat = edx * dx + (1.0 - edx) * self.dx_prev

        # 위치 컷오프 계산
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        ex = self.alpha(rate, cutoff)
        x_hat = ex * x + (1.0 - ex) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class YouTubePoseExtractor:
    """비디오 파일로부터 15-Joint 3D 월드 좌표(미터) 초고속 추출기 (RunningMode.VIDEO)"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "d:/configs/pose_landmarker_heavy.task"
        if not os.path.exists(self.model_path):
            fallback = "d:/pose_landmarker_heavy.task"
            if os.path.exists(fallback):
                self.model_path = fallback
            else:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                MediaPipeAdapter.get_detector(self.model_path) # Auto download fallback

    def extract_from_video(
        self,
        video_path: str,
        apply_smoothing: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        비디오 파일에서 프레임별 (15, 3) 3D 월드 좌표 및 (15,) 가시성 고속 추출
        반환: (kps_3d, visibilities, fps)
              kps_3d: (T, 15, 3)
              visibilities: (T, 15)
              fps: float
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            num_poses=1
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        frames_3d = []
        frames_vis = []
        filter_1euro = None

        frame_stride = max(1, int(round(fps / 30.0)))
        effective_fps = fps / frame_stride

        frame_idx = 0
        raw_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if raw_idx % frame_stride != 0:
                    raw_idx += 1
                    continue

                h, w, _ = frame.shape
                if w > 640:
                    new_h = int(h * (640.0 / w))
                    frame = cv2.resize(frame, (640, new_h))

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

                timestamp_ms = int(raw_idx * 1000.0 / fps)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                frame_kps_3d = np.zeros((15, 3), dtype=np.float32)
                frame_vis = np.zeros((15,), dtype=np.float32)

                if result.pose_landmarks and result.pose_world_landmarks:
                    landmarks_2d = result.pose_landmarks[0]
                    world_lms = result.pose_world_landmarks[0]

                    pts_33_3d = np.zeros((33, 3), dtype=np.float32)
                    vis_33 = np.zeros((33,), dtype=np.float32)

                    for i in range(len(world_lms)):
                        pts_33_3d[i, 0] = world_lms[i].x
                        pts_33_3d[i, 1] = world_lms[i].y
                        pts_33_3d[i, 2] = world_lms[i].z
                        vis_33[i] = float(landmarks_2d[i].visibility)

                    # 12개 직접 관절 매핑
                    for mp_idx, j15_idx in MediaPipeAdapter.DIRECT_MP_TO_15_MAP.items():
                        frame_kps_3d[j15_idx] = pts_33_3d[mp_idx]
                        frame_vis[j15_idx] = vis_33[mp_idx]

                    # 3개 파생 중점 관절 계산
                    pelvis_pt = (pts_33_3d[23] + pts_33_3d[24]) / 2.0
                    neck_pt = (pts_33_3d[11] + pts_33_3d[12]) / 2.0
                    torso_pt = (pelvis_pt + neck_pt) / 2.0

                    frame_kps_3d[Joint15.PELVIS] = pelvis_pt
                    frame_vis[Joint15.PELVIS] = min(vis_33[23], vis_33[24])

                    frame_kps_3d[Joint15.NECK] = neck_pt
                    frame_vis[Joint15.NECK] = min(vis_33[11], vis_33[12])

                    frame_kps_3d[Joint15.TORSO] = torso_pt
                    frame_vis[Joint15.TORSO] = min(frame_vis[Joint15.PELVIS], frame_vis[Joint15.NECK])

                # 1-Euro Filter 적용
                t = frame_idx / fps
                if apply_smoothing:
                    if filter_1euro is None:
                        filter_1euro = OneEuroFilter(t0=t, x0=frame_kps_3d, min_cutoff=1.0, beta=0.005)
                        smoothed_3d = frame_kps_3d
                    else:
                        smoothed_3d = filter_1euro.filter(t, frame_kps_3d)
                    frames_3d.append(smoothed_3d)
                else:
                    frames_3d.append(frame_kps_3d)

                frames_vis.append(frame_vis)
                frame_idx += 1
                raw_idx += 1
        finally:
            cap.release()
            detector.close()

        if len(frames_3d) == 0:
            return np.zeros((0, 15, 3), dtype=np.float32), np.zeros((0, 15), dtype=np.float32), fps

        kps_3d = np.array(frames_3d, dtype=np.float32)
        visibilities = np.array(frames_vis, dtype=np.float32)
        return kps_3d, visibilities, effective_fps
