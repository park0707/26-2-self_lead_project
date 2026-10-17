# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/pipeline/spinepose_tracker.py
Purpose: 웹 서빙 전용 SpinePose 고속 유도형 척추 9분절 추적기 (SpinePoseTracker)
         - 방식 3: 1회 반복(1-Repetition) 구간 내 핵심 16프레임 균등 샘플링 + MediaPipe BBox 유도 추론 + 64F 3차 스플라인 보간
         - MediaPipe 2D 관절점으로부터 사람 BBox를 즉각 산출하여 무거운 RF-DETR 탐지기(1,400ms) 완전 스킵
         - DFKI SpinePose (Small) + 3D Lifter를 통해 9대 척추 분절(S1, L5, L3, L1, T8, T3, C7, C4, C1) 3D 좌표 추출
         - 1회 반복당 약 2.7초(16F x 170ms)의 초경량 레이턴시로 척추 중립(c1) 모델에 정밀 3D 척추 피처 제공
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import cv2
import numpy as np
from scipy.interpolate import interp1d

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from spinepose import SpinePoseEstimator
except ImportError:
    SpinePoseEstimator = None

logger = logging.getLogger("spinepose_tracker")

# SpinePose 공식 37개 랜드마크 중 척추 체인 9개 인덱스
# 19: S1 (hip/sacrum), 26: L5, 27: L3, 28: L1, 29: T8, 30: T3, 18: C7 (neck), 35: C4, 36: C1
SPINEPOSE_9_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]
SPINEPOSE_9_NAMES = ['S1', 'L5', 'L3', 'L1', 'T8', 'T3', 'C7', 'C4', 'C1']


class SpinePoseTracker:
    """SpinePose 9대 척추 분절 고속 유도형 3D 추적기 (싱글톤)"""

    _instance = None

    def __init__(self, mode: str = "small", enable_lifting: bool = True, device: str = "cpu"):
        self.mode = mode
        self.enable_lifting = enable_lifting
        self.device = device
        self._estimator = None
        self._is_initialized = False

    @classmethod
    def get_instance(cls) -> "SpinePoseTracker":
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_estimator(self):
        """SpinePoseEstimator 지연 초기화 (Lazy Init)"""
        if self._estimator is None:
            if SpinePoseEstimator is None:
                logger.error("spinepose 라이브러리가 설치되어 있지 않습니다.")
                return None
            logger.info("SpinePoseEstimator (Small + 3D Lifter) 초기화 중...")
            t0 = time.time()
            self._estimator = SpinePoseEstimator(
                mode=self.mode,
                enable_lifting=self.enable_lifting
            )
            self._is_initialized = True
            logger.info(f"SpinePoseEstimator 초기화 완료 ({time.time() - t0:.2f}초 소요)")
        return self._estimator

    @staticmethod
    def _compute_bbox_from_mp_landmarks(
        mp_2d: np.ndarray,
        frame_w: int,
        frame_h: int,
        padding: float = 1.25
    ) -> np.ndarray:
        """
        MediaPipe 2D 관절점((33, 2) 픽셀)으로부터 사람 BBox를 즉시 계산하여
        무거운 RF-DETR 사람 탐지기(1,400ms) 연산을 0ms로 스킵
        """
        valid = (mp_2d[:, 0] > 0) & (mp_2d[:, 1] > 0)
        if np.sum(valid) < 4:
            return np.array([0, 0, frame_w, frame_h], dtype=np.float32)

        x_min, y_min = np.min(mp_2d[valid, 0]), np.min(mp_2d[valid, 1])
        x_max, y_max = np.max(mp_2d[valid, 0]), np.max(mp_2d[valid, 1])

        bw = x_max - x_min
        bh = y_max - y_min
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        new_bw = bw * padding
        new_bh = bh * padding

        b_x1 = max(0.0, cx - new_bw / 2.0)
        b_y1 = max(0.0, cy - new_bh / 2.0)
        b_x2 = min(float(frame_w), cx + new_bw / 2.0)
        b_y2 = min(float(frame_h), cy + new_bh / 2.0)

        return np.array([b_x1, b_y1, b_x2, b_y2], dtype=np.float32)

    @staticmethod
    def interpolate_seq_to_64f(kps_arr: np.ndarray) -> np.ndarray:
        """
        (T, J, C) 관절 배열을 시간축 64프레임으로 3차 스플라인 보간 (규정 준수)
        """
        T, J, C = kps_arr.shape
        if T == 64:
            return kps_arr.astype(np.float32)

        t_orig = np.linspace(0, 1, T)
        t_64 = np.linspace(0, 1, 64)
        kind = 'cubic' if T >= 4 else 'linear'

        flat_orig = kps_arr.reshape(T, -1)
        f = interp1d(t_orig, flat_orig, kind=kind, axis=0, fill_value='extrapolate')
        flat_64 = f(t_64)
        return flat_64.reshape(64, J, C).astype(np.float32)

    def extract_repetition_spine_64f(
        self,
        frames_source: Any,  # List[np.ndarray] (캐시된 프레임 리스트) 또는 str (비디오 경로)
        start_frame: int,
        end_frame: int,
        mp_kps_33_2d: np.ndarray,
        sample_stride: int = 2,
        num_sample_frames: int = 16,
        orig_w: int = 1920,
        orig_h: int = 1080,
        scale: float = 1.0,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        [방식 3] 1회 반복 구간에서 핵심 16프레임을 균등 샘플링하여 SpinePose 3D 추론 후 64프레임 3차 스플라인 보간
        
        반환:
          - spine_9_3d_64f: (64, 9, 3) 3D 척추 좌표 시퀀스 (c1 모델 입력용)
          - spine_9_2d_norm_64f: (64, 9, 3) [norm_x, norm_y, score] 2D 정규화 척추 좌표 (오버레이 뷰어용)
        """
        estimator = self._get_estimator()
        if estimator is None:
            logger.warning("[SpinePoseTracker] 추론기 미초기화: 더미 제로 시퀀스를 반환합니다.")
            dummy_3d = np.zeros((64, 9, 3), dtype=np.float32)
            dummy_2d = np.zeros((64, 9, 3), dtype=np.float32)
            return dummy_3d, dummy_2d

        # 1. 1회 반복 구간 내 핵심 N프레임 균등 인덱스 결정
        rep_len = max(1, end_frame - start_frame + 1)
        k = min(num_sample_frames, rep_len)
        sampled_indices = np.linspace(start_frame, end_frame, k).round().astype(int)

        raw_spine_9_3d = []
        raw_spine_9_2d = []

        is_cached_list = isinstance(frames_source, list)
        cap = None
        if not is_cached_list:
            cap = cv2.VideoCapture(str(frames_source))

        try:
            for i, frame_idx in enumerate(sampled_indices):
                # 프레임 획득
                frame = None
                if is_cached_list:
                    if 0 <= frame_idx < len(frames_source):
                        frame = frames_source[frame_idx]
                else:
                    if cap and cap.isOpened():
                        cap_pos = frame_idx * sample_stride
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cap_pos)
                        ret, f_read = cap.read()
                        if ret:
                            frame = f_read

                if frame is None:
                    # 프레임 로드 실패 시 마지막 프레임 또는 더미 복제
                    if len(raw_spine_9_3d) > 0:
                        raw_spine_9_3d.append(raw_spine_9_3d[-1].copy())
                        raw_spine_9_2d.append(raw_spine_9_2d[-1].copy())
                    else:
                        raw_spine_9_3d.append(np.zeros((9, 3), dtype=np.float32))
                        raw_spine_9_2d.append(np.zeros((9, 3), dtype=np.float32))
                    continue

                fh, fw = frame.shape[:2]

                # MediaPipe 2D 관절점으로부터 BBox 계산
                if 0 <= frame_idx < len(mp_kps_33_2d):
                    mp_2d_cur = mp_kps_33_2d[frame_idx].copy()
                    # 프레임 크기 scale에 맞춰 관절 좌표 스케일링
                    if scale != 1.0:
                        mp_2d_cur = mp_2d_cur * scale
                    raw_bbox = self._compute_bbox_from_mp_landmarks(mp_2d_cur, fw, fh, padding=1.25)
                else:
                    raw_bbox = np.array([0, 0, fw, fh], dtype=np.float32)

                bbox_input = np.array([raw_bbox], dtype=np.float32)

                # SpinePose 추론 (BBox 주입으로 탐지기 생략)
                try:
                    res = estimator.estimate(frame, bbox_input)
                    # res: (kps_2d, scores, kps_3d)
                    kps_2d = res[0][0]  # (37, 2)
                    scores = res[1][0]  # (37,)
                    kps_3d = res[2][0] if len(res) > 2 else np.zeros((37, 3), dtype=np.float32)

                    # 척추 9개 분절 슬라이싱
                    s9_3d = kps_3d[SPINEPOSE_9_INDICES]  # (9, 3)
                    s9_2d_px = kps_2d[SPINEPOSE_9_INDICES]  # (9, 2)
                    s9_scores = scores[SPINEPOSE_9_INDICES]  # (9,)

                    # 2D 좌표 정규화 (0.0 ~ 1.0)
                    s9_2d_norm = s9_2d_px.copy()
                    s9_2d_norm[:, 0] /= max(1.0, float(fw))
                    s9_2d_norm[:, 1] /= max(1.0, float(fh))
                    s9_2d_combined = np.column_stack([s9_2d_norm, s9_scores])  # (9, 3)

                    raw_spine_9_3d.append(s9_3d)
                    raw_spine_9_2d.append(s9_2d_combined)
                except Exception as e:
                    logger.warning(f"[SpinePoseTracker] 프레임 {frame_idx} 추론 중 오류: {e}")
                    if len(raw_spine_9_3d) > 0:
                        raw_spine_9_3d.append(raw_spine_9_3d[-1].copy())
                        raw_spine_9_2d.append(raw_spine_9_2d[-1].copy())
                    else:
                        raw_spine_9_3d.append(np.zeros((9, 3), dtype=np.float32))
                        raw_spine_9_2d.append(np.zeros((9, 3), dtype=np.float32))

                if progress_callback:
                    pct = float(i + 1) / float(k) * 100.0
                    progress_callback("extracting_spinepose", pct, f"SpinePose 16F 척추 추론 중... ({i+1}/{k})")

        finally:
            if cap is not None:
                cap.release()

        arr_raw_3d = np.array(raw_spine_9_3d, dtype=np.float32)  # (K, 9, 3)
        arr_raw_2d = np.array(raw_spine_9_2d, dtype=np.float32)  # (K, 9, 3)

        # 2. 64프레임 3차 스플라인 보간 표준 엄격 적용
        spine_9_3d_64f = self.interpolate_seq_to_64f(arr_raw_3d)
        spine_9_2d_norm_64f = self.interpolate_seq_to_64f(arr_raw_2d)

        return spine_9_3d_64f, spine_9_2d_norm_64f
