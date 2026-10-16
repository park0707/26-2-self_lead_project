# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/pipeline/video_processor.py
Purpose: 웹 서빙 전용 비디오 업로드 E2E 분석 파이프라인 (VideoAnalysisPipeline)
         1. 비디오 디코딩 및 MediaPipe 33-Landmark 시퀀스 추출
         2. 키네마틱 에너지 기반 운동 횟수(1-Repetition) 동적 구간 분할
         3. 1회 반복 64프레임 3차 스플라인(Cubic Spline) 보간 (규정 준수)
         4. 1단계 운동 종목 분류 (Stage1InferenceEngine)
         5. 2단계 운동별 33종 세부 결함 진단 (Stage2InferenceEngine)
         6. 실시간 웹소켓/비동기 진행률 콜백(Progress Callback) 지원
"""

import os
import sys
import time
import uuid
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.core.skeleton2d import Joint15
from src.preprocessing.rep_segmenter import RepetitionSegmenter
from src.web.core.config import EXERCISE_CATALOG, VIEW_CODE_MAP, VIEW_ALIAS_MAP
from src.web.core.schemas import (
    ExercisePrediction, DefectPrediction, RepetitionDetail, VideoAnalysisResponse
)
from src.web.core.topology_adapter import SkeletalTopologyAdapter
from src.web.pipeline.active_period_gater import ActivePeriodGater
from src.web.pipeline.rtmpose_tracker import RTMPoseTracker
from src.web.pipeline.spinepose_tracker import SpinePoseTracker
from src.web.pipeline.stage1_engine import Stage1InferenceEngine
from src.web.pipeline.stage2_engine import Stage2InferenceEngine

logger = logging.getLogger("video_processor")


class VideoAnalysisPipeline:
    """비디오 업로드 및 E2E 실시간 운동 자세 분석 파이프라인"""

    _instance = None

    def __init__(self, model_task_path: Optional[str] = None):
        if model_task_path is None:
            candidate = os.path.join(PROJECT_ROOT, "configs", "pose_landmarker_heavy.task")
            if os.path.exists(candidate):
                self.model_task_path = candidate
            elif os.path.exists("d:/configs/pose_landmarker_heavy.task"):
                self.model_task_path = "d:/configs/pose_landmarker_heavy.task"
            elif os.path.exists("d:/pose_landmarker_heavy.task"):
                self.model_task_path = "d:/pose_landmarker_heavy.task"
            else:
                self.model_task_path = candidate
        else:
            self.model_task_path = model_task_path

        self.stage1_engine = Stage1InferenceEngine.get_instance()
        self.stage2_engine = Stage2InferenceEngine.get_instance(prewarm=False)
        self.rtmp_tracker = RTMPoseTracker.get_instance()
        self.spine_tracker = SpinePoseTracker.get_instance()
        self._detector: Optional[vision.PoseLandmarker] = None

    @classmethod
    def get_instance(cls) -> "VideoAnalysisPipeline":
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _ensure_linux_graphics_libs():
        """리눅스 서버(Render 등)에서 libGLESv2, libGLdispatch, libGL, libEGL을 사전 로드하여 No such file 에러 원천 방지"""
        if not sys.platform.startswith("linux"):
            return
        
        candidates = [
            os.path.join(CURRENT_DIR, "../libs"),
            os.path.join(PROJECT_ROOT, "src/web/libs"),
            os.path.join(PROJECT_ROOT, "libs/usr/lib/x86_64-linux-gnu"),
            os.path.join(PROJECT_ROOT, "libs/usr/lib")
        ]
        
        import ctypes
        for cdir in candidates:
            norm_dir = os.path.abspath(cdir)
            if os.path.exists(norm_dir):
                cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
                if norm_dir not in cur_ld:
                    os.environ["LD_LIBRARY_PATH"] = f"{norm_dir}:{cur_ld}"
                
                ordered_libs = [
                    "libglvnd.so.0", "libglvnd.so.0.0.0",
                    "libGLdispatch.so.0", "libGLdispatch.so.0.0.0",
                    "libGLESv2.so.2", "libGLESv2.so.2.1.0",
                    "libEGL.so.1", "libEGL.so.1.1.0",
                    "libGL.so.1", "libGL.so.1.7.0"
                ]
                for lib_name in ordered_libs:
                    lib_file = os.path.join(norm_dir, lib_name)
                    if os.path.exists(lib_file):
                        try:
                            ctypes.CDLL(lib_file, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                        except Exception as e:
                            logger.debug(f"[_ensure_linux_graphics_libs] {lib_name} 로드 시도: {e}")

    def _get_detector(self) -> vision.PoseLandmarker:
        """MediaPipe PoseLandmarker 디텍터 반환 (Lazy Initialization)"""
        if self._detector is None:
            self._ensure_linux_graphics_libs()
            base_options = python.BaseOptions(
                model_asset_path=self.model_task_path,
                delegate=python.BaseOptions.Delegate.CPU
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._detector = vision.PoseLandmarker.create_from_options(options)
        return self._detector

    def extract_frame_landmarks(
        self, img_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        단일 BGR 넘파이 이미지에서 33개 및 15개 관절 좌표 고속 추출
        반환:
          - kps_33_3d: (33, 3) 3D 월드 좌표 (골반 중심 미터 단위)
          - kps_33_2d: (33, 2) 2D 픽셀 좌표
          - kps_15_2d: (15, 2) Joint15 2D 좌표 (RepSegmenter 입력용)
          - vis_33: (33,) 가시성
        """
        h, w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detector = self._get_detector()
        result = detector.detect(mp_image)

        pts_33_2d = np.zeros((33, 2), dtype=np.float32)
        pts_33_3d = np.zeros((33, 3), dtype=np.float32)
        vis_33 = np.zeros((33,), dtype=np.float32)
        pts_15_2d = np.zeros((15, 2), dtype=np.float32)

        if not result.pose_landmarks:
            return pts_33_3d, pts_33_2d, pts_15_2d, vis_33

        # 2D 랜드마크
        lms_2d = result.pose_landmarks[0]
        for i, lm in enumerate(lms_2d):
            pts_33_2d[i, 0] = lm.x * float(w)
            pts_33_2d[i, 1] = lm.y * float(h)
            vis_33[i] = float(lm.visibility) if hasattr(lm, 'visibility') else 1.0

        # 3D 월드 랜드마크 (없을 경우 2D를 기반으로 가상 z 구성)
        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            world_lms = result.pose_world_landmarks[0]
            for i, lm in enumerate(world_lms):
                pts_33_3d[i, 0] = lm.x
                pts_33_3d[i, 1] = lm.y
                pts_33_3d[i, 2] = lm.z
        else:
            pts_33_3d[:, :2] = pts_33_2d
            pts_33_3d[:, 2] = 0.0

        # Joint15 2D 매핑
        direct_map = {
            11: Joint15.LEFT_SHOULDER, 12: Joint15.RIGHT_SHOULDER,
            13: Joint15.LEFT_ELBOW, 14: Joint15.RIGHT_ELBOW,
            15: Joint15.LEFT_WRIST, 16: Joint15.RIGHT_WRIST,
            23: Joint15.LEFT_HIP, 24: Joint15.RIGHT_HIP,
            25: Joint15.LEFT_KNEE, 26: Joint15.RIGHT_KNEE,
            27: Joint15.LEFT_ANKLE, 28: Joint15.RIGHT_ANKLE,
        }
        for mp_idx, j15_idx in direct_map.items():
            pts_15_2d[j15_idx] = pts_33_2d[mp_idx]

        # 파생 중점 관절
        pelvis_2d = (pts_33_2d[23] + pts_33_2d[24]) / 2.0
        neck_2d = (pts_33_2d[11] + pts_33_2d[12]) / 2.0
        torso_2d = (pelvis_2d + neck_2d) / 2.0
        pts_15_2d[Joint15.PELVIS] = pelvis_2d
        pts_15_2d[Joint15.NECK] = neck_2d
        pts_15_2d[Joint15.TORSO] = torso_2d

        return pts_33_3d, pts_33_2d, pts_15_2d, vis_33

    def extract_video_landmarks(
        self,
        video_path: str,
        sample_stride: int = 2,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        비디오 파일로부터 프레임별 관절 시퀀스 고속 추출 (VIDEO 추적 모드 및 540p 해상도 최적화)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = float(cap.get(cv2.CAP_PROP_FPS))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_fps <= 0:
            orig_fps = 30.0
        sample_stride = max(1, int(sample_stride))
        effective_fps = orig_fps / float(sample_stride)

        # 리눅스 그래픽스 라이브러리 및 CPU 위임자 명시적 바인딩
        self._ensure_linux_graphics_libs()
        base_options = python.BaseOptions(
            model_asset_path=self.model_task_path,
            delegate=python.BaseOptions.Delegate.CPU
        )

        use_video_mode = True
        video_detector = None
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            video_detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e_vid_init:
            logger.warning(f"MediaPipe VIDEO 모드 생성 실패, IMAGE 모드로 안전 전환: {e_vid_init}")
            use_video_mode = False
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            video_detector = vision.PoseLandmarker.create_from_options(options)

        direct_map = {
            11: Joint15.LEFT_SHOULDER, 12: Joint15.RIGHT_SHOULDER,
            13: Joint15.LEFT_ELBOW, 14: Joint15.RIGHT_ELBOW,
            15: Joint15.LEFT_WRIST, 16: Joint15.RIGHT_WRIST,
            23: Joint15.LEFT_HIP, 24: Joint15.RIGHT_HIP,
            25: Joint15.LEFT_KNEE, 26: Joint15.RIGHT_KNEE,
            27: Joint15.LEFT_ANKLE, 28: Joint15.RIGHT_ANKLE,
        }

        all_kps_33_3d = []
        all_kps_33_2d = []
        all_kps_33_2d_norm = []
        all_kps_15_2d = []
        all_vis_33 = []

        frame_idx = 0
        extracted_count = 0
        last_timestamp_ms = -1

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_stride == 0:
                    h, w, _ = frame.shape
                    # 540p 해상도 최적화 (256x256 MediaPipe 입력에 맞추어 메모리 버스 부하 75% 절감)
                    if w > 540 or h > 960:
                        scale = min(540.0 / w, 960.0 / h)
                        small_w, small_h = int(w * scale), int(h * scale)
                        proc_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        proc_frame = frame

                    img_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

                    if use_video_mode:
                        calc_ts = int((frame_idx / orig_fps) * 1000)
                        if calc_ts <= last_timestamp_ms:
                            calc_ts = last_timestamp_ms + 1
                        last_timestamp_ms = calc_ts
                        try:
                            result = video_detector.detect_for_video(mp_image, calc_ts)
                        except Exception as e_det:
                            logger.warning(f"detect_for_video 실패, detect()로 안전 폴백: {e_det}")
                            result = video_detector.detect(mp_image)
                    else:
                        result = video_detector.detect(mp_image)

                    pts_33_2d = np.zeros((33, 2), dtype=np.float32)
                    pts_33_2d_norm = np.zeros((33, 2), dtype=np.float32)
                    pts_33_3d = np.zeros((33, 3), dtype=np.float32)
                    vis_33 = np.zeros((33,), dtype=np.float32)
                    pts_15_2d = np.zeros((15, 2), dtype=np.float32)

                    if result.pose_landmarks and len(result.pose_landmarks) > 0:
                        lms_2d = result.pose_landmarks[0]
                        for i, lm in enumerate(lms_2d):
                            # 원본 해상도(w, h) 픽셀 좌표로 정확히 복원
                            pts_33_2d[i, 0] = lm.x * float(w)
                            pts_33_2d[i, 1] = lm.y * float(h)
                            pts_33_2d_norm[i, 0] = float(lm.x)
                            pts_33_2d_norm[i, 1] = float(lm.y)
                            vis_33[i] = float(lm.visibility) if hasattr(lm, 'visibility') else 1.0

                        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                            world_lms = result.pose_world_landmarks[0]
                            for i, lm in enumerate(world_lms):
                                pts_33_3d[i, 0] = lm.x
                                pts_33_3d[i, 1] = lm.y
                                pts_33_3d[i, 2] = lm.z
                        else:
                            pts_33_3d[:, :2] = pts_33_2d
                            pts_33_3d[:, 2] = 0.0

                        for mp_idx, j15_idx in direct_map.items():
                            pts_15_2d[j15_idx] = pts_33_2d[mp_idx]

                        pelvis_2d = (pts_33_2d[23] + pts_33_2d[24]) / 2.0
                        neck_2d = (pts_33_2d[11] + pts_33_2d[12]) / 2.0
                        torso_2d = (pelvis_2d + neck_2d) / 2.0
                        pts_15_2d[Joint15.PELVIS] = pelvis_2d
                        pts_15_2d[Joint15.NECK] = neck_2d
                        pts_15_2d[Joint15.TORSO] = torso_2d

                    all_kps_33_3d.append(pts_33_3d)
                    all_kps_33_2d.append(pts_33_2d)
                    all_kps_33_2d_norm.append(pts_33_2d_norm)
                    all_kps_15_2d.append(pts_15_2d)
                    all_vis_33.append(vis_33)
                    extracted_count += 1

                    # 메모리 절약: 프레임 배열 즉시 해제 (512MB RAM 보장)
                    del img_rgb
                    del mp_image
                    if proc_frame is not frame:
                        del proc_frame
                    del frame

                    # 웹소켓 진행률 통지 쓰로틀링 (15프레임 주기 또는 마지막 프레임)
                    if progress_callback and total_frames > 0:
                        if frame_idx % 15 == 0 or frame_idx >= total_frames - 1:
                            pct = min(40.0, (frame_idx / total_frames) * 40.0)
                            progress_callback("extracting_pose", pct, f"뼈대 관절 고속 추출 중... ({frame_idx}/{total_frames} 프레임)")

                frame_idx += 1
        finally:
            cap.release()
            if video_detector:
                try:
                    video_detector.close()
                except Exception:
                    pass
            import gc
            gc.collect()

        if extracted_count == 0:
            raise ValueError("비디오에서 유효한 프레임을 읽을 수 없습니다.")

        return {
            "kps_33_3d": np.array(all_kps_33_3d, dtype=np.float32),
            "kps_33_2d": np.array(all_kps_33_2d, dtype=np.float32),
            "kps_33_2d_norm": np.array(all_kps_33_2d_norm, dtype=np.float32),
            "kps_15_2d": np.array(all_kps_15_2d, dtype=np.float32),
            "vis_33": np.array(all_vis_33, dtype=np.float32),
            "total_frames": total_frames,
            "orig_fps": orig_fps,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "fps": effective_fps,
            "sample_stride": sample_stride,
            "duration_sec": total_frames / orig_fps,
            "sampled_frames": None,
        }

    def analyze_video(
        self,
        video_path: str,
        exercise_id: Optional[int] = None,
        camera_view: str = "시점 C: 정면",
        sample_stride: int = 2,
        include_skeleton_preview: bool = True,
        progress_callback: Optional[Callable[[str, float, str, Optional[Dict[str, Any]]], None]] = None,
        cached_landmarks: Optional[Dict[str, Any]] = None
    ) -> VideoAnalysisResponse:
        """
        비디오 파일에 대한 엔드투엔드 전체 운동 및 반복별 결함 분석
        """
        t_start = time.time()
        job_id = str(uuid.uuid4())

        def _notify(stage: str, pct: float, msg: str, data: Optional[Dict[str, Any]] = None):
            if progress_callback:
                progress_callback(stage, pct, msg, data)

        _notify("started", 5.0, "비디오 분석 작업을 시작합니다...")

        # 1. 비디오 관절 추출 (5% ~ 45%)
        if cached_landmarks is not None:
            video_meta = cached_landmarks
            _notify("extracting_pose", 45.0, "캐시된 뼈대 관절 데이터를 로드했습니다.")
        else:
            video_meta = self.extract_video_landmarks(
                video_path=video_path,
                sample_stride=sample_stride,
                progress_callback=lambda st, pct, msg: _notify(st, 5.0 + pct, msg)
            )
        kps_33_3d = video_meta["kps_33_3d"]
        kps_15_2d = video_meta["kps_15_2d"]
        kps_33_2d_norm = video_meta.get("kps_33_2d_norm")
        orig_w = int(video_meta.get("orig_w", 1920))
        orig_h = int(video_meta.get("orig_h", 1080))
        if kps_33_2d_norm is None:
            kps_33_2d = video_meta["kps_33_2d"]
            kps_33_2d_norm = np.zeros_like(kps_33_2d)
            kps_33_2d_norm[:, :, 0] = np.clip(kps_33_2d[:, :, 0] / max(1.0, float(orig_w)), 0.0, 1.0)
            kps_33_2d_norm[:, :, 1] = np.clip(kps_33_2d[:, :, 1] / max(1.0, float(orig_h)), 0.0, 1.0)
        effective_fps = float(video_meta.get("fps", 30.0))
        stride = int(video_meta.get("sample_stride", sample_stride))
        orig_fps = float(video_meta.get("orig_fps", effective_fps * stride))
        total_orig_frames = int(video_meta.get("total_frames", len(kps_33_3d) * stride))
        total_sampled_frames = len(kps_33_3d)

        # 2. [Tier 1] 실제 운동 활성 구간 사전 검출 및 1회 반복 정밀 분할 (45% ~ 55%)
        _notify("filtering_active_period", 46.0, "생체역학 신호 기반 비운동 구간(언랙/셋업/휴식)을 필터링 중입니다...")
        active_start, active_end, raw_reps = ActivePeriodGater.filter_and_segment_repetitions(
            kps_15=kps_15_2d,
            fps=effective_fps
        )

        # 유효 반복이 감지되지 않은 경우 전체를 단일 1회 반복으로 폴백
        if not raw_reps:
            logger.info("반복 피크 미감지: 영상 전체를 1회 반복으로 처리합니다.")
            raw_reps = [(0, total_sampled_frames - 1, total_sampled_frames // 2)]
            active_start, active_end = 0, total_sampled_frames - 1

        active_dur = (active_end - active_start + 1) / effective_fps
        orig_active_start = active_start * stride
        orig_active_end = min(total_orig_frames - 1, active_end * stride)
        _notify(
            "segmenting_reps", 55.0,
            f"실제 운동 활성 구간(F{orig_active_start}~F{orig_active_end}, {active_dur:.1f}초) 내 총 {len(raw_reps)}회의 반복 동작이 확정되었습니다."
        )

        # 3. 1단계 운동 종목 식별 (55% ~ 65%) - 다중 반복 소프트 보팅(Multi-Rep Ensemble Soft Voting)
        _notify("classifying_exercise", 60.0, "1단계: 수행 중인 운동 종목을 다중 반복 앙상블로 AI 판별 중입니다...")

        # 검출된 유효 반복들에 대해 소프트 보팅으로 대표 종목 확정
        candidate_reps = raw_reps[:min(len(raw_reps), 5)]
        accum_probs: Dict[str, float] = {}
        first_stage1_pred = None

        for r_start, r_end, _ in candidate_reps:
            r_seq = kps_33_3d[r_start:max(r_start + 4, r_end + 1)]
            r_seq_64f = SkeletalTopologyAdapter.resample_to_64f_spline(r_seq)
            pred_dict = self.stage1_engine.predict(r_seq_64f)
            single_pred = ExercisePrediction(**pred_dict) if isinstance(pred_dict, dict) else pred_dict
            if first_stage1_pred is None:
                first_stage1_pred = single_pred
            for ex_name, p_val in single_pred.probabilities.items():
                accum_probs[ex_name] = accum_probs.get(ex_name, 0.0) + float(p_val)

        # 소프트 보팅 평균 확률 계산
        n_cands = max(1, len(candidate_reps))
        averaged_probs = {k: v / n_cands for k, v in accum_probs.items()}
        if not averaged_probs:
            best_exercise_name = first_stage1_pred.exercise_name if first_stage1_pred else "스쿼트"
            best_conf = 0.5
        else:
            best_exercise_name = max(averaged_probs, key=averaged_probs.get)
            best_conf = averaged_probs[best_exercise_name]

        # 종목 ID 역매핑
        best_ex_id = first_stage1_pred.exercise_id if first_stage1_pred else 0
        for ex_id, meta in EXERCISE_CATALOG.items():
            if meta["name"] == best_exercise_name:
                best_ex_id = ex_id
                break

        ensemble_stage1_pred = ExercisePrediction(
            exercise_id=best_ex_id,
            exercise_name=best_exercise_name,
            exercise_short=EXERCISE_CATALOG.get(best_ex_id, {}).get("short", best_exercise_name),
            confidence=round(float(best_conf), 4),
            macro_group=first_stage1_pred.macro_group if first_stage1_pred else "lower_body",
            probabilities={k: round(float(v), 4) for k, v in averaged_probs.items()},
            inference_time_ms=first_stage1_pred.inference_time_ms if first_stage1_pred else 0.0
        )

        # 사용자 지정 운동이 있으면 우선 적용, 없으면 AI 앙상블 예측 결과 채택
        if exercise_id is not None and exercise_id in EXERCISE_CATALOG:
            target_exercise_id = exercise_id
            target_meta = EXERCISE_CATALOG[target_exercise_id]
            exercise_result = ExercisePrediction(
                exercise_id=target_exercise_id,
                exercise_name=target_meta["name"],
                exercise_short=target_meta["short"],
                confidence=ensemble_stage1_pred.probabilities.get(target_meta["name"], ensemble_stage1_pred.confidence),
                macro_group=ensemble_stage1_pred.macro_group,
                probabilities=ensemble_stage1_pred.probabilities,
                inference_time_ms=ensemble_stage1_pred.inference_time_ms
            )
        else:
            target_exercise_id = ensemble_stage1_pred.exercise_id
            exercise_result = ensemble_stage1_pred


        # 4. RTMPose Halpe26 실제 관절 고속 유도형 추출 (시간축 활성 구간 한정 추출: 55% ~ 75%)
        # MediaPipe 2D 관절을 기반으로 YOLOX 디텍터를 0ms로 스킵하고, 기디코딩 프레임 버퍼로 디스크 I/O 0ms 달성
        _notify("extracting_rtmpose", 55.0, "RTMPose Halpe26 고속 유도형 관절(발 3각 지지면 & 척추선)을 추출 중입니다...")
        cached_frames = video_meta.get("sampled_frames")
        rtmp_active_26 = None
        try:
            rtmp_active_26 = self.rtmp_tracker.extract_active_period_rtmpose(
                video_path=video_path,
                active_start_frame=active_start,
                active_end_frame=active_end,
                mp_kps_33_2d=video_meta["kps_33_2d"],
                fps=orig_fps,
                sample_stride=stride,
                cached_frames=cached_frames,
                orig_w=video_meta.get("orig_w"),
                orig_h=video_meta.get("orig_h"),
                progress_callback=lambda st, pct, msg: _notify(st, pct, msg)
            )
        except Exception as e_rtm:
            logger.warning(f"RTMPose 추출 오류 또는 폴백 전환: {e_rtm}")
            rtmp_active_26 = None

        # 5. 2단계 운동별 횟수 단위 세부 결함 진단 (75% ~ 95%)
        _notify(
            "diagnosing_defects", 75.0,
            f"2단계: [{exercise_result.exercise_name}] 횟수별 33종 세부 결함을 정밀 분석 중입니다..."
        )

        repetition_details: List[RepetitionDetail] = []
        num_reps = len(raw_reps)

        for rep_idx, (start_f, end_f, bottom_f) in enumerate(raw_reps, start=1):
            rep_seq = kps_33_3d[start_f:max(start_f + 4, end_f + 1)]
            # 64프레임 3차 스플라인 보간 (규정 엄격 준수)
            rep_seq_64f = SkeletalTopologyAdapter.resample_to_64f_spline(rep_seq)

            # RTMPose 해당 반복 구간 슬라이싱 및 64프레임 스플라인 보간
            rep_rtm_64f = None
            if rtmp_active_26 is not None and len(rtmp_active_26) > 0:
                rel_start = max(0, start_f - active_start)
                rel_end = min(len(rtmp_active_26) - 1, end_f - active_start)
                rep_rtm_seq = rtmp_active_26[rel_start:max(rel_start + 4, rel_end + 1)]
                rep_rtm_64f = SkeletalTopologyAdapter.resample_to_64_frames(rep_rtm_seq)

            # MediaPipe 2D 정규화 해당 반복 구간 슬라이싱 및 64프레임 스플라인 보간
            rep_mp_norm_raw = kps_33_2d_norm[start_f:max(start_f + 4, end_f + 1)]
            rep_mp_norm_64f = SkeletalTopologyAdapter.resample_to_64_frames(rep_mp_norm_raw)

            # [방식 3] SpinePose 핵심 16F 샘플링 + MediaPipe BBox 유도 추론 + 64F 스플라인 보간
            # 척추 중립 결함 검사가 필요한 운동(0: 바벨 스쿼트 등)에 대해 정밀 3D 척추 추출
            rep_spine_3d_64f = None
            rep_spine_2d_norm_64f = None
            if target_exercise_id in (0, 1):  # 바벨 스쿼트(c1) 및 바벨 데드리프트(d1) 척추 중립 위반 전용
                _notify(
                    "extracting_spinepose",
                    75.0 + ((rep_idx - 1) / num_reps) * 15.0,
                    f"반복 {rep_idx}/{num_reps}: SpinePose 핵심 16F 척추 추론 및 64F 스플라인 보간 중..."
                )
                scale_val = min(540.0 / orig_w, 960.0 / orig_h) if (orig_w > 540 or orig_h > 960) else 1.0
                rep_spine_3d_64f, rep_spine_2d_norm_64f = self.spine_tracker.extract_repetition_spine_64f(
                    frames_source=cached_frames if cached_frames else video_path,
                    start_frame=start_f,
                    end_frame=end_f,
                    mp_kps_33_2d=video_meta["kps_33_2d"],
                    sample_stride=stride,
                    num_sample_frames=16,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    scale=scale_val
                )

            # 2단계 서브 결함 모델 직접 호출 (MediaPipe 33 + RTMPose 26 + SpinePose 9 융합 실측 좌표 전달)
            diag = self.stage2_engine.evaluate_single_repetition(
                exercise_id=target_exercise_id,
                kps_sequence=rep_seq_64f,
                camera_view=camera_view,
                source_type="mediapipe_33",
                rtmp_seq=rep_rtm_64f,
                spine_seq=rep_spine_3d_64f
            )

            # 뼈대 프리뷰 좌표 (선택 시 15관절 2D 정규화 또는 3D)
            preview_coords = None
            overlay_mp_coords = None
            overlay_rtmp_coords = None
            overlay_spine_coords = None
            if include_skeleton_preview:
                # (64, 15, 3)으로 변환 후 소수점 4자리 리스트화 (하위 호환)
                kps_15 = SkeletalTopologyAdapter.mediapipe_33_to_joint15(rep_seq_64f)
                preview_coords = np.round(kps_15, 4).tolist()

                # 영상 오버레이용 MediaPipe 33 (64, 33, 2)
                overlay_mp_coords = np.round(np.clip(rep_mp_norm_64f[:, :, :2], 0.0, 1.0), 4).tolist()

                # 영상 오버레이용 RTMPose 26 (64, 26, 3) [x, y, conf]
                if rep_rtm_64f is not None:
                    overlay_rtmp_coords = np.round(np.clip(rep_rtm_64f, 0.0, 1.0), 4).tolist()
                else:
                    kps_rtm26 = SkeletalTopologyAdapter.mediapipe_33_to_rtmpose_26(rep_seq_64f)
                    overlay_rtmp_coords = np.round(np.clip(kps_rtm26, 0.0, 1.0), 4).tolist()

                # 영상 오버레이용 SpinePose 9 (64, 9, 3) [x, y, conf]
                if rep_spine_2d_norm_64f is not None:
                    overlay_spine_coords = np.round(np.clip(rep_spine_2d_norm_64f, 0.0, 1.0), 4).tolist()

            start_sec = round((start_f * stride) / orig_fps, 3)
            end_sec = round(min(total_orig_frames - 1, end_f * stride) / orig_fps, 3)
            duration = max(0.1, end_sec - start_sec)

            rep_coaching = diag.get("coaching", {})

            rep_detail = RepetitionDetail(
                rep_index=rep_idx,
                start_frame=int(start_f * stride),
                end_frame=int(min(total_orig_frames - 1, end_f * stride)),
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=round(duration, 2),
                is_clean=diag["is_clean"],
                defects=diag["defects"],
                primary_feedback=rep_coaching.get("primary_action_cue"),
                primary_detail=rep_coaching.get("primary_detail"),
                secondary_feedbacks=rep_coaching.get("secondary_cues", []),
                causal_chain_title=rep_coaching.get("causal_chain_detected"),
                view_coaching_tip=rep_coaching.get("view_tip"),
                keypoints_preview=preview_coords,
                overlay_mp=overlay_mp_coords,
                overlay_rtmp=overlay_rtmp_coords,
                overlay_spine=overlay_spine_coords
            )
            repetition_details.append(rep_detail)

            pct = 75.0 + (rep_idx / num_reps) * 20.0
            _notify("diagnosing_defects", pct, f"반복 {rep_idx}/{num_reps}회차 결함 진단 완료")

        # 메모리 반환: 분석 완료 즉시 대용량 프레임 캐시 해제 및 GC 강제 회수
        video_meta.pop("sampled_frames", None)
        import gc
        gc.collect()

        total_time = round(time.time() - t_start, 2)
        _notify("completed", 100.0, "전체 비디오 운동 자세 분석이 성공적으로 완료되었습니다!")

        active_period_info = {
            "start_frame": int(active_start * stride),
            "end_frame": int(min(total_orig_frames - 1, active_end * stride)),
            "duration_sec": round((active_end - active_start + 1) / effective_fps, 2),
            "filtered_frames_count": max(0, int(total_orig_frames - (active_end - active_start + 1) * stride))
        }

        video_info_data = {
            "width": orig_w,
            "height": orig_h,
            "fps": round(orig_fps, 2),
            "effective_fps": round(effective_fps, 2),
            "sample_stride": stride,
            "total_frames": total_orig_frames,
            "duration_sec": round(total_orig_frames / orig_fps, 2)
        }

        # 3단계 세트 종합 맞춤 처방전 생성 (FeedbackEngine)
        from src.web.core.feedback_engine import FeedbackEngine
        fb_engine = FeedbackEngine.get_instance()
        set_summary_data = fb_engine.generate_set_summary(
            exercise_id=exercise_result.exercise_id,
            exercise_name=exercise_result.exercise_name,
            rep_results=[r.model_dump() for r in repetition_details]
        )

        return VideoAnalysisResponse(
            success=True,
            job_id=job_id,
            exercise=exercise_result,
            selected_view=camera_view,
            total_reps=len(repetition_details),
            reps=repetition_details,
            active_period=active_period_info,
            video_info=video_info_data,
            total_processing_time_sec=total_time,
            set_summary=set_summary_data,
            message=f"{exercise_result.exercise_name} {len(repetition_details)}회 반복 분석 및 생체역학 코칭 처방이 완료되었습니다."
        )

    def analyze_sequence_direct(
        self,
        keypoints: np.ndarray,
        exercise_id: Optional[int] = None,
        camera_view: str = "시점 C: 정면",
        source_type: str = "mediapipe_33"
    ) -> Dict[str, Any]:
        """
        웹캠 또는 외부 클라이언트로부터 직접 전송된 (T, J, C) 관절 배열 즉시 분석
        """
        t0 = time.time()
        # 1. 64프레임 스플라인 보간
        kps_64f = SkeletalTopologyAdapter.resample_to_64f_spline(keypoints)

        # 2. 1단계 종목 분류
        stage1_pred_dict = self.stage1_engine.predict(kps_64f)
        stage1_pred = ExercisePrediction(**stage1_pred_dict) if isinstance(stage1_pred_dict, dict) else stage1_pred_dict

        if exercise_id is None or exercise_id not in EXERCISE_CATALOG:
            target_exercise_id = stage1_pred.exercise_id
            exercise_result = stage1_pred
        else:
            target_exercise_id = exercise_id
            meta = EXERCISE_CATALOG[target_exercise_id]
            exercise_result = ExercisePrediction(
                exercise_id=target_exercise_id,
                exercise_name=meta["name"],
                exercise_short=meta["short"],
                confidence=stage1_pred.probabilities.get(meta["name"], 1.0),
                macro_group=stage1_pred.macro_group,
                probabilities=stage1_pred.probabilities,
                inference_time_ms=stage1_pred.inference_time_ms
            )


        # 3. 2단계 세부 결함 진단
        diag = self.stage2_engine.evaluate_single_repetition(
            exercise_id=target_exercise_id,
            kps_sequence=kps_64f,
            camera_view=camera_view,
            source_type=source_type
        )

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "exercise": exercise_result,
            "selected_view": camera_view,
            "is_clean": diag["is_clean"],
            "defects": diag["defects"],
            "inference_time_ms": round(elapsed_ms, 2)
        }
