# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/api/routers/analyze.py
Purpose: 비디오 파일 업로드 및 실시간 관절 시퀀스 분석 REST 엔드포인트
"""

import sys
import os
import shutil
import tempfile
import uuid
import asyncio
import logging
from typing import Optional, Dict, Any
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.core.schemas import (
    VideoAnalysisResponse, SequenceAnalysisRequest, SequenceAnalysisResponse
)
from src.web.pipeline.video_processor import VideoAnalysisPipeline
from src.web.api.websocket import ws_manager

logger = logging.getLogger("analyze_router")
router = APIRouter(prefix="/api/analyze", tags=["Exercise Analysis"])


@router.post("/video", response_model=VideoAnalysisResponse)
async def analyze_video_endpoint(
    video_file: UploadFile = File(..., description="분석할 운동 영상 파일 (.mp4, .mov, .avi)"),
    exercise_id: Optional[int] = Form(None, description="운동 종목 ID (0~7, 미입력 시 1단계 AI 자동 판별)"),
    camera_view: str = Form("시점 C: 정면", description="카메라 촬영 시점 (시점 A~E)"),
    sample_stride: int = Form(2, description="프레임 서브샘플링 간격 (1: 전수 30FPS, 2: 15FPS 권장 2.3배속)"),
    channel_id: Optional[str] = Form(None, description="실시간 진행률 수신용 웹소켓 채널 ID")
) -> VideoAnalysisResponse:
    """
    운동 비디오 파일 업로드 및 E2E 자동 자세 분석 엔드포인트
    1. 비디오 파일 수신 및 임시 파일 저장
    2. 프레임 서브샘플링(Stride 2 = 15FPS) 및 비디오 뼈대(MediaPipe) 초고속 추출
    3. [Tier 1] 비운동 구간(언랙/셋업) 소거 및 1회 반복(Repetition) 정밀 분할
    4. [Tier 1] 1회 반복 64프레임 3차 스플라인(Cubic Spline) 정밀 보간
    5. [Tier 2] MediaPipe 유도형 RTMPose Halpe26 발 3각 지지면 & 척추선 추출
    6. 1단계 종목 분류 및 2단계 33종 세부 결함 진단
    7. 실시간 진행률 WebSocket 브로드캐스트
    """
    # 1. 파일 확장자 검증
    allowed_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    file_ext = os.path.splitext(video_file.filename)[1].lower()
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 비디오 포맷입니다: {file_ext} (지원: {', '.join(allowed_exts)})"
        )

    # 2. 임시 파일 안전 저장
    temp_dir = os.path.join(PROJECT_ROOT, "src", "web", "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"upload_{uuid.uuid4().hex}{file_ext}"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        # 3. 비동기 이벤트 루프 캡처 (WebSocket 통지용)
        loop = asyncio.get_event_loop()

        def _sync_progress_callback(stage: str, pct: float, msg: str, data: Optional[Dict[str, Any]] = None):
            if channel_id:
                asyncio.run_coroutine_threadsafe(
                    ws_manager.send_progress(channel_id, stage, pct, msg, data),
                    loop
                )

        # 4. 비디오 분석 파이프라인 실행 (FastAPI 이벤트 루프 블로킹 방지를 위해 워커 스레드로 비동기 위임)
        pipeline = VideoAnalysisPipeline.get_instance()
        result = await asyncio.to_thread(
            pipeline.analyze_video,
            video_path=temp_path,
            exercise_id=exercise_id,
            camera_view=camera_view,
            sample_stride=sample_stride,
            include_skeleton_preview=True,
            progress_callback=_sync_progress_callback
        )
        return result

    except Exception as e:
        logger.error(f"비디오 분석 실패: {e}", exc_info=True)
        if channel_id:
            await ws_manager.send_progress(channel_id, "error", 0.0, f"분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"비디오 분석 중 내부 오류 발생: {str(e)}")

    finally:
        # 임시 파일 정리
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {temp_path}, {e}")


@router.post("/sequence", response_model=SequenceAnalysisResponse)
def analyze_sequence_endpoint(
    request: SequenceAnalysisRequest
) -> SequenceAnalysisResponse:
    """
    클라이언트에서 추출한 관절 좌표 시퀀스((T, J, C) 배열) 직접 전달 분석 엔드포인트
    - 웹캠 실시간 반복 분석이나 경량 프론트엔드 연동에 적합
    """
    try:
        kps_np = np.array(request.keypoints, dtype=np.float32)
        if kps_np.ndim != 3:
            raise HTTPException(
                status_code=400,
                detail=f"관절 시퀀스 규격 오류: (T, J, C)의 3차원 배열이어야 합니다. (입력 차원: {kps_np.shape})"
            )

        pipeline = VideoAnalysisPipeline.get_instance()
        res = pipeline.analyze_sequence_direct(
            keypoints=kps_np,
            exercise_id=request.exercise_id,
            camera_view=request.camera_view,
            source_type=request.source_type
        )

        return SequenceAnalysisResponse(
            success=True,
            exercise=res["exercise"],
            selected_view=res["selected_view"],
            is_clean=res["is_clean"],
            defects=res["defects"],
            inference_time_ms=res["inference_time_ms"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"시퀀스 분석 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"시퀀스 분석 중 내부 오류 발생: {str(e)}")
