# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/api/main.py
Purpose: FastAPI 메인 웹 서비스 애플리케이션 진입점
         - Lifespan 이벤트 핸들러: 서버 기동 시 1단계 및 2단계(33종) 전 모델 사전 워밍 (콜드 스타트 방지)
         - CORS 미들웨어 등록 (웹 프론트엔드 크로스 오리진 지원)
         - 헬스체크, 비디오 분석, 시퀀스 분석, 실시간 웹소켓 라우터 통합
         - 정적 파일(Static UI) 마운트 및 루트 서빙
"""

import sys
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# 프로젝트 루트 및 모듈 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
FEATURES_DIR = os.path.join(SRC_ROOT, "features")

for p in [PROJECT_ROOT, SRC_ROOT, FEATURES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.web.api.routers import health, analyze
from src.web.api import websocket
from src.web.pipeline.stage1_engine import Stage1InferenceEngine
from src.web.pipeline.stage2_engine import Stage2InferenceEngine
from src.web.pipeline.video_processor import VideoAnalysisPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s"
)
logger = logging.getLogger("web_api_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 수명 주기(Lifespan) 관리:
    클라우드 프리 티어(512MB RAM) 최적화:
    - 1단계 종목 분류 엔진 싱글톤 로드 (~5.5MB)
    - 2단계 33종 결함 모델 및 비전 디텍터는 사용 시점 온디맨드 지연 로딩(Lazy Loading)으로 OOM 방지
    """
    import gc
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    logger.info("=" * 70)
    logger.info("★ [vidio-health-trainer] 웹 서비스 백엔드 기동 시작 (Cloud Memory-Optimized Mode)")
    logger.info("=" * 70)

    t0 = time.time()

    # 1. 1단계 종목 분류 엔진 싱글톤 로드
    logger.info("[1/2] 1단계 운동 분류 엔진 로드 중...")
    stage1 = Stage1InferenceEngine.get_instance()
    logger.info(f"      ✔ 1단계 모델 로드 완료 (버전: {stage1.bundle.get('version', 'v6.0')}, 지원 운동: {len(stage1.exercise_names_ko)}종)")

    # 2. 2단계 엔진 싱글톤 등록 (사전 워밍 생략, On-Demand Lazy Loading 유지)
    logger.info("[2/2] 2단계 33종 결함 엔진 및 비전 파이프라인 지연 로딩 모드 활성화 (512MB RAM 보호)")
    _ = Stage2InferenceEngine.get_instance(prewarm=False)

    gc.collect()
    total_startup_sec = round(time.time() - t0, 2)
    logger.info("=" * 70)
    logger.info(f"★ [vidio-health-trainer] 백엔드 기동 완료 ({total_startup_sec}초) - 포트 바인딩 및 요청 수신 준비 완료!")
    logger.info("=" * 70)

    yield

    logger.info("★ [vidio-health-trainer] 웹 서비스 종료 처리 중...")


# FastAPI 인스턴스 초기화
app = FastAPI(
    title="vidio-health-trainer AI Video Analysis API",
    description="운동 동영상 분석 및 8대 운동 33종 세부 자세 결함 진단 실시간 백엔드 서비스",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(websocket.router)

# 정적 파일 및 UI 디렉토리 마운트 (Step 4 프론트엔드 연동 대비)
STATIC_DIR = os.path.join(CURRENT_DIR, "..", "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["Root"])
def root_endpoint() -> Dict[str, Any]:
    """서비스 기본 정보 및 API 문서 링크 안내"""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)

    return {
        "service": "vidio-health-trainer API Server",
        "status": "online",
        "version": "1.0.0",
        "description": "8대 운동 분류 및 33종 세부 결함 실시간 비디오 분석 서비스",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_endpoints": {
            "health": "/api/health",
            "exercises": "/api/exercises",
            "views": "/api/views",
            "analyze_video": "POST /api/analyze/video",
            "analyze_sequence": "POST /api/analyze/sequence",
            "websocket_progress": "WS /api/ws/analyze/{channel_id}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web.api.main:app", host="0.0.0.0", port=8000, reload=False)
