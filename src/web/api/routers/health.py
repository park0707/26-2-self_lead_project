# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/api/routers/health.py
Purpose: 시스템 헬스체크, 모델 사전 워밍 상태 점검 및 메타데이터 API 엔드포인트
"""

import sys
import os
from typing import Dict, Any, List
from fastapi import APIRouter

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.core.config import EXERCISE_CATALOG, VIEW_METADATA, VIEW_ALIAS_MAP
from src.web.core.schemas import HealthResponse
from src.web.core.model_registry import DefectModelRegistry
from src.web.pipeline.stage1_engine import Stage1InferenceEngine

router = APIRouter(prefix="/api", tags=["Health & Metadata"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    서버 헬스체크 및 1단계/2단계 모델 상태 조회
    """
    stage1 = Stage1InferenceEngine.get_instance()
    stage1_ready = getattr(stage1, "bundle", None) is not None

    registry = DefectModelRegistry.get_instance()
    warmed_count = sum(1 for is_w in registry._is_warmed.values() if is_w)
    stage2_ready = True  # 온디맨드 지연 로딩 레지스트리 상시 준비 완료 (512MB RAM 최적화)

    exercises_list = [
        {
            "id": ex["id"],
            "code": ex["code"],
            "name": ex["name"],
            "short": ex["short"],
            "defects_count": ex["defects_count"]
        }
        for ex in EXERCISE_CATALOG.values()
    ]

    views_list = [
        {
            "code": code,
            "name": meta["name"],
            "alias": meta.get("code", code),
            "angle": meta.get("desc", "")
        }
        for code, meta in VIEW_METADATA.items()
    ]


    return HealthResponse(
        status="healthy",
        service_name="vidio-health-trainer-web-api",
        version="1.0.0",
        stage1_ready=stage1_ready,
        stage2_ready=stage2_ready,
        warmed_exercises_count=warmed_count,
        total_defects_count=33,
        supported_exercises=exercises_list,
        camera_views=views_list
    )


@router.get("/exercises")
def list_exercises() -> Dict[str, Any]:
    """8대 지원 운동 목록 반환"""
    return {
        "count": len(EXERCISE_CATALOG),
        "exercises": list(EXERCISE_CATALOG.values())
    }


@router.get("/views")
def list_views() -> Dict[str, Any]:
    """공식 5대 촬영 시점(시점 A~E) 반환"""
    return {
        "count": len(VIEW_METADATA),
        "views": VIEW_METADATA,
        "aliases": VIEW_ALIAS_MAP
    }
