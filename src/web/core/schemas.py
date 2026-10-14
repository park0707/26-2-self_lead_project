# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/core/schemas.py
Purpose: Pydantic 기반 웹 서비스 입출력 REST / WebSocket JSON 표준 스키마
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ExercisePrediction(BaseModel):
    """1단계 운동 종목 분류 결과"""
    exercise_id: int = Field(..., description="운동 고유 ID (0~7)")
    exercise_name: str = Field(..., description="공식 운동 명칭")
    exercise_short: str = Field(..., description="약식 운동 명칭")
    confidence: float = Field(..., description="분류 확신도 (0.0~1.0)")
    macro_group: str = Field(..., description="소속 6대 상위 기능 그룹")
    probabilities: Dict[str, float] = Field(..., description="8대 운동별 사후 확률")
    inference_time_ms: float = Field(..., description="추론 지연시간 (ms)")


class DefectPrediction(BaseModel):
    """2단계 세부 결함 진단 결과"""
    defect_code: str = Field(..., description="결함 코드 (예: c1, bl2, d3)")
    defect_name: str = Field(..., description="세부 결함 명칭")
    probability: float = Field(..., description="결함 발생 확률 (0.0~1.0)")
    state: int = Field(default=0, description="3-State (0: 정상, 1: 결함, 2: 주의/보류)")
    state_label: str = Field(default="정상", description="3-State 한글 레이블")
    feedback: str = Field(default="피드백 자리", description="생체역학적 코칭 피드백")
    # 3단계 생체역학 세부 필드 (선택적)
    target_joint: Optional[str] = Field(default=None, description="부하 집중 대상 관절/부위")
    action_cue: Optional[str] = Field(default=None, description="즉각적 1문장 핵심 행동 지침")
    biomechanical_reason: Optional[str] = Field(default=None, description="생체역학적 원인 및 관절 부하 설명")
    optimal_view: Optional[str] = Field(default=None, description="관측 최적 권장 카메라 시점")


class RepetitionDetail(BaseModel):
    """단일 횟수(1-Rep) 분석 결과"""
    rep_index: int = Field(..., description="반복 회차 (1부터 시작)")
    start_frame: int = Field(..., description="반복 시작 원천 프레임 번호")
    end_frame: int = Field(..., description="반복 종료 원천 프레임 번호")
    start_sec: float = Field(default=0.0, description="반복 시작 시각 (초)")
    end_sec: float = Field(default=0.0, description="반복 종료 시각 (초)")
    duration_sec: float = Field(..., description="반복 수행 시간 (초)")
    is_clean: bool = Field(..., description="전체 결함 없는 완전 정상 반복 여부")
    defects: List[DefectPrediction] = Field(default_factory=list, description="세부 결함 진단 목록")
    # 3단계 단일 횟수 우선순위 코칭 필드
    primary_feedback: Optional[str] = Field(default=None, description="1회차 최우선 행동 지침 큐")
    primary_detail: Optional[str] = Field(default=None, description="생체역학적 배경 설명")
    secondary_feedbacks: List[str] = Field(default_factory=list, description="보조 행동 지침 목록 (최대 1개)")
    causal_chain_title: Optional[str] = Field(default=None, description="복합 연쇄 결함 인과 진단명")
    view_coaching_tip: Optional[str] = Field(default=None, description="카메라 시점 최적화 안내 팁")
    # 프론트엔드 스켈레톤 렌더링용 기존 3D 키포인트 (하위 호환)
    keypoints_preview: Optional[List[List[List[float]]]] = Field(
        default=None, description="프론트엔드 뼈대 시각화용 64프레임 정규화 좌표"
    )
    # 영상 위 2D 오버레이 렌더링용 정규화 좌표 (0.0~1.0)
    overlay_mp: Optional[List[List[List[float]]]] = Field(
        default=None, description="MediaPipe 33관절 2D 정규화 64프레임 시퀀스 (x, y)"
    )
    overlay_rtmp: Optional[List[List[List[float]]]] = Field(
        default=None, description="RTMPose Halpe26 2D 정규화 64프레임 시퀀스 (x, y, conf)"
    )
    overlay_spine: Optional[List[List[List[float]]]] = Field(
        default=None, description="SpinePose 9대 척추 분절 2D 정규화 64프레임 시퀀스 (x, y, conf)"
    )


class VideoAnalysisResponse(BaseModel):
    """비디오 E2E 분석 완료 최종 응답"""
    success: bool = True
    job_id: str = Field(..., description="작업 고유 UUID")
    exercise: ExercisePrediction = Field(..., description="1단계 운동 분류 결과")
    selected_view: str = Field(..., description="카메라 촬영 시점 (시점 A~E)")
    total_reps: int = Field(..., description="총 완료 횟수 (Reps)")
    reps: List[RepetitionDetail] = Field(default_factory=list, description="횟수별 세부 진단 결과")
    active_period: Optional[Dict[str, Any]] = Field(None, description="사전 필터링된 실제 운동 활성 구간 (start_frame, end_frame, duration_sec 등)")
    video_info: Optional[Dict[str, Any]] = Field(default=None, description="비디오 원천 메타데이터 (width, height, fps, total_frames)")
    total_processing_time_sec: float = Field(..., description="전체 분석 소요 시간 (초)")
    set_summary: Optional[Dict[str, Any]] = Field(default=None, description="3단계 세트 종합 맞춤 처방전 및 보강 운동 드릴")
    message: str = "분석이 성공적으로 완료되었습니다."


class SequenceAnalysisRequest(BaseModel):
    """실시간 웹캠/외부 클라이언트 단일 시퀀스 분석 요청"""
    exercise_id: Optional[int] = Field(None, description="운동 고유 ID (0~7, None이면 1단계 모델로 자동 분류)")
    camera_view: str = Field(default="시점 C: 정면", description="카메라 촬영 시점 (시점 A~E)")
    keypoints: List[List[List[float]]] = Field(..., description="(T, J, C) 관절 좌표 배열")
    source_type: str = Field(default="mediapipe_33", description="키포인트 유형 (mediapipe_33 또는 rtmpose_26)")


class SequenceAnalysisResponse(BaseModel):
    """단일 시퀀스 분석 응답"""
    success: bool = True
    exercise: ExercisePrediction = Field(..., description="1단계 운동 분류 결과")
    selected_view: str = Field(..., description="카메라 촬영 시점 (시점 A~E)")
    is_clean: bool = Field(..., description="전체 결함 없는 완전 정상 반복 여부")
    defects: List[DefectPrediction] = Field(default_factory=list, description="세부 결함 진단 목록")
    inference_time_ms: float = Field(..., description="추론 소요 시간 (ms)")


class HealthResponse(BaseModel):
    """서버 헬스체크 및 모델 준비 상태 응답"""
    status: str = "healthy"
    service_name: str = "vidio-health-trainer-web-api"
    version: str = "1.0.0"
    stage1_ready: bool = True
    stage2_ready: bool = True
    warmed_exercises_count: int = 8
    total_defects_count: int = 33
    supported_exercises: List[Dict[str, Any]] = Field(default_factory=list)
    camera_views: List[Dict[str, str]] = Field(default_factory=list)


class WebSocketProgressMessage(BaseModel):
    """웹소켓 실시간 진행률 알림 메시지"""
    job_id: str
    stage: str = Field(..., description="진행 단계 (extracting_pose, segmenting_reps, classifying_exercise, diagnosing_defects, completed, error)")
    progress_percent: float = Field(..., description="진행률 (0.0~100.0)")
    message: str = Field(..., description="사용자 안내 메시지")
    data: Optional[Dict[str, Any]] = None
