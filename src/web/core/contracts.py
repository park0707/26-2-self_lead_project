# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/core/contracts.py
Purpose: 웹 서비스 4대 계층(비전-전처리-피처-추론) 완전 분리 아키텍처 공통 데이터 계약(Data Contracts)
         - Contract-First / Interface-Driven 아키텍처 준수
         - 불변 데이터클래스(Immutable Dataclasses) 기반으로 계층 간 구현 세부사항 은닉
         - AGENTS.md 제2조 64프레임 스플라인 보간 표준 및 피험자/시점 전수 무결성 지원
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


# ==============================================================================
# [Tier 1] 비전 관절 추출 계층 출력 계약 (Raw Vision Landmarks)
# ==============================================================================

@dataclass(frozen=True)
class RawVisionLandmarks:
    """
    Tier 1 비전 관절 추적기가 영상으로부터 추출한 원시 시계열 관절 좌표 계약
    """
    tracker_type: str         # 'mediapipe_33' | 'rtmpose_26' | 'spinepose_9'
    keypoints: np.ndarray     # (T, J, C) 가변 프레임 원시 좌표 텐서 (float32)
    fps: float                # 영상 원본 FPS
    sample_stride: int = 1    # 프레임 샘플링 주기 (1 또는 2)
    orig_w: int = 1920        # 원본 영상 너비
    orig_h: int = 1080        # 원본 영상 높이
    frame_indices: Optional[List[int]] = None # 추출된 실제 프레임 인덱스 리스트

    def __post_init__(self):
        if not isinstance(self.keypoints, np.ndarray):
            object.__setattr__(self, 'keypoints', np.asarray(self.keypoints, dtype=np.float32))


# ==============================================================================
# [Tier 2] 골격 표준 전처리 계층 출력 계약 (Preprocessed Skeletons Container)
# ==============================================================================

@dataclass(frozen=True)
class PreprocessedSkeletons:
    """
    Tier 2 골격 표준 전처리 허브가 1회 반복 단위로 슬라이싱 및 64F 스플라인 보간을 완료한 표준 컨테이너
    - 불변 원칙: MediaPipe 33과 RTMPose 26을 억지로 상호 역변환하지 않고, 각각 순수 원본 좌표계 보존
    """
    mp33_64f: np.ndarray                   # (64, 33, 3) 순수 MediaPipe 3D 64프레임 스플라인 (float32)
    rtm26_64f: Optional[np.ndarray] = None # (64, 26, 3) 순수 RTMPose Halpe26 64프레임 스플라인 (float32)
    spine9_64f: Optional[np.ndarray] = None# (64, 9, 3) 순수 SpinePose 9척추 64프레임 스플라인 (float32)
    view_code: str = "view3"               # 공식 시점 코드 ('view1' ~ 'view5')
    exercise_id: int = 0                   # 표준 운동 ID (0~7)
    rep_index: int = 1                     # 세트 내 반복 횟수 번호
    start_frame: int = 0                   # 원본 영상 기준 시작 프레임
    end_frame: int = 0                     # 원본 영상 기준 종료 프레임

    def __post_init__(self):
        # 64프레임 무결성 방어 검증
        if self.mp33_64f is not None and len(self.mp33_64f) != 64:
            raise ValueError(f"PreprocessedSkeletons mp33_64f 길이는 반드시 64여야 합니다: {len(self.mp33_64f)}")
        if self.rtm26_64f is not None and len(self.rtm26_64f) != 64:
            raise ValueError(f"PreprocessedSkeletons rtm26_64f 길이는 반드시 64여야 합니다: {len(self.rtm26_64f)}")
        if self.spine9_64f is not None and len(self.spine9_64f) != 64:
            raise ValueError(f"PreprocessedSkeletons spine9_64f 길이는 반드시 64여야 합니다: {len(self.spine9_64f)}")


# ==============================================================================
# [Tier 4] 결함 모델 추론 계층 출력 계약 (Defect Prediction)
# ==============================================================================

@dataclass(frozen=True)
class DefectPredictionResult:
    """
    Tier 4 결함 모델 추론 엔진이 산출한 단일 결함 최종 진단 계약
    """
    defect_code: str          # 결함 코드 (예: 'c1', 'c2', 'c3', 'c4', 'd1' ...)
    defect_name: str          # 결함 공식 명칭 (예: '발바닥 지면 고정')
    probability: float        # 보정된 결함 확률 (0.0 ~ 1.0)
    state: int                # 3-State 상태 코드 (0: 정상 CLEAN, 1: 결함 DEFECT, 2: 판정보류 RETAKE)
    state_label: str          # 상태 명칭 (예: 'CLEAN (발바닥 지면 밀착 정상)')
    feedback: str = ""        # 3단계 코칭 피드백 문구
    target_joint: str = ""    # 대상 관절
    action_cue: str = ""      # 즉각 행동 큐
    corrective_action: str = "" # 교정 운동
    biomechanical_reason: str = "" # 생체역학적 원인
    optimal_view: str = ""    # 최적 관측 시점
