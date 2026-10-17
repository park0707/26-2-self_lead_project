# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/pipeline/stage2_engine.py
Purpose: 웹 서빙 전용 2단계 운동별 세부 결함(33종) 실시간 단일 추론 엔진 (Stage2InferenceEngine)
         - 마스터 클래스 없이 8대 운동별 개별 세부 결함 모델을 직접 호출
         - MediaPipe 33J / RTMPose 26J 관절 토폴로지 상호 변환 및 64F 스플라인 보간
         - 싱글톤 사전 워밍 지원 (추론 지연시간 20~50ms 달성)
         - 각 결함 모델 출력 규격: {defect_code, defect_name, probability, state, state_label, feedback: '피드백 자리'}
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.core.config import EXERCISE_CATALOG, VIEW_ALIAS_MAP, VIEW_METADATA
from src.web.core.schemas import DefectPrediction
from src.web.core.topology_adapter import SkeletalTopologyAdapter
from src.web.core.model_registry import DefectModelRegistry


class Stage2InferenceEngine:
    """2단계 8종 운동 세부 결함 실시간 초고속 단일 추론 엔진 (웹 서비스 전용)"""

    _instance = None

    def __init__(self, prewarm: bool = False):
        self.registry = DefectModelRegistry.get_instance()
        if prewarm:
            self.warm_up()

    @classmethod
    def get_instance(cls, prewarm: bool = False) -> "Stage2InferenceEngine":
        """싱글톤 패턴 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls(prewarm=prewarm)
        return cls._instance

    def warm_up(self, exercise_id: Optional[int] = None) -> float:
        """사전 워밍 수행"""
        return self.registry.warm_up(exercise_id=exercise_id)

    @staticmethod
    def _resolve_exercise_id(exercise: Union[int, str]) -> int:
        """운동 정수 ID 또는 문자열 이름으로부터 표준 exercise_id (0~7) 변환"""
        if isinstance(exercise, int):
            if 0 <= exercise <= 7:
                return exercise
            raise ValueError(f"유효하지 않은 운동 ID입니다: {exercise} (범위: 0~7)")

        ex_str = str(exercise).strip()
        for eid, meta in EXERCISE_CATALOG.items():
            if ex_str == meta['name'] or ex_str == meta['short'] or ex_str == meta['code'] or str(eid) == ex_str:
                return eid
            if meta['short'] in ex_str or meta['name'] in ex_str:
                return eid

        # 폴더명 매핑
        folder_names = {
            '01_바벨_스쿼트': 0, '스쿼트': 0, 'squat': 0,
            '02_바벨_데드리프트': 1, '데드리프트': 1, 'deadlift': 1,
            '03_바벨_스티프_데드리프트': 2, '스티프데드': 2, 'stiff': 2,
            '04_바벨_런지': 3, '바벨런지': 3,
            '05_다이나믹_런지': 4, '다이나믹런지': 4, 'dynamic': 4,
            '06_사이드_런지': 5, '사이드런지': 5, 'side': 5,
            '07_크로스_런지': 6, '크로스런지': 6, 'cross': 6,
            '08_굿모닝': 7, '09_굿모닝': 7, '굿모닝': 7, 'good_morning': 7
        }
        for k, v in folder_names.items():
            if k in ex_str:
                return v

        raise ValueError(f"알 수 없는 운동 종목입니다: '{exercise}'")

    @staticmethod
    def _resolve_view_code(camera_view: str) -> Tuple[str, str]:
        """시점 문자열('시점 A'~'시점 E' 또는 'view1'~'view5')을 내부 코드('view1'~'view5') 및 표준명으로 변환"""
        v_clean = str(camera_view).strip()
        v_code = VIEW_ALIAS_MAP.get(v_clean, "view3")
        meta = VIEW_METADATA.get(v_code, {"code": "시점 C", "name": "정면 (0도)"})
        full_name = f"{meta['code']} ({meta['name']})"
        return v_code, full_name

    def _harmonize_keypoints(
        self,
        keypoints_seq: np.ndarray,
        rtmp_seq: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        입력 키포인트 시퀀스를 64프레임 MediaPipe (64, 33, 3) 및 RTMPose (64, 26, 3) 듀얼 스트림으로 상호 정합
        """
        kps = np.asarray(keypoints_seq, dtype=np.float32)

        # 1. 2D 평탄화 배열인 경우 차원 복원
        if kps.ndim == 2:
            if kps.shape == (64, 98):
                from src.features.kinematics_v6 import KinematicsEngineV6
                kps_15 = KinematicsEngineV6.reconstruct_3d_keypoints_from_raw_b(kps)
                kps = kps_15
            elif kps.shape == (64, 45):
                kps = kps.reshape(64, 15, 3)

        # 2. 64 프레임 시간축 스플라인 보간
        if len(kps) != 64:
            kps = SkeletalTopologyAdapter.resample_to_64_frames(kps)

        # 3. 보조 RTMPose 시퀀스가 직접 전달된 경우
        if rtmp_seq is not None:
            rtm = np.asarray(rtmp_seq, dtype=np.float32)
            if len(rtm) != 64:
                rtm = SkeletalTopologyAdapter.resample_to_64_frames(rtm)
            mp = kps if kps.shape[1] == 33 else SkeletalTopologyAdapter.rtmpose_26_to_mediapipe_33(kps)
            return mp, rtm

        # 4. 단일 시퀀스로부터 상호 역복원
        T, J, C = kps.shape
        if J == 33:
            mp33 = kps[:, :, :3]
            rtm26 = SkeletalTopologyAdapter.mediapipe_33_to_rtmpose_26(mp33)
            return mp33, rtm26
        elif J == 26:
            rtm26 = kps[:, :, :3]
            mp33 = SkeletalTopologyAdapter.rtmpose_26_to_mediapipe_33(rtm26)
            return mp33, rtm26
        elif J == 15:
            # 15관절을 33 및 26으로 패딩 확장
            mp33 = np.zeros((64, 33, 3), dtype=np.float32)
            rtm26 = np.zeros((64, 26, 3), dtype=np.float32)
            for j15_idx, mp_idx in [(2, 11), (3, 12), (4, 13), (5, 14), (6, 15), (7, 16),
                                    (8, 23), (9, 24), (10, 25), (11, 26), (12, 27), (13, 28)]:
                mp33[:, mp_idx] = kps[:, j15_idx, :3]
            rtm26 = SkeletalTopologyAdapter.mediapipe_33_to_rtmpose_26(mp33)
            return mp33, rtm26

        raise ValueError(f"지원하지 않는 관절 토폴로지입니다: J={J} (지원: 15, 26, 33)")

    def diagnose(
        self,
        exercise: Union[int, str],
        keypoints_seq: np.ndarray,
        camera_view: str = "시점 C",
        rtmp_seq: Optional[np.ndarray] = None,
        spine_seq: Optional[np.ndarray] = None,
        mp_vis_seq: Optional[np.ndarray] = None,
        rtm_conf_seq: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        단일 반복(Single Repetition 64프레임)에 대한 2단계 세부 결함 전수 진단 수행
        - 마스터 클래스 없이 개별 세부 결함 모델 직접 호출
        - spine_seq: SpinePose 9대 척추 분절 (64, 9, 3) 3D 좌표 (선택적 주입)
        - mp_vis_seq, rtm_conf_seq: 관절 가시성 및 신뢰도 배열 (KQS 품질 게이팅 평가용)
        - KQS 품질 미달(is_retake=True) 시 2단계 모델 추론을 바이패스하고 재촬영 큐 즉시 직결
        """
        t0 = time.time()

        # 1. 운동 종목 및 촬영 시점 해석
        exercise_id = self._resolve_exercise_id(exercise)
        view_code, view_full_name = self._resolve_view_code(camera_view)

        # 2. 관절 좌표계 듀얼 정합 (MediaPipe 33J + RTMPose 26J)
        mp33, rtm26 = self._harmonize_keypoints(keypoints_seq, rtmp_seq=rtmp_seq)

        # 3. 키포인트 품질 평가(KQS) 사전 게이팅
        from src.web.core.feedback_engine import FeedbackEngine
        fb_engine = FeedbackEngine.get_instance()
        kqs_info = fb_engine.evaluate_keypoint_quality(
            exercise_id=exercise_id,
            mp_vis_seq=mp_vis_seq,
            rtm_conf_seq=rtm_conf_seq,
            threshold=0.60
        )

        is_retake = kqs_info.get('is_retake', False)
        ex_meta = EXERCISE_CATALOG[exercise_id]

        # 4. 결함 모델 평가 또는 품질 미달 시 바이패스
        if is_retake:
            # 관절 가림 및 신뢰도 미달 시: 오진단 방지를 위해 2단계 머신러닝 추론을 안전하게 바이패스
            defect_preds: List[DefectPrediction] = []
            for code in ex_meta.get('defect_codes', []):
                fb_def = fb_engine.get_feedback_def(code)
                name = fb_def.defect_name if fb_def else code
                defect_preds.append(DefectPrediction(
                    defect_code=code,
                    defect_name=name,
                    probability=0.0,
                    state=2,  # 주의/보류 상태
                    state_label="보류",
                    feedback="키포인트 가시성 저하로 판정보류",
                    target_joint=fb_def.target_joint if fb_def else None
                ))
            is_clean = False
        else:
            # 정상 품질 시: 마스터 클래스 없이 세부 결함 모델 직접 호출
            defect_preds: List[DefectPrediction] = self.registry.evaluate_exercise_defects(
                exercise_id=exercise_id,
                mp33_seq=mp33,
                rtm26_seq=rtm26,
                view_code=view_code,
                spine_seq=spine_seq
            )
            # 전체 반복 무결성(Clean 여부) 판정: 모든 결함이 state == 0
            is_clean = all(d.state == 0 for d in defect_preds)

        elapsed_ms = (time.time() - t0) * 1000.0

        # 5. 3단계 다중 결함 우선순위 큐 및 재촬영 게이팅 코칭 피드백 생성
        rep_coaching = fb_engine.prioritize_rep_defects(
            exercise_id=exercise_id,
            defect_predictions=[d.model_dump() for d in defect_preds],
            view_code=view_code,
            kqs_info=kqs_info
        )

        return {
            'exercise_id': exercise_id,
            'exercise_name': ex_meta['name'],
            'exercise_short': ex_meta['short'],
            'selected_view': view_full_name,
            'view_code': view_code,
            'is_clean': is_clean,
            'defects_count': len(defect_preds),
            'defects': [d.model_dump() for d in defect_preds],
            'coaching': rep_coaching,
            'kqs_info': kqs_info,
            'inference_time_ms': round(elapsed_ms, 2)
        }

    def evaluate_single_repetition(
        self,
        exercise: Union[int, str] = 0,
        keypoints_seq: Optional[np.ndarray] = None,
        camera_view: str = "시점 C: 정면",
        rtmp_seq: Optional[np.ndarray] = None,
        spine_seq: Optional[np.ndarray] = None,
        mp_vis_seq: Optional[np.ndarray] = None,
        rtm_conf_seq: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """evaluate_single_repetition 유연 래퍼 (키워드 인자 호환 지원)"""
        ex = kwargs.get("exercise_id", exercise)
        kps = kwargs.get("kps_sequence", keypoints_seq)
        sp = kwargs.get("spine_seq", spine_seq)
        vis = kwargs.get("mp_vis_seq", mp_vis_seq)
        conf = kwargs.get("rtm_conf_seq", rtm_conf_seq)
        return self.diagnose(
            exercise=ex,
            keypoints_seq=kps,
            camera_view=camera_view,
            rtmp_seq=rtmp_seq,
            spine_seq=sp,
            mp_vis_seq=vis,
            rtm_conf_seq=conf
        )


