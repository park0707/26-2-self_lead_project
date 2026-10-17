# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/pipeline/skeleton_preprocessor.py
Purpose: [Tier 2] 골격 표준 전처리 허브 (Skeleton Preprocessor Hub)
         - 1회 반복 단위 슬라이싱 및 AGENTS.md 제2조 표준 64프레임 3차 스플라인(Cubic Spline) 보간
         - MediaPipe 33관절과 RTMPose 26관절을 각각의 순수 원본 좌표계로 보존
         - PreprocessedSkeletons 불변 표준 컨테이너 생성 및 반환
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional, Union, List, Dict, Any

from src.web.core.contracts import RawVisionLandmarks, PreprocessedSkeletons


class SkeletonPreprocessorAPI:
    """골격 표준 전처리 API 허브"""

    @staticmethod
    def resample_to_64f_spline(seq: np.ndarray) -> np.ndarray:
        """
        임의 길이 시퀀스 (T, J, C) 또는 (T, D)를 정확히 64프레임으로 3차 스플라인(Cubic Spline) 보간
        (AGENTS.md 제2조 규정 및 T<=3 극단적 엣지 케이스 완벽 방어)
        """
        seq = np.asarray(seq, dtype=np.float32)
        T = len(seq)
        if T == 64:
            return seq

        if T <= 1:
            if T == 0:
                raise ValueError("입력 시퀀스가 비어 있습니다 (T=0).")
            # 1프레임 복제 확장
            return np.repeat(seq, 64, axis=0)

        orig_times = np.linspace(0, 1, T)
        target_times = np.linspace(0, 1, 64)

        if T < 4:
            # 3차 스플라인은 최소 4개 점 필요 -> 선형 보간 적용
            flat = seq.reshape(T, -1)
            resampled = np.zeros((64, flat.shape[1]), dtype=np.float32)
            for d in range(flat.shape[1]):
                resampled[:, d] = np.interp(target_times, orig_times, flat[:, d])
            return resampled.reshape((64,) + seq.shape[1:])

        # 3차 스플라인(Cubic Spline, bc_type='natural') 적용
        flat = seq.reshape(T, -1)
        cs = CubicSpline(orig_times, flat, axis=0, bc_type='natural')
        resampled = cs(target_times)
        return resampled.reshape((64,) + seq.shape[1:]).astype(np.float32)

    @classmethod
    def preprocess_repetition(
        cls,
        raw_mp: Union[RawVisionLandmarks, np.ndarray],
        raw_rtm: Optional[Union[RawVisionLandmarks, np.ndarray]] = None,
        raw_spine: Optional[Union[RawVisionLandmarks, np.ndarray]] = None,
        rep_start_frame: int = 0,
        rep_end_frame: Optional[int] = None,
        view_code: str = "view3",
        exercise_id: int = 0,
        rep_index: int = 1
    ) -> PreprocessedSkeletons:
        """
        원시 비전 시퀀스 또는 랜드마크 객체로부터 1회 반복 구간을 슬라이싱하여 64F 전처리 컨테이너 구축
        """
        # 1. MediaPipe 33 추출 및 슬라이싱
        mp_kps = raw_mp.keypoints if isinstance(raw_mp, RawVisionLandmarks) else np.asarray(raw_mp, dtype=np.float32)
        total_mp_f = len(mp_kps)
        
        end_f = rep_end_frame if rep_end_frame is not None else (total_mp_f - 1)
        start_f = max(0, min(rep_start_frame, total_mp_f - 1))
        end_f = max(start_f, min(end_f, total_mp_f - 1))

        # 반복 구간 슬라이싱
        mp_slice = mp_kps[start_f : max(start_f + 4, end_f + 1)]
        mp33_64f = cls.resample_to_64f_spline(mp_slice)

        # 2. RTMPose 26 추출 및 슬라이싱 (옵션)
        rtm26_64f = None
        if raw_rtm is not None:
            rtm_kps = raw_rtm.keypoints if isinstance(raw_rtm, RawVisionLandmarks) else np.asarray(raw_rtm, dtype=np.float32)
            if len(rtm_kps) > 0:
                if len(rtm_kps) == 64 and rep_end_frame is None:
                    rtm26_64f = rtm_kps
                else:
                    rtm_slice = rtm_kps[min(start_f, len(rtm_kps)-1) : max(start_f + 4, min(end_f + 1, len(rtm_kps)))]
                    rtm26_64f = cls.resample_to_64f_spline(rtm_slice)

        # 3. SpinePose 9 추출 및 슬라이싱 (옵션)
        spine9_64f = None
        if raw_spine is not None:
            spine_kps = raw_spine.keypoints if isinstance(raw_spine, RawVisionLandmarks) else np.asarray(raw_spine, dtype=np.float32)
            if len(spine_kps) > 0:
                if len(spine_kps) == 64 and rep_end_frame is None:
                    spine9_64f = spine_kps
                else:
                    sp_slice = spine_kps[min(start_f, len(spine_kps)-1) : max(start_f + 4, min(end_f + 1, len(spine_kps)))]
                    spine9_64f = cls.resample_to_64f_spline(sp_slice)

        return PreprocessedSkeletons(
            mp33_64f=mp33_64f,
            rtm26_64f=rtm26_64f,
            spine9_64f=spine9_64f,
            view_code=view_code,
            exercise_id=exercise_id,
            rep_index=rep_index,
            start_frame=start_f,
            end_frame=end_f
        )

    @classmethod
    def create_from_64f_tensors(
        cls,
        mp33_64f: np.ndarray,
        rtm26_64f: Optional[np.ndarray] = None,
        spine9_64f: Optional[np.ndarray] = None,
        view_code: str = "view3",
        exercise_id: int = 0,
        rep_index: int = 1
    ) -> PreprocessedSkeletons:
        """이미 64프레임으로 보간된 텐서들로부터 컨테이너를 직접 생성하는 편의 팩토리"""
        mp = cls.resample_to_64f_spline(mp33_64f)
        rtm = cls.resample_to_64f_spline(rtm26_64f) if rtm26_64f is not None else None
        sp = cls.resample_to_64f_spline(spine9_64f) if spine9_64f is not None else None

        return PreprocessedSkeletons(
            mp33_64f=mp,
            rtm26_64f=rtm,
            spine9_64f=sp,
            view_code=view_code,
            exercise_id=exercise_id,
            rep_index=rep_index
        )
