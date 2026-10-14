# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/web/pipeline/active_period_gater.py
Purpose: [Tier 1] 생체역학 키네마틱 에너지 및 관절 각도 기반 비운동 구간(언랙/셋업/휴식) 선제 필터링
         및 실제 운동 활성 구간(Active Exercise Period) 정밀 검출 모듈
"""

import logging
from itertools import groupby
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from src.core.skeleton2d import Joint15

logger = logging.getLogger("active_period_gater")


class ActivePeriodGater:
    """
    [Tier 1] 운동 영상의 비운동 구간(언랙/셋업 요동/휴식)을 사전에 필터링하고,
    순수 운동 활성 구간(Active Period: [start_frame, end_frame]) 및 
    정밀 1회 반복(Repetitions)을 식별하는 생체역학 게이트키퍼
    """

    @staticmethod
    def compute_joint_angles(kps_15: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Joint15 2D 좌표로부터 무릎 굴곡 각도, 고관절 힌지 각도, 체간 경사도를 계산
        kps_15: (T, 15, 2)
        """
        def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
            ba = a - b
            bc = c - b
            norm_ba = np.linalg.norm(ba, axis=-1)
            norm_bc = np.linalg.norm(bc, axis=-1)
            denom = norm_ba * norm_bc + 1e-6
            cos_angle = np.sum(ba * bc, axis=-1) / denom
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        # 좌우 무릎 각도 (Hip - Knee - Ankle)
        l_knee = _angle(kps_15[:, Joint15.LEFT_HIP], kps_15[:, Joint15.LEFT_KNEE], kps_15[:, Joint15.LEFT_ANKLE])
        r_knee = _angle(kps_15[:, Joint15.RIGHT_HIP], kps_15[:, Joint15.RIGHT_KNEE], kps_15[:, Joint15.RIGHT_ANKLE])
        min_knee = np.minimum(l_knee, r_knee)

        # 좌우 고관절 힌지 각도 (Torso - Hip - Knee)
        l_hip = _angle(kps_15[:, Joint15.TORSO], kps_15[:, Joint15.LEFT_HIP], kps_15[:, Joint15.LEFT_KNEE])
        r_hip = _angle(kps_15[:, Joint15.TORSO], kps_15[:, Joint15.RIGHT_HIP], kps_15[:, Joint15.RIGHT_KNEE])
        min_hip = np.minimum(l_hip, r_hip)

        # 체간 길이 (Neck - Pelvis)
        torso_vec = kps_15[:, Joint15.NECK] - kps_15[:, Joint15.PELVIS]
        torso_len = np.linalg.norm(torso_vec, axis=-1)

        return {
            "left_knee": l_knee,
            "right_knee": r_knee,
            "min_knee": min_knee,
            "left_hip": l_hip,
            "right_hip": r_hip,
            "min_hip": min_hip,
            "torso_len": torso_len
        }

    @classmethod
    def compute_composite_depth(cls, kps_15: np.ndarray) -> np.ndarray:
        """
        8대 운동에 모두 반응하는 복합 깊이 지표(Depth Signal) 산출
        - 스쿼트/런지: 골반 수직 하강(Pelvis Y) 중심
        - 데드리프트/굿모닝: 골반 하강 + 체간 전방 숙임(Neck-Pelvis Y) 중심
        """
        py = kps_15[:, Joint15.PELVIS, 1].copy()
        neck_y = kps_15[:, Joint15.NECK, 1].copy()

        # 급격한 비정상 1프레임 튀김(Jitter) 보정
        if len(py) > 1 and abs(py[0] - py[1]) > 80:
            py[0] = py[1]
        if len(neck_y) > 1 and abs(neck_y[0] - neck_y[1]) > 80:
            neck_y[0] = neck_y[1]

        pitch_y = neck_y - py  # 체간 숙임 깊이 (숙일수록 neck_y가 커져 pitch_y 증가)

        def _norm(sig: np.ndarray) -> np.ndarray:
            ptp = np.ptp(sig)
            if ptp < 1e-4:
                return np.zeros_like(sig)
            return (sig - np.min(sig)) / (ptp + 1e-6)

        norm_py = _norm(py)
        norm_pitch = _norm(pitch_y)
        return 0.65 * norm_py + 0.35 * norm_pitch

    @classmethod
    def filter_and_segment_repetitions(
        cls,
        kps_15: np.ndarray,
        fps: float = 30.0,
        min_knee_thresh: float = 142.0,
        min_hip_thresh: float = 132.0,
        min_burst_sec: float = 0.35,
        min_rep_duration_sec: float = 0.70
    ) -> Tuple[int, int, List[Tuple[int, int, int]]]:
        """
        [핵심 인터페이스] 비운동 구간(언랙, 셋업 요동, 정지 대기)을 필터링하고
        순수 활성 운동 구간 [active_start, active_end]과
        각 1회 반복 [(start_f, end_f, bottom_f), ...]을 반환

        반환:
          - active_start: int (순수 운동 시작 프레임)
          - active_end: int (순수 운동 종료 프레임)
          - reps: List[Tuple[start_frame, end_frame, bottom_frame]]
        """
        fps = max(1.0, float(fps))
        T = len(kps_15)
        if T == 0:
            logger.warning("입력 관절 배열이 비어있습니다.")
            return 0, 0, []
        if T < int(fps * 2):
            logger.info("입력 시퀀스가 너무 짧습니다 (< 2초). 전체를 단일 구간으로 반환합니다.")
            return 0, max(0, T - 1), [(0, max(0, T - 1), max(0, T // 2))]

        angles = cls.compute_joint_angles(kps_15)
        min_knee = angles["min_knee"]
        min_hip = angles["min_hip"]
        torso_len = angles["torso_len"]
        median_torso = float(np.median(torso_len)) if len(torso_len) > 0 else 300.0

        depth_signal = cls.compute_composite_depth(kps_15)
        pelvis_y = kps_15[:, Joint15.PELVIS, 1]

        # 1. 활성 굴곡 프레임 마스크 (스쿼트/런지 무릎 굽힘 or 데드/굿모닝 힙 힌지)
        active_mask = (min_knee < min_knee_thresh) | (min_hip < min_hip_thresh)

        # 2. 미세 노이즈 버스트 필터링 (0.35초 미만 단기 요동 배제)
        min_burst_frames = max(4, int(fps * min_burst_sec))
        valid_blocks: List[Tuple[int, int]] = []
        for is_active, group in groupby(enumerate(active_mask), key=lambda x: x[1]):
            if is_active:
                items = list(group)
                if len(items) >= min_burst_frames:
                    valid_blocks.append((items[0][0], items[-1][0]))

        # 활성 블록이 전혀 없는 경우 (운동을 하지 않고 서 있는 영상 등)
        if not valid_blocks:
            logger.warning("유효한 운동 굴곡 블록이 감지되지 않았습니다. 전체 구간을 단일 반복으로 폴백합니다.")
            return 0, T - 1, [(0, T - 1, T // 2)]

        # 3. 각 활성 블록 내에서 최하사점(Bottom/Inflection Point) 검출
        bottoms: List[int] = []
        for s, e in valid_blocks:
            bot = s + int(np.argmax(depth_signal[s:e+1]))
            bottoms.append(bot)

        # 0.8초 이내의 중복 피크 병합 (다중 극점 방지)
        min_interval_frames = int(fps * 0.8)
        pruned_bottoms: List[int] = [bottoms[0]]
        for b in bottoms[1:]:
            if b - pruned_bottoms[-1] >= min_interval_frames:
                pruned_bottoms.append(b)
            else:
                # 더 깊은 피크(depth_signal이 더 큰 쪽) 유지
                if depth_signal[b] > depth_signal[pruned_bottoms[-1]]:
                    pruned_bottoms[-1] = b

        # 4. 각 피크를 기준으로 [시작점, 종료점] 경계 정밀 산출 (경계 공유 방식)
        reps: List[Tuple[int, int, int]] = []
        n_bottoms = len(pruned_bottoms)

        for i in range(n_bottoms):
            bot = pruned_bottoms[i]

            # 시작점: 이전 피크와 현재 피크 사이의 기립 복귀점(Standing Lockout)
            if i == 0:
                # 첫 반복: 피크 앞 3.5초 범위 내에서 depth_signal 최솟값 (출발 직전 정지선)
                search_back = max(0, bot - int(fps * 3.5))
                if search_back < bot:
                    start_f = search_back + int(np.argmin(depth_signal[search_back:bot]))
                else:
                    start_f = bot
            else:
                prev_bot = pruned_bottoms[i - 1]
                if prev_bot < bot:
                    start_f = prev_bot + int(np.argmin(depth_signal[prev_bot:bot]))
                else:
                    start_f = bot

            # 종료점: 현재 피크와 다음 피크 사이의 기립 복귀점 (다음 반복의 시작점과 공유)
            if i < n_bottoms - 1:
                next_bot = pruned_bottoms[i + 1]
                if bot < next_bot:
                    end_f = bot + int(np.argmin(depth_signal[bot:next_bot]))
                else:
                    end_f = bot
            else:
                # 마지막 반복: 피크 뒤 3.5초 범위 내에서 기립 완성점 탐색
                search_fwd = min(T, bot + int(fps * 3.5))
                if bot < search_fwd:
                    end_f = bot + int(np.argmin(depth_signal[bot:search_fwd]))
                else:
                    end_f = bot

            # 물리적 유효성 검증 (최소 시간 및 최소 가동범위 필터링)
            duration_sec = (end_f - start_f + 1) / fps
            pelvis_drop = pelvis_y[bot] - min(pelvis_y[start_f], pelvis_y[end_f])
            depth_delta = depth_signal[bot] - min(depth_signal[start_f], depth_signal[end_f])

            # 유효성 가드: 0.7초 이상 지속 + 유의미한 수직 변위(체간의 12% 이상) 또는 깊이 변화
            if duration_sec >= min_rep_duration_sec and (pelvis_drop >= median_torso * 0.12 or depth_delta >= 0.10):
                reps.append((int(start_f), int(end_f), int(bot)))
            else:
                logger.info(f"비운동 미세 요동 필터링: F{start_f}~F{end_f} (지속: {duration_sec:.2f}s, 드롭: {pelvis_drop:.1f}px)")

        if not reps:
            logger.warning("가드 통과 반복이 없습니다. 전체 시퀀스 폴백 적용.")
            return 0, T - 1, [(0, T - 1, T // 2)]

        # 5. 활성 운동 구간 확정
        active_start = reps[0][0]
        active_end = reps[-1][1]

        logger.info(
            f"[Tier 1 ActivePeriodGater] 실제 운동 활성 구간 확정: F{active_start} ~ F{active_end} "
            f"({(active_end - active_start + 1) / fps:.2f}초) | 총 {len(reps)}회 반복 검출"
        )
        return int(active_start), int(active_end), reps
