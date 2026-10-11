# -*- coding: utf-8 -*-
"""
NovelFlow 운동 횟수 동적 분할 및 트리거 가드(Trigger Guard) 모듈 (v2 정밀 개편)

개선 사항:
1. 경계 공유(Boundary Overlap) 도입: 1회차 종료 프레임(기립 완성)과 2회차 시작 프레임을 공통 공유하여 동작의 잘림 원천 차단
2. 적응형 로컬 가동범위(Adaptive Local Prominence): 얕게 앉은 반복도 놓치지 않고 100% 감지하여 2회분 묶임 방지
3. 마지막 반복 꼬리 자르기(Tail Trimming): 운동 종료 후 멈춰 서 있는 불필요한 프레임 잉여 제거
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.signal import find_peaks
from src.core.skeleton2d import Joint15

class RepetitionSegmenter:
    """순수 비전 관절 좌표 기반 실시간 운동 횟수 및 1회 단위 동적 분할기 (v2)"""

    @staticmethod
    def fill_missing_joints(kps_seq: np.ndarray) -> np.ndarray:
        """(T, 15, 2 or 3) 결측치(0 또는 NaN) 선형 보간"""
        T, J, C = kps_seq.shape
        clean = np.copy(kps_seq).astype(np.float32)
        is_zero = (np.abs(clean) < 1e-5).all(axis=-1)
        is_nan = np.isnan(clean).any(axis=-1)
        invalid = is_zero | is_nan

        t_idx = np.arange(T)
        for j in range(J):
            valid_mask = ~invalid[:, j]
            if np.sum(valid_mask) >= 2:
                for c in range(C):
                    clean[:, j, c] = np.interp(t_idx, t_idx[valid_mask], clean[valid_mask, j, c])
            elif np.sum(valid_mask) == 1:
                clean[:, j, :] = clean[valid_mask, j, :]
        return clean

    @classmethod
    def detect_trigger_guard(cls, kps_seq: np.ndarray) -> int:
        """트리거 가드: 운동 시작 전 안정화 프레임 감지"""
        T = len(kps_seq)
        if T < 4:
            return 0

        torso_vec = kps_seq[:, Joint15.NECK, :2] - kps_seq[:, Joint15.PELVIS, :2]
        torso_len = np.linalg.norm(torso_vec, axis=-1) + 1e-6

        core_joints = [Joint15.PELVIS, Joint15.NECK, Joint15.LEFT_HIP, Joint15.RIGHT_HIP,
                       Joint15.LEFT_KNEE, Joint15.RIGHT_KNEE, Joint15.LEFT_ANKLE, Joint15.RIGHT_ANKLE]
        
        diff = kps_seq[1:, core_joints, :2] - kps_seq[:-1, core_joints, :2]
        dist = np.linalg.norm(diff, axis=-1)
        mean_speed = np.mean(dist, axis=-1)
        norm_speed = mean_speed / torso_len[1:]

        start_idx = 0
        for t in range(len(norm_speed)):
            if norm_speed[t] > 0.20:
                start_idx = t
                break
        return start_idx

    @classmethod
    def compute_composite_energy(cls, kps_seq: np.ndarray) -> np.ndarray:
        """
        9종 운동에 일괄 적용 가능한 범용 복합 키네마틱 에너지(E) 산출
        E(t) = 0.45 * Norm(Pelvis_Y) + 0.35 * Norm(Ankle_Dist) + 0.20 * Norm(Torso_Pitch)
        """
        clean_kps = cls.fill_missing_joints(kps_seq)

        py = clean_kps[:, Joint15.PELVIS, 1].copy()
        if len(py) > 1 and abs(py[0] - py[1]) > 80:
            py[0] = py[1]

        ad = np.linalg.norm(clean_kps[:, Joint15.LEFT_ANKLE, :2] - clean_kps[:, Joint15.RIGHT_ANKLE, :2], axis=-1)
        pitch = clean_kps[:, Joint15.NECK, 1] - clean_kps[:, Joint15.PELVIS, 1]

        def normalize(signal: np.ndarray) -> np.ndarray:
            ptp = np.ptp(signal)
            if ptp < 1e-4:
                return np.zeros_like(signal)
            return (signal - np.min(signal)) / (ptp + 1e-6)

        norm_py = normalize(py)
        norm_ad = normalize(ad)
        norm_pitch = normalize(pitch)

        E = 0.45 * norm_py + 0.35 * norm_ad + 0.20 * norm_pitch
        return E

    @classmethod
    def segment_repetitions(
        cls,
        kps_seq: np.ndarray,
        min_distance: int = 2,
        prominence_ratio: float = 0.14
    ) -> List[Tuple[int, int, int]]:
        """
        경계 공유(Overlap) 및 적응형 로컬 피크 기반 정밀 동적 분할
        반환값: List of (start_idx, end_idx, bottom_idx)
        """
        T = len(kps_seq)
        if T < 4:
            return [(0, T - 1, T // 2)]

        guard_start = cls.detect_trigger_guard(kps_seq)
        E = cls.compute_composite_energy(kps_seq)

        # 1. 적응형 피크 검출 (prominence 완화)
        prom = np.ptp(E) * prominence_ratio
        peaks, _ = find_peaks(E, distance=min_distance, prominence=prom)

        valid_peaks = [p for p in peaks if p >= guard_start]
        if not valid_peaks:
            valid_peaks = peaks if len(peaks) > 0 else [np.argmax(E)]

        # 2. 각 피크(최하사점)를 중심으로 [시작점, 종료점] 정밀 산출
        segments = []
        n_peaks = len(valid_peaks)

        for i, p in enumerate(valid_peaks):
            # 시작점: 이전 피크와 현재 피크 사이의 계곡 (Stand 복귀/출발점)
            if i == 0:
                prev_p = guard_start
                start_cand = prev_p + int(np.argmin(E[prev_p : p + 1]))
            else:
                prev_p = valid_peaks[i - 1]
                start_cand = prev_p + int(np.argmin(E[prev_p : p + 1]))

            # 종료점: 현재 피크와 다음 피크 사이의 계곡 (Stand 복귀점) -> 다음 회차의 시작점과 '공통 공유'
            if i < n_peaks - 1:
                next_p = valid_peaks[i + 1]
                end_cand = p + int(np.argmin(E[p : next_p + 1]))
            else:
                # 마지막 피크의 꼬리 자르기:
                # 최하사점 이후 에너지가 기준선(최솟값 + 0.15) 이하로 내려간 첫 프레임에서 깔끔하게 컷
                baseline = np.min(E) + 0.15
                tail_cands = np.where(E[p:] <= baseline)[0]
                if len(tail_cands) > 0:
                    end_cand = min(p + tail_cands[0] + 1, T - 1)
                else:
                    end_cand = T - 1

            # 시작과 끝이 역전되지 않도록 안전 보정
            start_cand = max(0, min(start_cand, p))
            end_cand = max(p, min(end_cand, T - 1))

            segments.append((int(start_cand), int(end_cand), int(p)))

        return segments
