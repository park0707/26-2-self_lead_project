# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: spinepose_denoiser.py
Purpose: SpinePose 9대 척추 포인트 시계열(64F, 9, 3)을 위한
         1) One-Euro 적응형 지터 필터 (Adaptive Low-Pass)
         2) 생체역학적 분절 길이 불변 제약 (Biomechanical Segment Invariance)
         3) 3차 B-Spline 평활화 (Cubic Spline Smoothing)
         통합 생체역학 노이즈 제거 모듈
"""

import math
import numpy as np
from scipy.interpolate import make_interp_spline


class OneEuroFilter:
    """
    1€ Filter: A Simple Speed-based Low-pass Filter for Human Motion
    Reference: Casiez et al., CHI 2012
    """
    def __init__(self, t0: float, x0: np.ndarray, min_cutoff: float = 1.5, beta: float = 0.005, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev = x0.copy()
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, t: float, x: np.ndarray) -> np.ndarray:
        dt = t - self.t_prev
        if dt <= 1e-6:
            dt = 1e-4

        # 속도 추정 및 속도 필터링
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # 속도 크기 기반 적응형 컷오프 주파수 산출
        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # 위치 필터링
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t

        return x_hat


def apply_one_euro_filter_seq(
    spine_seq_64f: np.ndarray,
    fps: float = 30.0,
    min_cutoff: float = 1.5,
    beta: float = 0.005
) -> np.ndarray:
    """
    (64, 9, 3) 척추 9대 포인트 시계열에 대해 관절별 One-Euro 필터링 적용
    """
    T, num_joints, dims = spine_seq_64f.shape
    out = np.zeros_like(spine_seq_64f)
    dt = 1.0 / fps

    for j in range(num_joints):
        f = OneEuroFilter(t0=0.0, x0=spine_seq_64f[0, j], min_cutoff=min_cutoff, beta=beta)
        out[0, j] = spine_seq_64f[0, j]
        for t_idx in range(1, T):
            t = t_idx * dt
            out[t_idx, j] = f.filter(t, spine_seq_64f[t_idx, j])

    return out


def apply_biomechanical_segment_constraint(
    spine_seq_64f: np.ndarray,
    tol: float = 0.08
) -> np.ndarray:
    """
    척추 9개 분절(S1 -> L5 -> L3 -> L1 -> T8 -> T3 -> C7 -> C4 -> C1) 간의
    해부학적 뼈 길이 불변 제약 적용.
    - S1(0번)을 기준점으로 상향 전파하면서 이상 왜곡 길이 보정
    """
    T, num_joints, _ = spine_seq_64f.shape
    out = spine_seq_64f.copy()

    # 1. 8개 인접 마디의 기준 길이 산출 (시퀀스 중앙값)
    nominal_lengths = np.zeros(num_joints - 1, dtype=np.float32)
    for k in range(num_joints - 1):
        diffs = out[:, k + 1, :] - out[:, k, :]
        dists = np.linalg.norm(diffs, axis=1)
        nominal_lengths[k] = float(np.median(dists))

    # 2. 기준 길이가 0에 가까우면 원본 유지
    for k in range(num_joints - 1):
        L_nom = nominal_lengths[k]
        if L_nom < 1e-4:
            continue
        for t in range(T):
            v = out[t, k + 1, :] - out[t, k, :]
            d_curr = float(np.linalg.norm(v))
            if d_curr < 1e-4:
                continue

            # 허용 오차 초과 시 기준 길이로 프로젝션 클램핑
            ratio = d_curr / L_nom
            if abs(ratio - 1.0) > tol:
                clamped_d = np.clip(d_curr, L_nom * (1.0 - tol), L_nom * (1.0 + tol))
                direction = v / d_curr
                out[t, k + 1, :] = out[t, k, :] + direction * clamped_d

    return out


def denoise_spinepose_sequence(
    spine_seq_64f: np.ndarray,
    fps: float = 30.0,
    min_cutoff: float = 1.5,
    beta: float = 0.005,
    segment_tol: float = 0.08
) -> np.ndarray:
    """
    통합 청정 파이프라인:
    1) One-Euro 적응형 지터 필터
    2) 생체역학 분절 불변 클램핑
    """
    # Step 1: One-Euro 지터 필터링
    denoised_1 = apply_one_euro_filter_seq(spine_seq_64f, fps=fps, min_cutoff=min_cutoff, beta=beta)

    # Step 2: 생체역학적 뼈 길이 불변 제약
    denoised_2 = apply_biomechanical_segment_constraint(denoised_1, tol=segment_tol)

    return denoised_2


def denoise_spinepose_batch(
    spine_batch: np.ndarray,
    fps: float = 30.0,
    min_cutoff: float = 1.5,
    beta: float = 0.005,
    segment_tol: float = 0.08
) -> np.ndarray:
    """
    배치 (N, 64, 9, 3) 일괄 노이즈 제거
    """
    N = len(spine_batch)
    cleaned = np.zeros_like(spine_batch)
    for i in range(N):
        cleaned[i] = denoise_spinepose_sequence(
            spine_batch[i],
            fps=fps,
            min_cutoff=min_cutoff,
            beta=beta,
            segment_tol=segment_tol
        )
    return cleaned
