# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지) / Defect: c4 (척추의 중립 위반)
Module: lunge_c4_spinepose_features.py
Purpose: SpinePose 9분절 3D 척추 좌표 (64, 9, 3)로부터 시점 불변(view-invariant) 척추 곡률 피처 추출.
         - 입력은 100% 이미지 기반 SpinePose 출력 (AI-Hub JSON/라벨 미사용)
         - 관절 3D 각도(분절 간 내각)만 사용 -> 카메라 좌표축/시점에 의존하지 않음
Interface:
    extract_lunge_c4_spinepose_features(spine3d, spine2d) -> np.ndarray (len(SPINEPOSE_FEATURE_NAMES),)
    extract_lunge_c4_spinepose_temporal(spine3d)          -> np.ndarray (64, len(SPINEPOSE_TEMPORAL_NAMES))
    SPINEPOSE_FEATURE_NAMES, SPINEPOSE_TEMPORAL_NAMES  (이름 기반 조회용)
관절 인덱스: 0=S1, 1=L5, 2=L3, 3=L1, 4=T8, 5=T3, 6=C7, 7=C4, 8=C1
"""

from typing import List
import numpy as np

_SERIES = ['lumbar', 'thoracic', 'cervical', 'straight', 'bow', 'lean']
_STATS = ['bot', 'min', 'max', 'rng', 'dbt']  # bottom, min, max, range, bottom-minus-top

SPINEPOSE_FEATURE_NAMES: List[str] = [f"sp_{s}_{st}" for s in _SERIES for st in _STATS] + [
    'sp_lumbar_jitter', 'sp_mean_ang_L5_bot'
]
SPINEPOSE_TEMPORAL_NAMES: List[str] = ['sp_lumbar_t', 'sp_thoracic_t', 'sp_cervical_t',
                                       'sp_straight_t', 'sp_bow_t', 'sp_lean_t']


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """b 꼭짓점 내각(deg), 입력 (T,3)."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _series(s: np.ndarray) -> np.ndarray:
    """(64,9,3) -> (64,6) [lumbar, thoracic, cervical, straight, bow, lean]."""
    ang = np.stack([_angle(s[:, k - 1], s[:, k], s[:, k + 1]) for k in range(1, 8)], axis=1)  # (64,7)
    lumbar = ang[:, 0:3].mean(1)
    thoracic = ang[:, 3:5].mean(1)
    cervical = ang[:, 5:7].mean(1)
    seg = np.linalg.norm(np.diff(s, axis=1), axis=-1)  # (64,8)
    chord = np.linalg.norm(s[:, 8] - s[:, 0], axis=-1)
    straight = chord / (seg.sum(1) + 1e-8)
    # 활모양 편차: L3, T8이 S1-C7 현에서 떨어진 최대 수직거리 / 현 길이
    d = s[:, 6] - s[:, 0]
    dn = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
    u = d / dn
    dev = []
    for k in (2, 4):
        w = s[:, k] - s[:, 0]
        proj = (w * u).sum(-1, keepdims=True) * u
        dev.append(np.linalg.norm(w - proj, axis=-1))
    bow = np.max(np.stack(dev, 1), axis=1) / dn[:, 0]
    # 체간 방향 변화: 시작 구간(앞 8프레임) 평균 체간벡터 대비 각도
    ref = u[:8].mean(0)
    ref = ref / (np.linalg.norm(ref) + 1e-8)
    lean = np.degrees(np.arccos(np.clip((u * ref).sum(-1), -1.0, 1.0)))
    return np.stack([lumbar, thoracic, cervical, straight, bow, lean], axis=1)


def _bottom_idx(spine2d: np.ndarray) -> int:
    """SpinePose 2D에서 S1(골반) 픽셀 y가 가장 큰(가장 낮은) 프레임."""
    y = spine2d[:, 0, 1]
    return int(np.argmax(y[8:56]) + 8) if len(y) == 64 else int(np.argmax(y))


def extract_lunge_c4_spinepose_features(spine3d: np.ndarray, spine2d: np.ndarray) -> np.ndarray:
    s = np.nan_to_num(np.asarray(spine3d, dtype=np.float64))
    ser = _series(s)                      # (64,6)
    b = _bottom_idx(np.asarray(spine2d, dtype=np.float64))
    lo, hi = max(0, b - 2), min(64, b + 3)
    top = ser[:8].mean(0)
    feats = []
    for j in range(ser.shape[1]):
        x = ser[:, j]
        bot = x[lo:hi].mean()
        feats += [bot, x.min(), x.max(), x.max() - x.min(), bot - top[j]]
    ang_L5 = _angle(s[:, 0], s[:, 1], s[:, 2])
    feats += [np.abs(np.diff(ser[:, 0])).mean(), ang_L5[lo:hi].mean()]
    out = np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    assert out.shape[0] == len(SPINEPOSE_FEATURE_NAMES)
    return out


def extract_lunge_c4_spinepose_temporal(spine3d: np.ndarray) -> np.ndarray:
    """TCN 추가 채널: 고정 스케일 정규화(데이터셋 통계 비의존, 누수 없음)."""
    ser = _series(np.nan_to_num(np.asarray(spine3d, dtype=np.float64)))
    scale = np.array([180.0, 180.0, 180.0, 1.0, 0.5, 90.0])
    return np.nan_to_num((ser / scale).astype(np.float32))
