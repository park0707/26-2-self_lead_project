# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Module: side_lunge_spatial_leg_separator.py
Purpose: 사이드 런지 전수 서브모델(c1~c5) 공통 표준 공간 거리 기반 다리 분리 엔진 (c3 동일 표준)
         - 최하사점: 양 무릎 굴곡 최소 각도(Knee Flexion Min) 시점 탐색
         - 다리 분리: 골반 중심(mid_hip)과의 수평(X축) 공간 거리 복합 지표 (무릎 0.6 + 발목 0.4)
           * 골반 아래 지탱하는 다리 = 앞다리 / 작업 다리 (Lead Leg)
           * 외측으로 멀리 뻗은 다리 = 뒷다리 / 신전 다리 (Trail Leg)
         - 100% 무누수 순수 비전 MediaPipe 33관절 3D 포즈 텐서 기반
"""

from typing import Tuple
import numpy as np

MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28


def detect_side_lunge_lead_leg_spatial(kps_64f: np.ndarray) -> Tuple[bool, int]:
    """
    사이드 런지 전용 표준 공간 거리 다리 분리 엔진 (c3 원천 동일 수식)
    
    Args:
        kps_64f: (64, 33, 3) 또는 (64, 33, C) 넘파이 배열
    
    Returns:
        is_left_lead: bool (True면 왼쪽 다리가 앞/작동 다리, False면 오른쪽 다리)
        bottom_idx: int (최하사점 프레임 인덱스, 0 ~ 63)
    """
    lh = kps_64f[:, MP_LEFT_HIP, :3]
    rh = kps_64f[:, MP_RIGHT_HIP, :3]
    lk = kps_64f[:, MP_LEFT_KNEE, :3]
    rk = kps_64f[:, MP_RIGHT_KNEE, :3]
    la = kps_64f[:, MP_LEFT_ANKLE, :3]
    ra = kps_64f[:, MP_RIGHT_ANKLE, :3]

    mid_hip = (lh + rh) * 0.5

    # 1. 양 무릎 3D 굴곡 각도 시계열 산출 (대퇴 벡터와 정강이 벡터의 사잇각)
    v_femur_l = lk - lh
    v_shank_l = la - lk
    v_femur_r = rk - rh
    v_shank_r = ra - rk

    norm_fl = np.linalg.norm(v_femur_l, axis=-1, keepdims=True) + 1e-7
    norm_sl = np.linalg.norm(v_shank_l, axis=-1, keepdims=True) + 1e-7
    norm_fr = np.linalg.norm(v_femur_r, axis=-1, keepdims=True) + 1e-7
    norm_sr = np.linalg.norm(v_shank_r, axis=-1, keepdims=True) + 1e-7

    dot_l = np.sum((v_femur_l / norm_fl) * (v_shank_l / norm_sl), axis=-1)
    dot_r = np.sum((v_femur_r / norm_fr) * (v_shank_r / norm_sr), axis=-1)

    # 180도 기준 굴곡각
    ang_l = 180.0 - np.degrees(np.arccos(np.clip(dot_l, -1.0, 1.0)))
    ang_r = 180.0 - np.degrees(np.arccos(np.clip(dot_r, -1.0, 1.0)))

    # 2. 최하사점: 양 무릎 중 더 깊게 굽혀진 시점 탐색
    min_combined = np.minimum(ang_l, ang_r)
    bottom_idx = int(np.argmin(min_combined))
    w_start = max(0, bottom_idx - 2)
    w_end = min(64, bottom_idx + 3)

    # 3. 공간 거리 분리: 골반 중심 대비 X축 수평 거리
    dx_knee_l = float(np.mean(np.abs(lk[w_start:w_end, 0] - mid_hip[w_start:w_end, 0])))
    dx_knee_r = float(np.mean(np.abs(rk[w_start:w_end, 0] - mid_hip[w_start:w_end, 0])))
    dx_ank_l = float(np.mean(np.abs(la[w_start:w_end, 0] - mid_hip[w_start:w_end, 0])))
    dx_ank_r = float(np.mean(np.abs(ra[w_start:w_end, 0] - mid_hip[w_start:w_end, 0])))

    dist_l_body = dx_knee_l * 0.6 + dx_ank_l * 0.4
    dist_r_body = dx_knee_r * 0.6 + dx_ank_r * 0.4

    is_left_lead = bool(dist_l_body < dist_r_body)
    return is_left_lead, bottom_idx


def detect_lead_leg_side_str(kps_64f: np.ndarray) -> str:
    """'left' 또는 'right' 문자열 반환 편의 함수"""
    is_left, _ = detect_side_lunge_lead_leg_spatial(kps_64f)
    return 'left' if is_left else 'right'
