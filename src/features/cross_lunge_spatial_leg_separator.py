# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Module: cross_lunge_spatial_leg_separator.py
Purpose: 크로스 런지 전용 3D 공간 기구학 기반 앞다리(Lead/Front Leg) 및 최하사점(t_bot) 탐색 엔진
         - 최하사점: 양 무릎 굴곡 각도 최소점(Knee Flexion Min) 및 골반 하강(Hip Y Max) 결합 탐색
         - 다리 분리: 앞쪽에서 체중을 지탱하는 앞다리(Lead/Front Leg) vs 뒤쪽 대각선으로 교차하는 뒷다리(Crossing/Trail Leg)
           * 앞다리: 바닥에 뒤꿈치가 확고히 밀착되며 전방(카메라/골반 기준 상대적 Z 전방)에 위치
           * 뒷다리: 후방으로 교차 굴곡되며 뒤꿈치가 들려 있음 (상대적 Z 후방)
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
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_TOE = 31
MP_RIGHT_TOE = 32


def detect_cross_lunge_lead_leg_spatial(kps_64f: np.ndarray) -> Tuple[bool, int]:
    """
    크로스 런지 전용 기구학적 앞다리(Lead Leg) 및 최하사점(Bottom Index) 탐지기
    
    Args:
        kps_64f: (64, 33, 3) 또는 (64, 33, C) 넘파이 배열
        
    Returns:
        is_left_lead: bool (True면 왼쪽 다리가 앞다리/지지다리, False면 오른쪽 다리가 앞다리)
        bottom_idx: int (최하사점 프레임 인덱스, 0 ~ 63)
    """
    lh = kps_64f[:, MP_LEFT_HIP, :3]
    rh = kps_64f[:, MP_RIGHT_HIP, :3]
    lk = kps_64f[:, MP_LEFT_KNEE, :3]
    rk = kps_64f[:, MP_RIGHT_KNEE, :3]
    la = kps_64f[:, MP_LEFT_ANKLE, :3]
    ra = kps_64f[:, MP_RIGHT_ANKLE, :3]
    lhe = kps_64f[:, MP_LEFT_HEEL, :3]
    rhe = kps_64f[:, MP_RIGHT_HEEL, :3]

    mid_hip = (lh + rh) * 0.5

    # 1. 양 무릎 3D 굴곡 각도 시계열 산출 (Femur vs Shank)
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

    ang_l = 180.0 - np.degrees(np.arccos(np.clip(dot_l, -1.0, 1.0)))
    ang_r = 180.0 - np.degrees(np.arccos(np.clip(dot_r, -1.0, 1.0)))

    # 2. 최하사점(t_bot): 무릎 굴곡 최대점과 골반 하강(mid_hip Y 최대)의 융합
    min_combined_ang = np.minimum(ang_l, ang_r)
    bot_ang_idx = int(np.argmin(min_combined_ang))
    bot_hip_idx = int(np.argmax(mid_hip[:, 1]))

    # 둘의 가중 결합 (가장 굴곡이 깊은 유효 구간 5~58 프레임 클램핑)
    bottom_idx = int(round(bot_ang_idx * 0.6 + bot_hip_idx * 0.4))
    bottom_idx = max(5, min(58, bottom_idx))

    # 3. 최하사점 부근 윈도우(±2프레임)에서 앞다리(Lead Leg) 판정
    w_start = max(0, bottom_idx - 2)
    w_end = min(64, bottom_idx + 3)

    # 3-1. Z축 전후 깊이 (MediaPipe에서 Z가 작을수록 카메라 쪽 전방)
    # 골반 중심 대비 발목의 상대적 Z 위치
    rel_z_l = float(np.mean(la[w_start:w_end, 2] - mid_hip[w_start:w_end, 2]))
    rel_z_r = float(np.mean(ra[w_start:w_end, 2] - mid_hip[w_start:w_end, 2]))

    # 3-2. 뒤꿈치 바닥 밀착도 (Y 좌표: 클수록 바닥 쪽)
    heel_y_l = float(np.mean(lhe[w_start:w_end, 1]))
    heel_y_r = float(np.mean(rhe[w_start:w_end, 1]))

    # 3-3. 무릎 전방 돌출도 (Z 위치)
    knee_z_l = float(np.mean(lk[w_start:w_end, 2]))
    knee_z_r = float(np.mean(rk[w_start:w_end, 2]))

    # 전방 스코어: Z가 더 앞쪽(-Z)이고 뒤꿈치가 더 바닥에 밀착(+Y)될수록 앞다리
    score_l = -rel_z_l * 1.5 - knee_z_l * 0.5 + (heel_y_l - heel_y_r) * 0.5
    score_r = -rel_z_r * 1.5 - knee_z_r * 0.5 + (heel_y_r - heel_y_l) * 0.5

    is_left_lead = bool(score_l >= score_r)
    return is_left_lead, bottom_idx
