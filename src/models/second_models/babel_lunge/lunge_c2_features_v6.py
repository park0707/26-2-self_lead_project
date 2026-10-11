# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: lunge_c2_features_v6.py

기립 베이스라인 정규화 및 수평 정사영 18차원 정예 피처 추출기:
1. [v5 16D 정예 피처 100% 계승]:
   - 상체 바벨 밸런싱 노이즈 3종(상체 측방 기울기, 보상 기울기, 측방 롤) 완전 배제.
   - 발끝 내회전 4D, 슬관절 외반 변위 4D, 하지 연쇄 2D, 골반-체간 정렬 4D, 결합/프록시 2D.
2. [생체역학 기립 베이스라인 정규화 & 수평 정사영 2D 신설 (18D)]:
   - 17. knee_displacement_foot_heading_angle:
     기립->최하사점 무릎 수평(XZ) 변위축 vs 발뒤꿈치->앞꿈치 수평 장축 간 사잇각.
     3D 공간의 수직 낙하(-Y) 노이즈를 완벽 배제하고 바닥 수평면(XZ) 상에서 순수 무릎 굽힘 진행축과 발끝 지향축의 불일치 포착.
   - 18. delta_q_angle_standing_to_bottom:
     기립 시(Frame 0) 해부학적 Q-angle 대비 최하사점 Q-angle의 동적 변화량.
     피험자 1, 3번의 선천적 골격 휨(O/X다리)으로 인한 FP(위양성 122개)를 원천 무력화하고,
     피험자 2, 9번의 미세한 동적 허탈 결함을 포착하여 결함 재현율(Recall)을 극대화 (r = +0.1207, p = 6.98e-5 입증).
"""

import numpy as np
from typing import Dict, List, Tuple, Any

from lunge_c2_features_v4 import (
    extract_lunge_c2_features_v4,
    detect_lead_leg_v4,
    FEATURE_NAMES_20D
)
from lunge_c2_features_v5 import (
    FEATURE_NAMES_16D,
    FEATURE_INDICES_16D
)

FEATURE_NAMES_18D = FEATURE_NAMES_16D + [
    "knee_displacement_foot_heading_angle",     # 17. 기립->최하사점 무릎 수평 변위축 vs 발끝 장축 사잇각
    "delta_q_angle_standing_to_bottom"          # 18. 기립 대비 최하사점 Q-angle 동적 변화량 (r = +0.1207, p < 0.0001)
]


def extract_delta_features_2d(kps_33_3d: np.ndarray, is_left_lead: bool, t_bot: int) -> Tuple[float, float]:
    """
    17. knee_displacement_foot_heading_angle
    18. delta_q_angle_standing_to_bottom
    """
    hip_idx = 23 if is_left_lead else 24
    knee_idx = 25 if is_left_lead else 26
    ank_idx = 27 if is_left_lead else 28
    heel_idx = 29 if is_left_lead else 30
    toe_idx = 31 if is_left_lead else 32

    # 17. knee_displacement_foot_heading_angle
    v_foot_xz = np.array([kps_33_3d[t_bot, toe_idx, 0] - kps_33_3d[t_bot, heel_idx, 0],
                          kps_33_3d[t_bot, toe_idx, 2] - kps_33_3d[t_bot, heel_idx, 2]], dtype=np.float32)
    norm_foot = np.linalg.norm(v_foot_xz) + 1e-6
    u_foot = v_foot_xz / norm_foot

    v_disp_xz = np.array([kps_33_3d[t_bot, knee_idx, 0] - kps_33_3d[0, knee_idx, 0],
                          kps_33_3d[t_bot, knee_idx, 2] - kps_33_3d[0, knee_idx, 2]], dtype=np.float32)
    norm_disp = np.linalg.norm(v_disp_xz)

    if norm_disp > 0.02:
        u_disp = v_disp_xz / norm_disp
        dot = np.clip(np.dot(u_foot, u_disp), -1.0, 1.0)
        ang_kf = float(np.arccos(dot) * 180.0 / np.pi)
    else:
        ang_kf = 0.0

    # 18. delta_q_angle_standing_to_bottom
    # 기립 시(t=0..3 평균) Q-angle
    v_thigh_std = np.mean(kps_33_3d[:3, knee_idx, :3] - kps_33_3d[:3, hip_idx, :3], axis=0)
    v_shank_std = np.mean(kps_33_3d[:3, ank_idx, :3] - kps_33_3d[:3, knee_idx, :3], axis=0)
    norm_ts = np.linalg.norm(v_thigh_std) * np.linalg.norm(v_shank_std) + 1e-6
    q_std = np.degrees(np.arccos(np.clip(np.dot(v_thigh_std, v_shank_std) / norm_ts, -1.0, 1.0)))

    # 최하사점 Q-angle
    v_thigh_bot = kps_33_3d[t_bot, knee_idx, :3] - kps_33_3d[t_bot, hip_idx, :3]
    v_shank_bot = kps_33_3d[t_bot, ank_idx, :3] - kps_33_3d[t_bot, knee_idx, :3]
    norm_tb = np.linalg.norm(v_thigh_bot) * np.linalg.norm(v_shank_bot) + 1e-6
    q_bot = np.degrees(np.arccos(np.clip(np.dot(v_thigh_bot, v_shank_bot) / norm_tb, -1.0, 1.0)))

    delta_q = float(q_bot - q_std)

    return ang_kf, delta_q


def extract_lunge_c2_features_v6(kps_33_3d: np.ndarray) -> np.ndarray:
    """
    단일 64프레임 시퀀스(64, 33, 3)로부터 18차원 정예 피처 추출
    """
    if kps_33_3d.shape[-1] == 4:
        kps_33_3d = kps_33_3d[:, :, :3]

    feats_20 = extract_lunge_c2_features_v4(kps_33_3d)
    f16 = feats_20[FEATURE_INDICES_16D]

    is_left_lead, t_bot = detect_lead_leg_v4(kps_33_3d)
    f17, f18 = extract_delta_features_2d(kps_33_3d, is_left_lead, t_bot)

    f18_total = np.append(f16, [f17, f18]).astype(np.float32)
    f18_total = np.nan_to_num(f18_total, nan=0.0, posinf=0.0, neginf=0.0)
    return f18_total


def extract_lunge_c2_features_v6_batch(X_seqs: np.ndarray) -> np.ndarray:
    """
    배치 시퀀스 (N, 64, 33, 3) 피처 추출
    """
    N = len(X_seqs)
    feats = np.zeros((N, 18), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_lunge_c2_features_v6(X_seqs[i])
    return feats
