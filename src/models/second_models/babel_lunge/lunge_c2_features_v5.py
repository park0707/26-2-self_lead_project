# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: lunge_c2_features_v5.py

상체 노이즈(Trunk Canting/Counterbalance/Roll) 완전 제거 16차원 정예 피처 추출기:
1. [노이즈 제거]: 바벨 중량 밸런싱으로 인해 정상인도 크게 발생하는 상체 횡방향 기울기 3종 전면 배제
   - trunk_canting_rom (제거)
   - trunk_counterbalance_score (제거)
   - decoupled_trunk_lateral_lean_bottom (제거)
2. [핵심 신호 보존]: 하지 11자 기계축 및 체간-발끝 진행 방향 사잇각 16D 엄선
   - 발끝 내회전 4D (lead_foot_inward_angle, stability, ratio, heel_toe_skew)
   - 슬관절 동적 외반 4D (valgus_peak, lateral_ratio, velocity_ml, alignment)
   - 하지 연쇄 2D (fppa_delta, thigh_foot_misalignment)
   - 골반 수평축 3D (pelvic_rotation, hip_adduction, inter_knee_dist)
   - 방향 기준축 2D (trunk_pelvis_yaw_torsion, trunk_foot_angle_deviation)
   - 가림 극복 SOTA 1D (dkcd_medial_deviation_bottom)
"""

import numpy as np
from typing import Dict, List, Tuple, Any

from lunge_c2_features_v4 import (
    extract_lunge_c2_features_v4,
    FEATURE_NAMES_20D
)

# 16D 정예 피처 명칭
FEATURE_NAMES_16D = [
    # Group A: 선행 발 지향 및 족부 안정성 (4D)
    "lead_foot_inward_angle_bottom",        # 1. 최하사점 선행 발목-발끝 벡터와 체간 정면 법선의 수평면 내각
    "lead_foot_axis_stability",             # 2. 하강/상승 구간 발끝 방향 벡터의 요동 분산
    "lead_foot_inward_ratio",               # 3. 발끝이 12° 이상 안쪽으로 꺾여 있는 프레임 비율
    "lead_heel_toe_skew",                   # 4. 발뒤꿈치 대비 발끝의 내측 편차 거리비

    # Group B: 슬관절 동적 정렬 및 외반 변위 (4D)
    "lead_knee_valgus_peak",                # 5. 최하사점 부근 무릎 외반 최대 피크 각도
    "lead_knee_lateral_ratio",              # 6. 골반 너비 대비 선행 무릎의 횡방향 이탈 비율
    "lead_knee_velocity_ml",                # 7. 최하사점 진입 시 무릎 관절의 내측 횡방향 이동 속도
    "lead_knee_ankle_alignment",            # 8. 무릎 중심과 발목 중심의 수직 정렬 오차

    # Group C: 관상면 동적 변화량 및 하지 연쇄 (2D)
    "fppa_eccentric_to_bottom_delta",       # 9. 신전 상태 대비 최하사점에서의 FPPA 각도 붕괴량
    "directional_misalignment_thigh_foot",  # 10. 신체 직교계 상 대퇴 진행 벡터 vs 발 장축 사잇각

    # Group D: 체간 진행 방향 및 골반 회전 (2D)
    "trunk_pelvis_yaw_torsion",             # 11. 흉곽 평면 법선과 골반 축 법선 간의 수평면 비틀림 각도
    "pelvic_rotation_bottom",               # 12. 최하사점에서 골반 축의 수평 회전 편차각

    # Group E: 상하체 결합 및 진행 사잇각 (3D)
    "lead_hip_adduction_bottom",            # 13. 선행 대퇴골의 관상면 내전 각도
    "trunk_foot_angle_deviation",           # 14. 체간 전방 벡터와 선행 발 장축 간의 3차원 입체 편차각
    "inter_knee_dist_ratio",                # 15. 양 무릎 간 3D 이격거리 대비 골반 너비 비율

    # Group F [SOTA]: 하지 기계축 편위 프록시 (1D)
    "dkcd_medial_deviation_bottom"          # 16. 고관절-발목 기계적 축 대비 무릎 중심 내측 변위량 (DKCD)
]

# 원천 20D 피처 인덱스 매핑
_V4_NAMES = FEATURE_NAMES_20D[:19]
FEATURE_INDICES_16D = [_V4_NAMES.index(name) for name in FEATURE_NAMES_16D]


def extract_lunge_c2_features_v5(kps_33_3d: np.ndarray) -> np.ndarray:
    """
    단일 64프레임 시퀀스(64, 33, 3)로부터 16차원 정예 피처 추출
    """
    feats_20 = extract_lunge_c2_features_v4(kps_33_3d)
    return feats_20[FEATURE_INDICES_16D]


def extract_lunge_c2_features_v5_batch(X_seqs: np.ndarray) -> np.ndarray:
    """
    배치 시퀀스 (N, 64, 33, 3) 피처 추출
    """
    N = len(X_seqs)
    feats = np.zeros((N, 16), dtype=np.float32)
    for i in range(N):
        feats[i] = extract_lunge_c2_features_v5(X_seqs[i])
    return feats
