# -*- coding: utf-8 -*-
"""
NovelFlow 스쿼트 최하사점(Bottom Turnaround) 생체역학 피처 추출 모듈 (Solution 3)
- ECCV 2022 Fitness-AQA 및 CVPR 2022/2023 RepCount/CoRe 기반
- Format B (16F x 92D) 시계열로부터 최하사점(변곡점) 8종 핵심 생체역학 지표 산출
- 기존 Format A (161D)와 결합하여 169차원 확장 피처셋 생성
"""

import numpy as np
from typing import Tuple, List

BOTTOM_FEATURE_NAMES_8 = [
    'bottom_knee_to_ankle_ratio',          # 최하사점 무릎간격/발목간격 비 (무릎 모임/벌어짐)
    'bottom_torso_shin_parallelism_diff',   # 최하사점 척추각-정강이각 평행도 오차
    'bottom_knee_valgus_mean',             # 최하사점 좌우 무릎 외반각 평균
    'bottom_knee_angle_mean',              # 최하사점 무릎 굴곡 각도 (스쿼트 깊이)
    'bottom_neck_spine_error',             # 최하사점 경추-흉추-골반 일직선 오차
    'bottom_3frame_accel',                 # 최하사점 전후 3프레임 수직 반환 가속도
    'bottom_knee_ankle_dist_diff',         # 최하사점 무릎간격 - 발목간격 차이 (cm 단위)
    'bottom_knee_asymmetry'                # 최하사점 좌우 무릎 굴곡 비대칭도
]

def extract_squat_bottom_biomechanics(X_b: np.ndarray) -> np.ndarray:
    """
    Format B (N, 16, 92)에서 각 반복의 최하사점 8대 생체역학 지표 산출 -> Shape: (N, 8)
    
    Format B 채널 맵:
      - 56: knee_angle_l_3d, 57: knee_angle_r_3d
      - 64: torso_pitch_3d
      - 66: knee_valgus_l_3d, 67: knee_valgus_r_3d
      - 69: knee_distance_3d, 70: ankle_distance_3d
      - 82: neck_spine_alignment_3d
      - 89: shank_verticality_angle_l_3d, 90: shank_verticality_angle_r_3d
    """
    N, T, C = X_b.shape
    knee_mean_all = (X_b[:, :, 56] + X_b[:, :, 57]) / 2.0 # (N, 16)
    bottom_indices = np.argmin(knee_mean_all, axis=1)      # (N,)

    feats = np.zeros((N, 8), dtype=np.float32)

    for i in range(N):
        b = bottom_indices[i]
        
        # 1. 최하사점 무릎 간격 / 발목 간격 비율
        kd = X_b[i, b, 69]
        ad = X_b[i, b, 70]
        feats[i, 0] = kd / (ad + 1e-5)

        # 2. 최하사점 척추각과 정강이각의 평행도 차이
        tp = X_b[i, b, 64]
        sv = (X_b[i, b, 89] + X_b[i, b, 90]) / 2.0
        feats[i, 1] = np.abs(tp - sv)

        # 3. 최하사점 좌우 무릎 외반각 평균
        feats[i, 2] = (X_b[i, b, 66] + X_b[i, b, 67]) / 2.0

        # 4. 최하사점 무릎 굴곡 각도 (깊이)
        feats[i, 3] = knee_mean_all[i, b]

        # 5. 최하사점 경추-흉추-골반 일직선(180도) 오차
        feats[i, 4] = np.abs(180.0 - X_b[i, b, 82])

        # 6. 최하사점 전후 3프레임 수직 반환 가속도 (바운싱/안정성)
        b_prev = max(0, b - 1)
        b_next = min(T - 1, b + 1)
        vel_prev = knee_mean_all[i, b] - knee_mean_all[i, b_prev]
        vel_next = knee_mean_all[i, b_next] - knee_mean_all[i, b]
        feats[i, 5] = vel_next - vel_prev

        # 7. 최하사점 무릎 간격 - 발목 간격 차이
        feats[i, 6] = kd - ad

        # 8. 최하사점 좌우 무릎 굴곡 각도 비대칭도
        feats[i, 7] = np.abs(X_b[i, b, 56] - X_b[i, b, 57])

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats

def build_augmented_squat_features(X_a: np.ndarray, X_b: np.ndarray, base_feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Format A (161D) + 최하사점 8종 피처 -> 169D 확장 피처 행렬 생성
    """
    feats_8 = extract_squat_bottom_biomechanics(X_b)
    X_aug = np.concatenate([X_a, feats_8], axis=-1).astype(np.float32)
    
    if base_feature_names is not None:
        aug_names = list(base_feature_names) + BOTTOM_FEATURE_NAMES_8
    else:
        aug_names = [f"f_{i}" for i in range(X_a.shape[1])] + BOTTOM_FEATURE_NAMES_8
        
    return X_aug, aug_names
