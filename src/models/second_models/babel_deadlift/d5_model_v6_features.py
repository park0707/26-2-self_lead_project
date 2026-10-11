# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Module: d5_model_v6_features.py
Purpose: v5.1 Pruned(32D) 베이스라인 계승 + 동적 어드레스 영점화(Setup Zero-Calibration)
         및 해부학적 체형 정규화(Anatomical Ratio) 피처 5종 결합 -> 총 37D 최적 피처셋
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v1 import (
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_ELBOW, MP_RIGHT_ELBOW,
    MP_LEFT_WRIST, MP_RIGHT_WRIST, MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE, MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
    MP_LEFT_HEEL, MP_RIGHT_HEEL, MP_LEFT_FOOT_INDEX, MP_RIGHT_FOOT_INDEX,
)
from src.models.second_models.babel_deadlift.d5_model_v5_1 import (
    extract_d5_v5_1_features_from_seq,
    FEATURE_NAMES_32_D5_PRUNED
)

# 신규 추가되는 5종 체형/영점 보정 피처
NEW_V6_FEATURE_NAMES: List[str] = [
    'd5_zero_delta_bar_shank_max',         # [32] 셋업 대비 리프팅 중 바벨-정강이 추가 이탈량 (Δ max)
    'd5_zero_delta_bar_shank_mean',        # [33] 셋업 대비 리프팅 중 바벨-정강이 평균 추가 이탈량 (Δ mean)
    'd5_zero_delta_shoulder_wrist_angle',  # [34] 셋업 대비 어깨-손목 전방 경사각 추가 증가량 (Δ deg)
    'd5_anatomical_arm_to_torso_ratio',    # [35] 개인 체형: 팔 길이 / 몸통 길이 비율 (긴 팔 보정)
    'd5_anatomical_torso_to_body_ratio',   # [36] 개인 체형: 몸통 길이 / 신체 스케일 비율 (긴 상체 보정)
]

FEATURE_NAMES_37_D5_V6: List[str] = FEATURE_NAMES_32_D5_PRUNED + NEW_V6_FEATURE_NAMES


def extract_d5_v6_zero_calibrated_features(kps_arr: np.ndarray, view_name: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 3D 키포인트 배열에서:
    1) 기존 v5.1 32D 최적 피처 추출
    2) 셋업 어드레스 구간(0~10프레임 중 최하사점 직전) 기준 동적 영점화 및 체형비율 5D 피처 추출
    -> 총 37차원 v6 피처 벡터 반환
    """
    # 1. 기존 검증된 32D 피처 추출 (Baseline Clone)
    f32 = extract_d5_v5_1_features_from_seq(kps_arr, view_name=view_name)

    T = kps_arr.shape[0]
    pts = kps_arr[:, :, :3].astype(np.float32)

    shoulder_l = pts[:, MP_LEFT_SHOULDER]
    shoulder_r = pts[:, MP_RIGHT_SHOULDER]
    wrist_l = pts[:, MP_LEFT_WRIST]
    wrist_r = pts[:, MP_RIGHT_WRIST]
    elbow_l = pts[:, MP_LEFT_ELBOW]
    elbow_r = pts[:, MP_RIGHT_ELBOW]
    hip_l = pts[:, MP_LEFT_HIP]
    hip_r = pts[:, MP_RIGHT_HIP]
    knee_l = pts[:, MP_LEFT_KNEE]
    knee_r = pts[:, MP_RIGHT_KNEE]
    ankle_l = pts[:, MP_LEFT_ANKLE]
    ankle_r = pts[:, MP_RIGHT_ANKLE]
    heel_l = pts[:, MP_LEFT_HEEL]
    heel_r = pts[:, MP_RIGHT_HEEL]
    toe_l = pts[:, MP_LEFT_FOOT_INDEX]
    toe_r = pts[:, MP_RIGHT_FOOT_INDEX]

    shoulder = (shoulder_l + shoulder_r) / 2.0
    bar = (wrist_l + wrist_r) / 2.0
    elbow = (elbow_l + elbow_r) / 2.0
    hip = (hip_l + hip_r) / 2.0
    knee = (knee_l + knee_r) / 2.0
    ankle = (ankle_l + ankle_r) / 2.0
    midfoot = (heel_l + heel_r + toe_l + toe_r) / 4.0

    # 신체 스케일
    thigh_len = np.mean(np.linalg.norm(hip - knee, axis=-1))
    shank_len = np.mean(np.linalg.norm(knee - ankle, axis=-1))
    torso_len = np.mean(np.linalg.norm(shoulder - hip, axis=-1)) + 1e-6
    arm_len = np.mean(np.linalg.norm(shoulder - wrist_l, axis=-1) + np.linalg.norm(shoulder - wrist_r, axis=-1)) / 2.0 + 1e-6
    ref_len = float(max(thigh_len + shank_len, 0.2))

    # 위상 탐색 (Nadir)
    bar_y_smooth = gaussian_filter1d(bar[:, 1], sigma=2.0)
    nadir_idx = int(np.argmax(bar_y_smooth))
    knee_y_smooth = gaussian_filter1d(knee[:, 1], sigma=2.0)

    # 셋업 어드레스 구간 (Nadir 기준 직전/직후 안정된 정지 프레임)
    setup_start = max(0, nadir_idx - 6)
    setup_end = min(T, nadir_idx + 4)
    if setup_end <= setup_start:
        setup_slice = slice(0, min(10, T))
    else:
        setup_slice = slice(setup_start, setup_end)

    # 바벨과 정강이 중심 간의 수평(X, Z) 거리 시계열
    shank_mid = (ankle + knee) / 2.0
    horiz_bar_shank_dist = np.sqrt((bar[:, 0] - shank_mid[:, 0])**2 + (bar[:, 2] - shank_mid[:, 2])**2) / ref_len

    # 정강이 통과 구간(바벨이 무릎 아래) 마스크
    shank_zone_mask = (bar_y_smooth >= knee_y_smooth - 0.02)
    if np.any(shank_zone_mask):
        bar_shank_max = float(np.max(horiz_bar_shank_dist[shank_zone_mask]))
        bar_shank_mean = float(np.mean(horiz_bar_shank_dist[shank_zone_mask]))
    else:
        bar_shank_max = float(np.max(horiz_bar_shank_dist))
        bar_shank_mean = float(np.mean(horiz_bar_shank_dist))

    # [핵심 1] 셋업 기준선 (Setup Baseline)
    setup_bar_shank = float(np.mean(horiz_bar_shank_dist[setup_slice]))

    # [피처 32, 33] 동적 영점 차감 상대 변위
    delta_bar_shank_max = float(bar_shank_max - setup_bar_shank)
    delta_bar_shank_mean = float(bar_shank_mean - setup_bar_shank)

    # 어깨-손목 전방 경사각 시계열
    arm_vec = bar - shoulder
    arm_norm = np.linalg.norm(arm_vec, axis=-1) + 1e-6
    # Y축은 아래 방향 (0, 1, 0)
    cos_angle = np.clip(arm_vec[:, 1] / arm_norm, -1.0, 1.0)
    arm_angle_deg = np.arccos(cos_angle) * 180.0 / np.pi

    setup_arm_angle = float(np.mean(arm_angle_deg[setup_slice]))
    max_arm_angle = float(np.max(arm_angle_deg))
    # [피처 34] 셋업 대비 어깨-손목 전방 경사각 추가 변위
    delta_shoulder_wrist_angle = float(max_arm_angle - setup_arm_angle)

    # [피처 35, 36] 개인 체형 정규화 지표
    arm_to_torso_ratio = float(arm_len / torso_len)
    torso_to_body_ratio = float(torso_len / (ref_len + 1e-6))

    f_new = np.array([
        delta_bar_shank_max,
        delta_bar_shank_mean,
        delta_shoulder_wrist_angle,
        arm_to_torso_ratio,
        torso_to_body_ratio
    ], dtype=np.float32)

    return np.concatenate([f32, f_new])


if __name__ == "__main__":
    # 단위 테스트: 더미 시퀀스에 대한 Shape 및 NaN 무결성 검증
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)
    feats = extract_d5_v6_zero_calibrated_features(dummy_seq, view_name="view3")
    print(f"[*] Dummy Unit Test Output Shape: {feats.shape}")
    assert feats.shape[0] == 37, f"Shape mismatch: {feats.shape[0]} != 37"
    assert not np.isnan(feats).any(), "NaN detected in unit test!"
    print("[+] Unit Test PASSED: 37D features extracted cleanly.")
