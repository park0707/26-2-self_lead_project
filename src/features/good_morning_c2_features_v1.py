# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 굿모닝 (Good Morning - 09)
Defect: c2 (시선 방향 유지 오류 - Gaze direction maintenance error)
Module: good_morning_c2_features_v1.py
Purpose: 굿모닝 시선 방향 유지 오류 모델 (gm2_v1)을 위한 생체역학 도메인 피처셋 (간접 측정 방식 적용)
         - 얼굴 피쳐(코, 눈)가 가려지는 후면/측후면 시점에서도 동작하도록 어깨(Shoulder), 귀(Ear), 골반(Hip) 기반 간접 측정 피처 구축
         - Trunk-Neck Alignment Angle, Global Torso vs Head Inclination 등 선행연구 기반 피처 적용
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.signal import savgol_filter

# MediaPipe 33 Landmark Indices
MP_NOSE = 0
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28

GOOD_MORNING_C2_FEATURE_NAMES_V1: List[str] = [
    # [그룹 1: 경추 중립성 / Trunk-Neck Alignment Angle (5D)]
    "trunk_neck_angle_min",            # [ 1] 흉동-경추 정렬 각도 최소값 (중립시 180도 부근)
    "trunk_neck_angle_max",            # [ 2] 흉동-경추 정렬 각도 최대값
    "trunk_neck_angle_mean",           # [ 3] 흉동-경추 정렬 각도 평균
    "trunk_neck_angle_var",            # [ 4] 흉동-경추 정렬 각도 분산 (변동성)
    "trunk_neck_angle_bottom",         # [ 5] 최저점(최대 굴곡)에서의 흉동-경추 정렬 각도

    # [그룹 2: 글로벌 체간 및 머리 기울기 (Torso vs Head Inclination) (5D)]
    "head_torso_inclination_diff_max", # [ 6] 머리와 체간 기울기 차이 최대값
    "head_torso_inclination_diff_min", # [ 7] 머리와 체간 기울기 차이 최소값
    "head_torso_inclination_diff_bot", # [ 8] 최저점에서의 머리-체간 기울기 차이
    "head_inclination_bottom",         # [ 9] 최저점에서의 글로벌 머리 기울기
    "torso_inclination_bottom",        # [10] 최저점에서의 글로벌 체간 기울기

    # [그룹 3: Ear-to-Shoulder 상대 변위 (목의 강체 유지도) (2D)]
    "ear_shoulder_dist_mean",          # [11] 귀-어깨 거리 평균 (체간 정규화)
    "ear_shoulder_dist_var",           # [12] 귀-어깨 거리 분산 (목을 뻗거나 당기는지 감지)

    # [그룹 4: 힌지(Hinge) 기본 동작 통제 및 기타 (3D)]
    "hip_flexion_angle_min",           # [13] 최저점에서의 고관절 굴곡 각도
    "knee_flexion_angle_mean",         # [14] 무릎 굴곡 각도 평균
    "bottom_dwell_ratio",              # [15] 최저점 체류 비율
]


def vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 180.0
    dot = np.dot(v1, v2)
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def safe_savgol_filter(arr: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    if len(arr) < window:
        return arr.copy()
    try:
        return savgol_filter(arr, window_length=window, polyorder=poly, mode='interp')
    except Exception:
        return arr.copy()


def extract_good_morning_c2_features_v1(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    굿모닝 c2(시선 방향 유지) 피처 추출
    """
    # 1. 미드포인트 계산 (어깨, 고관절, 귀)
    mid_shoulder = (kps_64f[:, MP_LEFT_SHOULDER] + kps_64f[:, MP_RIGHT_SHOULDER]) / 2.0
    mid_hip = (kps_64f[:, MP_LEFT_HIP] + kps_64f[:, MP_RIGHT_HIP]) / 2.0
    
    # 귀는 한쪽이 가려질 수 있으므로, 시점에 따라 안전한 귀 좌표를 쓴다. (단순 평균도 3D 추정치에서는 안정적)
    mid_ear = (kps_64f[:, MP_LEFT_EAR] + kps_64f[:, MP_RIGHT_EAR]) / 2.0
    mid_knee = (kps_64f[:, MP_LEFT_KNEE] + kps_64f[:, MP_RIGHT_KNEE]) / 2.0
    mid_ankle = (kps_64f[:, MP_LEFT_ANKLE] + kps_64f[:, MP_RIGHT_ANKLE]) / 2.0

    # 2. 기준점 (최대 굴곡점 = 어깨 Y 좌표가 가장 낮아지는 지점 또는 고관절 각도가 최소가 되는 지점)
    # MediaPipe 좌표계는 Y가 아래로 향할수록 커지므로 어깨 Y값이 최대인 지점이 최저점(Bottom)
    inflection_idx = int(np.argmax(mid_shoulder[:, 1]))
    inflection_idx = max(5, min(inflection_idx, 58))

    trunk_neck_angles = np.zeros(64, dtype=np.float32)
    head_inclinations = np.zeros(64, dtype=np.float32)
    torso_inclinations = np.zeros(64, dtype=np.float32)
    ear_shoulder_dists = np.zeros(64, dtype=np.float32)
    
    hip_flexions = np.zeros(64, dtype=np.float32)
    knee_flexions = np.zeros(64, dtype=np.float32)
    
    vertical_axis = np.array([0.0, -1.0, 0.0]) # 위쪽 방향 벡터 (Y 역방향)

    torso_lengths = []
    
    for f in range(64):
        # 몸통 길이 (정규화용)
        t_len = np.linalg.norm(mid_shoulder[f] - mid_hip[f])
        if t_len > 1e-4:
            torso_lengths.append(t_len)
            
        # 벡터 계산
        trunk_vec = mid_shoulder[f] - mid_hip[f]
        neck_vec = mid_ear[f] - mid_shoulder[f]
        
        # Trunk-Neck Angle (경추 정렬 각도)
        trunk_neck_angles[f] = vector_angle_3d(trunk_vec, neck_vec)
        
        # 글로벌 기울기
        torso_inclinations[f] = vector_angle_3d(trunk_vec, vertical_axis)
        head_inclinations[f] = vector_angle_3d(neck_vec, vertical_axis)
        
        # 상대 변위
        ear_shoulder_dists[f] = np.linalg.norm(neck_vec)
        
        # 기본 관절 각도
        v_femur = mid_knee[f] - mid_hip[f]
        v_torso_down = mid_hip[f] - mid_shoulder[f]
        hip_flexions[f] = vector_angle_3d(v_femur, v_torso_down)
        
        v_tibia = mid_ankle[f] - mid_knee[f]
        knee_flexions[f] = vector_angle_3d(-v_femur, v_tibia)

    torso_scale = float(np.median(torso_lengths)) if torso_lengths else 1.0

    # 스무딩
    trunk_neck_angles_sg = safe_savgol_filter(trunk_neck_angles)
    diff_inclinations = head_inclinations - torso_inclinations
    
    # [그룹 1]
    trunk_neck_angle_min = float(np.min(trunk_neck_angles_sg))
    trunk_neck_angle_max = float(np.max(trunk_neck_angles_sg))
    trunk_neck_angle_mean = float(np.mean(trunk_neck_angles_sg))
    trunk_neck_angle_var = float(np.var(trunk_neck_angles_sg))
    trunk_neck_angle_bottom = float(trunk_neck_angles_sg[inflection_idx])
    
    # [그룹 2]
    head_torso_inclination_diff_max = float(np.max(diff_inclinations))
    head_torso_inclination_diff_min = float(np.min(diff_inclinations))
    head_torso_inclination_diff_bot = float(diff_inclinations[inflection_idx])
    head_inclination_bottom = float(head_inclinations[inflection_idx])
    torso_inclination_bottom = float(torso_inclinations[inflection_idx])
    
    # [그룹 3]
    norm_es_dists = ear_shoulder_dists / torso_scale
    ear_shoulder_dist_mean = float(np.mean(norm_es_dists))
    ear_shoulder_dist_var = float(np.var(norm_es_dists))
    
    # [그룹 4]
    hip_flexion_angle_min = float(np.min(safe_savgol_filter(hip_flexions)))
    knee_flexion_angle_mean = float(np.mean(knee_flexions))
    
    bottom_threshold = np.max(mid_shoulder[:, 1]) * 0.95
    bottom_dwell_ratio = float(np.mean(mid_shoulder[:, 1] >= bottom_threshold))

    features = {
        "trunk_neck_angle_min": trunk_neck_angle_min,
        "trunk_neck_angle_max": trunk_neck_angle_max,
        "trunk_neck_angle_mean": trunk_neck_angle_mean,
        "trunk_neck_angle_var": trunk_neck_angle_var,
        "trunk_neck_angle_bottom": trunk_neck_angle_bottom,
        
        "head_torso_inclination_diff_max": head_torso_inclination_diff_max,
        "head_torso_inclination_diff_min": head_torso_inclination_diff_min,
        "head_torso_inclination_diff_bot": head_torso_inclination_diff_bot,
        "head_inclination_bottom": head_inclination_bottom,
        "torso_inclination_bottom": torso_inclination_bottom,
        
        "ear_shoulder_dist_mean": ear_shoulder_dist_mean,
        "ear_shoulder_dist_var": ear_shoulder_dist_var,
        
        "hip_flexion_angle_min": hip_flexion_angle_min,
        "knee_flexion_angle_mean": knee_flexion_angle_mean,
        "bottom_dwell_ratio": bottom_dwell_ratio,
    }

    return features


def extract_good_morning_c2_feature_vector_v1(
    kps_64f: np.ndarray,
    vis_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    feat_dict = extract_good_morning_c2_features_v1(kps_64f, vis_64f)
    vec = np.array([feat_dict[name] for name in GOOD_MORNING_C2_FEATURE_NAMES_V1], dtype=np.float32)
    return vec
