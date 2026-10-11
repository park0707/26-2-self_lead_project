# -*- coding: utf-8 -*-
"""
NovelFlow: Face & Foot 12D Fine-Grained Biomechanical Feature Extractor
- AI-Hub 24-Joint 3D JSON 데이터를 직접 파싱하여 진짜 얼굴(Nose, Neck, Eyes, Ears) 및 발(Foot, Ankle) 지표 산출
- 순수 운동 반복 구간(orig_start_frame ~ orig_end_frame)만을 엄밀 슬라이싱하여 준비/정리 동작 오차 완전 제거
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

FINE_FEATURE_NAMES_12: List[str] = [
    # 1. 고개 정면 관련 6대 정밀 지표
    'fine_head_gaze_pitch_mean',        # 시선 고도각 평균 (지면 수평 기준 도, 0도 근처 정상)
    'fine_head_gaze_pitch_min',         # 최저 시선 고도각 (바닥 숙임 최댓값)
    'fine_cervical_spine_angle_mean',   # 코-목-등 3D 경추 정렬각 평균 (180도 일자 정상)
    'fine_cervical_spine_angle_min',    # 최저 경추 정렬각 (목 꺾임)
    'fine_head_yaw_asymmetry_mean',     # 좌우 귀 대비 코의 회전각 (고개 돌림)
    'fine_head_acceleration_jerk',      # 머리 3차원 가속도 저더 (고개 흔들림)
    # 2. 발바닥 지면 고정 관련 6대 정밀 지표
    'fine_foot_pitch_left_mean',        # 좌측 발(Ankle-Foot) 지면 경사각 평균 (도)
    'fine_foot_pitch_right_mean',       # 우측 발 지면 경사각 평균
    'fine_foot_pitch_left_max',         # 좌측 뒤꿈치 들림각 최댓값 (Heel Lift)
    'fine_foot_pitch_right_max',        # 우측 뒤꿈치 들림각 최댓값 (Heel Lift)
    'fine_foot_heel_toe_delta_std',     # 발목-발끝 상대 수직 높이 편차의 표준편차 (들썩임)
    'fine_foot_stance_width_drift'      # 양발 사이 거리(Stance Width) 변동폭 (미끄러짐)
]

def calc_angle_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot = np.sum(v1 * v2, axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7
    return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))

def extract_features_from_3d_json_rep(
    json_path: str,
    start_frame: int,
    end_frame: int
) -> Optional[np.ndarray]:
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    total_f = len(frames)
    if total_f < 2:
        return None

    sf = max(0, min(start_frame, total_f - 2))
    ef = max(sf + 2, min(end_frame, total_f))
    rep_frames = frames[sf:ef]
    T = len(rep_frames)
    if T < 2:
        return None

    def get_pt(name: str) -> np.ndarray:
        arr = np.zeros((T, 3), dtype=np.float32)
        for t, rf in enumerate(rep_frames):
            p = rf['pts'].get(name, {'x': 0.0, 'y': 0.0, 'z': 0.0})
            arr[t] = [p['x'], p['y'], p['z']]
        return arr

    nose = get_pt('Nose')
    neck = get_pt('Neck')
    back = get_pt('Back')
    left_ear = get_pt('Left Ear')
    right_ear = get_pt('Right Ear')
    left_ankle = get_pt('Left Ankle')
    right_ankle = get_pt('Right Ankle')
    left_foot = get_pt('Left Foot')
    right_foot = get_pt('Right Foot')

    # 1. 고개 지표
    v_gaze = nose - neck
    gaze_horiz = np.sqrt(v_gaze[:, 0]**2 + v_gaze[:, 2]**2) + 1e-7
    gaze_pitch = np.degrees(np.arctan2(v_gaze[:, 1], gaze_horiz))

    pitch_mean = float(np.mean(gaze_pitch))
    pitch_min = float(np.min(gaze_pitch))

    v_spine = neck - back
    cervical_angle = calc_angle_3d(v_gaze, v_spine)
    cervical_mean = float(np.mean(cervical_angle))
    cervical_min = float(np.min(cervical_angle))

    v_ears = right_ear - left_ear
    ear_horiz = np.degrees(np.arctan2(v_ears[:, 2], v_ears[:, 0] + 1e-7))
    gaze_yaw = np.degrees(np.arctan2(v_gaze[:, 0], v_gaze[:, 2] + 1e-7))
    head_yaw_diff = float(np.mean(np.abs(gaze_yaw - (ear_horiz - 90.0))))

    acc = np.diff(v_gaze, n=2, axis=0) if T >= 3 else np.zeros((1, 3))
    head_jerk = float(np.mean(np.linalg.norm(acc, axis=-1)))

    # 2. 발바닥 접지 지표
    v_foot_l = left_foot - left_ankle
    v_foot_r = right_foot - right_ankle

    foot_pitch_l = np.degrees(np.arctan2(v_foot_l[:, 1], np.abs(v_foot_l[:, 2]) + 1e-7))
    foot_pitch_r = np.degrees(np.arctan2(v_foot_r[:, 1], np.abs(v_foot_r[:, 2]) + 1e-7))

    fp_l_mean = float(np.mean(foot_pitch_l))
    fp_r_mean = float(np.mean(foot_pitch_r))
    fp_l_max = float(np.max(foot_pitch_l))
    fp_r_max = float(np.max(foot_pitch_r))

    diff_y_l = v_foot_l[:, 1]
    diff_y_r = v_foot_r[:, 1]
    heel_toe_std = float((np.std(diff_y_l) + np.std(diff_y_r)) / 2.0)

    stance_dist = np.linalg.norm(left_foot - right_foot, axis=-1)
    stance_drift = float(np.max(stance_dist) - np.min(stance_dist))

    feats = np.array([
        pitch_mean, pitch_min, cervical_mean, cervical_min, head_yaw_diff, head_jerk,
        fp_l_mean, fp_r_mean, fp_l_max, fp_r_max, heel_toe_std, stance_drift
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
