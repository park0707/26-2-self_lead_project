# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (08_크로스_런지)
Defect: c3 (상체 정면 균형 잡기 오류 / Torso Sagittal & Coronal Postural Defect)
Module: cross_lunge_c3_features.py
Purpose:
  - 크로스 런지 3번 결함("상체 정면 균형잡기" 미흡) 판정을 위한 22차원 정예 생체역학 도메인 피처 추출기.
  - 100% 순수 비전 MediaPipe 33관절 3D (64, 33, 3) 및 15관절 호환.
  - 결함 유형 커버리지:
    1) 상체 뒤로 과도 젖힘 (Key 178, 180, 182 - Sagittal Posterior Lean / Hyperextension)
    2) 머리 젖혀 하늘 보기 (Key 183 - Cervical Spine Hyperextension / Head Pitch Defect)
    3) 몸통 옆으로 구부리기 (Key 184 - Coronal Lateral Flexion / Shoulder Tilt)
    4) 상체 좌측 회전/비틀림 (Key 184 - Transverse Torso Twist & Yaw Asymmetry)
    5) 5대 카메라 시점(A, B, C, D, E) 투영 편향 자가 보정을 위한 기하학적 정규화
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional, Union


FEATURE_NAMES_22D: List[str] = [
    # [1] 상체 시상면 전후 기울기 & 과신전 지표 (5D)
    'torso_sagittal_lean_bottom',             # 1. 최하사점 상체 시상면 전후 기울기 각도 (°) (정상: ~10°전방, 결함: 후방젖힘 음수/과신전)
    'torso_sagittal_lean_min',                # 2. 전 시퀀스 상체 후방 최대 젖힘 각도 (°)
    'torso_sagittal_lean_bottom_win',        # 3. 최하사점 전후 5프레임 상체 전후각 평균 (°)
    'torso_posterior_defect_margin',          # 4. 기준 정상 전방 굴곡각 대비 후방 이탈 결손 마진 (°)
    'torso_sagittal_rom',                     # 5. 상체 시상면 흔들림 가동범위 (max - min) (°)

    # [2] 두부 및 경추 과신전 지표 ("하늘 보고") (4D)
    'head_neck_pitch_bottom',                 # 6. 최하사점 두부(Nose-Ear)의 수직 상향 앙각 (°) (하늘 볼수록 증가)
    'head_neck_alignment_bottom',             # 7. 머리-목-체간 3D 일직선 정렬각 (°) (정상: ~170-180, 하늘보고 꺾임: <145)
    'head_neck_pitch_max',                    # 8. 전 시퀀스 머리 수직 상향 최대 각도 (°)
    'head_to_torso_relative_pitch',           # 9. 체간 벡터 대비 두부 지향각의 상대 굴곡/신전 각도 (°)

    # [3] 체간 관상면(좌우) 측면 굴곡 지표 ("몸통 옆으로 구부리고") (4D)
    'torso_coronal_tilt_bottom',              # 10. 최하사점 체간 좌우 측면 기울기 각도 (|Roll|) (°)
    'torso_coronal_tilt_max',                 # 11. 전 시퀀스 체간 좌우 최대 기울기 각도 (°)
    'shoulder_coronal_tilt_bottom',           # 12. 최하사점 양 어깨 수평 대비 기울기 각도 (°) (어깨 처짐)
    'shoulder_hip_lateral_offset_bottom',     # 13. 최하사점 어깨 중심의 골반 대비 횡방향(X) 변위 (체간길이 정규화)

    # [4] 상체 횡단면 회전 & 비틀림 지표 ("왼쪽 보고") (4D)
    'shoulder_pelvis_twist_bottom',           # 14. 최하사점 어깨축과 골반축 간 3D 비틀림 사이각 (°)
    'shoulder_pelvis_twist_max',              # 15. 전 시퀀스 어깨-골반축 최대 비틀림 각도 (°)
    'torso_yaw_deviation_bottom',             # 16. 최하사점 상체 법선 벡터의 정면(Z축) 방위각 편차 (°)
    'head_yaw_deviation_bottom',              # 17. 최하사점 시선 벡터의 좌우 편향각 ("왼쪽 보고") (°)

    # [5] 동역학적 안정성 및 각가속도 지표 (3D)
    'torso_lean_velocity_bottom',             # 18. 최하사점 상체 전후각 각속도 (deg/frame)
    'torso_lean_accel_bottom',                # 19. 최하사점 상체 전후각 각가속도 (deg/frame^2)
    'torso_stability_std_bottom',             # 20. 최하사점 윈도우 체간 각도 표준편차 (흔들림/자세 동요도)

    # [6] 시점 불변 기하학적 앵커 지표 (2D)
    'pelvis_width_to_torso_ratio',            # 21. 골반폭 대비 체간길이 비율 (카메라 방위각/거리 추정용)
    'lead_knee_angle_bottom'                  # 22. 최하사점 굴곡 무릎 3D 각도 (운동 깊이 기준 앵커) (°)
]


def _calc_vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 3D 벡터 간 사이각 (0° ~ 180°) 계산"""
    norm1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-7
    norm2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-7
    dot = np.sum((v1 / norm1) * (v2 / norm2), axis=-1)
    cos_theta = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _calc_cosine_similarity_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 3D 벡터 간 코사인 유사도 (-1.0 ~ 1.0)"""
    norm1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-7
    norm2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-7
    dot = np.sum((v1 / norm1) * (v2 / norm2), axis=-1)
    return np.clip(dot, -1.0, 1.0)


def _calc_joint_angle_3d(p_prox: np.ndarray, p_joint: np.ndarray, p_dist: np.ndarray) -> np.ndarray:
    """세 3D 관절 좌표가 이루는 사잇각 (0° ~ 180°) 계산"""
    v1 = p_prox - p_joint
    v2 = p_dist - p_joint
    return _calc_vector_angle_3d(v1, v2)


def extract_cross_lunge_c3_features_22d(kps_seq: np.ndarray) -> np.ndarray:
    """
    단일 64프레임 관절 좌표 배열로부터 22D 정예 크로스 런지 c3 피처 추출
    Input:
        kps_seq: (64, 33, 3) MediaPipe 33관절 3D 또는 (64, 15, 3) 15관절
    Output:
        feat_vec: (22,) np.float32 배열
    """
    assert kps_seq.ndim == 3 and kps_seq.shape[0] == 64, f"입력 규격 불일치: {kps_seq.shape}"
    num_joints = kps_seq.shape[1]

    if num_joints == 33:
        nose = kps_seq[:, 0, :3]
        lear = kps_seq[:, 7, :3]
        rear = kps_seq[:, 8, :3]
        ls = kps_seq[:, 11, :3]
        rs = kps_seq[:, 12, :3]
        lh = kps_seq[:, 23, :3]
        rh = kps_seq[:, 24, :3]
        lk = kps_seq[:, 25, :3]
        rk = kps_seq[:, 26, :3]
        la = kps_seq[:, 27, :3]
        ra = kps_seq[:, 28, :3]
    elif num_joints >= 15:
        nose = kps_seq[:, 0, :3]
        lear = kps_seq[:, 3, :3] if num_joints > 3 else kps_seq[:, 0, :3]
        rear = kps_seq[:, 4, :3] if num_joints > 4 else kps_seq[:, 0, :3]
        ls = kps_seq[:, 5, :3]
        rs = kps_seq[:, 6, :3]
        lh = kps_seq[:, 8, :3]
        rh = kps_seq[:, 9, :3]
        lk = kps_seq[:, 10, :3]
        rk = kps_seq[:, 11, :3]
        la = kps_seq[:, 12, :3]
        ra = kps_seq[:, 13, :3]
    else:
        raise ValueError(f"지원하지 않는 관절 수: {num_joints}")

    # 1. 신체 기본 중심 및 벡터 산출
    mid_hip = (lh + rh) / 2.0
    mid_shoulder = (ls + rs) / 2.0
    mid_ear = (lear + rear) / 2.0
    torso_vec = mid_shoulder - mid_hip  # (64, 3), 힙->어깨 벡터
    torso_len = np.linalg.norm(torso_vec, axis=-1, keepdims=True) + 1e-6

    # 2. 양 무릎 3D 각도 산출 및 최하사점(Bottom Phase) 동기화
    lk_angles = _calc_joint_angle_3d(lh, lk, la)
    rk_angles = _calc_joint_angle_3d(rh, rk, ra)
    min_knee_curve = np.minimum(lk_angles, rk_angles)

    # Savitzky-Golay 스무딩으로 최하사점 안정적 탐색
    try:
        smooth_knee = savgol_filter(min_knee_curve, window_length=11, polyorder=2)
    except Exception:
        smooth_knee = min_knee_curve

    bottom_idx = int(np.argmin(smooth_knee))
    # 바닥 윈도우 인덱스 (±3 프레임 클램핑)
    win_start = max(0, bottom_idx - 3)
    win_end = min(64, bottom_idx + 4)

    # 3. 상체 시상면 전후 각도 시계열 (Sagittal Lean Angle)
    # MediaPipe: Y축이 하향(+), 수직 상향은 -Y.
    # 전후 각도: Y-Z 평면에서 상향 수직 벡터 [0, -1] 대비 Z축 기울기
    # 전방 굴곡(앞으로 숙임) -> 양수(+), 후방 젖힘(Hyperextension) -> 음수(-)
    sagittal_lean = np.degrees(np.arctan2(torso_vec[:, 2], -torso_vec[:, 1] + 1e-6))
    try:
        smooth_sagittal = savgol_filter(sagittal_lean, window_length=9, polyorder=2)
    except Exception:
        smooth_sagittal = sagittal_lean

    torso_sagittal_lean_bottom = float(smooth_sagittal[bottom_idx])
    torso_sagittal_lean_min = float(np.min(smooth_sagittal))  # 가장 뒤로 많이 젖혀진 각도
    torso_sagittal_lean_bottom_win = float(np.mean(smooth_sagittal[win_start:win_end]))
    # 기준 정상 전방 굴곡(약 +10도) 대비 후방 이탈 결손 마진
    torso_posterior_defect_margin = float(max(0.0, 10.0 - torso_sagittal_lean_bottom))
    torso_sagittal_rom = float(np.ptp(smooth_sagittal))

    # 4. 두부 및 경추 과신전 지표 ("하늘 보고")
    # 두부 벡터: mid_ear -> nose
    head_vec = nose - mid_ear
    # 머리 수직 앙각 (Pitch): 수평선 대비 위로 쳐든 각도
    horiz_dist = np.linalg.norm(head_vec[:, [0, 2]], axis=-1) + 1e-6
    head_pitch = np.degrees(np.arctan2(-head_vec[:, 1], horiz_dist))  # -Y가 상향이므로 위로 들면 양수 증가
    try:
        smooth_head_pitch = savgol_filter(head_pitch, window_length=9, polyorder=2)
    except Exception:
        smooth_head_pitch = head_pitch

    head_neck_pitch_bottom = float(smooth_head_pitch[bottom_idx])
    head_neck_pitch_max = float(np.max(smooth_head_pitch))

    # 머리-목-체간 3D 일직선 정렬각 (Nose -> Mid-Shoulder -> Mid-Hip)
    neck_alignment = _calc_joint_angle_3d(nose, mid_shoulder, mid_hip)
    head_neck_alignment_bottom = float(neck_alignment[bottom_idx])

    # 체간 대비 두부의 상대 지향각
    rel_pitch = _calc_vector_angle_3d(head_vec, torso_vec)
    head_to_torso_relative_pitch = float(rel_pitch[bottom_idx])

    # 5. 체간 관상면(좌우) 측면 굴곡 지표 ("몸통 옆으로 구부리고")
    # 체간 X축 기울기 (Roll): |X| / (-Y)
    coronal_tilt = np.degrees(np.arctan2(np.abs(torso_vec[:, 0]), -torso_vec[:, 1] + 1e-6))
    torso_coronal_tilt_bottom = float(coronal_tilt[bottom_idx])
    torso_coronal_tilt_max = float(np.max(coronal_tilt))

    # 양 어깨 수평 기울기 (Shoulder Roll)
    shoulder_diff = rs - ls
    shoulder_roll = np.degrees(np.arctan2(np.abs(shoulder_diff[:, 1]), np.linalg.norm(shoulder_diff[:, [0, 2]], axis=-1) + 1e-6))
    shoulder_coronal_tilt_bottom = float(shoulder_roll[bottom_idx])

    # 어깨 중심의 골반 대비 횡방향(X) 변위
    lat_offset = np.abs(mid_shoulder[:, 0] - mid_hip[:, 0]) / torso_len[:, 0]
    shoulder_hip_lateral_offset_bottom = float(lat_offset[bottom_idx])

    # 6. 상체 횡단면 회전 & 비틀림 지표 ("왼쪽 보고")
    pelvis_axis = rh - lh
    shoulder_axis = rs - ls
    twist_angles = _calc_vector_angle_3d(shoulder_axis, pelvis_axis)
    shoulder_pelvis_twist_bottom = float(twist_angles[bottom_idx])
    shoulder_pelvis_twist_max = float(np.max(twist_angles))

    # 상체 전면 법선 벡터의 Z축 이탈 각도 (Torso Yaw)
    torso_normal = np.cross(shoulder_axis, torso_vec)
    torso_yaw = np.degrees(np.arctan2(np.abs(torso_normal[:, 0]), np.abs(torso_normal[:, 2]) + 1e-6))
    torso_yaw_deviation_bottom = float(torso_yaw[bottom_idx])

    # 시선(Head) 벡터의 좌우 편향각 (Head Yaw)
    head_yaw = np.degrees(np.arctan2(np.abs(head_vec[:, 0]), np.abs(head_vec[:, 2]) + 1e-6))
    head_yaw_deviation_bottom = float(head_yaw[bottom_idx])

    # 7. 동역학적 안정성 및 각속도/각가속도
    vel = np.gradient(smooth_sagittal)
    acc = np.gradient(vel)
    torso_lean_velocity_bottom = float(np.abs(vel[bottom_idx]))
    torso_lean_accel_bottom = float(np.abs(acc[bottom_idx]))
    torso_stability_std_bottom = float(np.std(smooth_sagittal[win_start:win_end]))

    # 8. 시점 불변 기하학적 앵커 지표
    pelvis_w = np.linalg.norm(rh - lh, axis=-1)
    pelvis_ratio = float(np.mean(pelvis_w / (torso_len[:, 0] + 1e-6)))
    lead_knee_angle = float(smooth_knee[bottom_idx])

    feat_vec = np.array([
        torso_sagittal_lean_bottom,
        torso_sagittal_lean_min,
        torso_sagittal_lean_bottom_win,
        torso_posterior_defect_margin,
        torso_sagittal_rom,

        head_neck_pitch_bottom,
        head_neck_alignment_bottom,
        head_neck_pitch_max,
        head_to_torso_relative_pitch,

        torso_coronal_tilt_bottom,
        torso_coronal_tilt_max,
        shoulder_coronal_tilt_bottom,
        shoulder_hip_lateral_offset_bottom,

        shoulder_pelvis_twist_bottom,
        shoulder_pelvis_twist_max,
        torso_yaw_deviation_bottom,
        head_yaw_deviation_bottom,

        torso_lean_velocity_bottom,
        torso_lean_accel_bottom,
        torso_stability_std_bottom,

        pelvis_ratio,
        lead_knee_angle
    ], dtype=np.float32)

    # NaN / Inf 안전 클리핑
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=180.0, neginf=-180.0)
    return feat_vec


if __name__ == '__main__':
    # 합성 더미 시퀀스로 피처 추출기 테스트
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)
    feats = extract_cross_lunge_c3_features_22d(dummy_seq)
    print(f"Feature Vector Shape: {feats.shape}")
    print(f"Features count: {len(FEATURE_NAMES_22D)}")
    for name, val in zip(FEATURE_NAMES_22D, feats):
        print(f"  {name:35s}: {val:8.3f}")
    assert len(feats) == 22, "피처 차원 수 불일치"
    print("\n[OK] cross_lunge_c3_features_22d 무결성 통과!")
