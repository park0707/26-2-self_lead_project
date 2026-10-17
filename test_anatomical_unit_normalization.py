# -*- coding: utf-8 -*-
"""
vidio-health-trainer
사용자 제안: 인체 부위별 분할 단위 좌표계 정규화 (Anatomical Unit Normalization)

[수학적 원리]
1. 가로 좌표계 단위 u_x: 사용자 어깨 너비 (Shoulder Width)의 1/N
2. 세로 좌표계 단위 u_y: 사용자 대퇴골 길이 (Thigh Length) 또는 체간 길이 (Torso Length)의 1/M
3. 기립 구간(Stand Phase)에서 사용자 신체 척도(u_x, u_y)를 계측하여 동적 굴곡에 의한 왜곡 방지
4. AI-Hub 표준 인체 척도(u_x_ref, u_y_ref)와의 비율을 통해 종횡비 왜곡(카메라 해상도/각도 왜곡)을 상쇄:
   s_x = W_shoulder_ref / W_shoulder_user
   s_y = L_thigh_ref / L_thigh_user
   s_z = sqrt(s_x * s_y)  (또는 s_y)
5. 골반 중심(Mid-Hip)을 원점으로 하여 신체 중심 기준으로 스케일링 수행:
   P'_i(t) = P_pelvis(t) + S * (P_i(t) - P_pelvis(t))
"""

import numpy as np

# AI-Hub 바벨 운동 데이터셋 전수 통계 기반 표준 인체 척도 (단위: 미터)
AIHUB_REF_METRICS = {
    "shoulder_width": 0.3117,  # 어깨 너비 (좌우 견봉 간 거리)
    "thigh_length": 0.3696,    # 대퇴골 길이 (고관절-슬관절 간 거리)
    "torso_length": 0.4874,    # 체간 길이 (어깨 중심-골반 중심 간 거리)
}

def extract_stand_phase_metrics(kps_64f, stand_frames=8):
    """
    동작 초기 기립 상태(Stand Phase)에서 노이즈를 평균화하여
    사용자의 정적 신체 척도(어깨 너비, 대퇴골 길이, 체간 길이)를 측정합니다.
    """
    n_frames = min(stand_frames, len(kps_64f))
    stand_kps = np.mean(kps_64f[:n_frames], axis=0) # [33, 3]
    
    # 1. 어깨 너비 (좌 11, 우 12)
    sh_w = float(np.linalg.norm(stand_kps[11] - stand_kps[12]))
    if sh_w < 1e-4:
        sh_w = AIHUB_REF_METRICS["shoulder_width"]
        
    # 2. 대퇴골 길이 (좌 23-25, 우 24-26 평균)
    thigh_l = np.linalg.norm(stand_kps[23] - stand_kps[25])
    thigh_r = np.linalg.norm(stand_kps[24] - stand_kps[26])
    thigh = float((thigh_l + thigh_r) / 2.0)
    if thigh < 1e-4:
        thigh = AIHUB_REF_METRICS["thigh_length"]
        
    # 3. 체간 길이 (어깨 중심 - 골반 중심)
    sh_mid = (stand_kps[11] + stand_kps[12]) / 2.0
    hip_mid = (stand_kps[23] + stand_kps[24]) / 2.0
    torso = float(np.linalg.norm(sh_mid - hip_mid))
    if torso < 1e-4:
        torso = AIHUB_REF_METRICS["torso_length"]
        
    return {
        "shoulder_width": sh_w,
        "thigh_length": thigh,
        "torso_length": torso
    }

def apply_anatomical_unit_normalization(kps_64f, vertical_anchor="thigh", stand_frames=8):
    """
    사용자가 제안한 인체 분할 단위 좌표계 변환을 적용합니다.
    
    Args:
        kps_64f: [64, 33, 3] 64프레임 3D 랜드마크 시퀀스
        vertical_anchor: 'thigh' (사용자 원안: 대퇴골 M등분) 또는 'torso' (체간 변형)
        stand_frames: 기립 기준 프레임 수
        
    Returns:
        norm_kps: [64, 33, 3] 인체 단위 좌표계로 정규화된 시퀀스
        scale_info: 계산된 스케일 팩터 딕셔너리
    """
    user_metrics = extract_stand_phase_metrics(kps_64f, stand_frames)
    
    # 1. 가로 스케일 팩터 (어깨 너비 기준)
    s_x = AIHUB_REF_METRICS["shoulder_width"] / user_metrics["shoulder_width"]
    
    # 2. 세로 스케일 팩터 (대퇴골 또는 체간 기준)
    if vertical_anchor == "torso":
        s_y = AIHUB_REF_METRICS["torso_length"] / user_metrics["torso_length"]
    else: # 'thigh'
        s_y = AIHUB_REF_METRICS["thigh_length"] / user_metrics["thigh_length"]
        
    # 3. 깊이(Z) 스케일 팩터 (등방성 기하 평균)
    s_z = np.sqrt(s_x * s_y)
    
    # 4. 프레임별 골반 중심 기준 신체 스케일링
    norm_kps = np.zeros_like(kps_64f)
    
    for t in range(len(kps_64f)):
        # 골반 중심 (Mid-Hip)
        pelvis = (kps_64f[t, 23] + kps_64f[t, 24]) / 2.0
        
        # 골반 중심 상대 좌표계로 전환
        rel = kps_64f[t] - pelvis
        
        # 축별 독립 인체 단위 스케일링 적용
        rel[:, 0] *= s_x
        rel[:, 1] *= s_y
        rel[:, 2] *= s_z
        
        # 골반 중심 복원
        norm_kps[t] = rel + pelvis
        
    scale_info = {
        "user_shoulder": user_metrics["shoulder_width"],
        "user_thigh": user_metrics["thigh_length"],
        "user_torso": user_metrics["torso_length"],
        "s_x": float(s_x),
        "s_y": float(s_y),
        "s_z": float(s_z),
        "aspect_correction_ratio": float(s_x / s_y)
    }
    return norm_kps, scale_info
