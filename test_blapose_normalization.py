# -*- coding: utf-8 -*-
"""
vidio-health-trainer
선행 연구 후보 2: BLAPose 기반 전향 기구학(Forward Kinematics) 골격 표준화
(Enhancing 3D Human Pose Estimation with Bone Length Adjustment, ACCV 2024 / CVPR)

[핵심 알고리즘]
1. 3D 키포인트 시퀀스에서 각 관절 간 3차원 단위 방향 벡터(Unit Direction Vector) D 추출
   u_ij(t) = (P_j(t) - P_i(t)) / ||P_j(t) - P_i(t)||
2. AI-Hub 표준 인체 골격 템플릿의 고정 골격 길이 L_ref_ij 매핑
3. 골반 중심(Mid-Hip)을 원점으로 하여 기구학 트리(Kinematic Tree)를 따라 전향 기구학(FK) 재구성
   P'_j(t) = P'_i(t) + L_ref_ij * u_ij(t)
4. 결과: 모든 뼈의 길이가 AI-Hub 표준치와 100% 일치하면서도, 사용자의 관절 회전/동작 궤적은 완전 보존!
"""

import numpy as np

# AI-Hub 바벨 운동 데이터셋 전수 통계 기반 표준 골격 길이 (미터 단위)
AIHUB_BONE_LENGTHS_REF = {
    "torso": 0.4874,         # Mid-Hip -> Mid-Shoulder
    "shoulder_half": 0.1558,  # Mid-Shoulder -> Left/Right Shoulder (어깨 반폭)
    "hip_half": 0.1050,       # Mid-Hip -> Left/Right Hip (골반 반폭)
    "thigh": 0.3696,          # Hip -> Knee (대퇴골)
    "shank": 0.3950,          # Knee -> Ankle (정강이)
    "foot": 0.1450,           # Ankle -> Toe (발)
    "heel": 0.0750,           # Ankle -> Heel (뒤꿈치)
    "upper_arm": 0.2850,      # Shoulder -> Elbow (상완)
    "forearm": 0.2600,        # Elbow -> Wrist (전완)
    "neck": 0.1800,           # Mid-Shoulder -> Nose (목-안면)
}

def apply_blapose_fk_normalization(kps_64f: np.ndarray) -> np.ndarray:
    """
    BLAPose 전향 기구학 골격 표준화 변환
    kps_64f: [64, 33, 3] 랜드마크 시퀀스
    Returns: [64, 33, 3] 표준화된 시퀀스
    """
    T = len(kps_64f)
    out_kps = np.zeros_like(kps_64f)
    
    eps = 1e-6
    
    for t in range(T):
        pts = kps_64f[t]
        
        # 1. 루트 조인트 정의: 골반 중심 (Mid-Hip)
        mid_hip = (pts[23] + pts[24]) / 2.0
        out_kps[t, 23] = mid_hip # 임시
        out_kps[t, 24] = mid_hip
        
        # 2. 골반 좌우 배치 (Mid-Hip -> 23 Left Hip, 24 Right Hip)
        v_lhip = pts[23] - mid_hip
        d_lhip = v_lhip / (np.linalg.norm(v_lhip) + eps)
        out_kps[t, 23] = mid_hip + d_lhip * AIHUB_BONE_LENGTHS_REF["hip_half"]
        
        v_rhip = pts[24] - mid_hip
        d_rhip = v_rhip / (np.linalg.norm(v_rhip) + eps)
        out_kps[t, 24] = mid_hip + d_rhip * AIHUB_BONE_LENGTHS_REF["hip_half"]
        
        # 3. 체간 축 (Mid-Hip -> Mid-Shoulder)
        mid_shoulder_orig = (pts[11] + pts[12]) / 2.0
        v_torso = mid_shoulder_orig - mid_hip
        d_torso = v_torso / (np.linalg.norm(v_torso) + eps)
        mid_shoulder_norm = mid_hip + d_torso * AIHUB_BONE_LENGTHS_REF["torso"]
        
        # 4. 어깨 좌우 배치 (Mid-Shoulder -> 11 Left Shoulder, 12 Right Shoulder)
        v_lsh = pts[11] - mid_shoulder_orig
        d_lsh = v_lsh / (np.linalg.norm(v_lsh) + eps)
        out_kps[t, 11] = mid_shoulder_norm + d_lsh * AIHUB_BONE_LENGTHS_REF["shoulder_half"]
        
        v_rsh = pts[12] - mid_shoulder_orig
        d_rsh = v_rsh / (np.linalg.norm(v_rsh) + eps)
        out_kps[t, 12] = mid_shoulder_norm + d_rsh * AIHUB_BONE_LENGTHS_REF["shoulder_half"]
        
        # 5. 두부/안면 (Mid-Shoulder -> 0 Nose)
        v_nose = pts[0] - mid_shoulder_orig
        d_nose = v_nose / (np.linalg.norm(v_nose) + eps)
        out_kps[t, 0] = mid_shoulder_norm + d_nose * AIHUB_BONE_LENGTHS_REF["neck"]
        # 안면 세부 관절들은 nose 상대 위치 유지
        for f_idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            out_kps[t, f_idx] = out_kps[t, 0] + (pts[f_idx] - pts[0])
            
        # 6. 좌측 팔 기구학 (11 Shoulder -> 13 Elbow -> 15 Wrist)
        v_lelb = pts[13] - pts[11]
        d_lelb = v_lelb / (np.linalg.norm(v_lelb) + eps)
        out_kps[t, 13] = out_kps[t, 11] + d_lelb * AIHUB_BONE_LENGTHS_REF["upper_arm"]
        
        v_lwri = pts[15] - pts[13]
        d_lwri = v_lwri / (np.linalg.norm(v_lwri) + eps)
        out_kps[t, 15] = out_kps[t, 13] + d_lwri * AIHUB_BONE_LENGTHS_REF["forearm"]
        for hand_idx in [17, 19, 21]:
            out_kps[t, hand_idx] = out_kps[t, 15] + (pts[hand_idx] - pts[15])
            
        # 7. 우측 팔 기구학 (12 Shoulder -> 14 Elbow -> 16 Wrist)
        v_relb = pts[14] - pts[12]
        d_relb = v_relb / (np.linalg.norm(v_relb) + eps)
        out_kps[t, 14] = out_kps[t, 12] + d_relb * AIHUB_BONE_LENGTHS_REF["upper_arm"]
        
        v_rwri = pts[16] - pts[14]
        d_rwri = v_rwri / (np.linalg.norm(v_rwri) + eps)
        out_kps[t, 16] = out_kps[t, 14] + d_rwri * AIHUB_BONE_LENGTHS_REF["forearm"]
        for hand_idx in [18, 20, 22]:
            out_kps[t, hand_idx] = out_kps[t, 16] + (pts[hand_idx] - pts[16])
            
        # 8. 좌측 하체 기구학 (23 Hip -> 25 Knee -> 27 Ankle -> 29 Heel / 31 Toe)
        v_lkne = pts[25] - pts[23]
        d_lkne = v_lkne / (np.linalg.norm(v_lkne) + eps)
        out_kps[t, 25] = out_kps[t, 23] + d_lkne * AIHUB_BONE_LENGTHS_REF["thigh"]
        
        v_lank = pts[27] - pts[25]
        d_lank = v_lank / (np.linalg.norm(v_lank) + eps)
        out_kps[t, 27] = out_kps[t, 25] + d_lank * AIHUB_BONE_LENGTHS_REF["shank"]
        
        v_lhel = pts[29] - pts[27]
        d_lhel = v_lhel / (np.linalg.norm(v_lhel) + eps)
        out_kps[t, 29] = out_kps[t, 27] + d_lhel * AIHUB_BONE_LENGTHS_REF["heel"]
        
        v_ltoe = pts[31] - pts[27]
        d_ltoe = v_ltoe / (np.linalg.norm(v_ltoe) + eps)
        out_kps[t, 31] = out_kps[t, 27] + d_ltoe * AIHUB_BONE_LENGTHS_REF["foot"]
        
        # 9. 우측 하체 기구학 (24 Hip -> 26 Knee -> 28 Ankle -> 30 Heel / 32 Toe)
        v_rkne = pts[26] - pts[24]
        d_rkne = v_rkne / (np.linalg.norm(v_rkne) + eps)
        out_kps[t, 26] = out_kps[t, 24] + d_rkne * AIHUB_BONE_LENGTHS_REF["thigh"]
        
        v_rank = pts[28] - pts[26]
        d_rank = v_rank / (np.linalg.norm(v_rank) + eps)
        out_kps[t, 28] = out_kps[t, 26] + d_rank * AIHUB_BONE_LENGTHS_REF["shank"]
        
        v_rhel = pts[30] - pts[28]
        d_rhel = v_rhel / (np.linalg.norm(v_rhel) + eps)
        out_kps[t, 30] = out_kps[t, 28] + d_rhel * AIHUB_BONE_LENGTHS_REF["heel"]
        
        v_rtoe = pts[32] - pts[28]
        d_rtoe = v_rtoe / (np.linalg.norm(v_rtoe) + eps)
        out_kps[t, 32] = out_kps[t, 28] + d_rtoe * AIHUB_BONE_LENGTHS_REF["foot"]
        
    return out_kps
