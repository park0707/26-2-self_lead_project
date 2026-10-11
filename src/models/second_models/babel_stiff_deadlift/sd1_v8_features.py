"""
sd1_v8_features.py
바벨 스티프 데드리프트 1번 결함 (sd1: 무릎 구부린 채 고정 위반)
v8.0 생체역학 도메인 피처 엔지니어링 모듈 (MediaPipe 33관절 64프레임 표준)
규정 준수: vidio-health-trainer, 순수 비전 3D 좌표, 원근 왜곡 극복 신체 중심 시상면 투영
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple

# MediaPipe 33 Joint Indices
MP_L_SHOULDER = 11
MP_R_SHOULDER = 12
MP_L_HIP = 23
MP_R_HIP = 24
MP_L_KNEE = 25
MP_R_KNEE = 26
MP_L_ANKLE = 27
MP_R_ANKLE = 28
MP_L_HEEL = 29
MP_R_HEEL = 30
MP_L_FOOT = 31
MP_R_FOOT = 32

# 하체 핵심 12관절 인덱스 (TCN 입력용: 12관절 x 3D = 36채널)
KEY_JOINTS_12 = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

VIEW_INDEX_MAP = {
    'view1': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view5': 4,
    '시점 E': 0, '시점 B': 1, '시점 C': 2, '시점 D': 3, '시점 A': 4
}

FEATURE_NAMES_30D_SD1 = [
    'knee_angle_bottom_deg',       # [0] 최저점 무릎 3D 내각
    'knee_rom_deg',                # [1] 무릎 3D 각도 변화폭 (max - min)
    'knee_delta_address_bottom',   # [2] 어드레스 대비 최저점 무릎 각도 변화량
    'knee_hip_rom_ratio',          # [3] 고관절 대비 무릎 ROM 비율 (ROM_knee / ROM_hip)
    'hip_knee_velocity_corr',      # [4] 고관절-무릎 각속도 피어슨 상관계수 (동위상 결합도)
    'mcrp',                        # [5] Mean Continuous Relative Phase (평균 연속 상대 위상)
    'vcrp',                        # [6] CRP 가변성 (Continuous Relative Phase Variability)
    'knee_angular_accel_max',      # [7] 무릎 최대 각가속도 (주저앉는 스파이크)
    'knee_accel_sign_changes',     # [8] 무릎 각가속도 부호 반전 횟수 (미세 떨림 지터)
    'excess_flexion_integral',     # [9] 어드레스 대비 5도 이상 단방향 추가 굴곡 적분량
    'excess_flexion_max',          # [10] 어드레스 대비 최대 추가 굴곡량
    'kadr_max',                    # [11] 신체 정규 시상축(e_z) 발목 대비 무릎 전방 돌출비 최대값 (KT-DC)
    'kadr_mean',                   # [12] 발목 대비 무릎 전방 돌출비 평균값
    'kadr_rom',                    # [13] 발목 대비 무릎 전방 돌출비 가동범위
    'shank_svii',                  # [14] 정강이 수직 불변성 지수 (지면 수직 대비 정강이 경사각 ROM)
    'femur_rom',                   # [15] 대퇴골 지면 경사각 ROM
    'norm_knee_disp_y',            # [16] 정강이 길이 대비 무릎 수직(Y축) 변위 ROM
    'norm_knee_disp_z',            # [17] 대퇴골 길이 대비 신체 시상축(Z) 무릎 변위 ROM
    'pli',                         # [18] 고관절-무릎 각속도 위상 분리 지수 (Phase Lag Index)
    'hyperextension_ratio',        # [19] 무릎 각도가 180도를 초과한 프레임 비율 (과신전)
    'coronal_knee_wobble',         # [20] 관상면(좌우 X축) 무릎 흔들림 표준편차
    'sagittal_proj_knee_rom',      # [21] 신체 중심 동적 시상면 투영 무릎 각도 ROM
    'sagittal_proj_knee_bottom',   # [22] 동적 시상면 투영 최저점 무릎 각도
    'bilateral_knee_asym',         # [23] 좌우 무릎 각도 비대칭도 평균
    'view_code',                   # [24] 시점 코드 (0~4)
    'v_oh_1',                      # [25] 시점 원-핫 1 (view1 / 시점 E)
    'v_oh_2',                      # [26] 시점 원-핫 2 (view2 / 시점 B)
    'v_oh_3',                      # [27] 시점 원-핫 3 (view3 / 시점 C)
    'v_oh_4',                      # [28] 시점 원-핫 4 (view4 / 시점 D)
    'v_oh_5'                       # [29] 시점 원-핫 5 (view5 / 시점 A)
]

def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """3D 벡터 ba와 bc 사이의 사잇각(도) 산출"""
    ba = a - b
    bc = c - b
    dot = np.sum(ba * bc, axis=-1)
    norm_ba = np.linalg.norm(ba, axis=-1) + 1e-7
    norm_bc = np.linalg.norm(bc, axis=-1) + 1e-7
    cos_angle = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def compute_continuous_relative_phase(ang_hip: np.ndarray, ang_knee: np.ndarray) -> Tuple[float, float, float]:
    """
    고관절-무릎 연속 상대 위상 (CRP) 및 위상각 속도 상관계수 계산
    ang_hip, ang_knee: (64,)
    반환: (MCRP, VCRP, vel_corr)
    """
    # 중심차분 각속도 계산 (스무딩 적용)
    h_smooth = gaussian_filter1d(ang_hip, sigma=1.5)
    k_smooth = gaussian_filter1d(ang_knee, sigma=1.5)
    
    omega_h = np.gradient(h_smooth)
    omega_k = np.gradient(k_smooth)
    
    # 1. 속도 피어슨 상관계수 (결함 시 > 0.7 동위상 결합)
    std_h = np.std(omega_h) + 1e-7
    std_k = np.std(omega_k) + 1e-7
    vel_corr = float(np.corrcoef(omega_h, omega_k)[0, 1])
    if np.isnan(vel_corr):
        vel_corr = 0.0

    # 2. 위상각 정규화 [-1, 1]
    rom_h = (np.max(h_smooth) - np.min(h_smooth)) + 1e-7
    rom_k = (np.max(k_smooth) - np.min(k_smooth)) + 1e-7
    
    norm_h = 2.0 * (h_smooth - np.min(h_smooth)) / rom_h - 1.0
    norm_k = 2.0 * (k_smooth - np.min(k_smooth)) / rom_k - 1.0
    
    norm_omega_h = omega_h / (np.max(np.abs(omega_h)) + 1e-7)
    norm_omega_k = omega_k / (np.max(np.abs(omega_k)) + 1e-7)
    
    phi_h = np.arctan2(norm_omega_h, norm_h) # [-pi, pi]
    phi_k = np.arctan2(norm_omega_k, norm_k)
    
    # 3. CRP 시계열 (|phi_h - phi_k|)
    crp_deg = np.degrees(np.abs(phi_h - phi_k))
    crp_deg = np.where(crp_deg > 180.0, 360.0 - crp_deg, crp_deg)
    
    mcrp = float(np.mean(crp_deg))
    vcrp = float(np.std(crp_deg))
    
    return mcrp, vcrp, vel_corr

def compute_canonical_pelvis_sagittal(p_hip_l, p_hip_r, p_sh_l, p_sh_r, p_knee_l, p_knee_r, p_ank_l, p_ank_r):
    """
    Pelvis-Gravity Canonical Transformation & Dynamic Sagittal Plane
    - e_x: 골반 좌우 관상축 (오른쪽 -> 왼쪽)
    - e_y: 중력 수직 상향축
    - e_z: 신체 정규 시상 전후축 (정면 방향)
    """
    p_root = 0.5 * (p_hip_l + p_hip_r)
    p_mid_ank = 0.5 * (p_ank_l + p_ank_r)
    p_knee_mid = 0.5 * (p_knee_l + p_knee_r)

    # 1. Lateral 축 (e_x)
    v_lateral = p_hip_l - p_hip_r
    e_x = v_lateral / (np.linalg.norm(v_lateral, axis=-1, keepdims=True) + 1e-7)

    # 2. Vertical 축 (e_y): 발목 중심 -> 골반 상향
    v_up = p_root - p_mid_ank
    dot_xy = np.sum(v_up * e_x, axis=-1, keepdims=True)
    v_y_ortho = v_up - dot_xy * e_x
    e_y = v_y_ortho / (np.linalg.norm(v_y_ortho, axis=-1, keepdims=True) + 1e-7)

    # 3. Sagittal 전방 축 (e_z): e_x x e_y
    e_z = np.cross(e_x, e_y)

    # 4. 발목 기준 무릎 상대 벡터 투영
    rel_knee = p_knee_mid - p_mid_ank
    z_knee_rel = np.sum(rel_knee * e_z, axis=-1) # (T,) 전방 이동량
    y_knee_rel = np.sum(rel_knee * e_y, axis=-1) # (T,)

    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    len_shank = 0.5 * (np.linalg.norm(v_shank_l, axis=-1).mean() + np.linalg.norm(v_shank_r, axis=-1).mean()) + 1e-5

    kadr = z_knee_rel / len_shank # Knee-Ankle Displacement Ratio (무차원)

    # 5. 동적 시상면 투영 무릎 각도
    # 법선 벡터: e_x (관상면 축)
    # 대퇴골 및 경골 벡터를 시상면(e_y, e_z 평면)에 투영
    v_thigh_mid = p_knee_mid - p_root
    v_shank_mid = p_mid_ank - p_knee_mid

    v_thigh_proj = v_thigh_mid - np.sum(v_thigh_mid * e_x, axis=-1, keepdims=True) * e_x
    v_shank_proj = v_shank_mid - np.sum(v_shank_mid * e_x, axis=-1, keepdims=True) * e_x

    dot_sag = np.sum(v_thigh_proj * v_shank_proj, axis=-1)
    norm_t = np.linalg.norm(v_thigh_proj, axis=-1) + 1e-7
    norm_s = np.linalg.norm(v_shank_proj, axis=-1) + 1e-7
    ang_sag_knee = np.degrees(np.arccos(np.clip(dot_sag / (norm_t * norm_s), -1.0, 1.0)))

    return kadr, z_knee_rel, ang_sag_knee, len_shank

def extract_sd1_v8_features(kps: np.ndarray, view_name: str = "view4") -> np.ndarray:
    """
    (64, 33, 3) MediaPipe 3D 관절 좌표에서 30차원 v8 생체역학 도메인 피처 추출
    """
    arr = np.array(kps, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    p_sh_l = arr[:, MP_L_SHOULDER]
    p_sh_r = arr[:, MP_R_SHOULDER]
    p_hip_l = arr[:, MP_L_HIP]
    p_hip_r = arr[:, MP_R_HIP]
    p_knee_l = arr[:, MP_L_KNEE]
    p_knee_r = arr[:, MP_R_KNEE]
    p_ank_l = arr[:, MP_L_ANKLE]
    p_ank_r = arr[:, MP_R_ANKLE]

    # 1. 3D 공간 각도 시계열 (Hip-Knee-Ankle)
    ang_knee_l = calculate_angle_3d(p_hip_l, p_knee_l, p_ank_l)
    ang_knee_r = calculate_angle_3d(p_hip_r, p_knee_r, p_ank_r)
    ang_knee = 0.5 * (ang_knee_l + ang_knee_r) # (64,)

    # 2. 고관절 각도 시계열 (Shoulder-Hip-Knee)
    ang_hip_l = calculate_angle_3d(p_sh_l, p_hip_l, p_knee_l)
    ang_hip_r = calculate_angle_3d(p_sh_r, p_hip_r, p_knee_r)
    ang_hip = 0.5 * (ang_hip_l + ang_hip_r) # (64,)

    # 3. 최저점(고관절 최대 굴곡 = 최소 각도 프레임)
    idx_bot = int(np.argmin(ang_hip))

    # [0] 최저점 무릎 3D 내각
    f0_bottom_angle = float(ang_knee[idx_bot])

    # [1] 무릎 3D 각도 변화폭 (ROM)
    f1_knee_rom = float(np.max(ang_knee) - np.min(ang_knee))

    # [2] 어드레스 대비 최저점 무릎 각도 변화량
    init_knee = float(np.mean(ang_knee[:6]))
    f2_delta_address_bot = float(abs(ang_knee[idx_bot] - init_knee))

    # [3] 고관절 대비 무릎 ROM 비율
    hip_rom = float(np.max(ang_hip) - np.min(ang_hip))
    f3_rom_ratio = float(np.clip(f1_knee_rom / (hip_rom + 1e-5), 0.0, 5.0))

    # [4~6] 고관절-무릎 CRP 및 각속도 상관계수
    mcrp, vcrp, vel_corr = compute_continuous_relative_phase(ang_hip, ang_knee)
    f4_vel_corr = float(vel_corr)
    f5_mcrp = float(mcrp)
    f6_vcrp = float(vcrp)

    # [7~8] 무릎 각가속도 및 부호 반전 지터
    k_smooth = gaussian_filter1d(ang_knee, sigma=1.5)
    omega_k = np.gradient(k_smooth)
    alpha_k = np.gradient(omega_k)
    f7_accel_max = float(np.max(np.abs(alpha_k)))
    sign_changes = np.sum(np.diff(np.sign(alpha_k)) != 0)
    f8_sign_changes = float(sign_changes)

    # [9~10] 단방향 초과 굴곡 적분량 & 최대치 (어드레스 대비 5도 이상 굴곡)
    excess_flex = np.maximum(0.0, init_knee - ang_knee - 5.0)
    f9_excess_integral = float(np.mean(excess_flex))
    f10_excess_max = float(np.max(excess_flex))

    # [11~14, 21~22] Canonical Transformation & KADR & 동적 시상면 투영
    kadr_seq, z_knee_seq, ang_sag_seq, len_shank = compute_canonical_pelvis_sagittal(
        p_hip_l, p_hip_r, p_sh_l, p_sh_r, p_knee_l, p_knee_r, p_ank_l, p_ank_r
    )
    f11_kadr_max = float(np.max(kadr_seq))
    f12_kadr_mean = float(np.mean(kadr_seq))
    f13_kadr_rom = float(np.max(kadr_seq) - np.min(kadr_seq))

    # [14] 정강이 수직 불변성 지수 (SVII: 정강이 지면 수직선 대비 각도 ROM)
    v_shank_l = p_ank_l - p_knee_l
    v_shank_r = p_ank_r - p_knee_r
    pitch_shank_l = np.degrees(np.arctan2(np.abs(v_shank_l[:, 1]), np.linalg.norm(v_shank_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_shank_r = np.degrees(np.arctan2(np.abs(v_shank_r[:, 1]), np.linalg.norm(v_shank_r[:, [0, 2]], axis=-1) + 1e-7))
    f14_shank_svii = float(0.5 * ((np.max(pitch_shank_l) - np.min(pitch_shank_l)) + (np.max(pitch_shank_r) - np.min(pitch_shank_r))))

    # [15] 대퇴골 지면 경사각 ROM
    v_femur_l = p_knee_l - p_hip_l
    v_femur_r = p_knee_r - p_hip_r
    pitch_femur_l = np.degrees(np.arctan2(np.abs(v_femur_l[:, 1]), np.linalg.norm(v_femur_l[:, [0, 2]], axis=-1) + 1e-7))
    pitch_femur_r = np.degrees(np.arctan2(np.abs(v_femur_r[:, 1]), np.linalg.norm(v_femur_r[:, [0, 2]], axis=-1) + 1e-7))
    f15_femur_rom = float(0.5 * ((np.max(pitch_femur_l) - np.min(pitch_femur_l)) + (np.max(pitch_femur_r) - np.min(pitch_femur_r))))

    # [16] 정강이 길이 대비 무릎 수직(Y) 변위 ROM
    disp_y = 0.5 * (p_knee_l[:, 1] + p_knee_r[:, 1])
    f16_norm_disp_y = float((np.max(disp_y) - np.min(disp_y)) / (len_shank + 1e-5))

    # [17] 대퇴골 길이 대비 신체 시상축(Z) 무릎 변위 ROM
    len_femur = 0.5 * (np.mean(np.linalg.norm(v_femur_l, axis=-1)) + np.mean(np.linalg.norm(v_femur_r, axis=-1))) + 1e-5
    f17_norm_disp_z = float((np.max(z_knee_seq) - np.min(z_knee_seq)) / (len_femur + 1e-5))

    # [18] PLI (Phase Lag Index / 위상 분리 지수)
    sum_omega_k = np.sum(np.abs(np.diff(ang_knee)))
    sum_omega_h = np.sum(np.abs(np.diff(ang_hip)))
    f18_pli = float(np.clip(1.0 - (sum_omega_k / (sum_omega_h + 1e-5)), -1.0, 1.0))

    # [19] 과신전 프레임 비율 (무릎 내각 > 180도)
    f19_hyperext_ratio = float(np.mean(ang_knee > 180.0))

    # [20] 관상면(X축) 무릎 흔들림 표준편차
    knee_x = 0.5 * (p_knee_l[:, 0] + p_knee_r[:, 0])
    f20_coronal_wobble = float(np.std(knee_x))

    # [21~22] 동적 시상면 투영 무릎 각도
    f21_sag_rom = float(np.max(ang_sag_seq) - np.min(ang_sag_seq))
    f22_sag_bottom = float(ang_sag_seq[idx_bot])

    # [23] 좌우 무릎 각도 비대칭도 평균
    f23_bilateral_asym = float(np.mean(np.abs(ang_knee_l - ang_knee_r)))

    # [24~29] 시점 정보 (시점 코드 및 원-핫)
    v_idx = VIEW_INDEX_MAP.get(view_name, 3) # default: view4 (idx 3)
    f24_view_code = float(v_idx)
    v_oh = np.zeros(5, dtype=np.float32)
    v_oh[v_idx] = 1.0

    feat_30d = np.array([
        f0_bottom_angle, f1_knee_rom, f2_delta_address_bot, f3_rom_ratio,
        f4_vel_corr, f5_mcrp, f6_vcrp, f7_accel_max, f8_sign_changes,
        f9_excess_integral, f10_excess_max, f11_kadr_max, f12_kadr_mean, f13_kadr_rom,
        f14_shank_svii, f15_femur_rom, f16_norm_disp_y, f17_norm_disp_z,
        f18_pli, f19_hyperext_ratio, f20_coronal_wobble, f21_sag_rom,
        f22_sag_bottom, f23_bilateral_asym, f24_view_code,
        v_oh[0], v_oh[1], v_oh[2], v_oh[3], v_oh[4]
    ], dtype=np.float32)

    return np.nan_to_num(feat_30d, nan=0.0, posinf=0.0, neginf=0.0)

def extract_sd1_temporal_12j(kps: np.ndarray) -> np.ndarray:
    """
    Dilated-TCN 시계열 딥러닝 스트림용 12관절 3D 좌표 텐서 추출 (64, 36)
    12개 관절 x (X, Y, Z) = 36채널
    """
    arr = np.array(kps, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    seq_12j = arr[:, KEY_JOINTS_12, :3].reshape(64, 36)
    return seq_12j
