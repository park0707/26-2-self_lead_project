"""
stiff_spine_features_v2.py
- 스티프 데드리프트 척추 중립(Spine Neutrality) 피처 최적화 v2 모듈
- 30D 원본 피처 중:
  * 분산 0 (상수 피처)
  * 다중공선성 (|r| >= 0.85) 중복 피처
  * LOFO 성능 저해(Negative Impact) 피처
  * 순수 노이즈(Cohen's d ~ 0, AUC ~ 0.50) 피처를 전면 정리한 정예 피처셋 도출
"""

import numpy as np

# v1 30D 피처 전체 이름 목록 (레거시 인덱스 보존용)
V1_FEATURE_NAMES = [
    "stiff_lumbar_poly_a",                # 0
    "stiff_lumbar_stiffness_sd",          # 1
    "stiff_spine_lateral_dev_max",        # 2
    "stiff_trunk_flex_speed_max",         # 3
    "stiff_cervical_min",                 # 4
    "stiff_signed_thoracolumbar_delta",   # 5
    "stiff_active_rom",                   # 6
    "stiff_cobb_mean",                    # 7
    "stiff_shoulder_hip_twist_max",       # 8
    "stiff_lateral_asym_integral",        # 9
    "stiff_flex_accel_peak",              # 10
    "stiff_pitch_standing",               # 11
    "stiff_view_normal_z",                # 12
    "stiff_shear_stress_proxy",           # 13
    "stiff_lockout_hyperextension",       # 14
    "stiff_pitch_rom",                    # 15
    "stiff_view_occlusion",               # 16
    "stiff_view_comp_curv",               # 17
    "stiff_res_lumbar_curv",              # 18
    "stiff_ratio_arm_to_torso",           # 19
    "stiff_ratio_femur_to_tibia",         # 20
    "stiff_ratio_leg_to_torso",           # 21
    "stiff_shoulder_roll_max",            # 22
    "stiff_hip_roll_max",                 # 23
    "stiff_thoracic_kyphosis_peak",       # 24
    "stiff_lumbopelvic_rhythm_ratio",     # 25
    "stiff_knee_fixation_sd",             # 26
    "stiff_torso_pelvis_lateral_offset",  # 27
    "stiff_lateral_drift_direction",      # 28
    "stiff_sagittal_spine_stiffness"      # 29
]

# 제거 대상 피처 10개 (이유 명시)
PRUNED_FEATURES_INFO = {
    "stiff_res_lumbar_curv": {"index": 18, "reason": "분산 0.0000 (상수 피처, 정보량 전무)"},
    "stiff_pitch_standing": {"index": 11, "reason": "제거 시 F1 +0.11%p 상승(성능 저해) 및 stiff_pitch_rom과 r=-0.986 초고상관"},
    "stiff_flex_accel_peak": {"index": 10, "reason": "stiff_trunk_flex_speed_max와 r=+0.976 초고상관 중복 피처"},
    "stiff_signed_thoracolumbar_delta": {"index": 5, "reason": "stiff_lumbar_stiffness_sd와 r=+0.888 고상관 중복"},
    "stiff_view_normal_z": {"index": 12, "reason": "stiff_thoracic_kyphosis_peak와 r=-0.895 고상관 중복 및 단독 AUC 51.18% 미약"},
    "stiff_shoulder_roll_max": {"index": 22, "reason": "stiff_hip_roll_max와 r=+0.895 고상관 중복 (핵심 기여 피처인 hip_roll_max 보존)"},
    "stiff_lockout_hyperextension": {"index": 14, "reason": "제거 시 F1 +0.05%p 상승(성능 저해/노이즈)"},
    "stiff_lumbopelvic_rhythm_ratio": {"index": 25, "reason": "제거 시 F1 +0.11%p 상승(성능 저해/노이즈)"},
    "stiff_lateral_asym_integral": {"index": 9, "reason": "Cohen's d=-0.006, AUC 50.12% 순수 무효 노이즈"},
    "stiff_lateral_drift_direction": {"index": 28, "reason": "Cohen's d=-0.002, AUC 50.33% 순수 무효 노이즈"}
}

# v2 엄선 피처 인덱스 (20D 정예 피처셋)
V2_SELECTED_INDICES = [
    0,   # stiff_lumbar_poly_a (곡률 곡선 2차항)
    1,   # stiff_lumbar_stiffness_sd (요추 안정성/강성 변동)
    2,   # stiff_spine_lateral_dev_max (측만 최대 편차)
    3,   # stiff_trunk_flex_speed_max (체간 굴곡 최대 속도)
    4,   # stiff_cervical_min (경추 신전 최소각)
    6,   # stiff_active_rom (척추 굴곡 가동범위)
    7,   # stiff_cobb_mean (평균 콥스각)
    8,   # stiff_shoulder_hip_twist_max (어깨-골반 비틀림)
    13,  # stiff_shear_stress_proxy (전단 응력 대리 지표, 핵심 기여)
    15,  # stiff_pitch_rom (체간 피치 가동범위)
    16,  # stiff_view_occlusion (시점 차폐율)
    17,  # stiff_view_comp_curv (시점 보정 곡률)
    19,  # stiff_ratio_arm_to_torso (팔-체간 비율, 개인화 체형 보정)
    20,  # stiff_ratio_femur_to_tibia (대퇴-하퇴 비율)
    21,  # stiff_ratio_leg_to_torso (하지-체간 비율)
    23,  # stiff_hip_roll_max (골반 롤 최대각, 핵심 기여)
    24,  # stiff_thoracic_kyphosis_peak (흉추 후만 피크각)
    26,  # stiff_knee_fixation_sd (무릎 고정 편차)
    27,  # stiff_torso_pelvis_lateral_offset (체간-골반 횡편차)
    29   # stiff_sagittal_spine_stiffness (시상면 척추 강성)
]

V2_FEATURE_NAMES = [V1_FEATURE_NAMES[idx] for idx in V2_SELECTED_INDICES]

def select_v2_features(x_30d: np.ndarray) -> np.ndarray:
    """
    30D 피처 배열을 20D 정예 v2 피처 배열로 축소 필터링.
    x_30d: (N, 30) 또는 (30,)
    """
    if x_30d.ndim == 1:
        return x_30d[V2_SELECTED_INDICES]
    elif x_30d.ndim == 2:
        return x_30d[:, V2_SELECTED_INDICES]
    else:
        raise ValueError(f"지원하지 않는 텐서 차원: {x_30d.shape}")
