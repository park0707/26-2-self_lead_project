# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Module: stiff_spine_augmentation.py
Purpose: 사용자 엄격 제약 준수 골격 데이터 증강 모듈
  - [규칙 1]: 인위적 노이즈(Gaussian Jitter 등) 절대 사용 금지!
  - [규칙 2]: 뼈대 길이 가변(Bone Length Scaling)만 적용
  - [규칙 3]: DTW(Dynamic Time Warping) 기반 비선형 시간 축 변경(Temporal Warping)만 적용
"""

import numpy as np
from scipy.interpolate import CubicSpline

# MediaPipe 관절 인덱스
MP = dict(
    NOSE=0, LEAR=7, REAR=8,
    LSH=11, RSH=12, LELB=13, RELB=14, LWR=15, RWR=16,
    LHIP=23, RHIP=24, LKNE=25, RKNE=26, LANK=27, RANK=28
)

# 계층적 골격 키네마틱 트리 정의 (부모 -> 자식 관절)
KINEMATIC_CHAINS = [
    # 하지 (골반 -> 무릎 -> 발목)
    (MP['LHIP'], MP['LKNE']),
    (MP['LKNE'], MP['LANK']),
    (MP['RHIP'], MP['RKNE']),
    (MP['RKNE'], MP['RANK']),
    # 상지 (어깨 -> 팔꿈치 -> 손목)
    (MP['LSH'], MP['LELB']),
    (MP['LELB'], MP['LWR']),
    (MP['RSH'], MP['RELB']),
    (MP['RELB'], MP['RWR'])
]


def augment_bone_length_scaling(kps_64f: np.ndarray, scale_range: Tuple[float, float] = (0.93, 1.07)) -> np.ndarray:
    """
    [증강 1] 뼈대 길이 가변 (Bone Length Scaling)
    - 체형 다양성(대퇴 대 하퇴 비, 팔 길이 비)을 물리적으로 정밀 모사
    - 인위적 노이즈 없음. 골격의 기하학적 길이만 안전 범위(±7%) 내에서 신축
    kps_64f: (64, 33, 3)
    """
    aug_kps = kps_64f.copy()
    T = kps_64f.shape[0]

    # 체간/사지별 스케일 계수 무작위 추출
    s_femur = float(np.random.uniform(scale_range[0], scale_range[1]))
    s_tibia = float(np.random.uniform(scale_range[0], scale_range[1]))
    s_arm = float(np.random.uniform(scale_range[0], scale_range[1]))

    for t in range(T):
        # 1. 좌측 하지
        v_l_femur = aug_kps[t, MP['LKNE']] - aug_kps[t, MP['LHIP']]
        aug_kps[t, MP['LKNE']] = aug_kps[t, MP['LHIP']] + v_l_femur * s_femur
        v_l_tibia = kps_64f[t, MP['LANK']] - kps_64f[t, MP['LKNE']]
        aug_kps[t, MP['LANK']] = aug_kps[t, MP['LKNE']] + v_l_tibia * s_tibia

        # 2. 우측 하지
        v_r_femur = aug_kps[t, MP['RKNE']] - aug_kps[t, MP['RHIP']]
        aug_kps[t, MP['RKNE']] = aug_kps[t, MP['RHIP']] + v_r_femur * s_femur
        v_r_tibia = kps_64f[t, MP['RANK']] - kps_64f[t, MP['RKNE']]
        aug_kps[t, MP['RANK']] = aug_kps[t, MP['RKNE']] + v_r_tibia * s_tibia

        # 3. 좌측 상지
        v_l_arm = aug_kps[t, MP['LELB']] - aug_kps[t, MP['LSH']]
        aug_kps[t, MP['LELB']] = aug_kps[t, MP['LSH']] + v_l_arm * s_arm
        v_l_forearm = kps_64f[t, MP['LWR']] - kps_64f[t, MP['LELB']]
        aug_kps[t, MP['LWR']] = aug_kps[t, MP['LELB']] + v_l_forearm * s_arm

        # 4. 우측 상지
        v_r_arm = aug_kps[t, MP['RELB']] - aug_kps[t, MP['RSH']]
        aug_kps[t, MP['RELB']] = aug_kps[t, MP['RSH']] + v_r_arm * s_arm
        v_r_forearm = kps_64f[t, MP['RWR']] - kps_64f[t, MP['RELB']]
        aug_kps[t, MP['RWR']] = aug_kps[t, MP['RELB']] + v_r_forearm * s_arm

    return aug_kps


def augment_dtw_temporal_warping(kps_64f: np.ndarray, warp_range: Tuple[float, float] = (0.75, 1.30)) -> np.ndarray:
    """
    [증강 2] DTW 기반 비선형 시간 축 변경 (Dynamic Time Warping)
    - 운동의 하강(Eccentric) 및 상승(Concentric) 구간 템포 편차를 비선형 시간 맵핑으로 모사
    - 인위적 노이즈 없음. 3차 스플라인(Cubic Spline)으로 부드럽게 64프레임 보간
    kps_64f: (64, 33, 3)
    """
    T = kps_64f.shape[0]  # 64
    t_orig = np.linspace(0, 1, T)

    # 손목 수직 궤적 기준 최저 변곡점(Inflexion Point) 탐색
    wrist_y = (kps_64f[:, MP['LWR'], 1] + kps_64f[:, MP['RWR'], 1]) / 2.0
    inflexion_idx = int(np.argmax(wrist_y))
    inflexion_t = float(inflexion_idx / max(1, T - 1))

    # 하강 구간과 상승 구간의 비대칭 속도 스케일링 계수
    s_eccentric = float(np.random.uniform(warp_range[0], warp_range[1]))
    # 단조 증가를 보장하는 시간 앵커 포인트 5개 생성
    anchor_t_orig = np.array([
        0.0,
        inflexion_t * 0.5,
        inflexion_t,
        inflexion_t + (1.0 - inflexion_t) * 0.5,
        1.0
    ])

    delta_mid = (s_eccentric - 1.0) * 0.1
    anchor_t_warped = np.array([
        0.0,
        max(0.05, min(0.45, anchor_t_orig[1] + delta_mid)),
        max(0.20, min(0.80, anchor_t_orig[2] + delta_mid * 1.5)),
        max(0.55, min(0.95, anchor_t_orig[3] + delta_mid * 0.5)),
        1.0
    ])
    # 단조 증가 강제 정렬
    anchor_t_warped = np.sort(anchor_t_warped)

    # DTW 맵핑 함수 (Warped Time -> Original Time)
    cs_time = CubicSpline(anchor_t_warped, anchor_t_orig, bc_type='clamped')
    t_sampled = np.clip(cs_time(t_orig), 0.0, 1.0)

    # 33개 관절 3D 궤적에 대해 3차 스플라인 리샘플링
    aug_kps = np.zeros_like(kps_64f)
    for j in range(33):
        for c in range(3):
            cs_joint = CubicSpline(t_orig, kps_64f[:, j, c], bc_type='natural')
            aug_kps[:, j, c] = cs_joint(t_sampled)

    return aug_kps


def apply_stiff_spine_augmentation(kps_64f: np.ndarray) -> np.ndarray:
    """
    뼈대 길이 가변 + DTW 시간 왜곡 복합 증강 (노이즈 0%)
    """
    # 1. 뼈대 길이 가변
    kps_aug = augment_bone_length_scaling(kps_64f)
    # 2. DTW 시간 축 변경
    kps_aug = augment_dtw_temporal_warping(kps_aug)
    return kps_aug
