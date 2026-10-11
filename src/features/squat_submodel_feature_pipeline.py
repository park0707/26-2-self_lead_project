# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: squat_submodel_feature_pipeline.py
Purpose: c1~c4 세부 결함 모델을 위한 인터페이스 기반 독립 피처 파이프라인
         - 규정 준수:
           1) 인터페이스 기반 독립 모듈화 (외부에서 숫자 인덱스 직접 슬라이싱 영구 금지)
           2) Format A(161D) 피처 이름 매핑(Name-based Lookup)을 통한 정밀 추출 (인덱스 버그 100% 방지)
           3) 기존 프로덕션 버전(c1 v7, c2 v4, c3 v5, c4 v3) 핵심 도메인 피처 계승
           4) c2 VOR 18D 시계열 경추 협응 및 c4 16D 정강이-발목 커플링 완벽 융합
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

sys.path.insert(0, 'd:/')
from src.features.kinematics_3d import KinematicsExtractor3D

VOR_COUP_PATH = r"D:\Data\피처_데이터\squat_c2_vor_and_c4_coupling_feats.npz"

# Format A 161개 피처 이름 리스트 로드 및 딕셔너리 구축
dummy_arr = np.zeros((64, 15, 3), dtype=np.float32)
_, FORMAT_A_NAMES = KinematicsExtractor3D.build_format_a_summary_3d(dummy_arr)
FORMAT_A_NAME_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(FORMAT_A_NAMES)}


def get_format_a_slice_by_names(X_fa: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Format A 161D 배열에서 이름 기반으로 안전하게 피처 슬라이싱 수행
    """
    indices = []
    selected_names = []
    for fn in feature_names:
        if fn in FORMAT_A_NAME_TO_IDX:
            indices.append(FORMAT_A_NAME_TO_IDX[fn])
            selected_names.append(fn)
        else:
            raise KeyError(f"[!] Format A에 존재하지 않는 피처명: '{fn}'")
    
    return X_fa[:, indices].astype(np.float32), selected_names


def get_feature_group_names(prefix: str, stats: Tuple[str, ...] = ('mean', 'std', 'min', 'max', 'delta')) -> List[str]:
    """접두사(prefix)에 5대 통계량 접미사를 붙여 피처명 리스트 생성"""
    return [f"{prefix}_{s}" for s in stats]


# =========================================================================
# 1. c1 척추 중립 (Spine Neutrality) 피처 빌더
# =========================================================================
C1_FA_FEATURE_NAMES = (
    get_feature_group_names("torso_pitch_3d") +
    get_feature_group_names("torso_roll_3d") +
    get_feature_group_names("sh_hip_twist_3d") +
    get_feature_group_names("neck_spine_alignment_3d") +
    get_feature_group_names("barbell_body_distance_3d") +
    get_feature_group_names("knee_hip_synchrony_3d") +
    [
        "torso_pitch_3d_bottom_snapshot",
        "torso_roll_3d_bottom_snapshot",
        "neck_spine_alignment_3d_bottom_snapshot"
    ]
)

def build_c1_features(X_fa: np.ndarray, X_c1_raw: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """c1 특화 10D 피처 + Format A 이름기반 33D 피처 결합 -> 43D 피처셋 반환"""
    fa_slice, fa_names = get_format_a_slice_by_names(X_fa, C1_FA_FEATURE_NAMES)
    c1_names = [f"c1_spine_raw_{i}" for i in range(X_c1_raw.shape[1])] + fa_names
    X_c1_total = np.concatenate([X_c1_raw, fa_slice], axis=1)
    return X_c1_total, c1_names


# =========================================================================
# 2. c2 고개 정면 (Head/Gaze Alignment) 피처 빌더
# =========================================================================
C2_FA_FEATURE_NAMES = (
    get_feature_group_names("neck_spine_alignment_3d") +
    get_feature_group_names("torso_pitch_3d") +
    get_feature_group_names("torso_roll_3d") +
    [
        "neck_spine_alignment_3d_bottom_snapshot",
        "torso_pitch_3d_bottom_snapshot"
    ]
)

C2_HEAD_PATH = r"D:\Data\피처_데이터\squat_c2_full33_head_feats.npz"

def build_c2_features(
    X_fa: np.ndarray,
    X_c2_raw: np.ndarray,
    X_head_18d: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    c2 특화 3D 피처 + Full-33 18D 진정한 경추-체간/시선 3D 각도 피처 + Format A 이름기반 17D 피처 + 상대피치 -> 39D 피처셋 반환
    (후방 뷰 view1에서도 체간-경추 협응 지표를 통해 완벽 커버)
    """
    if X_head_18d is None:
        if os.path.exists(C2_HEAD_PATH):
            d_head = np.load(C2_HEAD_PATH, allow_pickle=True)
            X_head_18d = d_head['X_c2_head_18d']
        elif os.path.exists(VOR_COUP_PATH):
            d_vc = np.load(VOR_COUP_PATH, allow_pickle=True)
            X_head_18d = d_vc['X_c2_vor_18d']

    fa_slice, fa_names = get_format_a_slice_by_names(X_fa, C2_FA_FEATURE_NAMES)
    
    # 파생 피처: 체간 대비 경추 상대 피치 평균
    torso_pitch_mean = X_fa[:, FORMAT_A_NAME_TO_IDX["torso_pitch_3d_mean"]]
    neck_spine_mean = X_fa[:, FORMAT_A_NAME_TO_IDX["neck_spine_alignment_3d_mean"]]
    rel_pitch = (neck_spine_mean - torso_pitch_mean).reshape(-1, 1).astype(np.float32)
    
    c2_names = [f"c2_gaze_raw_{i}" for i in range(X_c2_raw.shape[1])]
    parts = [X_c2_raw]

    if X_head_18d is not None:
        c2_names += [f"c2_head_angle_{i}" for i in range(X_head_18d.shape[1])]
        parts.append(X_head_18d)

    c2_names += fa_names + ["c2_head_to_torso_relative_pitch"]
    parts.extend([fa_slice, rel_pitch])

    X_c2_total = np.concatenate(parts, axis=1)
    return X_c2_total, c2_names


# =========================================================================
# 3. c3 발과 무릎 외반 (Knee Valgus / Alignment) 피처 빌더
# =========================================================================
C3_FA_FEATURE_NAMES = (
    get_feature_group_names("knee_valgus_l_3d") +
    get_feature_group_names("knee_valgus_r_3d") +
    get_feature_group_names("knee_distance_3d") +
    get_feature_group_names("ankle_distance_3d") +
    get_feature_group_names("knee_asymmetry_3d") +         # 정확한 인덱스 (80~84) 매핑
    get_feature_group_names("knee_angle_l_3d") +
    get_feature_group_names("knee_angle_r_3d") +
    [
        "knee_valgus_l_3d_bottom_snapshot",
        "knee_valgus_r_3d_bottom_snapshot",
        "knee_distance_3d_bottom_snapshot"
    ]
)

def build_c3_features(X_fa: np.ndarray, X_c3_raw: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """c3 v5.0 특화 10D 절대너비비 피처 + Format A 이름기반 38D 피처 결합 -> 48D 피처셋 반환"""
    fa_slice, fa_names = get_format_a_slice_by_names(X_fa, C3_FA_FEATURE_NAMES)
    c3_names = [f"c3_valgus_raw_{i}" for i in range(X_c3_raw.shape[1])] + fa_names
    X_c3_total = np.concatenate([X_c3_raw, fa_slice], axis=1)
    return X_c3_total, c3_names


# =========================================================================
# 4. c4 발바닥 지면 고정 (Foot Ground Fixation) 피처 빌더
# =========================================================================
C4_FA_FEATURE_NAMES = (
    get_feature_group_names("shank_verticality_angle_l_3d") +  # 정강이 수직각 L (125~129)
    get_feature_group_names("shank_verticality_angle_r_3d") +  # 정강이 수직각 R (130~134) 완벽 복원
    get_feature_group_names("ankle_angle_l_3d") +              # 발목 각도 L (20~24)
    get_feature_group_names("ankle_angle_r_3d") +              # 발목 각도 R (25~29)
    get_feature_group_names("ankle_z_spread_3d") +             # 발목 Z축 분산 (75~79)
    get_feature_group_names("signed_ankle_z_rel_l_3d") +       # 서명 발목 Z L (105~109)
    get_feature_group_names("signed_ankle_z_rel_r_3d") +       # 서명 발목 Z R (110~114)
    [
        "shank_verticality_angle_l_3d_bottom_snapshot",
        "shank_verticality_angle_r_3d_bottom_snapshot",
        "ankle_angle_l_3d_bottom_snapshot",
        "ankle_angle_r_3d_bottom_snapshot",
        "ankle_z_spread_3d_bottom_snapshot"
    ]
)

def build_c4_features(
    X_fa: np.ndarray,
    X_c4_raw: np.ndarray,
    X_coup_16d: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    c4 v3.0 특화 9D 피처 + 16D 정강이-발목 커플링 피처 + Format A 이름기반 40D 피처 결합 -> 65D 피처셋 반환
    """
    if X_coup_16d is None and os.path.exists(VOR_COUP_PATH):
        d_vc = np.load(VOR_COUP_PATH, allow_pickle=True)
        X_coup_16d = d_vc['X_c4_coup_16d']

    fa_slice, fa_names = get_format_a_slice_by_names(X_fa, C4_FA_FEATURE_NAMES)
    c4_names = [f"c4_foot_raw_{i}" for i in range(X_c4_raw.shape[1])]
    parts = [X_c4_raw]

    if X_coup_16d is not None:
        c4_names += [f"c4_coupling_{i}" for i in range(X_coup_16d.shape[1])]
        parts.append(X_coup_16d)

    c4_names += fa_names
    parts.append(fa_slice)

    X_c4_total = np.concatenate(parts, axis=1)
    return X_c4_total, c4_names
