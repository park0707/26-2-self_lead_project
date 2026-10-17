# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Module: cl2_hybrid_1.py
Version: cl2_hybrid_1 (Dual Pose Estimator Hybrid: MediaPipe 3D + RTMPose Halpe26 2D)

주요 사양:
  1. MediaPipe 3D + RTMPose Halpe26 크로스모달 융합 피처 (86D):
     - MediaPipe 33관절 기반 3D PACS 골반 직교계 및 체간-대퇴 3D 공간 벡터 (51D)
     - RTMPose Halpe26 기반 고정밀 2D 발 3점(엄지·새끼·뒤꿈치) 외반 및 상체 비대칭 (29D)
     - MediaPipe-RTMPose 크로스모달 상호작용 피처 (6D)
  2. CatBoost + LightGBM + XGBoost 3-way 최적 가중치 블렌딩 앙상블
  3. Guo et al. (ICML 2017) NLL 및 ECE 최소화 온도 스케일링(Temperature Scaling) 캘리브레이션
  4. AGENTS.md 공식 5대 시점별 적응형 3-State 신뢰도 게이팅 (보류율 <= 10.0% 제약)
  5. 100% 무누수 순수 비전 텐서 기반 (AI-Hub 라벨 철저 격리)
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.cross_lunge.cross_lunge_c2_v7_features import (
    extract_cross_lunge_c2_v7_features_51d,
    safe_savgol_filter_kps,
    FEATURE_NAMES_51D
)
from src.features.cross_lunge_spatial_leg_separator import detect_cross_lunge_lead_leg_spatial

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "cl2_hybrid_1_bundle.pkl")

# AGENTS.md 공식 5대 시점 매핑 규격 준수
VIEW_NAME_TO_CODE = {
    '시점 A': 'view5', 'view5': 'view5', 'view_5': 'view5', '시점 A: 우측 후방 (45도)': 'view5',
    '시점 B': 'view2', 'view2': 'view2', 'view_2': 'view2', '시점 B: 우측 전방 (45도)': 'view2',
    '시점 C': 'view3', 'view3': 'view3', 'view_3': 'view3', '시점 C: 정면 (0도)': 'view3',
    '시점 D': 'view4', 'view4': 'view4', 'view_4': 'view4', '시점 D: 좌측 전방 (45도)': 'view4',
    '시점 E': 'view1', 'view1': 'view1', 'view_1': 'view1', '시점 E: 좌측 후방 (45도)': 'view1',
}

VIEW_CODE_TO_NAME = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}

# RTMPose 고유 29D 피처 명칭 (시점 One-Hot 5D 제외)
RTM_UNIQUE_FEATURE_NAMES_29D = [
    "rtm_c2_cp_trunk_thigh",          # 1. 2D 체간-대퇴골 2D 외적 (좌우 편향)
    "rtm_c2_dp_trunk_thigh",          # 2. 2D 체간-대퇴골 2D 내적 (전방 일치)
    "rtm_c2_cp_thigh_foot",           # 3. 2D 대퇴골-발 2D 외적 (대퇴 대비 발 회전)
    "rtm_c2_dp_thigh_foot",           # 4. 2D 대퇴골-발 2D 내적
    "rtm_c2_cp_trunk_foot",           # 5. 2D 체간-발 2D 외적
    "rtm_c2_dp_trunk_foot",           # 6. 2D 체간-발 2D 내적
    "rtm_c2_cp_shank_foot",           # 7. 2D 정강이-발 2D 외적
    "rtm_c2_dp_shank_foot",           # 8. 2D 정강이-발 2D 내적
    "rtm_c2_cp_thigh_shank",          # 9. 2D 대퇴-정강이 2D 외적
    "rtm_c2_dp_thigh_shank",          # 10. 2D 대퇴-정강이 2D 내적
    "rtm_c2_toe_heel_x_diff",         # 11. 2D 발끝-뒤꿈치 X 편차 비율 (외반 정량화)
    "rtm_c2_toe_heel_y_diff",         # 12. 2D 발끝-뒤꿈치 Y 편차 비율
    "rtm_c2_foot_width_ratio",        # 13. 2D 엄지-새끼발가락 횡폭 비율
    "rtm_c2_ankle_knee_x_offset",     # 14. 2D 발목-무릎 횡방향 이탈 거리
    "rtm_c2_knee_toe_x_offset",       # 15. 2D 무릎-발끝 횡방향 이탈 거리
    "rtm_c2_delta_foot_dp",           # 16. 기립 대비 최하사점 발 방향 내적 변화량
    "rtm_c2_delta_foot_cp",           # 17. 기립 대비 최하사점 발 외적 회전 변화량
    "rtm_c2_sh_diff_y",               # 18. 어깨 좌우 수직 고저차 비율
    "rtm_c2_sh_diff_x",               # 19. 어깨 수평 너비 정규화
    "rtm_c2_hip_diff_y",              # 20. 골반 좌우 수직 고저차 비율
    "rtm_c2_hip_diff_x",              # 21. 골반 수평 너비 정규화
    "rtm_c2_torso_coronal_angle",     # 22. 체간 관상면 기울기 각도 (°)
    "rtm_c2_head_yaw_asym",           # 23. 두부 시선 좌우 비대칭 편차
    "rtm_c2_nose_to_ear_l",           # 24. 코-좌측 귀 거리 비율
    "rtm_c2_nose_to_ear_r",           # 25. 코-우측 귀 거리 비율
    "rtm_c2_dp3_thigh_shank",         # 26. 단안 3D 역투영 대퇴-정강이 내적
    "rtm_c2_dp3_thigh_foot",          # 27. 단안 3D 역투영 대퇴-발 내적
    "rtm_c2_dp3_shank_foot",          # 28. 단안 3D 역투영 정강이-발 내적
    "rtm_c2_is_left_lead",            # 29. RTMPose 기준 좌측 앞다리 여부
]

CROSS_MODAL_FEATURE_NAMES_6D = [
    "cross_mp_trunk_rtm_foot_dp",      # 1. MediaPipe 3D 체간 2D 투영 vs RTMPose 발 벡터 내적
    "cross_mp_trunk_rtm_foot_cp",      # 2. MediaPipe 3D 체간 2D 투영 vs RTMPose 발 벡터 외적
    "cross_mp_thigh_rtm_foot_dp",      # 3. MediaPipe 앞대퇴 벡터 vs RTMPose 발 벡터 내적
    "cross_mp_thigh_rtm_foot_cp",      # 4. MediaPipe 앞대퇴 벡터 vs RTMPose 발 벡터 외적
    "cross_mp_rtm_knee_x_diff",        # 5. MediaPipe-RTMPose 앞무릎 횡방향 정합 오차
    "cross_mp_rtm_lead_leg_agree"      # 6. 앞다리 판정 상호 일치 여부 (1.0 / 0.0)
]

CL2_HYBRID_1_FEATURE_NAMES = FEATURE_NAMES_51D + RTM_UNIQUE_FEATURE_NAMES_29D + CROSS_MODAL_FEATURE_NAMES_6D


def normalize_2d_vec(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm < 1e-6, 0.0, v / (norm + 1e-6))


def reconstruct_3d_vector(v_2d_seq: np.ndarray, sign_z: float = 1.0) -> np.ndarray:
    l_2d = np.linalg.norm(v_2d_seq, axis=-1)
    L_true = np.percentile(l_2d, 95)
    z_mag = np.sqrt(np.clip(L_true**2 - l_2d**2, 0, None))
    z = z_mag * sign_z
    v_3d = np.column_stack([v_2d_seq, z])
    norms = np.linalg.norm(v_3d, axis=-1, keepdims=True) + 1e-6
    return v_3d / norms


def extract_cl2_hybrid_1_features(
    mp33_64f: np.ndarray,
    kps_26_64f: np.ndarray,
    view_code: str = 'view3'
) -> np.ndarray:
    """
    MediaPipe 33 (64, 33, 3) + RTMPose 26 (64, 26, 3) 텐서로부터
    cl2_hybrid_1 전용 86차원 크로스모달 정예 피처 벡터 추출
    """
    assert mp33_64f.shape[0] == 64 and mp33_64f.shape[1] == 33, f"MediaPipe 규격 불일치: {mp33_64f.shape}"
    assert kps_26_64f.shape[0] == 64 and kps_26_64f.shape[1] == 26, f"RTMPose 규격 불일치: {kps_26_64f.shape}"

    v_code_clean = VIEW_NAME_TO_CODE.get(str(view_code).strip(), 'view3')

    # =========================================================================
    # 1. MediaPipe 3D 51D 피처 추출
    # =========================================================================
    mp_clean = safe_savgol_filter_kps(mp33_64f[:, :, :3], window=7, poly=2)
    f_mp_51d = extract_cross_lunge_c2_v7_features_51d(mp_clean, view_code=v_code_clean)
    is_left_lead_mp, t_bot_mp = detect_cross_lunge_lead_leg_spatial(mp_clean)

    # =========================================================================
    # 2. RTMPose Halpe26 고유 29D 피처 추출
    # =========================================================================
    kps_2d = kps_26_64f[:, :, :2]
    conf = kps_26_64f[:, :, 2]
    T = kps_2d.shape[0]

    l_knee_y = kps_2d[:, 13, 1]
    r_knee_y = kps_2d[:, 14, 1]
    t_l = int(np.argmax(l_knee_y))
    t_r = int(np.argmax(r_knee_y))
    if l_knee_y[t_l] >= r_knee_y[t_r]:
        is_left_lead_rtm = True
        t_bot_rtm = t_l
        hip_l, knee_l, ank_l = 11, 13, 15
        toe_big_l, toe_sml_l, heel_l = 20, 22, 24
    else:
        is_left_lead_rtm = False
        t_bot_rtm = t_r
        hip_l, knee_l, ank_l = 12, 14, 16
        toe_big_l, toe_sml_l, heel_l = 21, 23, 25

    t_bot_rtm = max(4, min(T - 5, t_bot_rtm))
    win = slice(max(0, t_bot_rtm - 3), min(T, t_bot_rtm + 4))

    k_std = np.mean(kps_2d[:5], axis=0)
    k_bot = np.mean(kps_2d[win], axis=0)

    pelvis_bot = k_bot[19] if conf[t_bot_rtm, 19] > 0.3 else (k_bot[11] + k_bot[12]) * 0.5
    neck_bot = k_bot[18] if conf[t_bot_rtm, 18] > 0.3 else (k_bot[5] + k_bot[6]) * 0.5
    v_trunk_rtm = normalize_2d_vec(neck_bot - pelvis_bot)
    trunk_len_rtm = np.linalg.norm(neck_bot - pelvis_bot) + 1e-6

    v_thigh_rtm = normalize_2d_vec(k_bot[knee_l] - k_bot[hip_l])
    v_shank_rtm = normalize_2d_vec(k_bot[ank_l] - k_bot[knee_l])
    mid_toe_bot = (k_bot[toe_big_l] + k_bot[toe_sml_l]) * 0.5
    v_foot_rtm = normalize_2d_vec(mid_toe_bot - k_bot[heel_l])

    cp_trunk_thigh = float(v_trunk_rtm[0]*v_thigh_rtm[1] - v_trunk_rtm[1]*v_thigh_rtm[0])
    dp_trunk_thigh = float(v_trunk_rtm[0]*v_thigh_rtm[0] + v_trunk_rtm[1]*v_thigh_rtm[1])
    cp_thigh_foot = float(v_thigh_rtm[0]*v_foot_rtm[1] - v_thigh_rtm[1]*v_foot_rtm[0])
    dp_thigh_foot = float(v_thigh_rtm[0]*v_foot_rtm[0] + v_thigh_rtm[1]*v_foot_rtm[1])
    cp_trunk_foot = float(v_trunk_rtm[0]*v_foot_rtm[1] - v_trunk_rtm[1]*v_foot_rtm[0])
    dp_trunk_foot = float(v_trunk_rtm[0]*v_foot_rtm[0] + v_trunk_rtm[1]*v_foot_rtm[1])
    cp_shank_foot = float(v_shank_rtm[0]*v_foot_rtm[1] - v_shank_rtm[1]*v_foot_rtm[0])
    dp_shank_foot = float(v_shank_rtm[0]*v_foot_rtm[0] + v_shank_rtm[1]*v_foot_rtm[1])
    cp_thigh_shank = float(v_thigh_rtm[0]*v_shank_rtm[1] - v_thigh_rtm[1]*v_shank_rtm[0])
    dp_thigh_shank = float(v_thigh_rtm[0]*v_shank_rtm[0] + v_thigh_rtm[1]*v_shank_rtm[1])

    toe_heel_x_diff = float((mid_toe_bot[0] - k_bot[heel_l][0]) / trunk_len_rtm)
    toe_heel_y_diff = float((mid_toe_bot[1] - k_bot[heel_l][1]) / trunk_len_rtm)
    foot_width = float(np.linalg.norm(k_bot[toe_sml_l] - k_bot[toe_big_l]) / trunk_len_rtm)
    ankle_knee_x_offset = float((k_bot[ank_l][0] - k_bot[knee_l][0]) / trunk_len_rtm)
    knee_toe_x_offset = float((mid_toe_bot[0] - k_bot[knee_l][0]) / trunk_len_rtm)

    v_foot_std = normalize_2d_vec((k_std[toe_big_l] + k_std[toe_sml_l]) * 0.5 - k_std[heel_l])
    delta_foot_dp = float(np.dot(v_foot_rtm, v_foot_std))
    delta_foot_cp = float(v_foot_rtm[0]*v_foot_std[1] - v_foot_rtm[1]*v_foot_std[0])

    sh_diff_y = float((k_bot[5][1] - k_bot[6][1]) / trunk_len_rtm)
    sh_diff_x = float(abs(k_bot[5][0] - k_bot[6][0]) / trunk_len_rtm)
    hip_diff_y = float((k_bot[11][1] - k_bot[12][1]) / trunk_len_rtm)
    hip_diff_x = float(abs(k_bot[11][0] - k_bot[12][0]) / trunk_len_rtm)
    torso_coronal_angle = float(np.degrees(np.arctan2(k_bot[5][1] - k_bot[6][1], k_bot[5][0] - k_bot[6][0])))

    nose_to_ear_l = float(np.linalg.norm(k_bot[0] - k_bot[3]) / trunk_len_rtm)
    nose_to_ear_r = float(np.linalg.norm(k_bot[0] - k_bot[4]) / trunk_len_rtm)
    head_yaw_asym = float(nose_to_ear_l - nose_to_ear_r)

    v_thigh_seq = kps_2d[:, knee_l] - kps_2d[:, hip_l]
    v_shank_seq = kps_2d[:, ank_l] - kps_2d[:, knee_l]
    v_foot_seq = kps_2d[:, toe_big_l] - kps_2d[:, heel_l]
    u3_thigh = reconstruct_3d_vector(v_thigh_seq)
    u3_shank = reconstruct_3d_vector(v_shank_seq)
    u3_foot = reconstruct_3d_vector(v_foot_seq)

    dp3_thigh_shank = float(np.dot(u3_thigh[t_bot_rtm], u3_shank[t_bot_rtm]))
    dp3_thigh_foot = float(np.dot(u3_thigh[t_bot_rtm], u3_foot[t_bot_rtm]))
    dp3_shank_foot = float(np.dot(u3_shank[t_bot_rtm], u3_foot[t_bot_rtm]))

    f_rtm_29d = [
        cp_trunk_thigh, dp_trunk_thigh,
        cp_thigh_foot, dp_thigh_foot,
        cp_trunk_foot, dp_trunk_foot,
        cp_shank_foot, dp_shank_foot,
        cp_thigh_shank, dp_thigh_shank,
        toe_heel_x_diff, toe_heel_y_diff,
        foot_width, ankle_knee_x_offset, knee_toe_x_offset,
        delta_foot_dp, delta_foot_cp,
        sh_diff_y, sh_diff_x, hip_diff_y, hip_diff_x, torso_coronal_angle,
        head_yaw_asym, nose_to_ear_l, nose_to_ear_r,
        dp3_thigh_shank, dp3_thigh_foot, dp3_shank_foot,
        float(is_left_lead_rtm)
    ]

    # =========================================================================
    # 3. MediaPipe 3D <-> RTMPose 2D 크로스모달 상호작용 피처 (6D)
    # =========================================================================
    # MediaPipe의 체간 2D 투영 벡터
    sh_mid_mp = (mp_clean[t_bot_mp, 11, :2] + mp_clean[t_bot_mp, 12, :2]) * 0.5
    hip_mid_mp = (mp_clean[t_bot_mp, 23, :2] + mp_clean[t_bot_mp, 24, :2]) * 0.5
    v_trunk_mp_2d = normalize_2d_vec(sh_mid_mp - hip_mid_mp)

    # MediaPipe 앞대퇴 2D 투영 벡터
    hip_lead_idx_mp = 23 if is_left_lead_mp else 24
    knee_lead_idx_mp = 25 if is_left_lead_mp else 26
    v_thigh_mp_2d = normalize_2d_vec(mp_clean[t_bot_mp, knee_lead_idx_mp, :2] - mp_clean[t_bot_mp, hip_lead_idx_mp, :2])

    # 1~4: MediaPipe 체간/대퇴 vs RTMPose 발 벡터 사잇각 (내적/외적)
    cross_mp_trunk_rtm_foot_dp = float(np.dot(v_trunk_mp_2d, v_foot_rtm))
    cross_mp_trunk_rtm_foot_cp = float(v_trunk_mp_2d[0]*v_foot_rtm[1] - v_trunk_mp_2d[1]*v_foot_rtm[0])
    cross_mp_thigh_rtm_foot_dp = float(np.dot(v_thigh_mp_2d, v_foot_rtm))
    cross_mp_thigh_rtm_foot_cp = float(v_thigh_mp_2d[0]*v_foot_rtm[1] - v_thigh_mp_2d[1]*v_foot_rtm[0])

    # 5: 앞무릎 횡방향 정합 오차 (스케일 정규화)
    mp_knee_x_norm = float(mp_clean[t_bot_mp, knee_lead_idx_mp, 0] - hip_mid_mp[0])
    rtm_knee_x_norm = float(k_bot[knee_l, 0] - pelvis_bot[0]) / trunk_len_rtm
    cross_mp_rtm_knee_x_diff = float(mp_knee_x_norm - rtm_knee_x_norm)

    # 6: 앞다리 판정 상호 일치 여부
    cross_mp_rtm_lead_leg_agree = 1.0 if (is_left_lead_mp == is_left_lead_rtm) else 0.0

    f_cross_6d = [
        cross_mp_trunk_rtm_foot_dp,
        cross_mp_trunk_rtm_foot_cp,
        cross_mp_thigh_rtm_foot_dp,
        cross_mp_thigh_rtm_foot_cp,
        cross_mp_rtm_knee_x_diff,
        cross_mp_rtm_lead_leg_agree
    ]

    f_total_86d = np.concatenate([
        f_mp_51d,
        np.array(f_rtm_29d, dtype=np.float32),
        np.array(f_cross_6d, dtype=np.float32)
    ])

    return np.nan_to_num(f_total_86d, nan=0.0)


class CL2Hybrid1Classifier:
    """
    크로스 런지 c2 하이브리드 프로덕션 분류기 (MediaPipe 3D + RTMPose Halpe26 2D)
    """
    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        if not os.path.exists(self.bundle_path):
            raise FileNotFoundError(f"번들 파일을 찾을 수 없습니다: {self.bundle_path}")
        with open(self.bundle_path, 'rb') as f:
            self.bundle = pickle.load(f)

        self.models_cat = self.bundle['models_cat']
        self.models_lgb = self.bundle['models_lgb']
        self.models_xgb = self.bundle['models_xgb']
        self.weights = self.bundle.get('weights', [0.34, 0.33, 0.33])
        self.temperature = self.bundle.get('temperature', 1.0)
        self.view_thresholds = self.bundle['view_thresholds']
        self.metadata = self.bundle.get('metadata', {})

    def predict_raw_prob(self, features_86d: np.ndarray) -> float:
        feat = features_86d.reshape(1, -1)
        p_cat = np.mean([m.predict_proba(feat)[0, 1] for m in self.models_cat])
        p_lgb = np.mean([m.predict(feat)[0] for m in self.models_lgb])

        import xgboost as xgb
        dmat = xgb.DMatrix(feat)
        p_xgb = np.mean([m.predict(dmat)[0] for m in self.models_xgb])

        p_blend = self.weights[0]*p_cat + self.weights[1]*p_lgb + self.weights[2]*p_xgb

        # 온도 스케일링 캘리브레이션
        eps = 1e-7
        p_blend = np.clip(p_blend, eps, 1.0 - eps)
        logit = np.log(p_blend / (1.0 - p_blend))
        calibrated_prob = 1.0 / (1.0 + np.exp(-logit / max(0.1, self.temperature)))
        return float(np.clip(calibrated_prob, 0.0, 1.0))

    def predict_3state(
        self,
        mp33_64f: np.ndarray,
        kps_26_64f: np.ndarray,
        view_code: str = 'view3'
    ) -> Dict[str, Any]:
        v_code_clean = VIEW_NAME_TO_CODE.get(str(view_code).strip(), 'view3')
        v_name = VIEW_CODE_TO_NAME.get(v_code_clean, '시점 C (정면 0°)')

        features_86d = extract_cl2_hybrid_1_features(mp33_64f, kps_26_64f, view_code=v_code_clean)
        prob = self.predict_raw_prob(features_86d)

        th = self.view_thresholds.get(v_code_clean, {'tau_clean': 0.40, 'tau_defect': 0.60})
        tau_clean = th['tau_clean']
        tau_defect = th['tau_defect']

        if prob <= tau_clean:
            state = "CLEAN"
            state_kr = "정상"
            defect_prediction = False
            confidence = float((tau_clean - prob) / max(1e-4, tau_clean))
            retake_reason = None
        elif prob >= tau_defect:
            state = "DEFECT"
            state_kr = "오류 (몸통-앞발-앞무릎 불일치)"
            defect_prediction = True
            confidence = float((prob - tau_defect) / max(1e-4, 1.0 - tau_defect))
            retake_reason = None
        else:
            state = "RETAKE"
            state_kr = "판정 보류 (재촬영 권고)"
            defect_prediction = None
            dist_to_edge = min(prob - tau_clean, tau_defect - prob)
            margin_width = max(1e-4, tau_defect - tau_clean)
            confidence = float(1.0 - (dist_to_edge / (margin_width * 0.5)))
            retake_reason = (
                f"{v_name}에서 몸통, 앞무릎, 앞발의 방향 회전각 정렬이 모호합니다. "
                "몸통과 앞발, 앞무릎이 같은 전방 방향을 유지하도록 정렬하여 다시 촬영해 주세요."
            )

        return {
            'defect_code': 'cl2',
            'defect_name': 'Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부',
            'state': state,
            'state_kr': state_kr,
            'defect_prediction': defect_prediction,
            'defect_probability': prob,
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'view_code': v_code_clean,
            'view_name': v_name,
            'tau_clean': float(tau_clean),
            'tau_defect': float(tau_defect),
            'retake_reason': retake_reason,
            'features_86d': features_86d
        }


def predict_cross_lunge_c2_hybrid(
    mp33_64f: np.ndarray,
    kps_26_64f: np.ndarray,
    view_code: str = 'view3',
    bundle_path: Optional[str] = None
) -> Dict[str, Any]:
    clf = CL2Hybrid1Classifier(bundle_path=bundle_path)
    return clf.predict_3state(mp33_64f, kps_26_64f, view_code=view_code)
