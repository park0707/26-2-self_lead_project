# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Module: cl2_rtmp_1.py
Version: cl2_rtmp_1 (RTMPose-m Halpe26 Optimized Production Classifier)

주요 사양:
  1. RTMPose Halpe26 전용 생체역학 피처 (82D):
     - 베이스라인 MediaPipe 호환 48D 정예 도메인 피처 (골반 직교계, 시계열 동역학)
     - RTMPose 고유 2D 평면 기하 정합 (외적·내적·발끝 외반각) 17D
     - 상체 관상면 측면 굴곡 및 두부 좌향 회전 비틀림 정량화 피처 8D (Key 184 대응)
     - 단안 3D 역투영 복원(Monocular 3D Reconstruction via Bone Length Constancy) 3D 기하 정합 4D
     - 공식 5대 시점 One-Hot 인코딩 피처 (5D)
  2. CatBoost + LightGBM + XGBoost 3-way 가중 블렌딩 앙상블
  3. Guo et al. (ICML 2017) NLL 및 ECE 최소화 온도 스케일링(Temperature Scaling) 캘리브레이션
  4. AGENTS.md 공식 5대 시점별 적응형 3-State 신뢰도 게이팅 (보류율 <= 10.0% 제약)
  5. 100% 무누수 순수 비전 RTMPose 2D 키포인트 기반 (AI-Hub 라벨 격리)
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union

import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.cross_lunge.cross_lunge_c2_v7_features import (
    extract_cross_lunge_c2_v7_features_51d,
    safe_savgol_filter_kps
)
from src.models.second_models.cross_lunge.cross_lunge_c2_v2_features import (
    compute_v2_cross_lunge_specialized_features_8d,
    FEATURE_NAMES_59D
)
from src.features.cross_lunge_spatial_leg_separator import detect_cross_lunge_lead_leg_spatial
from src.models.second_models.cross_lunge.cl2_v3 import FEATURES_TO_ELIMINATE_11

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "cl2_rtmp_1_bundle.pkl")

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

RETAINED_C2_INDICES = [i for i, fn in enumerate(FEATURE_NAMES_59D) if fn not in FEATURES_TO_ELIMINATE_11]
RETAINED_C2_NAMES = [fn for fn in FEATURE_NAMES_59D if fn not in FEATURES_TO_ELIMINATE_11]

CL2_RTMP_1_EXTRA_FEATURE_NAMES = [
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
    "rtm_c2_sh_diff_y",               # 18. 어깨 좌우 수직 고저차 비율 (몸통 옆으로 구부리기)
    "rtm_c2_sh_diff_x",               # 19. 어깨 수평 너비 정규화
    "rtm_c2_hip_diff_y",              # 20. 골반 좌우 수직 고저차 비율
    "rtm_c2_hip_diff_x",              # 21. 골반 수평 너비 정규화
    "rtm_c2_torso_coronal_angle",     # 22. 체간 관상면 기울기 각도 (°)
    "rtm_c2_head_yaw_asym",           # 23. 두부 시선 좌우 비대칭 편차 (왼쪽 보고)
    "rtm_c2_nose_to_ear_l",           # 24. 코-좌측 귀 거리 비율
    "rtm_c2_nose_to_ear_r",           # 25. 코-우측 귀 거리 비율
    "rtm_c2_dp3_thigh_shank",         # 26. 단안 3D 역투영 대퇴-정강이 내적
    "rtm_c2_dp3_thigh_foot",          # 27. 단안 3D 역투영 대퇴-발 내적
    "rtm_c2_dp3_shank_foot",          # 28. 단안 3D 역투영 정강이-발 내적
    "rtm_c2_is_left_lead",            # 29. 좌측 앞다리 여부
    "view_is_view5",                  # 30. 시점 A (우측 후방 45°)
    "view_is_view2",                  # 31. 시점 B (우측 전방 45°)
    "view_is_view3",                  # 32. 시점 C (정면 0°)
    "view_is_view4",                  # 33. 시점 D (좌측 전방 45°)
    "view_is_view1"                   # 34. 시점 E (좌측 후방 45°)
]

CL2_RTMP_1_FEATURE_NAMES = RETAINED_C2_NAMES + CL2_RTMP_1_EXTRA_FEATURE_NAMES


def halpe26_to_mp33_c2(kps_26: np.ndarray) -> np.ndarray:
    """Halpe26 키포인트를 MediaPipe33 호환 텐서로 매핑"""
    T = kps_26.shape[0]
    mp33 = np.zeros((T, 33, 3), dtype=np.float32)
    mapping = {
        0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14,
        9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
        20: 31, 21: 32, 24: 29, 25: 30
    }
    for h_idx, m_idx in mapping.items():
        mp33[:, m_idx] = kps_26[:, h_idx]
    return mp33


def normalize_2d_c2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm < 1e-6, 0.0, v / (norm + 1e-6))


def reconstruct_3d_vector_c2(v_2d_seq: np.ndarray, sign_z: float = 1.0) -> np.ndarray:
    l_2d = np.linalg.norm(v_2d_seq, axis=-1)
    L_true = np.percentile(l_2d, 95)
    z_mag = np.sqrt(np.clip(L_true**2 - l_2d**2, 0, None))
    z = z_mag * sign_z
    v_3d = np.column_stack([v_2d_seq, z])
    norms = np.linalg.norm(v_3d, axis=-1, keepdims=True) + 1e-6
    return v_3d / norms


def extract_cl2_rtmp_1_features(kps_26_64f: np.ndarray, view_code: str = 'view3') -> np.ndarray:
    """
    (64, 26, 3) RTMPose 텐서로부터 cl2_rtmp_1 전용 82차원 정예 피처 벡터 추출
    """
    assert kps_26_64f.ndim == 3 and kps_26_64f.shape[0] == 64, f"입력 규격 불일치: {kps_26_64f.shape}"
    kps_2d = kps_26_64f[:, :, :2]
    conf = kps_26_64f[:, :, 2]
    T = kps_2d.shape[0]

    v_code_clean = VIEW_NAME_TO_CODE.get(str(view_code).strip(), 'view3')

    # 1. MediaPipe 호환 48D 피처 추출
    mp33 = halpe26_to_mp33_c2(kps_26_64f)
    kps_clean = safe_savgol_filter_kps(mp33[:, :, :3], window=7, poly=2)
    is_left_lead_sp, t_bot_sp = detect_cross_lunge_lead_leg_spatial(kps_clean)
    f_51d = extract_cross_lunge_c2_v7_features_51d(kps_clean, view_code=v_code_clean)
    f_8d = compute_v2_cross_lunge_specialized_features_8d(kps_clean, is_left_lead_sp, t_bot_sp)
    f_59d = np.concatenate([f_51d, f_8d])
    f_base_48d = f_59d[RETAINED_C2_INDICES]

    # 2. RTMPose 고유 34D 피처 추출
    l_knee_y = kps_2d[:, 13, 1]
    r_knee_y = kps_2d[:, 14, 1]
    t_l = int(np.argmax(l_knee_y))
    t_r = int(np.argmax(r_knee_y))
    if l_knee_y[t_l] >= r_knee_y[t_r]:
        is_left_lead = True
        t_bot = t_l
        hip_l, knee_l, ank_l = 11, 13, 15
        toe_big_l, toe_sml_l, heel_l = 20, 22, 24
    else:
        is_left_lead = False
        t_bot = t_r
        hip_l, knee_l, ank_l = 12, 14, 16
        toe_big_l, toe_sml_l, heel_l = 21, 23, 25

    t_bot = max(4, min(T - 5, t_bot))
    win = slice(max(0, t_bot - 3), min(T, t_bot + 4))

    k_std = np.mean(kps_2d[:5], axis=0)
    k_bot = np.mean(kps_2d[win], axis=0)

    pelvis_bot = k_bot[19] if conf[t_bot, 19] > 0.3 else (k_bot[11] + k_bot[12]) * 0.5
    neck_bot = k_bot[18] if conf[t_bot, 18] > 0.3 else (k_bot[5] + k_bot[6]) * 0.5
    v_trunk = normalize_2d_c2(neck_bot - pelvis_bot)
    trunk_len = np.linalg.norm(neck_bot - pelvis_bot) + 1e-6

    v_thigh = normalize_2d_c2(k_bot[knee_l] - k_bot[hip_l])
    v_shank = normalize_2d_c2(k_bot[ank_l] - k_bot[knee_l])
    mid_toe_bot = (k_bot[toe_big_l] + k_bot[toe_sml_l]) * 0.5
    v_foot = normalize_2d_c2(mid_toe_bot - k_bot[heel_l])

    cp_trunk_thigh = float(v_trunk[0]*v_thigh[1] - v_trunk[1]*v_thigh[0])
    dp_trunk_thigh = float(v_trunk[0]*v_thigh[0] + v_trunk[1]*v_thigh[1])
    cp_thigh_foot = float(v_thigh[0]*v_foot[1] - v_thigh[1]*v_foot[0])
    dp_thigh_foot = float(v_thigh[0]*v_foot[0] + v_thigh[1]*v_foot[1])
    cp_trunk_foot = float(v_trunk[0]*v_foot[1] - v_trunk[1]*v_foot[0])
    dp_trunk_foot = float(v_trunk[0]*v_foot[0] + v_trunk[1]*v_foot[1])
    cp_shank_foot = float(v_shank[0]*v_foot[1] - v_shank[1]*v_foot[0])
    dp_shank_foot = float(v_shank[0]*v_foot[0] + v_shank[1]*v_foot[1])
    cp_thigh_shank = float(v_thigh[0]*v_shank[1] - v_thigh[1]*v_shank[0])
    dp_thigh_shank = float(v_thigh[0]*v_shank[0] + v_thigh[1]*v_shank[1])

    toe_heel_x_diff = float((mid_toe_bot[0] - k_bot[heel_l][0]) / trunk_len)
    toe_heel_y_diff = float((mid_toe_bot[1] - k_bot[heel_l][1]) / trunk_len)
    foot_width = float(np.linalg.norm(k_bot[toe_sml_l] - k_bot[toe_big_l]) / trunk_len)
    ankle_knee_x_offset = float((k_bot[ank_l][0] - k_bot[knee_l][0]) / trunk_len)
    knee_toe_x_offset = float((mid_toe_bot[0] - k_bot[knee_l][0]) / trunk_len)

    v_foot_std = normalize_2d_c2((k_std[toe_big_l] + k_std[toe_sml_l]) * 0.5 - k_std[heel_l])
    delta_foot_dp = float(np.dot(v_foot, v_foot_std))
    delta_foot_cp = float(v_foot[0]*v_foot_std[1] - v_foot[1]*v_foot_std[0])

    sh_diff_y = float((k_bot[5][1] - k_bot[6][1]) / trunk_len)
    sh_diff_x = float(abs(k_bot[5][0] - k_bot[6][0]) / trunk_len)
    hip_diff_y = float((k_bot[11][1] - k_bot[12][1]) / trunk_len)
    hip_diff_x = float(abs(k_bot[11][0] - k_bot[12][0]) / trunk_len)
    torso_coronal_angle = float(np.degrees(np.arctan2(k_bot[5][1] - k_bot[6][1], k_bot[5][0] - k_bot[6][0])))

    nose_to_ear_l = float(np.linalg.norm(k_bot[0] - k_bot[3]) / trunk_len)
    nose_to_ear_r = float(np.linalg.norm(k_bot[0] - k_bot[4]) / trunk_len)
    head_yaw_asym = float(nose_to_ear_l - nose_to_ear_r)

    v_thigh_seq = kps_2d[:, knee_l] - kps_2d[:, hip_l]
    v_shank_seq = kps_2d[:, ank_l] - kps_2d[:, knee_l]
    v_foot_seq = kps_2d[:, toe_big_l] - kps_2d[:, heel_l]
    u3_thigh = reconstruct_3d_vector_c2(v_thigh_seq)
    u3_shank = reconstruct_3d_vector_c2(v_shank_seq)
    u3_foot = reconstruct_3d_vector_c2(v_foot_seq)

    dp3_thigh_shank = float(np.dot(u3_thigh[t_bot], u3_shank[t_bot]))
    dp3_thigh_foot = float(np.dot(u3_thigh[t_bot], u3_foot[t_bot]))
    dp3_shank_foot = float(np.dot(u3_shank[t_bot], u3_foot[t_bot]))

    v_vec = [
        1.0 if v_code_clean == 'view5' else 0.0,
        1.0 if v_code_clean == 'view2' else 0.0,
        1.0 if v_code_clean == 'view3' else 0.0,
        1.0 if v_code_clean == 'view4' else 0.0,
        1.0 if v_code_clean == 'view1' else 0.0
    ]

    extra_34d = [
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
        float(is_left_lead)
    ] + v_vec

    f_total_82d = np.concatenate([f_base_48d, np.array(extra_34d, dtype=np.float32)])
    return np.nan_to_num(f_total_82d, nan=0.0)


class CL2RTMP1Classifier:
    """cl2_rtmp_1 공식 프로덕션 추론기"""
    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"cl2_rtmp_1 가중치 번들이 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = "cl2_rtmp_1"
        self.defect_name = "몸통 앞발 앞무릎 방향 일치 여부 (RTMPose 82D 최적화 v1.0)"
        self.feature_names = self.bundle['feature_names']
        self.models_cb = self.bundle['models_cat']
        self.models_lgb = self.bundle['models_lgb']
        self.models_xgb = self.bundle['models_xgb']
        self.weights = self.bundle['ensemble_weights']
        self.temperature = float(self.bundle.get('temperature', 1.0))
        self.view_thresholds = self.bundle['view_thresholds']
        self.global_threshold = float(self.bundle.get('global_threshold', 0.50))

    def predict_features(self, feature_vector: np.ndarray, camera_view: str = '시점 C') -> Dict[str, Any]:
        """82D 피처 벡터 기반 추론"""
        X_in = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if X_in.shape[1] != 82:
            raise ValueError(f"입력 피처 차원이 82D여야 합니다. (현재: {X_in.shape[1]}D)")

        v_code = VIEW_NAME_TO_CODE.get(str(camera_view).strip(), 'view3')
        v_name = VIEW_CODE_TO_NAME[v_code]

        p_cb = float(np.mean([m.predict_proba(X_in)[0, 1] for m in self.models_cb]))
        p_lgb = float(np.mean([m.predict_proba(X_in)[0, 1] for m in self.models_lgb]))
        p_xgb = float(np.mean([m.predict_proba(X_in)[0, 1] for m in self.models_xgb]))

        w_cb, w_lgb, w_xgb = self.weights
        p_ens = float(w_cb * p_cb + w_lgb * p_lgb + w_xgb * p_xgb)

        # 온도 보정
        eps = 1e-7
        p_clipped = np.clip(p_ens, eps, 1.0 - eps)
        logit = np.log(p_clipped / (1.0 - p_clipped))
        prob_calibrated = float(1.0 / (1.0 + np.exp(-logit / max(0.01, self.temperature))))

        th_info = self.view_thresholds.get(v_code, {'tau_clean': 0.45, 'tau_defect': 0.50})
        tau_clean = float(th_info['tau_clean'])
        tau_defect = float(th_info['tau_defect'])

        is_defect_2s = bool(prob_calibrated >= self.global_threshold)
        state_2s = "DEFECT" if is_defect_2s else "NORMAL"

        if prob_calibrated < tau_clean:
            state_3s = "NORMAL"
        elif prob_calibrated >= tau_defect:
            state_3s = "DEFECT"
        else:
            state_3s = "UNCERTAIN"

        return {
            'model_version': self.model_name,
            'defect_name': self.defect_name,
            'camera_view': v_name,
            'camera_view_code': v_code,
            'raw_probability': p_ens,
            'defect_probability': prob_calibrated,
            'temperature': self.temperature,
            'prediction_2state': state_2s,
            'is_defect': is_defect_2s,
            'prediction_3state': state_3s,
            'tau_clean': tau_clean,
            'tau_defect': tau_defect
        }

    def predict(self, kps_26_64f: np.ndarray, camera_view: str = '시점 C') -> Dict[str, Any]:
        """(64, 26, 3) 관절 텐서 기반 엔드투엔드 추론"""
        v_code = VIEW_NAME_TO_CODE.get(str(camera_view).strip(), 'view3')
        feats = extract_cl2_rtmp_1_features(kps_26_64f, view_code=v_code)
        return self.predict_features(feats, camera_view=camera_view)


# 싱글톤 캐시
_CACHED_CL2_CLASSIFIER: Optional[CL2RTMP1Classifier] = None

def predict_cross_lunge_c2_rtmp_1(
    kps_or_features: np.ndarray,
    camera_view: str = '시점 C',
    bundle_path: Optional[str] = None
) -> Dict[str, Any]:
    global _CACHED_CL2_CLASSIFIER
    if _CACHED_CL2_CLASSIFIER is None or bundle_path is not None:
        _CACHED_CL2_CLASSIFIER = CL2RTMP1Classifier(bundle_path=bundle_path)

    arr = np.asarray(kps_or_features)
    if arr.ndim == 3:
        return _CACHED_CL2_CLASSIFIER.predict(arr, camera_view=camera_view)
    elif arr.ndim in (1, 2):
        vec = arr.flatten()
        return _CACHED_CL2_CLASSIFIER.predict_features(vec, camera_view=camera_view)
    else:
        raise ValueError(f"지원하지 않는 입력 배열 규격: {arr.shape}")
