# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 1 - 앞다리 무릎 각도 90도 오류 (Lead Knee 90 Degree Angle Defect)
Module: cl1_rtmp_1.py
Version: cl1_rtmp_1 (RTMPose-m Halpe26 Optimized Production Classifier)

주요 사양:
  1. RTMPose Halpe26 전용 생체역학 피처 (74D):
     - 베이스라인 MediaPipe 호환 55D 도메인 피처 (ISB JCS, 힌지 구속, 골격 길이 보존)
     - RTMPose 고유 2D 정밀 굴곡각 및 대퇴 수평도, 정강이 수직도 (10D)
     - 단안 3D 역투영 복원(Monocular 3D Reconstruction via Bone Length Constancy) 무릎 굴곡각 (4D)
     - 공식 5대 시점 One-Hot 인코딩 피처 (5D)
  2. CatBoost + LightGBM + XGBoost 3-way 최적 앙상블
  3. Guo et al. (ICML 2017) 음의 로그 우도(NLL) 및 ECE 최소화 온도 스케일링(Temperature Scaling) 보정
  4. AGENTS.md 공식 5대 시점별 적응형 3-State 신뢰도 게이팅 (보류율 <= 10.0% 제약)
  5. 100% 무누수 순수 비전 RTMPose 2D 키포인트 기반 (AI-Hub 라벨 격리)
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

from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_feature_vector_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "cl1_rtmp_1_bundle.pkl")

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

CL1_RTMP_1_EXTRA_FEATURE_NAMES = [
    "rtm_knee_angle_2d_min",          # 1. 2D 최하사점 앞무릎 최소 굴곡 각도 (°)
    "rtm_knee_angle_2d_mean_bot",     # 2. 2D 최하사점 윈도우 평균 무릎 각도 (°)
    "rtm_knee_90_err_2d",             # 3. 2D |최소 굴곡각 - 90°| 절대 오차 (°)
    "rtm_knee_90_err_bot_2d",         # 4. 2D |최하사점 평균각 - 90°| 절대 오차 (°)
    "rtm_knee_rom_2d",                # 5. 2D 무릎 가동범위 (max - min) (°)
    "rtm_knee_angle_recon3d_min",     # 6. 단안 3D 역투영 복원 최하사점 무릎 각도 (°)
    "rtm_knee_angle_recon3d_mean_bot",# 7. 단안 3D 역투영 복원 최하사점 평균 각도 (°)
    "rtm_knee_90_err_recon3d",        # 8. 단안 3D 역투영 복원 |최소각 - 90°| 절대 오차 (°)
    "rtm_knee_90_err_bot_recon3d",    # 9. 단안 3D 역투영 복원 |평균각 - 90°| 절대 오차 (°)
    "rtm_femur_horiz_level_ratio",    # 10. 2D 대퇴골 지면 수평 레벨링 비율 (|y_hip - y_knee| / L_femur)
    "rtm_tibia_vert_align_ratio",     # 11. 2D 정강이뼈 수직 정렬 비율 (|x_knee - x_ank| / L_tibia)
    "rtm_hip_vertical_drop_ratio",    # 12. 기립 대비 골반 수직 하강비
    "rtm_trail_knee_angle_2d_min",    # 13. 2D 뒷다리 무릎 최저 굴곡 각도 (°)
    "rtm_is_left_lead",               # 14. 앞다리 좌우 여부 (좌측 앞다리: 1.0, 우측: 0.0)
    "view_is_view5",                  # 15. 시점 A (우측 후방 45°)
    "view_is_view2",                  # 16. 시점 B (우측 전방 45°)
    "view_is_view3",                  # 17. 시점 C (정면 0°)
    "view_is_view4",                  # 18. 시점 D (좌측 전방 45°)
    "view_is_view1"                   # 19. 시점 E (좌측 후방 45°)
]

CL1_RTMP_1_FEATURE_NAMES = SIDE_LUNGE_C1_FEATURE_NAMES_V11 + CL1_RTMP_1_EXTRA_FEATURE_NAMES


def halpe26_to_mp33_c1(kps_26: np.ndarray) -> np.ndarray:
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


def reconstruct_3d_vector_c1(v_2d_seq: np.ndarray, sign_z: float = 1.0) -> np.ndarray:
    """생체역학 강체 뼈 길이 불변 구속(Bone Length Constancy) 기반 단안 3D 역투영 복원"""
    l_2d = np.linalg.norm(v_2d_seq, axis=-1)
    L_true = np.percentile(l_2d, 95)
    z_mag = np.sqrt(np.clip(L_true**2 - l_2d**2, 0, None))
    z = z_mag * sign_z
    v_3d = np.column_stack([v_2d_seq, z])
    norms = np.linalg.norm(v_3d, axis=-1, keepdims=True) + 1e-6
    return v_3d / norms


def extract_cl1_rtmp_1_features(kps_26_64f: np.ndarray, view_code: str = 'view3') -> np.ndarray:
    """
    (64, 26, 3) RTMPose 텐서로부터 cl1_rtmp_1 전용 74차원 정예 피처 벡터 추출
    """
    assert kps_26_64f.ndim == 3 and kps_26_64f.shape[0] == 64, f"입력 규격 불일치: {kps_26_64f.shape}"
    kps_2d = kps_26_64f[:, :, :2]
    T = kps_2d.shape[0]

    # 1. 기존 MediaPipe 호환 55D 베이스라인 피처 추출
    mp33 = halpe26_to_mp33_c1(kps_26_64f)
    f_base_55d = extract_side_lunge_c1_feature_vector_v11(mp33)

    # 2. 앞다리(Lead Leg) 및 최하사점 식별
    l_knee_y = kps_2d[:, 13, 1]
    r_knee_y = kps_2d[:, 14, 1]
    if np.max(l_knee_y) >= np.max(r_knee_y):
        hip_l, knee_l, ank_l = 11, 13, 15
        hip_t, knee_t, ank_t = 12, 14, 16
        is_left_lead = True
        t_bot = int(np.argmax(l_knee_y))
    else:
        hip_l, knee_l, ank_l = 12, 14, 16
        hip_t, knee_t, ank_t = 11, 13, 15
        is_left_lead = False
        t_bot = int(np.argmax(r_knee_y))

    t_bot = max(4, min(T - 5, t_bot))
    win = slice(max(0, t_bot - 3), min(T, t_bot + 4))

    # 2D 앞다리 대퇴골/경골 벡터
    v_thigh_seq = kps_2d[:, knee_l] - kps_2d[:, hip_l]
    v_shank_seq = kps_2d[:, ank_l] - kps_2d[:, knee_l]
    norm_thigh = np.linalg.norm(v_thigh_seq, axis=-1) + 1e-6
    norm_shank = np.linalg.norm(v_shank_seq, axis=-1) + 1e-6

    # 2D 무릎 굴곡각
    dot_seq = np.sum(v_thigh_seq * v_shank_seq, axis=-1) / (norm_thigh * norm_shank)
    dot_seq = np.clip(dot_seq, -1.0, 1.0)
    angles_2d = np.degrees(np.arccos(dot_seq))

    min_ang_2d = float(np.min(angles_2d))
    mean_ang_bot_2d = float(np.mean(angles_2d[win]))
    err_90_2d = float(abs(min_ang_2d - 90.0))
    err_90_bot_2d = float(abs(mean_ang_bot_2d - 90.0))
    rom_ang_2d = float(np.max(angles_2d) - min_ang_2d)

    # 단안 3D 역투영 복원 굴곡각
    u3_thigh = reconstruct_3d_vector_c1(v_thigh_seq)
    u3_shank = reconstruct_3d_vector_c1(v_shank_seq)
    dot3_seq = np.sum(u3_thigh * u3_shank, axis=-1)
    dot3_seq = np.clip(dot3_seq, -1.0, 1.0)
    angles_3d = np.degrees(np.arccos(dot3_seq))

    min_ang_3d = float(np.min(angles_3d))
    mean_ang_bot_3d = float(np.mean(angles_3d[win]))
    err_90_3d = float(abs(min_ang_3d - 90.0))
    err_90_bot_3d = float(abs(mean_ang_bot_3d - 90.0))

    # 기하학적 정규화 비율
    k_bot = np.mean(kps_2d[win], axis=0)
    k_std = np.mean(kps_2d[:5], axis=0)
    femur_len = np.linalg.norm(k_bot[knee_l] - k_bot[hip_l]) + 1e-6
    tibia_len = np.linalg.norm(k_bot[ank_l] - k_bot[knee_l]) + 1e-6
    trunk_len = np.linalg.norm(k_bot[18] - k_bot[19]) + 1e-6

    femur_horiz_ratio = float(abs(k_bot[hip_l, 1] - k_bot[knee_l, 1]) / femur_len)
    tibia_vert_ratio = float(abs(k_bot[knee_l, 0] - k_bot[ank_l, 0]) / tibia_len)
    hip_drop = float((k_bot[hip_l, 1] - k_std[hip_l, 1]) / trunk_len)

    # 뒷다리 2D 무릎각
    v_thigh_t = kps_2d[:, knee_t] - kps_2d[:, hip_t]
    v_shank_t = kps_2d[:, ank_t] - kps_2d[:, knee_t]
    dot_t = np.sum(v_thigh_t * v_shank_t, axis=-1) / (np.linalg.norm(v_thigh_t, axis=-1)*np.linalg.norm(v_shank_t, axis=-1) + 1e-6)
    ang_t_2d = np.degrees(np.arccos(np.clip(dot_t, -1.0, 1.0)))
    min_ang_t_2d = float(np.min(ang_t_2d))

    v_code_clean = VIEW_NAME_TO_CODE.get(str(view_code).strip(), 'view3')
    v_vec = [
        1.0 if v_code_clean == 'view5' else 0.0,
        1.0 if v_code_clean == 'view2' else 0.0,
        1.0 if v_code_clean == 'view3' else 0.0,
        1.0 if v_code_clean == 'view4' else 0.0,
        1.0 if v_code_clean == 'view1' else 0.0
    ]

    extra_19d = [
        min_ang_2d, mean_ang_bot_2d, err_90_2d, err_90_bot_2d, rom_ang_2d,
        min_ang_3d, mean_ang_bot_3d, err_90_3d, err_90_bot_3d,
        femur_horiz_ratio, tibia_vert_ratio, hip_drop, min_ang_t_2d,
        float(is_left_lead)
    ] + v_vec

    f_total_74d = np.concatenate([f_base_55d, np.array(extra_19d, dtype=np.float32)])
    return np.nan_to_num(f_total_74d, nan=0.0)


class CL1RTMP1Classifier:
    """cl1_rtmp_1 공식 프로덕션 추론기"""
    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"cl1_rtmp_1 가중치 번들이 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = "cl1_rtmp_1"
        self.defect_name = "앞다리 무릎 각도 90도 오류 (RTMPose 74D 최적화 v1.0)"
        self.feature_names = self.bundle['feature_names']
        self.models_cb = self.bundle['models_cat']
        self.models_lgb = self.bundle['models_lgb']
        self.models_xgb = self.bundle['models_xgb']
        self.weights = self.bundle['ensemble_weights']
        self.temperature = float(self.bundle.get('temperature', 1.0))
        self.view_thresholds = self.bundle['view_thresholds']
        self.global_threshold = float(self.bundle.get('global_threshold', 0.50))

    def predict_features(self, feature_vector: np.ndarray, camera_view: str = '시점 C') -> Dict[str, Any]:
        """74D 피처 벡터 기반 추론"""
        X_in = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if X_in.shape[1] != 74:
            raise ValueError(f"입력 피처 차원이 74D여야 합니다. (현재: {X_in.shape[1]}D)")

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

        th_info = self.view_thresholds.get(v_code, {'tau_clean': 0.35, 'tau_defect': 0.65})
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
        feats = extract_cl1_rtmp_1_features(kps_26_64f, view_code=v_code)
        return self.predict_features(feats, camera_view=camera_view)


# 싱글톤 캐시
_CACHED_CL1_CLASSIFIER: Optional[CL1RTMP1Classifier] = None

def predict_cross_lunge_c1_rtmp_1(
    kps_or_features: np.ndarray,
    camera_view: str = '시점 C',
    bundle_path: Optional[str] = None
) -> Dict[str, Any]:
    global _CACHED_CL1_CLASSIFIER
    if _CACHED_CL1_CLASSIFIER is None or bundle_path is not None:
        _CACHED_CL1_CLASSIFIER = CL1RTMP1Classifier(bundle_path=bundle_path)

    arr = np.asarray(kps_or_features)
    if arr.ndim == 3:
        return _CACHED_CL1_CLASSIFIER.predict(arr, camera_view=camera_view)
    elif arr.ndim in (1, 2):
        vec = arr.flatten()
        return _CACHED_CL1_CLASSIFIER.predict_features(vec, camera_view=camera_view)
    else:
        raise ValueError(f"지원하지 않는 입력 배열 규격: {arr.shape}")
