# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 3 - 상체 정면 균형잡기 오류 (Trunk Frontal Balance & Stability Defect)
Module: cl3_rtmp_1.py
Version: cl3_rtmp_1 (RTMPose-m Halpe26 Optimized Production Classifier)

주요 사양:
  1. RTMPose Halpe26 전용 생체역학 피처 (46D):
     - 베이스라인 MediaPipe 호환 22D 정예 도메인 피처 (상체 시상면 전후각, 경추 신전)
     - RTMPose 고유 2D 체간 시상면/관상면 경사각 및 동요도 5D
     - 두부-경추 수직 앙각(하늘 보기, Key 183) 및 좌향 시선 회전(왼쪽 보기, Key 184) 6D
     - 양 어깨/골반 관상면 기울기 및 체간 횡방향 이탈 거리 6D (몸통 옆으로 구부리기, Key 184)
     - 기립 대비 최하사점 상대 변화량 2D
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

from src.features.cross_lunge_c3_features import (
    extract_cross_lunge_c3_features_22d,
    FEATURE_NAMES_22D
)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "cl3_rtmp_1_bundle.pkl")

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

CL3_RTMP_1_EXTRA_FEATURE_NAMES = [
    "rtm_c3_tilt_bot",                # 1. 2D 최하사점 상체 수직 기울기 각도 (°)
    "rtm_c3_tilt_min",                # 2. 2D 전 시퀀스 상체 최소 기울기 각도 (후방 젖힘) (°)
    "rtm_c3_tilt_max",                # 3. 2D 전 시퀀스 상체 최대 기울기 각도 (°)
    "rtm_c3_tilt_rom",                # 4. 2D 상체 기울기 가동범위 (max - min) (°)
    "rtm_c3_tilt_std_bot",            # 5. 2D 최하사점 윈도우 상체 동요도 (표준편차)
    "rtm_c3_head_pitch_bot",          # 6. 2D 최하사점 머리 수직 앙각 (하늘 보기) (°)
    "rtm_c3_head_pitch_max",          # 7. 2D 전 시퀀스 머리 최대 앙각 (°)
    "rtm_c3_head_pitch_min",          # 8. 2D 전 시퀀스 머리 최소 앙각 (°)
    "rtm_c3_head_pitch_rom",          # 9. 2D 머리 앙각 가동범위 (°)
    "rtm_c3_ear_asym_bot",            # 10. 2D 최하사점 귀-코 수평 비대칭도 (왼쪽 보기)
    "rtm_c3_ear_asym_max",            # 11. 2D 전 시퀀스 귀-코 수평 최대 비대칭도
    "rtm_c3_sh_tilt_bot",             # 12. 2D 최하사점 어깨 좌우 기울기 비율 (몸통 옆 굽힘)
    "rtm_c3_sh_tilt_max",             # 13. 2D 전 시퀀스 어깨 좌우 최대 기울기
    "rtm_c3_hip_tilt_bot",            # 14. 2D 최하사점 골반 좌우 기울기 비율
    "rtm_c3_hip_tilt_max",            # 15. 2D 전 시퀀스 골반 좌우 최대 기울기
    "rtm_c3_lat_offset_bot",          # 16. 2D 최하사점 어깨-골반 횡방향 이탈 비율
    "rtm_c3_lat_offset_max",          # 17. 2D 전 시퀀스 어깨-골반 횡방향 최대 이탈 비율
    "rtm_c3_delta_tilt",              # 18. 기립 대비 최하사점 상체 기울기 변화량 (°)
    "rtm_c3_delta_head_pitch",        # 19. 기립 대비 최하사점 머리 앙각 변화량 (°)
    "view_is_view5",                  # 20. 시점 A (우측 후방 45°)
    "view_is_view2",                  # 21. 시점 B (우측 전방 45°)
    "view_is_view3",                  # 22. 시점 C (정면 0°)
    "view_is_view4",                  # 23. 시점 D (좌측 전방 45°)
    "view_is_view1"                   # 24. 시점 E (좌측 후방 45°)
]

CL3_RTMP_1_FEATURE_NAMES = FEATURE_NAMES_22D + CL3_RTMP_1_EXTRA_FEATURE_NAMES


def halpe26_to_mp33_c3(kps_26: np.ndarray) -> np.ndarray:
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


def extract_cl3_rtmp_1_features(kps_26_64f: np.ndarray, view_code: str = 'view3') -> np.ndarray:
    """
    (64, 26, 3) RTMPose 텐서로부터 cl3_rtmp_1 전용 46차원 정예 피처 벡터 추출
    """
    assert kps_26_64f.ndim == 3 and kps_26_64f.shape[0] == 64, f"입력 규격 불일치: {kps_26_64f.shape}"
    kps_2d = kps_26_64f[:, :, :2]
    conf = kps_26_64f[:, :, 2]
    T = kps_2d.shape[0]

    # 1. MediaPipe 호환 22D 베이스라인 피처 추출
    mp33 = halpe26_to_mp33_c3(kps_26_64f)
    f_base_22d = extract_cross_lunge_c3_features_22d(mp33)

    # 2. RTMPose 고유 24D 피처 추출
    l_knee_y = kps_2d[:, 13, 1]
    r_knee_y = kps_2d[:, 14, 1]
    t_bot = int(np.argmax(np.maximum(l_knee_y, r_knee_y)))
    t_bot = max(4, min(T - 5, t_bot))
    win = slice(max(0, t_bot - 3), min(T, t_bot + 4))

    pelvis_seq = kps_2d[:, 19] if np.mean(conf[:, 19]) > 0.3 else (kps_2d[:, 11] + kps_2d[:, 12]) * 0.5
    neck_seq = kps_2d[:, 18] if np.mean(conf[:, 18]) > 0.3 else (kps_2d[:, 5] + kps_2d[:, 6]) * 0.5
    v_trunk_seq = neck_seq - pelvis_seq
    trunk_len = np.mean(np.linalg.norm(v_trunk_seq, axis=-1)) + 1e-6

    # 상체 전후/좌우 기울기 각도
    trunk_tilt_rad = np.arctan2(v_trunk_seq[:, 0], -v_trunk_seq[:, 1])
    trunk_tilt_deg = np.degrees(trunk_tilt_rad)

    tilt_bot = float(np.mean(trunk_tilt_deg[win]))
    tilt_min = float(np.min(trunk_tilt_deg))
    tilt_max = float(np.max(trunk_tilt_deg))
    tilt_rom = float(tilt_max - tilt_min)
    tilt_std = float(np.std(trunk_tilt_deg[win]))

    # 두부 앙각 (하늘 보기)
    v_head_seq = kps_2d[:, 0] - neck_seq
    head_pitch_rad = np.arctan2(v_head_seq[:, 0], -v_head_seq[:, 1])
    head_pitch_deg = np.degrees(head_pitch_rad)
    head_pitch_bot = float(np.mean(head_pitch_deg[win]))
    head_pitch_max = float(np.max(head_pitch_deg))
    head_pitch_min = float(np.min(head_pitch_deg))
    head_pitch_rom = float(head_pitch_max - head_pitch_min)

    # 시선 좌우 회전 (왼쪽 보기)
    ear_asym_seq = (kps_2d[:, 0, 0] - kps_2d[:, 3, 0]) - (kps_2d[:, 4, 0] - kps_2d[:, 0, 0])
    ear_asym_norm = ear_asym_seq / trunk_len
    ear_asym_bot = float(np.mean(ear_asym_norm[win]))
    ear_asym_max = float(np.max(abs(ear_asym_norm)))

    # 어깨 및 골반 관상면 기울기
    sh_tilt_seq = (kps_2d[:, 5, 1] - kps_2d[:, 6, 1]) / (np.linalg.norm(kps_2d[:, 5] - kps_2d[:, 6], axis=-1) + 1e-6)
    hip_tilt_seq = (kps_2d[:, 11, 1] - kps_2d[:, 12, 1]) / (np.linalg.norm(kps_2d[:, 11] - kps_2d[:, 12], axis=-1) + 1e-6)

    sh_tilt_bot = float(np.mean(sh_tilt_seq[win]))
    sh_tilt_max = float(np.max(abs(sh_tilt_seq)))
    hip_tilt_bot = float(np.mean(hip_tilt_seq[win]))
    hip_tilt_max = float(np.max(abs(hip_tilt_seq)))

    # 체간 횡방향 이탈
    mid_sh = (kps_2d[:, 5] + kps_2d[:, 6]) * 0.5
    mid_hip = (kps_2d[:, 11] + kps_2d[:, 12]) * 0.5
    lateral_offset = (mid_sh[:, 0] - mid_hip[:, 0]) / trunk_len
    lat_offset_bot = float(np.mean(lateral_offset[win]))
    lat_offset_max = float(np.max(abs(lateral_offset)))

    # 기립 대비 최하사점 변화량
    tilt_stand = float(np.mean(trunk_tilt_deg[:5]))
    delta_tilt = float(tilt_bot - tilt_stand)
    head_stand = float(np.mean(head_pitch_deg[:5]))
    delta_head = float(head_pitch_bot - head_stand)

    v_code_clean = VIEW_NAME_TO_CODE.get(str(view_code).strip(), 'view3')
    v_vec = [
        1.0 if v_code_clean == 'view5' else 0.0,
        1.0 if v_code_clean == 'view2' else 0.0,
        1.0 if v_code_clean == 'view3' else 0.0,
        1.0 if v_code_clean == 'view4' else 0.0,
        1.0 if v_code_clean == 'view1' else 0.0
    ]

    extra_24d = [
        tilt_bot, tilt_min, tilt_max, tilt_rom, tilt_std,
        head_pitch_bot, head_pitch_max, head_pitch_min, head_pitch_rom,
        ear_asym_bot, ear_asym_max,
        sh_tilt_bot, sh_tilt_max, hip_tilt_bot, hip_tilt_max,
        lat_offset_bot, lat_offset_max,
        delta_tilt, delta_head
    ] + v_vec

    f_total_46d = np.concatenate([f_base_22d, np.array(extra_24d, dtype=np.float32)])
    return np.nan_to_num(f_total_46d, nan=0.0)


class CL3RTMP1Classifier:
    """cl3_rtmp_1 공식 프로덕션 추론기"""
    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"cl3_rtmp_1 가중치 번들이 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = "cl3_rtmp_1"
        self.defect_name = "상체 정면 균형잡기 오류 (RTMPose 46D 최적화 v1.0)"
        self.feature_names = self.bundle['feature_names']
        self.models_cb = self.bundle['models_cat']
        self.models_lgb = self.bundle['models_lgb']
        self.models_xgb = self.bundle['models_xgb']
        self.weights = self.bundle['ensemble_weights']
        self.temperature = float(self.bundle.get('temperature', 1.0))
        self.view_thresholds = self.bundle['view_thresholds']
        self.global_threshold = float(self.bundle.get('global_threshold', 0.50))

    def predict_features(self, feature_vector: np.ndarray, camera_view: str = '시점 C') -> Dict[str, Any]:
        """46D 피처 벡터 기반 추론"""
        X_in = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if X_in.shape[1] != 46:
            raise ValueError(f"입력 피처 차원이 46D여야 합니다. (현재: {X_in.shape[1]}D)")

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

        th_info = self.view_thresholds.get(v_code, {'tau_clean': 0.35, 'tau_defect': 0.55})
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
        feats = extract_cl3_rtmp_1_features(kps_26_64f, view_code=v_code)
        return self.predict_features(feats, camera_view=camera_view)


# 싱글톤 캐시
_CACHED_CL3_CLASSIFIER: Optional[CL3RTMP1Classifier] = None

def predict_cross_lunge_c3_rtmp_1(
    kps_or_features: np.ndarray,
    camera_view: str = '시점 C',
    bundle_path: Optional[str] = None
) -> Dict[str, Any]:
    global _CACHED_CL3_CLASSIFIER
    if _CACHED_CL3_CLASSIFIER is None or bundle_path is not None:
        _CACHED_CL3_CLASSIFIER = CL3RTMP1Classifier(bundle_path=bundle_path)

    arr = np.asarray(kps_or_features)
    if arr.ndim == 3:
        return _CACHED_CL3_CLASSIFIER.predict(arr, camera_view=camera_view)
    elif arr.ndim in (1, 2):
        vec = arr.flatten()
        return _CACHED_CL3_CLASSIFIER.predict_features(vec, camera_view=camera_view)
    else:
        raise ValueError(f"지원하지 않는 입력 배열 규격: {arr.shape}")
