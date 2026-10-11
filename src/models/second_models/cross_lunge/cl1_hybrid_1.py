# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 1 - 앞다리 무릎 각도 90도 오류 (Lead Knee 90 Degree Angle Defect)
Module: cl1_hybrid_1.py
Version: cl1_hybrid_1 (MediaPipe 3D + RTMPose 2D Hybrid 134D Production Classifier)

주요 사양:
  1. MediaPipe 3D + RTMPose 2D 상호보완 융합 134D 피처:
     - MediaPipe 33관절 3D 고유 시상면/ISB JCS/신체 로컬 직교 좌표계 불변량 (55D)
     - RTMPose Halpe26 2D 서브픽셀 정밀 굴곡각, 대퇴 수평도, 정강이 수직도, 단안 3D 역투영 (74D)
     - 3D-2D 투영 단축 및 사잇각 불일치 정량화 교차 상호작용 피처 (5D)
  2. CatBoost + LightGBM + XGBoost 3-way 앙상블 및 피험자 완전 격리(StratifiedGroupKFold) 학습
  3. Guo et al. (ICML 2017) 음의 로그 우도(NLL) 및 ECE 최소화 온도 스케일링(Temperature Scaling) 보정
  4. AGENTS.md 공식 5대 시점별 적응형 3-State 신뢰도 게이팅 (보류율 <= 10.0% 제약)
  5. 100% 무누수 순수 비전 추출 포인트 기반 (AI-Hub 라벨 격리)
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional, Union, List

import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_feature_vector_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)
from src.models.second_models.cross_lunge.cl1_rtmp_1 import (
    extract_cl1_rtmp_1_features,
    CL1_RTMP_1_FEATURE_NAMES
)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "cl1_hybrid_1_bundle.pkl")

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

CL1_CROSS_INTERACTION_FEATURE_NAMES = [
    "cross_diff_mp_3d_vs_rtm_2d_angle",   # 1. MP 3D 무릎각 vs RTM 2D 무릎각 절대 편차 (시상면 단축 왜곡 감지)
    "cross_diff_mp_3d_vs_rtm_2d_90_err",  # 2. MP 3D 90도 오차 vs RTM 2D 90도 오차 절대 편차
    "cross_diff_mp_isb_pure_vs_rtm_2d",   # 3. MP 순수 ISB 시상면 굴곡각 - RTM 2D 무릎각 사잇각차
    "cross_ratio_mp_to_rtm_flexion",      # 4. MP 3D 무릎각 / RTM 2D 무릎각 비율
    "cross_foreshortening_disparity",     # 5. MP 광축 투영 단축 계수 * RTM 대퇴골 수평 레벨링 비율
]

CL1_HYBRID_1_FEATURE_NAMES = (
    [f"mp_{name}" for name in SIDE_LUNGE_C1_FEATURE_NAMES_V11] +
    [f"rtm_{name}" for name in CL1_RTMP_1_FEATURE_NAMES] +
    CL1_CROSS_INTERACTION_FEATURE_NAMES
)

assert len(CL1_HYBRID_1_FEATURE_NAMES) == 134, f"피처 차원 수 불일치: {len(CL1_HYBRID_1_FEATURE_NAMES)}"


def compute_cross_interaction_features(f_mp_55d: np.ndarray, f_rtm_74d: np.ndarray) -> np.ndarray:
    """MediaPipe 3D 피처와 RTMPose 2D 피처 간 교차 상호작용 피처(5D) 산출"""
    # MP: idx 0(lead_knee_angle_min), idx 2(lead_knee_90_error_abs), idx 32(optical_axis_foreshortening_factor), idx 51(isb_jcs_pure_flexion_min)
    # RTM: idx 55(rtm_knee_angle_2d_min), idx 57(rtm_knee_90_err_2d), idx 64(rtm_femur_horiz_level_ratio)
    mp_knee_ang = f_mp_55d[0]
    mp_90_err = f_mp_55d[2]
    mp_opt_factor = f_mp_55d[32]
    mp_isb_flex = f_mp_55d[51]

    rtm_knee_ang = f_rtm_74d[55]
    rtm_90_err = f_rtm_74d[57]
    rtm_femur_horiz = f_rtm_74d[64]

    diff_knee_ang = float(abs(mp_knee_ang - rtm_knee_ang))
    diff_90_err_val = float(abs(mp_90_err - rtm_90_err))
    diff_isb_val = float(mp_isb_flex - rtm_knee_ang)
    ratio_ang = float(mp_knee_ang / (rtm_knee_ang + 1e-4))
    disp_factor = float(mp_opt_factor * rtm_femur_horiz)

    return np.array([diff_knee_ang, diff_90_err_val, diff_isb_val, ratio_ang, disp_factor], dtype=np.float32)


def extract_cl1_hybrid_1_features(
    kps_26_64f: np.ndarray,
    kps_33_3d_64f: np.ndarray,
    view_code: str = 'view3',
    vis_64f: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    (64, 26, 3) RTMPose 텐서와 (64, 33, 3) MediaPipe 3D 텐서로부터
    cl1_hybrid_1 전용 134차원 정예 하이브리드 피처 벡터 추출
    """
    assert kps_26_64f.shape[0] == 64, f"RTMPose 프레임 수 불일치: {kps_26_64f.shape}"
    assert kps_33_3d_64f.shape[0] == 64, f"MediaPipe 프레임 수 불일치: {kps_33_3d_64f.shape}"

    # 1. MediaPipe 55D 추출
    f_mp_55d = extract_side_lunge_c1_feature_vector_v11(kps_33_3d_64f, vis_64f=vis_64f)

    # 2. RTMPose 74D 추출
    f_rtm_74d = extract_cl1_rtmp_1_features(kps_26_64f, view_code=view_code)

    # 3. 교차 상호작용 피처 (5D)
    f_cross_5d = compute_cross_interaction_features(f_mp_55d, f_rtm_74d)

    # 4. 결합
    f_total_134d = np.concatenate([f_mp_55d, f_rtm_74d, f_cross_5d])
    return np.nan_to_num(f_total_134d, nan=0.0)


class CL1Hybrid1Classifier:
    """cl1_hybrid_1 공식 프로덕션 추론기"""
    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            bundle_path = DEFAULT_BUNDLE_PATH
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"cl1_hybrid_1 가중치 번들이 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.model_name = "cl1_hybrid_1"
        self.defect_name = "앞다리 무릎 각도 90도 오류 (MediaPipe 3D + RTMPose 2D 하이브리드 최적화 v1.0)"
        self.feature_names = self.bundle['feature_names']
        self.models_cb = self.bundle['models_cat']
        self.models_lgb = self.bundle['models_lgb']
        self.models_xgb = self.bundle['models_xgb']
        self.weights = self.bundle['ensemble_weights']
        self.temperature = float(self.bundle.get('temperature', 1.0))
        self.view_thresholds = self.bundle['view_thresholds']
        self.global_threshold = float(self.bundle.get('global_threshold', 0.50))

    def predict_features(self, feature_vector: np.ndarray, camera_view: str = '시점 C') -> Dict[str, Any]:
        """134D 피처 벡터 기반 추론"""
        X_in = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if X_in.shape[1] != 134:
            raise ValueError(f"입력 피처 차원이 134D여야 합니다. (현재: {X_in.shape[1]}D)")

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

    def predict(
        self,
        kps_26_64f: np.ndarray,
        kps_33_3d_64f: np.ndarray,
        camera_view: str = '시점 C',
        vis_64f: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """(64, 26, 3) RTMPose 및 (64, 33, 3) MediaPipe 텐서 기반 엔드투엔드 추론"""
        v_code = VIEW_NAME_TO_CODE.get(str(camera_view).strip(), 'view3')
        feats = extract_cl1_hybrid_1_features(
            kps_26_64f=kps_26_64f,
            kps_33_3d_64f=kps_33_3d_64f,
            view_code=v_code,
            vis_64f=vis_64f
        )
        return self.predict_features(feats, camera_view=camera_view)


# 싱글톤 캐시
_CACHED_CL1_HYBRID_CLASSIFIER: Optional[CL1Hybrid1Classifier] = None

def predict_cross_lunge_c1_hybrid_1(
    kps_26_or_features: np.ndarray,
    kps_33_3d_64f: Optional[np.ndarray] = None,
    camera_view: str = '시점 C',
    bundle_path: Optional[str] = None,
    vis_64f: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """공식 탑레벨 추론 함수"""
    global _CACHED_CL1_HYBRID_CLASSIFIER
    if _CACHED_CL1_HYBRID_CLASSIFIER is None or bundle_path is not None:
        _CACHED_CL1_HYBRID_CLASSIFIER = CL1Hybrid1Classifier(bundle_path=bundle_path)

    arr = np.asarray(kps_26_or_features)
    if arr.ndim == 3 and kps_33_3d_64f is not None:
        return _CACHED_CL1_HYBRID_CLASSIFIER.predict(
            kps_26_64f=arr,
            kps_33_3d_64f=kps_33_3d_64f,
            camera_view=camera_view,
            vis_64f=vis_64f
        )
    elif arr.ndim in (1, 2):
        vec = arr.flatten()
        return _CACHED_CL1_HYBRID_CLASSIFIER.predict_features(vec, camera_view=camera_view)
    else:
        raise ValueError("유효하지 않은 입력 인자: (64, 26, 3) 및 (64, 33, 3) 텐서 쌍 또는 134D 피처 벡터가 필요합니다.")
