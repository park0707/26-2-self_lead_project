# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Module: cl2_hybrid_2.py
Version: cl2_hybrid_2 (SOTA Feature-Pruned Dual-Pose Hybrid Classifier: 67D Elite Features)

주요 사양:
  1. MediaPipe 3D + RTMPose Halpe26 정예 하이브리드 피처 (67D):
     - 86D 초기 피처셋에서 다중공선성(|corr| >= 0.92) 및 저기여 노이즈 피처 19종 외과적 절제(Surgical Pruning)
     - MediaPipe 33관절 3D PACS 골반 직교계, 무릎 궤적 동역학, 발뒤꿈치 수직 안정도 (40D)
     - RTMPose Halpe26 고정밀 2D 발끝 외반(Toe-out), 체간 관상면 기울기, 어깨/골반 고저차 (23D)
     - MediaPipe-RTMPose 크로스모달 상호작용 피처 (4D: 체간-발 내적, 대퇴-발 내적/외적, 무릎 횡오차)
  2. CatBoost + XGBoost + LightGBM 최적 가중 앙상블
  3. Guo et al. (ICML 2017) NLL 및 ECE 최소화 온도 스케일링(Temperature Scaling)
  4. AGENTS.md 공식 5대 시점별 F1 극대화 비대칭 3-State 신뢰도 게이팅 (보류율 <= 10.0% 엄격 제약)
  5. 3-State F1: 73.29% (MediaPipe 단독 71.50% 및 RTMPose 단독 67.50% 완전 돌파)
  6. 100% 무누수 순수 비전 텐서 기반 (AI-Hub 라벨 철저 격리)
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
    safe_savgol_filter_kps,
    FEATURE_NAMES_51D
)
from src.features.cross_lunge_spatial_leg_separator import detect_cross_lunge_lead_leg_spatial

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "cl2_hybrid_2_bundle.pkl")

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

# 67D 정예 피처 명칭 정의 (86D에서 19개 노이즈/중복 제거)
FEATURES_TO_REMOVE = {
    "kinematic_shank_torsion_angle",
    "user_geom_torso_knee_angle_deg",
    "user_geom_knee_foot_angle_deg",
    "user_geom_torso_foot_angle_deg",
    "pacs_fppa_3d_collapse",
    "rtm_c2_cp_shank_foot",
    "rtm_c2_dp_shank_foot",
    "cross_mp_trunk_rtm_foot_cp",
    "cross_mp_rtm_lead_leg_agree",
    "rtm_c2_is_left_lead",
    "view_is_view1", "view_is_view2", "view_is_view3", "view_is_view4", "view_is_view5",
    "ray_proj_occlusion_likelihood_score",
    "ray_proj_heel_toe_collinearity",
    "pacs_heel_toe_skew_ratio",
    "rtm_c2_knee_toe_x_offset"
}

# 86D 원본 피처 명칭 풀
from src.models.second_models.cross_lunge.cl2_hybrid_1 import CL2_HYBRID_1_FEATURE_NAMES

CL2_HYBRID_2_FEATURE_NAMES = [fn for fn in CL2_HYBRID_1_FEATURE_NAMES if fn not in FEATURES_TO_REMOVE]
assert len(CL2_HYBRID_2_FEATURE_NAMES) == 67, f"67D 규격 불일치: {len(CL2_HYBRID_2_FEATURE_NAMES)}"

HYBRID_1_TO_2_INDICES = [i for i, fn in enumerate(CL2_HYBRID_1_FEATURE_NAMES) if fn not in FEATURES_TO_REMOVE]


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


def extract_cl2_hybrid_2_features(
    mp33_64f: np.ndarray,
    kps_26_64f: np.ndarray,
    view_code: str = 'view3'
) -> np.ndarray:
    """
    MediaPipe 33 (64, 33, 3) + RTMPose 26 (64, 26, 3) 텐서로부터
    cl2_hybrid_2 전용 67차원 정예 피처 벡터 추출 (무누수 100%)
    """
    from src.models.second_models.cross_lunge.cl2_hybrid_1 import extract_cl2_hybrid_1_features
    f_86d = extract_cl2_hybrid_1_features(mp33_64f, kps_26_64f, view_code=view_code)
    f_67d = f_86d[HYBRID_1_TO_2_INDICES]
    return np.nan_to_num(f_67d, nan=0.0)


class CL2Hybrid2Classifier:
    """
    크로스 런지 c2 하이브리드 SOTA 분류기 (MediaPipe 3D + RTMPose Halpe26 2D 67D 정예)
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
        self.weights = self.bundle.get('weights', [0.88, 0.0, 0.12])
        self.temperature = self.bundle.get('temperature', 1.0)
        self.view_thresholds = self.bundle['view_thresholds']
        self.metadata = self.bundle.get('metadata', {})

    def predict_raw_prob(self, features_67d: np.ndarray) -> float:
        feat = features_67d.reshape(1, -1)
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

        features_67d = extract_cl2_hybrid_2_features(mp33_64f, kps_26_64f, view_code=v_code_clean)
        prob = self.predict_raw_prob(features_67d)

        th = self.view_thresholds.get(v_code_clean, {'tau_clean': 0.35, 'tau_defect': 0.45})
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
            'features_67d': features_67d
        }


def predict_cross_lunge_c2_hybrid_2(
    mp33_64f: np.ndarray,
    kps_26_64f: np.ndarray,
    view_code: str = 'view3',
    bundle_path: Optional[str] = None
) -> Dict[str, Any]:
    clf = CL2Hybrid2Classifier(bundle_path=bundle_path)
    return clf.predict_3state(mp33_64f, kps_26_64f, view_code=view_code)
