# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Defect: Condition 5 - 상체 과도한 숙임/젖힘 (Torso Sagittal Neutrality)
Module: bl5_hybrid_1.py
Model: bl5_hybrid_1 (MediaPipe 3D 36D + RTMPose 2D 37D + Cross-Modal 6D = 79D Fusion)
Status: [PRODUCTION CONFIRMED] 사용자 승인 최종 모델 확정 (2026-10-11)
Performance:
  - 3-State F1-Score: 75.86% (보류율 9.62% <= 10.0% 준수)
  - ROC-AUC: 0.8555 (역대 최고치 경신)
  - 선택적 확정 정확도 (Accuracy): 80.66%
  - 선택적 결함 검출 재현율 (Recall): 82.50%
  - 선택적 정밀도 (Precision): 70.21%
"""

import os
import sys
if "D:/" not in sys.path:
    sys.path.insert(0, "D:/")
import pickle
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.features.lunge_c5_features_hybrid import (
    extract_lunge_c5_hybrid_features,
    compute_cross_interactions,
    LUNGE_C5_HYBRID_79D_NAMES
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "bl5_hybrid_1_bundle.pkl")

# AGENTS.md 공식 5대 시점 명칭 매핑
VIEW_CANONICAL_MAP = {
    'view1': 'view1',
    'view2': 'view2',
    'view3': 'view3',
    'view4': 'view4',
    'view5': 'view5',
    '시점 A': 'view5',
    '시점 B': 'view2',
    '시점 C': 'view3',
    '시점 D': 'view4',
    '시점 E': 'view1',
    'A': 'view5',
    'B': 'view2',
    'C': 'view3',
    'D': 'view4',
    'E': 'view1',
}

VIEW_DISPLAY_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}

VIEW_ALIAS_MAP = VIEW_DISPLAY_MAP

VIEW_THRESHOLDS = {
    'view5': (0.200, 0.290), # 시점 A (우측 후방 45°)
    'view2': (0.270, 0.490), # 시점 B (우측 전방 45°)
    'view3': (0.180, 0.225), # 시점 C (정면 0°)
    'view4': (0.245, 0.370), # 시점 D (좌측 전방 45°)
    'view1': (0.475, 0.670), # 시점 E (좌측 후방 45°)
}


class BabelLungeCondition5ModelHybrid:
    """
    바벨 런지 c5(상체 과도한 숙임/젖힘) 공식 프로덕션 하이브리드 추론 엔진
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        if not os.path.exists(self.bundle_path):
            raise FileNotFoundError(f"bl5_hybrid_1 번들 파일이 존재하지 않습니다: {self.bundle_path}")

        with open(self.bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.model_name = bundle.get('model_name', 'bl5_hybrid_1')
        self.feature_names = bundle.get('feature_names', LUNGE_C5_HYBRID_79D_NAMES)
        self.weights = bundle.get('ensemble_weights', [0.5, 0.3, 0.2])
        self.model_cat = bundle['model_cat']
        self.model_lgb = bundle['model_lgb']
        self.model_xgb = bundle['model_xgb']
        self.tau_clean = float(bundle.get('tau_clean', 0.230))
        self.tau_defect = float(bundle.get('tau_defect', 0.340))
        self.view_thresholds = bundle.get('view_thresholds', VIEW_THRESHOLDS)
        self.view_audit = bundle.get('view_audit', {})
        self.overall_performance = bundle.get('overall_performance', {})
        self.is_loaded = True

    @classmethod
    def extract_features(
        cls,
        mp_kps_33_3d: np.ndarray,
        rtmp_kps_26_2d: np.ndarray
    ) -> np.ndarray:
        """
        MediaPipe 3D 및 RTMPose 2D 64프레임 시퀀스로부터 79차원 하이브리드 피처 벡터 추출
        """
        return extract_lunge_c5_hybrid_features(mp_kps_33_3d, rtmp_kps_26_2d)

    def predict_proba(self, feat_vec: np.ndarray) -> float:
        """
        79차원 피처 벡터로부터 3-Way 앙상블 결함 확률 추론
        """
        if feat_vec.ndim == 1:
            feat_vec = feat_vec.reshape(1, -1)

        p_cb = float(self.model_cat.predict_proba(feat_vec)[0, 1])
        p_lgb = float(self.model_lgb.predict_proba(feat_vec)[0, 1])
        p_xgb = float(self.model_xgb.predict_proba(feat_vec)[0, 1])

        p_ens = (self.weights[0] * p_cb +
                 self.weights[1] * p_lgb +
                 self.weights[2] * p_xgb)
        return float(np.clip(p_ens, 0.0, 1.0))

    def predict_sequence(
        self,
        mp_kps_33_3d: np.ndarray,
        rtmp_kps_26_2d: np.ndarray,
        view: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스에 대한 c5 세부 결함 추론
        - Input:
            * mp_kps_33_3d: (64, 33, 3) MediaPipe 3D 키포인트 배열
            * rtmp_kps_26_2d: (64, 26, 2) 또는 (64, 26, 3) RTMPose 키포인트 배열
            * view: 카메라 시점 코드 ('view1'~'view5' 또는 '시점 A'~'시점 E')
        - Output:
            * status: 'Normal' (정상), 'Defect' (결함), 'Caution_Retake' (판단 보류)
            * state_code: 0 (정상), 1 (결함), 2 (보류)
            * defect_prob: float (0.0 ~ 1.0)
            * confidence: float
            * feedback: 상세 생체역학 코칭 피드백 메시지
        """
        feats = self.extract_features(mp_kps_33_3d, rtmp_kps_26_2d)
        prob = self.predict_proba(feats)

        # 시점별 최적 적응형 임계값 결정
        v_canon = VIEW_CANONICAL_MAP.get(str(view), None) if view else None
        if v_canon in self.view_thresholds:
            t_clean, t_defect = self.view_thresholds[v_canon]
        else:
            t_clean, t_defect = self.tau_clean, self.tau_defect

        if prob >= t_defect:
            state = "Defect"
            state_code = 1
            confidence = float((prob - t_defect) / (1.0 - t_defect + 1e-6))
            is_defect = True
        elif prob < t_clean:
            state = "Normal"
            state_code = 0
            confidence = float((t_clean - prob) / (t_clean + 1e-6))
            is_defect = False
        else:
            state = "Caution_Retake"
            state_code = 2
            confidence = 0.5
            is_defect = bool(prob >= 0.5)

        # 주요 생체역학 지표 추출 (MediaPipe 36D 중 주요 피처)
        pitch_bottom = float(feats[0])     # MP_lunge_vspa_torso_pitch_bottom
        pitch_peak = float(feats[1])       # MP_lunge_vspa_torso_pitch_peak
        pitch_min = float(feats[2])        # MP_lunge_vspa_torso_pitch_min
        tibia_parallel = float(feats[7])   # MP_lunge_tibia_torso_parallelism
        trunk_comp = float(feats[44])      # RTMP_trunk_compression (36 + 8)

        defect_types = []
        feedback_msgs = []

        if is_defect or state == "Caution_Retake":
            # 1. 전방 과숙임 (Excessive Trunk Flexion)
            if pitch_peak > 25.0 or pitch_bottom > 22.0 or trunk_comp < 0.85:
                defect_types.append("상체 과도한 전방 숙임 (Excessive Trunk Flexion)")
                feedback_msgs.append(
                    f"런지 동작 중 상체가 전방으로 {pitch_peak:.1f}° 과도하게 숙여져 요추 및 대퇴사두근에 불균형한 전단력이 가해집니다. "
                    f"가슴을 열고 시선을 정면 수평으로 유지하며 상체를 정강이 각도({tibia_parallel:.1f}° 오차)와 평행하게 세우세요."
                )
            # 2. 후방 과젖힘 (Trunk Hyperextension)
            if pitch_min < -5.0:
                defect_types.append("상체 과도한 후방 젖힘 (Trunk Hyperextension)")
                feedback_msgs.append(
                    f"상체가 뒤로 {abs(pitch_min):.1f}° 과도하게 젖혀져 요추 후관절에 과도한 압축 스트레스가 발생하고 있습니다. "
                    f"복압을 단단히 유지하고 흉곽을 아래로 당겨 중립 척추(Neutral Spine)를 유지하세요."
                )

        if not defect_types:
            feedback_msgs.append("상체의 전후 중립 척추 정렬이 매우 이상적으로 유지되고 있습니다.")

        return {
            "status": state,
            "state_code": state_code,
            "is_defect": is_defect,
            "defect_prob": prob,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "applied_view": VIEW_DISPLAY_MAP.get(v_canon, "global") if v_canon else "global",
            "thresholds": {"tau_clean": t_clean, "tau_defect": t_defect},
            "defect_types": defect_types,
            "feedback": " ".join(feedback_msgs),
            "key_metrics": {
                "torso_pitch_bottom_deg": pitch_bottom,
                "torso_pitch_peak_deg": pitch_peak,
                "torso_pitch_min_deg": pitch_min,
                "tibia_torso_parallelism_err": tibia_parallel,
                "rtmp_trunk_compression": trunk_comp
            }
        }


# 하위 호환 별칭
BL5Hybrid1Classifier = BabelLungeCondition5ModelHybrid
