# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드, 06_스텝_백워드)
Defect: Condition 5 - 상체의 과도한 숙임/젖힘 여부 (Torso Sagittal Neutrality)
Module: dl5_rtmp_1.py
Model: dl5_rtmp_1 (MediaPipe 3D 36D + RTMPose 2D 37D + Cross-Modal 6D = 79D Hybrid SOTA)
Purpose: bl5_final 아키텍처를 계승하여 다이나믹 런지 운동학에 최적화된 공식 c5 프로덕션 추론 엔진
Status: [PRODUCTION CONFIRMED] 사용자 승인 최종 모델 확정 (2026-10-11)
Performance (5-Fold StratifiedGroupKFold 피험자 1~9 전원 격리 전수 실측):
  - 3-State 적응형 F1 (보류율 9.13% <= 10.0%): 77.41% (바벨런지 제로샷 69.82% 대비 +7.59%p 대폭 향상)
  - 3-State 적응형 F1 (보류율 13.96% <= 15.0%): 78.36%
  - ROC-AUC: 0.8565 (역대 최고치)
  - 결함 검출 재현율 (Recall): 86.65%
  - 선택적 확정 정확도 (Accuracy): 79.84%
  - 선택적 정밀도 (Precision): 69.95%
  - 공식 5대 시점(A~E)별 3-State F1:
    * 시점 A (우측 후방 45°, view5): 83.48% (AUC: 0.8959)
    * 시점 B (우측 전방 45°, view2): 81.58% (AUC: 0.8893)
    * 시점 C (정면 0°, view3)     : 65.96% (AUC: 0.7211, 제로샷 58.26% 대비 +7.70%p)
    * 시점 D (좌측 전방 45°, view4): 73.87% (AUC: 0.8359)
    * 시점 E (좌측 후방 45°, view1): 84.96% (AUC: 0.9217)
  - 세부 운동 유형별 3-State F1:
    * 05_스텝_포워드 다이나믹 런지 (875건): 71.15% (AUC: 0.8311)
    * 06_스텝_백워드 다이나믹 런지 (593건): 83.10% (AUC: 0.8964)
"""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pickle
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.features.lunge_c5_features_hybrid import (
    extract_lunge_c5_hybrid_features,
    compute_cross_interactions,
    LUNGE_C5_HYBRID_79D_NAMES
)

DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "dl5_rtmp_1_bundle.pkl")

# AGENTS.md 공식 5대 시점 명칭 매핑 규격
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

VIEW_THRESHOLDS_10 = {
    'view5': (0.2066, 0.3236), # 시점 A (우측 후방 45°)
    'view2': (0.2847, 0.4734), # 시점 B (우측 전방 45°)
    'view3': (0.1394, 0.1752), # 시점 C (정면 0°)
    'view4': (0.3273, 0.4228), # 시점 D (좌측 전방 45°)
    'view1': (0.4274, 0.6058), # 시점 E (좌측 후방 45°)
}

VIEW_THRESHOLDS_15 = {
    'view5': (0.1676, 0.3821),
    'view2': (0.2658, 0.5111),
    'view3': (0.1215, 0.1752),
    'view4': (0.3083, 0.4609),
    'view1': (0.4472, 0.7445),
}


class DL5RTMP1Classifier:
    """
    다이나믹 런지 c5 (상체 과도한 숙임/젖힘 여부) 공식 프로덕션 dl5_rtmp_1 분류기
    - bl5_final 79D Hybrid 구조 계승 (MediaPipe 36D + RTMPose 37D + Cross-modal 6D)
    - 3-Way 최적 앙상블 (CatBoost 50% + LightGBM 30% + XGBoost 20%)
    - 공식 5대 시점(시점 A~E) 적응형 3-State 캘리브레이션
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        if not os.path.exists(self.bundle_path):
            raise FileNotFoundError(f"dl5_rtmp_1 번들 파일이 존재하지 않습니다: {self.bundle_path}")

        with open(self.bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.model_name = bundle.get('model_name', 'dl5_rtmp_1')
        self.exercise = bundle.get('exercise', '다이나믹 런지')
        self.condition = bundle.get('condition', 'c5 (상체의 과도한 숙임/젖힘 여부)')
        self.feature_names = bundle.get('feature_names', LUNGE_C5_HYBRID_79D_NAMES)
        self.weights = bundle.get('ensemble_weights', [0.5, 0.3, 0.2])
        self.model_cat = bundle['model_cat']
        self.model_lgb = bundle['model_lgb']
        self.model_xgb = bundle['model_xgb']
        self.tau_clean_default = float(bundle.get('tau_clean_default', 0.250))
        self.tau_defect_default = float(bundle.get('tau_defect_default', 0.400))
        self.view_thresholds_10 = bundle.get('view_thresholds_10', VIEW_THRESHOLDS_10)
        self.view_thresholds_15 = bundle.get('view_thresholds_15', VIEW_THRESHOLDS_15)
        self.view_thresholds = bundle.get('view_thresholds', self.view_thresholds_10)
        self.benchmark_results = bundle.get('benchmark_results', {})
        self.is_loaded = True

    @classmethod
    def extract_features(
        cls,
        mp_kps_33_3d: np.ndarray,
        rtmp_kps_26_2d: np.ndarray
    ) -> np.ndarray:
        """
        MediaPipe 3D (64, 33, 3) 및 RTMPose 2D (64, 26, 2/3)로부터 79차원 하이브리드 피처 벡터 추출
        """
        return extract_lunge_c5_hybrid_features(mp_kps_33_3d, rtmp_kps_26_2d)

    def predict_proba(
        self,
        mp_kps_33_3d: np.ndarray,
        rtmp_kps_26_2d: np.ndarray,
        view: Optional[str] = None
    ) -> float:
        """
        단일 시퀀스로부터 79D 피처 추출 후 3-Way 앙상블 결함 확률 반환 (0.0 ~ 1.0)
        """
        feat_vec = self.extract_features(mp_kps_33_3d, rtmp_kps_26_2d)
        return self.predict_proba_from_features(feat_vec)

    def predict_proba_from_features(self, feat_vec: np.ndarray) -> float:
        """
        79차원 사전 추출 피처 벡터로부터 3-Way 앙상블 결함 확률 반환
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
        view: Optional[str] = None,
        max_rejection: float = 0.10
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스에 대한 c5 세부 결함 정밀 추론
        - Input:
            * mp_kps_33_3d: (64, 33, 3) MediaPipe 3D 키포인트 배열
            * rtmp_kps_26_2d: (64, 26, 2) 또는 (64, 26, 3) RTMPose 2D 키포인트 배열
            * view: 카메라 시점 코드 ('view1'~'view5' 또는 '시점 A'~'시점 E')
            * max_rejection: 보류율 제약 (0.10: 10% 이내 엄격, 0.15: 15% 이내)
        - Output:
            * status: 'Normal' (정상), 'Defect' (결함), 'Caution_Retake' (판단 보류)
            * state_code: 0 (정상), 1 (결함), 2 (보류)
            * defect_prob: float (0.0 ~ 1.0)
            * confidence: float (0.0 ~ 1.0)
            * applied_view: 공식 카메라 시점 명칭
            * thresholds: 적용된 (tau_clean, tau_defect)
            * defect_types: 감지된 세부 운동학적 결함 목록
            * feedback: 다이나믹 런지 맞춤형 코칭 가이드 메시지
            * key_metrics: 주요 생체역학 지표 요약
        """
        feats = self.extract_features(mp_kps_33_3d, rtmp_kps_26_2d)
        prob = self.predict_proba_from_features(feats)

        # 시점별 최적 임계값 선택
        v_canon = VIEW_CANONICAL_MAP.get(str(view), None) if view else None
        threshold_table = self.view_thresholds_15 if max_rejection > 0.12 else self.view_thresholds_10

        if v_canon in threshold_table:
            t_clean, t_defect = threshold_table[v_canon]
            display_view = VIEW_DISPLAY_MAP.get(v_canon, v_canon)
        else:
            t_clean, t_defect = self.tau_clean_default, self.tau_defect_default
            display_view = "전체 시점 통합 기본값 (Global View)"

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

        # 주요 생체역학 지표 역추적 (MediaPipe 36D + RTMP 37D + Cross-modal)
        pitch_bottom = float(feats[0])     # MP_lunge_vspa_torso_pitch_bottom
        pitch_peak = float(feats[1])       # MP_lunge_vspa_torso_pitch_peak
        pitch_min = float(feats[2])        # MP_lunge_vspa_torso_pitch_min
        tibia_parallel = float(feats[7])   # MP_lunge_tibia_torso_parallelism
        trunk_comp = float(feats[44])      # RTMP_trunk_compression
        spec_ratio = float(feats[52])      # RTMP_spectral_energy_ratio
        burst_jerk = float(feats[26])      # MP_lunge_early_ascent_burst_jerk
        cross_diff = float(feats[73])      # cross_mp_pitch_rtmp_angle_diff

        defect_types = []
        feedback_msgs = []

        if is_defect or state == "Caution_Retake":
            # 1. 다이나믹 스텝 착지 시 상체 과도한 전방 쏠림 (Trunk Flexion)
            if pitch_peak > 28.0 or (pitch_bottom > 22.0 and trunk_comp > 0.12):
                defect_types.append("상체 과도한 전방 쏠림 (Excessive Forward Trunk Lean)")
                feedback_msgs.append(
                    f"다이나믹 스텝 착지 및 최하사점에서 상체가 앞으로 {pitch_peak:.1f}° 과도하게 숙여졌습니다. "
                    f"스텝을 디딜 때 복압을 잡고 가슴을 들어 상체를 지면과 수직에 가깝게 세우세요."
                )

            # 2. 추진 복귀 시 상체 과도한 후방 젖힘 (Trunk Hyperextension)
            if pitch_min < -6.0 or feats[4] > 0.05:
                defect_types.append("상체 과도한 후방 젖힘 (Trunk Hyperextension)")
                feedback_msgs.append(
                    f"원위치 복귀 추진 시 상체가 뒤로 {abs(pitch_min):.1f}° 젖혀져 허리에 부담이 가해지고 있습니다. "
                    f"갈비뼈를 아래로 당기고 척추 중립을 유지한 채 앞발로 지면을 밀어내세요."
                )

            # 3. 정강이-체간 평행도 붕괴 (Tibia-Torso Parallelism Error)
            if tibia_parallel > 13.0:
                defect_types.append("정강이-상체 축 불일치 (Tibia-Torso Parallelism Deficit)")
                feedback_msgs.append(
                    f"앞다리 정강이 각도와 상체 기울기 간에 {tibia_parallel:.1f}°의 불일치가 감지되었습니다. "
                    f"착지 발목과 고관절 힌지 각도의 밸런스를 맞춰 상체와 정강이가 평행하도록 조절하세요."
                )

            # 4. 동적 지면 반력 불안정성 / 버스트 저크 (Dynamic Burst Jerk)
            if burst_jerk > 0.65 or spec_ratio > 1.20:
                defect_types.append("동적 반환 구간 체간 반동 저크 (Trunk Burst Jerk)")
                feedback_msgs.append(
                    f"방향 전환 추진 시 상체의 순간적인 반동/튕김 저크 현상이 관측되었습니다. "
                    f"급격한 반동을 줄이고 둔근과 대퇴사두근의 통제된 긴장감으로 부드럽게 복귀하세요."
                )

        if not defect_types:
            if is_defect:
                defect_types.append("체간 시상면 안정성 저하 (Trunk Sagittal Instability)")
                feedback_msgs.append("상체의 전후 궤적 흔들림이 감지되었습니다. 코어에 단단히 힘을 주고 시선을 정면에 고정하세요.")
            elif state == "Caution_Retake":
                feedback_msgs.append("상체 궤적이 경계 구간에 있습니다. 신체 전신이 카메라에 명확히 들어오도록 각도를 맞추고 1회 다시 수행해 주세요.")
            else:
                feedback_msgs.append("스텝 착지와 복귀 전 과정에서 상체 척추 중립이 완벽하게 유지된 뛰어난 다이나믹 런지입니다.")

        return {
            'model': 'dl5_rtmp_1',
            'status': state,
            'state_code': state_code,
            'is_defect': is_defect,
            'defect_prob': round(prob, 4),
            'confidence': round(confidence, 4),
            'applied_view': display_view,
            'thresholds': (round(t_clean, 4), round(t_defect, 4)),
            'defect_types': defect_types,
            'feedback': feedback_msgs,
            'key_metrics': {
                'torso_pitch_peak_deg': round(pitch_peak, 2),
                'torso_pitch_bottom_deg': round(pitch_bottom, 2),
                'torso_pitch_min_deg': round(pitch_min, 2),
                'tibia_torso_parallelism_err_deg': round(tibia_parallel, 2),
                'trunk_compression_ratio': round(trunk_comp, 3),
                'spectral_energy_ratio': round(spec_ratio, 3),
                'burst_jerk': round(burst_jerk, 3),
                'cross_modal_pitch_diff': round(cross_diff, 2)
            }
        }


# 하위 호환 및 공식 싱글톤 인스턴스 인터페이스
DynamicLungeCondition5ModelRTMP1 = DL5RTMP1Classifier
_default_dl5_rtmp_1_classifier: Optional[DL5RTMP1Classifier] = None


def get_dl5_rtmp_1_classifier(bundle_path: Optional[str] = None) -> DL5RTMP1Classifier:
    global _default_dl5_rtmp_1_classifier
    if _default_dl5_rtmp_1_classifier is None or bundle_path is not None:
        _default_dl5_rtmp_1_classifier = DL5RTMP1Classifier(bundle_path)
    return _default_dl5_rtmp_1_classifier


def predict_dynamic_lunge_c5_rtmp_1(
    mp_kps_33_3d: np.ndarray,
    rtmp_kps_26_2d: np.ndarray,
    view: Optional[str] = None,
    max_rejection: float = 0.10
) -> Dict[str, Any]:
    clf = get_dl5_rtmp_1_classifier()
    return clf.predict_sequence(mp_kps_33_3d, rtmp_kps_26_2d, view=view, max_rejection=max_rejection)
