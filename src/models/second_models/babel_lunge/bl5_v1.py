# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: bl5_v1.py
Model Name: bl5_v1 (Barbell Lunge Defect 5 - Torso Sagittal Neutrality Classifier)
Purpose: 바벨 런지 5번째 세부 오류인 '상체의 과도한 숙임/젖힘 여부' 프로덕션 추론 서빙 인터페이스
         - 100% 순수 비전 MediaPipe 33관절 3D (64F 3차 스플라인) 독립 입력
         - 32차원 도메인 생체역학 피처 고속 추출
         - 5-Fold Tri-Ensemble (LightGBM + CatBoost + Random Forest) 확률 추론
         - 시점 적응형 캘리브레이션 및 3-State 선택적 판정(Clean, Defect, Retake)
         - 사용자 친화적 교정 피드백 자동 생성
"""

import os
import sys
import pickle
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features import (
    extract_lunge_c5_features,
    extract_lunge_c5_feature_vector,
    LUNGE_C5_FEATURE_NAMES
)

DEFAULT_BUNDLE_PATH = r"D:\src\models\second_models\babel_lunge\weights\bl5_v1_bundle.pkl"

# AGENTS.md 공식 5대 시점 표준 매핑
VIEW_ALIAS_MAP = {
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
    'view5': 'view5',
    'view2': 'view2',
    'view3': 'view3',
    'view4': 'view4',
    'view1': 'view1'
}


class BL5V1Classifier:
    """
    바벨 런지 5번째 오류 '상체의 과도한 숙임/젖힘 여부' 공식 추론 엔진
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        self.is_loaded = False
        self.lgb_models = []
        self.cb_models = []
        self.rf_models = []
        self.feature_names = LUNGE_C5_FEATURE_NAMES
        self.best_params = {
            'tau': 0.50,
            'margin': 0.05
        }
        self.view_audit = {}

        if os.path.exists(self.bundle_path):
            self._load_bundle(self.bundle_path)

    def _load_bundle(self, path: str):
        try:
            with open(path, 'rb') as f:
                bundle = pickle.load(f)
            self.lgb_models = bundle.get('lgb_models', [])
            self.cb_models = bundle.get('cb_models', [])
            self.rf_models = bundle.get('rf_models', [])
            self.feature_names = bundle.get('feature_names', LUNGE_C5_FEATURE_NAMES)
            self.best_params = bundle.get('best_3state_params', self.best_params)
            self.view_audit = bundle.get('view_audit', {})
            self.is_loaded = True
        except Exception as e:
            print(f"[!] 번들 로드 실패 ({path}): {e}")

    @classmethod
    def extract_features(cls, kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None) -> Dict[str, float]:
        """AGENTS.md 표준 인터페이스: 순수 64F 시퀀스로부터 32D 생체역학 피처 추출"""
        return extract_lunge_c5_features(kps_33_3d_64f, vis_64f)

    def predict_proba(self, kps_33_3d_64f: np.ndarray, view: Optional[str] = None) -> float:
        """
        단일 런지 시퀀스에 대한 결함 확률(0.0 ~ 1.0) 추론
        - kps_33_3d_64f: (64, 33, 3) 관절 좌표 배열
        - view: 공식 시점 명칭 ('시점 A'~'시점 E') 또는 코드 ('view1'~'view5')
        """
        if not self.is_loaded:
            # 모델 미로드 시 룰베이스 안전 대체 (Safe Fallback)
            feats = self.extract_features(kps_33_3d_64f)
            pitch_min = feats.get("lunge_torso_pitch_min", 0.0)
            pitch_peak = feats.get("lunge_torso_pitch_peak", 0.0)
            if pitch_min < -8.0 or pitch_peak > 25.0:
                return 0.85
            return 0.15

        vec = extract_lunge_c5_feature_vector(kps_33_3d_64f).reshape(1, -1)

        # 5-Fold Tri-Ensemble 평균 확률 계산
        lgb_probs = [m.predict_proba(vec)[0, 1] for m in self.lgb_models]
        cb_probs = [m.predict_proba(vec)[0, 1] for m in self.cb_models]
        rf_probs = [m.predict_proba(vec)[0, 1] for m in self.rf_models]

        p_lgb = float(np.mean(lgb_probs))
        p_cb = float(np.mean(cb_probs))
        p_rf = float(np.mean(rf_probs))

        prob = 0.45 * p_lgb + 0.35 * p_cb + 0.20 * p_rf

        # 시점 C(정면 0°) 보정 레이어
        canonical_view = VIEW_ALIAS_MAP.get(view, None) if view else None
        if canonical_view == 'view3':
            # 정면 원근 수축 보정: 3D 가상 시상각 및 원근 단축비 보정
            feats = self.extract_features(kps_33_3d_64f)
            foreshorten = feats.get("lunge_torso_foreshortening_ratio", 1.0)
            if foreshorten < 0.88:
                prob = min(1.0, prob * 1.15)

        return float(np.clip(prob, 0.0, 1.0))

    def predict_3state(self, kps_33_3d_64f: np.ndarray, view: Optional[str] = None) -> Dict[str, Any]:
        """
        3-State 선택적 분류 판정 (Clean / Defect / Retake)
        반환:
          - state: 'Clean' | 'Defect' | 'Retake'
          - state_code: 0 (정상) | 1 (결함) | 2 (보류)
          - prob: 결함 확률
          - confidence: 경계선 대비 마진 확신도
          - defect_type: 세부 결함 유형 ('정상', '상체 과도한 뒤로 젖힘', '상체 과도한 앞으로 숙임')
          - feedback: 코칭 피드백 메시지
          - features_summary: 핵심 생체역학 지표
        """
        prob = self.predict_proba(kps_33_3d_64f, view=view)
        tau = self.best_params.get('tau', 0.50)
        margin = self.best_params.get('margin', 0.05)

        low_b = tau - margin
        high_b = tau + margin

        feats = self.extract_features(kps_33_3d_64f)
        pitch_min = feats.get("lunge_torso_pitch_min", 0.0)
        pitch_peak = feats.get("lunge_torso_pitch_peak", 0.0)
        pitch_bottom = feats.get("lunge_torso_pitch_bottom_win", 0.0)

        confidence = float(abs(prob - tau) / (1.0 - tau if prob >= tau else tau))

        if prob >= high_b:
            state = "Defect"
            state_code = 1
            if pitch_min < -6.0:
                defect_type = "상체 과도한 뒤로 젖힘 (Torso Hyperextension)"
                feedback = (
                    f"런지 하강 시 상체가 뒤로 {abs(pitch_min):.1f}° 과도하게 젖혀졌습니다. "
                    "골반을 앞으로 밀지 말고 척추 기립근과 복부에 힘을 주어 상체를 지면과 수직(직립)으로 유지하세요."
                )
            elif pitch_peak > 22.0:
                defect_type = "상체 과도한 앞으로 숙임 (Excessive Forward Flexion)"
                feedback = (
                    f"런지 최하사점에서 상체가 앞으로 {pitch_peak:.1f}° 과도하게 고꾸라졌습니다. "
                    "가슴을 펴고 시선을 정면에 고정하여 상체 전경각을 10° 이내로 통제하세요."
                )
            else:
                defect_type = "시상면 체간 중립 불안정 (Trunk Pitch Instability)"
                feedback = (
                    "동작 전반에 걸쳐 상체 전후 흔들림이 감지되었습니다. 코어 안정성을 유지하세요."
                )
        elif prob <= low_b:
            state = "Clean"
            state_code = 0
            defect_type = "정상 자세 (Clean Upright Torso)"
            feedback = (
                f"상체 중립이 훌륭하게 유지되었습니다. (최하사점 피치: {pitch_bottom:.1f}°, 가동범위: {feats.get('lunge_torso_pitch_rom', 0.0):.1f}°)"
            )
        else:
            state = "Retake"
            state_code = 2
            defect_type = "신뢰도 보류 (Uncertain / Retake Recommended)"
            feedback = (
                "상체 기울기 판정 경계선 부근에 위치하여 정확한 판정이 어렵습니다. "
                "카메라 앵글을 점검하고 명확한 직립 자세로 1회 다시 수행해 주세요."
            )

        return {
            "model": "bl5_v1",
            "exercise": "04_바벨_런지",
            "condition": "c5. 상체의 과도한 숙임/젖힘 여부",
            "state": state,
            "state_code": state_code,
            "prob": float(prob),
            "confidence": float(confidence),
            "defect_type": defect_type,
            "feedback": feedback,
            "features_summary": {
                "pitch_bottom_win_deg": round(pitch_bottom, 2),
                "pitch_peak_forward_deg": round(pitch_peak, 2),
                "pitch_min_backward_deg": round(pitch_min, 2),
                "pitch_rom_deg": round(feats.get("lunge_torso_pitch_rom", 0.0), 2),
                "foreshortening_ratio": round(feats.get("lunge_torso_foreshortening_ratio", 1.0), 4),
                "delta_from_setup_deg": round(feats.get("lunge_delta_pitch_from_setup", 0.0), 2)
            }
        }
