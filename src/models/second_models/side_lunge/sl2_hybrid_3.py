# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통 발 무릎 방향 일치 여부 (Trunk-Foot-Knee Directional Alignment Defect)
Module: sl2_hybrid_3.py
Purpose: AI-Hub 4,130건 전수 및 5대 공식 시점(시점 A~E) 전수 포괄
         Top-30D 최적화 정예 피처셋 기반 프로덕션 하이브리드 추론 엔진 (sl2_hybrid_3)
         - MediaPipe 3D 핵심 19D + RTMPose 2D 정밀 11D = 30D 정예 생체역학 벡터
         - CatBoost(0.55) + XGBoost(0.25) + LightGBM(0.20) 3-way GBDT 앙상블
         - NLL 최소화 온도 스케일링 (Temperature Scaling)
         - 듀얼 추론 모드:
           1) 시점 지정 시 (View-Aware 35D): 시점 B F1 71.35%, 시점 D F1 63.79% 등 시점별 맞춤 판정
           2) 시점 미지정 시 (Pure 30D): 시점 무관 순수 역학 판정
         - 3-State 선택적 분류 (정상 0, 결함 1, 불확실/재촬영 -1)
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

BUNDLE_PATH = os.path.join(CURRENT_DIR, "sl2_hybrid_3_bundle.pkl")
EVAL_RESULTS_PATH = os.path.join(CURRENT_DIR, "sl2_hybrid_3_evaluation_results.json")

# 공식 5대 시점 명칭 및 코드 매핑
OFFICIAL_VIEW_TAXONOMY = {
    'view5': '시점 A: 우측 후방 (45°)',
    'view2': '시점 B: 우측 전방 (45°)',
    'view3': '시점 C: 정면 (0°)',
    'view4': '시점 D: 좌측 전방 (45°)',
    'view1': '시점 E: 좌측 후방 (45°)'
}

VIEW_ALIAS_TO_CODE = {
    'view1': 'view1', 'view_1': 'view1', '시점 E': 'view1', '시점 E: 좌측 후방 (45°)': 'view1', 'left_back': 'view1',
    'view2': 'view2', 'view_2': 'view2', '시점 B': 'view2', '시점 B: 우측 전방 (45°)': 'view2', 'right_front': 'view2',
    'view3': 'view3', 'view_3': 'view3', '시점 C': 'view3', '시점 C: 정면 (0°)': 'view3', 'front': 'view3',
    'view4': 'view4', 'view_4': 'view4', '시점 D': 'view4', '시점 D: 좌측 전방 (45°)': 'view4', 'left_front': 'view4',
    'view5': 'view5', 'view_5': 'view5', '시점 A': 'view5', '시점 A: 우측 후방 (45°)': 'view5', 'right_back': 'view5',
}

VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']

# 84D 중 30D 정예 인덱스
TOP30_INDICES = [2, 4, 7, 13, 21, 22, 24, 26, 27, 35, 36, 39, 40, 41, 46, 47, 48, 49, 52, 58, 60, 61, 62, 63, 65, 70, 73, 75, 77, 80]


class SL2Hybrid3Classifier:
    """
    사이드 런지 c2 (몸통·발·무릎 방향 불일치) 5대 전 시점 포괄 sl2_hybrid_3 분류기
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or BUNDLE_PATH
        self.is_loaded = False
        self.bundle_data = None
        self.pure_30d = None
        self.view_aware_35d = None
        self.weights = {'catboost': 0.55, 'xgboost': 0.25, 'lightgbm': 0.20}

        if os.path.exists(self.bundle_path):
            self.load_bundle(self.bundle_path)

    def load_bundle(self, path: str):
        with open(path, 'rb') as f:
            b = pickle.load(f)
        self.bundle_data = b
        self.pure_30d = b['pure_30d']
        self.view_aware_35d = b['view_aware_35d']
        self.weights = b.get('weights', self.weights)
        self.is_loaded = True

    def _prepare_vector(self, X: np.ndarray, view: Optional[str] = None) -> Tuple[np.ndarray, bool]:
        """
        입력 배열의 규격에 따라 30D 또는 35D로 변환
        - X가 84D이면 TOP30_INDICES로 30D 슬라이싱
        - X가 30D이고 view가 제공되면 35D로 확장
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        D = X.shape[1]
        if D == 84:
            X_30d = X[:, TOP30_INDICES]
        elif D == 30:
            X_30d = X
        elif D == 35:
            return X, True
        else:
            raise ValueError(f"Expected 30, 35, or 84 features, got {D}")

        if view is not None:
            v_code = VIEW_ALIAS_TO_CODE.get(str(view).strip(), None)
            if v_code is not None:
                # 5-view one-hot: [view5, view2, view3, view4, view1]
                one_hot = np.zeros((X_30d.shape[0], 5), dtype=np.float32)
                col_idx = VIEW_ORDER.index(v_code)
                one_hot[:, col_idx] = 1.0
                X_35d = np.hstack([X_30d, one_hot])
                return X_35d, True

        return X_30d, False

    def predict_proba(self, X: np.ndarray, view: Optional[str] = None) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Model bundle is not loaded. Train or load bundle first.")

        X_input, is_view_aware = self._prepare_vector(X, view)

        if is_view_aware:
            sub = self.view_aware_35d
            expected_dim = 35
        else:
            sub = self.pure_30d
            expected_dim = 30

        assert X_input.shape[1] == expected_dim, f"Expected {expected_dim} features, got {X_input.shape[1]}"

        p_cb = np.mean([m.predict_proba(X_input)[:, 1] for m in sub['cb_models']], axis=0)
        p_xgb = np.mean([m.predict_proba(X_input)[:, 1] for m in sub['xgb_models']], axis=0)
        p_lgb = np.mean([m.predict_proba(X_input)[:, 1] for m in sub['lgb_models']], axis=0)

        w_cb = self.weights.get('catboost', 0.55)
        w_xgb = self.weights.get('xgboost', 0.25)
        w_lgb = self.weights.get('lightgbm', 0.20)
        p_raw = w_cb * p_cb + w_xgb * p_xgb + w_lgb * p_lgb

        # 온도 스케일링
        eps = 1e-7
        p_clip = np.clip(p_raw, eps, 1.0 - eps)
        z = np.log(p_clip / (1.0 - p_clip))
        T = sub['temperature']
        p_cal = 1.0 / (1.0 + np.exp(-z / T))
        return p_cal

    def predict_3state(
        self,
        X: np.ndarray,
        view: str = 'view2',
        profile: str = 'f1_breakthrough'
    ) -> Dict[str, Any]:
        """
        3-State 선택적 신뢰도 분류
        Returns:
            dict: {prediction, label, state, probability, view, view_name, profile, tau_clean, tau_defect}
        """
        v_code = VIEW_ALIAS_TO_CODE.get(str(view).strip(), 'view2')
        prob = float(self.predict_proba(X, view=v_code)[0])

        sub = self.view_aware_35d
        v_info = sub['view_metrics'].get(v_code, sub['view_metrics']['view2'])

        if profile == 'clinical_strict':
            th = v_info['clinical_strict']
        else: # f1_breakthrough
            th = v_info['f1_breakthrough']

        tau_c = th['tau_clean']
        tau_d = th['tau_defect']

        if prob <= tau_c:
            pred = 0
            label_kr = "정상 (Clean)"
            state = "DECISION_NORMAL"
        elif prob >= tau_d:
            pred = 1
            label_kr = "결함 감지 (몸통·발·무릎 방향 불일치)"
            state = "DECISION_DEFECT"
        else:
            pred = -1
            label_kr = "불확실 (재촬영 권고)"
            state = "UNCERTAIN_RETAKE"

        return {
            'prediction': pred,
            'label': label_kr,
            'state': state,
            'probability': round(prob, 4),
            'view': v_code,
            'view_name': OFFICIAL_VIEW_TAXONOMY[v_code],
            'profile': profile,
            'tau_clean': tau_c,
            'tau_defect': tau_d
        }
