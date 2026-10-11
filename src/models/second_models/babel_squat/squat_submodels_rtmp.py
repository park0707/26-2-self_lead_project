# -*- coding: utf-8 -*-
# ==============================================================================
# [공식 최종 확정 프로덕션 통합 엔진 / OFFICIAL FINAL PRODUCTION INTEGRATED ENGINE]
# 프로젝트: vidio-health-trainer
# 운동 종목: 바벨 스쿼트 (Barbell Squat)
# 파일 역할: 바벨 스쿼트 4대 세부 결함(c1~c4) 공식 최종 챔피언 통합 배포 엔진
# 패키지 표준 별칭: BabelSquatIntegratedEngine, BabelSquatEngine (src.models.second_models.babel_squat)
# ==============================================================================
"""
vidio-health-trainer: 바벨 스쿼트 4대 세부 결함 공식 최종 챔피언 통합 엔진 (OFFICIAL FINAL ENGINE)
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: squat_submodels_rtmp.py
Status: [공식 최종 확정 배포 엔진 / OFFICIAL FINAL PRODUCTION ENGINE]
        - 1) c1 v12_rtmp: 척추 중립 위반 (Spine Neutrality, 38D)
        - 2) c2 v9_rtmp: 고개 정면 및 시선 (Head/Gaze Orientation, 19D)
        - 3) c3 v7_rtmp: 발-무릎 방향 일치 (Knee Valgus, 22D)
        - 4) c4 v6_rtmp: 발바닥 지면 고정 (Foot Grounding, 25D)
        - 3계층 의사결정 파이프라인 탑재:
          * Level 1: Guo et al. (ICML 2017) NLL 최소화 기반 온도 스케일링
          * Level 2: 5대 공식 시점(시점 A~E)별 불확실성 적응형 3-State 신뢰도 게이팅
          * Level 3: Calibrated Soft-Risk Probability Fusion 종합 세션 판정 (Clean / Defect / Retake) & 4-Defect EMR
"""

import os
import sys
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from c1_model_v12_rtmp import BabelSquatC1ModelV12RTMP
from c2_model_v9_rtmp import BabelSquatC2ModelV9RTMP
from c3_model_v7_rtmp import BabelSquatC3ModelV7RTMP
from c4_model_v6_rtmp import BabelSquatC4ModelV6RTMP

VIEW_ALIAS_MAP: Dict[str, str] = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면 (0도)',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)'
}

DEFAULT_INTEGRATED_PARAMS: Dict[str, Any] = {
    'soft_risk_tau_clean': 0.45,
    'soft_risk_tau_defect': 0.15,
    'weights_path': os.path.join(CURRENT_DIR, "weights_squat_submodels_rtmp.pkl"),
    'calibration_params_path': os.path.join(CURRENT_DIR, "squat_submodels_rtmp_calibration_params.json")
}


class BabelSquatSubmodelsRTMP:
    """
    바벨 스쿼트 4대 세부 결함(c1~c4) 통합 RTMPose 엔진
    """
    EXERCISE_NAME = "바벨 스쿼트"
    VERSION = "v13_rtmp"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        calibration_path: Optional[str] = None
    ):
        weights_path = weights_path or DEFAULT_INTEGRATED_PARAMS['weights_path']
        calibration_path = calibration_path or DEFAULT_INTEGRATED_PARAMS['calibration_params_path']

        # 4대 서브모델 인스턴스화
        self.c1_model = BabelSquatC1ModelV12RTMP()
        self.c2_model = BabelSquatC2ModelV9RTMP()
        self.c3_model = BabelSquatC3ModelV7RTMP()
        self.c4_model = BabelSquatC4ModelV6RTMP()

        self.tau_clean = DEFAULT_INTEGRATED_PARAMS['soft_risk_tau_clean']
        self.is_loaded = False

        if calibration_path and os.path.exists(calibration_path):
            self.load_calibration_params(calibration_path)

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def load_calibration_params(self, param_path: str):
        with open(param_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
            if 'c1' in params:
                self.c1_model.params = params['c1']
            if 'c2' in params:
                self.c2_model.params = params['c2']
            if 'c3' in params:
                self.c3_model.params = params['c3']
            if 'c4' in params:
                self.c4_model.params = params['c4']
            if 'integrated' in params and 'tau_clean' in params['integrated']:
                self.tau_clean = params['integrated']['tau_clean']

    def load_weights(self, weights_path: str):
        with open(weights_path, 'rb') as f:
            data = pickle.load(f)
            if 'c1_models' in data:
                self.c1_model.models = data['c1_models']
                self.c1_model.is_loaded = True
            if 'c2_models' in data:
                self.c2_model.models = data['c2_models']
                self.c2_model.is_loaded = True
            if 'c3_models' in data:
                self.c3_model.models = data['c3_models']
                self.c3_model.is_loaded = True
            if 'c4_models' in data:
                self.c4_model.models = data['c4_models']
                self.c4_model.is_loaded = True
        self.is_loaded = True

    def extract_features(
        self,
        kps_64f: np.ndarray,
        view_name: str = 'view3',
        spine_9_64f: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        4대 서브모델 피처 일괄 추출
        """
        feats_c1 = self.c1_model.extract_features(kps_64f, spine_9_64f=spine_9_64f)
        feats_c2 = self.c2_model.extract_features(kps_64f, view_name=view_name)
        feats_c3 = self.c3_model.extract_features(kps_64f, view_name=view_name)
        feats_c4 = self.c4_model.extract_features(kps_64f, view_name=view_name)
        return {
            'c1': feats_c1,
            'c2': feats_c2,
            'c3': feats_c3,
            'c4': feats_c4
        }

    def predict_sequence(
        self,
        kps_64f: np.ndarray,
        view_name: str = 'view3',
        spine_9_64f: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스로부터 4대 결함 및 세션 종합 진단
        """
        feats_dict = self.extract_features(kps_64f, view_name=view_name, spine_9_64f=spine_9_64f)
        return self.predict_features_dict(feats_dict, view_name=view_name)

    def predict(
        self,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]],
        view_name: str = 'view3',
        spine_9_64f: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        유연한 통합 예측 인터페이스 (시퀀스 ndarray 또는 사전 추출된 피처 dict 모두 지원)
        """
        if isinstance(input_data, dict):
            return self.predict_features_dict(input_data, view_name=view_name)
        else:
            return self.predict_sequence(input_data, view_name=view_name, spine_9_64f=spine_9_64f)

    def predict_features_dict(
        self,
        feats_dict: Dict[str, np.ndarray],
        view_name: str = 'view3'
    ) -> Dict[str, Any]:
        """
        4대 피처 딕셔너리 {'c1': (38,), 'c2': (19,), 'c3': (22,), 'c4': (25,)} 로부터
        Level 1, 2, 3 종합 진단 수행
        """
        # Level 1 & 2: 서브모델별 예측
        p1 = float(self.c1_model.predict_proba(feats_dict['c1'], view_name=view_name, calibrate=True)[0])
        p2 = float(self.c2_model.predict_proba(feats_dict['c2'], view_name=view_name, calibrate=True)[0])
        p3 = float(self.c3_model.predict_proba(feats_dict['c3'], view_name=view_name, calibrate=True)[0])
        p4 = float(self.c4_model.predict_proba(feats_dict['c4'], view_name=view_name, calibrate=True)[0])

        s1_code, s1_name = self.c1_model.classify_3state(p1, view_name)
        s2_code, s2_name = self.c2_model.classify_3state(p2, view_name)
        s3_code, s3_name = self.c3_model.classify_3state(p3, view_name)
        s4_code, s4_name = self.c4_model.classify_3state(p4, view_name)

        # Level 3: Calibrated Soft-Risk Fusion
        p_clean = float((1.0 - p1) * (1.0 - p2) * (1.0 - p3) * (1.0 - p4))
        has_retake = any(s == 2 for s in [s1_code, s2_code, s3_code, s4_code])
        has_defect = any(s == 1 for s in [s1_code, s2_code, s3_code, s4_code])

        if p_clean >= self.tau_clean and not has_defect:
            session_decision = "CLEAN (전신 무결점 정상 스쿼트)"
            session_state = 0
        elif has_defect or p_clean < 0.20:
            session_decision = "DEFECT (세부 결함 발견)"
            session_state = 1
        elif has_retake:
            session_decision = "RETAKE (불확실성 구간 감지 / 재촬영 안내)"
            session_state = 2
        else:
            session_decision = "CLEAN (관측 가능 정상)"
            session_state = 0

        return {
            'exercise': self.EXERCISE_NAME,
            'version': self.VERSION,
            'view_code': view_name,
            'view_name': VIEW_ALIAS_MAP.get(view_name, view_name),
            'session_state': session_state,
            'session_decision': session_decision,
            'p_clean_fusion': round(p_clean, 4),
            'tau_clean_applied': self.tau_clean,
            'submodels': {
                'c1': {'name': '척추 중립 위반', 'prob': round(p1, 4), 'state_code': s1_code, 'decision': s1_name},
                'c2': {'name': '고개 정면 및 시선', 'prob': round(p2, 4), 'state_code': s2_code, 'decision': s2_name},
                'c3': {'name': '발-무릎 방향 일치', 'prob': round(p3, 4), 'state_code': s3_code, 'decision': s3_name},
                'c4': {'name': '발바닥 지면 고정', 'prob': round(p4, 4), 'state_code': s4_code, 'decision': s4_name}
            }
        }

    def predict_batch(
        self,
        X_dict: Dict[str, np.ndarray],
        views: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        N개 시퀀스 일괄 예측 (OOF 및 평가용)
        """
        N = len(views)
        probs_calibrated = np.zeros((N, 4), dtype=np.float32)
        states_3state = np.zeros((N, 4), dtype=np.int32)

        models = [
            (self.c1_model, 'c1', 0),
            (self.c2_model, 'c2', 1),
            (self.c3_model, 'c3', 2),
            (self.c4_model, 'c4', 3)
        ]

        for m, k, idx in models:
            for vn in np.unique(views):
                mask = (views == vn)
                if np.any(mask):
                    probs_calibrated[mask, idx] = m.predict_proba(X_dict[k][mask], view_name=vn, calibrate=True)
                    for i in np.where(mask)[0]:
                        states_3state[i, idx] = m.classify_3state(probs_calibrated[i, idx], view_name=vn)[0]

        probs_clean = np.prod(1.0 - probs_calibrated, axis=1)
        pred_clean = (probs_clean >= self.tau_clean).astype(int)

        return {
            'probs_calibrated': probs_calibrated,
            'states_3state': states_3state,
            'probs_clean': probs_clean,
            'pred_clean': pred_clean
        }
