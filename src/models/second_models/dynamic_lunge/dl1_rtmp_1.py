# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05 Forward + 06 Backward)
Defect: c1 (앞다리 무릎 각도 90도 미달/과도 굴곡 오류)
Module: dl1_rtmp_1.py
Purpose: 다이나믹 런지 c1 공식 프로덕션 모델 배포 인터페이스 (dl1_rtmp_1)
Architecture:
  - Backbone: RTMPose 2D Halpe26 -> MediaPipe 33 Fake-Z 사상
  - Feature Extractor: lunge_c1_features_v4.py (30D 도메인 기구학 정예 피처)
  - Classifier: CatBoost / GBDT 앙상블 최적화
  - Decision Logic: 공식 5대 시점(A~E)별 적응형 3-State 선택적 분류 (보류율 <= 10.0% 준수)
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional

import numpy as np

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(r"D:\src\features")
from lunge_c1_features_v4 import (
    extract_lunge_c1_feature_vector_v4,
    extract_lunge_c1_features_v4,
    LUNGE_C1_FEATURE_NAMES_V4
)

BUNDLE_PATH = r"D:\src\models\second_models\dynamic_lunge\dl1_rtmp_1_bundle.pkl"
CBM_PATH = r"D:\src\models\second_models\dynamic_lunge\dl1_rtmp_1.cbm"

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def map_rtmp26_to_mp33_with_fake_z(kps26: np.ndarray) -> np.ndarray:
    """RTMPose Halpe26 2D -> MediaPipe 33 format with fake Z depth"""
    kps33 = np.zeros((64, 33, 3), dtype=np.float32)
    kps33[:, 0] = kps26[:, 0]
    kps33[:, 2] = kps26[:, 1]
    kps33[:, 5] = kps26[:, 2]
    kps33[:, 7] = kps26[:, 3]
    kps33[:, 8] = kps26[:, 4]
    kps33[:, 11] = kps26[:, 5]
    kps33[:, 12] = kps26[:, 6]
    kps33[:, 13] = kps26[:, 7]
    kps33[:, 14] = kps26[:, 8]
    kps33[:, 15] = kps26[:, 9]
    kps33[:, 16] = kps26[:, 10]
    kps33[:, 23] = kps26[:, 11]
    kps33[:, 24] = kps26[:, 12]
    kps33[:, 25] = kps26[:, 13]
    kps33[:, 26] = kps26[:, 14]
    kps33[:, 27] = kps26[:, 15]
    kps33[:, 28] = kps26[:, 16]
    kps33[:, 29] = kps26[:, 24]  # LEFT_HEEL
    kps33[:, 30] = kps26[:, 25]  # RIGHT_HEEL
    kps33[:, 31] = kps26[:, 20]  # LEFT_FOOT_INDEX
    kps33[:, 32] = kps26[:, 21]  # RIGHT_FOOT_INDEX

    left_mp_indices = [1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    right_mp_indices = [4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    kps33[:, left_mp_indices, 2] = -0.1
    kps33[:, right_mp_indices, 2] = 0.1
    return kps33


class DL1RTMP1Classifier:
    """다이나믹 런지 c1(앞다리 무릎 각도 90도) 공식 프로덕션 분류기"""

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"dl1_rtmp_1 프로덕션 번들 모델 파일 없음: {bundle_path}")

        with open(bundle_path, 'rb') as f:
            self.bundle = pickle.load(f)

        self.model_name = self.bundle.get('model_name', 'dl1_rtmp_1')
        self.feature_names = self.bundle.get('feature_names', LUNGE_C1_FEATURE_NAMES_V4)
        self.weights = self.bundle.get('ensemble_weights', (1.0, 0.0, 0.0))
        self.model_cb = self.bundle.get('model_cb')
        self.model_lgb = self.bundle.get('model_lgb')
        self.model_xgb = self.bundle.get('model_xgb')
        self.view_thresholds = self.bundle.get('view_thresholds', {})
        self.global_thresholds = self.bundle.get('global_thresholds', {'tau_clean': 0.35, 'tau_defect': 0.60})

    def predict_sequence(
        self,
        kps_26_64f: np.ndarray,
        view_code: str = 'view3'
    ) -> Dict[str, Any]:
        """
        단일 64프레임 시퀀스에 대한 c1 세부 결함 추론
        - Input: kps_26_64f (64, 26, 3) RTMPose 정규화 키포인트 배열
        - view_code: 'view1' ~ 'view5' (공식 5대 시점)
        - Output: 상태(0: 정상, 1: 결함, 2: 보류), 결함 확률, 상세 피처 딕셔너리
        """
        assert kps_26_64f.shape[:2] == (64, 26), f"입력 텐서 규격 불일치: {kps_26_64f.shape}"

        # 1. Fake Z 주입 및 MP33 변환
        kps33 = map_rtmp26_to_mp33_with_fake_z(kps_26_64f)

        # 2. 도메인 피처 추출
        feat_dict = extract_lunge_c1_features_v4(kps33)
        feat_vec = np.array([feat_dict[name] for name in self.feature_names], dtype=np.float32).reshape(1, -1)
        if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)

        # 3. 모델 확률 추론
        p_cb = float(self.model_cb.predict_proba(feat_vec)[0, 1])
        if self.model_lgb is not None and self.weights[1] > 1e-4:
            p_lgb = float(self.model_lgb.predict_proba(feat_vec)[0, 1])
        else:
            p_lgb = p_cb

        if self.model_xgb is not None and self.weights[2] > 1e-4:
            p_xgb = float(self.model_xgb.predict_proba(feat_vec)[0, 1])
        else:
            p_xgb = p_cb

        prob = float(self.weights[0] * p_cb + self.weights[1] * p_lgb + self.weights[2] * p_xgb)

        # 4. 시점별 적응형 임계값 로드
        v_th = self.view_thresholds.get(view_code, self.global_thresholds)
        tau_clean = float(v_th.get('tau_clean', 0.35))
        tau_defect = float(v_th.get('tau_defect', 0.60))

        # 5. 3-State 판정
        if prob < tau_clean:
            state = 0
            state_str = "CLEAN (정상)"
            feedback = "앞다리 무릎 각도가 이상적인 90도 범위에 부합합니다. 안정적인 착지와 굴곡을 유지하고 있습니다."
        elif prob >= tau_defect:
            state = 1
            state_str = "DEFECT (결함)"
            feedback = "앞다리 무릎이 90도까지 충분히 굽혀지지 않았습니다. 착지 후 체중을 앞다리에 싣고 더 깊게 앉으세요."
        else:
            state = 2
            state_str = "UNCERTAIN (판단 보류 / 재촬영)"
            feedback = "동작 경계 영역이거나 시점 가림으로 판정 신뢰도가 미달합니다. 카메라 각도를 재조정하고 다시 시도하세요."

        return {
            'model_name': self.model_name,
            'defect_code': 'c1',
            'defect_name': '앞다리 무릎 각도 90도 미달/과도 굴곡',
            'state': state,
            'state_str': state_str,
            'defect_probability': prob,
            'feedback': feedback,
            'view_code': view_code,
            'view_name': VIEW_ALIAS_MAP.get(view_code, view_code),
            'metrics': {
                'lead_knee_angle_min': feat_dict['lead_knee_angle_min'],
                'lead_knee_90_error_abs': feat_dict['lead_knee_90_error_abs'],
                'lead_femur_vertical_drop_ratio': feat_dict['lead_femur_vertical_drop_ratio'],
                'lead_tibia_vertical_tilt': feat_dict['lead_tibia_vertical_tilt'],
                'lead_knee_rom_ratio': feat_dict['lead_knee_rom_ratio'],
                'stride_length_normalized': feat_dict['stride_length_normalized'],
            },
            'thresholds': {
                'tau_clean': tau_clean,
                'tau_defect': tau_defect
            }
        }


# 전역 싱글톤 인스턴스
_DEFAULT_CLASSIFIER: Optional[DL1RTMP1Classifier] = None


def get_dl1_classifier() -> DL1RTMP1Classifier:
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = DL1RTMP1Classifier()
    return _DEFAULT_CLASSIFIER


def predict_dynamic_lunge_c1(
    kps_26_64f: np.ndarray,
    view_code: str = 'view3'
) -> Dict[str, Any]:
    """공식 표준 1-함수 추론 인터페이스"""
    clf = get_dl1_classifier()
    return clf.predict_sequence(kps_26_64f, view_code=view_code)
