# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Module: cl2_final.py
Purpose: 크로스 런지 Condition 2 공식 최종 확정 모델 (Final Production Model)
         - 공식 채택 모델: cl2_hybrid_2 (MediaPipe 3D + RTMPose Halpe26 2D 67D 정예 하이브리드)
         - 3-State F1: 73.29% (MediaPipe 단독 71.50% 및 RTMPose 단독 67.50% 완전 돌파)
         - ROC-AUC: 73.65% (역대 최고 기록)
         - 3-State Accuracy: 66.88%, Recall: 86.53%, Coverage: 91.02% (보류율 8.98%)
         - Guo et al. 온도 스케일링 캘리브레이션 (T=0.8504, ECE=3.52%)
         - 5대 공식 시점(시점 A~시점 E) F1 극대화 비대칭 3-State 신뢰도 게이팅 적용
"""

import os
import sys
import pickle
from typing import Dict, Any, Union, Optional

import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(CURRENT_DIR, "cl2_final_bundle.pkl")

from src.models.second_models.cross_lunge.cl2_hybrid_2 import (
    extract_cl2_hybrid_2_features,
    CL2_HYBRID_2_FEATURE_NAMES,
    VIEW_NAME_TO_CODE,
    VIEW_CODE_TO_NAME
)

# 싱글톤 캐시
_CACHED_BUNDLE = None


def load_final_model_bundle(bundle_path: str = BUNDLE_PATH) -> Dict[str, Any]:
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None:
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"최종 모델 번들을 찾을 수 없습니다: {bundle_path}")
        with open(bundle_path, 'rb') as f:
            _CACHED_BUNDLE = pickle.load(f)
    return _CACHED_BUNDLE


def predict_cross_lunge_c2(
    input_data: np.ndarray,
    kps_26_64f: Optional[np.ndarray] = None,
    camera_view: str = '시점 C',
    bundle_path: str = BUNDLE_PATH
) -> Dict[str, Any]:
    """
    크로스 런지 Condition 2 공식 최종 판정 함수 (cl2_final / cl2_hybrid_2)

    Args:
        input_data:
            1) (64, 33, 3) 넘파이 배열: MediaPipe 33 키포인트 시퀀스 (kps_26_64f와 함께 제공)
            2) (67,) 넘파이 배열: 이미 추출된 67D 정예 하이브리드 피처 벡터
        kps_26_64f: (64, 26, 3) RTMPose Halpe26 키포인트 시퀀스 (input_data가 MP 시퀀스일 때 필수)
        camera_view: 공식 시점 명칭('시점 A'~'시점 E') 또는 내부 코드('view1'~'view5')
        bundle_path: 최종 모델 번들 경로

    Returns:
        3-State 진단 결과 딕셔너리 (CLEAN/DEFECT/RETAKE, 결함확률, 신뢰도, 임계값 정보 등)
    """
    bundle = load_final_model_bundle(bundle_path)
    v_code = VIEW_NAME_TO_CODE.get(str(camera_view).strip(), 'view3')
    v_name = VIEW_CODE_TO_NAME.get(v_code, '시점 C (정면 0°)')

    input_arr = np.asarray(input_data, dtype=np.float32)

    # 1. 입력 형태에 따른 피처 벡터 결정
    if input_arr.ndim == 3 and input_arr.shape == (64, 33, 3):
        if kps_26_64f is None:
            raise ValueError("MediaPipe 텐서(64, 33, 3) 입력 시 RTMPose 텐서(kps_26_64f, shape: 64, 26, 3)를 반드시 함께 제공해야 합니다.")
        kps_rtm = np.asarray(kps_26_64f, dtype=np.float32)
        if kps_rtm.shape != (64, 26, 3):
            raise ValueError(f"RTMPose 텐서 규격 불일치: 기대값 (64, 26, 3), 실제값 {kps_rtm.shape}")
        feat_67d = extract_cl2_hybrid_2_features(input_arr, kps_rtm, view_code=v_code)
    elif input_arr.ndim in (1, 2) and input_arr.size == 67:
        feat_67d = input_arr.reshape(67)
    else:
        raise ValueError(
            f"지원하지 않는 입력 규격입니다: shape={input_arr.shape}. "
            "(64, 33, 3) MP 텐서 + (64, 26, 3) RTM 텐서, 또는 67차원 피처 벡터를 입력해 주세요."
        )

    # 2. 앙상블 확률 계산
    X_in = feat_67d.reshape(1, -1)
    models_cat = bundle['models_cat']
    models_lgb = bundle['models_lgb']
    models_xgb = bundle['models_xgb']
    w_cat, w_lgb, w_xgb = bundle.get('weights', [0.8862, 0.0, 0.1138])
    temperature = bundle.get('temperature', 0.8504)

    p_cat = float(np.mean([m.predict_proba(X_in)[0, 1] for m in models_cat]))
    p_lgb = float(np.mean([m.predict(X_in)[0] for m in models_lgb])) if len(models_lgb) > 0 else 0.0

    import xgboost as xgb
    dmat = xgb.DMatrix(X_in)
    p_xgb = float(np.mean([m.predict(dmat)[0] for m in models_xgb])) if len(models_xgb) > 0 else 0.0

    p_blend = float(w_cat * p_cat + w_lgb * p_lgb + w_xgb * p_xgb)

    # 3. 온도 스케일링 보정
    eps = 1e-7
    p_blend_clip = float(np.clip(p_blend, eps, 1.0 - eps))
    logit = float(np.log(p_blend_clip / (1.0 - p_blend_clip)))
    prob = float(1.0 / (1.0 + np.exp(-logit / max(0.1, temperature))))
    prob = float(np.clip(prob, 0.0, 1.0))

    # 4. 시점별 임계값 로드
    view_th = bundle['view_thresholds'].get(v_code, {'tau_clean': 0.35, 'tau_defect': 0.45})
    tau_clean = float(view_th['tau_clean'])
    tau_defect = float(view_th['tau_defect'])

    # 5. 2-State 판정 (단순 0.5 또는 균형 분할)
    is_defect_2s = bool(prob >= 0.5)
    state_2s = "DEFECT" if is_defect_2s else "NORMAL"

    # 6. 3-State 판정 (공식 SOTA F1: 73.29%, 커버리지: 91.02%)
    if prob <= tau_clean:
        state_3s = "CLEAN"
        state_kr = "정상"
        defect_pred = False
        confidence = float((tau_clean - prob) / max(1e-4, tau_clean))
        retake_reason = None
    elif prob >= tau_defect:
        state_3s = "DEFECT"
        state_kr = "오류 (몸통-앞발-앞무릎 불일치)"
        defect_pred = True
        confidence = float((prob - tau_defect) / max(1e-4, 1.0 - tau_defect))
        retake_reason = None
    else:
        state_3s = "RETAKE"
        state_kr = "판정 보류 (재촬영 권고)"
        defect_pred = None
        dist_to_edge = min(prob - tau_clean, tau_defect - prob)
        margin_width = max(1e-4, tau_defect - tau_clean)
        confidence = float(1.0 - (dist_to_edge / (margin_width * 0.5)))
        retake_reason = (
            f"{v_name}에서 몸통, 앞무릎, 앞발의 방향 회전각 정렬이 모호합니다. "
            "몸통과 앞발, 앞무릎이 같은 전방 방향을 유지하도록 정렬하여 다시 촬영해 주세요."
        )

    return {
        'model_version': 'cl2_final (cl2_hybrid_2 확정본)',
        'defect_code': 'cl2',
        'defect_name': 'Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부',
        'camera_view': v_name,
        'camera_view_code': v_code,
        'defect_probability': prob,
        'confidence': float(np.clip(confidence, 0.0, 1.0)),
        'state': state_3s,
        'state_kr': state_kr,
        'prediction_2state': state_2s,
        'prediction_3state': state_3s,
        'is_defect': defect_pred,
        'applied_thresholds_3state': {
            'tau_clean': tau_clean,
            'tau_defect': tau_defect
        },
        'retake_reason': retake_reason,
        'feature_dim': 67,
        'features_67d': feat_67d
    }


if __name__ == "__main__":
    print("[*] cl2_final 무결성 셀프 테스트 시작...")
    dummy_feat = np.zeros(67, dtype=np.float32)
    res = predict_cross_lunge_c2(dummy_feat, camera_view='시점 C')
    print("[+] cl2_final 67D 피처 단일 추론 무결성 테스트 통과!")
    print(f"    - Model: {res['model_version']}")
    print(f"    - State: {res['state_kr']} ({res['state']}), Prob: {res['defect_probability']:.4f}")
    print(f"    - Thresholds: {res['applied_thresholds_3state']}")
