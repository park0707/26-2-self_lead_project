# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Defect: d1. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Module: stiff_d1_inference.py
Purpose: 단일 64프레임 시퀀스에 대한 프로덕션 실서비스 추론 파이프라인
         - 38D Zero-Calibrated 도메인 피처 추출
         - LightGBM + CatBoost Soft-Voting 앙상블
         - 3-State View-Calibrated Gating 판정 (정상, 결함, 판정보류)
Rule: AGENTS.md 인터페이스 기반 독립 모듈화 규정 준수
"""

import os
import sys
import pickle
from typing import Dict, Any, Optional

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import (
    extract_stiff_d1_features, extract_stiff_d1_v2_features,
    FEATURE_NAMES_38_STIFF_D1, FEATURE_NAMES_32_STIFF_D1, VIEW_ALIAS_MAP
)

DEFAULT_BUNDLE_NAME = "sd3_v2_bundle.pkl"
WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", DEFAULT_BUNDLE_NAME)
_MODEL_CACHE: Optional[Dict[str, Any]] = None


def load_model_bundle(weights_path: str = WEIGHTS_PATH) -> Dict[str, Any]:
    """가중치 번들 캐싱 로드 (v2 우선, v1 fallback)"""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not os.path.exists(weights_path):
        fallback_v1 = os.path.join(CURRENT_DIR, "weights", "sd3_v1_bundle.pkl")
        fallback_legacy = os.path.join(CURRENT_DIR, "weights", "stiff_deadlift_d1_best.pkl")
        if os.path.exists(fallback_v1):
            weights_path = fallback_v1
        elif os.path.exists(fallback_legacy):
            weights_path = fallback_legacy
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
    with open(weights_path, 'rb') as fp:
        _MODEL_CACHE = pickle.load(fp)
    return _MODEL_CACHE


def infer_stiff_d1_sequence(kps_64f: np.ndarray,
                            view_name: str = "view4",
                            uncertainty_margin: Optional[float] = None) -> Dict[str, Any]:
    """
    (64, 33, 3) 3차 스플라인 보간 관절 시퀀스 단일 입력에 대해 3-State 최종 결함 판정 수행

    Parameters:
    -----------
    kps_64f : np.ndarray
        (64, 33, 3) MediaPipe 3D 관절 좌표 배열
    view_name : str
        카메라 시점 코드 ('view1' ~ 'view5')
    uncertainty_margin : Optional[float]
        신뢰도 마진 대역 (기본값: 모델 번들 내 최적 tau 대역 적용)

    Returns:
    --------
    Dict[str, Any]:
        defect_prob: 결함 발생 확률 (0.0 ~ 1.0)
        prediction: 0 (정상), 1 (결함), -1 (판정 보류)
        is_decided: 확정 판정 여부 (True: 확정, False: 보류)
        decision_label: 한글 판정 라벨
        view_alias: 공식 시점 명칭 (예: '시점 D (좌측 전방 45°)')
        features_dim: 38
    """
    if kps_64f.shape != (64, 33, 3):
        raise ValueError(f"Input shape must be (64, 33, 3), got {kps_64f.shape}")

    # 1. 모델 번들 로드
    bundle = load_model_bundle()
    f_dim = bundle.get('feature_dim', 32)

    # 2. 도메인 피처 추출 (v2 32D 또는 v1 38D 적응형 추출)
    if f_dim == 32:
        f_vec = extract_stiff_d1_v2_features(kps_64f, view_name=view_name)
    else:
        f_vec = extract_stiff_d1_features(kps_64f, view_name=view_name)
    X = f_vec.reshape(1, -1)

    clf_lgb = bundle['lgb_model']
    clf_cb = bundle['cb_model']
    w_lgb, w_cb = bundle.get('weights', (0.55, 0.45))

    p_lgb = clf_lgb.predict_proba(X)[0, 1]
    p_cb = clf_cb.predict_proba(X)[0, 1]
    prob = float(w_lgb * p_lgb + w_cb * p_cb)

    # 3. 3-State 판정 임계값 획득
    opt_3state = bundle.get('optimal_3state', {})
    tau_clean = opt_3state.get('tau_clean', 0.44)
    tau_defect = opt_3state.get('tau_defect', 0.56)

    if uncertainty_margin is not None:
        tau_clean = 0.50 - uncertainty_margin
        tau_defect = 0.50 + uncertainty_margin

    # 4. 3-State 분류 로직
    if prob <= tau_clean:
        pred = 0
        is_decided = True
        label = "정상 (바벨 밀착 궤적 양호)"
    elif prob >= tau_defect:
        pred = 1
        is_decided = True
        label = "결함 감지 (바벨이 몸에서 과도하게 멀어짐)"
    else:
        pred = -1
        is_decided = False
        label = "판정 보류 (신뢰도 경계 구간, 재촬영 권장)"

    view_alias = VIEW_ALIAS_MAP.get(view_name, view_name)

    return {
        'defect_prob': round(prob, 4),
        'prediction': pred,
        'is_decided': is_decided,
        'decision_label': label,
        'view_name': view_name,
        'view_alias': view_alias,
        'tau_clean': round(tau_clean, 3),
        'tau_defect': round(tau_defect, 3),
        'features_dim': len(f_vec)
    }
