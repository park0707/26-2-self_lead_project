# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit & Integration Test: sl1_v11 Pipeline (Option 2: CatBoost heavy [0.15, 0.65, 0.20] + 3-State [0.35, 0.65])
"""

import os
import sys
import glob
import pickle
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_feature_vector_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)

BUNDLE_PATH = r"D:\src\models\second_models\side_lunge\sl1_v11_bundle.pkl"
DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"


def test_sl1_v11():
    print("=" * 70)
    print("[*] sl1_v11 파이프라인 무결성 검증 테스트 시작 (55D Option 2)")
    print("=" * 70)

    # 1. Feature Dimension
    assert len(SIDE_LUNGE_C1_FEATURE_NAMES_V11) == 55
    print("[+] 55D 도메인 피처 명칭 일관성 100% 통과")

    # 2. Sample Extraction
    sample = glob.glob(os.path.join(DATA_DIR, "*.npz"))[0]
    data = np.load(sample)
    kps = data['kps_33_3d_64f']
    vis = data['vis_64f'] if 'vis_64f' in data else None
    vec = extract_side_lunge_c1_feature_vector_v11(kps, vis)
    assert vec.shape == (55,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
    print("[+] 단일 시퀀스 55D 피처 추출 및 무결성(0 NaN/Inf) 통과")

    # 3. Bundle Inference Test
    with open(BUNDLE_PATH, 'rb') as f:
        bundle = pickle.load(f)

    assert bundle['features_dimension'] == 55
    assert len(bundle['models_lgb']) == 5
    assert len(bundle['models_cat']) == 5
    assert len(bundle['models_xgb']) == 5
    assert bundle['ensemble_weights'] == [0.15, 0.65, 0.20]

    import xgboost as xgb
    vec_2d = vec.reshape(1, -1)
    w = bundle['ensemble_weights']
    p_lgb = np.mean([m.predict(vec_2d)[0] for m in bundle['models_lgb']])
    p_cat = np.mean([m.predict_proba(vec_2d)[0, 1] for m in bundle['models_cat']])
    p_xgb = np.mean([m.predict(xgb.DMatrix(vec_2d))[0] for m in bundle['models_xgb']])
    p_ens = float(w[0] * p_lgb + w[1] * p_cat + w[2] * p_xgb)

    # Check 3-state logic
    def_th = bundle['default_thresholds']
    if p_ens < def_th['tau_clean']:
        state = 'Clean'
    elif p_ens >= def_th['tau_defect']:
        state = 'Defect'
    else:
        state = 'Retake'

    print(f"[+] 앙상블 추론 테스트 완료: 확률={p_ens:.4f} -> 3-State 판정결과={state}")
    print("[+] sl1_v11 55D 배포 번들 무결성 검증 100% 성공!")


if __name__ == '__main__':
    test_sl1_v11()
