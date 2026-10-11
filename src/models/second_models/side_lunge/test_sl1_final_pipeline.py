# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit & Integration Test: sl1_final Official Champion Pipeline (Hybrid 110D MediaPipe + RTMPose)
"""

import os
import sys
import pickle
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BUNDLE_PATH = r"D:\src\models\second_models\side_lunge\sl1_final_bundle.pkl"
CACHE_PATH = r"D:\Data\피처_데이터\sl1_hybrid_features.npz"


def test_sl1_final():
    print("=" * 70)
    print("[*] sl1_final 공식 최종 챔피언 파이프라인 무결성 검증 테스트 시작 (Hybrid 110D)")
    print("=" * 70)

    # 1. Cache & Feature Integrity
    assert os.path.exists(CACHE_PATH), f"캐시 부재: {CACHE_PATH}"
    data = np.load(CACHE_PATH)
    X_110d = data['X_hybrid_110d']
    assert X_110d.shape == (4130, 110)
    assert np.isnan(X_110d).sum() == 0
    assert np.isinf(X_110d).sum() == 0
    print("[+] 110D 하이브리드 피처 4,130개 시퀀스 무결성(0 NaN/Inf) 통과")

    # 2. Bundle Load Test
    assert os.path.exists(BUNDLE_PATH), f"번들 부재: {BUNDLE_PATH}"
    with open(BUNDLE_PATH, 'rb') as f:
        bundle = pickle.load(f)

    assert bundle['features_dimension'] == 110
    assert len(bundle['models_lgb']) == 5
    assert len(bundle['models_cat']) == 5
    assert len(bundle['models_xgb']) == 5
    assert 0.05 < bundle['temperature'] < 5.0
    print(f"[+] 번들 로드 통과: {bundle['features_dimension']}D, 온도: {bundle['temperature']:.4f}")

    # 3. Dual Interface Test (Dictionary & Object)
    sample_vec = X_110d[0]
    p_prob = bundle.predict_proba(sample_vec)
    assert len(p_prob) == 1
    assert 0.0 <= p_prob[0] <= 1.0

    res_3s = bundle.predict_3state(sample_vec, view_code='view3')
    assert res_3s['state'] in [0, 1, 2]
    assert 'state_label' in res_3s
    assert 'feedback' in res_3s
    assert res_3s['view_code'] == 'view3'

    print(f"[+] 단일 시퀀스 3-State 추론 성공: 확률={res_3s['calibrated_prob']} -> {res_3s['state_label']}")
    print(f"[+] 피드백: {res_3s['feedback']}")
    print("[+] sl1_final 공식 최종 챔피언 파이프라인 무결성 검증 100% 성공!")


if __name__ == '__main__':
    test_sl1_final()
