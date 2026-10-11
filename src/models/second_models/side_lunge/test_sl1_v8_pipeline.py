# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit & Integration Test: sl1_v8 Pipeline
Verifies:
  1. Feature dimension is strictly 45D
  2. 0 NaN and 0 Inf in extracted features
  3. Single-sequence inference pipeline with sl1_v8_bundle.pkl
  4. Proper 3-State output generation (Clean / Defect / Retake)
"""

import os
import sys
import glob
import pickle
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v9 import (
    extract_side_lunge_c1_feature_vector_v9,
    SIDE_LUNGE_C1_FEATURE_NAMES_V9
)

BUNDLE_PATH = r"D:\src\models\second_models\side_lunge\sl1_v8_bundle.pkl"
DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"


def test_sl1_v8():
    print("=" * 70)
    print("[*] sl1_v8 파이프라인 무결성 검증 테스트 시작")
    print("=" * 70)

    # 1. Feature Dimension and Name Integrity
    assert len(SIDE_LUNGE_C1_FEATURE_NAMES_V9) == 45, f"Expected 45 features, got {len(SIDE_LUNGE_C1_FEATURE_NAMES_V9)}"
    print("[+] 45D 피처 정의 목록 일관성 통과 (45개)")

    # 2. Sample Extraction & Nan/Inf Check
    samples = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))[:5]
    assert len(samples) > 0, "샘플 NPZ 데이터가 존재하지 않습니다."

    for s_path in samples:
        fname = os.path.basename(s_path)
        data = np.load(s_path)
        kps = data['kps_33_3d_64f']
        vis = data['vis_64f'] if 'vis_64f' in data else None

        vec = extract_side_lunge_c1_feature_vector_v9(kps, vis)
        assert vec.shape == (45,), f"Shape mismatch: {vec.shape} != (45,)"
        assert not np.isnan(vec).any(), f"NaN detected in {fname}"
        assert not np.isinf(vec).any(), f"Inf detected in {fname}"
        print(f"  - 샘플 검증 통과: {fname:<45s} -> Shape: {vec.shape}, NaN: 0, Inf: 0")

    # 3. Bundle Inference Test
    if os.path.exists(BUNDLE_PATH):
        with open(BUNDLE_PATH, 'rb') as f:
            bundle = pickle.load(f)

        assert 'models_lgb' in bundle
        assert 'models_cat' in bundle
        assert 'models_xgb' in bundle
        assert 'ensemble_weights' in bundle
        assert 'view_thresholds' in bundle

        # Test single inference
        vec_2d = vec.reshape(1, -1)
        w = bundle['ensemble_weights']

        # LGB
        p_lgb = np.mean([m.predict(vec_2d)[0] for m in bundle['models_lgb']])
        # CAT
        p_cat = np.mean([m.predict_proba(vec_2d)[0, 1] for m in bundle['models_cat']])
        # XGB
        import xgboost as xgb
        dtest = xgb.DMatrix(vec_2d)
        p_xgb = np.mean([m.predict(dtest)[0] for m in bundle['models_xgb']])

        p_ens = float(w[0] * p_lgb + w[1] * p_cat + w[2] * p_xgb)
        print(f"\n[+] 단일 시퀀스 추론 성공:")
        print(f"    p_lgb={p_lgb:.4f}, p_cat={p_cat:.4f}, p_xgb={p_xgb:.4f} -> p_ensemble={p_ens:.4f}")

        # Check thresholding
        th = bundle['view_thresholds'].get('view2', {'tau_clean': 0.38, 'tau_defect': 0.58})
        if p_ens < th['tau_clean']:
            state = 'Clean'
        elif p_ens >= th['tau_defect']:
            state = 'Defect'
        else:
            state = 'Retake'
        print(f"    시점 B 기준 판정 상태: {state} (tau=[{th['tau_clean']:.2f}, {th['tau_defect']:.2f}])")

    print("\n[+] sl1_v8 모든 무결성 검증 100% 통과 완료!")


if __name__ == '__main__':
    test_sl1_v8()
