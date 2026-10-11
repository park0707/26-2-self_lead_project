# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit test for side_lunge c1 sl1_v10 pipeline (55D)
"""
import os
import sys
import pickle
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_feature_vector_v11,
    extract_side_lunge_c1_features_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)

def test_feature_extractor():
    print("[TEST 1] Testing 55D Feature Extractor...")
    assert len(SIDE_LUNGE_C1_FEATURE_NAMES_V11) == 55, f"Expected 55 features, got {len(SIDE_LUNGE_C1_FEATURE_NAMES_V11)}"
    
    # Random dummy input
    dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
    feat_vec = extract_side_lunge_c1_feature_vector_v11(dummy_kps)
    
    assert feat_vec.shape == (55,), f"Expected shape (55,), got {feat_vec.shape}"
    assert not np.isnan(feat_vec).any(), "Found NaN in feature vector"
    assert not np.isinf(feat_vec).any(), "Found Inf in feature vector"
    print("  -> Feature extractor test PASSED!")

def test_model_bundle_if_ready():
    bundle_path = r"D:\src\models\second_models\side_lunge\sl1_v10_bundle.pkl"
    if not os.path.exists(bundle_path):
        print(f"[TEST 2] Model bundle not yet created: {bundle_path} (Skipping until trained)")
        return
    print("[TEST 2] Testing sl1_v10 Model Bundle...")
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
    assert bundle['model_name'] == 'sl1_v10'
    assert bundle['features_dimension'] == 55
    assert len(bundle['models_lgb']) == 5
    assert len(bundle['models_cat']) == 5
    assert len(bundle['models_xgb']) == 5
    
    # Test single-sample inference
    dummy_feat = np.random.randn(1, 55).astype(np.float32)
    w = bundle['ensemble_weights']
    p_lgb = np.mean([m.predict(dummy_feat)[0] for m in bundle['models_lgb']])
    p_cat = np.mean([m.predict_proba(dummy_feat)[0, 1] for m in bundle['models_cat']])
    
    import xgboost as xgb
    dmat = xgb.DMatrix(dummy_feat)
    p_xgb = np.mean([m.predict(dmat)[0] for m in bundle['models_xgb']])
    
    final_prob = w[0] * p_lgb + w[1] * p_cat + w[2] * p_xgb
    assert 0.0 <= final_prob <= 1.0, f"Invalid probability: {final_prob}"
    print(f"  -> Single inference test PASSED! Probability: {final_prob:.4f}")

if __name__ == '__main__':
    test_feature_extractor()
    test_model_bundle_if_ready()
    print("[ALL TESTS PASSED]")
