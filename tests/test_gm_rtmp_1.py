# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests for Good Morning RTMPose Optimized Models (gm1_rtmp_1, gm2_rtmp_1, gm3_rtmp_1)
Test File: tests/test_gm_rtmp_1.py
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = r"D:\src\models\second_models\babel_good_morning"
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from gm1_rtmp_1 import BabelGoodMorningGM1RTMP1, GoodMorningGM1RTMP1
from gm2_rtmp_1 import BabelGoodMorningGM2RTMP1, GoodMorningGM2RTMP1
from gm3_rtmp_1 import BabelGoodMorningGM3RTMP1, GoodMorningGM3RTMP1

CACHE_PATH = r"D:\Data\피처_데이터\good_morning_rtmpose_features_cache.npz"
PARAMS_JSON = os.path.join(MODEL_DIR, "good_morning_submodels_rtmp_1_params.json")
RESULTS_JSON = os.path.join(MODEL_DIR, "good_morning_rtmp_1_optimization_results.json")
DOCX_PATH = r"D:\자주프 문서\7주차\vidio-health-trainer_바벨굿모닝_세부오류3종_RTMPose_최적화_gm1_gm2_gm3_rtmp_1_공식감사보고서.docx"
MD_PATH = r"D:\자주프 문서\7주차\vidio-health-trainer_바벨굿모닝_세부오류3종_RTMPose_최적화_gm1_gm2_gm3_rtmp_1_공식감사보고서.md"

VIEW_LIST = ['view1', 'view2', 'view3', 'view4', 'view5']


def test_cache_integrity():
    assert os.path.exists(CACHE_PATH), f"Cache not found: {CACHE_PATH}"
    c = np.load(CACHE_PATH, allow_pickle=True)
    assert c['X_c1'].shape == (1020, 13)
    assert c['X_c2'].shape == (1020, 15)
    assert c['X_c3'].shape == (1020, 45)
    assert np.isnan(c['X_c1']).sum() == 0
    assert np.isnan(c['X_c2']).sum() == 0
    assert np.isnan(c['X_c3']).sum() == 0
    assert len(c['subjects']) == 1020
    assert len(np.unique(c['subjects'])) == 9
    assert len(np.unique(c['views'])) == 5


def test_model_instantiation_and_prediction():
    # 1. gm1
    m1 = BabelGoodMorningGM1RTMP1()
    assert m1.DEFECT_CODE == 'c1'
    assert m1.FEATURE_DIM == 13
    assert m1.is_loaded is True
    dummy_kps26 = np.zeros((64, 26, 3), dtype=np.float32)
    feat1 = m1.extract_features(dummy_kps26)
    assert feat1.shape == (13,)
    res1 = m1.predict_sequence(dummy_kps26, view_name='view3')
    assert 'raw_probability' in res1
    assert 'calibrated_probability' in res1
    assert res1['state_code'] in [0, 1, 2]

    # 2. gm2
    m2 = BabelGoodMorningGM2RTMP1()
    assert m2.DEFECT_CODE == 'c2'
    assert m2.FEATURE_DIM == 15
    assert m2.is_loaded is True
    feat2 = m2.extract_features(dummy_kps26)
    assert feat2.shape == (15,)
    res2 = m2.predict_sequence(dummy_kps26, view_name='view5')
    assert 'calibrated_probability' in res2
    assert res2['state_code'] in [0, 1, 2]

    # 3. gm3
    m3 = BabelGoodMorningGM3RTMP1()
    assert m3.DEFECT_CODE == 'c3'
    assert m3.FEATURE_DIM == 45
    assert m3.is_loaded is True
    dummy_spine = np.zeros((64, 9, 3), dtype=np.float32)
    feat3 = m3.extract_features(dummy_kps26, dummy_spine)
    assert feat3.shape == (45,)
    res3 = m3.predict_sequence(dummy_kps26, view_name='view1', spine_9_64f=dummy_spine)
    assert 'calibrated_probability' in res3
    assert res3['state_code'] in [0, 1, 2]


def test_calibration_params_json():
    assert os.path.exists(PARAMS_JSON), f"Params JSON not found: {PARAMS_JSON}"
    with open(PARAMS_JSON, 'r', encoding='utf-8') as fp:
        params = json.load(fp)

    for c in ['c1', 'c2', 'c3']:
        assert c in params
        p = params[c]
        assert 'global_temperature' in p and p['global_temperature'] > 0
        assert 'view_temperatures' in p
        assert 'global_thresholds' in p
        assert 'view_thresholds' in p
        for vn in VIEW_LIST:
            assert vn in p['view_temperatures']
            assert vn in p['view_thresholds']
            th = p['view_thresholds'][vn]
            assert 0.0 < th['tau_norm'] < th['tau_def'] < 1.0


def test_optimization_results_json():
    assert os.path.exists(RESULTS_JSON), f"Results JSON not found: {RESULTS_JSON}"
    with open(RESULTS_JSON, 'r', encoding='utf-8') as fp:
        res = json.load(fp)

    assert res['project'] == 'vidio-health-trainer'
    assert res['exercise'] == '바벨 굿모닝 (09_굿모닝)'
    assert res['total_sequences'] == 1020
    assert res['subjects_count'] == 9
    assert res['views_count'] == 5

    sub = res['submodels']
    for c in ['c1', 'c2', 'c3']:
        assert c in sub
        post = sub[c]['post_opt']
        assert post['coverage'] >= 90.0, f"{c} coverage {post['coverage']} < 90.0%"
        assert post['retake'] <= 10.0, f"{c} retake {post['retake']} > 10.0%"
        assert post['sel_f1'] >= 70.0
        assert post['roc_auc'] >= 75.0

    assert 'session_emr' in sub
    assert sub['session_emr']['emr_post_opt'] >= sub['session_emr']['emr_pre_opt']


def test_reports_generated():
    assert os.path.exists(DOCX_PATH), f"Word report not found: {DOCX_PATH}"
    assert os.path.getsize(DOCX_PATH) > 10000, "Word report too small"
    assert os.path.exists(MD_PATH), f"Markdown report not found: {MD_PATH}"
    assert os.path.getsize(MD_PATH) > 1000, "Markdown report too small"


if __name__ == '__main__':
    print("[TEST 1] Testing cache integrity...")
    test_cache_integrity()
    print(" -> PASSED")

    print("[TEST 2] Testing calibration params JSON...")
    test_calibration_params_json()
    print(" -> PASSED")

    print("[TEST 3] Testing optimization results JSON...")
    test_optimization_results_json()
    print(" -> PASSED")

    print("[TEST 4] Testing model instantiation and prediction...")
    test_model_instantiation_and_prediction()
    print(" -> PASSED")

    print("[TEST 5] Testing reports exist...")
    test_reports_generated()
    print(" -> PASSED")

    print("\nALL HARNESS TESTS PASSED SUCCESSFULLY!")
