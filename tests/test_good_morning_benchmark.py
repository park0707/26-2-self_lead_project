# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit tests for Good Morning MediaPipe vs RTMPose Benchmark
Test File: tests/test_good_morning_benchmark.py
"""

import os
import json
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

BENCHMARK_JSON = r"D:\src\models\second_models\babel_good_morning\good_morning_mediapipe_vs_rtmpose_benchmark_results.json"
CACHE_PATH = r"D:\Data\피처_데이터\good_morning_rtmpose_features_cache.npz"
DOCX_PATH = r"D:\자주프 문서\7주차\vidio-health-trainer_바벨굿모닝_세부오류3종_MediaPipe_vs_RTMPose_성능비교_공식감사보고서.docx"
MD_PATH = r"D:\자주프 문서\7주차\vidio-health-trainer_바벨굿모닝_세부오류3종_MediaPipe_vs_RTMPose_성능비교_공식감사보고서.md"

def test_cache_integrity():
    assert os.path.exists(CACHE_PATH), f"Cache file not found: {CACHE_PATH}"
    c = np.load(CACHE_PATH, allow_pickle=True)
    
    assert c['X_c1'].shape == (1020, 13)
    assert c['X_c2'].shape == (1020, 15)
    assert c['X_c3'].shape == (1020, 45)
    
    assert np.isnan(c['X_c1']).sum() == 0
    assert np.isnan(c['X_c2']).sum() == 0
    assert np.isnan(c['X_c3']).sum() == 0
    
    assert np.isinf(c['X_c1']).sum() == 0
    assert np.isinf(c['X_c2']).sum() == 0
    assert np.isinf(c['X_c3']).sum() == 0
    
    # Ground truth labels
    assert np.sum(c['y_c1'] == 0) == 510 and np.sum(c['y_c1'] == 1) == 510
    assert np.sum(c['y_c2'] == 0) == 515 and np.sum(c['y_c2'] == 1) == 505
    assert np.sum(c['y_c3'] == 0) == 505 and np.sum(c['y_c3'] == 1) == 515

def test_benchmark_results_json():
    assert os.path.exists(BENCHMARK_JSON), f"Benchmark JSON not found: {BENCHMARK_JSON}"
    with open(BENCHMARK_JSON, 'r', encoding='utf-8') as f:
        d = json.load(f)
        
    assert d['project'] == 'vidio-health-trainer'
    assert d['exercise'] == '바벨 굿모닝 (09_굿모닝)'
    assert d['total_sequences'] == 1020
    assert d['subjects_count'] == 9
    assert d['views_count'] == 5
    
    for model_key in ['c1', 'c2', 'c3']:
        assert model_key in d['mediapipe_metrics']
        assert model_key in d['rtmpose_metrics']
        
        rtm = d['rtmpose_metrics'][model_key]
        assert 'forced_f1' in rtm and rtm['forced_f1'] > 50.0
        assert 'sel_f1' in rtm and rtm['sel_f1'] > 60.0
        assert 'retake' in rtm and rtm['retake'] <= 10.0
        assert 'coverage' in rtm and rtm['coverage'] >= 90.0
        
        # 5 views check
        for vn in ['view1', 'view2', 'view3', 'view4', 'view5']:
            assert vn in rtm['views']
            assert rtm['views'][vn]['sel_f1'] > 0.0

def test_benchmark_reports_exist():
    assert os.path.exists(DOCX_PATH), f"Word report not found: {DOCX_PATH}"
    assert os.path.getsize(DOCX_PATH) > 10000
    
    assert os.path.exists(MD_PATH), f"Markdown report not found: {MD_PATH}"
    assert os.path.getsize(MD_PATH) > 1000

if __name__ == '__main__':
    print("[TEST 1] Testing cache integrity...")
    test_cache_integrity()
    print(" -> PASSED")
    
    print("[TEST 2] Testing benchmark results JSON...")
    test_benchmark_results_json()
    print(" -> PASSED")
    
    print("[TEST 3] Testing benchmark reports exist...")
    test_benchmark_reports_exist()
    print(" -> PASSED")
    
    print("\nALL HARNESS TESTS PASSED SUCCESSFULLY!")
