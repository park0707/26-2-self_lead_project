# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 3: sd3 (바벨 궤적과 몸 밀착 위반)
Version: v4.0 (Official Production Standard)
Script: test_sd3_v4_classifier.py
Purpose: SD3V4Classifier 및 SD3Classifier 통합 추론, 엣지 케이스 방어, 5대 시점 호환성 단위 테스트
"""

import os
import sys
import pickle
import numpy as np

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd3_v4_classifier import SD3V4Classifier
from sd3_classifier import SD3Classifier
from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import VIEW_ALIAS_MAP

CACHE_V3_PKL = os.path.join(CURRENT_DIR, "stiff_d1_v3_dataset_cache.pkl")


def test_sd3_v4_and_alias_init():
    clf_v4 = SD3V4Classifier()
    assert clf_v4.model_name == 'sd3_v4'
    assert clf_v4.coverage >= 90.0
    assert clf_v4.abstain_rate <= 10.0
    assert len(clf_v4.feature_names) == 36

    clf_alias = SD3Classifier()
    assert clf_alias.model_name == 'sd3_v4'
    print("[PASS] test_sd3_v4_and_alias_init passed successfully.")


def test_sd3_v4_shapes_and_edge_cases():
    clf = SD3Classifier()

    # 1. 3D (64, 33, 3) All Zeros
    zeros_3d = np.zeros((64, 33, 3), dtype=np.float32)
    diag_zeros = clf.diagnose(zeros_3d, view_name="view4")
    assert 'probability' in diag_zeros
    assert 0.0 <= diag_zeros['probability'] <= 1.0
    assert diag_zeros['state_3state'] in [0, 1, 2]

    # 2. 2D (64, 99)
    zeros_2d_99 = np.zeros((64, 99), dtype=np.float32)
    p_99 = clf.predict_proba(zeros_2d_99, view_name="view2")
    assert 0.0 <= p_99 <= 1.0

    # 3. 2D (64, 98) with padding
    zeros_2d_98 = np.zeros((64, 98), dtype=np.float32)
    p_98 = clf.predict_proba(zeros_2d_98, view_name="view3")
    assert 0.0 <= p_98 <= 1.0

    # 4. NaN / Inf robustness
    nan_tensor = np.full((64, 33, 3), np.nan, dtype=np.float32)
    p_nan = clf.predict_proba(nan_tensor, view_name="view1")
    assert not np.isnan(p_nan)
    assert 0.0 <= p_nan <= 1.0

    print("[PASS] test_sd3_v4_shapes_and_edge_cases passed successfully.")


def test_sd3_v4_all_5_views():
    clf = SD3Classifier()
    dummy = np.random.randn(64, 33, 3).astype(np.float32) * 0.1

    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        diag = clf.diagnose(dummy, view_name=v_code)
        assert diag['view_name'] == v_code
        assert diag['view_alias'] == v_alias
        assert 0.0 <= diag['probability'] <= 1.0
        assert diag['state_3state'] in [0, 1, 2]
        assert diag['thresholds']['abstain_rate'] <= 10.0

    print("[PASS] test_sd3_v4_all_5_views passed successfully.")


def test_sd3_v4_real_samples():
    if not os.path.exists(CACHE_V3_PKL):
        print("[SKIP] Cache not found")
        return

    clf = SD3Classifier()
    with open(CACHE_V3_PKL, 'rb') as fp:
        cache = pickle.load(fp)

    df = cache['df']
    sample_indices = [0, 100, 250, 500, 750]
    for idx in sample_indices:
        row = df.iloc[idx]
        fpath = row['path']
        view = row['view']
        with np.load(fpath, allow_pickle=True) as d:
            kps_64f = d['kps_33_3d_64f'].astype(np.float32)

        diag = clf.diagnose(kps_64f, view_name=view)
        assert 0.0 <= diag['probability'] <= 1.0
        assert diag['state_3state'] in [0, 1, 2]
        print(f"  Sample {idx} ({row['subj']}, {row['key']}, {view}): P={diag['probability']:.3f}, State={diag['state_label']}")

    print("[PASS] test_sd3_v4_real_samples passed successfully.")


if __name__ == '__main__':
    test_sd3_v4_and_alias_init()
    test_sd3_v4_shapes_and_edge_cases()
    test_sd3_v4_all_5_views()
    test_sd3_v4_real_samples()
    print("\n[+] All SD3V4Classifier and SD3Classifier unit tests PASSED successfully!")
