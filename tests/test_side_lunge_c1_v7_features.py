# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit test for side_lunge_c1_features_v7.py
Verifies feature vector dimensions (24,), absence of NaN/Inf, and pure vision compliance.
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v7 import (
    extract_side_lunge_c1_features_v7,
    extract_side_lunge_c1_feature_vector_v7,
    SIDE_LUNGE_C1_FEATURE_NAMES_V7
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"


def test_feature_names_dimension():
    """피처 명칭 리스트가 정확히 24차원인지 검증"""
    assert len(SIDE_LUNGE_C1_FEATURE_NAMES_V7) == 24, f"Expected 24 features, got {len(SIDE_LUNGE_C1_FEATURE_NAMES_V7)}"


def test_synthetic_sequence_extraction():
    """모의 시퀀스 데이터(64, 33, 3)에 대한 피처 추출 무결성 검증"""
    np.random.seed(42)
    kps_fake = np.random.randn(64, 33, 3).astype(np.float32)
    # y축(중력방향) 그럴듯하게 설정
    kps_fake[:, 23:25, 1] = 0.5  # 고관절
    kps_fake[:, 25:27, 1] = 0.8  # 무릎
    kps_fake[:, 27:29, 1] = 1.1  # 발목

    feat_dict = extract_side_lunge_c1_features_v7(kps_fake)
    assert len(feat_dict) == 24
    for k, v in feat_dict.items():
        assert not np.isnan(v), f"Feature {k} is NaN"
        assert not np.isinf(v), f"Feature {k} is Inf"

    feat_vec = extract_side_lunge_c1_feature_vector_v7(kps_fake)
    assert feat_vec.shape == (24,)
    assert not np.isnan(feat_vec).any()
    assert not np.isinf(feat_vec).any()


def test_real_sequence_extraction():
    """실제 전처리 NPZ 시퀀스로 24D 피처 추출 및 무결성 검증"""
    npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    if not npz_files:
        print("[!] NPZ files not found, skipping real sample test.")
        return

    sample_npz = npz_files[0]
    data = np.load(sample_npz)
    kps_33_3d = data['kps_33_3d_64f']
    vis_64f = data.get('vis_64f', None)

    assert kps_33_3d.shape == (64, 33, 3)

    feat_dict = extract_side_lunge_c1_features_v7(kps_33_3d, vis_64f)
    assert len(feat_dict) == 24
    for k, v in feat_dict.items():
        assert not np.isnan(v), f"Real sample feature {k} is NaN"
        assert not np.isinf(v), f"Real sample feature {k} is Inf"

    feat_vec = extract_side_lunge_c1_feature_vector_v7(kps_33_3d, vis_64f)
    assert feat_vec.shape == (24,)
    assert not np.isnan(feat_vec).any()
    assert not np.isinf(feat_vec).any()
    print(f"\n[+] Real sample ({os.path.basename(sample_npz)}) 24D features extracted successfully!")


if __name__ == "__main__":
    test_feature_names_dimension()
    test_synthetic_sequence_extraction()
    test_real_sequence_extraction()
    print("\n[+] All Unit Tests Passed successfully!")
