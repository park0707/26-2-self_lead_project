# -*- coding: utf-8 -*-
"""
Harness Engineering Unit Test: test_side_lunge_c1_v6_features.py
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v6 import (
    extract_side_lunge_c1_features_v6,
    extract_side_lunge_c1_feature_vector_v6,
    SIDE_LUNGE_C1_FEATURE_NAMES_V6
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"

def run_test():
    print("[*] side_lunge_c1_features_v6 36D 피처 하네스 단위 테스트 시작...")
    assert len(SIDE_LUNGE_C1_FEATURE_NAMES_V6) == 36, f"피처 수 불일치: {len(SIDE_LUNGE_C1_FEATURE_NAMES_V6)}"

    # 1. 합성 텐서 무결성 테스트
    dummy_kps = np.random.randn(64, 33, 3).astype(np.float32)
    feat_dict = extract_side_lunge_c1_features_v6(dummy_kps)
    feat_vec = extract_side_lunge_c1_feature_vector_v6(dummy_kps)

    assert feat_vec.shape == (36,), f"벡터 Shape 오류: {feat_vec.shape}"
    assert not np.isnan(feat_vec).any(), "합성 텐서에서 NaN 검출!"
    assert not np.isinf(feat_vec).any(), "합성 텐서에서 Inf 검출!"
    print("  [+] 합성 더미 텐서 테스트 통과: Shape=(36,), NaN/Inf 0건")

    # 2. 실제 데이터셋 대표 10개 시퀀스 전수 검증
    sample_npzs = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))[:10]
    assert len(sample_npzs) > 0, "NPZ 파일이 존재하지 않습니다."

    for idx, npz_p in enumerate(sample_npzs):
        data = np.load(npz_p)
        kps_33_3d = data['kps_33_3d_64f']
        vis = data['vis_64f'] if 'vis_64f' in data else None
        
        vec = extract_side_lunge_c1_feature_vector_v6(kps_33_3d, vis)
        assert vec.shape == (36,), f"실제 데이터 {npz_p} Shape 오류: {vec.shape}"
        assert not np.isnan(vec).any(), f"실제 데이터 {npz_p} NaN 검출!"
        assert not np.isinf(vec).any(), f"실제 데이터 {npz_p} Inf 검출!"

    print(f"  [+] 실제 전처리 NPZ 10개 샘플 전수 검증 통과 (Shape: 36, NaN/Inf 0건)")
    print("[+] 모든 하네스 단위 테스트 100% 통과 (APPROVED)!")

if __name__ == '__main__':
    run_test()
