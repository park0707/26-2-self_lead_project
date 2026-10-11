# -*- coding: utf-8 -*-
"""
단위 테스트: test_side_lunge_c5_v7_features.py
검증 대상: side_lunge_c5_features_v7.py
- 피처 벡터 차원 규격 검증 (42D)
- 결측치 (NaN/Inf) 0% 검증
- 실제 전처리 npz 샘플 기반 동작 무결성 검증
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c5_features_v7 import (
    extract_side_lunge_c5_v7_feature_vector,
    SIDE_LUNGE_C5_V7_FEATURE_NAMES
)

def test_feature_names_length():
    print(f"[*] 피처 명칭 수 검증: {len(SIDE_LUNGE_C5_V7_FEATURE_NAMES)}개")
    assert len(SIDE_LUNGE_C5_V7_FEATURE_NAMES) == 42, f"피처 명칭 길이 불일치: {len(SIDE_LUNGE_C5_V7_FEATURE_NAMES)}"
    print("    -> [통과] 42차원 명칭 일치")

def test_synthetic_input():
    print("[*] 합성 텐서 무결성 테스트 (64, 33, 3)...")
    np.random.seed(42)
    fake_seq = np.random.randn(64, 33, 3).astype(np.float32)
    # Give some realistic keypoint offsets
    fake_seq[:, 23, 0] = -0.2 # Left hip
    fake_seq[:, 24, 0] = 0.2  # Right hip
    fake_seq[:, 25, 0] = -0.5 # Left knee
    fake_seq[:, 26, 0] = 0.5  # Right knee
    fake_seq[:, 27, 0] = -0.6 # Left ankle
    fake_seq[:, 28, 0] = 0.6  # Right ankle
    fake_seq[:, 11, 0] = -0.25 # Left shoulder
    fake_seq[:, 12, 0] = 0.25  # Right shoulder

    feats = extract_side_lunge_c5_v7_feature_vector(fake_seq)
    assert feats.shape == (42,), f"출력 차원 불일치: {feats.shape}"
    assert not np.isnan(feats).any(), "합성 텐서에서 NaN 발생!"
    assert not np.isinf(feats).any(), "합성 텐서에서 Inf 발생!"
    print(f"    -> [통과] 출력 차원 {feats.shape}, NaN=0, Inf=0")

def test_real_npz_samples():
    print("[*] 실제 전처리 npz 샘플 테스트 (정상 및 결함 10건 전수 검증)...")
    npz_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    assert len(npz_files) > 0, "테스트할 npz 파일이 없습니다!"

    sample_files = npz_files[:10]
    for i, sf in enumerate(sample_files):
        data = np.load(sf)
        seq = data['kps_33_3d_64f']
        feats = extract_side_lunge_c5_v7_feature_vector(seq)
        assert feats.shape == (42,), f"파일 {sf} 출력 차원 불일치: {feats.shape}"
        assert not np.isnan(feats).any(), f"파일 {sf}에서 NaN 발생!"
        assert not np.isinf(feats).any(), f"파일 {sf}에서 Inf 발생!"

    print(f"    -> [통과] 10개 실제 샘플 전수 통과 (Shape=(42,), NaN=0, Inf=0)")

if __name__ == '__main__':
    test_feature_names_length()
    test_synthetic_input()
    test_real_npz_samples()
    print("\n[V] 모든 단위 테스트 100% 통과 (APPROVED)!")
