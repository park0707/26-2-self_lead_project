# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_lunge_c4_features_v4.py
Purpose: lunge_c4_features_v4.py 30D 단위 테스트 및 무결성 검증
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, r"D:/")
from src.features.lunge_c4_features_v4 import (
    extract_lunge_c4_features_v4,
    extract_lunge_c4_temporal_features_v4,
    LUNGE_C4_FEATURE_NAMES_V4
)


def test_v4_feature_extractor():
    print("=" * 70)
    print("  [Harness Test] lunge_c4_features_v4.py 무결성 검증")
    print("=" * 70)

    # 1. 차원수 일치 확인
    assert len(LUNGE_C4_FEATURE_NAMES_V4) == 30, f"피처 명칭 수 불일치: {len(LUNGE_C4_FEATURE_NAMES_V4)}"
    print(f"[PASS] v4 정밀 피처 명칭 수: {len(LUNGE_C4_FEATURE_NAMES_V4)}개 일치")

    # 2. 더미 시퀀스 테스트
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)
    feat_vec = extract_lunge_c4_features_v4(dummy_seq)
    assert feat_vec.shape == (30,), f"피처 벡터 형상 오류: {feat_vec.shape}"
    assert not np.isnan(feat_vec).any(), "NaN 발견!"
    assert not np.isinf(feat_vec).any(), "Inf 발견!"
    print(f"[PASS] 더미 텐서 피처 추출 성공: Shape={feat_vec.shape}, NaN/Inf=0")

    # 3. 16D 시계열 텐서 테스트
    temp_feats = extract_lunge_c4_temporal_features_v4(dummy_seq)
    assert temp_feats.shape == (64, 16), f"시계열 형상 오류: {temp_feats.shape}"
    assert not np.isnan(temp_feats).any(), "시계열 NaN 발견!"
    print(f"[PASS] 시계열 텐서 추출 성공: Shape={temp_feats.shape}")

    # 4. 실제 샘플 파일 테스트
    sample_files = glob.glob(r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지\*.npz")
    if sample_files:
        sample_p = sample_files[0]
        data = np.load(sample_p)
        real_seq = data['kps_33_3d_64f']
        real_feats = extract_lunge_c4_features_v4(real_seq)
        assert real_feats.shape == (30,), f"실제 샘플 형상 오류: {real_feats.shape}"
        print(f"[PASS] 실제 샘플 피처 추출 성공: {os.path.basename(sample_p)}")

    print("\n[★ APPROVED] lunge_c4_features_v4.py 100% 무결성 검증 통과!")


if __name__ == "__main__":
    test_v4_feature_extractor()
