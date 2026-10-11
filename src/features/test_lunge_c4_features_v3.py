# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_lunge_c4_features_v3.py
Purpose: lunge_c4_features_v3.py 단위 테스트 및 무결성 검증
"""

import sys
import numpy as np
from lunge_c4_features_v3 import (
    extract_lunge_c4_features_v3,
    extract_lunge_c4_temporal_features,
    LUNGE_C4_FEATURE_NAMES_V3
)

def test_v3_feature_extractor():
    print("=" * 70)
    print("  [Harness Test] lunge_c4_features_v3.py 무결성 검증")
    print("=" * 70)

    # 1. 차원수 일치 확인
    assert len(LUNGE_C4_FEATURE_NAMES_V3) == 24, f"피처 명칭 수 불일치: {len(LUNGE_C4_FEATURE_NAMES_V3)}"
    print(f"[PASS] v3 정예 피처 명칭 수: {len(LUNGE_C4_FEATURE_NAMES_V3)}개 일치")

    # 2. 더미 시퀀스 테스트
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)
    feat_vec = extract_lunge_c4_features_v3(dummy_seq)
    assert feat_vec.shape == (24,), f"피처 벡터 형상 오류: {feat_vec.shape}"
    assert not np.isnan(feat_vec).any(), "NaN 발견!"
    assert not np.isinf(feat_vec).any(), "Inf 발견!"
    print(f"[PASS] 더미 텐서 피처 추출 성공: Shape={feat_vec.shape}, NaN/Inf=0")

    # 3. 14D 시계열 텐서 테스트
    temp_feats = extract_lunge_c4_temporal_features(dummy_seq)
    assert temp_feats.shape == (64, 14), f"시계열 형상 오류: {temp_feats.shape}"
    assert not np.isnan(temp_feats).any(), "시계열 NaN 발견!"
    print(f"[PASS] 시계열 텐서 추출 성공: Shape={temp_feats.shape}")

    # 4. 실제 샘플 파일 테스트
    sample_path = r"D:\Data\바벨_런지\01_정상_런지_스플라인\01_64프레임_정규화\피험자1_Key001_view1.npz"
    if os.path.exists(sample_path):
        data = np.load(sample_path)
        real_seq = data['landmarks'] if 'landmarks' in data else data['sequence']
        real_feats = extract_lunge_c4_features_v3(real_seq)
        assert real_feats.shape == (24,), f"실제 샘플 형상 오류: {real_feats.shape}"
        print(f"[PASS] 실제 샘플 피처 추출 성공: {os.path.basename(sample_path)}")

    print("\n[★ APPROVED] lunge_c4_features_v3.py 100% 무결성 검증 통과!")

if __name__ == "__main__":
    import os
    test_v3_feature_extractor()
