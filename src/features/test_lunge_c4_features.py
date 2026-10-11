# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_lunge_c4_features.py
Purpose: 바벨 런지 c4 28차원 피처 추출기 단위 테스트 및 무결성 검증
"""

import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, "D:/")

from src.features.lunge_c4_features import (
    extract_lunge_c4_features,
    LUNGE_C4_FEATURE_NAMES
)

def test_feature_extraction():
    print("=" * 60)
    print("  [Harness Test] lunge_c4_features 무결성 단위 테스트")
    print("=" * 60)

    # 1. 피처 명칭 차원 검증
    assert len(LUNGE_C4_FEATURE_NAMES) == 28, f"Feature count must be 28, got {len(LUNGE_C4_FEATURE_NAMES)}"
    print(f"[✓] 피처 목록 28개 규격 확인 완료")

    # 2. 합성 더미 데이터 (64, 33, 3) 테스트
    np.random.seed(42)
    dummy_seq = np.random.randn(64, 33, 3).astype(np.float32)
    feats = extract_lunge_c4_features(dummy_seq)
    
    assert feats.shape == (28,), f"Output shape must be (28,), got {feats.shape}"
    assert feats.dtype == np.float32, f"Output dtype must be float32, got {feats.dtype}"
    assert not np.isnan(feats).any(), "Output contains NaN values!"
    assert not np.isinf(feats).any(), "Output contains Inf values!"
    print(f"[✓] 더미 텐서 (64, 33, 3) 28차원 정상 추출 확인 (NaN/Inf 0%)")

    # 3. 실제 추출된 npz 샘플 파일 로드 테스트
    import glob, os
    sample_files = glob.glob(r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지\*.npz")
    if sample_files:
        data = np.load(sample_files[0])
        real_seq = data['kps_33_3d_64f']
        real_feats = extract_lunge_c4_features(real_seq)
        assert real_feats.shape == (28,)
        assert not np.isnan(real_feats).any()
        assert not np.isinf(real_feats).any()
        print(f"[✓] 실제 전처리 npz 샘플({os.path.basename(sample_files[0])}) 28D 추출 성공!")
        print("\n=== 상위 10개 피처값 샘플 ===")
        for name, val in zip(LUNGE_C4_FEATURE_NAMES[:10], real_feats[:10]):
            print(f"  {name:30s}: {val:.4f}")
    
    print("\n[★ APPROVED] lunge_c4_features 단위 테스트 100% 통과!\n")

if __name__ == '__main__':
    test_feature_extraction()
