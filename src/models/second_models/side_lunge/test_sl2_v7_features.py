# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Test: test_sl2_v7_features.py
Purpose: v7 51D 피처 추출기 단위 테스트 및 무결성 검증
"""

import os
import glob
import numpy as np
import sys

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = r"D:\src\models\second_models\side_lunge"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from side_lunge_c2_v7_features import (
    FEATURE_NAMES_51D,
    V7_NEW_FEATURE_NAMES_12D,
    extract_side_lunge_c2_v7_features_51d
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"


def run_test():
    npz_files = glob.glob(os.path.join(DATA_DIR, "*view5*.npz"))
    if not npz_files:
        npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    assert len(npz_files) > 0, "테스트용 NPZ 파일을 찾을 수 없습니다."

    sample_npz = npz_files[0]
    print(f"[*] 테스트 후방 샘플: {sample_npz}")

    data = np.load(sample_npz)
    kps_seq = data['kps_33_3d_64f'][:, :, :3]
    assert kps_seq.shape == (64, 33, 3), f"Shape 불일치: {kps_seq.shape}"

    feat_vec = extract_side_lunge_c2_v7_features_51d(kps_seq, view_code="view5")

    print(f"[*] 피처 벡터 차원: {feat_vec.shape}")
    assert feat_vec.shape == (51,), f"51D 기대했으나 {feat_vec.shape} 반환됨"
    assert not np.isnan(feat_vec).any(), "피처에 NaN이 존재합니다!"
    assert not np.isinf(feat_vec).any(), "피처에 Inf가 존재합니다!"

    print("\n[*] v7 3대 혁신 12D 신규 피처 세부 검증:")
    for name in V7_NEW_FEATURE_NAMES_12D:
        idx = FEATURE_NAMES_51D.index(name)
        val = feat_vec[idx]
        print(f"  - [{idx+1:2d}] {name:40s}: {val:.4f}")

    print("\n[+] [HARNESS TEST PASSED] v7 51D 피처 추출기 단위 테스트 100% 통과!")


if __name__ == '__main__':
    run_test()
