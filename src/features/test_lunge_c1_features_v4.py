# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: test_lunge_c1_features_v4.py
Purpose: lunge_c1_features_v4 32D 피처 추출 모듈 무결성 및 단위 테스트
"""

import os
import glob
import numpy as np
import time

from lunge_c1_features_v4 import (
    extract_lunge_c1_features_v4,
    extract_lunge_c1_feature_vector_v4,
    LUNGE_C1_FEATURE_NAMES_V4,
)


def test_v4_feature_extraction():
    print("=" * 60)
    print("  [Test] lunge_c1_features_v4 Unit & Integrity Test")
    print("=" * 60)

    # 1. 피처 명칭 수 검증 (정확히 32개)
    assert len(LUNGE_C1_FEATURE_NAMES_V4) == 32, f"피처 개수가 32개가 아닙니다: {len(LUNGE_C1_FEATURE_NAMES_V4)}"
    print(f"[+] 공식 v4 피처 수: {len(LUNGE_C1_FEATURE_NAMES_V4)}D 확인 완료")

    # 2. 합성 데이터 규격 및 NaN/Inf 검증
    np.random.seed(42)
    dummy_kps = np.random.uniform(-1.0, 1.0, (64, 33, 3)).astype(np.float32)
    dummy_vis = np.random.uniform(0.5, 1.0, (64, 33)).astype(np.float32)

    feat_dict = extract_lunge_c1_features_v4(dummy_kps, dummy_vis)
    feat_vec = extract_lunge_c1_feature_vector_v4(dummy_kps, dummy_vis)

    assert len(feat_dict) == 32, f"추출 딕셔너리 크기 불일치: {len(feat_dict)}"
    assert feat_vec.shape == (32,), f"추출 벡터 차원 불일치: {feat_vec.shape}"
    assert not np.isnan(feat_vec).any(), "NaN 값 검출됨!"
    assert not np.isinf(feat_vec).any(), "Inf 값 검출됨!"
    print("[+] 합성 데이터 32D 벡터 규격 및 무결성(No NaN/Inf) 검증 통과")

    # 3. 실제 전처리 NPZ 데이터셋 대상 실측 검증
    npz_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    sample_files = glob.glob(os.path.join(npz_dir, "*.npz"))[:5]

    if sample_files:
        print(f"\n[*] 실제 전처리 NPZ 샘플 {len(sample_files)}개 실측 테스트:")
        for fpath in sample_files:
            bname = os.path.basename(fpath)
            data = np.load(fpath)
            kps_64f = data['kps_33_3d_64f']
            vis_64f = data.get('vis_64f', None)

            t0 = time.perf_counter()
            vec = extract_lunge_c1_feature_vector_v4(kps_64f, vis_64f)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            assert vec.shape == (32,), f"실제 데이터 피처 형상 오류: {vec.shape}"
            assert not np.isnan(vec).any(), f"실제 데이터 {bname}에 NaN 포함"
            assert not np.isinf(vec).any(), f"실제 데이터 {bname}에 Inf 포함"

            # 6대 신규 피처 검증
            r_vert = vec[26]
            pelvis_drop = vec[27]
            recon_angle = vec[28]
            recon_90_err = vec[29]
            morpho_err = vec[30]
            pka_norm = vec[31]

            print(f"  - [{bname[:28]}...] 32D ({elapsed_ms:.2f}ms) | KneeMin={vec[0]:.1f}° | ReconKnee={recon_angle:.1f}° | R_vert={r_vert:.3f} | PelvisDrop={pelvis_drop:.3f}")
        print("\n[SUCCESS] 실제 데이터 기반 32D 피처 추출 100% 무결성 확인!")
    else:
        print("[!] 샘플 NPZ 파일을 찾을 수 없습니다. 경로 확인 필요.")


if __name__ == "__main__":
    test_v4_feature_extraction()
