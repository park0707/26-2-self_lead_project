# -*- coding: utf-8 -*-
"""
Unit test for lunge_c1_features_v2.py
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c1_features_v2 import (
    extract_lunge_c1_features_v2,
    extract_lunge_c1_feature_vector_v2,
    LUNGE_C1_FEATURE_NAMES_V2,
    detect_lead_leg,
    find_dual_anchor_inflection_point,
    build_isb_pelvis_frame
)


def test_v2_features():
    print("=" * 60)
    print("  [Test] Lunge C1 v2 Feature Extraction Unit Test")
    print("=" * 60)

    cache_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    npz_files = glob.glob(os.path.join(cache_dir, "*.npz"))
    assert len(npz_files) > 0, "테스트용 npz 파일 없음"

    sample_file = npz_files[0]
    print(f"[*] 테스트 샘플: {os.path.basename(sample_file)}")
    data = np.load(sample_file)
    kps = data['kps_33_3d_64f']
    vis = data['vis_64f'] if 'vis_64f' in data else None

    # 1. ISB 프레임 테스트
    R = build_isb_pelvis_frame(kps[0])
    assert R.shape == (3, 3), f"R shape 오류: {R.shape}"
    # 직교성 검증 (R @ R.T = I)
    ident = R @ R.T
    assert np.allclose(ident, np.eye(3), atol=1e-3), "ISB 프레임 직교성 위반"
    print("[+] ISB 골반 직교 기저(3x3) 직교성 완벽 검증 완료!")

    # 2. 38D 피처 딕셔너리 추출
    feats = extract_lunge_c1_features_v2(kps, vis)
    assert len(feats) == 38, f"피처 개수 오류: {len(feats)} != 38"
    for name in LUNGE_C1_FEATURE_NAMES_V2:
        assert name in feats, f"누락된 피처: {name}"
        assert not np.isnan(feats[name]), f"NaN: {name}"
        assert not np.isinf(feats[name]), f"Inf: {name}"

    # 3. 38D 벡터 변환
    vec = extract_lunge_c1_feature_vector_v2(kps, vis)
    assert vec.shape == (38,), f"벡터 shape 오류: {vec.shape}"
    assert np.isnan(vec).sum() == 0

    print("\n[+] v2 대표 신규 피처 수치 확인:")
    print(f"  - sagittal_knee_angle_bottom: {feats['sagittal_knee_angle_bottom']:.2f}°")
    print(f"  - sagittal_knee_90_error: {feats['sagittal_knee_90_error']:.2f}°")
    print(f"  - kot_normalized_displacement: {feats['kot_normalized_displacement']:.3f}")
    print(f"  - rigid_bone_length_consistency: {feats['rigid_bone_length_consistency']:.4f}")
    print(f"  - tibia_torso_parallelism_error: {feats['tibia_torso_parallelism_error']:.2f}°")
    print(f"  - rear_knee_ground_clearance: {feats['rear_knee_ground_clearance']:.3f}")

    print("\n[SUCCESS] lunge_c1_features_v2 단위 테스트를 100% 완벽히 통과하였습니다!")


if __name__ == "__main__":
    test_v2_features()
