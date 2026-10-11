# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_sl2_v8_front_features.py
Harness Engineering Test Suite for side_lunge_c2_v8_front_features.py (57D)
"""

import os
import glob
import numpy as np

from side_lunge_c2_v8_front_features import (
    FEATURE_NAMES_58D,
    compute_v8_front_breakthrough_features_7d,
    extract_side_lunge_c2_v8_front_features_58d
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"


def test_feature_names_dimension():
    """피처 이름 목록이 정확히 58개인지 검증"""
    assert len(FEATURE_NAMES_58D) == 58, f"58개 피처 목록 불일치: {len(FEATURE_NAMES_58D)}"
    assert len(set(FEATURE_NAMES_58D)) == 58, "피처 이름 중복 감지"


def test_v8_front_synthetic_pose():
    """합성 포즈 텐서에 대한 58D 추출 및 NaN/Inf 0% 무결성 검증"""
    np.random.seed(42)
    fake_kps = np.random.randn(64, 33, 3).astype(np.float32)

    feat_b = extract_side_lunge_c2_v8_front_features_58d(fake_kps, view_code="view2")
    feat_d = extract_side_lunge_c2_v8_front_features_58d(fake_kps, view_code="view4")

    assert feat_b.shape == (58,), f"시점 B 피처 형상 오류: {feat_b.shape}"
    assert feat_d.shape == (58,), f"시점 D 피처 형상 오류: {feat_d.shape}"

    assert not np.isnan(feat_b).any(), "시점 B 피처에 NaN 검출"
    assert not np.isinf(feat_b).any(), "시점 B 피처에 Inf 검출"
    assert not np.isnan(feat_d).any(), "시점 D 피처에 NaN 검출"
    assert not np.isinf(feat_d).any(), "시점 D 피처에 Inf 검출"


def test_v8_front_real_npz():
    """실제 전처리 npz 데이터 파일 2건(시점 B, D)에 대한 무결성 검증"""
    npz_b = glob.glob(os.path.join(DATA_DIR, "*_view2.npz"))
    npz_d = glob.glob(os.path.join(DATA_DIR, "*_view4.npz"))

    assert len(npz_b) > 0, "view2 npz 파일이 존재하지 않습니다."
    assert len(npz_d) > 0, "view4 npz 파일이 존재하지 않습니다."

    data_b = np.load(npz_b[0])['kps_33_3d_64f']
    data_d = np.load(npz_d[0])['kps_33_3d_64f']

    feat_b = extract_side_lunge_c2_v8_front_features_58d(data_b, view_code="view2")
    feat_d = extract_side_lunge_c2_v8_front_features_58d(data_d, view_code="view4")

    assert feat_b.shape == (58,)
    assert feat_d.shape == (58,)
    assert not np.isnan(feat_b).any()
    assert not np.isnan(feat_d).any()

    # 신규 7D 피처의 물리적 유효성 검증
    # 51: fpa_dynamic_foot_progression_angle (-180 ~ +180)
    assert -180.0 <= feat_b[51] <= 180.0
    # 52: front_screen_space_2d_torso_foot_cosine (-1.0 ~ +1.0)
    assert -1.05 <= feat_b[52] <= 1.05
    # 53: front_view_relative_ipsi_contra_yaw (-180 ~ +180)
    assert -180.0 <= feat_b[53] <= 180.0
    # 57: kappa_c2 scalar triple product (0.0 ~ 1.05)
    assert 0.0 <= feat_b[57] <= 1.05

    print(f"\n[+] 시점 B 58D 피처 정상 추출 완료 (앞 5개: {feat_b[:5]}, 신규 7개: {feat_b[51:]})")
    print(f"[+] 시점 D 58D 피처 정상 추출 완료 (신규 7개: {feat_d[51:]})")


if __name__ == '__main__':
    test_feature_names_dimension()
    test_v8_front_synthetic_pose()
    test_v8_front_real_npz()
    print("\n[ALL TESTS PASSED] sl2_v8_front 57D 피처 하네스 테스트 100% 통과!")
