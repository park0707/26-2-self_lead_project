# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Test: test_sl2_augmentation.py
Harness test for sl2_augmentation.py
"""

import numpy as np
from sl2_augmentation import (
    temporal_warp_pose_sequence,
    kinematic_bone_scaling,
    generate_augmented_defect_variants
)


def test_augmentation_integrity():
    np.random.seed(42)
    fake_kps = np.random.randn(64, 33, 3).astype(np.float32)

    # 1. 시간축 워핑 테스트
    w_slow = temporal_warp_pose_sequence(fake_kps, speed_factor=0.85)
    w_fast = temporal_warp_pose_sequence(fake_kps, speed_factor=1.15)
    assert w_slow.shape == (64, 33, 3)
    assert w_fast.shape == (64, 33, 3)
    assert not np.isnan(w_slow).any()
    assert not np.isnan(w_fast).any()

    # 2. 뼈 길이 스케일링 테스트
    scaled = kinematic_bone_scaling(fake_kps, scale_femur=1.1, scale_tibia=0.9, scale_foot=1.05, scale_torso=1.05)
    assert scaled.shape == (64, 33, 3)
    assert not np.isnan(scaled).any()

    # 3. 종합 변형 생성 테스트
    variants = generate_augmented_defect_variants(fake_kps, n_variants=2)
    assert len(variants) == 2
    for v in variants:
        assert v.shape == (64, 33, 3)
        assert not np.isnan(v).any()

    print("[ALL TESTS PASSED] sl2_augmentation 단위 테스트 100% 통과!")


if __name__ == '__main__':
    test_augmentation_integrity()
