# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit test for lunge_c5_features.py
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features import (
    extract_lunge_c5_features,
    extract_lunge_c5_feature_vector,
    LUNGE_C5_FEATURE_NAMES,
    detect_lead_leg,
    find_inflection_point
)


def test_lunge_c5_features():
    print("=" * 60)
    print("  [Test] Lunge C5 Feature Extraction Unit Test")
    print("=" * 60)

    # 1. 기 전처리된 샘플 파일 탐색
    cache_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    npz_files = glob.glob(os.path.join(cache_dir, "*.npz"))
    assert len(npz_files) > 0, "테스트를 위한 npz 파일이 존재하지 않습니다."

    sample_file = npz_files[0]
    print(f"[*] 테스트 대상 파일: {os.path.basename(sample_file)}")

    data = np.load(sample_file)
    kps_33_3d = data['kps_33_3d_64f']
    vis_64f = data['vis_64f'] if 'vis_64f' in data else None

    assert kps_33_3d.shape == (64, 33, 3), f"잘못된 shape: {kps_33_3d.shape}"

    # 2. 앞다리 식별 테스트
    lead_side = detect_lead_leg(kps_33_3d)
    print(f"[+] 앞다리 자율 식별 결과: {lead_side}")
    assert lead_side in ['left', 'right'], f"유효하지 않은 lead_side: {lead_side}"

    # 3. 최저점 검출 테스트
    t_bottom, w_start, w_end = find_inflection_point(kps_33_3d)
    print(f"[+] 최저점 검출 프레임: {t_bottom} (윈도우: {w_start} ~ {w_end})")
    assert 0 <= w_start < t_bottom <= w_end <= 64

    # 4. 32D 피처 딕셔너리 추출
    feats = extract_lunge_c5_features(kps_33_3d, vis_64f)
    assert len(feats) == 32, f"피처 개수 불일치: {len(feats)} != 32"
    for name in LUNGE_C5_FEATURE_NAMES:
        assert name in feats, f"누락된 피처: {name}"
        val = feats[name]
        assert not np.isnan(val), f"NaN 발견: {name}"
        assert not np.isinf(val), f"Inf 발견: {name}"

    # 5. 32D 벡터 변환 테스트
    vec = extract_lunge_c5_feature_vector(kps_33_3d, vis_64f)
    assert vec.shape == (32,), f"벡터 shape 불일치: {vec.shape}"
    assert np.isnan(vec).sum() == 0, "벡터에 NaN 존재"

    print("\n[+] 대표 피처 추출 수치 확인:")
    print(f"  - lunge_torso_pitch_bottom_win: {feats['lunge_torso_pitch_bottom_win']:.2f}°")
    print(f"  - lunge_torso_pitch_peak: {feats['lunge_torso_pitch_peak']:.2f}°")
    print(f"  - lunge_torso_pitch_min: {feats['lunge_torso_pitch_min']:.2f}°")
    print(f"  - lunge_torso_pitch_rom: {feats['lunge_torso_pitch_rom']:.2f}°")
    print(f"  - lunge_torso_foreshortening_ratio: {feats['lunge_torso_foreshortening_ratio']:.4f}")
    print(f"  - lunge_delta_pitch_from_setup: {feats['lunge_delta_pitch_from_setup']:.2f}°")

    print("\n[SUCCESS] 모든 단위 테스트를 완벽히 통과하였습니다!")


if __name__ == "__main__":
    test_lunge_c5_features()
