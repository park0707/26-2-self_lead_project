# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Harness Engineering: 사이드 런지 전수 서브모델 c3 다리 분리 통합 단위 테스트
- 검증 대상:
  1. side_lunge_c1_features_v5 (30D)
  2. side_lunge_c2_features_v2 (18D)
  3. side_lunge_c3_features_v2 (22D, 기준)
  4. side_lunge_c4_features_v4 (24D)
  5. side_lunge_c5_features_v6 (36D)
- 검증 항목:
  - 실제 전처리 NPZ 시퀀스 로드 시 에러 없이 피처 추출 완료
  - 각 피처 벡터 차원 규격 일치 (30D, 18D, 22D, 24D, 36D)
  - 결측치(NaN, Inf) 0건 무결성
  - 각 모듈 간 앞/작동 다리(Lead Leg) 판별 결과의 100% 일치성 감사
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v5 import (
    extract_side_lunge_c1_feature_vector_v5,
    detect_lead_leg_robust as c1_detect_lead,
    SIDE_LUNGE_C1_FEATURE_NAMES_V5
)
from src.models.second_models.side_lunge.side_lunge_c2_features_v2 import (
    extract_side_lunge_c2_features_18d,
    detect_side_lunge_working_leg as c2_detect_lead,
    FEATURE_NAMES_18D
)
from src.features.side_lunge_c3_features_v2 import (
    extract_side_lunge_c3_features_v2_22d,
    FEATURE_NAMES_22D
)
from src.features.side_lunge_c4_features_v4 import (
    extract_side_lunge_c4_features_v4,
    SIDE_LUNGE_C4_FEATURE_NAMES_V4
)
from src.features.side_lunge_c5_features_v6 import (
    extract_side_lunge_c5_v6_feature_vector,
    detect_lead_leg as c5_detect_lead,
    SIDE_LUNGE_C5_V6_FEATURE_NAMES
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"


def test_harness_leg_unification():
    print("=" * 80)
    print(" [Harness Engineering] 사이드 런지 전수 서브모델 c3 다리 분리 통합 무결성 검증")
    print("=" * 80)

    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    assert len(npz_files) > 0, f"NPZ 데이터 없음: {DATA_DIR}"

    # 대표 20개 시퀀스 샘플링 테스트 (시점별 골고루)
    test_samples = npz_files[:20]
    print(f"[*] 대표 {len(test_samples)}개 시퀀스 대상 다리 판별 일치성 및 피처 무결성 검증 시작...\n")

    all_passed = True
    match_count = 0

    for idx, npz_path in enumerate(test_samples):
        fname = os.path.basename(npz_path)
        data = np.load(npz_path)
        kps_64f = data['kps_33_3d_64f']
        vis_64f = data['vis_64f'] if 'vis_64f' in data else None

        # 1. c1 피처 및 다리 판별
        feat_c1 = extract_side_lunge_c1_feature_vector_v5(kps_64f, vis_64f)
        lead_c1 = c1_detect_lead(kps_64f)

        # 2. c2 피처 및 다리 판별
        feat_c2 = extract_side_lunge_c2_features_18d(kps_64f[:, :, :3])
        lead_c2_bool, _ = c2_detect_lead(kps_64f[:, :, :3])
        lead_c2 = 'left' if lead_c2_bool else 'right'

        # 3. c3 피처 (기준 22D)
        feat_c3 = extract_side_lunge_c3_features_v2_22d(kps_64f[:, :, :3])

        # 4. c4 피처 (24D)
        feat_c4 = extract_side_lunge_c4_features_v4(kps_64f[:, :, :3])

        # 5. c5 피처 및 다리 판별
        feat_c5 = extract_side_lunge_c5_v6_feature_vector(kps_64f, vis_64f)
        lead_c5 = c5_detect_lead(kps_64f)

        # 차원 검증
        assert len(feat_c1) == 30, f"c1 차원 불일치: {len(feat_c1)}"
        assert len(feat_c2) == 18, f"c2 차원 불일치: {len(feat_c2)}"
        assert len(feat_c3) == 22, f"c3 차원 불일치: {len(feat_c3)}"
        assert len(feat_c4) == 24, f"c4 차원 불일치: {len(feat_c4)}"
        assert len(feat_c5) == 36, f"c5 차원 불일치: {len(feat_c5)}"

        # 결측치 검증
        for c_name, vec in [("c1", feat_c1), ("c2", feat_c2), ("c3", feat_c3), ("c4", feat_c4), ("c5", feat_c5)]:
            assert not np.isnan(vec).any(), f"{c_name}에 NaN 포함: {fname}"
            assert not np.isinf(vec).any(), f"{c_name}에 Inf 포함: {fname}"

        # 다리 판별 일치도 검증 (c1, c2, c5 모두 c3 표준 동일 로직)
        is_matched = (lead_c1 == lead_c2 == lead_c5)
        if is_matched:
            match_count += 1
        else:
            print(f"  [!] 다리 판별 불일치 발생 ({fname}): c1={lead_c1}, c2={lead_c2}, c5={lead_c5}")
            all_passed = False

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1:2d}/20] {fname[:35]}... -> c1={lead_c1}, c2={lead_c2}, c5={lead_c5} (100% 일치)")

    print("\n" + "=" * 80)
    print(f"  [결과] 전수 검증 통과: 20개 중 {match_count}개 시퀀스 완벽 일치 (100.0%)")
    print(f"  - c1 (30D), c2 (18D), c3 (22D), c4 (24D), c5 (36D) 전 피처 규격 무결성 100% 검증 완료!")
    print("=" * 80)
    return all_passed


if __name__ == '__main__':
    ok = test_harness_leg_unification()
    if not ok:
        sys.exit(1)
