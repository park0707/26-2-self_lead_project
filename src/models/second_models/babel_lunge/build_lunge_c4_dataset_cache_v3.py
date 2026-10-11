# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_lunge_c4_dataset_cache_v3.py
Purpose: 바벨 런지 c4 v3 정예 24차원 피처 및 14차원 시계열 데이터셋 캐시 구축
"""

import os
import sys
import numpy as np

sys.path.insert(0, r"D:/")
from src.features.lunge_c4_features_v3 import LUNGE_C4_FEATURE_NAMES_V3

V2_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v2.npz"
V3_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz"


def build_v3_cache():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 런지 c4 v3 정예 24D 데이터셋 캐시 구축")
    print("=" * 80)

    if not os.path.exists(V2_CACHE_PATH):
        raise FileNotFoundError(f"v2 캐시 파일이 존재하지 않습니다: {V2_CACHE_PATH}")

    d_v2 = np.load(V2_CACHE_PATH, allow_pickle=True)
    X_36d = d_v2['X_36d']
    X_temp_14d = d_v2['X_temp_14d']
    y = d_v2['y']
    subjects = d_v2['subjects']
    views = d_v2['views']
    clip_ids = d_v2['clip_ids']
    ambiguities = d_v2['ambiguities']
    keys = d_v2['keys']
    v2_feature_names = list(d_v2['feature_names'])

    print(f"[*] 원본 v2 피처 형상: {X_36d.shape}")
    print(f"[*] v3 정예 피처 수: {len(LUNGE_C4_FEATURE_NAMES_V3)}개")

    # v3 피처에 해당하는 인덱스 맵 생성
    indices_v3 = []
    for name in LUNGE_C4_FEATURE_NAMES_V3:
        if name not in v2_feature_names:
            raise ValueError(f"v2 피처 목록에 존재하지 않는 피처: {name}")
        idx = v2_feature_names.index(name)
        indices_v3.append(idx)

    indices_v3 = np.array(indices_v3, dtype=np.int32)
    X_24d = X_36d[:, indices_v3]

    print(f"[*] 구축된 v3 피처 형상: {X_24d.shape}")
    assert X_24d.shape == (787, 24), f"예상 형상 (787, 24) 불일치: {X_24d.shape}"
    assert not np.isnan(X_24d).any(), "NaN 존재!"
    assert not np.isinf(X_24d).any(), "Inf 존재!"

    # 캐시 저장
    os.makedirs(os.path.dirname(V3_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        V3_CACHE_PATH,
        X_24d=X_24d,
        X_temp_14d=X_temp_14d,
        y=y,
        subjects=subjects,
        views=views,
        clip_ids=clip_ids,
        ambiguities=ambiguities,
        keys=keys,
        feature_names=np.array(LUNGE_C4_FEATURE_NAMES_V3)
    )

    print(f"[+] v3 정예 데이터셋 캐시 저장 완료: {V3_CACHE_PATH}")
    print(f"    - 총 샘플 수: {len(y)}개 (정상: {np.sum(y == 0)}, 결함: {np.sum(y == 1)})")
    print(f"    - 피처 수: 24차원")
    print(f"    - 시계열 텐서: {X_temp_14d.shape}")


if __name__ == "__main__":
    build_v3_cache()
