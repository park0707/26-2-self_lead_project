# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: align_lunge_c4_dataset_cache_v4.py
Purpose: v4 캐시를 v1/v2/v3와 완벽히 동일한 787개 시퀀스 및 순서로 정렬하여 공정한 1:1 비교 보장
"""

import os
import sys
import numpy as np

sys.path.insert(0, r"D:/")
from src.features.lunge_c4_features_v4 import (
    extract_lunge_c4_features_v4,
    extract_lunge_c4_temporal_features_v4,
    LUNGE_C4_FEATURE_NAMES_V4
)

V3_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz"
V4_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v4.npz"
MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"


def align_v4_cache():
    print("=" * 80)
    print("  [vidio-health-trainer] v4 데이터셋 캐시를 v3 787개 기준 1:1 정렬")
    print("=" * 80)

    d3 = np.load(V3_CACHE_PATH, allow_pickle=True)
    v3_clip_ids = list(d3['clip_ids'])
    v3_y = d3['y']
    v3_subjects = d3['subjects']
    v3_views = d3['views']
    v3_ambiguities = d3['ambiguities']
    v3_keys = d3['keys']

    print(f"[*] 기준 v3 시퀀스 수: {len(v3_clip_ids)}개 (정상: {np.sum(v3_y==0)}, 결함: {np.sum(v3_y==1)})")

    X_30d_list = []
    X_temp_16d_list = []

    for cid in v3_clip_ids:
        fpath = os.path.join(MP33_DIR, f"{cid}.npz")
        data = np.load(fpath)
        seq_64f = data['kps_33_3d_64f']

        f30 = extract_lunge_c4_features_v4(seq_64f)
        t16 = extract_lunge_c4_temporal_features_v4(seq_64f)

        X_30d_list.append(f30)
        X_temp_16d_list.append(t16)

    X_30d = np.array(X_30d_list, dtype=np.float32)
    X_temp_16d = np.array(X_temp_16d_list, dtype=np.float32)

    print(f"[*] 추출 완료된 v4 피처 형상: {X_30d.shape}")
    print(f"[*] 추출 완료된 v4 시계열 형상: {X_temp_16d.shape}")
    assert X_30d.shape == (787, 30), f"형상 불일치: {X_30d.shape}"
    assert X_temp_16d.shape == (787, 64, 16), f"시계열 형상 불일치: {X_temp_16d.shape}"

    np.savez_compressed(
        V4_CACHE_PATH,
        X_30d=X_30d,
        X_temp_16d=X_temp_16d,
        y=v3_y,
        subjects=v3_subjects,
        views=v3_views,
        clip_ids=v3_clip_ids,
        ambiguities=v3_ambiguities,
        keys=v3_keys,
        feature_names=np.array(LUNGE_C4_FEATURE_NAMES_V4)
    )

    print(f"[+] v4 정밀 캐시 787개 저장 완료: {V4_CACHE_PATH}")


if __name__ == "__main__":
    align_v4_cache()
