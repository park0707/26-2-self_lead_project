# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_all_views_58d_cache.py
Purpose: AI-Hub 사이드 런지 4,130건 전수 대상 MediaPipe 58D 피처 캐시 구축
         - 기존 전방 1,652건(view2, view4)의 7D 피처는 sl2_v8_front_58d_cache.pkl에서 직결 로드
         - 나머지 2,478건(view1, view3, view5)의 7D 피처만 병렬 추출하여 고속 합성
         - 최종 저장: D:\Data\피처_데이터\sl2_v8_all_views_58d_cache.pkl
"""

import os
import sys
import time
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.side_lunge.side_lunge_c2_v8_front_features import (
    compute_v8_front_breakthrough_features_7d,
    FEATURE_NAMES_58D
)
from src.models.second_models.side_lunge.side_lunge_c2_features_v3 import safe_savgol_filter_kps

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
V7_ALL_CACHE = r"D:\Data\피처_데이터\sl2_v7_all_views_51d_cache.pkl"
V8_FRONT_CACHE = r"D:\Data\피처_데이터\sl2_v8_front_58d_cache.pkl"
OUT_CACHE = r"D:\Data\피처_데이터\sl2_v8_all_views_58d_cache.pkl"


def process_single(item):
    idx, clip_name, view_code = item
    npz_path = os.path.join(DATA_DIR, f"{clip_name}_{view_code}.npz")
    d = np.load(npz_path)
    kps = safe_savgol_filter_kps(d['kps_33_3d_64f'][:, :, :3])
    f7 = compute_v8_front_breakthrough_features_7d(kps, view_code)
    return idx, f7


def build_cache():
    if os.path.exists(OUT_CACHE):
        print(f"[*] 이미 완성된 4,130건 58D 캐시가 존재합니다: {OUT_CACHE}")
        with open(OUT_CACHE, 'rb') as f:
            d = pickle.load(f)
        print(f"[*] 캐시 형상: X={d['X'].shape}, y={d['y'].shape}")
        return

    print("=" * 80)
    print("vidio-health-trainer: 4,130건 5대 시점 전수 58D 피처 캐시 구축 시작")
    print("=" * 80)

    # 1. v7 51D 로드
    with open(V7_ALL_CACHE, 'rb') as f:
        v7 = pickle.load(f)

    X_51d = v7['X']
    y = v7['y']
    groups = v7['groups']
    views = v7['views']
    clip_names = v7['clip_names']
    N = len(y)
    assert N == 4130, f"Expected 4130, got {N}"

    # 2. 기존 전방 1,652건 캐시 로드
    with open(V8_FRONT_CACHE, 'rb') as f:
        front_v8 = pickle.load(f)

    front_clips = front_v8['clip_names']
    front_views = front_v8['views']
    front_7d = front_v8['X'][:, 51:] # 마지막 7개 피처
    front_map = {(c, v): front_7d[i] for i, (c, v) in enumerate(zip(front_clips, front_views))}

    feat_7d_all = np.zeros((N, 7), dtype=np.float32)
    missing_items = []

    for i in range(N):
        key = (clip_names[i], views[i])
        if key in front_map:
            feat_7d_all[i] = front_map[key]
        else:
            missing_items.append((i, clip_names[i], views[i]))

    print(f"[*] 기존 전방 캐시 재사용: {len(front_map)}건 매핑 완료")
    print(f"[*] 신규 추출 필요 건수: {len(missing_items)}건 (view1, view3, view5)")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, f7 in executor.map(process_single, missing_items):
            feat_7d_all[idx] = f7

    elapsed = time.time() - t0
    print(f"[+] {len(missing_items)}건 병렬 추출 완료! 소요시간: {elapsed:.1f}초")

    X_58d = np.hstack([X_51d, feat_7d_all])
    print(f"[+] 최종 X_58d 결합 완료: 형상 = {X_58d.shape}")

    payload = {
        'X': X_58d,
        'y': y,
        'groups': groups,
        'views': views,
        'clip_names': clip_names,
        'feature_names': FEATURE_NAMES_58D
    }

    with open(OUT_CACHE, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[+] 4,130건 전수 58D 캐시 저장 완료: {OUT_CACHE}")


if __name__ == '__main__':
    build_cache()
