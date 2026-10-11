# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_dynamic_lunge_c2_hybrid_cache.py
Purpose: 다이나믹 런지(05_스텝_포워드, 06_스텝_백워드) 1,468개 전수 피처 추출 및 캐싱
         - MediaPipe 33관절 3D (64F)
         - RTMPose Halpe 26관절 2D (64F)
         - bl2_final 계승 46D 하이브리드 피처 (MP 18D + RTMP 18D + MP Frontal 5D + RTMP Frontal 5D)
         - AI-Hub 라벨 격리 (오직 정답 y 및 메타데이터로만 저장)
"""

import os
import sys
import glob
import json
import time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BABEL_LUNGE_DIR = r"D:\src\models\second_models\babel_lunge"
if BABEL_LUNGE_DIR not in sys.path:
    sys.path.insert(0, BABEL_LUNGE_DIR)

from lunge_c2_features_v6 import FEATURE_NAMES_18D
from lunge_c2_features_hybrid_2 import (
    FEATURE_NAMES_FRONTAL,
    extract_hybrid_2_features
)

MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지"
RTMP_DIRS = [
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\05_스텝_포워드_다이나믹_런지",
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\06_스텝_백워드_다이나믹_런지"
]
LABEL_DIRS = [
    r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D",
    r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"
]

OUTPUT_CACHE = r"D:\src\models\second_models\dynamic_lunge\dynamic_lunge_c2_hybrid_cache.npz"

FEATURE_NAMES_46D = (
    [f"MP_{f}" for f in FEATURE_NAMES_18D] + 
    [f"RTMP_{f}" for f in FEATURE_NAMES_18D] + 
    [f"MP_{f}" for f in FEATURE_NAMES_FRONTAL] + 
    [f"RTMP_{f}" for f in FEATURE_NAMES_FRONTAL]
)

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 다이나믹 런지 c2 하이브리드 피처 1,468개 전수 추출 시작")
    print("=" * 80)

    mp_files = sorted(glob.glob(os.path.join(MP33_DIR, "*.npz")))
    print(f"[*] 총 MediaPipe 파일 탐색: {len(mp_files)}개")

    matched_items = []
    for mp_path in mp_files:
        fname = os.path.basename(mp_path)
        view_code = None
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            if f"_{v}.npz" in fname:
                view_code = v
                break
        if not view_code:
            continue

        sub_id = None
        for s in range(1, 10):
            if f"피험자{s}" in fname:
                sub_id = s
                break
        if not sub_id:
            continue

        rtmp_path = None
        for rd in RTMP_DIRS:
            c = os.path.join(rd, fname)
            if os.path.exists(c):
                rtmp_path = c
                break
        if not rtmp_path:
            continue

        raw_key = fname.replace(f"_{view_code}.npz", "")
        label_path = None
        for ld in LABEL_DIRS:
            c = os.path.join(ld, f"{raw_key}.json")
            if os.path.exists(c):
                label_path = c
                break
        if not label_path:
            continue

        try:
            with open(label_path, 'r', encoding='utf-8', errors='ignore') as fp:
                d = json.load(fp)
            conds = d.get('type_info', {}).get('conditions', [])
            c2_val = conds[1].get('value')
            y_c2 = 1 if (c2_val is False) else 0
        except Exception:
            continue

        ex_type = "05_스텝_포워드" if ("05_" in rtmp_path or "포워드" in rtmp_path) else "06_스텝_백워드"
        matched_items.append({
            'fname': fname,
            'clip_name': raw_key,
            'mp_path': mp_path,
            'rtmp_path': rtmp_path,
            'view': view_code,
            'sub': sub_id,
            'y_c2': y_c2,
            'ex_type': ex_type
        })

    N = len(matched_items)
    print(f"[*] 매칭 완료: {N}개 시퀀스")
    assert N == 1468, f"예상 시퀀스 수 1,468개 불일치: {N}"

    X_46d = np.zeros((N, 46), dtype=np.float32)
    y_arr = np.zeros(N, dtype=np.int32)
    subjects = np.zeros(N, dtype=np.int32)
    views = []
    clip_names = []
    ex_types = []

    t0 = time.time()
    for idx, item in enumerate(matched_items):
        mp_data = np.load(item['mp_path'], allow_pickle=True)
        rtmp_data = np.load(item['rtmp_path'], allow_pickle=True)

        kps_mp_3d = mp_data['kps_33_3d_64f']
        kps_rtmp_2d = rtmp_data['kps_26_64f']

        f_46 = extract_hybrid_2_features(kps_mp_3d, kps_rtmp_2d)
        X_46d[idx] = f_46
        y_arr[idx] = item['y_c2']
        subjects[idx] = item['sub']
        views.append(item['view'])
        clip_names.append(item['clip_name'])
        ex_types.append(item['ex_type'])

        if (idx + 1) % 200 == 0 or (idx + 1) == N:
            print(f"  - [{idx+1}/{N}] 처리 중... ({time.time()-t0:.1f}s)")

    views_arr = np.array(views)
    clip_names_arr = np.array(clip_names)
    ex_types_arr = np.array(ex_types)

    # 무결성 검증
    nan_count = np.isnan(X_46d).sum()
    inf_count = np.isinf(X_46d).sum()
    print("\n[+] 무결성 사전 검증 (Pre-Modeling Validation Invariant):")
    print(f"  - X Shape: {X_46d.shape}")
    print(f"  - y Shape: {y_arr.shape} (정상: {np.sum(y_arr==0)}, 결함: {np.sum(y_arr==1)}, 비율: {np.mean(y_arr)*100:.2f}%)")
    print(f"  - NaN Count: {nan_count}")
    print(f"  - Inf Count: {inf_count}")
    print(f"  - Subjects (1~9): {sorted(np.unique(subjects))}")
    print(f"  - Views (5개 시점): {sorted(np.unique(views_arr))}")
    assert nan_count == 0, "NaN 감지!"
    assert inf_count == 0, "Inf 감지!"

    # 캐시 저장
    np.savez_compressed(
        OUTPUT_CACHE,
        X=X_46d,
        y=y_arr,
        subjects=subjects,
        views=views_arr,
        clip_names=clip_names_arr,
        ex_types=ex_types_arr,
        feature_names=FEATURE_NAMES_46D
    )
    print(f"\n[+] 46D 캐시 저장 완료: {OUTPUT_CACHE} ({os.path.getsize(OUTPUT_CACHE)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
