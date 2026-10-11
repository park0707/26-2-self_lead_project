# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드, 06_스텝_백워드)
Module: build_dynamic_lunge_c5_rtmp_dataset.py
Purpose: 다이나믹 런지 c5 (상체의 과도한 숙임/젖힘) RTMPose 73D 및 Hybrid 79D 피처 캐시 구축
         - MediaPipe 3D 36D (사전 추출 검증 완료)
         - RTMPose Halpe26 2D -> 73D Pure RTMP 피처
         - MediaPipe 36D + RTMP 37D + 6D Cross-Modal = 79D Hybrid 피처
         - 전체 1,468개 시퀀스 전수 추출 및 피험자(1~9)/5대시점(A~E) 무결성 보장
"""

import os
import sys
import glob
import time
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features_rtmp_3 import (
    extract_lunge_c5_rtmp_3_features,
    map_halpe26_to_mp33,
    LUNGE_C5_RTMP_3_FEATURE_NAMES
)
from src.features.lunge_c5_features_hybrid import (
    compute_cross_interactions,
    LUNGE_C5_HYBRID_79D_NAMES
)

MP_CACHE_PATH = r"D:\Data\피처_데이터\dynamic_lunge_c5_v5_36d_features.npz"
RTMP_DIRS = [
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\05_스텝_포워드_다이나믹_런지",
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\06_스텝_백워드_다이나믹_런지"
]
OUT_CACHE_PATH = r"D:\Data\피처_데이터\dynamic_lunge_c5_rtmp_dataset.npz"

def build_dataset():
    print("=" * 80)
    print("  [vidio-health-trainer] 다이나믹 런지 c5 RTMP(73D) 및 Hybrid(79D) 피처셋 구축")
    print("=" * 80)

    if not os.path.exists(MP_CACHE_PATH):
        raise FileNotFoundError(f"MediaPipe 36D 캐시 파일이 존재하지 않습니다: {MP_CACHE_PATH}")

    d_mp = np.load(MP_CACHE_PATH, allow_pickle=True)
    mp_fnames = list(d_mp['filenames'])
    X_mp = d_mp['X'].astype(np.float32)
    y_all = d_mp['y'].astype(int)
    subs_all = d_mp['subjects'].astype(int)
    views_all = d_mp['views'].astype(str)
    keys_all = list(d_mp['keys'])

    N = len(mp_fnames)
    print(f"[*] MediaPipe 캐시 로드 완료: {N}개 시퀀스")

    X_rtmp_73d_list = []
    X_hybrid_79d_list = []
    exercise_types = []

    start_time = time.time()
    for i, fname in enumerate(tqdm(mp_fnames, desc="Extracting RTMP & Hybrid features")):
        p1 = os.path.join(RTMP_DIRS[0], fname)
        p2 = os.path.join(RTMP_DIRS[1], fname)

        if os.path.exists(p1):
            rtmp_path = p1
            ex_type = "05_스텝_포워드"
        elif os.path.exists(p2):
            rtmp_path = p2
            ex_type = "06_스텝_백워드"
        else:
            raise FileNotFoundError(f"RTMP 파일을 찾을 수 없습니다: {fname}")

        rtmp_data = np.load(rtmp_path, allow_pickle=True)
        kps_26 = rtmp_data['kps_26_64f']

        # 1. Halpe26 -> MP33 mapping
        mp33 = map_halpe26_to_mp33(kps_26)

        # 2. Extract RTMP 73D
        feat_rtmp_73d = extract_lunge_c5_rtmp_3_features(mp33)
        feat_rtmp_73d = np.nan_to_num(feat_rtmp_73d, nan=0.0, posinf=0.0, neginf=0.0)
        X_rtmp_73d_list.append(feat_rtmp_73d)

        # 3. Extract Hybrid 79D
        rtmp_37d = feat_rtmp_73d[31:68]
        feat_mp_36d = X_mp[i]
        cross_6d = compute_cross_interactions(feat_mp_36d, rtmp_37d)
        feat_hybrid_79d = np.concatenate([feat_mp_36d, rtmp_37d, cross_6d]).astype(np.float32)
        feat_hybrid_79d = np.nan_to_num(feat_hybrid_79d, nan=0.0, posinf=0.0, neginf=0.0)
        X_hybrid_79d_list.append(feat_hybrid_79d)

        exercise_types.append(ex_type)

    X_rtmp_73d = np.array(X_rtmp_73d_list, dtype=np.float32)
    X_hybrid_79d = np.array(X_hybrid_79d_list, dtype=np.float32)
    exercise_types = np.array(exercise_types, dtype=object)

    elapsed = time.time() - start_time
    print(f"\n[+] 피처 추출 완료: {elapsed:.2f}초 소요 ({elapsed/N*1000:.1f}ms/시퀀스)")
    print(f"[*] X_rtmp_73d shape  : {X_rtmp_73d.shape} (NaNs: {np.isnan(X_rtmp_73d).sum()})")
    print(f"[*] X_hybrid_79d shape: {X_hybrid_79d.shape} (NaNs: {np.isnan(X_hybrid_79d).sum()})")
    print(f"[*] y distribution     : Normal(0)={np.sum(y_all==0)}, Defect(1)={np.sum(y_all==1)} ({np.mean(y_all)*100:.1f}%)")
    print(f"[*] Exercise distribution: 포워드={np.sum(exercise_types=='05_스텝_포워드')}, 백워드={np.sum(exercise_types=='06_스텝_백워드')}")
    print(f"[*] Subjects: {np.unique(subs_all)}")
    print(f"[*] Views   : {np.unique(views_all)}")

    os.makedirs(os.path.dirname(OUT_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_CACHE_PATH,
        X_rtmp_73d=X_rtmp_73d,
        X_hybrid_79d=X_hybrid_79d,
        X_mp_36d=X_mp,
        y=y_all,
        subjects=subs_all,
        views=views_all,
        keys=keys_all,
        filenames=mp_fnames,
        exercise_types=exercise_types,
        rtmp_feature_names=np.array(LUNGE_C5_RTMP_3_FEATURE_NAMES, dtype=object),
        hybrid_feature_names=np.array(LUNGE_C5_HYBRID_79D_NAMES, dtype=object)
    )
    print(f"[+] 캐시 파일 압축 저장 완료: {OUT_CACHE_PATH} ({os.path.getsize(OUT_CACHE_PATH):,} bytes)")

if __name__ == '__main__':
    build_dataset()
