# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Module: src/features/build_side_lunge_c1_hybrid_features.py
Purpose: MediaPipe 33관절(Pseudo-3D) 및 RTMPose Halpe26(고정밀 2D) 추출 포인트의
         장점을 상호 보완적으로 융합한 sl1 하이브리드 피처셋 생성 및 캐싱 파이프라인
         - 정예 55D 하이브리드 (RTMP 46D 2D 고정밀 + MP 9D 3D 시상면/불변량)
         - 확장 110D 풀 하이브리드 (RTMP 55D + MP 55D 상호작용 피처)
         - 4,130개 시퀀스 전수 1:1 완벽 정합 및 무결성 검증
"""

import os
import sys
import pickle
import numpy as np

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MP_CACHE_PATH = r"D:\Data\피처_데이터\sl1_v10_dataset_cache.pkl"
RTMP_CACHE_PATH = r"D:\Data\피처_데이터\side_lunge_rtmp_features.npz"
OUT_CACHE_PATH = r"D:\Data\피처_데이터\sl1_hybrid_features.npz"

DEGENERATE_RTMP_INDICES = [5, 17, 22, 27, 32, 44, 50, 53, 54]


def build_and_save_hybrid_features():
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 c1 MediaPipe + RTMPose 하이브리드 피처 생성")
    print("=" * 80)

    # 1. 기존 캐시 로드
    print("[*] MediaPipe 캐시 로드 중:", MP_CACHE_PATH)
    with open(MP_CACHE_PATH, 'rb') as f:
        mp_data = pickle.load(f)

    print("[*] RTMPose 캐시 로드 중:", RTMP_CACHE_PATH)
    rtmp_data = np.load(RTMP_CACHE_PATH)

    X_mp = mp_data['X'].astype(np.float32)
    X_rtmp = rtmp_data['X_c1'].astype(np.float32)
    y = rtmp_data['y_c1'].astype(int)
    groups = rtmp_data['subjects']
    views = rtmp_data['views']
    clip_names = rtmp_data['clip_names']
    feature_names_base = mp_data['feature_names']

    # 2. 정합성 검증
    assert len(X_mp) == 4130, f"MediaPipe 샘플 수 오류: {len(X_mp)}"
    assert len(X_rtmp) == 4130, f"RTMPose 샘플 수 오류: {len(X_rtmp)}"
    assert list(mp_data['clip_names']) == list(clip_names), "시퀀스 순서 불일치!"
    print(f"[*] 4,130개 시퀀스 1:1 완벽 정합 확인 완료.")

    # 3. 정예 55D 하이브리드 피처 생성 (RTMP 2D + MP 3D 대체)
    X_hybrid_55d = X_rtmp.copy()
    hybrid_55d_names = []
    for i, name in enumerate(feature_names_base):
        if i in DEGENERATE_RTMP_INDICES:
            X_hybrid_55d[:, i] = X_mp[:, i]
            hybrid_55d_names.append(f"mp3d_{name}")
        else:
            hybrid_55d_names.append(f"rtmp2d_{name}")

    # 4. 확장 110D 풀 하이브리드 피처 생성 (RTMP 55D + MP 55D)
    X_hybrid_110d = np.hstack([X_rtmp, X_mp]).astype(np.float32)
    hybrid_110d_names = [f"rtmp_{n}" for n in feature_names_base] + [f"mp_{n}" for n in feature_names_base]

    # 결측치 정합성 방어
    X_hybrid_55d = np.nan_to_num(X_hybrid_55d, nan=0.0, posinf=0.0, neginf=0.0)
    X_hybrid_110d = np.nan_to_num(X_hybrid_110d, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[*] X_hybrid_55d shape: {X_hybrid_55d.shape}")
    print(f"[*] X_hybrid_110d shape: {X_hybrid_110d.shape}")
    print(f"[*] NaN/Inf 검출: 55D={np.isnan(X_hybrid_55d).sum()}, 110D={np.isnan(X_hybrid_110d).sum()}")

    # 5. npz 압축 캐시 저장
    os.makedirs(os.path.dirname(OUT_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_CACHE_PATH,
        X_hybrid_55d=X_hybrid_55d,
        X_hybrid_110d=X_hybrid_110d,
        y=y,
        groups=groups,
        views=views,
        clip_names=clip_names,
        feature_names_55d=hybrid_55d_names,
        feature_names_110d=hybrid_110d_names,
        degenerate_indices=DEGENERATE_RTMP_INDICES
    )
    print(f"[+] 성공적으로 하이브리드 피처셋 저장 완료: {OUT_CACHE_PATH}")


if __name__ == '__main__':
    build_and_save_hybrid_features()
