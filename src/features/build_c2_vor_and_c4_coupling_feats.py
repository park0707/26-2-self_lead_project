# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_c2_vor_and_c4_coupling_feats.py
Purpose: 원천 64F 스플라인 2,605개 전수에 대한 c2 VOR 18D 및 c4 16D 정밀 시계열 피처 추출 및 영구 캐시
         - 규정 준수:
           1) 베이스라인 코드 계승 원칙 (c2_model_v1.py 18D VOR 및 run_c4_extended_pipeline 16D 연산기 100% 계승)
           2) AI-Hub 3D JSON 0% 참조, 순수 MediaPipe 64F 기반
           3) 저장 위치: D:\Data\피처_데이터\squat_c2_vor_and_c4_coupling_feats.npz
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, 'd:/')
from src.features.kinematics_3d import KinematicsExtractor3D
from src.models.second_models.babel_squat.c2_model_v1 import BabelSquatC2ModelV1, FEATURE_NAMES_18
from src.models.second_models.babel_squat.run_c4_extended_pipeline import extract_c4_16d_features, FEATURE_NAMES_16_C4

RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
OUT_PATH = r"D:\Data\피처_데이터\squat_c2_vor_and_c4_coupling_feats.npz"


def main():
    print("=" * 110, flush=True)
    print("  [vidio-health-trainer] c2 VOR 18D & c4 커플링 16D 전수 2,605개 추출 빌더", flush=True)
    print("  ★ 베이스라인 코드 계승 원칙: c2 v4 VOR 18D 및 c4 v3 정강이-발목 커플링 100% 온전 계승", flush=True)
    print("=" * 110, flush=True)

    master_d = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master_d['sequence_ids']
    N = len(seq_ids)

    print(f"[*] 총 처리 대상 시퀀스: {N}개", flush=True)

    c2_vor_list = []
    c4_coup_list = []

    t0 = time.time()

    for i, sid in enumerate(seq_ids, 1):
        fp = os.path.join(RAW_DIR, f"{sid}.npz")
        d = np.load(fp)
        kps_15 = d['kps_15_3d_64f'] # (64, 15, 3)

        # 1. Format B 시계열 텐서 생성 (64, 92)
        fb = KinematicsExtractor3D.build_format_b_timeseries_3d(kps_15) # (64, 92)
        fb_batch = fb[np.newaxis, ...] # (1, 64, 92)

        # 2. c2 VOR 18D 피처 추출
        c2_18d = BabelSquatC2ModelV1.extract_features(fb_batch, use_pruned=True)[0] # (18,)

        # 3. c4 16D 발목 지면 피처 추출
        c4_16d = extract_c4_16d_features(fb_batch)[0] # (16,)

        c2_vor_list.append(c2_18d)
        c4_coup_list.append(c4_16d)

        if i % 500 == 0 or i == N:
            elapsed = time.time() - t0
            speed = i / elapsed
            print(f"  - 진행률: {i:4d}/{N} ({i/N*100:5.1f}%) | 추출 속도: {speed:.1f}개/초 | 경과: {elapsed:.1f}초", flush=True)

    X_c2_vor = np.array(c2_vor_list, dtype=np.float32)
    X_c4_coup = np.array(c4_coup_list, dtype=np.float32)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X_c2_vor_18d=X_c2_vor,
        c2_vor_feature_names=np.array(FEATURE_NAMES_18),
        X_c4_coup_16d=X_c4_coup,
        c4_coup_feature_names=np.array(FEATURE_NAMES_16_C4),
        sequence_ids=seq_ids
    )

    print("\n" + "=" * 110, flush=True)
    print(f"[+] c2 VOR 18D & c4 16D 피처 데이터셋 구축 완료!")
    print(f"  • c2 VOR 18D 형상 : {X_c2_vor.shape}")
    print(f"  • c4 16D 형상     : {X_c4_coup.shape}")
    print(f"  • 저장 위치       : {OUT_PATH}")
    print("=" * 110, flush=True)


if __name__ == '__main__':
    main()
