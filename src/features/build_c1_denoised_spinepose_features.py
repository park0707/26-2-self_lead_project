# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_c1_denoised_spinepose_features.py
Purpose: One-Euro + Segment Clamping 노이즈 제거가 완료된 청정 SpinePose(64, 9, 3)로부터
         Model-B용 38D, 27D, 24D 피처셋 재구축
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c1_model_v11_a import extract_c1_v11_a_features_from_seq
from src.models.second_models.babel_squat.c1_model_v11_a_diet import extract_c1_v11_a_diet_features_from_seq
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
DENOISED_SPINE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_denoised_64f.npz"
OUT_FEATS_PATH = r"D:\Data\피처_데이터\squat_c1_v11_denoised_spinepose_feats.npz"

def build():
    print("=" * 110)
    print("  [vidio-health-trainer] 청정 SpinePose 기반 c1 피처셋 일괄 생성 파이프라인")
    print("=" * 110)

    t0 = time.time()
    master = np.load(MASTER_PATH, allow_pickle=True)
    master_sids = master['sequence_ids']
    N = len(master_sids)

    spine_data = np.load(DENOISED_SPINE_PATH, allow_pickle=True)
    spine_9_tensor = spine_data['spine_9_64f_vision']

    X_38d = np.zeros((N, 38), dtype=np.float32)
    X_27d = np.zeros((N, 27), dtype=np.float32)
    X_temporal = np.zeros((N, 64, 24), dtype=np.float32)

    for i, sid in enumerate(master_sids):
        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"    - 진행률: {i+1}/{N} ({(i+1)/N*100:.1f}%) | 소요 시간: {time.time()-t0:.1f}초", flush=True)

        mp_path = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
        mp_d = np.load(mp_path, allow_pickle=True)
        kps_33 = mp_d['kps_33_3d_64f']
        sp_9 = spine_9_tensor[i]

        X_38d[i] = extract_c1_v11_a_features_from_seq(kps_33, sp_9)
        X_27d[i] = extract_c1_v11_a_diet_features_from_seq(kps_33, sp_9)
        X_temporal[i] = extract_temporal_24d(kps_33, sp_9)

    os.makedirs(os.path.dirname(OUT_FEATS_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_FEATS_PATH,
        X_38d_vision=X_38d,
        X_27d_vision=X_27d,
        X_temporal_vision=X_temporal,
        sequence_ids=master_sids
    )
    print(f"[OK] 청정 SpinePose 피처셋 저장 완료: {OUT_FEATS_PATH} ({time.time()-t0:.1f}초)")

if __name__ == '__main__':
    build()
