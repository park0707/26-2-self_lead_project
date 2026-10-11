# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_lunge_c4_dataset_cache_flip.py
Purpose: 런지 동작에서 항상 '오른발이 앞으로 나오도록' 좌우 반전(Mirroring)을 적용하여 
         특징을 재추출한 캐시 구축 (v3의 24D 기반)
"""

import os
import sys
import glob
import numpy as np
import math
from tqdm import tqdm

sys.path.insert(0, r"D:/")
from src.features.lunge_c4_features_v3 import (
    extract_lunge_c4_features_v3,
    extract_lunge_c4_temporal_features,
    LUNGE_C4_FEATURE_NAMES_V3
)

MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
CACHE_V3 = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz"
OUT_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_flip.npz"

def _angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.sum(ba * bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_forward_leg_is_left(seq_33):
    from scipy.ndimage import gaussian_filter1d
    coords = seq_33[:, :, :3].astype(np.float32)
    smoothed = np.zeros_like(coords)
    for j in range(coords.shape[1]):
        for c in range(3):
            smoothed[:, j, c] = gaussian_filter1d(coords[:, j, c], sigma=1.0)
            
    # Find t_bottom
    hip_mid = (smoothed[:, 23] + smoothed[:, 24]) * 0.5
    t_bottom = np.argmax(hip_mid[:, 1])
    
    # Calculate knee angles at bottom
    left_knee_angle = _angle(smoothed[t_bottom, 23], smoothed[t_bottom, 25], smoothed[t_bottom, 27])
    right_knee_angle = _angle(smoothed[t_bottom, 24], smoothed[t_bottom, 26], smoothed[t_bottom, 28])
    
    # The leg with the smaller knee angle (more bent) is usually the forward leg in a lunge
    # Or, the forward ankle has a larger Z (or smaller Z? In MP, -Z is closer to camera).
    # Since we use left_knee_angle < right_knee_angle in v3, let's stick to it.
    return left_knee_angle < right_knee_angle

def mirror_pose(seq_33: np.ndarray) -> np.ndarray:
    out = seq_33.copy()
    out[:, :, 0] = -out[:, :, 0] # Flip X
    SWAP_PAIRS = [
        (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
        (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
        (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
    ]
    for left, right in SWAP_PAIRS:
        temp = out[:, left, :].copy()
        out[:, left, :] = out[:, right, :]
        out[:, right, :] = temp
    return out

def build_cache():
    print("=" * 80)
    print("  [vidio-health-trainer] 런지 좌우 반전(오른발 전방 통일) 24D 피처 캐시 구축")
    print("=" * 80)

    d_v3 = np.load(CACHE_V3, allow_pickle=True)
    y = d_v3['y']
    subjects = d_v3['subjects']
    views = d_v3['views']
    clip_ids = d_v3['clip_ids']
    
    print(f"[*] v3 캐시 로드 완료: {len(y)}개 클립")

    X_24d_list = []
    X_temp_14d_list = []
    
    missing = 0
    flipped_count = 0
    
    for i, cid in enumerate(tqdm(clip_ids)):
        mp_path = os.path.join(MP33_DIR, f"{cid}.npz")
        if not os.path.exists(mp_path):
            missing += 1
            # Fallback to zeros just in case, though it shouldn't happen
            X_24d_list.append(np.zeros(24, dtype=np.float32))
            X_temp_14d_list.append(np.zeros((64, 14), dtype=np.float32))
            continue
            
        data = np.load(mp_path)
        seq_33 = data['kps_33_3d_64f']
        
        is_left_lead = get_forward_leg_is_left(seq_33)
        if is_left_lead:
            seq_33 = mirror_pose(seq_33)
            flipped_count += 1
            
        feats_24 = extract_lunge_c4_features_v3(seq_33)
        temp_14 = extract_lunge_c4_temporal_features(seq_33)
        
        X_24d_list.append(feats_24)
        X_temp_14d_list.append(temp_14)

    X_24d = np.array(X_24d_list, dtype=np.float32)
    X_temp_14d = np.array(X_temp_14d_list, dtype=np.float32)

    print(f"\n[+] 추출 완료. 누락: {missing}, 좌우 반전 적용 횟수: {flipped_count}/{len(clip_ids)}")
    print(f"[*] X_24d: {X_24d.shape}, X_temp_14d: {X_temp_14d.shape}")

    np.savez_compressed(
        OUT_CACHE_PATH,
        X_24d=X_24d,
        X_temp_14d=X_temp_14d,
        y=y,
        subjects=subjects,
        views=views,
        clip_ids=clip_ids,
        feature_names=LUNGE_C4_FEATURE_NAMES_V3
    )
    print(f"[+] 캐시 저장 완료: {OUT_CACHE_PATH}")

if __name__ == "__main__":
    build_cache()
