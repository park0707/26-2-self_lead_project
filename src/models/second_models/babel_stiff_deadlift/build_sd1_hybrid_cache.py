# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd1. 무릎 구부린 채 고정 위반 (Knee Fixation Violation)
Purpose: MediaPipe 3D (24D) + RTMPose 2D (41D) + View Conditioning 캐시 구축
"""

import os
import sys
import pickle
import time
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sd1_v8_features import extract_sd1_v8_features, FEATURE_NAMES_30D_SD1
from sd1_rtmp_2_features import extract_sd1_rtmp_2_features, FEATURE_NAMES_47D_SD1_RTMP_2

CACHE_PKL = os.path.join(CURRENT_DIR, "stiff_d1_v3_dataset_cache.pkl")
OUT_CACHE_NPZ = os.path.join(CURRENT_DIR, "stiff_sd1_hybrid_features_cache.npz")

VIEW_TO_INT = {
    "view1": 1, # 시점 E: 좌측 후방 45°
    "view2": 2, # 시점 B: 우측 전방 45°
    "view3": 3, # 시점 C: 정면 0°
    "view4": 4, # 시점 D: 좌측 전방 45°
    "view5": 5  # 시점 A: 우측 후방 45°
}

def main():
    print("[*] 메타데이터 로드 중...", flush=True)
    with open(CACHE_PKL, "rb") as f:
        c = pickle.load(f)
    df = c["df"]
    N = len(df)
    print(f"[*] 총 {N}건 시퀀스에 대해 MediaPipe + RTMPose 하이브리드 피처 추출 시작...", flush=True)
    
    mp_names_24 = FEATURE_NAMES_30D_SD1[:24]
    rtmp_names_41 = FEATURE_NAMES_47D_SD1_RTMP_2[:41]
    
    X_mp_24 = np.zeros((N, 24), dtype=np.float32)
    X_rtmp_41 = np.zeros((N, 41), dtype=np.float32)
    view_ids = np.zeros(N, dtype=np.float32)
    view_oh = np.zeros((N, 5), dtype=np.float32)
    
    y = np.zeros(N, dtype=np.int32)
    subjects = np.array(df["subj"].values)
    views = np.array(df["view"].values)
    keys = np.array(df["key"].values)
    seq_ids = []
    
    t0 = time.time()
    for i in range(N):
        row = df.iloc[i]
        mp_path = row["path"]
        rtmp_path = mp_path.replace("MediaPipe_Full33_64F", "RTMPose_Halpe26_64F")
        v = row["view"]
        k = str(row["key"])
        
        # sd1 결함 = Key213, Key214, Key215 (390건)
        y[i] = 1 if k in ["Key213", "Key214", "Key215"] else 0
        sid = os.path.splitext(os.path.basename(mp_path))[0]
        seq_ids.append(sid)
        
        # Load kps
        k_mp = np.load(mp_path)["kps_33_3d_64f"]
        k_rtmp = np.load(rtmp_path)["kps_26_64f"]
        
        # Extract
        f_mp = extract_sd1_v8_features(k_mp, v)
        f_rtmp = extract_sd1_rtmp_2_features(k_rtmp, v)
        
        X_mp_24[i] = f_mp[:24]
        X_rtmp_41[i] = f_rtmp[:41]
        
        vid = VIEW_TO_INT[v]
        view_ids[i] = vid
        view_oh[i, vid - 1] = 1.0
        
        if (i + 1) % 200 == 0 or (i + 1) == N:
            print(f"  - [{i+1}/{N}] 추출 진행 중... ({time.time() - t0:.1f}s)", flush=True)
            
    print(f"[+] 피처 추출 완료: {time.time() - t0:.1f}s 소요", flush=True)
    
    # Save cache
    np.savez_compressed(
        OUT_CACHE_NPZ,
        X_mp_24=X_mp_24,
        X_rtmp_41=X_rtmp_41,
        view_ids=view_ids,
        view_oh=view_oh,
        y=y,
        subjects=subjects,
        views=views,
        keys=keys,
        seq_ids=np.array(seq_ids),
        mp_names_24=np.array(mp_names_24),
        rtmp_names_41=np.array(rtmp_names_41)
    )
    print(f"[+] 캐시 저장 완료: {OUT_CACHE_NPZ}", flush=True)

if __name__ == "__main__":
    main()
