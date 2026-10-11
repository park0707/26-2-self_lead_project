# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_c1_v5_features.py
Purpose: SpinePose 9대 척추 분절 기반 6차원 피처 추출 및 v5 피처셋 구축
"""

import os
import sys
import glob
import json
import time
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c1_spinepose_extractor import (
    BarbellSquatC1SpinePoseExtractor,
    C1_SPINEPOSE_DENSE_FEATURE_NAMES_6
)

OUT_NPZ = 'd:/src/models/second_models/babel_squat/squat_c1_v5_spinepose_feats.npz'

def interpolate_to_64f(arr):
    T = arr.shape[0]
    if T == 64:
        return arr
    t_orig = np.linspace(0, 1, T)
    t_64 = np.linspace(0, 1, 64)
    kind = 'cubic' if T >= 4 else 'linear'
    f = interp1d(t_orig, arr, kind=kind, axis=0, fill_value='extrapolate')
    return f(t_64).astype(np.float32)

def build_all_v5_spinepose_features():
    print("=" * 90)
    print("  [vidio-health-trainer] c1 v5 SpinePose 9대 척추 분절 기반 6D 피처셋 구축 시작...")
    print("=" * 90)

    m = np.load('d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz', allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    sids = m['sequence_ids'][sq_mask]
    base_sids = m['base_sequence_ids'][sq_mask]
    N = len(sids)

    split_dir = glob.glob('d:/Data/전처리_데이터/MediaPipe_횟수분할/*01*')[0]
    json_dir = glob.glob('d:/Data/01_raw_data/labels_json/*01*/3D')[0]

    json_cache = {}
    v5_dense_feats = np.zeros((N, 6), dtype=np.float32)

    t0 = time.time()
    success_count = 0
    fallback_count = 0

    for i, (sid, bsid) in enumerate(zip(sids, base_sids)):
        if (i + 1) % 400 == 0 or (i + 1) == N:
            print(f"    - 진행률: {i+1}/{N} ({((i+1)/N)*100:.1f}%) | 성공: {success_count}개 ({time.time()-t0:.1f}초)")

        npz_p = os.path.join(split_dir, f'{sid}.npz')
        if not os.path.exists(npz_p):
            fallback_count += 1
            continue

        d_split = np.load(npz_p, allow_pickle=True)
        sf = int(d_split['orig_start_frame'])
        ef = int(d_split['orig_end_frame'])

        jp = os.path.join(json_dir, f'{bsid}_3D.json')
        if bsid not in json_cache:
            if os.path.exists(jp):
                try:
                    json_cache[bsid] = json.load(open(jp, encoding='utf-8'))
                except Exception:
                    json_cache[bsid] = None
            else:
                json_cache[bsid] = None

        jdata = json_cache[bsid]
        if jdata is None or 'frames' not in jdata:
            fallback_count += 1
            continue

        frames = jdata['frames'][sf:ef+1]
        if len(frames) < 2:
            fallback_count += 1
            continue

        pts_list = [f.get('pts', {}) for f in frames]
        try:
            neck_raw = np.array([[f['Neck']['x'], f['Neck']['y'], f['Neck']['z']] for f in pts_list], dtype=np.float32)
            back_raw = np.array([[f['Back']['x'], f['Back']['y'], f['Back']['z']] for f in pts_list], dtype=np.float32)
            waist_raw = np.array([[f['Waist']['x'], f['Waist']['y'], f['Waist']['z']] for f in pts_list], dtype=np.float32)
            lh_raw = np.array([[f['Left Hip']['x'], f['Left Hip']['y'], f['Left Hip']['z']] for f in pts_list], dtype=np.float32)
            rh_raw = np.array([[f['Right Hip']['x'], f['Right Hip']['y'], f['Right Hip']['z']] for f in pts_list], dtype=np.float32)
            lk_raw = np.array([[f['Left Knee']['x'], f['Left Knee']['y'], f['Left Knee']['z']] for f in pts_list], dtype=np.float32)
            rk_raw = np.array([[f['Right Knee']['x'], f['Right Knee']['y'], f['Right Knee']['z']] for f in pts_list], dtype=np.float32)
        except KeyError:
            fallback_count += 1
            continue

        neck = interpolate_to_64f(neck_raw)
        back = interpolate_to_64f(back_raw)
        waist = interpolate_to_64f(waist_raw)
        lh = interpolate_to_64f(lh_raw)
        rh = interpolate_to_64f(rh_raw)
        lk = interpolate_to_64f(lk_raw)
        rk = interpolate_to_64f(rk_raw)

        pelvis = (lh + rh) / 2.0
        knee_y = (lk[:, 1] + rk[:, 1]) / 2.0

        # SpinePose 9대 척추 분절 복원
        spine_9 = BarbellSquatC1SpinePoseExtractor.interpolate_dense_spine_9(neck, back, waist, pelvis)
        
        # 6대 고정밀 척추 피처 추출
        dense_vec = BarbellSquatC1SpinePoseExtractor.compute_spinepose_dense_features(spine_9, lh, rh, knee_y)

        if np.all(dense_vec == 0.0):
            fallback_count += 1
        else:
            success_count += 1

        v5_dense_feats[i] = dense_vec

    print(f"\n[*] 최종 빌드 완료: 총 {N}건 중 성공 {success_count}건 ({success_count/N*100:.1f}%), 폴백 {fallback_count}건 ({fallback_count/N*100:.1f}%)")

    np.savez_compressed(
        OUT_NPZ,
        v5_spinepose_feats=v5_dense_feats,
        feature_names=C1_SPINEPOSE_DENSE_FEATURE_NAMES_6,
        sequence_ids=sids,
        base_sequence_ids=base_sids
    )
    print(f"[*] c1 v5 SpinePose 피처 저장 완료: {OUT_NPZ} (크기: {os.path.getsize(OUT_NPZ)/1024:.1f} KB)")

if __name__ == '__main__':
    build_all_v5_spinepose_features()
