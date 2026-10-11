# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_c1_temporal_24d_tensor.py
Purpose: c1(척추 중립 위반) E1(TCN) 실험을 위한 2,605개 전 시퀀스 표준 24D 시계열 텐서(64, 24) 추출 및 캐싱
"""

import os
import sys
import time
import json
import glob
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.ndimage import gaussian_filter1d

RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
JSON_DIR = r"d:\Data\01_raw_data\labels_json"
OUT_TENSOR_PATH = r"D:\Data\피처_데이터\squat_c1_temporal_24d_64f.npz"


def interpolate_to_64f(arr: np.ndarray) -> np.ndarray:
    T = arr.shape[0]
    if T == 64:
        return arr
    t_orig = np.linspace(0, 1, T)
    t_64 = np.linspace(0, 1, 64)
    kind = 'cubic' if T >= 4 else 'linear'
    f = interp1d(t_orig, arr, kind=kind, axis=0, fill_value='extrapolate')
    return f(t_64).astype(np.float32)


def build_spine_cache(base_sids: np.ndarray, json_map: dict) -> dict:
    spine_cache = {}
    z_src = np.array([0.0, 0.25, 0.60, 0.85], dtype=np.float32)
    z_dst = np.array([0.0, 0.12, 0.25, 0.42, 0.60, 0.75, 0.85, 0.92, 1.00], dtype=np.float32)

    for b in base_sids:
        jp = json_map.get(b)
        if jp is None or not os.path.exists(jp):
            spine_cache[b] = None
            continue
        with open(jp, 'r', encoding='utf-8') as f:
            jd = json.load(f)
        frames = jd.get('frames', [])
        pts_list = [f.get('pts', {}) for f in frames]
        neck_raw = np.array([[f['Neck']['x'], f['Neck']['y'], f['Neck']['z']] for f in pts_list], dtype=np.float32)
        back_raw = np.array([[f['Back']['x'], f['Back']['y'], f['Back']['z']] for f in pts_list], dtype=np.float32)
        waist_raw = np.array([[f['Waist']['x'], f['Waist']['y'], f['Waist']['z']] for f in pts_list], dtype=np.float32)
        lh_raw = np.array([[f['Left Hip']['x'], f['Left Hip']['y'], f['Left Hip']['z']] for f in pts_list], dtype=np.float32)
        rh_raw = np.array([[f['Right Hip']['x'], f['Right Hip']['y'], f['Right Hip']['z']] for f in pts_list], dtype=np.float32)

        neck = interpolate_to_64f(neck_raw)
        back = interpolate_to_64f(back_raw)
        waist = interpolate_to_64f(waist_raw)
        pelvis = interpolate_to_64f((lh_raw + rh_raw) / 2.0)

        T = 64
        s9 = np.zeros((T, 9, 3), dtype=np.float32)
        for t in range(T):
            pts_4 = np.array([pelvis[t], waist[t], back[t], neck[t]], dtype=np.float32)
            fx = PchipInterpolator(z_src, pts_4[:, 0], extrapolate=True)
            fy = PchipInterpolator(z_src, pts_4[:, 1], extrapolate=True)
            fz = PchipInterpolator(z_src, pts_4[:, 2], extrapolate=True)
            s9[t, :, 0] = fx(z_dst)
            s9[t, :, 1] = fy(z_dst)
            s9[t, :, 2] = fz(z_dst)
        spine_cache[b] = s9
    return spine_cache


def extract_temporal_24d(kps_33: np.ndarray, spine_9: np.ndarray) -> np.ndarray:
    T = 64
    mid_hip = (kps_33[:, 23] + kps_33[:, 24]) / 2.0
    mid_sh = (kps_33[:, 11] + kps_33[:, 12]) / 2.0
    mid_knee = (kps_33[:, 25] + kps_33[:, 26]) / 2.0

    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)
    v_sh = kps_33[:, 11] - kps_33[:, 12]
    z_hat = np.cross(y_hat, v_sh)
    z_hat /= (np.linalg.norm(z_hat, axis=1, keepdims=True) + 1e-7)
    x_hat = np.cross(y_hat, z_hat)
    x_hat /= (np.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-7)

    if spine_9 is None:
        spine_9 = np.zeros((T, 9, 3), dtype=np.float32)
        weights = [0.0, 0.12, 0.25, 0.42, 0.60, 0.75, 0.85, 0.92, 1.00]
        for idx_w, w in enumerate(weights):
            spine_9[:, idx_w, :] = mid_hip + w * (mid_sh - mid_hip)

    # 1. 핵심 척추 분절 (S1=0, L5=1, L3=2, L1=3, T8=4, C7=6)의 체간 로컬 y, z (12D)
    pts_spine = spine_9[:, [0, 1, 2, 3, 4, 6], :] # (64, 6, 3)
    spine_rel = pts_spine - mid_hip[:, np.newaxis, :]
    y_spine = np.sum(spine_rel * y_hat[:, np.newaxis, :], axis=2) # (64, 6)
    z_spine = np.sum(spine_rel * z_hat[:, np.newaxis, :], axis=2) # (64, 6)

    # 2. 골반, 어깨, 무릎 로컬 y, z (6D)
    pts_body = np.stack([mid_hip, mid_sh, mid_knee], axis=1) # (64, 3, 3)
    body_rel = pts_body - mid_hip[:, np.newaxis, :]
    y_body = np.sum(body_rel * y_hat[:, np.newaxis, :], axis=2) # (64, 3)
    z_body = np.sum(body_rel * z_hat[:, np.newaxis, :], axis=2) # (64, 3)

    # 3. 생체역학 각도 시계열 6개 (6D)
    # (1) 요추 Cobb 각도 시계열
    v_l5_s1 = spine_9[:, 0] - spine_9[:, 1]
    v_l3_l1 = spine_9[:, 3] - spine_9[:, 2]
    dot_cobb = np.sum(v_l5_s1 * v_l3_l1, axis=1)
    norm_l5 = np.linalg.norm(v_l5_s1, axis=1) + 1e-6
    norm_l1 = np.linalg.norm(v_l3_l1, axis=1) + 1e-6
    cross_cobb = np.cross(v_l5_s1, v_l3_l1)
    sin_cobb = np.sum(cross_cobb * x_hat, axis=1) / (norm_l5 * norm_l1)
    cos_cobb = np.clip(dot_cobb / (norm_l5 * norm_l1), -1.0, 1.0)
    cobb_s = gaussian_filter1d(np.degrees(np.arctan2(sin_cobb, cos_cobb)), 1.0) / 30.0 # 정규화

    # (2) 흉요추 곡률 각도 시계열
    w_lower = spine_9[:, 4] - spine_9[:, 2]
    v_upper = spine_9[:, 6] - spine_9[:, 4]
    cross_tl = np.cross(w_lower, v_upper)
    dot_tl = np.sum(w_lower * v_upper, axis=1)
    norm_w = np.linalg.norm(w_lower, axis=1) + 1e-6
    norm_u = np.linalg.norm(v_upper, axis=1) + 1e-6
    sin_tl = np.sum(cross_tl * x_hat, axis=1) / (norm_w * norm_u)
    cos_tl = np.clip(dot_tl / (norm_w * norm_u), -1.0, 1.0)
    phi_tl_s = gaussian_filter1d(np.degrees(np.arctan2(sin_tl, cos_tl)), 1.0) / 30.0

    # (3) 요추-골반 상대각 시계열
    v_pelvis = mid_hip - spine_9[:, 2]
    cross_lp = np.cross(-w_lower, v_pelvis)
    dot_lp = np.sum((-w_lower) * v_pelvis, axis=1)
    norm_p = np.linalg.norm(v_pelvis, axis=1) + 1e-6
    sin_lp = np.sum(cross_lp * x_hat, axis=1) / (norm_w * norm_p)
    cos_lp = np.clip(dot_lp / (norm_w * norm_p), -1.0, 1.0)
    phi_lp_s = gaussian_filter1d(np.degrees(np.arctan2(sin_lp, cos_lp)), 1.0) / 30.0

    # (4) 골반 대비 L5 후방 말림 변위 (cm / 10.0)
    diff_l5_s1 = spine_9[:, 1] - spine_9[:, 0]
    retro_z_s = gaussian_filter1d(np.sum(diff_l5_s1 * z_hat, axis=1) * 100.0, 1.0) / 10.0

    # (5) 체간 굴곡 각도 시계열 (deg / 90.0)
    v_vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    trunk_angles = np.degrees(np.arccos(np.clip(
        np.sum(v_torso * v_vert, axis=1) / (np.linalg.norm(v_torso, axis=1) + 1e-7), -1.0, 1.0
    )))
    trunk_s = gaussian_filter1d(trunk_angles, 1.0) / 90.0

    # (6) 골반 하강 위상 (y좌표 정규화)
    pelvis_y = mid_hip[:, 1]
    pelvis_y_s = gaussian_filter1d(pelvis_y, 1.2)
    p_min, p_max = np.min(pelvis_y_s), np.max(pelvis_y_s)
    phase_s = (pelvis_y_s - p_min) / (p_max - p_min + 1e-6)

    # 결합: (64, 12 + 6 + 6) = (64, 24)
    angles_6 = np.stack([cobb_s, phi_tl_s, phi_lp_s, retro_z_s, trunk_s, phase_s], axis=1)
    tensor_24 = np.concatenate([y_spine, z_spine, y_body, z_body, angles_6], axis=1).astype(np.float32)
    return np.nan_to_num(tensor_24, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    if os.path.exists(OUT_TENSOR_PATH):
        print(f"[*] 기존 시계열 텐서 캐시 파일 발견: {OUT_TENSOR_PATH}")
        return

    print("[*] 24D 시계열 텐서 생성 시작...", flush=True)
    t0 = time.time()

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    base_sids = master['base_sequence_ids']

    json_map = {}
    for jf in glob.glob(os.path.join(JSON_DIR, '**', '*_3D.json'), recursive=True):
        bn = os.path.basename(jf).replace('_3D.json', '')
        json_map[bn] = jf

    spine_cache = build_spine_cache(np.unique(base_sids), json_map)

    tensors = []
    N = len(seq_ids)
    for i, (sid, bsid) in enumerate(zip(seq_ids, base_sids), 1):
        fp = os.path.join(RAW_DIR, f"{sid}.npz")
        d = np.load(fp, allow_pickle=True)
        kps_33 = d['kps_33_3d_64f']
        s9 = spine_cache.get(bsid)

        t_24 = extract_temporal_24d(kps_33, s9)
        tensors.append(t_24)

        if i % 500 == 0 or i == N:
            print(f"    - 진행: {i}/{N} ({i/N*100:.1f}%) | {time.time()-t0:.1f}초", flush=True)

    X_temporal = np.array(tensors, dtype=np.float32) # (2605, 64, 24)
    print(f"[*] 시계열 텐서 완성: shape={X_temporal.shape} (소요: {time.time()-t0:.1f}초)")

    np.savez_compressed(OUT_TENSOR_PATH, X_temporal=X_temporal, sequence_ids=seq_ids)
    print(f"[*] 시계열 텐서 캐시 저장 완료: {OUT_TENSOR_PATH}")


if __name__ == '__main__':
    main()
