# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_denoise_spinepose_dataset.py
Purpose: 2,605건 SpinePose 척추 9대 포인트 시계열에 대해
         One-Euro 필터링 및 생체역학 분절 불변 클램핑 일괄 적용하여
         청정 척추 데이터셋 생성
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, 'd:/')
from src.features.spinepose_denoiser import denoise_spinepose_sequence

RAW_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_true_vision_64f.npz"
OUT_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_denoised_64f.npz"

print("[*] SpinePose 2,605건 원천 데이터 로드 중...", flush=True)
raw_d = np.load(RAW_PATH, allow_pickle=True)
raw_spines = raw_d['spine_9_64f_vision']
seq_ids = raw_d['sequence_ids']
valid_mask = raw_d['valid_mask']
N = len(raw_spines)

print(f"[*] 총 {N}건 시퀀스 노이즈 제거 파이프라인 가동 (One-Euro + Segment Clamping)...", flush=True)
t0 = time.time()
clean_spines = np.zeros_like(raw_spines)

# 지터 노이즈 측정 (1차 미분 가속도 변화량 측정)
raw_jerk_list = []
clean_jerk_list = []

for i in range(N):
    sp = raw_spines[i]
    cl = denoise_spinepose_sequence(sp, fps=30.0, min_cutoff=1.5, beta=0.005, segment_tol=0.08)
    clean_spines[i] = cl

    # 샘플링하여 지터 감소율 측정
    if i % 100 == 0:
        # 가속도(2차 미분) 노름 계산
        acc_raw = np.diff(sp, n=2, axis=0)
        acc_cl  = np.diff(cl, n=2, axis=0)
        raw_jerk_list.append(np.mean(np.linalg.norm(acc_raw, axis=-1)))
        clean_jerk_list.append(np.mean(np.linalg.norm(acc_cl, axis=-1)))

np.savez_compressed(
    OUT_PATH,
    spine_9_64f_vision=clean_spines,
    sequence_ids=seq_ids,
    valid_mask=valid_mask
)

elapsed = time.time() - t0
mean_raw_jerk = np.mean(raw_jerk_list)
mean_clean_jerk = np.mean(clean_jerk_list)
jitter_reduction = (1.0 - mean_clean_jerk / mean_raw_jerk) * 100.0

print(f"[OK] {N}건 청정 척추 데이터셋 생성 완료! ({elapsed:.1f}초)", flush=True)
print(f"     - 원시 가속도 노이즈 평균: {mean_raw_jerk:.4f}", flush=True)
print(f"     - 필터링 가속도 노이즈 평균: {mean_clean_jerk:.4f}", flush=True)
print(f"     - 고주파 지터 억제율: {jitter_reduction:.1f}% 감소 달성!", flush=True)
print(f"     - 저장 경로: {OUT_PATH}", flush=True)
