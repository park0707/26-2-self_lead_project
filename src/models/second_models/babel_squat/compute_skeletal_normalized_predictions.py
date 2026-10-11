# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: compute_skeletal_normalized_predictions.py
Purpose: AI-Hub 2,605건 전수(5대 시점 x 9명 피험자)에 대해 Skeletal Normalization(Dual-Stream) 적용 후
         c1(척추), c3(무릎) 결함 확률 및 9D 운동사슬 피처 추출 및 저장
"""

import os
import sys
import time
import glob
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features.skeletal_normalizer import normalize_skeletal_lengths
from src.models.second_models.babel_squat.c1_model_v11_1 import BabelSquatC1ModelV11_1
from src.models.second_models.babel_squat.c3_model_v6 import BabelSquatC3ModelV6
from src.models.second_models.babel_squat.babel_squat_integrated_v10_2 import BabelSquatIntegratedEngineV10_2

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
SAVE_PATH = os.path.join(CURRENT_DIR, "squat_skeletal_normalized_predictions_2605.npz")

print("=" * 100)
print("  [vidio-health-trainer] 2,605건 전수 Skeletal Normalization & Dual-Stream 피처 추출")
print("=" * 100)

# 1. 메타 데이터 순서 로드
meta = np.load(META_PATH, allow_pickle=True)
sequence_ids = meta['sequence_ids']
N = len(sequence_ids)
print(f"[*] 대상 시퀀스 수: {N}건")

# 2. 모델 초기화
print("[*] c1, c3 모델 및 운동사슬 추출기 로드 중...")
c1_model = BabelSquatC1ModelV11_1()
c3_model = BabelSquatC3ModelV6()
engine_helper = BabelSquatIntegratedEngineV10_2()

# 3. 2,605건 순회 추출
p_c1_norm = np.zeros(N, dtype=np.float32)
p_c3_norm = np.zeros(N, dtype=np.float32)
kinetic_9d_norm = np.zeros((N, 9), dtype=np.float32)
femur_wobble_raw = np.zeros(N, dtype=np.float32)
tibia_wobble_raw = np.zeros(N, dtype=np.float32)
residual_norms = np.zeros(N, dtype=np.float32)

t0 = time.time()
print("[*] 2,605건 처리 시작...")

for i, sid in enumerate(sequence_ids):
    file_path = os.path.join(DATA_DIR, f"{sid}.npz")
    d = np.load(file_path, allow_pickle=True)
    kps_raw = d['kps_33_3d_64f']

    # Skeletal Normalization (Dual-Stream)
    st_a, st_b, metrics = normalize_skeletal_lengths(kps_raw)

    # c1 & c3 예측 (Stream A)
    res_c1 = c1_model.predict_selective(st_a)
    res_c3 = c3_model.predict_selective(st_a)
    p1_val = res_c1.get('defect_probability')
    p3_val = res_c3.get('defect_probability')
    p_c1_norm[i] = 0.5 if p1_val is None else float(p1_val)
    p_c3_norm[i] = 0.5 if p3_val is None else float(p3_val)

    # 9D 운동사슬 (Stream A)
    k_vec = engine_helper.compute_9d_kinetic(st_a)
    kinetic_9d_norm[i] = k_vec

    # 품질 지표 수집
    femur_wobble_raw[i] = metrics['raw_femur_wobble_std']
    tibia_wobble_raw[i] = metrics['raw_tibia_wobble_std']
    residual_norms[i]   = metrics['stream_residual_norm']

    if (i + 1) % 500 == 0 or (i + 1) == N:
        elapsed = time.time() - t0
        speed = (i + 1) / elapsed
        rem = (N - (i + 1)) / speed if speed > 0 else 0
        print(f"  [{i+1:4d}/{N}] 처리 완료 ({speed:.1f} seq/s, 잔여 약 {rem:.1f}초)")

# 저장
np.savez_compressed(
    SAVE_PATH,
    sequence_ids=sequence_ids,
    p_c1_norm=p_c1_norm,
    p_c3_norm=p_c3_norm,
    kinetic_9d_norm=kinetic_9d_norm,
    femur_wobble_raw=femur_wobble_raw,
    tibia_wobble_raw=tibia_wobble_raw,
    residual_norms=residual_norms
)

print(f"\n[★ 완료] 2,605건 정규화 결과 저장 완료 -> {SAVE_PATH}")
print(f"  평균 대퇴골 Wobble Std: {np.mean(femur_wobble_raw):.4f}")
print(f"  평균 경골 Wobble Std:   {np.mean(tibia_wobble_raw):.4f}")
print(f"  평균 스트림 간 잔차:    {np.mean(residual_norms):.4f}")
print("=" * 100)
