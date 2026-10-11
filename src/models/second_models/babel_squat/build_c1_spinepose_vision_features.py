# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_c1_spinepose_vision_features.py
Purpose: SpinePose 순수 비전 척추 좌표(64, 9, 3)를 기반으로
         Model-B (신규 순수 비전 모델) 전용 38D, 27D, 24D 시계열 텐서 일괄 구축
- 원칙:
  기존 AI-Hub 보간 기반 Model-A와 100% 동일한 피처 추출 함수 적용 (Zero-Formula-Drift)
  오직 spine_9_64f 관절 원천만 SpinePose 비전 좌표로 교체
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
VISION_SPINE_PATH = r"D:\Data\피처_데이터\squat_c1_spinepose_true_vision_64f.npz"
OUT_VISION_FEATS_PATH = r"D:\Data\피처_데이터\squat_c1_v11_vision_spinepose_feats.npz"


def build_vision_features():
    print("=" * 110)
    print("  [vidio-health-trainer] Model-B (SpinePose 순수 비전 기반) c1 피처셋 일괄 생성")
    print("=" * 110)

    if not os.path.exists(VISION_SPINE_PATH):
        raise FileNotFoundError(f"SpinePose 순수 비전 캐시 파일이 존재하지 않습니다: {VISION_SPINE_PATH}")

    t0 = time.time()
    master = np.load(MASTER_PATH, allow_pickle=True)
    master_sids = master['sequence_ids']
    N = len(master_sids)
    print(f"[*] 총 {N}개 시퀀스 피처 변환 시작...")

    vision_data = np.load(VISION_SPINE_PATH, allow_pickle=True)
    spine_9_tensor = vision_data['spine_9_64f_vision'] # (N, 64, 9, 3)

    X_38d_vision = np.zeros((N, 38), dtype=np.float32)
    X_27d_vision = np.zeros((N, 27), dtype=np.float32)
    X_temporal_vision = np.zeros((N, 64, 24), dtype=np.float32)

    for i, sid in enumerate(master_sids):
        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"    - 진행률: {i+1}/{N} ({(i+1)/N*100:.1f}%) | 소요 시간: {time.time()-t0:.1f}초", flush=True)

        mp_path = os.path.join(RAW_MP33_DIR, f"{sid}.npz")
        mp_d = np.load(mp_path, allow_pickle=True)
        kps_33_64f = mp_d['kps_33_3d_64f'] # (64, 33, 3)
        spine_9 = spine_9_tensor[i]         # (64, 9, 3)

        f38 = extract_c1_v11_a_features_from_seq(kps_33_64f, spine_9)
        f27 = extract_c1_v11_a_diet_features_from_seq(kps_33_64f, spine_9)
        t24 = extract_temporal_24d(kps_33_64f, spine_9)

        X_38d_vision[i] = f38
        X_27d_vision[i] = f27
        X_temporal_vision[i] = t24

    os.makedirs(os.path.dirname(OUT_VISION_FEATS_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_VISION_FEATS_PATH,
        X_38d_vision=X_38d_vision,
        X_27d_vision=X_27d_vision,
        X_temporal_vision=X_temporal_vision,
        sequence_ids=master_sids
    )
    print(f"[OK] Model-B 비전 피처셋 저장 완료: {OUT_VISION_FEATS_PATH} ({time.time()-t0:.1f}초)")
    print(f"     - X_38d_vision: {X_38d_vision.shape}")
    print(f"     - X_27d_vision: {X_27d_vision.shape}")
    print(f"     - X_temporal_vision: {X_temporal_vision.shape}")


if __name__ == '__main__':
    build_vision_features()
