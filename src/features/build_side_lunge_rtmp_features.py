# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Module: src/features/build_side_lunge_rtmp_features.py
Purpose: RTMPose-m (Halpe26) 관절 포인트로부터 사이드 런지 세부오류 5대 모델(c1~c5)
         공식 도메인 피처 텐서 일괄 추출 및 캐싱 파이프라인
         - c1: 55D (앞다리 무릎 90도 오류, v11)
         - c2: 51D (몸통·발·무릎 방향 불일치, v7)
         - c3: 28D (안 굽히는 다리 신전 결함, v3)
         - c4: 24D (척추의 중립 위반, v4)
         - c5: 42D (상체 과도한 숙임/젖힘, v7)
         - 원천 전체 826개 폴더 x 5대 시점 = 4,130개 시퀀스 전수 커버
         - 피험자 9명 전원(1~9번) 및 공식 5대 시점(A~E, view1~view5) 전수 포함
"""

import os
import sys
import glob
import time
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Any, Optional

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SIDE_LUNGE_MODEL_DIR = r"D:\src\models\second_models\side_lunge"
if SIDE_LUNGE_MODEL_DIR not in sys.path:
    sys.path.insert(0, SIDE_LUNGE_MODEL_DIR)

from src.features.side_lunge_c1_features_v11 import extract_side_lunge_c1_feature_vector_v11
from side_lunge_c2_v7_features import extract_side_lunge_c2_v7_features_51d
from src.features.side_lunge_c3_features_v3 import extract_side_lunge_c3_features_v3_28d
from src.features.side_lunge_c4_features_v4 import extract_side_lunge_c4_features_v4
from src.features.side_lunge_c5_features_v7 import extract_side_lunge_c5_v7_feature_vector
from src.core.skeleton2d import SYMMETRIC_PAIRS_HALPE26

RTMP_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\07_사이드_런지"
OUT_NPZ_PATH = r"D:\Data\피처_데이터\side_lunge_rtmp_features.npz"

MP_DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
C1_CACHE = r"D:\Data\피처_데이터\sl1_v10_dataset_cache.pkl"
C2_CACHE = r"D:\Data\피처_데이터\sl2_v7_all_views_51d_cache.pkl"
C3_CACHE = r"D:\Data\피처_데이터\sl3_v3_dataset_cache.pkl"
C4_CACHE = r"D:\Data\피처_데이터\sl4_v3_dataset_cache.pkl"
C5_CACHE = r"D:\Data\피처_데이터\side_lunge_c5_v7_all_relabel_42d_features.npz"


def halpe26_to_mp33_unflipped(kps26: np.ndarray, is_flipped: bool) -> np.ndarray:
    """
    Halpe26 (64, 26, 3) 관절 데이터를 MediaPipe 33관절 (64, 33, 3) 규격으로 변환.
    - is_flipped가 True인 경우: 전처리 파이프라인에서 적용되었던 X축 반전 및 대칭쌍 스왑을
      역적용(Unflip)하여 원본 카메라 촬영 좌표계로 완벽 복원
    - Z 좌표는 순수 2D 추출 포인트 특성에 맞추어 0.0으로 정규화
    """
    kps = kps26.copy()
    if is_flipped:
        kps[:, :, 0] = -kps[:, :, 0]
        for l_idx, r_idx in SYMMETRIC_PAIRS_HALPE26:
            tmp = kps[:, l_idx, :].copy()
            kps[:, l_idx, :] = kps[:, r_idx, :]
            kps[:, r_idx, :] = tmp

    kps33 = np.zeros((64, 33, 3), dtype=np.float32)
    # Head & Face
    kps33[:, 0] = kps[:, 0]    # NOSE
    kps33[:, 1] = kps[:, 1]    # LEFT_EYE_INNER proxy
    kps33[:, 2] = kps[:, 1]    # LEFT_EYE
    kps33[:, 3] = kps[:, 1]    # LEFT_EYE_OUTER proxy
    kps33[:, 4] = kps[:, 2]    # RIGHT_EYE_INNER proxy
    kps33[:, 5] = kps[:, 2]    # RIGHT_EYE
    kps33[:, 6] = kps[:, 2]    # RIGHT_EYE_OUTER proxy
    kps33[:, 7] = kps[:, 3]    # LEFT_EAR
    kps33[:, 8] = kps[:, 4]    # RIGHT_EAR

    # Upper Body
    kps33[:, 11] = kps[:, 5]   # LEFT_SHOULDER
    kps33[:, 12] = kps[:, 6]   # RIGHT_SHOULDER
    kps33[:, 13] = kps[:, 7]   # LEFT_ELBOW
    kps33[:, 14] = kps[:, 8]   # RIGHT_ELBOW
    kps33[:, 15] = kps[:, 9]   # LEFT_WRIST
    kps33[:, 16] = kps[:, 10]  # RIGHT_WRIST

    # Lower Body & Feet
    kps33[:, 23] = kps[:, 11]  # LEFT_HIP
    kps33[:, 24] = kps[:, 12]  # RIGHT_HIP
    kps33[:, 25] = kps[:, 13]  # LEFT_KNEE
    kps33[:, 26] = kps[:, 14]  # RIGHT_KNEE
    kps33[:, 27] = kps[:, 15]  # LEFT_ANKLE
    kps33[:, 28] = kps[:, 16]  # RIGHT_ANKLE
    kps33[:, 29] = kps[:, 24]  # LEFT_HEEL
    kps33[:, 30] = kps[:, 25]  # RIGHT_HEEL
    kps33[:, 31] = kps[:, 20]  # LEFT_FOOT_INDEX (LEFT_BIG_TOE)
    kps33[:, 32] = kps[:, 21]  # RIGHT_FOOT_INDEX (RIGHT_BIG_TOE)

    # Z 채널은 0.0 초기화
    kps33[:, :, 2] = 0.0
    return kps33


def extract_features_single(file_path: str) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """단일 RTMPose npz 파일로부터 c1~c5 5종 피처 벡터 일괄 추출"""
    fname = os.path.basename(file_path)
    view_code = fname.replace('.npz', '').split('_')[-1]

    try:
        data = np.load(file_path)
        kps26 = data['kps_26_64f']
        is_flipped = bool(data['is_flipped'])

        kps33 = halpe26_to_mp33_unflipped(kps26, is_flipped)

        # 1. c1 (55D)
        fc1 = extract_side_lunge_c1_feature_vector_v11(kps33)

        # 2. c2 (51D)
        fc2 = extract_side_lunge_c2_v7_features_51d(kps33, view_code=view_code)

        # 3. c3 (28D)
        fc3 = extract_side_lunge_c3_features_v3_28d(kps33)

        # 4. c4 (24D)
        fc4_raw = extract_side_lunge_c4_features_v4(kps33)
        fc4 = np.array(list(fc4_raw.values()), dtype=np.float32) if isinstance(fc4_raw, dict) else np.asarray(fc4_raw, dtype=np.float32)

        # 5. c5 (42D)
        fc5 = extract_side_lunge_c5_v7_feature_vector(kps33)

        # 결측치(NaN/Inf) 정위 방어
        fc1 = np.nan_to_num(fc1, nan=0.0, posinf=0.0, neginf=0.0)
        fc2 = np.nan_to_num(fc2, nan=0.0, posinf=0.0, neginf=0.0)
        fc3 = np.nan_to_num(fc3, nan=0.0, posinf=0.0, neginf=0.0)
        fc4 = np.nan_to_num(fc4, nan=0.0, posinf=0.0, neginf=0.0)
        fc5 = np.nan_to_num(fc5, nan=0.0, posinf=0.0, neginf=0.0)

        return fname, fc1, fc2, fc3, fc4, fc5
    except Exception as e:
        print(f"[!] Error processing {fname}: {e}")
        return None


def build_and_save_side_lunge_rtmp_dataset():
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 원천 전체 826개 폴더(4,130 시퀀스)")
    print("  RTMPose Halpe26 추출 포인트 기반 c1~c5 세부 오류 5대 피처셋 일괄 생성")
    print("=" * 80)

    # 1. 파일 목록 및 정렬 (MediaPipe와 완벽히 1:1 매칭되는 정렬)
    files = sorted(glob.glob(os.path.join(RTMP_DIR, "*.npz")))
    total_files = len(files)
    print(f"[*] 발견된 RTMPose npz 파일 수: {total_files}개")
    if total_files != 4130:
        print(f"[!] 경고: 파일 수가 4,130개가 아닙니다 (현재 {total_files}개).")

    # 2. 정답 라벨 및 메타데이터 로드
    print("[*] 기존 MediaPipe 공식 캐시로부터 정합 라벨 및 메타데이터 로드 중...")
    with open(C1_CACHE, 'rb') as f:
        c1_meta = pickle.load(f)
    with open(C2_CACHE, 'rb') as f:
        c2_meta = pickle.load(f)
    with open(C3_CACHE, 'rb') as f:
        c3_meta = pickle.load(f)
    with open(C4_CACHE, 'rb') as f:
        c4_meta = pickle.load(f)
    c5_meta = np.load(C5_CACHE, allow_pickle=True)

    y_c1 = c1_meta['y'].astype(np.int32)
    y_c2 = c2_meta['y'].astype(np.int32)
    y_c4 = c4_meta['y'].astype(np.int32)
    y_c5 = c5_meta['y'].astype(np.int32)

    # c3 2,460개 벤치마크 매핑
    mp_basenames = [os.path.basename(f) for f in files]
    c3_fnames = [c + '_' + v + '.npz' for c, v in zip(c3_meta['clip_names'], c3_meta['views'])]
    c3_indices_map = {fn: idx for idx, fn in enumerate(mp_basenames)}
    c3_mask_2460 = np.zeros(total_files, dtype=bool)
    y_c3_full = np.zeros(total_files, dtype=np.int32)

    for i, c3_fn in enumerate(c3_fnames):
        if c3_fn in c3_indices_map:
            f_idx = c3_indices_map[c3_fn]
            c3_mask_2460[f_idx] = True
            y_c3_full[f_idx] = c3_meta['y'][i]

    print(f"    - c1 양성/음성: {np.sum(y_c1==1)} / {np.sum(y_c1==0)} (총 {len(y_c1)})")
    print(f"    - c2 양성/음성: {np.sum(y_c2==1)} / {np.sum(y_c2==0)} (총 {len(y_c2)})")
    print(f"    - c3 2460 양성/음성: {np.sum(c3_meta['y']==1)} / {np.sum(c3_meta['y']==0)} (총 {len(c3_meta['y'])})")
    print(f"    - c4 양성/음성: {np.sum(y_c4==1)} / {np.sum(y_c4==0)} (총 {len(y_c4)})")
    print(f"    - c5 양성/음성: {np.sum(y_c5==1)} / {np.sum(y_c5==0)} (총 {len(y_c5)})")

    # 메타 파싱
    subjects = np.zeros(total_files, dtype=np.int32)
    views = []
    clip_names = []

    for idx, f in enumerate(files):
        fname = os.path.basename(f).replace('.npz', '')
        parts = fname.split('_')
        # e.g., [오류]_D02_피험자1_Key146_view1
        s_num = int(''.join(filter(str.isdigit, parts[2])))
        subjects[idx] = s_num
        views.append(parts[-1])
        clip_names.append('_'.join(parts[:-1]))

    views = np.array(views)
    clip_names = np.array(clip_names)

    # 3. 멀티프로세싱 고속 피처 추출
    workers = min(8, os.cpu_count() or 4)
    print(f"[*] 멀티프로세싱 가동 (워커 수: {workers})...")
    t0 = time.time()

    feat_results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res in executor.map(extract_features_single, files, chunksize=25):
            if res is not None:
                fn, fc1, fc2, fc3, fc4, fc5 = res
                feat_results[fn] = (fc1, fc2, fc3, fc4, fc5)

    elapsed = time.time() - t0
    print(f"[*] 피처 추출 완료: {len(feat_results)}/{total_files} 파일 ({elapsed:.1f}초, {elapsed/total_files*1000:.1f}ms/파일)")

    # 4. 정렬된 배열 조립
    X_c1 = np.zeros((total_files, 55), dtype=np.float32)
    X_c2 = np.zeros((total_files, 51), dtype=np.float32)
    X_c3 = np.zeros((total_files, 28), dtype=np.float32)
    X_c4 = np.zeros((total_files, 24), dtype=np.float32)
    X_c5 = np.zeros((total_files, 42), dtype=np.float32)

    for idx, f in enumerate(files):
        fn = os.path.basename(f)
        fc1, fc2, fc3, fc4, fc5 = feat_results[fn]
        X_c1[idx] = fc1
        X_c2[idx] = fc2
        X_c3[idx] = fc3
        X_c4[idx] = fc4
        X_c5[idx] = fc5

    # 검증: NaNs 및 Shape
    print("[*] 피처 텐서 무결성 검증:")
    print(f"    - X_c1: {X_c1.shape}, NaNs: {np.isnan(X_c1).sum()}")
    print(f"    - X_c2: {X_c2.shape}, NaNs: {np.isnan(X_c2).sum()}")
    print(f"    - X_c3: {X_c3.shape}, NaNs: {np.isnan(X_c3).sum()}")
    print(f"    - X_c4: {X_c4.shape}, NaNs: {np.isnan(X_c4).sum()}")
    print(f"    - X_c5: {X_c5.shape}, NaNs: {np.isnan(X_c5).sum()}")

    # 5. npz 저장
    np.savez_compressed(
        OUT_NPZ_PATH,
        X_c1=X_c1,
        X_c2=X_c2,
        X_c3=X_c3,
        X_c4=X_c4,
        X_c5=X_c5,
        y_c1=y_c1,
        y_c2=y_c2,
        y_c3=y_c3_full,
        c3_mask_2460=c3_mask_2460,
        y_c4=y_c4,
        y_c5=y_c5,
        subjects=subjects,
        views=views,
        clip_names=clip_names,
        filenames=np.array(mp_basenames)
    )
    print(f"[+] 성공적으로 캐시 저장 완료: {OUT_NPZ_PATH} ({os.path.getsize(OUT_NPZ_PATH) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    build_and_save_side_lunge_rtmp_dataset()
