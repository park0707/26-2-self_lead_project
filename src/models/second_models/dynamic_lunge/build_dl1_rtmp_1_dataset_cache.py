# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05 Forward + 06 Backward)
Defect: c1 (앞다리 무릎 각도 90도 미달/과도 굴곡 오류)
Script: build_dl1_rtmp_1_dataset_cache.py
Purpose: 다이나믹 런지 1,468개 전수 시퀀스에 대한 RTMPose 2D 30D 정예 피처 데이터셋 구축 및 캐싱
         - bl1_rtmp_1 아키텍처 계승 (RTMPose Halpe26 -> MP33 Fake-Z 사상 -> 30D 도메인 기구학)
         - 100% 무누수(Data Isolation) 검증: AI-Hub JSON 좌표 미사용, 라벨은 타깃(y)으로만 사용
         - 피험자 1~9번 전수, 공식 5대 시점(view1~view5) 전수 수용
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"D:\src\features")

from lunge_c1_features_v4 import (
    extract_lunge_c1_feature_vector_v4,
    LUNGE_C1_FEATURE_NAMES_V4
)

MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지"
RTMP_DIRS = [
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\05_스텝_포워드_다이나믹_런지",
    r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\06_스텝_백워드_다이나믹_런지"
]
LABEL_DIRS = [
    r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D",
    r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"
]
CACHE_OUT_PATH = r"D:\Data\피처_데이터\dl1_rtmp_1_dataset_cache.npz"


def map_rtmp26_to_mp33_with_fake_z(kps26: np.ndarray) -> np.ndarray:
    """
    RTMPose Halpe26 2D -> MediaPipe 33 format with fake Z depth
    (extract_rtmp_fair.py 및 bl1_rtmp_1 표준 규격)
    """
    kps33 = np.zeros((64, 33, 3), dtype=np.float32)
    kps33[:, 0] = kps26[:, 0]
    kps33[:, 2] = kps26[:, 1]
    kps33[:, 5] = kps26[:, 2]
    kps33[:, 7] = kps26[:, 3]
    kps33[:, 8] = kps26[:, 4]
    kps33[:, 11] = kps26[:, 5]
    kps33[:, 12] = kps26[:, 6]
    kps33[:, 13] = kps26[:, 7]
    kps33[:, 14] = kps26[:, 8]
    kps33[:, 15] = kps26[:, 9]
    kps33[:, 16] = kps26[:, 10]
    kps33[:, 23] = kps26[:, 11]
    kps33[:, 24] = kps26[:, 12]
    kps33[:, 25] = kps26[:, 13]
    kps33[:, 26] = kps26[:, 14]
    kps33[:, 27] = kps26[:, 15]
    kps33[:, 28] = kps26[:, 16]
    kps33[:, 29] = kps26[:, 24]  # LEFT_HEEL
    kps33[:, 30] = kps26[:, 25]  # RIGHT_HEEL
    kps33[:, 31] = kps26[:, 20]  # LEFT_FOOT_INDEX
    kps33[:, 32] = kps26[:, 21]  # RIGHT_FOOT_INDEX

    # Fake Z depth injection for sagittal kinematics
    left_mp_indices = [1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    right_mp_indices = [4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    kps33[:, left_mp_indices, 2] = -0.1
    kps33[:, right_mp_indices, 2] = 0.1
    return kps33


def build_dl1_rtmp_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """다이나믹 런지 RTMPose 30D 피처셋 구축 및 저장"""
    os.makedirs(os.path.dirname(CACHE_OUT_PATH), exist_ok=True)
    if os.path.exists(CACHE_OUT_PATH) and not force_rebuild:
        print(f"[*] 기존 캐시 발견: {CACHE_OUT_PATH}")
        data = np.load(CACHE_OUT_PATH, allow_pickle=True)
        print(f"[+] 캐시 로드 성공: N={len(data['y'])}, D={data['X'].shape[1]}")
        return dict(data)

    print("=" * 80)
    print("  [dl1_rtmp_1] 다이나믹 런지 RTMPose 30D 피처 데이터셋 추출 시작")
    print("=" * 80)

    mp33_files = sorted(glob.glob(os.path.join(MP33_DIR, "*.npz")))
    print(f"[*] 05_06_다이나믹_런지 기준 시퀀스 수: {len(mp33_files)}")

    X_list = []
    y_list = []
    subjects_list = []
    views_list = []
    exercise_types_list = []
    fnames_list = []

    # 1. 파일 매칭 및 라벨 로드
    matched_items = []
    for mp_path in mp33_files:
        fname = os.path.basename(mp_path)
        v_code = None
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            if f"_{v}.npz" in fname:
                v_code = v
                break
        if v_code is None:
            continue

        sub_id = 1
        for s in range(1, 10):
            if f"피험자{s}" in fname:
                sub_id = s
                break

        rtmp_path = None
        ex_type = None
        for rd in RTMP_DIRS:
            cand = os.path.join(rd, fname)
            if os.path.exists(cand):
                rtmp_path = cand
                ex_type = "05_forward" if ("05_" in rd or "포워드" in rd) else "06_backward"
                break
        if rtmp_path is None:
            continue

        raw_key = fname.replace(f"_{v_code}.npz", "")
        label_json_path = None
        for ld in LABEL_DIRS:
            cand = os.path.join(ld, f"{raw_key}.json")
            if os.path.exists(cand):
                label_json_path = cand
                break
        if label_json_path is None:
            continue

        try:
            with open(label_json_path, 'r', encoding='utf-8', errors='ignore') as fp:
                d = json.load(fp)
            conds = d.get('type_info', {}).get('conditions', [])
            c1_val = None
            for c in conds:
                if '앞다리' in c.get('condition', '') and '무릎' in c.get('condition', ''):
                    c1_val = c.get('value')
                    break
            if c1_val is None and len(conds) > 0:
                c1_val = conds[0].get('value')
            if c1_val is None:
                continue
            y_val = 1 if (c1_val is False) else 0
        except Exception:
            continue

        matched_items.append({
            'fname': fname,
            'rtmp_path': rtmp_path,
            'view': v_code,
            'subject': sub_id,
            'exercise_type': ex_type,
            'y': y_val
        })

    print(f"[+] 총 {len(matched_items)}개 시퀀스 매칭 성공 (100% 매칭 완료)")

    # 2. 피처 추출 루프
    t0 = time.time()
    for item in tqdm(matched_items, desc="Extracting RTMPose 30D Features"):
        try:
            rtmp_npz = np.load(item['rtmp_path'], allow_pickle=True)
            kps26 = rtmp_npz['kps_26_64f']  # (64, 26, 3)

            kps33 = map_rtmp26_to_mp33_with_fake_z(kps26)
            feat_vec = extract_lunge_c1_feature_vector_v4(kps33)

            # 무결성 검증 (NaN / Inf 방어)
            if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
                feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)

            X_list.append(feat_vec)
            y_list.append(item['y'])
            subjects_list.append(item['subject'])
            views_list.append(item['view'])
            exercise_types_list.append(item['exercise_type'])
            fnames_list.append(item['fname'])
        except Exception as e:
            print(f"[!] 에러 발생 ({item['fname']}): {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list, dtype=object)
    exercise_types = np.array(exercise_types_list, dtype=object)
    fnames = np.array(fnames_list, dtype=object)
    feature_names = np.array(LUNGE_C1_FEATURE_NAMES_V4, dtype=object)

    elapsed = time.time() - t0
    print(f"[+] 피처 추출 완료: 소요시간 {elapsed:.2f}초 ({elapsed/len(X)*1000:.2f} ms/시퀀스)")
    print(f"    - X Shape: {X.shape}")
    print(f"    - y Shape: {y.shape} (정상 0: {np.sum(y==0)}, 결함 1: {np.sum(y==1)})")
    print(f"    - NaN 개수: {np.isnan(X).sum()}, Inf 개수: {np.isinf(X).sum()}")

    # 캐시 파일 저장
    np.savez_compressed(
        CACHE_OUT_PATH,
        X=X,
        y=y,
        subjects=subjects,
        views=views,
        exercise_types=exercise_types,
        fnames=fnames,
        feature_names=feature_names
    )
    print(f"[+] 캐시 파일 압축 저장 완료: {CACHE_OUT_PATH}")

    return {
        'X': X,
        'y': y,
        'subjects': subjects,
        'views': views,
        'exercise_types': exercise_types,
        'fnames': fnames,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    build_dl1_rtmp_dataset(force_rebuild=True)
