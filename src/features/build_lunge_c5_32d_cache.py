# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: build_lunge_c5_32d_cache.py
Purpose: MediaPipe Full 33관절 64F 캐시로부터 lunge_c5 32D 도메인 피처 텐서 및 정답 라벨 빌드
         - AI-Hub 3D JSON 배제 (MediaPipe 33 비전 키포인트만 투입)
         - 정답 라벨은 2D JSON type_info.conditions[4] (Condition 5) 파싱
         - 피험자(1~9), 시점(view1~view5), Key 번호 정밀 매핑
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Any, Optional

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features import extract_lunge_c5_feature_vector, LUNGE_C5_FEATURE_NAMES

RAW_MP_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
LABELS_JSON_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
OUT_NPZ_PATH = r"D:\Data\피처_데이터\lunge_c5_32d_features.npz"


def load_label_map() -> Dict[str, Dict[str, Any]]:
    """모든 2D JSON 파일로부터 Condition 5 라벨 및 메타데이터 추출"""
    label_map = {}
    json_files = glob.glob(os.path.join(LABELS_JSON_DIR, "*.json"))
    for jf in json_files:
        base_name = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            tinfo = data.get("type_info", {})
            conds = tinfo.get("conditions", [])
            key = tinfo.get("key", "329")
            desc = tinfo.get("description", "")
            
            # Condition 5 value (True: 정상 -> 0, False: 결함 -> 1)
            c5_val = conds[4].get("value") if len(conds) >= 5 else True
            label = 1 if c5_val is False else 0
            
            label_map[base_name] = {
                "key": key,
                "desc": desc,
                "label": label,
                "c5_raw": c5_val
            }
        except Exception as e:
            pass
    return label_map


def build_lunge_c5_dataset(min_files: int = 100):
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 런지 c5 32차원 피처 텐서 빌더")
    print("=" * 80)

    label_map = load_label_map()
    print(f"[*] 로드된 2D JSON 라벨 수: {len(label_map)}개")

    npz_files = sorted(glob.glob(os.path.join(RAW_MP_DIR, "*.npz")))
    print(f"[*] 탐색된 MediaPipe 64F npz 수: {len(npz_files)}개")

    if len(npz_files) < min_files:
        print(f"[!] 경고: 가용 파일 수({len(npz_files)})가 최소 요구({min_files})보다 적습니다.")

    X_list = []
    y_list = []
    subj_list = []
    view_list = []
    key_list = []
    file_list = []

    view_map = {
        "view5": 0,  # 시점 A (우측 후방 45°)
        "view2": 1,  # 시점 B (우측 전방 45°)
        "view3": 2,  # 시점 C (정면 0°)
        "view4": 3,  # 시점 D (좌측 전방 45°)
        "view1": 4   # 시점 E (좌측 후방 45°)
    }

    t0 = time.time()
    for nf in tqdm(npz_files, desc="피처 추출 중"):
        fname = os.path.basename(nf)
        # 예: [오류]_D03_피험자1_Key330_view5.npz
        parts = fname.replace(".npz", "").split("_")
        vn = parts[-1]  # view1 ~ view5
        clip_id = "_".join(parts[:-1])

        # 피험자 파싱
        subj = 1
        for s in range(1, 10):
            if f"피험자{s}" in clip_id or f"_{s}_" in clip_id:
                subj = s
                break

        # 라벨 매핑
        if clip_id in label_map:
            info = label_map[clip_id]
            lbl = info["label"]
            k_val = info["key"]
        elif "정상" in clip_id:
            lbl = 0
            k_val = "329"
        else:
            continue

        try:
            data = np.load(nf)
            kps_33_3d = data['kps_33_3d_64f']
            vis = data.get('vis_64f', None)

            feat_vec = extract_lunge_c5_feature_vector(kps_33_3d, vis)
            if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
                feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

            X_list.append(feat_vec)
            y_list.append(lbl)
            subj_list.append(subj)
            view_list.append(vn)
            key_list.append(k_val)
            file_list.append(fname)
        except Exception as e:
            continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subj_list, dtype=np.int32)
    views = np.array(view_list, dtype=object)
    keys = np.array(key_list, dtype=object)
    filenames = np.array(file_list, dtype=object)

    elapsed = time.time() - t0
    print(f"\n[+] 피처 추출 완료: {X.shape[0]}개 시퀀스 | 차원: {X.shape[1]}D | 소요: {elapsed:.1f}초")
    print(f"    - 정상(0): {np.sum(y == 0)}개 ({np.mean(y == 0)*100:.1f}%)")
    print(f"    - 결함(1): {np.sum(y == 1)}개 ({np.mean(y == 1)*100:.1f}%)")
    print(f"    - 피험자 분포: {dict(zip(*np.unique(subjects, return_counts=True)))}")
    print(f"    - 시점 분포: {dict(zip(*np.unique(views, return_counts=True)))}")

    os.makedirs(os.path.dirname(OUT_NPZ_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ_PATH,
        X=X,
        y=y,
        subjects=subjects,
        views=views,
        keys=keys,
        filenames=filenames,
        feature_names=np.array(LUNGE_C5_FEATURE_NAMES, dtype=object)
    )
    print(f"[+] 캐시 저장 완료: {OUT_NPZ_PATH} ({os.path.getsize(OUT_NPZ_PATH)/1024/1024:.2f} MB)")
    return OUT_NPZ_PATH


if __name__ == "__main__":
    build_lunge_c5_dataset(min_files=50)
