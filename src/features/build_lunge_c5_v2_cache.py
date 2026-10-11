# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: build_lunge_c5_v2_cache.py
Purpose: MediaPipe Full 33관절 64F 전처리 데이터로부터 lunge_c5 v2 32D 도메인 피처 텐서 및 정답 라벨 빌드
         - VSPA 및 Farrokhi 2008 생체역학 v2 피처 추출기 적용
         - AI-Hub 3D JSON 배제 (100% 비전 MediaPipe 33 키포인트)
         - 정답 라벨: 2D JSON type_info.conditions[4] 파싱
         - 피험자(1~9), 5대 시점(view1~view5) 정합 매핑
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c5_features_v2 import extract_lunge_c5_v2_feature_vector, LUNGE_C5_V2_FEATURE_NAMES

RAW_MP_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
LABELS_JSON_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
OUT_NPZ_PATH = r"D:\Data\피처_데이터\lunge_c5_v2_32d_features.npz"


def load_label_map() -> Dict[str, Dict[str, Any]]:
    """모든 2D JSON 파일로부터 Condition 5 라벨 파싱"""
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
            
            c5_val = conds[4].get("value") if len(conds) >= 5 else True
            label = 1 if c5_val is False else 0
            
            label_map[base_name] = {
                "key": key,
                "desc": desc,
                "label": label,
                "c5_raw": c5_val
            }
        except Exception:
            pass
    return label_map


def build_lunge_c5_v2_dataset():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 런지 c5 v2 32차원 피처 텐서 빌더")
    print("=" * 80)

    label_map = load_label_map()
    print(f"[*] 로드된 2D JSON 라벨 수: {len(label_map)}개")

    npz_files = sorted(glob.glob(os.path.join(RAW_MP_DIR, "*.npz")))
    print(f"[*] 탐색된 MediaPipe 64F npz 수: {len(npz_files)}개")

    X_list = []
    y_list = []
    subj_list = []
    view_list = []
    key_list = []
    file_list = []

    t0 = time.time()
    for nf in tqdm(npz_files, desc="v2 피처 추출 중"):
        fname = os.path.basename(nf)
        parts = fname.replace(".npz", "").split("_")
        vn = parts[-1]  # view1 ~ view5
        clip_id = "_".join(parts[:-1])

        # 피험자 파싱
        subj = 1
        for p in parts:
            if "피험자" in p:
                digits = "".join(filter(str.isdigit, p))
                if digits:
                    subj = int(digits)
                break

        # 라벨 매칭
        matched = False
        target_label = 0
        key_num = "329"

        if clip_id in label_map:
            target_label = label_map[clip_id]["label"]
            key_num = label_map[clip_id]["key"]
            matched = True
        else:
            for k, v in label_map.items():
                if clip_id in k or k in clip_id:
                    target_label = v["label"]
                    key_num = v["key"]
                    matched = True
                    break

        if not matched:
            if "[정상]" in fname:
                target_label = 0
                key_num = "329"
            else:
                continue

        # 키포인트 로드
        try:
            data = np.load(nf)
            kps_key = 'kps_33_3d_64f' if 'kps_33_3d_64f' in data else ('keypoints_64f' if 'keypoints_64f' in data else None)
            if kps_key is None:
                continue
            kps_64f = data[kps_key]
            vis_64f = data['vis_64f'] if 'vis_64f' in data else (data['visibilities_64f'] if 'visibilities_64f' in data else None)
        except Exception as e:
            continue

        feat_vec = extract_lunge_c5_v2_feature_vector(kps_64f, vis_64f)
        if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(feat_vec)
        y_list.append(target_label)
        subj_list.append(subj)
        view_list.append(vn)
        key_list.append(key_num)
        file_list.append(fname)

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int32)
    subj_arr = np.array(subj_list, dtype=np.int32)
    view_arr = np.array(view_list, dtype=object)
    key_arr = np.array(key_list, dtype=object)
    file_arr = np.array(file_list, dtype=object)
    feat_names = np.array(LUNGE_C5_V2_FEATURE_NAMES, dtype=object)

    os.makedirs(os.path.dirname(OUT_NPZ_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ_PATH,
        X=X_arr,
        y=y_arr,
        subjects=subj_arr,
        views=view_arr,
        keys=key_arr,
        filenames=file_arr,
        feature_names=feat_names
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"[+] v2 피처 추출 완료: {len(X_arr)}개 시퀀스 | 차원: {X_arr.shape[1]}D | 소요시간: {elapsed:.1f}초")
    print(f"    - 정상(0): {np.sum(y_arr == 0)}개 ({np.mean(y_arr == 0)*100:.1f}%)")
    print(f"    - 결함(1): {np.sum(y_arr == 1)}개 ({np.mean(y_arr == 1)*100:.1f}%)")
    unique_subjs, subj_counts = np.unique(subj_arr, return_counts=True)
    print(f"    - 피험자 분포: {dict(zip(unique_subjs, subj_counts))}")
    unique_views, view_counts = np.unique(view_arr, return_counts=True)
    print(f"    - 시점별 분포: {dict(zip(unique_views, view_counts))}")
    print(f"[+] v2 캐시 파일 저장 완료: {OUT_NPZ_PATH} ({os.path.getsize(OUT_NPZ_PATH) / 1024 / 1024:.2f} MB)")
    print("=" * 80)


if __name__ == '__main__':
    build_lunge_c5_v2_dataset()
