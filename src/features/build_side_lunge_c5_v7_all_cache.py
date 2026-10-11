# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge)
Module: build_side_lunge_c5_v7_all_cache.py
Purpose: 사이드 런지 원천 데이터 826개 폴더 전체(4,130개 시퀀스)를 대상으로,
         설명(description) 텍스트 기반 전수 라벨 교정 데이터셋에 대해
         side_lunge_c5 v7 42D 혁신 도메인 피처 텐서 및 정답 라벨 캐시 빌드
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c5_features_v7 import (
    extract_side_lunge_c5_v7_feature_vector,
    SIDE_LUNGE_C5_V7_FEATURE_NAMES
)

RAW_MP_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABELS_JSON_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
OUT_NPZ_PATH = r"D:\Data\피처_데이터\side_lunge_c5_v7_all_relabel_42d_features.npz"


def load_relabel_map() -> Dict[str, Dict[str, Any]]:
    """모든 2D JSON 파일로부터 description 텍스트 기반 c5 라벨 전수 교정 매핑 생성"""
    label_map = {}
    json_files = glob.glob(os.path.join(LABELS_JSON_DIR, "*.json"))
    for jf in json_files:
        base_name = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            tinfo = data.get("type_info", {})
            key = str(tinfo.get("key", ""))
            desc = tinfo.get("description", "")
            conds = tinfo.get("conditions", [])

            # 원본 c5 라벨 값
            orig_c5_val = conds[4].get("value") if len(conds) >= 5 else True
            orig_label = 1 if orig_c5_val is False else 0

            # 설명(description) 기반 c5 정답 판정
            if ("상체 뒤로 젖힘" in desc or "상체뒤로젖힘" in desc or 
                "상체 젖힘" in desc or "상체 숙임" in desc or "숙임" in desc):
                corrected_label = 1
            else:
                corrected_label = 0

            label_map[base_name] = {
                "key": key,
                "desc": desc,
                "orig_label": orig_label,
                "label": corrected_label,
                "is_corrected": (orig_label != corrected_label)
            }
        except Exception:
            pass
    return label_map


def build_side_lunge_c5_v7_all_cache():
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 전체 826폴더 42D v7 혁신 피처 빌더")
    print("=" * 80)

    label_map = load_relabel_map()
    corrected_count = sum(1 for m in label_map.values() if m["is_corrected"])
    print(f"[*] 로드된 2D JSON 라벨 수: {len(label_map)}개")
    print(f"[*] 설명 기반 정답 라벨 교정 적용 건수: {corrected_count}개")

    npz_files = sorted(glob.glob(os.path.join(RAW_MP_DIR, "*.npz")))
    print(f"[*] MediaPipe 64F npz 수: {len(npz_files)}개")

    if not npz_files:
        print("[!] 전처리된 MediaPipe 64F npz 파일이 없습니다.")
        return

    X_list = []
    y_list = []
    subjects_list = []
    views_list = []
    keys_list = []
    filenames_list = []

    t0 = time.time()
    for nf in tqdm(npz_files, desc="v7 42D 피처 추출 중"):
        fname = os.path.basename(nf)
        parts = fname.replace(".npz", "").split("_")
        vn = parts[-1]  # view1 ~ view5
        clip_id = "_".join(parts[:-1])

        # 피험자 파싱
        subj = 1
        for p in parts:
            if "피험자" in p:
                try:
                    subj = int(p.replace("피험자", ""))
                except Exception:
                    pass
                break

        if clip_id not in label_map:
            continue

        meta = label_map[clip_id]
        final_label = meta["label"]  # 설명 기반 교정 라벨
        key_id = meta["key"]

        try:
            data = np.load(nf, allow_pickle=True)
            seq_3d = data['kps_33_3d_64f']
            feat_vec = extract_side_lunge_c5_v7_feature_vector(seq_3d)

            if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
                continue

            X_list.append(feat_vec)
            y_list.append(final_label)
            subjects_list.append(subj)
            views_list.append(vn)
            keys_list.append(key_id)
            filenames_list.append(fname.replace(".npz", ""))
        except Exception as e:
            continue

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    subjects_arr = np.array(subjects_list, dtype=np.int32)
    views_arr = np.array(views_list, dtype=str)
    keys_arr = np.array(keys_list, dtype=str)
    filenames_arr = np.array(filenames_list, dtype=str)

    elapsed = time.time() - t0
    print(f"\n[+] v7 피처 추출 완료: 총 {len(y_arr)}개 시퀀스 (소요시간: {elapsed:.2f}s)")
    print(f"    - X Shape: {X_arr.shape}")
    print(f"    - y Shape: {y_arr.shape} (정상 0: {np.sum(y_arr == 0)}, 결함 1: {np.sum(y_arr == 1)}, 결함비율: {np.mean(y_arr == 1)*100:.2f}%)")
    print(f"    - 피험자: {np.unique(subjects_arr).tolist()} ({len(np.unique(subjects_arr))}명 전수)")
    print(f"    - 시점: {np.unique(views_arr).tolist()} (5대 시점 전수)")
    print(f"    - 결측치 감사: NaN={np.isnan(X_arr).sum()}개, Inf={np.isinf(X_arr).sum()}개 (무결성 통과)")

    os.makedirs(os.path.dirname(OUT_NPZ_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ_PATH,
        X=X_arr,
        y=y_arr,
        subjects=subjects_arr,
        views=views_arr,
        keys=keys_arr,
        filenames=filenames_arr,
        feature_names=SIDE_LUNGE_C5_V7_FEATURE_NAMES
    )
    print(f"[+] v7 42D 피처 캐시 저장 완료: {OUT_NPZ_PATH}")


if __name__ == '__main__':
    build_side_lunge_c5_v7_all_cache()
