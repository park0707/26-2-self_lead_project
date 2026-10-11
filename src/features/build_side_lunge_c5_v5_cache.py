# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge)
Module: build_side_lunge_c5_v5_cache.py
Purpose: MediaPipe Full 33관절 64F 전처리 데이터로부터 side_lunge_c5 v5 36D SOTA 도메인 피처 텐서 및 정답 라벨 빌드
         - 바벨 런지 c5 v5 36D 피처 아키텍처 100% 계승
         - AI-Hub 3D JSON 배제 (100% 비전 MediaPipe 33 키포인트)
         - 정답 라벨: 2D JSON type_info.conditions[4] 파싱 (c5: 상체의 과조한 숙임/젖힘 여부)
         - 라벨-설명 일치 정제 19개 Key 데이터셋 전수 포함
         - 피험자(1~9 전수), 5대 시점(view1~view5 전수) 정합 매핑
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

from src.features.lunge_c5_features_v5 import (
    extract_lunge_c5_v5_feature_vector,
    LUNGE_C5_V5_FEATURE_NAMES
)

RAW_MP_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABELS_JSON_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
OUT_NPZ_PATH = r"D:\Data\피처_데이터\side_lunge_c5_v5_36d_features.npz"

# 라벨과 설명이 일치하는 19개 Key 공식 정의
MATCHING_KEYS = {
    '145', '146', '147', '148', '149', '150', '151', '152',
    '164', '165', '166', '167', '168', '169', '170',
    '172', '173', '174', '175'
}


def load_label_map() -> Dict[str, Dict[str, Any]]:
    """모든 2D JSON 파일로부터 Condition 5 라벨 파싱 (라벨-설명 일치 Key만 수록)"""
    label_map = {}
    json_files = glob.glob(os.path.join(LABELS_JSON_DIR, "*.json"))
    for jf in json_files:
        base_name = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            tinfo = data.get("type_info", {})
            key = str(tinfo.get("key", ""))
            
            # 라벨-설명 불일치 Key는 배제
            if key not in MATCHING_KEYS:
                continue

            conds = tinfo.get("conditions", [])
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


def build_side_lunge_c5_v5_dataset():
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 c5 v5 36차원 SOTA 피처 텐서 빌더")
    print("=" * 80)

    label_map = load_label_map()
    print(f"[*] 로드된 2D JSON 일치 라벨 수: {len(label_map)}개")

    npz_files = sorted(glob.glob(os.path.join(RAW_MP_DIR, "*.npz")))
    print(f"[*] 탐색된 MediaPipe 64F npz 수: {len(npz_files)}개")

    if not npz_files:
        print("[!] 전처리된 MediaPipe 64F npz 파일이 없습니다. 전처리를 먼저 완료하세요.")
        return

    X_list = []
    y_list = []
    subj_list = []
    view_list = []
    key_list = []
    file_list = []

    t0 = time.time()
    for nf in tqdm(npz_files, desc="v5 36D 피처 추출 중"):
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
            # 라벨 맵에 없으면 (불일치 Key이거나 미포함 폴더) 스킵
            continue

        meta = label_map[clip_id]
        lbl = meta["label"]
        key = meta["key"]

        try:
            data = np.load(nf, allow_pickle=True)
            kps_33_3d = data["kps_33_3d_64f"]  # (64, 33, 3)
            kps_33_2d = data["kps_33_2d_64f"]  # (64, 33, 2)

            feat_vec = extract_lunge_c5_v5_feature_vector(kps_33_3d, kps_33_2d)

            # NaN / Inf 안전 처리
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

            X_list.append(feat_vec)
            y_list.append(lbl)
            subj_list.append(subj)
            view_list.append(vn)
            key_list.append(key)
            file_list.append(fname)
        except Exception as e:
            print(f"[WARN] Error processing {fname}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subj_list, dtype=np.int32)
    views = np.array(view_list, dtype=object)
    keys = np.array(key_list, dtype=object)
    filenames = np.array(file_list, dtype=object)

    elapsed = time.time() - t0
    print(f"\n[+] 피처 추출 완료: {len(X)}개 시퀀스 (소요시간: {elapsed:.2f}s)")
    print(f"    - X Shape: {X.shape}")
    print(f"    - y Shape: {y.shape} (정상 0: {np.sum(y == 0)}, 결함 1: {np.sum(y == 1)}, 결함비율: {np.mean(y == 1)*100:.2f}%)")
    print(f"    - 피험자: {np.unique(subjects).tolist()}")
    print(f"    - 시점: {np.unique(views).tolist()}")

    # NaN / Inf 검증
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"    - 결측치 감사: NaN={nan_count}개, Inf={inf_count}개 (무결성 통과)")

    os.makedirs(os.path.dirname(OUT_NPZ_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ_PATH,
        X=X,
        y=y,
        subjects=subjects,
        views=views,
        keys=keys,
        filenames=filenames,
        feature_names=np.array(LUNGE_C5_V5_FEATURE_NAMES, dtype=object)
    )
    print(f"[+] 피처 캐시 저장 완료: {OUT_NPZ_PATH}")


if __name__ == "__main__":
    build_side_lunge_c5_v5_dataset()
