# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: build_lunge_c4_dataset_cache_v4.py
Purpose: 바벨 런지 c4(척추의 중립 위반) v4 정밀 30차원 도메인 피처 및 16차원 시계열 텐서 캐시 구축
         - 100% 순수 비전 MediaPipe Full33 64F 입력 기반
         - 오분류 분석 기반 6D 상대적 동역학 정규화 피처 탑재
         - 피험자 1~9번 전수, 공식 5대 시점(view1~5) 전수 수용
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

from src.features.lunge_c4_features_v4 import (
    extract_lunge_c4_features_v4,
    extract_lunge_c4_temporal_features_v4,
    LUNGE_C4_FEATURE_NAMES_V4
)

MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
OUT_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v4.npz"
os.makedirs(os.path.dirname(OUT_CACHE_PATH), exist_ok=True)


def load_all_labels() -> Dict[str, Dict[str, Any]]:
    """모든 2D JSON 라벨 로드 및 척추 중립(조건 4) 라벨 추출"""
    label_files = glob.glob(os.path.join(LABEL_DIR, "**", "*.json"), recursive=True)
    labels_dict = {}
    for lf in label_files:
        bname = os.path.basename(lf).replace('.json', '')
        try:
            with open(lf, 'r', encoding='utf-8') as f:
                d = json.load(f)
            t_info = d.get('type_info', {})
            conds = t_info.get('conditions', [])
            if len(conds) > 3:
                is_normal = conds[3].get('value', True)
                defect_label = 0 if is_normal else 1
                key_code = t_info.get('key', '')
                desc = t_info.get('description', '')
                labels_dict[bname] = {
                    'label': defect_label,
                    'key': key_code,
                    'desc': desc
                }
        except Exception:
            continue
    return labels_dict


def build_cache_v4():
    print("=" * 75)
    print("  [vidio-health-trainer] 바벨 런지 c4 v4 정밀 30D 캐시 구축 파이프라인")
    print("=" * 75)

    labels_map = load_all_labels()
    print(f"[*] 라벨 데이터 로드 완료: {len(labels_map)}개 클립 매핑")

    all_npz = glob.glob(os.path.join(MP33_DIR, "*.npz"))
    print(f"[*] MediaPipe 64F 시퀀스 감지: {len(all_npz)}개")

    if not all_npz:
        print("[!] 전처리 npz 파일이 없습니다.")
        return

    X_30d_list = []
    X_temp_16d_list = []
    y_list = []
    subj_list = []
    view_list = []
    clip_id_list = []
    ambiguity_list = []
    key_list = []

    print("[*] v4 피처 및 16D 시계열 추출 진행 중...")
    for fpath in tqdm(all_npz):
        fname = os.path.basename(fpath).replace('.npz', '')
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        clip_base, view_code = parts[0], parts[1]

        if clip_base not in labels_map:
            continue

        l_info = labels_map[clip_base]
        label = l_info['label']
        key_code = l_info['key']

        subj = "피험자1"
        for i in range(1, 10):
            if f"피험자{i}" in fname:
                subj = f"피험자{i}"
                break

        try:
            npz_data = np.load(fpath)
            seq_64f = npz_data['kps_33_3d_64f']  # (64, 33, 4)

            feats_30d = extract_lunge_c4_features_v4(seq_64f)
            temp_16d = extract_lunge_c4_temporal_features_v4(seq_64f)

            X_30d_list.append(feats_30d)
            X_temp_16d_list.append(temp_16d)
            y_list.append(label)
            subj_list.append(subj)
            view_list.append(view_code)
            clip_id_list.append(fname)
            ambiguity_list.append(1)
            key_list.append(key_code)
        except Exception as e:
            continue

    X_30d_arr = np.array(X_30d_list, dtype=np.float32)
    X_temp_16d_arr = np.array(X_temp_16d_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    subj_arr = np.array(subj_list)
    view_arr = np.array(view_list)
    clip_id_arr = np.array(clip_id_list)
    ambiguity_arr = np.array(ambiguity_list, dtype=np.int32)
    key_arr = np.array(key_list)

    print("\n" + "=" * 70)
    print("  [캐시 구축 결과 요약]")
    print("=" * 70)
    print(f"[*] 총 추출 시퀀스 수: {len(y_arr)}개")
    print(f"    - 정상(0): {np.sum(y_arr == 0)}개, 결함(1): {np.sum(y_arr == 1)}개 (결함비율: {np.mean(y_arr)*100:.2f}%)")
    print(f"[*] 피처 형상 (X_30d): {X_30d_arr.shape}")
    print(f"[*] 시계열 형상 (X_temp_16d): {X_temp_16d_arr.shape}")

    np.savez_compressed(
        OUT_CACHE_PATH,
        X_30d=X_30d_arr,
        X_temp_16d=X_temp_16d_arr,
        y=y_arr,
        subjects=subj_arr,
        views=view_arr,
        clip_ids=clip_id_arr,
        ambiguities=ambiguity_arr,
        keys=key_arr,
        feature_names=np.array(LUNGE_C4_FEATURE_NAMES_V4)
    )
    print(f"[+] 캐시 저장 완료: {OUT_CACHE_PATH}")


if __name__ == "__main__":
    build_cache_v4()
