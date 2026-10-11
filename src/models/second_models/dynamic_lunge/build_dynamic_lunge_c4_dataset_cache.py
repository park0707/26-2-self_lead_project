# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 통합 (05_스텝_포워드_다이나믹_런지 + 06_스텝_백워드_다이나믹_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Module: build_dynamic_lunge_c4_dataset_cache.py
Purpose: 다이나믹 런지 c4 24차원 정예 도메인 피처 및 14차원 시계열 텐서 캐시 구축
         - 100% 순수 비전 MediaPipe Full33 64F 입력 기반
         - AI-Hub 2D JSON에서 조건 4 (척추의 중립: False->1 결함, True->0 정상) 매핑
         - 피험자 1~9번 전수, 공식 5대 시점(view1~5) 전수 수용
         - 물리적 임계치 기반 4단계 모호성(Level 1~4) 정량화
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

from src.features.lunge_c4_features_v3 import (
    extract_lunge_c4_features_v3,
    extract_lunge_c4_temporal_features,
    LUNGE_C4_FEATURE_NAMES_V3
)

DATA_SOURCES = [
    {
        'name': '05_06_다이나믹_런지_통합',
        'mp33_dir': r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_06_다이나믹_런지",
        'label_dirs': [
            r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D",
            r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"
        ],
    },
    {
        'name': '05_스텝_포워드_다이나믹_런지',
        'mp33_dir': r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\05_스텝_포워드_다이나믹_런지",
        'label_dirs': [r"D:\Data\01_raw_data\labels_json\05_스텝_포워드_다이나믹_런지\2D"],
    },
    {
        'name': '06_스텝_백워드_다이나믹_런지',
        'mp33_dir': r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\06_스텝_백워드_다이나믹_런지",
        'label_dirs': [r"D:\Data\01_raw_data\labels_json\06_스텝_백워드_다이나믹_런지\2D"],
    }
]

OUT_CACHE_PATH = r"D:\Data\피처_데이터\dynamic_lunge_c4_dataset_cache.npz"
os.makedirs(os.path.dirname(OUT_CACHE_PATH), exist_ok=True)

VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def load_all_labels(label_dir: str) -> Dict[str, Dict[str, Any]]:
    """지정 디렉토리의 모든 2D JSON 라벨 로드 및 척추 중립(조건 4) 라벨 추출"""
    label_files = glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True)
    labels_dict = {}
    for lf in label_files:
        bname = os.path.basename(lf).replace('.json', '')
        try:
            with open(lf, 'r', encoding='utf-8') as f:
                d = json.load(f)
            t_info = d.get('type_info', {})
            conds = t_info.get('conditions', [])
            if len(conds) > 3:
                # 조건 4: 척추의 중립 (True: 정상->0, False: 결함->1)
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


def compute_ambiguity_level(feats_24: np.ndarray, label: int) -> int:
    """
    물리적 임계치 기반 4단계 모호성 정량화
    Level 1: 명확군 (Clear)
    Level 2: 안정군 (Moderate)
    Level 3: 경계군 (Borderline)
    Level 4: 극단 모호군 (Ambiguous / 라벨 불일치 경계)
    """
    yaw_max = feats_24[0]
    pitch_max = feats_24[2]
    pitch_min = feats_24[3]
    roll_max = feats_24[8]

    cervical_dev = max(yaw_max, abs(pitch_max), abs(pitch_min))

    if label == 1:  # 결함군
        if cervical_dev >= 20.0 or roll_max >= 10.0:
            return 1
        elif cervical_dev >= 14.0 or roll_max >= 7.0:
            return 2
        elif cervical_dev >= 8.0 or roll_max >= 4.0:
            return 3
        else:
            return 4
    else:  # 정상군
        if cervical_dev < 10.0 and roll_max < 5.0:
            return 1
        elif cervical_dev < 15.0 and roll_max < 8.0:
            return 2
        elif cervical_dev < 20.0 and roll_max < 11.0:
            return 3
        else:
            return 4


def build_cache():
    print("=" * 80)
    print("  [vidio-health-trainer] 다이나믹 런지 c4 24D 정예 피처 및 14D 시계열 캐시 구축")
    print("=" * 80)

    X_24d_list = []
    X_temp_14d_list = []
    y_list = []
    subj_list = []
    view_list = []
    clip_id_list = []
    ambiguity_list = []
    key_list = []
    source_list = []
    seen_fnames = set()

    for src in DATA_SOURCES:
        s_name = src['name']
        mp33_dir = src['mp33_dir']
        labels_map = {}
        for ld in src.get('label_dirs', []):
            labels_map.update(load_all_labels(ld))
        print(f"\n[*] {s_name} 라벨 로드 완료: {len(labels_map)}개")

        all_npz = glob.glob(os.path.join(mp33_dir, "*.npz"))
        print(f"[*] {s_name} MediaPipe 64F 시퀀스 감지: {len(all_npz)}개")

        if not all_npz:
            print(f"[!] {s_name} 전처리 npz 파일이 없습니다. (건너뜀)")
            continue

        for fpath in tqdm(all_npz, desc=f"Extracting {s_name}"):
            fname = os.path.basename(fpath).replace('.npz', '')
            parts = fname.rsplit('_', 1)
            if len(parts) != 2:
                continue
            clip_base, view_code = parts[0], parts[1]

            if fname in seen_fnames:
                continue
            seen_fnames.add(fname)

            if clip_base not in labels_map:
                continue

            l_info = labels_map[clip_base]
            label = l_info['label']
            key_code = l_info['key']

            subj = "Unknown"
            for p in clip_base.split('_'):
                if '피험자' in p:
                    subj = p
                    break

            try:
                data = np.load(fpath)
                seq_33 = data['kps_33_3d_64f']
            except Exception:
                continue

            feats_24 = extract_lunge_c4_features_v3(seq_33)
            temp_14 = extract_lunge_c4_temporal_features(seq_33)
            amb_lvl = compute_ambiguity_level(feats_24, label)

            X_24d_list.append(feats_24)
            X_temp_14d_list.append(temp_14)
            y_list.append(label)
            subj_list.append(subj)
            view_list.append(view_code)
            clip_id_list.append(fname)
            ambiguity_list.append(amb_lvl)
            key_list.append(key_code)
            source_list.append(s_name)

    X_24d = np.array(X_24d_list, dtype=np.float32)
    X_temp_14d = np.array(X_temp_14d_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjs = np.array(subj_list)
    views = np.array(view_list)
    clip_ids = np.array(clip_id_list)
    ambiguities = np.array(ambiguity_list, dtype=np.int32)
    keys = np.array(key_list)
    sources = np.array(source_list)

    print("\n" + "=" * 80)
    print(f"[+] 다이나믹 런지 c4 데이터셋 캐시 구축 완료!")
    print(f"    - 총 유효 시퀀스: {len(y)}개")
    print(f"    - 정상(0): {np.sum(y == 0)}개, 결함(1): {np.sum(y == 1)}개 (결함비: {np.mean(y)*100:.2f}%)")
    print(f"    - 피험자 분포: {dict(zip(*np.unique(subjs, return_counts=True)))}")
    print(f"    - 시점 분포: {dict(zip(*np.unique(views, return_counts=True)))}")
    print(f"    - X_24d 형상: {X_24d.shape} (결측/무한대 검증: NaN={np.isnan(X_24d).any()}, Inf={np.isinf(X_24d).any()})")
    print(f"    - X_temp_14d 형상: {X_temp_14d.shape}")

    np.savez_compressed(
        OUT_CACHE_PATH,
        X_24d=X_24d,
        X_temp_14d=X_temp_14d,
        y=y,
        subjects=subjs,
        views=views,
        clip_ids=clip_ids,
        ambiguities=ambiguities,
        keys=keys,
        sources=sources,
        feature_names=np.array(LUNGE_C4_FEATURE_NAMES_V3)
    )
    print(f"    - 캐시 저장 완료: {OUT_CACHE_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    build_cache()
