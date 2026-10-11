# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: build_lunge_c4_dataset_cache.py
Purpose: 바벨 런지 c4(척추의 중립 위반) 28차원 도메인 피처 및 14차원 시계열 텐서 캐시 구축
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

from src.features.lunge_c4_features import (
    extract_lunge_c4_features,
    extract_lunge_c4_temporal_series,
    LUNGE_C4_FEATURE_NAMES
)

MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
OUT_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache.npz"
os.makedirs(os.path.dirname(OUT_CACHE_PATH), exist_ok=True)

VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


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


def compute_ambiguity_level(feats_28: np.ndarray, label: int) -> int:
    """
    물리적 임계치 기반 4단계 모호성 정량화
    Level 1: 명확군 (Clear)
    Level 2: 안정군 (Moderate)
    Level 3: 경계군 (Borderline)
    Level 4: 극단 모호군 (Ambiguous / 라벨 불일치 경계)
    """
    # feats_28 주요 지표:
    # idx 0: cervical_yaw_max
    # idx 2: cervical_pitch_max
    # idx 3: cervical_pitch_min
    # idx 8: trunk_roll_max
    yaw_max = feats_28[0]
    pitch_max = feats_28[2]
    pitch_min = feats_28[3]
    roll_max = feats_28[8]

    # 결함 지표 복합치
    cervical_dev = max(yaw_max, abs(pitch_max), abs(pitch_min))

    if label == 1:  # 결함군
        if cervical_dev >= 20.0 or roll_max >= 10.0:
            return 1  # 명확 결함
        elif cervical_dev >= 14.0 or roll_max >= 7.0:
            return 2  # 안정 결함
        elif cervical_dev >= 8.0 or roll_max >= 4.0:
            return 3  # 경계 결함
        else:
            return 4  # 극단 모호 결함 (물리적 편차 미미하나 결함 라벨)
    else:  # 정상군
        if cervical_dev < 10.0 and roll_max < 5.0:
            return 1  # 명확 정상
        elif cervical_dev < 15.0 and roll_max < 8.0:
            return 2  # 안정 정상
        elif cervical_dev < 20.0 and roll_max < 11.0:
            return 3  # 경계 정상
        else:
            return 4  # 극단 모호 정상 (물리적 편차 큰데 정상 라벨)


def build_cache():
    print("=" * 70)
    print("  [vidio-health-trainer] 바벨 런지 c4 피처 캐시 구축 파이프라인")
    print("=" * 70)

    labels_map = load_all_labels()
    print(f"[*] 라벨 데이터 로드 완료: {len(labels_map)}개 클립 매핑")

    all_npz = glob.glob(os.path.join(MP33_DIR, "*.npz"))
    print(f"[*] MediaPipe 64F 시퀀스 감지: {len(all_npz)}개")

    if not all_npz:
        print("[!] 전처리 npz 파일이 아직 생성되지 않았습니다.")
        return

    X_28d_list = []
    X_temp_14d_list = []
    y_list = []
    subj_list = []
    view_list = []
    clip_id_list = []
    ambiguity_list = []
    key_list = []

    print("[*] 피처 추출 및 모호성 정량화 진행 중...")
    for fpath in tqdm(all_npz):
        fname = os.path.basename(fpath).replace('.npz', '')
        # 예: [오류]_D03_피험자1_Key330_view1
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        clip_base, view_code = parts[0], parts[1]

        if clip_base not in labels_map:
            continue

        l_info = labels_map[clip_base]
        label = l_info['label']
        key_code = l_info['key']

        # 피험자 파싱
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

        feats_28 = extract_lunge_c4_features(seq_33)
        temp_14 = extract_lunge_c4_temporal_series(seq_33)
        amb_lvl = compute_ambiguity_level(feats_28, label)

        X_28d_list.append(feats_28)
        X_temp_14d_list.append(temp_14)
        y_list.append(label)
        subj_list.append(subj)
        view_list.append(view_code)
        clip_id_list.append(fname)
        ambiguity_list.append(amb_lvl)
        key_list.append(key_code)

    X_28d = np.array(X_28d_list, dtype=np.float32)
    X_temp_14d = np.array(X_temp_14d_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjs = np.array(subj_list)
    views = np.array(view_list)
    clip_ids = np.array(clip_id_list)
    ambiguities = np.array(ambiguity_list, dtype=np.int32)
    keys = np.array(key_list)

    print(f"\n[+] 피처 캐시 구축 완료:")
    print(f"    - 총 유효 시퀀스: {len(y)}개")
    print(f"    - 정상(0): {np.sum(y == 0)}개, 결함(1): {np.sum(y == 1)}개 (결함비: {np.mean(y)*100:.1f}%)")
    print(f"    - X_28d 규격: {X_28d.shape}")
    print(f"    - X_temp_14d 규격: {X_temp_14d.shape}")
    print(f"    - 모호성 등급: Level 1={np.sum(ambiguities==1)}, Level 2={np.sum(ambiguities==2)}, Level 3={np.sum(ambiguities==3)}, Level 4={np.sum(ambiguities==4)}")

    np.savez_compressed(
        OUT_CACHE_PATH,
        X_28d=X_28d,
        X_temp_14d=X_temp_14d,
        y=y,
        subjects=subjs,
        views=views,
        clip_ids=clip_ids,
        ambiguities=ambiguities,
        keys=keys,
        feature_names=LUNGE_C4_FEATURE_NAMES
    )
    print(f"[+] 캐시 저장 완료: {OUT_CACHE_PATH}")


if __name__ == '__main__':
    build_cache()
