# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Module: build_side_lunge_c4_dataset_cache.py
Purpose: 사이드 런지 c4 24차원 정예 도메인 피처 및 14차원 시계열 텐서 캐시 구축
         - 100% 순수 비전 MediaPipe Full33 64F 입력 기반 (AI-Hub 3D JSON 엄격 배제, 무누수 보장)
         - "라벨과 설명이 다른 거 제외한 버전": side_lunge_clean_folders.json (Key 145~152, 208폴더, 1,040 시퀀스) 대상
         - AI-Hub 2D JSON에서 조건 4 (척추의 중립: False->1 결함, True->0 정상) 매핑
         - 피험자 1~9번 전수, 공식 5대 시점(view1~5 / 시점 A~E) 전수 수용
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

MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
CLEAN_FOLDERS_JSON = os.path.join(MP33_DIR, "side_lunge_clean_folders.json")
OUT_CACHE_PATH = r"D:\Data\피처_데이터\side_lunge_c4_dataset_cache.npz"

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
            with open(lf, 'r', encoding='utf-8', errors='ignore') as f:
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
    물리적 임계치 기반 4단계 모호성 레벨 (Level 1~4) 산정:
    - cervical_yaw_max (#0): 정상 < 15°, 결함 >= 22°
    - trunk_roll_max (#8): 정상 < 6°, 결함 >= 10°
    - cervical_pitch_max (#2): 정상 < 12°, 결함 >= 18°
    """
    c_yaw = feats_24[0]
    t_roll = feats_24[8]
    c_pitch = feats_24[2]

    # 물리적 지표 기반 결함도 점수
    defect_score = 0
    if c_yaw >= 22.0 or t_roll >= 10.0 or c_pitch >= 18.0:
        defect_score += 2
    elif c_yaw >= 15.0 or t_roll >= 6.0 or c_pitch >= 12.0:
        defect_score += 1

    if label == 1:
        if defect_score >= 2:
            return 1 # 확실한 결함
        elif defect_score == 1:
            return 2 # 경미한 결함
        else:
            return 3 # 모호한 결함 (라벨은 결함이나 물리적 변위 미미)
    else:
        if defect_score == 0:
            return 1 # 확실한 정상
        elif defect_score == 1:
            return 2 # 경미한 자세 불안
        else:
            return 4 # 잠재적 이상치 (라벨은 정상이나 물리적 변위 큼)


def build_side_lunge_c4_cache():
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 c4 (척추의 중립) 정예 데이터셋 캐시 빌더")
    print("  [원천 정제 데이터]: 라벨-설명 일치 검증 버전 (side_lunge_clean_folders.json)")
    print("=" * 80)

    # 1. 클린 폴더 목록 로드
    if os.path.exists(CLEAN_FOLDERS_JSON):
        with open(CLEAN_FOLDERS_JSON, 'r', encoding='utf-8') as f:
            clean_folders = set(json.load(f))
        print(f"[*] 클린 폴더 목록 로드: {len(clean_folders)}개 폴더")
    else:
        clean_folders = None
        print("[!] clean_folders.json 없음, 전체 폴더 탐색")

    # 2. 라벨 로드
    labels_dict = load_all_labels(LABEL_DIR)
    print(f"[*] 로드된 2D JSON 라벨 수: {len(labels_dict)}개")

    # 3. npz 파일 탐색
    npz_pattern = os.path.join(MP33_DIR, "*.npz")
    all_npz = glob.glob(npz_pattern)
    print(f"[*] 탐색된 총 MediaPipe Full33 64F 파일: {len(all_npz)}개")

    matched_items = []
    for npz_path in all_npz:
        bname = os.path.basename(npz_path).replace('.npz', '')
        # bname format: {folder}_{view}
        parts = bname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        folder, view = parts[0], parts[1]
        if view not in VIEW_MAP:
            continue

        if clean_folders is not None and folder not in clean_folders:
            continue

        if folder not in labels_dict:
            continue

        matched_items.append((npz_path, folder, view, labels_dict[folder]))

    print(f"[*] 조건 일치 타깃 시퀀스: {len(matched_items)}개")
    if len(matched_items) == 0:
        print("[!] 타깃 시퀀스가 없습니다. 전처리가 완료될 때까지 대기하거나 확인 필요.")
        return False

    # 피처 추출
    features_list = []
    temporal_list = []
    labels_list = []
    subjects_list = []
    views_list = []
    folders_list = []
    keys_list = []
    ambiguity_list = []

    print("[*] 24D 도메인 피처 및 14ch 시계열 텐서 추출 시작...")
    for npz_path, folder, view, l_info in tqdm(matched_items, desc="Feature Extraction"):
        try:
            data = np.load(npz_path)
            seq_33 = data['kps_33_3d_64f']
        except Exception:
            continue

        f24 = extract_lunge_c4_features_v3(seq_33)
        t14 = extract_lunge_c4_temporal_features(seq_33)
        lbl = l_info['label']
        amb = compute_ambiguity_level(f24, lbl)

        # 피험자 번호 추출 (e.g., [오류]_D02_피험자1_Key146 -> subj 1)
        subj = 1
        for part in folder.split('_'):
            if '피험자' in part:
                digits = ''.join(c for c in part if c.isdigit())
                if digits:
                    subj = int(digits)
                break

        features_list.append(f24)
        temporal_list.append(t14)
        labels_list.append(lbl)
        subjects_list.append(subj)
        views_list.append(view)
        folders_list.append(folder)
        keys_list.append(l_info['key'])
        ambiguity_list.append(amb)

    X_static = np.array(features_list, dtype=np.float32)
    X_temporal = np.array(temporal_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)
    folders = np.array(folders_list)
    keys = np.array(keys_list)
    ambiguity = np.array(ambiguity_list, dtype=np.int32)

    os.makedirs(os.path.dirname(OUT_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_CACHE_PATH,
        X_static=X_static,
        X_temporal=X_temporal,
        y=y,
        subjects=subjects,
        views=views,
        folders=folders,
        keys=keys,
        ambiguity=ambiguity,
        feature_names=np.array(LUNGE_C4_FEATURE_NAMES_V3)
    )

    print("\n" + "=" * 80)
    print(f"  [+] 캐시 저장 성공: {OUT_CACHE_PATH}")
    print(f"  [+] 총 샘플 수: {len(y)}개")
    print(f"  [+] 클래스 분포: 정상(0) = {np.sum(y == 0)}개, 결함(1) = {np.sum(y == 1)}개 (결함비율 {np.mean(y)*100:.2f}%)")
    print(f"  [+] 정적 피처 shape: {X_static.shape}")
    print(f"  [+] 시계열 텐서 shape: {X_temporal.shape}")
    print(f"  [+] 피험자 분포: {np.unique(subjects, return_counts=True)}")
    print(f"  [+] 시점 분포: {np.unique(views, return_counts=True)}")
    print("=" * 80)
    return True


if __name__ == '__main__':
    build_side_lunge_c4_cache()
