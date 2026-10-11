import os
import sys
import glob
import json
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.adapters.aihub_adapter import AIHubAdapter
from src.adapters.mediapipe_adapter import MediaPipeAdapter
from src.preprocessing.normalizer2d import GeometricNormalizer2D
from src.preprocessing.normalizer3d import GeometricNormalizer3D
from src.features.kinematics_2d import KinematicsExtractor2D
from src.features.kinematics_3d import KinematicsExtractor3D
from src.features.feature_fusion import FeatureFusionEngine
from src.features.build_dual_splits import (
    RAW_LABELS_BASE,
    TRAIN_SUBJECTS, VALID_SUBJECTS, TEST_SUBJECTS,
    TRAIN_VIEWS, VALID_VIEWS, TEST_VIEWS,
    ALL_VIEWS, extract_clean_subject, get_dual_split_category
)

RAW_IMAGES_BASE = 'd:/Data/01_raw_data/images'
BASE_DUAL_DIR = 'd:/Data/피처_데이터/Dual_Splits'

OUTPUT_COND1_DIR = 'd:/Data/피처_데이터/Condition1_AIHub_2D'
OUTPUT_COND2_DIR = 'd:/Data/피처_데이터/Condition2_Fused_2D_3D'
OUTPUT_COND3_DIR = 'd:/Data/피처_데이터/Condition3_MediaPipe_3D'

SPLIT_MAP = {
    'Dual_Train.npz': 'Train_Main',
    'Dual_Valid.npz': 'Valid_Main',
    'Dual_Test_Strict.npz': 'Test_Strict_Dual_Holdout',
    'Dual_Test_Unseen_Subj.npz': 'Test_Unseen_Subject_Only',
    'Dual_Test_Unseen_View.npz': 'Test_Unseen_View_Only'
}

VIEW_KEY_TO_PREFIX = {
    'view1': '시점A',
    'view2': '시점B',
    'view3': '시점C',
    'view4': '시점D',
    'view5': '시점E'
}

def process_single_sequence_3source(args: Tuple[str, str, str]) -> List[Dict[str, Any]]:
    """단일 시퀀스(이미지 폴더 + JSON)에 대해 5개 시점의 2D, 3D, 융합 피처를 동시 추출"""
    seq_dir, json_path, exercise_name = args
    results = []
    seq_id = os.path.basename(os.path.normpath(seq_dir))
    subject_id = extract_clean_subject(seq_id)

    # Pose Landmarker Detector는 프로세스별로 캐싱됨
    det = MediaPipeAdapter.get_detector()

    for view_key in ALL_VIEWS:
        view_prefix = VIEW_KEY_TO_PREFIX.get(view_key, '시점A')
        split_cat = get_dual_split_category(subject_id, view_key)
        if split_cat == 'Other':
            continue

        try:
            # 1. AI Hub 2D 파싱 및 피처 추출
            seq_2d = AIHubAdapter.parse_2d_json(json_path, view_key=view_key) if json_path and os.path.exists(json_path) else None
            feat_a_2d = None
            feat_b_2d = None
            feat_names_2d = []
            
            if seq_2d is not None and seq_2d.keypoints_2d is not None and len(seq_2d.keypoints_2d) >= 4:
                norm_res_2d = GeometricNormalizer2D.process_sequence_2d(seq_2d.keypoints_2d, exercise_name)
                if norm_res_2d is not None:
                    feat_a_2d, feat_names_2d = KinematicsExtractor2D.build_format_a_summary(norm_res_2d['keypoints_2d_norm'])
                    feat_b_2d = KinematicsExtractor2D.build_format_b_timeseries(norm_res_2d['keypoints_2d_norm'])

            # 2. MediaPipe 3D & 2D 파싱 및 피처 추출
            seq_mp = MediaPipeAdapter.parse_sequence_from_images(seq_dir, view_prefix=view_prefix, min_frames=4, detector=det)
            feat_a_3d = None
            feat_b_3d = None
            feat_names_3d = []

            if seq_mp is not None and seq_mp.keypoints_3d is not None and len(seq_mp.keypoints_3d) >= 4:
                norm_res_3d = GeometricNormalizer3D.process_sequence_3d(seq_mp.keypoints_3d, exercise_name)
                if norm_res_3d is not None:
                    feat_a_3d, feat_names_3d = KinematicsExtractor3D.build_format_a_summary_3d(norm_res_3d['keypoints_3d_norm'])
                    feat_b_3d = KinematicsExtractor3D.build_format_b_timeseries_3d(norm_res_3d['keypoints_3d_norm'])

            # 3. 융합 피처 생성 (2D + 3D)
            feat_a_fused = None
            feat_b_fused = None
            feat_names_fused = []

            if feat_a_2d is not None and feat_a_3d is not None:
                feat_a_fused, feat_names_fused = FeatureFusionEngine.fuse_format_a_features(
                    feat_a_2d, feat_a_3d, feat_names_2d, feat_names_3d
                )
                feat_b_fused = FeatureFusionEngine.fuse_format_b_timeseries(feat_b_2d, feat_b_3d)

            # 라벨 및 조건
            label = seq_2d.label if seq_2d is not None else (seq_mp.label if seq_mp is not None else 1)
            conditions = seq_2d.conditions if seq_2d is not None else []

            results.append({
                'sequence_id': f"{seq_id}_{view_key}",
                'subject': subject_id,
                'view': view_key,
                'category': split_cat,
                'label': label,
                'conditions': conditions,
                'feat_a_cond1': feat_a_2d,
                'feat_b_cond1': feat_b_2d,
                'feat_names_cond1': feat_names_2d,
                'feat_a_cond2': feat_a_fused,
                'feat_b_cond2': feat_b_fused,
                'feat_names_cond2': feat_names_fused,
                'feat_a_cond3': feat_a_3d,
                'feat_b_cond3': feat_b_3d,
                'feat_names_cond3': feat_names_3d,
            })
        except Exception:
            continue

    return results

def build_all_3source_datasets(max_workers: int = 8):
    """3대 키포인트 소스 조건 데이터셋 일괄 구축 및 NPZ 저장"""
    print("=" * 80)
    print(" [*] 연구 계획서 1.5: 3가지 키포인트 소스 조건 Dual Splits 빌더 실행")
    print(f"     1. 조건 1: AI Hub 2D 단독 -> {OUTPUT_COND1_DIR}")
    print(f"     2. 조건 2: 2D+3D 융합     -> {OUTPUT_COND2_DIR}")
    print(f"     3. 조건 3: MediaPipe 3D  -> {OUTPUT_COND3_DIR}")
    print("=" * 80)

    for d in [OUTPUT_COND1_DIR, OUTPUT_COND2_DIR, OUTPUT_COND3_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1. Condition 1은 기존 검증된 Dual_Splits 파일들 복사로 즉시 구성
    print("\n[1/3] Condition 1 (AI Hub 2D) 데이터셋 복사 및 검증 중...")
    for fn in SPLIT_MAP.keys():
        src_p = os.path.join(BASE_DUAL_DIR, fn)
        dst_p = os.path.join(OUTPUT_COND1_DIR, fn)
        if os.path.exists(src_p):
            shutil.copy2(src_p, dst_p)
            print(f"  [+] 복사 완료: {dst_p}")

    # 2. 이미지 폴더 및 JSON 매핑 목록 구성
    print("\n[2/3] 이미지 시퀀스 및 라벨 JSON 매핑 목록 탐색 중...")
    tasks = []
    exercise_folders = sorted(os.listdir(RAW_IMAGES_BASE))

    for ex_name in exercise_folders:
        ex_img_dir = os.path.join(RAW_IMAGES_BASE, ex_name)
        ex_lbl_dir = os.path.join(RAW_LABELS_BASE, ex_name, '2D')
        if not os.path.isdir(ex_img_dir):
            continue

        for sid in os.listdir(ex_img_dir):
            seq_img_p = os.path.join(ex_img_dir, sid)
            if not os.path.isdir(seq_img_p):
                continue
            
            json_p = os.path.join(ex_lbl_dir, f"{sid}.json") if os.path.exists(ex_lbl_dir) else ""
            tasks.append((seq_img_p, json_p, ex_name))

    print(f"[*] 총 {len(tasks)}개 시퀀스 대상 3D 및 융합 피처 추출 시작 (Worker: {max_workers})...")

    cond2_pools: Dict[str, Dict[str, List]] = {cat: {'A': [], 'B': [], 'y': [], 'seq_ids': [], 'subj': [], 'views': []} for cat in SPLIT_MAP.values()}
    cond3_pools: Dict[str, Dict[str, List]] = {cat: {'A': [], 'B': [], 'y': [], 'seq_ids': [], 'subj': [], 'views': []} for cat in SPLIT_MAP.values()}

    cond2_names = []
    cond3_names = []

    # 멀티프로세싱 실행
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_sequence_3source, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting 3D/Fused"):
            try:
                seq_results = fut.result()
                for r in seq_results:
                    cat = r['category']
                    if cat not in cond2_pools:
                        continue

                    # Condition 2 (Fused 2D+3D)
                    if r['feat_a_cond2'] is not None and r['feat_b_cond2'] is not None:
                        cond2_pools[cat]['A'].append(r['feat_a_cond2'])
                        cond2_pools[cat]['B'].append(r['feat_b_cond2'])
                        cond2_pools[cat]['y'].append(r['label'])
                        cond2_pools[cat]['seq_ids'].append(r['sequence_id'])
                        cond2_pools[cat]['subj'].append(r['subject'])
                        cond2_pools[cat]['views'].append(r['view'])
                        if not cond2_names:
                            cond2_names = r['feat_names_cond2']

                    # Condition 3 (Pure MediaPipe 3D)
                    if r['feat_a_cond3'] is not None and r['feat_b_cond3'] is not None:
                        cond3_pools[cat]['A'].append(r['feat_a_cond3'])
                        cond3_pools[cat]['B'].append(r['feat_b_cond3'])
                        cond3_pools[cat]['y'].append(r['label'])
                        cond3_pools[cat]['seq_ids'].append(r['sequence_id'])
                        cond3_pools[cat]['subj'].append(r['subject'])
                        cond3_pools[cat]['views'].append(r['view'])
                        if not cond3_names:
                            cond3_names = r['feat_names_cond3']
            except Exception:
                continue

    # 3. NPZ 파일 저장
    print("\n[3/3] Condition 2 및 Condition 3 분할 NPZ 파일 저장 중...")
    for fn, cat in SPLIT_MAP.items():
        # Condition 2 저장
        c2 = cond2_pools[cat]
        if len(c2['A']) > 0:
            c2_path = os.path.join(OUTPUT_COND2_DIR, fn)
            np.savez_compressed(
                c2_path,
                X_Format_A=np.array(c2['A'], dtype=np.float32),
                X_Format_B=np.array(c2['B'], dtype=np.float32),
                y=np.array(c2['y'], dtype=np.int32),
                sequence_ids=np.array(c2['seq_ids']),
                subjects=np.array(c2['subj']),
                views=np.array(c2['views']),
                feature_names=np.array(cond2_names)
            )
            print(f"  [+] Condition 2 ({fn}): A={np.array(c2['A']).shape}, B={np.array(c2['B']).shape}")

        # Condition 3 저장
        c3 = cond3_pools[cat]
        if len(c3['A']) > 0:
            c3_path = os.path.join(OUTPUT_COND3_DIR, fn)
            np.savez_compressed(
                c3_path,
                X_Format_A=np.array(c3['A'], dtype=np.float32),
                X_Format_B=np.array(c3['B'], dtype=np.float32),
                y=np.array(c3['y'], dtype=np.int32),
                sequence_ids=np.array(c3['seq_ids']),
                subjects=np.array(c3['subj']),
                views=np.array(c3['views']),
                feature_names=np.array(cond3_names)
            )
            print(f"  [+] Condition 3 ({fn}): A={np.array(c3['A']).shape}, B={np.array(c3['B']).shape}")

    print("\n[+] 3대 키포인트 소스 조건 데이터셋 전수 구축 완료!")

if __name__ == '__main__':
    build_all_3source_datasets(max_workers=8)
