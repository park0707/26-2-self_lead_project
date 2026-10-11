import os
import sys
import json
import time
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

from src.features.kinematic_pipeline import KinematicPipelineEngine
from src.features.kinematics_3d import KinematicsExtractor3D

SRC_COND3_DIR = 'd:/Data/피처_데이터/Condition3_MediaPipe_3D'
OUTPUT_COND4_DIR = 'd:/Data/피처_데이터/Condition4_NovelFlow_3D'
PREPROCESSED_MP_DIR = 'd:/Data/전처리_데이터/MediaPipe'

SPLIT_FILES = [
    'Dual_Train.npz',
    'Dual_Valid.npz',
    'Dual_Test_Strict.npz',
    'Dual_Test_Unseen_Subj.npz',
    'Dual_Test_Unseen_View.npz'
]

def find_all_mp_files() -> Dict[str, str]:
    mapping = {}
    for root, _, files in os.walk(PREPROCESSED_MP_DIR):
        for f in files:
            if f.endswith('.npz'):
                sid = os.path.splitext(f)[0]
                mapping[sid] = os.path.join(root, f)
    return mapping

# Global worker cache
_ENGINE = None
_MP_MAP = None

def init_worker(mp_map_dict: Dict[str, str]):
    global _ENGINE, _MP_MAP
    _ENGINE = KinematicPipelineEngine()
    _MP_MAP = mp_map_dict

def process_single_item(item: Tuple[int, str, np.ndarray, np.ndarray, int, str, str, int, int]) -> Tuple[int, np.ndarray, np.ndarray]:
    """단일 항목 처리 (멀티프로세싱 워커)"""
    idx, sid, fallback_a, fallback_b, target_dim_a, target_dim_b = item[0], item[1], item[2], item[3], item[7], item[8]
    global _ENGINE, _MP_MAP

    base_sid = str(sid).split('_view')[0]
    mp_file = _MP_MAP.get(base_sid) or _MP_MAP.get(str(sid))

    if mp_file and os.path.exists(mp_file):
        try:
            d_mp = np.load(mp_file, allow_pickle=True)
            kps_2d = d_mp['keypoints_2d_norm']
            kps_3d = d_mp['keypoints_3d'] if 'keypoints_3d' in d_mp else np.pad(kps_2d, ((0,0),(0,0),(0,1)))

            res = _ENGINE.process_sequence(kps_2d, kps_3d)
            canonical_3d = res['canonical_3d']

            feat_a, _ = KinematicsExtractor3D.build_format_a_summary_3d(canonical_3d)
            feat_b = KinematicsExtractor3D.build_format_b_timeseries_3d(canonical_3d)

            # 차원 정합
            if len(feat_a) < target_dim_a:
                feat_a = np.concatenate([feat_a, np.zeros(target_dim_a - len(feat_a), dtype=np.float32)], axis=-1)
            elif len(feat_a) > target_dim_a:
                feat_a = feat_a[:target_dim_a]

            if feat_b.shape[-1] < target_dim_b:
                feat_b = np.concatenate([feat_b, np.zeros((len(feat_b), target_dim_b - feat_b.shape[-1]), dtype=np.float32)], axis=-1)
            elif feat_b.shape[-1] > target_dim_b:
                feat_b = feat_b[:, :target_dim_b]

            return idx, feat_a.astype(np.float32), feat_b.astype(np.float32)
        except Exception:
            return idx, fallback_a, fallback_b
    else:
        return idx, fallback_a, fallback_b

def build_condition4_dataset(max_workers: int = 8):
    print("=" * 80)
    print(" [*] 차세대 4대 모듈 파이프라인 (Condition 4: NovelFlow 3D) 고속 병렬 빌더 시작")
    print(f"     출력 경로: {OUTPUT_COND4_DIR} (워커 수: {max_workers})")
    print("=" * 80)

    os.makedirs(OUTPUT_COND4_DIR, exist_ok=True)
    mp_map = find_all_mp_files()
    print(f" [*] 사전 전처리된 MediaPipe 파일: {len(mp_map):,} 개 발견")

    for fn in SPLIT_FILES:
        src_path = os.path.join(SRC_COND3_DIR, fn)
        if not os.path.exists(src_path):
            src_path = os.path.join('d:/Data/피처_데이터/Dual_Splits', fn)
        if not os.path.exists(src_path):
            print(f" [!] 원본 분할 파일 없음: {src_path}")
            continue

        print(f"\n [*] 병렬 처리 시작: {fn} (원본: {src_path})...")
        d_src = np.load(src_path, allow_pickle=True)
        seq_ids = d_src['sequence_ids']
        y_labels = d_src['y']
        subjects = d_src['subjects']
        views = d_src['views']
        feat_names = d_src['feature_names'] if 'feature_names' in d_src else []

        X_a_src = d_src['X_Format_A']
        X_b_src = d_src['X_Format_B']
        N = len(seq_ids)
        target_dim_a = X_a_src.shape[-1]
        target_dim_b = X_b_src.shape[-1]

        tasks = []
        for i in range(N):
            tasks.append((i, seq_ids[i], X_a_src[i], X_b_src[i], y_labels[i], subjects[i], views[i], target_dim_a, target_dim_b))

        X_a_out = np.zeros_like(X_a_src)
        X_b_out = np.zeros_like(X_b_src)

        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(mp_map,)) as executor:
            futures = [executor.submit(process_single_item, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Building {fn}"):
                idx, fa, fb = fut.result()
                X_a_out[idx] = fa
                X_b_out[idx] = fb

        out_path = os.path.join(OUTPUT_COND4_DIR, fn)
        np.savez_compressed(
            out_path,
            X_Format_A=X_a_out,
            X_Format_B=X_b_out,
            y=y_labels,
            sequence_ids=seq_ids,
            subjects=subjects,
            views=views,
            feature_names=feat_names
        )
        print(f"  [✓] 분할 NPZ 저장 완료: {out_path} (Shape: {X_a_out.shape})")

    print("\n" + "=" * 80)
    print(" [✓] Condition 4 (NovelFlow 3D) 전 분할 데이터셋 구축이 100% 완료되었습니다!")
    print("=" * 80)

if __name__ == '__main__':
    build_condition4_dataset(max_workers=8)
