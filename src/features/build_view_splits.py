import os
import sys
import glob
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

from src.adapters.aihub_adapter import AIHubAdapter
from src.preprocessing.normalizer2d import GeometricNormalizer2D
from src.features.kinematics_2d import KinematicsExtractor2D

RAW_LABELS_BASE = 'd:/Data/01_raw_data/labels_json'
OUTPUT_SPLITS_DIR = 'd:/Data/피처_데이터/View_Splits'

VIEW_ROLES = {
    'view1': 'Train',   # 시점A (정면 0°)
    'view2': 'Train',   # 시점B (전방 대각선 45°)
    'view5': 'Train',   # 시점E (후면 180°)
    'view4': 'Valid',   # 시점D (후방 대각선 135°)
    'view3': 'Test',    # 시점C (완전 측면 90° - Hold-out Test)
}

def process_single_json_all_views(json_path: str) -> List[Dict[str, Any]]:
    """단일 JSON 파일에서 5개 시점(view1~view5)을 파싱, 정규화 및 피처 추출"""
    results = []
    for view_key, role in VIEW_ROLES.items():
        try:
            seq = AIHubAdapter.parse_2d_json(json_path, view_key=view_key)
            if seq is None:
                continue

            norm_res = GeometricNormalizer2D.process_sequence_2d(seq.keypoints_2d, seq.exercise_name)
            if norm_res is None:
                continue

            feat_a, feat_names = KinematicsExtractor2D.build_format_a_summary(norm_res['keypoints_2d_norm'])
            feat_b = KinematicsExtractor2D.build_format_b_timeseries(norm_res['keypoints_2d_norm'])

            results.append({
                'seq_id': seq.sequence_id,
                'view_key': view_key,
                'role': role,
                'exercise': seq.exercise_name,
                'subject': seq.subject_id,
                'label': seq.label,
                'conditions': seq.conditions,
                'feat_a': feat_a,
                'feat_b': feat_b,
                'feat_names': feat_names
            })
        except Exception:
            continue
    return results

def build_all_view_splits(num_workers: int = 12):
    """전체 AI Hub JSON 파일에 대해 5개 시점 피처를 일괄 추출하고 Train/Valid/Test 세트로 분할 패킹"""
    print("=" * 70)
    print(f"[*] 5대 시점(Multi-View) 전수 추출 및 View-Split 데이터셋 빌드 시작 (프로세스: {num_workers})")
    print("    - Train (3개 시점): view1 (시점A, 정면), view2 (시점B, 전측면 45°), view5 (시점E, 후면)")
    print("    - Valid (1개 시점): view4 (시점D, 후측면 135°)")
    print("    - Test  (1개 시점): view3 (시점C, 완전 측면 90° - Hold-out Test)")
    print("=" * 70)

    start_time = time.time()
    os.makedirs(OUTPUT_SPLITS_DIR, exist_ok=True)

    # 1. 전체 JSON 목록 수집
    json_pattern = os.path.join(RAW_LABELS_BASE, '**', '2D', '*.json')
    json_files = glob.glob(json_pattern, recursive=True)
    print(f"[*] 대상 2D 라벨 JSON 파일: {len(json_files):,}개 (총 예상 시퀀스: {len(json_files) * 5:,}개)")

    all_records = {'Train': [], 'Valid': [], 'Test': []}
    view_counts = {v: 0 for v in VIEW_ROLES}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_json_all_views, p): p for p in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Multi-View 피처 추출"):
            res_list = future.result()
            for r in res_list:
                role = r['role']
                all_records[role].append(r)
                view_counts[r['view_key']] += 1

    elapsed = time.time() - start_time
    print(f"\n[+] Multi-View 피처 추출 완료 (소요 시간: {elapsed:.1f}초)")
    for v, cnt in sorted(view_counts.items()):
        print(f"    - {v} ({VIEW_ROLES[v]}): {cnt:,}개 시퀀스")

    # 2. 역할별 데이터 패킹 및 저장
    feat_names = None
    split_summary = {}

    for role in ['Train', 'Valid', 'Test']:
        recs = all_records[role]
        if not recs:
            continue
        recs.sort(key=lambda x: (x['seq_id'], x['view_key']))

        if feat_names is None:
            feat_names = recs[0]['feat_names']

        X_a = np.stack([r['feat_a'] for r in recs], axis=0) # (N, 97)
        X_b = np.stack([r['feat_b'] for r in recs], axis=0) # (N, 64, 69)
        y = np.array([r['label'] for r in recs], dtype=np.int32)
        seq_ids = [f"{r['seq_id']}_{r['view_key']}" for r in recs]
        subjects = [r['subject'] for r in recs]
        views = [r['view_key'] for r in recs]

        # 저장
        npz_filename = f"AIHub_{role}_Split.npz"
        npz_path = os.path.join(OUTPUT_SPLITS_DIR, npz_filename)

        np.savez_compressed(
            npz_path,
            X_Format_A=X_a,
            X_Format_B=X_b,
            y=y,
            sequence_ids=seq_ids,
            subjects=subjects,
            views=views,
            feature_names=feat_names
        )

        normal_cnt = int(np.sum(y == 1))
        error_cnt = int(np.sum(y == 0))
        split_summary[role] = {
            'file': npz_filename,
            'total_samples': len(y),
            'shape_A': list(X_a.shape),
            'shape_B': list(X_b.shape),
            'normal_count': normal_cnt,
            'error_count': error_cnt,
            'normal_ratio': round(normal_cnt / len(y) * 100, 2)
        }
        print(f"[+] {role:5s} 세트 저장 완료: {npz_filename} | Total={len(y):,}, Shape_A={X_a.shape}, Normal={normal_cnt:,}, Error={error_cnt:,}")

    # 요약 JSON 저장
    meta_summary_path = os.path.join(OUTPUT_SPLITS_DIR, 'view_splits_summary.json')
    with open(meta_summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'elapsed_sec': elapsed,
            'view_roles': VIEW_ROLES,
            'view_counts': view_counts,
            'splits': split_summary
        }, f, indent=2, ensure_ascii=False)
    print(f"[+] 분할 요약 메타데이터 저장: {meta_summary_path}")

if __name__ == '__main__':
    build_all_view_splits(num_workers=12)
