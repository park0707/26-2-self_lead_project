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
OUTPUT_DUAL_DIR = 'd:/Data/피처_데이터/Dual_Splits'

# 1. 피험자 그룹 정의 (Subject Groups)
TRAIN_SUBJECTS = {'피험자1', '피험자2', '피험자3', '피험자4', '피험자5', '피험자6'}
VALID_SUBJECTS = {'피험자7', '피험자8'}
TEST_SUBJECTS  = {'피험자9'}

# 2. 시점 그룹 정의 (View Groups)
TRAIN_VIEWS = {'view1', 'view2', 'view5'} # 시점A(0°), 시점B(45°), 시점E(180°)
VALID_VIEWS = {'view4'}                   # 시점D(135°)
TEST_VIEWS  = {'view3'}                   # 시점C(90° 측면)

ALL_VIEWS = ['view1', 'view2', 'view3', 'view4', 'view5']

def extract_clean_subject(seq_id: str) -> str:
    """시퀀스 ID에서 피험자 번호 추출"""
    for part in seq_id.split('_'):
        if '피험자' in part:
            return part
    return '피험자1'

def get_dual_split_category(subject: str, view: str) -> str:
    """피험자 x 시점 2차원 교차 분할 카테고리 결정"""
    is_train_subj = subject in TRAIN_SUBJECTS
    is_valid_subj = subject in VALID_SUBJECTS
    is_test_subj  = subject in TEST_SUBJECTS

    is_train_view = view in TRAIN_VIEWS
    is_valid_view = view in VALID_VIEWS
    is_test_view  = view in TEST_VIEWS

    # 1. 순수 학습용 (Train Subj x Train View)
    if is_train_subj and is_train_view:
        return 'Train_Main'
    
    # 2. 순수 검증용 (Valid Subj x Valid View)
    if is_valid_subj and is_valid_view:
        return 'Valid_Main'

    # 3. 궁극의 엄격한 복합 홀드아웃 테스트 (Test Subj x Test View)
    if is_test_subj and is_test_view:
        return 'Test_Strict_Dual_Holdout'

    # 4. 인물 일반화 테스트 (Test Subj x Train View: 미학습 인물, 학습 각도)
    if is_test_subj and is_train_view:
        return 'Test_Unseen_Subject_Only'

    # 5. 시점 일반화 테스트 (Train Subj x Test View: 학습 인물, 미학습 각도)
    if is_train_subj and is_test_view:
        return 'Test_Unseen_View_Only'

    # 기타 조합
    if is_valid_subj and is_train_view:
        return 'Train_Aux_Valid_Subj'
    if is_train_subj and is_valid_view:
        return 'Valid_Aux_Train_Subj'
    if is_test_subj and is_valid_view:
        return 'Test_Aux_Valid_View'
    if is_valid_subj and is_test_view:
        return 'Test_Aux_Valid_Subj'

    return 'Other'

def process_single_json_dual(json_path: str) -> List[Dict[str, Any]]:
    """단일 JSON 파일에서 5개 시점을 추출하고 2차원 분할 태깅"""
    results = []
    seq_id = os.path.splitext(os.path.basename(json_path))[0]
    subject_id = extract_clean_subject(seq_id)

    for view_key in ALL_VIEWS:
        try:
            seq = AIHubAdapter.parse_2d_json(json_path, view_key=view_key)
            if seq is None:
                continue

            norm_res = GeometricNormalizer2D.process_sequence_2d(seq.keypoints_2d, seq.exercise_name)
            if norm_res is None:
                continue

            feat_a, feat_names = KinematicsExtractor2D.build_format_a_summary(norm_res['keypoints_2d_norm'])
            feat_b = KinematicsExtractor2D.build_format_b_timeseries(norm_res['keypoints_2d_norm'])

            split_cat = get_dual_split_category(subject_id, view_key)

            results.append({
                'seq_id': seq_id,
                'view_key': view_key,
                'subject': subject_id,
                'category': split_cat,
                'exercise': seq.exercise_name,
                'label': seq.label,
                'conditions': seq.conditions,
                'feat_a': feat_a,
                'feat_b': feat_b,
                'feat_names': feat_names
            })
        except Exception:
            continue
    return results

def build_all_dual_splits(num_workers: int = 12):
    """전체 AI Hub JSON에 대해 2차원 복합 분할 데이터셋을 빌드하고 패킹 저장"""
    print("=" * 70)
    print(f"[*] 2차원 복합 분할 (Subject x View Dual Split) 데이터셋 빌드 시작")
    print(f"    - Train Subjects (6명): {sorted(list(TRAIN_SUBJECTS))}")
    print(f"    - Valid Subjects (2명): {sorted(list(VALID_SUBJECTS))}")
    print(f"    - Test  Subjects (1명): {sorted(list(TEST_SUBJECTS))}")
    print(f"    - Train Views (3개): {sorted(list(TRAIN_VIEWS))} (시점A, B, E)")
    print(f"    - Valid Views (1개): {sorted(list(VALID_VIEWS))} (시점D)")
    print(f"    - Test  Views (1개): {sorted(list(TEST_VIEWS))} (시점C - 측면 90°)")
    print("=" * 70)

    start_time = time.time()
    os.makedirs(OUTPUT_DUAL_DIR, exist_ok=True)

    json_pattern = os.path.join(RAW_LABELS_BASE, '**', '2D', '*.json')
    json_files = glob.glob(json_pattern, recursive=True)
    print(f"[*] 대상 2D 라벨 JSON 파일: {len(json_files):,}개")

    cat_records = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_json_dual, p): p for p in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="2차원 복합 분할 추출"):
            res_list = future.result()
            for r in res_list:
                cat = r['category']
                if cat not in cat_records:
                    cat_records[cat] = []
                cat_records[cat].append(r)

    elapsed = time.time() - start_time
    print(f"\n[+] 2차원 복합 분할 추출 완료 (소요 시간: {elapsed:.1f}초)")

    # 주요 5대 데이터셋 패킹
    target_splits = {
        'Dual_Train.npz': 'Train_Main',                          # 피험자1~6 x 시점A,B,E
        'Dual_Valid.npz': 'Valid_Main',                          # 피험자7~8 x 시점D
        'Dual_Test_Strict.npz': 'Test_Strict_Dual_Holdout',      # 피험자9   x 시점C (복합 홀드아웃)
        'Dual_Test_Unseen_Subj.npz': 'Test_Unseen_Subject_Only', # 피험자9   x 시점A,B,E (인물 일반화)
        'Dual_Test_Unseen_View.npz': 'Test_Unseen_View_Only',    # 피험자1~6 x 시점C (시점 일반화)
    }

    feat_names = None
    summary_dict = {}

    for filename, cat_key in target_splits.items():
        recs = cat_records.get(cat_key, [])
        if not recs:
            print(f"[!] 카테고리 {cat_key}에 데이터가 없습니다.")
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

        out_path = os.path.join(OUTPUT_DUAL_DIR, filename)
        np.savez_compressed(
            out_path,
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
        summary_dict[filename] = {
            'category': cat_key,
            'total_samples': len(y),
            'shape_A': list(X_a.shape),
            'shape_B': list(X_b.shape),
            'normal_count': normal_cnt,
            'error_count': error_cnt,
            'normal_ratio': round(normal_cnt / len(y) * 100, 2),
            'subjects': sorted(list(set(subjects))),
            'views': sorted(list(set(views)))
        }
        print(f"[+] {filename:<26s}: Total={len(y):>5,}, Shape_A={str(X_a.shape):<10}, Normal={normal_cnt:>4,}, Error={error_cnt:>5,} | Subjs={sorted(list(set(subjects)))}, Views={sorted(list(set(views)))}")

    # 요약 JSON 저장
    meta_path = os.path.join(OUTPUT_DUAL_DIR, 'dual_splits_summary.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'elapsed_sec': elapsed,
            'train_subjects': sorted(list(TRAIN_SUBJECTS)),
            'valid_subjects': sorted(list(VALID_SUBJECTS)),
            'test_subjects': sorted(list(TEST_SUBJECTS)),
            'train_views': sorted(list(TRAIN_VIEWS)),
            'valid_views': sorted(list(VALID_VIEWS)),
            'test_views': sorted(list(TEST_VIEWS)),
            'splits': summary_dict
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 2차원 복합 분할 메타데이터 저장: {meta_path}")

if __name__ == '__main__':
    build_all_dual_splits(num_workers=12)
