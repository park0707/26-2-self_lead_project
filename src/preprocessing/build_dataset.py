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
from src.adapters.uiprmd_adapter import UIPRMDAdapter
from src.adapters.rehab_adapter import REHABAdapter
from src.preprocessing.normalizer2d import GeometricNormalizer2D

OUTPUT_BASE_DIR = 'd:/Data/전처리_데이터'
AIHUB_OUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'AIHub')
UIPRMD_OUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'UI-PRMD')
REHAB_OUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'REHAB24-6')

def process_single_aihub_json(json_path: str) -> Optional[Dict[str, Any]]:
    """단일 AIHub 2D JSON 파일 파싱, 2D 5단계 정규화 및 .npz 저장"""
    try:
        seq = AIHubAdapter.parse_2d_json(json_path)
        if seq is None:
            return None

        norm_res = GeometricNormalizer2D.process_sequence_2d(seq.keypoints_2d, seq.exercise_name)
        if norm_res is None:
            return None

        rel_path = os.path.relpath(os.path.dirname(json_path), 'd:/Data/01_raw_data/labels_json')
        target_dir = os.path.join(AIHUB_OUT_DIR, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        npz_filename = f"{seq.sequence_id}.npz"
        npz_path = os.path.join(target_dir, npz_filename)

        meta_dict = {
            'sequence_id': seq.sequence_id,
            'exercise_name': seq.exercise_name,
            'subject_id': seq.subject_id,
            'source_dataset': seq.source_dataset,
            'fps': seq.fps,
            'orig_frames': norm_res['orig_frames'],
            'rot_angle_deg': norm_res['rot_angle_deg'],
            'torso_scale': norm_res['torso_scale'],
            'is_flipped': norm_res['is_flipped']
        }

        np.savez_compressed(
            npz_path,
            keypoints_2d_norm=norm_res['keypoints_2d_norm'],     # (64, 15, 2)
            keypoints_2d_raw=seq.keypoints_2d,                   # (T_orig, 15, 2)
            label=np.int32(seq.label),
            conditions=np.array(seq.conditions or [], dtype=bool),
            metadata=json.dumps(meta_dict, ensure_ascii=False)
        )

        return {
            'dataset': 'AIHub',
            'seq_id': seq.sequence_id,
            'exercise': seq.exercise_name,
            'subject': seq.subject_id,
            'label': seq.label,
            'npz_path': npz_path
        }
    except Exception:
        return None

def process_single_uiprmd_file(txt_path: str) -> Optional[Dict[str, Any]]:
    """단일 UI-PRMD Kinect 파일 처리 및 .npz 저장"""
    try:
        fname = os.path.basename(txt_path)
        is_correct = 1 if '정상' in txt_path or 'Segmented' in txt_path else 0
        
        movement_name = 'UI-PRMD_Movement'
        for m_code in [f"m{i:02d}" for i in range(1, 11)]:
            if m_code in fname:
                movement_name = m_code
                break

        seq = UIPRMDAdapter.parse_kinect_txt_2d(txt_path, movement_name, is_correct)
        if seq is None:
            return None

        norm_res = GeometricNormalizer2D.process_sequence_2d(seq.keypoints_2d, seq.exercise_name)
        if norm_res is None:
            return None

        rel_dir = os.path.relpath(os.path.dirname(txt_path), 'd:/Data/06_external_benchmarks/01_UI-PRMD')
        target_dir = os.path.join(UIPRMD_OUT_DIR, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

        npz_filename = f"{seq.sequence_id}.npz"
        npz_path = os.path.join(target_dir, npz_filename)

        meta_dict = {
            'sequence_id': seq.sequence_id,
            'exercise_name': seq.exercise_name,
            'source_dataset': seq.source_dataset,
            'fps': seq.fps,
            'orig_frames': norm_res['orig_frames'],
            'rot_angle_deg': norm_res['rot_angle_deg'],
            'torso_scale': norm_res['torso_scale'],
            'is_flipped': norm_res['is_flipped']
        }

        np.savez_compressed(
            npz_path,
            keypoints_2d_norm=norm_res['keypoints_2d_norm'],
            keypoints_2d_raw=seq.keypoints_2d,
            label=np.int32(seq.label),
            conditions=np.array([], dtype=bool),
            metadata=json.dumps(meta_dict, ensure_ascii=False)
        )

        return {
            'dataset': 'UI-PRMD',
            'seq_id': seq.sequence_id,
            'exercise': seq.exercise_name,
            'label': seq.label,
            'npz_path': npz_path
        }
    except Exception:
        return None

def process_single_rehab_rep(item: Tuple[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """단일 REHAB24-6 repetition 구간 처리 및 .npz 저장"""
    npy_path, rec = item
    try:
        vid = rec['video_id']
        ex_id = rec['exercise_id']
        rep_num = rec['repetition_number']
        person_id = rec['person_id']
        correctness = rec['correctness']
        s_frame = rec['first_frame']
        e_frame = rec['last_frame']

        # 30fps 파일인지 120fps 파일인지 판별
        fps = 120.0 if '120fps' in npy_path else 30.0
        # 120fps의 경우 segmentation frame 인덱스 스케일링 (120/30 = 4배)
        if '120fps' in npy_path:
            s_frame = s_frame * 4
            e_frame = e_frame * 4

        cam_tag = 'c17' if 'c17' in npy_path else ('c18' if 'c18' in npy_path else 'cam')
        seq_id = f"REHAB_Ex{ex_id}_{vid}_{cam_tag}_rep{rep_num:02d}"
        exercise_name = f"REHAB_Ex{ex_id}"
        subject_id = f"Person_{person_id:02d}"

        seq = REHABAdapter.parse_repetition_2d(
            npy_path=npy_path,
            start_frame=s_frame,
            end_frame=e_frame,
            exercise_name=exercise_name,
            label=correctness,
            seq_id=seq_id,
            subject_id=subject_id,
            fps=fps
        )
        if seq is None:
            return None

        norm_res = GeometricNormalizer2D.process_sequence_2d(seq.keypoints_2d, seq.exercise_name)
        if norm_res is None:
            return None

        target_dir = os.path.join(REHAB_OUT_DIR, f"Ex{ex_id}", f"{int(fps)}fps")
        os.makedirs(target_dir, exist_ok=True)

        npz_filename = f"{seq.sequence_id}.npz"
        npz_path = os.path.join(target_dir, npz_filename)

        meta_dict = {
            'sequence_id': seq.sequence_id,
            'exercise_name': seq.exercise_name,
            'subject_id': seq.subject_id,
            'source_dataset': seq.source_dataset,
            'fps': seq.fps,
            'orig_frames': norm_res['orig_frames'],
            'rot_angle_deg': norm_res['rot_angle_deg'],
            'torso_scale': norm_res['torso_scale'],
            'is_flipped': norm_res['is_flipped'],
            'mocap_erroneous': rec.get('mocap_erroneous', 0)
        }

        np.savez_compressed(
            npz_path,
            keypoints_2d_norm=norm_res['keypoints_2d_norm'],
            keypoints_2d_raw=seq.keypoints_2d,
            label=np.int32(seq.label),
            conditions=np.array([], dtype=bool),
            metadata=json.dumps(meta_dict, ensure_ascii=False)
        )

        return {
            'dataset': 'REHAB24-6',
            'seq_id': seq.sequence_id,
            'exercise': seq.exercise_name,
            'subject': seq.subject_id,
            'label': seq.label,
            'npz_path': npz_path
        }
    except Exception:
        return None

def build_subject_wise_splits(aihub_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """피험자(Subject) 단위 누수 방지 Train(70%) : Val(15%) : Test(15%) 분할 메타데이터 생성"""
    subjects = sorted(list(set(r['subject'] for r in aihub_records if r.get('subject'))))
    np.random.seed(42)
    shuffled_subjects = np.random.permutation(subjects).tolist()

    n_total = len(shuffled_subjects)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_subjs = set(shuffled_subjects[:n_train])
    val_subjs = set(shuffled_subjects[n_train:n_train + n_val])
    test_subjs = set(shuffled_subjects[n_train + n_val:])

    train_ids = [r['seq_id'] for r in aihub_records if r.get('subject') in train_subjs]
    val_ids = [r['seq_id'] for r in aihub_records if r.get('subject') in val_subjs]
    test_ids = [r['seq_id'] for r in aihub_records if r.get('subject') in test_subjs]

    return {
        'split_strategy': 'Subject-wise Group Split',
        'train_subjects': list(train_subjs),
        'val_subjects': list(val_subjs),
        'test_subjects': list(test_subjs),
        'train_count': len(train_ids),
        'val_count': len(val_ids),
        'test_count': len(test_ids),
        'train_sequences': train_ids,
        'val_sequences': val_ids,
        'test_sequences': test_ids
    }

def main():
    start_time = time.time()
    os.makedirs(AIHUB_OUT_DIR, exist_ok=True)
    os.makedirs(UIPRMD_OUT_DIR, exist_ok=True)
    os.makedirs(REHAB_OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("[START] 2D 15-Joint 전처리 및 로컬 영구 저장 파이프라인 시작")
    print("=" * 60)

    num_workers = min(12, os.cpu_count() or 4)
    print(f"[*] 멀티프로세싱 가동: {num_workers}개 워커 프로세스")

    # 1. AIHub 2D JSON 처리
    aihub_jsons = glob.glob('d:/Data/01_raw_data/labels_json/**/*.json', recursive=True)
    aihub_2d_jsons = [f for f in aihub_jsons if '_3D.json' not in f and not f.endswith('_3D.json')]
    print(f"[*] AI Hub 2D JSON 대상 파일: {len(aihub_2d_jsons):,}개")

    aihub_success_records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_aihub_json, p): p for p in aihub_2d_jsons}
        for future in tqdm(as_completed(futures), total=len(futures), desc="AIHub 2D 전처리"):
            res = future.result()
            if res is not None:
                aihub_success_records.append(res)
    print(f"[OK] AI Hub 전처리 완료: {len(aihub_success_records):,} / {len(aihub_2d_jsons):,}개 성공")

    # 2. UI-PRMD Kinect 처리
    uiprmd_files = glob.glob('d:/Data/06_external_benchmarks/01_UI-PRMD/Kinect_키넥트/**/*.txt', recursive=True)
    uiprmd_pos_files = [f for f in uiprmd_files if '위치' in f or 'Positions' in f or 'positions' in f]
    print(f"[*] UI-PRMD Kinect 위치 파일: {len(uiprmd_pos_files):,}개")

    uiprmd_success_records = []
    if uiprmd_pos_files:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_uiprmd_file, p): p for p in uiprmd_pos_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="UI-PRMD 2D 전처리"):
                res = future.result()
                if res is not None:
                    uiprmd_success_records.append(res)
        print(f"[OK] UI-PRMD 전처리 완료: {len(uiprmd_success_records):,} / {len(uiprmd_pos_files):,}개 성공")

    # 3. REHAB24-6 처리 (Segmentation.csv 기반 반복 단위 추출)
    seg_csv_path = 'd:/Data/06_external_benchmarks/02_REHAB24-6/metadata/Segmentation.csv'
    rehab_seg_records = REHABAdapter.load_segmentation_records(seg_csv_path)
    print(f"[*] REHAB24-6 세그멘테이션 반복 레코드: {len(rehab_seg_records):,}개")

    # REHAB 2D/3D NPY 파일들과 세그멘테이션 레코드 매칭 작업 큐 생성
    rehab_tasks = []
    rehab_joints_root = 'd:/Data/06_external_benchmarks/02_REHAB24-6/2D_관절좌표_joints'
    if not os.path.exists(rehab_joints_root):
        rehab_joints_root = 'd:/Data/06_external_benchmarks/02_REHAB24-6/3D_관절좌표_joints'

    for rec in rehab_seg_records:
        ex_dir = f"Ex{rec['exercise_id']}"
        vid = rec['video_id']
        # 30fps c17 및 c18 파일 매칭
        for fps_tag in ['30fps', '120fps']:
            for cam_tag in ['c17', 'c18']:
                npy_name = f"{vid}-{cam_tag}-{fps_tag}.npy"
                npy_path = os.path.join(rehab_joints_root, ex_dir, npy_name)
                if os.path.exists(npy_path):
                    rehab_tasks.append((npy_path, rec))

    print(f"[*] REHAB24-6 추출 대상 반복 작업 수: {len(rehab_tasks):,}개")

    rehab_success_records = []
    if rehab_tasks:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_rehab_rep, task): task for task in rehab_tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="REHAB24-6 2D 전처리"):
                res = future.result()
                if res is not None:
                    rehab_success_records.append(res)
        print(f"[OK] REHAB24-6 전처리 완료: {len(rehab_success_records):,} / {len(rehab_tasks):,}개 성공")

    # 4. 피험자 분할 메타데이터 생성
    print("[*] 피험자(Subject-wise) 누수 방지 분할 데이터셋 구축 중...")
    splits_dict = build_subject_wise_splits(aihub_success_records)
    splits_path = os.path.join(OUTPUT_BASE_DIR, 'dataset_splits.json')
    with open(splits_path, 'w', encoding='utf-8') as f:
        json.dump(splits_dict, f, ensure_ascii=False, indent=2)

    summary_dict = {
        'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_elapsed_sec': round(time.time() - start_time, 2),
        'aihub_total_processed': len(aihub_success_records),
        'uiprmd_total_processed': len(uiprmd_success_records),
        'rehab_total_processed': len(rehab_success_records),
        'splits_summary': {
            'train_count': splits_dict['train_count'],
            'val_count': splits_dict['val_count'],
            'test_count': splits_dict['test_count']
        }
    }
    summary_path = os.path.join(OUTPUT_BASE_DIR, 'metadata_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"[DONE] 전체 전처리 파이프라인 완료! (총 소요시간: {summary_dict['total_elapsed_sec']}초)")
    print(f"[*] 저장 경로: {OUTPUT_BASE_DIR}")
    print(f"[*] AIHub: {len(aihub_success_records):,}개 | UI-PRMD: {len(uiprmd_success_records):,}개 | REHAB24-6: {len(rehab_success_records):,}개")
    print(f"[*] Train: {splits_dict['train_count']} | Val: {splits_dict['val_count']} | Test: {splits_dict['test_count']}")
    print("=" * 60)

if __name__ == '__main__':
    main()
