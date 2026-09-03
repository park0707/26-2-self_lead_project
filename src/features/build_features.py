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

from src.features.kinematics_2d import KinematicsExtractor2D

INPUT_PREPROCESSED_DIR = 'd:/Data/전처리_데이터'
OUTPUT_FEATURES_DIR = 'd:/Data/피처_데이터'
FORMAT_A_DIR = os.path.join(OUTPUT_FEATURES_DIR, 'Format_A_통계요약')
FORMAT_B_DIR = os.path.join(OUTPUT_FEATURES_DIR, 'Format_B_시계열텐서')

def process_single_npz_feature(npz_path: str) -> Optional[Dict[str, Any]]:
    """단일 전처리 .npz 파일 로드 및 Format A (97차원), Format B (64, 69) 피처 추출"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        kps_norm = data['keypoints_2d_norm'] # (64, 15, 2)
        label = int(data['label'])
        conds = data['conditions']
        seq_id = os.path.splitext(os.path.basename(npz_path))[0]

        # 1. Format A (97차원 요약 통계량)
        feat_a, feat_names = KinematicsExtractor2D.build_format_a_summary(kps_norm)
        
        # 2. Format B (64, 69 시계열 텐서)
        feat_b = KinematicsExtractor2D.build_format_b_timeseries(kps_norm)

        return {
            'seq_id': seq_id,
            'feat_a': feat_a,
            'feat_b': feat_b,
            'label': label,
            'conditions': conds
        }
    except Exception:
        return None

def extract_and_save_dataset_features(dataset_name: str, file_pattern: str, num_workers: int = 12):
    """특정 데이터셋에 대해 멀티프로세싱 피처 일괄 추출 및 캐시 저장"""
    npz_files = glob.glob(file_pattern, recursive=True)
    print(f"\n[*] {dataset_name} 대상 전처리 파일: {len(npz_files):,}개")
    if not npz_files:
        return

    records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_npz_feature, p): p for p in npz_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset_name} 피처 추출"):
            res = future.result()
            if res is not None:
                records.append(res)

    print(f"[OK] {dataset_name} 피처 추출 완료: {len(records):,} / {len(npz_files):,}개 성공")

    # 배열 패킹
    seq_ids = [r['seq_id'] for r in records]
    feat_a_arr = np.stack([r['feat_a'] for r in records], axis=0) # (N, 97)
    feat_b_arr = np.stack([r['feat_b'] for r in records], axis=0) # (N, 64, 69)
    labels_arr = np.array([r['label'] for r in records], dtype=np.int32)
    conditions_list = [r['conditions'] for r in records]

    # Format A 저장
    out_a_path = os.path.join(FORMAT_A_DIR, f"{dataset_name}_features_A.npz")
    np.savez_compressed(
        out_a_path,
        features=feat_a_arr,
        labels=labels_arr,
        sequence_ids=seq_ids,
        conditions=np.array(conditions_list, dtype=object)
    )
    print(f"  -> Format A 저장 완료: {out_a_path} (Shape: {feat_a_arr.shape})")

    # Format B 저장
    out_b_path = os.path.join(FORMAT_B_DIR, f"{dataset_name}_features_B.npz")
    np.savez_compressed(
        out_b_path,
        features=feat_b_arr,
        labels=labels_arr,
        sequence_ids=seq_ids
    )
    print(f"  -> Format B 저장 완료: {out_b_path} (Shape: {feat_b_arr.shape})")

def main():
    start_time = time.time()
    os.makedirs(FORMAT_A_DIR, exist_ok=True)
    os.makedirs(FORMAT_B_DIR, exist_ok=True)

    print("=" * 60)
    print("[START] 2D 생체역학 97차원 피처 일괄 추출 및 캐시 빌드")
    print("=" * 60)

    num_workers = min(12, os.cpu_count() or 4)
    print(f"[*] 멀티프로세싱 가동: {num_workers}개 워커 프로세스")

    # 1. AI Hub 피처 추출
    extract_and_save_dataset_features(
        'AIHub',
        os.path.join(INPUT_PREPROCESSED_DIR, 'AIHub/**/*.npz'),
        num_workers=num_workers
    )

    # 2. UI-PRMD 피처 추출
    extract_and_save_dataset_features(
        'UIPRMD',
        os.path.join(INPUT_PREPROCESSED_DIR, 'UI-PRMD/**/*.npz'),
        num_workers=num_workers
    )

    # 3. REHAB24-6 피처 추출
    extract_and_save_dataset_features(
        'REHAB',
        os.path.join(INPUT_PREPROCESSED_DIR, 'REHAB24-6/**/*.npz'),
        num_workers=num_workers
    )

    # 4. Feature Names 메타데이터 JSON 저장
    sample_kps = np.zeros((64, 15, 2), dtype=np.float32)
    _, feat_names_97 = KinematicsExtractor2D.build_format_a_summary(sample_kps)
    names_json_path = os.path.join(FORMAT_A_DIR, 'feature_names_A.json')
    with open(names_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_features': len(feat_names_97),
            'feature_names': feat_names_97,
            'stream_summary': {
                'bone_stream_channels': 42,
                'joint_angles_channels': 8,
                'alignment_defects_channels': 6,
                'asymmetry_channels': 2,
                'velocity_channels': 8,
                'aihub_special_channels': 3,
                'format_b_total_channels': 69,
                'format_a_total_features': 97
            }
        }, f, ensure_ascii=False, indent=2)

    elapsed = round(time.time() - start_time, 2)
    print("=" * 60)
    print(f"[DONE] 전체 97차원 피처 데이터셋 빌드 완료! (총 소요시간: {elapsed}초)")
    print(f"[*] Format A 디렉터리: {FORMAT_A_DIR}")
    print(f"[*] Format B 디렉터리: {FORMAT_B_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
