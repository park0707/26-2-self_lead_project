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

from src.features.kinematics_2d import KinematicsExtractor2D

INPUT_PREPROCESSED_MP = 'd:/Data/전처리_데이터/MediaPipe'
INPUT_PREPROCESSED_AIHUB = 'd:/Data/전처리_데이터/AIHub'

OUTPUT_FEATURES_DIR = 'd:/Data/피처_데이터'
OUTPUT_MP_DIR = 'd:/Data/피처_데이터/MediaPipe'
OUTPUT_MEDIA_ROOT = 'd:/Data/MediaPipe/피처'

FORMAT_A_DIR = os.path.join(OUTPUT_FEATURES_DIR, 'Format_A_통계요약')
FORMAT_B_DIR = os.path.join(OUTPUT_FEATURES_DIR, 'Format_B_시계열텐서')

def find_all_npz_files(base_dir: str) -> Dict[str, str]:
    """디렉토리를 재귀 탐색하여 {seq_id: file_path} 매핑 딕셔너리 생성 (대괄호 이슈 완전 해결)"""
    file_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith('.npz'):
                seq_id = os.path.splitext(f)[0]
                file_map[seq_id] = os.path.join(root, f)
    return file_map

def process_single_mp_pair(args: Tuple[str, Optional[str]]) -> Optional[Dict[str, Any]]:
    """단일 MediaPipe .npz 파일 로드 및 Format A (97D), Format B (64, 69), Delta 피처 추출"""
    mp_npz_path, aihub_npz_path = args
    try:
        data = np.load(mp_npz_path, allow_pickle=True)
        kps_norm_mp = data['keypoints_2d_norm'] # (64, 15, 2)
        label = int(data['label'])
        conds = data['conditions']
        seq_id = os.path.splitext(os.path.basename(mp_npz_path))[0]
        meta = json.loads(str(data['metadata'])) if 'metadata' in data else {}
        exercise_name = meta.get('exercise_name', '')

        # 1. MediaPipe 단독 피처
        feat_a_mp, feat_names = KinematicsExtractor2D.build_format_a_summary(kps_norm_mp)
        feat_b_mp = KinematicsExtractor2D.build_format_b_timeseries(kps_norm_mp)

        # 2. AI Hub 라벨 파일 매칭 시 차이 피처 계산
        feat_a_aihub = None
        feat_b_aihub = None
        delta_a = None
        delta_b = None
        concat_a = None

        if aihub_npz_path and os.path.exists(aihub_npz_path):
            aihub_data = np.load(aihub_npz_path, allow_pickle=True)
            kps_norm_aihub = aihub_data['keypoints_2d_norm'] # (64, 15, 2)
            
            feat_a_aihub, _ = KinematicsExtractor2D.build_format_a_summary(kps_norm_aihub)
            feat_b_aihub = KinematicsExtractor2D.build_format_b_timeseries(kps_norm_aihub)
            
            # 소스 간 차이 피처 (Δ = MediaPipe - AIHub)
            delta_a = feat_a_mp - feat_a_aihub
            delta_b = feat_b_mp - feat_b_aihub
            concat_a = np.concatenate([feat_a_aihub, feat_a_mp], axis=-1) # (194,)

        return {
            'seq_id': seq_id,
            'exercise': exercise_name,
            'label': label,
            'conditions': conds,
            'feat_a_mp': feat_a_mp,
            'feat_b_mp': feat_b_mp,
            'feat_names': feat_names,
            'feat_a_aihub': feat_a_aihub,
            'feat_b_aihub': feat_b_aihub,
            'delta_a': delta_a,
            'delta_b': delta_b,
            'concat_a': concat_a
        }
    except Exception:
        return None

def build_all_mediapipe_features(num_workers: int = 8):
    """전체 MediaPipe 데이터셋에 대해 운동학 피처 및 차이 피처 일괄 추출 및 캐시 저장"""
    print("=" * 70)
    print(f"[*] MediaPipe 운동학 피처 추출 시작 (프로세스: {num_workers})")
    print("=" * 70)

    start_time = time.time()
    
    # 1. 파일 맵 구축
    mp_file_map = find_all_npz_files(INPUT_PREPROCESSED_MP)
    aihub_file_map = find_all_npz_files(INPUT_PREPROCESSED_AIHUB)
    print(f"[*] 발견된 MediaPipe 전처리 파일: {len(mp_file_map):,}개")
    print(f"[*] 발견된 AI Hub 전처리 파일: {len(aihub_file_map):,}개")

    if not mp_file_map:
        print("[!] 전처리된 MediaPipe 파일이 없습니다.")
        return

    tasks = []
    for seq_id, mp_path in mp_file_map.items():
        aihub_path = aihub_file_map.get(seq_id)
        tasks.append((mp_path, aihub_path))

    records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_mp_pair, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="MediaPipe 피처 추출"):
            res = future.result()
            if res is not None:
                records.append(res)

    if not records:
        print("[!] 피처 추출 실패 (기록 없음)")
        return

    records.sort(key=lambda x: x['seq_id'])

    # 1. MediaPipe 단독 피처 배열 구성
    X_a_mp = np.stack([r['feat_a_mp'] for r in records], axis=0) # (N, 97)
    X_b_mp = np.stack([r['feat_b_mp'] for r in records], axis=0) # (N, 64, 69)
    y = np.array([r['label'] for r in records], dtype=np.int32)
    seq_ids = [r['seq_id'] for r in records]
    feat_names = records[0]['feat_names']

    # 2. AI Hub 매칭 및 차이 피처 구성
    matched_records = [r for r in records if r['delta_a'] is not None]
    print(f"[*] AI Hub 라벨 매칭 완료된 시퀀스: {len(matched_records):,}개")

    # 저장 디렉토리 생성
    for d in [FORMAT_A_DIR, FORMAT_B_DIR, OUTPUT_MP_DIR, OUTPUT_MEDIA_ROOT]:
        os.makedirs(d, exist_ok=True)

    # 3. MediaPipe 단독 피처 저장
    mp_a_path = os.path.join(FORMAT_A_DIR, 'MediaPipe_Format_A.npz')
    mp_b_path = os.path.join(FORMAT_B_DIR, 'MediaPipe_Format_B.npz')
    
    np.savez_compressed(
        mp_a_path,
        X=X_a_mp,
        y=y,
        feature_names=feat_names,
        sequence_ids=seq_ids
    )
    np.savez_compressed(
        mp_b_path,
        X=X_b_mp,
        y=y,
        sequence_ids=seq_ids
    )

    # 4. MediaPipe 전용 디렉토리(Data/피처_데이터/MediaPipe/ & Data/MediaPipe/피처/)에도 저장
    for root_dir in [OUTPUT_MP_DIR, OUTPUT_MEDIA_ROOT]:
        np.savez_compressed(os.path.join(root_dir, 'MediaPipe_Only_Format_A.npz'), X=X_a_mp, y=y, feature_names=feat_names, sequence_ids=seq_ids)
        np.savez_compressed(os.path.join(root_dir, 'MediaPipe_Only_Format_B.npz'), X=X_b_mp, y=y, sequence_ids=seq_ids)

        if matched_records:
            X_delta_a = np.stack([r['delta_a'] for r in matched_records], axis=0) # (M, 97)
            X_delta_b = np.stack([r['delta_b'] for r in matched_records], axis=0) # (M, 64, 69)
            X_concat_a = np.stack([r['concat_a'] for r in matched_records], axis=0) # (M, 194)
            y_matched = np.array([r['label'] for r in matched_records], dtype=np.int32)
            seq_matched = [r['seq_id'] for r in matched_records]

            # Delta 피처 저장
            np.savez_compressed(os.path.join(root_dir, 'Delta_Features_Format_A.npz'), X=X_delta_a, y=y_matched, feature_names=feat_names, sequence_ids=seq_matched)
            np.savez_compressed(os.path.join(root_dir, 'Delta_Features_Format_B.npz'), X=X_delta_b, y=y_matched, sequence_ids=seq_matched)
            np.savez_compressed(os.path.join(root_dir, 'Concat_AIHub_MediaPipe_Format_A.npz'), X=X_concat_a, y=y_matched, sequence_ids=seq_matched)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[+] MediaPipe 피처 추출 및 캐시 저장 완료 (소요 시간: {elapsed:.1f}초)")
    print(f"    - 총 시퀀스 수: {X_a_mp.shape[0]:,}개")
    print(f"    - Format A Shape: {X_a_mp.shape} (97차원)")
    print(f"    - Format B Shape: {X_b_mp.shape} (64 frames, 69 dims)")
    if matched_records:
        print(f"    - Delta/Concat 피처 매칭 시퀀스: {len(matched_records):,}개 (194차원 결합)")
    print(f"    - 저장 경로 1: {OUTPUT_MP_DIR}")
    print(f"    - 저장 경로 2: {OUTPUT_MEDIA_ROOT}")
    print("=" * 70)

if __name__ == '__main__':
    build_all_mediapipe_features(num_workers=8)
