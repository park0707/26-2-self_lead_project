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

from src.adapters.mediapipe_adapter import MediaPipeAdapter
from src.preprocessing.normalizer2d import GeometricNormalizer2D

RAW_IMAGES_BASE = 'd:/Data/01_raw_data/images'
RAW_LABELS_BASE = 'd:/Data/01_raw_data/labels_json'
OUTPUT_BASE_DIR = 'd:/Data/전처리_데이터'
MEDIAPIPE_OUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'MediaPipe')
MEDIAPIPE_ROOT_DIR = 'd:/Data/MediaPipe/전처리'

_worker_detector = None

def _get_worker_detector():
    global _worker_detector
    if _worker_detector is None:
        _worker_detector = MediaPipeAdapter.get_detector()
    return _worker_detector

def process_single_image_seq(args: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
    """단일 시퀀스 이미지 디렉토리로부터 MediaPipe 추출, 정규화 및 .npz 저장"""
    seq_dir, exercise_dir_name, view_prefix = args
    try:
        det = _get_worker_detector()
        seq_id = os.path.basename(os.path.normpath(seq_dir))
        
        # 대응되는 AI Hub JSON 경로 탐색
        json_path = os.path.join(RAW_LABELS_BASE, exercise_dir_name, '2D', f"{seq_id}.json")
        if not os.path.exists(json_path):
            json_path = None

        seq = MediaPipeAdapter.parse_sequence_from_images(
            seq_dir,
            view_prefix=view_prefix,
            min_frames=4,
            json_path=json_path,
            detector=det
        )
        if seq is None:
            return None

        norm_res = GeometricNormalizer2D.process_sequence_2d(seq.keypoints_2d, seq.exercise_name)
        if norm_res is None:
            return None

        # 1. 전처리_데이터/MediaPipe/ 에 저장
        target_dir = os.path.join(MEDIAPIPE_OUT_DIR, exercise_dir_name)
        os.makedirs(target_dir, exist_ok=True)
        npz_filename = f"{seq.sequence_id}.npz"
        npz_path = os.path.join(target_dir, npz_filename)

        # 2. Data/MediaPipe/전처리/ 에도 동시 백업 저장
        root_target_dir = os.path.join(MEDIAPIPE_ROOT_DIR, exercise_dir_name)
        os.makedirs(root_target_dir, exist_ok=True)
        root_npz_path = os.path.join(root_target_dir, npz_filename)

        meta_dict = {
            'sequence_id': seq.sequence_id,
            'exercise_name': seq.exercise_name,
            'subject_id': seq.subject_id,
            'source_dataset': 'MediaPipe',
            'fps': seq.fps,
            'orig_frames': norm_res['orig_frames'],
            'rot_angle_deg': norm_res['rot_angle_deg'],
            'torso_scale': norm_res['torso_scale'],
            'is_flipped': norm_res['is_flipped'],
            'visibility_mean': seq.metadata.get('visibility_mean', 0.0),
            'view': view_prefix
        }

        save_kwargs = {
            'keypoints_2d_norm': norm_res['keypoints_2d_norm'],     # (64, 15, 2)
            'keypoints_2d_raw': seq.keypoints_2d,                   # (T_orig, 15, 2)
            'label': np.int32(seq.label),
            'conditions': np.array(seq.conditions or [], dtype=bool),
            'metadata': json.dumps(meta_dict, ensure_ascii=False)
        }

        np.savez_compressed(npz_path, **save_kwargs)
        np.savez_compressed(root_npz_path, **save_kwargs)

        return {
            'dataset': 'MediaPipe',
            'seq_id': seq.sequence_id,
            'exercise': seq.exercise_name,
            'subject': seq.subject_id,
            'label': seq.label,
            'npz_path': npz_path,
            'visibility': seq.metadata.get('visibility_mean', 0.0)
        }
    except Exception:
        return None

def build_all_mediapipe_dataset(view_prefix: str = "시점A", num_workers: int = 8):
    """전체 9개 운동의 모든 시퀀스에 대해 MediaPipe 일괄 추출 및 전처리 수행"""
    print("=" * 70)
    print(f"[*] MediaPipe 15-Joint 데이터셋 추출 및 전처리 시작 (시점: {view_prefix}, 프로세스: {num_workers})")
    print("=" * 70)

    start_time = time.time()
    
    # 작업 대상 시퀀스 목록 수집
    tasks = []
    exercise_folders = [f for f in os.listdir(RAW_IMAGES_BASE) if os.path.isdir(os.path.join(RAW_IMAGES_BASE, f))]
    exercise_folders.sort()

    for ex_folder in exercise_folders:
        ex_path = os.path.join(RAW_IMAGES_BASE, ex_folder)
        seq_dirs = [os.path.join(ex_path, s) for s in os.listdir(ex_path) if os.path.isdir(os.path.join(ex_path, s))]
        for s_dir in seq_dirs:
            tasks.append((s_dir, ex_folder, view_prefix))

    print(f"[*] 총 감지된 시퀀스 폴더: {len(tasks):,}개")

    success_records = []
    failed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image_seq, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="MediaPipe 전처리"):
            res = future.result()
            if res is not None:
                success_records.append(res)
            else:
                failed_count += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[+] MediaPipe 데이터셋 추출 완료 (소요 시간: {elapsed:.1f}초)")
    print(f"    - 성공 시퀀스: {len(success_records):,}개")
    print(f"    - 실패/스킵 시퀀스: {failed_count:,}개")
    print(f"    - 성공률: {(len(success_records) / max(1, len(tasks))) * 100:.1f}%")
    print("=" * 70)

    # 요약 메타데이터 저장
    summary_path = os.path.join(MEDIAPIPE_OUT_DIR, 'mediapipe_extraction_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_tasks': len(tasks),
            'success_count': len(success_records),
            'failed_count': failed_count,
            'elapsed_sec': elapsed,
            'view': view_prefix,
            'records': success_records
        }, f, indent=2, ensure_ascii=False)
    print(f"[+] 추출 요약 저장: {summary_path}")

if __name__ == '__main__':
    build_all_mediapipe_dataset(view_prefix="시점A", num_workers=8)
