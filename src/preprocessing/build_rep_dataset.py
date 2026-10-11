# -*- coding: utf-8 -*-
"""
NovelFlow 전체 데이터셋 대상 횟수(Repetition) 단위 동적 분할 및 전처리 빌더
- 입력: d:/Data/전처리_데이터/MediaPipe/ (6,047개 시퀀스)
- 분할 엔진: RepetitionSegmenter (복합 키네마틱 에너지 + 트리거 가드 + 경계 공유)
- 출력: d:/Data/전처리_데이터/MediaPipe_횟수분할/
  ├── 01_바벨_스쿼트/
  │   ├── [오류]_D03_피험자1_Key314_rep01.npz
  │   ├── [오류]_D03_피험자1_Key314_rep02.npz
  │   └── ...
  ├── 02_바벨_데드리프트/
  └── ...
- 메타데이터 및 프레임 길이 통계 자동 산출
"""

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

# 프로젝트 루트 등록
sys.path.insert(0, 'd:/')

from src.preprocessing.rep_segmenter import RepetitionSegmenter
from src.models.first_model.dataset import FOLDER_TO_EX_IDX, EXERCISE_NAMES_KO

SRC_MP_DIR = 'd:/Data/전처리_데이터/MediaPipe'
OUT_REP_DIR = 'd:/Data/전처리_데이터/MediaPipe_횟수분할'
SUMMARY_JSON_PATH = os.path.join(OUT_REP_DIR, 'segmentation_summary.json')

def process_single_file(args: Tuple[str, str, str]) -> Dict[str, Any]:
    """단일 npz 파일 로드 -> 횟수별 분할 -> 서브 npz 저장"""
    src_path, ex_folder, out_folder = args
    try:
        d = np.load(src_path, allow_pickle=True)
        kps_raw = d['keypoints_2d_raw'] # (T, 15, 2)
        label = int(d['label']) if 'label' in d else 1
        conds = d['conditions'] if 'conditions' in d else np.array([], dtype=bool)
        
        meta = {}
        if 'metadata' in d:
            try:
                meta = json.loads(str(d['metadata']))
            except Exception:
                pass

        base_sid = os.path.splitext(os.path.basename(src_path))[0]
        ex_name = meta.get('exercise_name', ex_folder)
        subj_id = meta.get('subject_id', '피험자1')
        view = meta.get('view', '시점A')
        fps = meta.get('fps', 30.0)
        ex_idx = FOLDER_TO_EX_IDX.get(ex_folder, -1)

        # RepetitionSegmenter 동적 분할 실행
        segments = RepetitionSegmenter.segment_repetitions(kps_raw)

        # 3프레임 미만 노이즈 필터링
        valid_segs = []
        for s, e, b in segments:
            if (e - s + 1) >= 3:
                valid_segs.append((s, e, b))

        # 세그먼트가 없을 경우 전체를 단일 세그먼트로 보존
        if not valid_segs:
            valid_segs = [(0, len(kps_raw) - 1, len(kps_raw) // 2)]

        total_reps = len(valid_segs)
        saved_reps = []
        durations = []

        for rep_idx, (s, e, b) in enumerate(valid_segs, 1):
            sub_kps_raw = kps_raw[s : e + 1] # (T_sub, 15, 2)
            dur = int(e - s + 1)
            durations.append(dur)

            rep_sid = f"{base_sid}_rep{rep_idx:02d}"
            out_filename = f"{rep_sid}.npz"
            out_path = os.path.join(out_folder, out_filename)

            rep_meta = {
                'sequence_id': rep_sid,
                'base_sequence_id': base_sid,
                'exercise_name': ex_name,
                'exercise_folder': ex_folder,
                'exercise_idx': ex_idx,
                'subject_id': subj_id,
                'view': view,
                'fps': fps,
                'label': label,
                'rep_index': rep_idx,
                'total_reps': total_reps,
                'orig_start_frame': int(s),
                'orig_end_frame': int(e),
                'orig_bottom_frame': int(b),
                'duration_frames': dur
            }

            save_dict = {
                'keypoints_2d_raw': sub_kps_raw.astype(np.float32),
                'sequence_id': rep_sid,
                'base_sequence_id': base_sid,
                'exercise_name': ex_name,
                'exercise_idx': np.int64(ex_idx),
                'subject_id': subj_id,
                'view': view,
                'label': np.int32(label),
                'conditions': conds,
                'rep_index': np.int32(rep_idx),
                'total_reps': np.int32(total_reps),
                'orig_start_frame': np.int32(s),
                'orig_end_frame': np.int32(e),
                'orig_bottom_frame': np.int32(b),
                'duration_frames': np.int32(dur),
                'metadata': json.dumps(rep_meta, ensure_ascii=False)
            }

            np.savez_compressed(out_path, **save_dict)
            saved_reps.append(rep_sid)

        return {
            'status': 'success',
            'base_sid': base_sid,
            'exercise_folder': ex_folder,
            'total_reps': total_reps,
            'durations': durations,
            'saved_count': len(saved_reps)
        }
    except Exception as err:
        return {
            'status': 'error',
            'base_sid': os.path.basename(src_path),
            'exercise_folder': ex_folder,
            'error': str(err)
        }

def run_dataset_segmentation(num_workers: int = 8):
    print("=" * 90)
    print("  ★ [NovelFlow] 전체 MediaPipe 데이터셋 운동 횟수(Repetition) 고속 병렬 분할기")
    print(f"    - 원본 경로: {SRC_MP_DIR}")
    print(f"    - 출력 경로: {OUT_REP_DIR}")
    print(f"    - 병렬 워커: {num_workers} processes")
    print("=" * 90)

    os.makedirs(OUT_REP_DIR, exist_ok=True)

    # 작업 목록 생성
    tasks = []
    exercise_folders = sorted([
        d for d in os.listdir(SRC_MP_DIR)
        if os.path.isdir(os.path.join(SRC_MP_DIR, d)) and not d.startswith('.')
    ])

    for ex_folder in exercise_folders:
        src_ex_path = os.path.join(SRC_MP_DIR, ex_folder)
        out_ex_path = os.path.join(OUT_REP_DIR, ex_folder)
        os.makedirs(out_ex_path, exist_ok=True)

        files = [f for f in os.listdir(src_ex_path) if f.endswith('.npz')]
        for f in files:
            src_file_path = os.path.join(src_ex_path, f)
            tasks.append((src_file_path, ex_folder, out_ex_path))

    total_tasks = len(tasks)
    print(f"[*] 총 {len(exercise_folders)}개 운동 폴더에서 {total_tasks:,}개 시퀀스 작업 등록 완료.")

    # 병렬 실행
    start_time = time.time()
    results = []
    success_count = 0
    error_count = 0

    all_rep_counts = []
    all_durations = []
    ex_stats = {ex: {'files': 0, 'reps': 0, 'durations': []} for ex in exercise_folders}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_file, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=total_tasks, desc="횟수 동적 분할 진행률"):
            res = fut.result()
            results.append(res)
            if res['status'] == 'success':
                success_count += 1
                r_cnt = res['total_reps']
                durs = res['durations']
                all_rep_counts.append(r_cnt)
                all_durations.extend(durs)

                ef = res['exercise_folder']
                if ef in ex_stats:
                    ex_stats[ef]['files'] += 1
                    ex_stats[ef]['reps'] += r_cnt
                    ex_stats[ef]['durations'].extend(durs)
            else:
                error_count += 1

    elapsed = time.time() - start_time
    total_reps_generated = len(all_durations)

    print("\n" + "=" * 90)
    print("  ★ [분할 완료 보고서]")
    print(f"    - 처리 완료 시퀀스: {success_count:,} / {total_tasks:,} (에러: {error_count}개)")
    print(f"    - 총 생성된 횟수(Repetition) 서브 샘플: {total_reps_generated:,}개 (약 {total_reps_generated/max(success_count,1):.2f}배 증강)")
    print(f"    - 총 소요 시간: {elapsed:.1f}초 ({elapsed/max(total_tasks,1)*1000:.1f} ms/시퀀스)")
    print("-" * 90)

    # 전체 통계치 계산
    rep_arr = np.array(all_rep_counts) if all_rep_counts else np.array([0])
    dur_arr = np.array(all_durations) if all_durations else np.array([0])

    summary_data = {
        'total_source_sequences': total_tasks,
        'success_sequences': success_count,
        'error_sequences': error_count,
        'total_generated_repetitions': total_reps_generated,
        'elapsed_seconds': round(elapsed, 2),
        'reps_per_sequence_stats': {
            'mean': float(np.mean(rep_arr)),
            'median': float(np.median(rep_arr)),
            'std': float(np.std(rep_arr)),
            'min': int(np.min(rep_arr)),
            'max': int(np.max(rep_arr))
        },
        'duration_frames_stats': {
            'mean': float(np.mean(dur_arr)),
            'median': float(np.median(dur_arr)),
            'std': float(np.std(dur_arr)),
            'min': int(np.min(dur_arr)),
            'max': int(np.max(dur_arr)),
            'p25': float(np.percentile(dur_arr, 25)),
            'p75': float(np.percentile(dur_arr, 75)),
            'p90': float(np.percentile(dur_arr, 90)),
            'p95': float(np.percentile(dur_arr, 95))
        },
        'exercise_breakdown': {}
    }

    print("  [1] 시퀀스당 분할 횟수(Reps per Sequence) 통계:")
    print(f"      평균: {summary_data['reps_per_sequence_stats']['mean']:.2f}회 | 중앙값: {summary_data['reps_per_sequence_stats']['median']:.1f}회 | 범위: {summary_data['reps_per_sequence_stats']['min']} ~ {summary_data['reps_per_sequence_stats']['max']}회")
    print("\n  [2] 단일 횟수(Repetition) 프레임 길이(Duration Frames) 통계:")
    print(f"      평균: {summary_data['duration_frames_stats']['mean']:.2f} 프레임 | 중앙값: {summary_data['duration_frames_stats']['median']:.1f} 프레임 | 표준편차: {summary_data['duration_frames_stats']['std']:.2f}")
    print(f"      25%: {summary_data['duration_frames_stats']['p25']:.1f}F | 75%: {summary_data['duration_frames_stats']['p75']:.1f}F | 90%: {summary_data['duration_frames_stats']['p90']:.1f}F | 95%: {summary_data['duration_frames_stats']['p95']:.1f}F")
    print(f"      최소: {summary_data['duration_frames_stats']['min']} 프레임 | 최대: {summary_data['duration_frames_stats']['max']} 프레임")
    print("-" * 90)

    print("  [3] 운동 종목별 세부 분할 현황:")
    for ef, stat in sorted(ex_stats.items()):
        f_cnt = stat['files']
        r_cnt = stat['reps']
        d_list = stat['durations']
        avg_r = r_cnt / max(f_cnt, 1)
        avg_d = np.mean(d_list) if d_list else 0.0
        med_d = np.median(d_list) if d_list else 0.0

        summary_data['exercise_breakdown'][ef] = {
            'sequences': f_cnt,
            'repetitions': r_cnt,
            'avg_reps': round(avg_r, 2),
            'avg_duration_frames': round(float(avg_d), 2),
            'median_duration_frames': round(float(med_d), 2)
        }
        print(f"      - {ef:<26}: {f_cnt:4d}개 시퀀스 -> {r_cnt:5d}개 횟수 (평균 {avg_r:.2f}회/시퀀스, 평균 길이 {avg_d:.1f}F)")

    # JSON 저장
    with open(SUMMARY_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n  [★] 분할 요약 메타데이터 저장 완료: " + SUMMARY_JSON_PATH)
    print("=" * 90)

if __name__ == '__main__':
    run_dataset_segmentation()
