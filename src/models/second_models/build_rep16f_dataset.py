# -*- coding: utf-8 -*-
"""
NovelFlow 16프레임 단일 횟수(Repetition) 피처 데이터셋 빌더 (Condition 5)
- 입력: d:/Data/전처리_데이터/MediaPipe_횟수분할/ (22,562개 분할 파일)
- 정규화: KinematicPipelineEngine (Soft-Slack IK + SE(2) 정렬) -> 16프레임 3차 스플라인 리샘플링
- 피처 추출:
  - Format A: KinematicsExtractor3D 161D 요약 통계 피처
  - Format B: KinematicsExtractor3D 92D x 16프레임 시계열 텐서
- 출력: d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz
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

sys.path.insert(0, 'd:/')

from src.features.kinematic_pipeline import KinematicPipelineEngine
from src.features.kinematics_3d import KinematicsExtractor3D
from src.preprocessing.normalizer3d import GeometricNormalizer3D
from src.models.first_model.dataset import FOLDER_TO_EX_IDX, EXERCISE_NAMES_KO

SRC_REP_DIR = 'd:/Data/전처리_데이터/MediaPipe_횟수분할'
OUT_COND5_DIR = 'd:/Data/피처_데이터/Condition5_Repetition_16F'
MASTER_NPZ_PATH = os.path.join(OUT_COND5_DIR, 'Master_Repetitions_16F.npz')

_WORKER_ENGINE = None

def _get_worker_engine():
    global _WORKER_ENGINE
    if _WORKER_ENGINE is None:
        _WORKER_ENGINE = KinematicPipelineEngine()
    return _WORKER_ENGINE

def process_single_rep_file(args: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    """단일 분할 npz 파일 로드 -> 16F 3D 정규화 -> Format A (161D) & Format B (16, 92) 추출"""
    filepath, ex_folder = args
    try:
        engine = _get_worker_engine()
        d = np.load(filepath, allow_pickle=True)
        
        k2 = d['keypoints_2d_raw'] # (T, 15, 2)
        if len(k2) < 2:
            return None

        # 3D 역기구학 및 SE(2) 중력 정렬
        k3_prior = np.pad(k2, ((0,0),(0,0),(0,1)))
        res = engine.process_sequence(k2, k3_prior)
        can_3d = res['canonical_3d'] # (T, 15, 3)

        # 16프레임 3차 스플라인 리샘플링
        can_16f = GeometricNormalizer3D.canonical_resample_time_3d(can_3d, target_frames=16)

        # Format A & Format B 추출
        feat_a, feat_names = KinematicsExtractor3D.build_format_a_summary_3d(can_16f)
        feat_b = KinematicsExtractor3D.build_format_b_timeseries_3d(can_16f)

        # 메타데이터 로드
        seq_id = str(d['sequence_id'])
        base_sid = str(d['base_sequence_id'])
        y_act = int(d['exercise_idx'])
        label = int(d['label'])
        subj_str = str(d['subject_id'])
        view_str = str(d['view'])
        rep_idx = int(d['rep_index'])
        dur = int(d['duration_frames'])

        # 피험자 번호 정수화 (1~9)
        subj_num = 1
        for digit in ['1','2','3','4','5','6','7','8','9']:
            if f"피험자{digit}" in subj_str or subj_str == digit or f"_{digit}_" in subj_str:
                subj_num = int(digit)
                break

        # 시점 명칭 정규화 (view1 ~ view5)
        view_map = {
            '시점A': 'view1', '시점B': 'view2', '시점C': 'view3', '시점D': 'view4', '시점E': 'view5',
            'view1': 'view1', 'view2': 'view2', 'view3': 'view3', 'view4': 'view4', 'view5': 'view5'
        }
        norm_view = view_map.get(view_str, 'view1')

        return {
            'feat_a': feat_a.astype(np.float32),
            'feat_b': feat_b.astype(np.float32),
            'sequence_id': seq_id,
            'base_sequence_id': base_sid,
            'y_action': y_act,
            'y_binary': label,
            'subject_num': subj_num,
            'view': norm_view,
            'rep_index': rep_idx,
            'duration': dur,
            'feature_names': feat_names
        }
    except Exception:
        return None

def build_condition5_dataset(num_workers: int = 8):
    print("=" * 90)
    print("  ★ [NovelFlow] Condition 5 (Repetition 16F 3D) 고속 병렬 피처 데이터셋 빌더")
    print(f"    - 입력 경로: {SRC_REP_DIR}")
    print(f"    - 출력 파일: {MASTER_NPZ_PATH}")
    print(f"    - 워커 수: {num_workers} processes")
    print("=" * 90)

    os.makedirs(OUT_COND5_DIR, exist_ok=True)

    # 1. 파일 목록 수집
    tasks = []
    exercise_folders = sorted([
        d for d in os.listdir(SRC_REP_DIR)
        if os.path.isdir(os.path.join(SRC_REP_DIR, d)) and not d.startswith('.')
    ])

    for ex_folder in exercise_folders:
        folder_path = os.path.join(SRC_REP_DIR, ex_folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
        for f in files:
            tasks.append((os.path.join(folder_path, f), ex_folder))

    total_tasks = len(tasks)
    print(f"[*] 총 {total_tasks:,}개 분할 파일 수집 완료. 16프레임 3D 정규화 및 피처 추출 시작...")

    start_time = time.time()
    all_feat_a = []
    all_feat_b = []
    all_sids = []
    all_base_sids = []
    all_y_act = []
    all_y_bin = []
    all_subjs = []
    all_views = []
    all_rep_idx = []
    all_durs = []
    feat_names = None

    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_rep_file, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=total_tasks, desc="Condition 5 16F 빌드"):
            res = fut.result()
            if res is not None:
                success_count += 1
                all_feat_a.append(res['feat_a'])
                all_feat_b.append(res['feat_b'])
                all_sids.append(res['sequence_id'])
                all_base_sids.append(res['base_sequence_id'])
                all_y_act.append(res['y_action'])
                all_y_bin.append(res['y_binary'])
                all_subjs.append(res['subject_num'])
                all_views.append(res['view'])
                all_rep_idx.append(res['rep_index'])
                all_durs.append(res['duration'])
                if feat_names is None:
                    feat_names = res['feature_names']

    elapsed = time.time() - start_time
    print(f"\n[*] 추출 완료: {success_count:,} / {total_tasks:,}개 성공 (소요: {elapsed:.1f}초, {elapsed/max(total_tasks,1)*1000:.1f} ms/샘플)")

    # 2. NumPy 배열 스택
    print("[*] 마스터 텐서 결합 및 압축 패키징 중...")
    X_a = np.stack(all_feat_a, axis=0) # (N, 161)
    X_b = np.stack(all_feat_b, axis=0) # (N, 16, 92)
    y_act = np.array(all_y_act, dtype=np.int64)
    y_bin = np.array(all_y_bin, dtype=np.int32)
    subjs = np.array(all_subjs, dtype=np.int32)
    views = np.array(all_views)
    sids = np.array(all_sids)
    base_sids = np.array(all_base_sids)

    # 3. 마스터 .npz 저장
    np.savez_compressed(
        MASTER_NPZ_PATH,
        X_Format_A=X_a,
        X_Format_B=X_b,
        y_action=y_act,
        y_binary=y_bin,
        subjects=subjs,
        views=views,
        sequence_ids=sids,
        base_sequence_ids=base_sids,
        rep_indices=np.array(all_rep_idx, dtype=np.int32),
        durations=np.array(all_durs, dtype=np.int32),
        feature_names=np.array(feat_names or []),
        exercise_names=np.array(EXERCISE_NAMES_KO)
    )

    file_size_mb = os.path.getsize(MASTER_NPZ_PATH) / (1024 * 1024)
    print("=" * 90)
    print("  ★ [Condition 5 마스터 데이터셋 구축 완료]")
    print(f"    - 저장 경로: {MASTER_NPZ_PATH} ({file_size_mb:.1f} MB)")
    print(f"    - X_Format_A 형상: {X_a.shape} (161D 요약 통계)")
    print(f"    - X_Format_B 형상: {X_b.shape} (16프레임 x 92채널 시계열)")
    print(f"    - 총 샘플 수: {len(y_act):,}개")
    print(f"    - 피험자 분포: {dict(zip(*np.unique(subjs, return_counts=True)))}")
    print(f"    - 시점 분포: {dict(zip(*np.unique(views, return_counts=True)))}")
    print("=" * 90)

if __name__ == '__main__':
    build_condition5_dataset()
