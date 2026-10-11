# =========================================================================================================
# [스크립트명]: build_video_benchmark_3d_features.py
# [위치]: src/features/build_video_benchmark_3d_features.py
# [목적]: 외부 RGB 비디오 벤치마크(RepCount, UCF101)로부터 MediaPipe 3D 월드 관절 좌표 궤적 추출,
#         3D 기하 정규화 및 1단계 8종 분류기 전용 Format A (191D) / Format B (64, 98) 피처 빌더
# =========================================================================================================

import os
import sys
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 프로젝트 루트 경로 등록
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

import mediapipe as mp
from src.adapters.mediapipe_adapter import MediaPipeAdapter
from src.core.skeleton2d import Joint15
from src.preprocessing.normalizer3d import GeometricNormalizer3D
from src.features.kinematics_3d import KinematicsExtractor3D

REPCOUNT_DIR = 'd:/Data/06_external_benchmarks/03_RepCount'
UCF101_DIR = 'd:/Data/06_external_benchmarks/04_UCF101'
OUTPUT_BENCHMARK_DIR = 'd:/Data/피처_데이터/External_Benchmarks_3D'

TARGET_DIM_A = 191
TARGET_DIM_B = 98

def pad_feature_a(feat_a: np.ndarray, target_dim: int = TARGET_DIM_A) -> np.ndarray:
    curr_dim = feat_a.shape[0]
    if curr_dim < target_dim:
        pad = np.zeros(target_dim - curr_dim, dtype=np.float32)
        return np.concatenate([feat_a, pad])
    elif curr_dim > target_dim:
        return feat_a[:target_dim]
    return feat_a.astype(np.float32)

def pad_feature_b(feat_b: np.ndarray, target_dim: int = TARGET_DIM_B) -> np.ndarray:
    T, curr_dim = feat_b.shape
    if curr_dim < target_dim:
        pad = np.zeros((T, target_dim - curr_dim), dtype=np.float32)
        return np.concatenate([feat_b, pad], axis=-1)
    elif curr_dim > target_dim:
        return feat_b[:, :target_dim]
    return feat_b.astype(np.float32)

def process_single_video(task_item: Tuple[str, int, str, str]) -> Optional[Dict[str, Any]]:
    video_path, y_action, ex_name, dataset_name = task_item
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 8:
            cap.release()
            return None

        # 64프레임 균등 샘플링 인덱스 생성
        num_samples = min(64, total_frames)
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        idx_set = set(indices)

        frames_dict = {}
        curr_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if curr_idx in idx_set:
                frames_dict[curr_idx] = frame
            curr_idx += 1
        cap.release()

        if len(frames_dict) < 6:
            return None

        # MediaPipe Pose Heavy 모델 로드
        det = MediaPipeAdapter.get_detector()

        kps_list = []
        for i in indices:
            if i not in frames_dict:
                continue
            frame = frames_dict[i]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = det.detect(mp_img)

            pts_15 = np.zeros((15, 3), dtype=np.float32)
            if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
                wlms = res.pose_world_landmarks[0]
                for mp_i, j15_i in MediaPipeAdapter.DIRECT_MP_TO_15_MAP.items():
                    pts_15[j15_i, 0] = wlms[mp_i].x
                    pts_15[j15_i, 1] = wlms[mp_i].y
                    pts_15[j15_i, 2] = wlms[mp_i].z

                pts_15[Joint15.PELVIS] = (pts_15[Joint15.LEFT_HIP] + pts_15[Joint15.RIGHT_HIP]) / 2.0
                pts_15[Joint15.NECK] = (pts_15[Joint15.LEFT_SHOULDER] + pts_15[Joint15.RIGHT_SHOULDER]) / 2.0
                pts_15[Joint15.TORSO] = (pts_15[Joint15.PELVIS] + pts_15[Joint15.NECK]) / 2.0

            kps_list.append(pts_15)

        kps_seq = np.array(kps_list, dtype=np.float32)

        # 3D 기하 정규화 파이프라인 적용
        norm_res = GeometricNormalizer3D.process_sequence_3d(kps_seq, exercise_name=ex_name)
        if norm_res is None:
            return None

        kps_norm_64 = norm_res['keypoints_3d_norm'] # (64, 15, 3)

        # 3D 생체역학 피처 추출
        feat_a_raw, _ = KinematicsExtractor3D.build_format_a_summary_3d(kps_norm_64)
        feat_b_raw = KinematicsExtractor3D.build_format_b_timeseries_3d(kps_norm_64)

        feat_a = pad_feature_a(feat_a_raw, TARGET_DIM_A)
        feat_b = pad_feature_b(feat_b_raw, TARGET_DIM_B)

        seq_id = os.path.splitext(os.path.basename(video_path))[0]

        return {
            'seq_id': f'{dataset_name}_{seq_id}',
            'dataset': dataset_name,
            'y_action': y_action,
            'feat_a': feat_a,
            'feat_b': feat_b
        }
    except Exception as e:
        return None

def build_all_video_benchmarks(max_workers: int = 6):
    print('=' * 95)
    print(' [NovelFlow] 외부 비디오 벤치마크 (RepCount, UCF101) MediaPipe 3D 피처 빌더 가동')
    print('=' * 95)
    t_start = time.time()
    os.makedirs(OUTPUT_BENCHMARK_DIR, exist_ok=True)

    # 1. RepCount Tasks
    out_rep = os.path.join(OUTPUT_BENCHMARK_DIR, 'RepCount_3D_features.npz')
    if os.path.exists(out_rep):
        print(f'\n[1/2] RepCount 이미 완료됨 -> 스킵 ({out_rep})')
    else:
        print('\n[1/2] RepCount 스쿼트 비디오 3D 피처 추출 준비...')
        rep_squat_dir = os.path.join(REPCOUNT_DIR, 'videos', 'squat')
        rep_tasks = []
        if os.path.exists(rep_squat_dir):
            for f in sorted(os.listdir(rep_squat_dir)):
                if f.endswith(('.mp4', '.avi')):
                    vp = os.path.join(rep_squat_dir, f)
                    rep_tasks.append((vp, 0, '01_바벨_스쿼트', 'RepCount'))
        print(f'  [*] RepCount 스쿼트 대상 비디오: {len(rep_tasks)}개')

        rep_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_video, t): t for t in rep_tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc='RepCount 3D 추출'):
                res = fut.result()
                if res is not None:
                    rep_results.append(res)

        print(f'  [OK] RepCount 성공: {len(rep_results)} / {len(rep_tasks)}개 영상')

        if rep_results:
            r_sids = np.array([r['seq_id'] for r in rep_results])
            r_y_act = np.array([r['y_action'] for r in rep_results], dtype=np.int64)
            r_fa = np.stack([r['feat_a'] for r in rep_results], axis=0)
            r_fb = np.stack([r['feat_b'] for r in rep_results], axis=0)

            np.savez_compressed(
                out_rep,
                X_Format_A=r_fa,
                X_Format_B=r_fb,
                y_action=r_y_act,
                sequence_ids=r_sids
            )
            print(f'  -> RepCount 저장 완료: {out_rep} (A: {r_fa.shape}, B: {r_fb.shape})')

    # 2. UCF101 Tasks
    print('\n[2/2] UCF101 스쿼트 및 런지 비디오 3D 피처 추출 준비...')
    ucf_squat_dir = os.path.join(UCF101_DIR, 'videos', 'BodyWeightSquats')
    ucf_lunge_dir = os.path.join(UCF101_DIR, 'videos', 'Lunges')
    ucf_tasks = []
    if os.path.exists(ucf_squat_dir):
        for f in sorted(os.listdir(ucf_squat_dir)):
            if f.endswith(('.mp4', '.avi')):
                vp = os.path.join(ucf_squat_dir, f)
                ucf_tasks.append((vp, 0, '01_바벨_스쿼트', 'UCF101'))
    if os.path.exists(ucf_lunge_dir):
        for f in sorted(os.listdir(ucf_lunge_dir)):
            if f.endswith(('.mp4', '.avi')):
                vp = os.path.join(ucf_lunge_dir, f)
                # 런지 매핑: 4번 다이나믹 런지 (또는 Macro Group Lunge Family)
                ucf_tasks.append((vp, 4, '05_다이나믹_런지', 'UCF101'))
    print(f'  [*] UCF101 대상 비디오: {len(ucf_tasks)}개 (스쿼트 {len(os.listdir(ucf_squat_dir))}, 런지 {len(os.listdir(ucf_lunge_dir))})')

    ucf_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, t): t for t in ucf_tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='UCF101 3D 추출'):
            res = fut.result()
            if res is not None:
                ucf_results.append(res)

    print(f'  [OK] UCF101 성공: {len(ucf_results)} / {len(ucf_tasks)}개 영상')

    if ucf_results:
        u_sids = np.array([r['seq_id'] for r in ucf_results])
        u_y_act = np.array([r['y_action'] for r in ucf_results], dtype=np.int64)
        u_fa = np.stack([r['feat_a'] for r in ucf_results], axis=0)
        u_fb = np.stack([r['feat_b'] for r in ucf_results], axis=0)

        out_ucf = os.path.join(OUTPUT_BENCHMARK_DIR, 'UCF101_3D_features.npz')
        np.savez_compressed(
            out_ucf,
            X_Format_A=u_fa,
            X_Format_B=u_fb,
            y_action=u_y_act,
            sequence_ids=u_sids
        )
        print(f'  -> UCF101 저장 완료: {out_ucf} (A: {u_fa.shape}, B: {u_fb.shape})')

    t_total = time.time() - t_start
    print('\n' + '=' * 95)
    print(f'  ★ 모든 비디오 벤치마크 3D 피처 추출 완료! (총 소요 시간: {t_total:.1f}초)')
    print('=' * 95)

if __name__ == '__main__':
    build_all_video_benchmarks(max_workers=6)
