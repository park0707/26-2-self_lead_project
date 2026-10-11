# =========================================================================================================
# [스크립트명]: build_benchmark_3d_features.py
# [위치]: src/features/build_benchmark_3d_features.py
# [목적]: 외부 벤치마크(UI-PRMD, REHAB24-6) 원본 3D 모션캡처/키넥트 데이터 파싱, 3D 기하 정규화 및
#         1단계 8종 운동 분류 모델(v4.0 2-Stage 캐스케이드) 전용 Format A (191D) / Format B (64, 98) 피처 빌더
# =========================================================================================================

import os
import sys
import glob
import csv
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

# 프로젝트 루트 경로 등록
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.core.skeleton2d import Joint15
from src.preprocessing.normalizer3d import GeometricNormalizer3D
from src.features.kinematics_3d import KinematicsExtractor3D

# 입출력 기본 경로 정의
UIPRMD_RAW_DIR = 'd:/Data/06_external_benchmarks/01_UI-PRMD'
REHAB_RAW_DIR = 'd:/Data/06_external_benchmarks/02_REHAB24-6'
OUTPUT_BENCHMARK_DIR = 'd:/Data/피처_데이터/External_Benchmarks_3D'

TARGET_DIM_A = 191
TARGET_DIM_B = 98

# 1. UI-PRMD Kinect 22 관절 계층 트리 (부모 관절 인덱스)
# 0: Waist(Root), 1: Spine, 2: Chest, 3: Neck, 4: Head, 5: HeadTip
# 6: L_Collar, 7: L_UpArm, 8: L_Forearm, 9: L_Hand
# 10: R_Collar, 11: R_UpArm, 12: R_Forearm, 13: R_Hand
# 14: L_UpLeg, 15: L_LowLeg, 16: L_Foot, 17: L_Toes
# 18: R_UpLeg, 19: R_LowLeg, 20: R_Foot, 21: R_Toes
UIPRMD_PARENTS = {
    0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    6: 2, 7: 6, 8: 7, 9: 8,
    10: 2, 11: 10, 12: 11, 13: 12,
    14: 0, 15: 14, 16: 15, 17: 16,
    18: 0, 19: 18, 20: 19, 21: 20
}

# UI-PRMD 22관절 -> 15개 핵심 생체역학 관절(Joint15) 매핑
UIPRMD_TO_15_MAP = {
    Joint15.PELVIS: 0,          # Waist
    Joint15.TORSO: 1,           # Spine
    Joint15.LEFT_SHOULDER: 7,   # L_UpArm
    Joint15.RIGHT_SHOULDER: 11, # R_UpArm
    Joint15.LEFT_ELBOW: 8,      # L_Forearm
    Joint15.RIGHT_ELBOW: 12,    # R_Forearm
    Joint15.LEFT_WRIST: 9,      # L_Hand
    Joint15.RIGHT_WRIST: 13,    # R_Hand
    Joint15.LEFT_HIP: 14,       # L_UpLeg
    Joint15.RIGHT_HIP: 18,      # R_UpLeg
    Joint15.LEFT_KNEE: 15,      # L_LowLeg
    Joint15.RIGHT_KNEE: 19,     # R_LowLeg
    Joint15.LEFT_ANKLE: 16,     # L_Foot
    Joint15.RIGHT_ANKLE: 20,    # R_Foot
    Joint15.NECK: 3             # Neck
}

# AI Hub 8대 운동 매핑 (Class 0~7)
# 0: 스쿼트, 1: 일반데드, 2: 스티프데드, 3: 바벨런지, 4: 다이나믹런지, 5: 사이드런지, 6: 크로스런지, 7: 굿모닝
UIPRMD_ACTION_MAP = {
    'm01': 0, # Deep Squat -> 01_바벨_스쿼트
    'm03': 4, # In-line Lunge -> 05_다이나믹_런지
    'm04': 5  # Side Lunge -> 06_사이드_런지
}

# 2. REHAB24-6 26개 관절 -> 15개 핵심 관절(Joint15) 매핑
REHAB_TO_15_MAP = {
    Joint15.PELVIS: 0,          # Hips
    Joint15.TORSO: 1,           # Spine
    Joint15.LEFT_SHOULDER: 6,   # LeftShoulder
    Joint15.RIGHT_SHOULDER: 11, # RightShoulder
    Joint15.LEFT_ELBOW: 7,      # LeftArm
    Joint15.RIGHT_ELBOW: 12,    # RightArm
    Joint15.LEFT_WRIST: 8,      # LeftForeArm
    Joint15.RIGHT_WRIST: 13,    # RightForeArm
    Joint15.LEFT_HIP: 16,       # LeftUpLeg
    Joint15.RIGHT_HIP: 21,      # RightUpLeg
    Joint15.LEFT_KNEE: 17,      # LeftLeg
    Joint15.RIGHT_KNEE: 22,     # RightLeg
    Joint15.LEFT_ANKLE: 18,     # LeftFoot
    Joint15.RIGHT_ANKLE: 23,    # RightFoot
    Joint15.NECK: 3             # Neck
}

REHAB_ACTION_MAP = {
    6: 0, # Ex6 (Squats) -> 01_바벨_스쿼트
    5: 4  # Ex5 (Leg Lunge) -> 05_다이나믹_런지
}

def pad_feature_a(feat_a: np.ndarray, target_dim: int = TARGET_DIM_A) -> np.ndarray:
    """Format A 161차원 피처를 모델 저장 규격(191D)으로 제로 패딩"""
    curr_len = len(feat_a)
    if curr_len < target_dim:
        return np.concatenate([feat_a, np.zeros(target_dim - curr_len, dtype=np.float32)], axis=-1)
    elif curr_len > target_dim:
        return feat_a[:target_dim]
    return feat_a.astype(np.float32)

def pad_feature_b(feat_b: np.ndarray, target_dim: int = TARGET_DIM_B) -> np.ndarray:
    """Format B (64, 92) 텐서를 모델 저장 규격(64, 98)으로 제로 패딩"""
    T, curr_dim = feat_b.shape
    if curr_dim < target_dim:
        pad = np.zeros((T, target_dim - curr_dim), dtype=np.float32)
        return np.concatenate([feat_b, pad], axis=-1)
    elif curr_dim > target_dim:
        return feat_b[:, :target_dim]
    return feat_b.astype(np.float32)

# =========================================================================================================
# UI-PRMD 단일 파일 워커 함수
# =========================================================================================================
def process_single_uiprmd_file(txt_path: str) -> Optional[Dict[str, Any]]:
    try:
        fname = os.path.basename(txt_path)
        is_correct = 1 if ('정상' in txt_path or 'Segmented' in txt_path) else 0

        # 운동 코드 추출 (m01~m10)
        movement_id = 'm00'
        for m_code in [f"m{i:02d}" for i in range(1, 11)]:
            if m_code in fname:
                movement_id = m_code
                break

        # 타깃 운동 매핑 (0: 스쿼트, 4: 다이나믹 런지, 5: 사이드 런지, 그 외: -1)
        y_action = UIPRMD_ACTION_MAP.get(movement_id, -1)
        is_target = 1 if y_action != -1 else 0

        # TXT 데이터 로드 (66D)
        data = []
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 66:
                    vals = [float(p.strip()) for p in parts[:66]]
                    data.append(vals)

        if not data or len(data) < 6:
            return None

        rel_3d = np.array(data, dtype=np.float32).reshape(len(data), 22, 3) # (T, 22, 3)
        T = rel_3d.shape[0]

        # 22개 관절 전방 운동학(Forward Kinematics) 누적 좌표 복원
        abs_pos = np.zeros_like(rel_3d)
        for i in range(22):
            p_idx = UIPRMD_PARENTS[i]
            if p_idx == -1:
                abs_pos[:, i, :] = rel_3d[:, i, :]
            else:
                abs_pos[:, i, :] = abs_pos[:, p_idx, :] + rel_3d[:, i, :]

        # 15개 핵심 관절 추출
        kps_15 = np.zeros((T, 15, 3), dtype=np.float32)
        for j15, ui_idx in UIPRMD_TO_15_MAP.items():
            kps_15[:, j15, :] = abs_pos[:, ui_idx, :]

        # 3D 기하 정규화 파이프라인 적용
        norm_res = GeometricNormalizer3D.process_sequence_3d(kps_15, exercise_name=movement_id)
        if norm_res is None:
            return None

        kps_norm_64 = norm_res['keypoints_3d_norm'] # (64, 15, 3)

        # 3D 생체역학 피처 추출
        feat_a_raw, _ = KinematicsExtractor3D.build_format_a_summary_3d(kps_norm_64)
        feat_b_raw = KinematicsExtractor3D.build_format_b_timeseries_3d(kps_norm_64)

        # 규격 패딩 적용
        feat_a = pad_feature_a(feat_a_raw, TARGET_DIM_A)
        feat_b = pad_feature_b(feat_b_raw, TARGET_DIM_B)

        seq_id = os.path.splitext(fname)[0]

        return {
            'seq_id': f"UIPRMD_{seq_id}",
            'movement_id': movement_id,
            'y_action': y_action,
            'y_binary': is_correct,
            'is_target': is_target,
            'feat_a': feat_a,
            'feat_b': feat_b
        }
    except Exception:
        return None

# =========================================================================================================
# REHAB24-6 단일 Repetition 워커 함수
# =========================================================================================================
def process_single_rehab_rep(item: Tuple[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    npy_path, rec = item
    try:
        vid = rec['video_id']
        ex_id = int(rec['exercise_id'])
        rep_num = int(rec['repetition_number'])
        person_id = int(rec['person_id'])
        correctness = int(rec['correctness'])
        s_frame = int(rec['first_frame'])
        e_frame = int(rec['last_frame'])
        mocap_err = int(rec.get('mocap_erroneous', 0))

        y_action = REHAB_ACTION_MAP.get(ex_id, -1)
        is_target = 1 if y_action != -1 else 0

        if not os.path.exists(npy_path):
            return None

        arr = np.load(npy_path) # (Total_Frames, 26, 4)
        if arr.ndim != 3 or arr.shape[1] != 26:
            return None

        total_t = arr.shape[0]
        s_idx = max(0, s_frame)
        e_idx = min(total_t, e_frame + 1)
        if s_idx >= e_idx or (e_idx - s_idx) < 6:
            return None

        rep_slice = arr[s_idx:e_idx, :, :3].astype(np.float32) # (T_rep, 26, 3)
        T_rep = rep_slice.shape[0]

        # 15개 핵심 관절 추출
        kps_15 = np.zeros((T_rep, 15, 3), dtype=np.float32)
        for j15, r_idx in REHAB_TO_15_MAP.items():
            kps_15[:, j15, :] = rep_slice[:, r_idx, :]

        # 3D 기하 정규화
        norm_res = GeometricNormalizer3D.process_sequence_3d(kps_15, exercise_name=f"Ex{ex_id}")
        if norm_res is None:
            return None

        kps_norm_64 = norm_res['keypoints_3d_norm'] # (64, 15, 3)

        feat_a_raw, _ = KinematicsExtractor3D.build_format_a_summary_3d(kps_norm_64)
        feat_b_raw = KinematicsExtractor3D.build_format_b_timeseries_3d(kps_norm_64)

        feat_a = pad_feature_a(feat_a_raw, TARGET_DIM_A)
        feat_b = pad_feature_b(feat_b_raw, TARGET_DIM_B)

        seq_id = f"REHAB_Ex{ex_id}_{vid}_rep{rep_num:02d}_p{person_id:02d}"

        return {
            'seq_id': seq_id,
            'exercise_id': f"Ex{ex_id}",
            'y_action': y_action,
            'y_binary': correctness,
            'is_target': is_target,
            'mocap_err': mocap_err,
            'feat_a': feat_a,
            'feat_b': feat_b
        }
    except Exception:
        return None

# =========================================================================================================
# 메인 실행 파이프라인
# =========================================================================================================
def build_all_external_benchmarks(max_workers: int = 8):
    os.makedirs(OUTPUT_BENCHMARK_DIR, exist_ok=True)
    t_start = time.time()

    print("\n" + "=" * 95)
    print("  ★ [NovelFlow] 외부 벤치마크 3D 데이터 정규화 및 v4.0 피처 일괄 가공 파이프라인")
    print(f"  ★ 대상: UI-PRMD (Kinect 3D 66D) & REHAB24-6 (OptiTrack 3D 26-Joints)")
    print(f"  ★ 출력 규격: Format A ({TARGET_DIM_A}D) / Format B (64, {TARGET_DIM_B})")
    print("=" * 95)

    # -------------------------------------------------------------
    # 1. UI-PRMD 처리
    # -------------------------------------------------------------
    print("\n[1/2] UI-PRMD Kinect 3D 파일 탐색 및 피처 추출 시작...")
    uiprmd_all_txts = glob.glob(os.path.join(UIPRMD_RAW_DIR, '**/*.txt'), recursive=True)
    uiprmd_pos_txts = [
        f for f in uiprmd_all_txts
        if ('positions' in f.lower() or '위치' in f) and ('kinect' in f.lower() or 'ŰƮ' in f or '키넥트' in f)
    ]
    print(f"  [*] 대상 파일 발견: {len(uiprmd_pos_txts):,}개")

    uiprmd_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_uiprmd_file, p): p for p in uiprmd_pos_txts}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="UI-PRMD 3D 추출"):
            res = fut.result()
            if res is not None:
                uiprmd_results.append(res)

    print(f"  [OK] UI-PRMD 성공: {len(uiprmd_results):,} / {len(uiprmd_pos_txts):,}개 시퀀스")

    # 배열 패킹 및 저장
    if uiprmd_results:
        u_sids = np.array([r['seq_id'] for r in uiprmd_results])
        u_mids = np.array([r['movement_id'] for r in uiprmd_results])
        u_y_act = np.array([r['y_action'] for r in uiprmd_results], dtype=np.int64)
        u_y_bin = np.array([r['y_binary'] for r in uiprmd_results], dtype=np.int32)
        u_target = np.array([r['is_target'] for r in uiprmd_results], dtype=np.int32)
        u_fa = np.stack([r['feat_a'] for r in uiprmd_results], axis=0) # (N, 191)
        u_fb = np.stack([r['feat_b'] for r in uiprmd_results], axis=0) # (N, 64, 98)

        u_out_path = os.path.join(OUTPUT_BENCHMARK_DIR, 'UIPRMD_3D_features.npz')
        np.savez_compressed(
            u_out_path,
            X_Format_A=u_fa,
            X_Format_B=u_fb,
            y_action=u_y_act,
            y_binary=u_y_bin,
            is_target=u_target,
            sequence_ids=u_sids,
            movement_ids=u_mids
        )
        print(f"  -> UI-PRMD 저장 완료: {u_out_path} (A: {u_fa.shape}, B: {u_fb.shape})")

    # -------------------------------------------------------------
    # 2. REHAB24-6 처리
    # -------------------------------------------------------------
    print("\n[2/2] REHAB24-6 OptiTrack 3D 세그멘테이션 파싱 및 피처 추출 시작...")
    seg_csv = os.path.join(REHAB_RAW_DIR, 'metadata', 'Segmentation.csv')
    rehab_tasks = []
    if os.path.exists(seg_csv):
        with open(seg_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                vid = row['video_id'].strip()
                ex_id = row['exercise_id'].strip()
                # 30fps 파일 탐색
                npy_path = os.path.join(REHAB_RAW_DIR, '3D_관절좌표_joints', f"Ex{ex_id}", f"{vid}-30fps.npy")
                if not os.path.exists(npy_path):
                    npy_path = os.path.join(REHAB_RAW_DIR, '3D_ǥ_joints', f"Ex{ex_id}", f"{vid}-30fps.npy")
                if not os.path.exists(npy_path):
                    # fallback glob search
                    matches = glob.glob(os.path.join(REHAB_RAW_DIR, '**', f"{vid}-30fps.npy"), recursive=True)
                    if matches:
                        npy_path = matches[0]

                if os.path.exists(npy_path):
                    rehab_tasks.append((npy_path, row))

    print(f"  [*] 대상 유효 Repetition 발견: {len(rehab_tasks):,}개")

    rehab_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_rehab_rep, t): t for t in rehab_tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="REHAB 3D 추출"):
            res = fut.result()
            if res is not None:
                rehab_results.append(res)

    print(f"  [OK] REHAB24-6 성공: {len(rehab_results):,} / {len(rehab_tasks):,}개 시퀀스")

    # 배열 패킹 및 저장
    if rehab_results:
        r_sids = np.array([r['seq_id'] for r in rehab_results])
        r_eids = np.array([r['exercise_id'] for r in rehab_results])
        r_y_act = np.array([r['y_action'] for r in rehab_results], dtype=np.int64)
        r_y_bin = np.array([r['y_binary'] for r in rehab_results], dtype=np.int32)
        r_target = np.array([r['is_target'] for r in rehab_results], dtype=np.int32)
        r_err = np.array([r['mocap_err'] for r in rehab_results], dtype=np.int32)
        r_fa = np.stack([r['feat_a'] for r in rehab_results], axis=0) # (N, 191)
        r_fb = np.stack([r['feat_b'] for r in rehab_results], axis=0) # (N, 64, 98)

        r_out_path = os.path.join(OUTPUT_BENCHMARK_DIR, 'REHAB_3D_features.npz')
        np.savez_compressed(
            r_out_path,
            X_Format_A=r_fa,
            X_Format_B=r_fb,
            y_action=r_y_act,
            y_binary=r_y_bin,
            is_target=r_target,
            mocap_erroneous=r_err,
            sequence_ids=r_sids,
            exercise_ids=r_eids
        )
        print(f"  -> REHAB24-6 저장 완료: {r_out_path} (A: {r_fa.shape}, B: {r_fb.shape})")

    t_total = time.time() - t_start
    print("\n" + "=" * 95)
    print(f"  ★ 외부 벤치마크 3D 피처 가공 완료! (총 소요 시간: {t_total:.1f}초)")
    print(f"  ★ 저장 디렉토리: {OUTPUT_BENCHMARK_DIR}")
    print("=" * 95)

if __name__ == '__main__':
    build_all_external_benchmarks(max_workers=8)
