# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: extract_squat_kinetic_chain_features.py
Purpose: 해결책 5 (Kinetic Chain Part-Aware Representation & 3-Phase Alignment)
         - MediaPipe Full-33 64F 전수 2,605개 시퀀스로부터 9대 운동사슬 및 3단계 위상 피처 추출
         - 대상: D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트
         - 출력: D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz
"""

import os
import sys
import time
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
OUT_PATH = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    s_a = float(np.std(a))
    s_b = float(np.std(b))
    if s_a < 1e-6 or s_b < 1e-6:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if np.isnan(c) else c

def compute_kinetic_chain_features(kps: np.ndarray) -> np.ndarray:
    """
    단일 시퀀스 (64, 33, 3) 3D 관절 궤적으로부터 9대 운동사슬 및 3위상 특성 계산
    """
    sh_mid = (kps[:, 11, :] + kps[:, 12, :]) / 2.0
    hip_mid = (kps[:, 23, :] + kps[:, 24, :]) / 2.0
    knee_mid = (kps[:, 25, :] + kps[:, 26, :]) / 2.0
    ank_mid = (kps[:, 27, :] + kps[:, 28, :]) / 2.0
    ear_mid = (kps[:, 7, :] + kps[:, 8, :]) / 2.0

    torso_len = float(np.mean(np.linalg.norm(sh_mid - hip_mid, axis=1))) + 1e-6

    # 1. Nadir (최저점) anchor 프레임
    hip_y = hip_mid[:, 1]
    nadir_idx = int(np.argmax(hip_y))
    nadir_idx = int(np.clip(nadir_idx, 8, 55))

    # 3-Phase 인덱스 분할
    idx_descent = np.arange(0, max(1, nadir_idx - 4))
    idx_nadir = np.arange(max(0, nadir_idx - 4), min(64, nadir_idx + 5))
    idx_ascent = np.arange(min(63, nadir_idx + 5), 64)

    if len(idx_descent) == 0: idx_descent = np.arange(0, nadir_idx)
    if len(idx_ascent) == 0: idx_ascent = np.arange(nadir_idx, 64)

    # 2. 부위별 키네마틱스 궤적 (64프레임)
    # 체간 전경각 (Trunk Forward Tilt)
    trunk_vec = sh_mid - hip_mid
    trunk_tilt = np.arctan2(np.abs(trunk_vec[:, 2]), np.abs(trunk_vec[:, 1]) + 1e-6)

    # 무릎 외반 너비비 (Knee Valgus Ratio: 무릎 거리 / 골반 거리)
    knee_dist = np.linalg.norm(kps[:, 25, :2] - kps[:, 26, :2], axis=1)
    hip_dist = np.linalg.norm(kps[:, 23, :2] - kps[:, 24, :2], axis=1) + 1e-6
    valgus_ratio = knee_dist / hip_dist

    # 발목/발뒤꿈치 상대 수직 변위
    ank_y = ank_mid[:, 1]
    heel_lift = (np.min(ank_y) - ank_y) / torso_len

    # 두경부-경추 피치
    neck_vec = kps[:, 0, :] - ear_mid
    neck_pitch = neck_vec[:, 1] / torso_len

    # 3. 4대 부위 간 운동사슬 상관계수 (Cross-Part Coupling)
    corr_foot_knee = safe_corr(heel_lift, valgus_ratio)
    corr_knee_trunk = safe_corr(valgus_ratio, trunk_tilt)
    corr_trunk_neck = safe_corr(trunk_tilt, neck_pitch)
    corr_foot_trunk = safe_corr(heel_lift, trunk_tilt)

    # 4. 3단계 위상별 도메인 피처 (Phase-Specific Tokens)
    descent_valgus_delta = float(valgus_ratio[idx_descent[-1]] - valgus_ratio[idx_descent[0]])
    nadir_trunk_tilt_max = float(np.max(trunk_tilt[idx_nadir]))
    nadir_valgus_min = float(np.min(valgus_ratio[idx_nadir]))
    ascent_heel_lift_max = float(np.max(heel_lift[idx_ascent]) - np.min(heel_lift[idx_ascent]))

    # 5. 관절 간 도달 위상차 (Kinematic Phase Lag)
    knee_y = knee_mid[:, 1]
    knee_nadir_idx = int(np.argmax(knee_y))
    phase_lag = float((nadir_idx - knee_nadir_idx) / 63.0)

    return np.array([
        corr_foot_knee,         # [0] 발-무릎 운동사슬 커플링 상관계수 (-1~1)
        corr_knee_trunk,        # [1] 무릎-체간 운동사슬 커플링 상관계수 (-1~1)
        corr_trunk_neck,        # [2] 체간-경추 운동사슬 커플링 상관계수 (-1~1)
        corr_foot_trunk,        # [3] 발-체간 원격 운동사슬 커플링 상관계수 (-1~1)
        descent_valgus_delta,   # [4] 하강 구간 무릎 외반 변위량
        nadir_trunk_tilt_max,   # [5] 최저점 구간 최대 체간 숙임각 (rad)
        nadir_valgus_min,       # [6] 최저점 구간 최소 무릎 너비비 (최대 내측 모임)
        ascent_heel_lift_max,   # [7] 상승 구간 발뒤꿈치 최대 들림 폭
        phase_lag               # [8] 고관절-슬관절 최저점 도달 위상차 (Phase Lag)
    ], dtype=np.float32)

def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 해결책 5 - 9대 운동사슬(Kinetic Chain) 피처 전수 추출 시작")
    print("=" * 115)

    meta = np.load(META_PATH, allow_pickle=True)
    sequence_ids = meta['sequence_ids']
    N = len(sequence_ids)
    print(f"  타깃 메타 시퀀스 수: {N}건")

    feat_names = [
        'corr_foot_knee', 'corr_knee_trunk', 'corr_trunk_neck', 'corr_foot_trunk',
        'descent_valgus_delta', 'nadir_trunk_tilt_max', 'nadir_valgus_min',
        'ascent_heel_lift_max', 'phase_lag'
    ]

    X_kinetic = np.zeros((N, 9), dtype=np.float32)
    t0 = time.time()

    for i, sid in enumerate(sequence_ids):
        npz_name = f"{sid}.npz"
        npz_path = os.path.join(DATA_DIR, npz_name)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing sequence file: {npz_path}")

        seq_data = np.load(npz_path)
        kps_3d = seq_data['kps_33_3d_64f']
        X_kinetic[i] = compute_kinetic_chain_features(kps_3d)

        if (i + 1) % 500 == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            print(f"  진행률: [{i+1:4d} / {N}] ({((i+1)/N)*100:.1f}%) - 소요시간: {elapsed:.2f}초")

    print("\n  [피처별 통계 요약 (전체 2,605건)]")
    for j, name in enumerate(feat_names):
        vals = X_kinetic[:, j]
        print(f"    {j:02d}. {name:<22s} | 평균: {np.mean(vals):.4f} | 표준편차: {np.std(vals):.4f} | 최소: {np.min(vals):.4f} | 최대: {np.max(vals):.4f}")

    np.savez_compressed(
        OUT_PATH,
        X_kinetic_9d=X_kinetic,
        feature_names=feat_names,
        sequence_ids=sequence_ids
    )
    print(f"\n  [저장 완료] -> {OUT_PATH}")
    print("=" * 115)

if __name__ == '__main__':
    main()
