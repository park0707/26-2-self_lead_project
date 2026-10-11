# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: extract_squat_quality_features.py
Purpose: 해결책 4 (관측성 인지형 특성 - Visibility & Quality Gate Features)
         - MediaPipe Full-33 64F 전수 2,605개 시퀀스로부터 12대 관측성/품질 지표 추출
         - 대상: D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트
         - 출력: D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz
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
OUT_PATH = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"

def compute_12d_quality_features(kps_3d: np.ndarray) -> np.ndarray:
    """
    단일 시퀀스 (64, 33, 3) 3D 관절 궤적으로부터 12대 관측성/품질 피처 계산
    """
    # 랜드마크 인덱스:
    # 0: 코(Nose), 7: 좌측 귀, 8: 우측 귀
    # 11: 좌측 어깨, 12: 우측 어깨, 23: 좌측 골반, 24: 우측 골반
    # 25: 좌측 무릎, 26: 우측 무릎, 27: 좌측 발목, 28: 우측 발목
    sh_mid = (kps_3d[:, 11, :] + kps_3d[:, 12, :]) / 2.0
    hip_mid = (kps_3d[:, 23, :] + kps_3d[:, 24, :]) / 2.0

    # 1. 체간 길이 (Torso length)
    torso_len = np.linalg.norm(sh_mid - hip_mid, axis=1) # (64,)
    t_mean = float(np.mean(torso_len)) + 1e-6
    t_rel_std = float(np.std(torso_len) / t_mean)

    # 2. 대퇴부 길이 (Femur length)
    femur_l = np.linalg.norm(kps_3d[:, 23, :] - kps_3d[:, 25, :], axis=1)
    femur_r = np.linalg.norm(kps_3d[:, 24, :] - kps_3d[:, 26, :], axis=1)
    femur_rel_std = float((np.std(femur_l) / (np.mean(femur_l) + 1e-6) + np.std(femur_r) / (np.mean(femur_r) + 1e-6)) / 2.0)

    # 3. 정강이 길이 (Tibia length)
    tibia_l = np.linalg.norm(kps_3d[:, 25, :] - kps_3d[:, 27, :], axis=1)
    tibia_r = np.linalg.norm(kps_3d[:, 26, :] - kps_3d[:, 28, :], axis=1)
    tibia_rel_std = float((np.std(tibia_l) / (np.mean(tibia_l) + 1e-6) + np.std(tibia_r) / (np.mean(tibia_r) + 1e-6)) / 2.0)

    # 골격 불안정성 지표
    bone_instability = float((t_rel_std + femur_rel_std + tibia_rel_std) / 3.0)
    q_bone = float(np.exp(-10.0 * bone_instability))

    # 4. 가속도 지터 (2nd derivative of hip_mid)
    acc = np.diff(hip_mid, n=2, axis=0) # (62, 3)
    jitter = float(np.mean(np.linalg.norm(acc, axis=1)) / t_mean)
    q_jitter = float(np.exp(-20.0 * jitter))

    # 5. 종합 최소 품질 게이트
    q_overall = float(min(q_bone, q_jitter))

    # 6. 스쿼트 최저점 (Nadir) 정규 위치 (0~1)
    hip_y = hip_mid[:, 1] # 이미지 좌표계상 y는 아래로 증가
    nadir_idx = int(np.argmax(hip_y))
    nadir_norm = float(nadir_idx / 63.0)

    # 7. 스쿼트 수직 가동범위 (Range of Motion, ROM)
    rom_depth = float((np.max(hip_y) - np.min(hip_y)) / t_mean)

    # 8. 어깨 너비 / 체간 길이 비 (관상면 투영 원근도)
    sh_w = float(np.mean(np.linalg.norm(kps_3d[:, 11, :] - kps_3d[:, 12, :], axis=1)) / t_mean)

    # 9. 골반 너비 / 체간 길이 비
    hip_w = float(np.mean(np.linalg.norm(kps_3d[:, 23, :] - kps_3d[:, 24, :], axis=1)) / t_mean)

    # 10. 좌우 무릎 Z축 깊이차 (45도 사선 시점 감지 지표)
    z_knee_diff = float(np.mean(np.abs(kps_3d[:, 25, 2] - kps_3d[:, 26, 2])) / t_mean)

    # 11. 좌우 발목 Z축 깊이차
    z_ank_diff = float(np.mean(np.abs(kps_3d[:, 27, 2] - kps_3d[:, 28, 2])) / t_mean)

    # 12. 두경부 관측 가시성 비율 (코-귀 거리 / 어깨 너비)
    ear_mid = (kps_3d[:, 7, :] + kps_3d[:, 8, :]) / 2.0
    head_dist = np.linalg.norm(kps_3d[:, 0, :] - ear_mid, axis=1) # 코와 양 귀 중심 간 거리
    head_vis = float(np.mean(head_dist) / t_mean)

    return np.array([
        q_bone,             # [0] 골격 길이 불변성 품질 (0~1)
        q_jitter,           # [1] 가속도 평활도 품질 (0~1)
        q_overall,          # [2] 종합 최소 품질 (0~1)
        bone_instability,   # [3] 골격 불안정도 계수
        jitter,             # [4] 가속도 지터 변동량
        nadir_norm,         # [5] 스쿼트 최저점 프레임 위치 (0~1)
        rom_depth,          # [6] 수직 가동범위 (ROM / 체간길이)
        sh_w,               # [7] 어깨 너비 투영비
        hip_w,              # [8] 골반 너비 투영비
        z_knee_diff,        # [9] 좌우 무릎 Z차이 (사선 뷰 인지)
        z_ank_diff,         # [10] 좌우 발목 Z차이
        head_vis            # [11] 두경부 코-귀 관측 비율 (후방 가림 인지)
    ], dtype=np.float32)

def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 해결책 4 - 12대 관측성/품질 게이트 피처 전수 추출 시작")
    print("=" * 115)

    meta = np.load(META_PATH, allow_pickle=True)
    sequence_ids = meta['sequence_ids']
    N = len(sequence_ids)
    print(f"  타깃 메타 시퀀스 수: {N}건")

    feat_names = [
        'q_bone', 'q_jitter', 'q_overall', 'bone_instability', 'jitter',
        'nadir_norm', 'rom_depth', 'sh_width_ratio', 'hip_width_ratio',
        'z_knee_diff', 'z_ank_diff', 'head_vis_ratio'
    ]

    X_quality = np.zeros((N, 12), dtype=np.float32)
    t0 = time.time()

    for i, sid in enumerate(sequence_ids):
        npz_name = f"{sid}.npz"
        npz_path = os.path.join(DATA_DIR, npz_name)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing sequence file: {npz_path}")

        seq_data = np.load(npz_path)
        kps_3d = seq_data['kps_33_3d_64f'] # (64, 33, 3)
        X_quality[i] = compute_12d_quality_features(kps_3d)

        if (i + 1) % 500 == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            print(f"  진행률: [{i+1:4d} / {N}] ({((i+1)/N)*100:.1f}%) - 소요시간: {elapsed:.2f}초")

    # 통계 검증
    print("\n  [피처별 통계 요약 (전체 2,605건)]")
    for j, name in enumerate(feat_names):
        vals = X_quality[:, j]
        print(f"    {j:02d}. {name:<18s} | 평균: {np.mean(vals):.4f} | 표준편차: {np.std(vals):.4f} | 최소: {np.min(vals):.4f} | 최대: {np.max(vals):.4f}")

    np.savez_compressed(
        OUT_PATH,
        X_quality_12d=X_quality,
        feature_names=feat_names,
        sequence_ids=sequence_ids
    )
    print(f"\n  [저장 완료] -> {OUT_PATH}")
    print("=" * 115)

if __name__ == '__main__':
    main()
