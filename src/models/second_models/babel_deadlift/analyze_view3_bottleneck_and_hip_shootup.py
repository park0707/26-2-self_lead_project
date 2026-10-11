# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: analyze_view3_bottleneck_and_hip_shootup.py
Purpose: 시점 C(정면)에서 발뒤꿈치 2D 지표가 왜 무력화되는지,
         그리고 엉덩이 선행 거상(Hip Shoot-up) 및 정강이/체간 투영 피처가
         시점 C의 결함(D3)과 얼마나 강한 상관관계를 가지는지 정량 분석
"""

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, f1_score

CURRENT_DIR = r"D:\src\models\second_models\babel_deadlift"
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHE_META = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")
RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"

def halpe26_to_mediapipe33(kps_26: np.ndarray) -> np.ndarray:
    T = kps_26.shape[0]
    mp33 = np.zeros((T, 33, 3), dtype=np.float32)
    mapping = {
        0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
        5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
        20: 31, 21: 32, 24: 29, 25: 30
    }
    for h_idx, m_idx in mapping.items():
        mp33[:, m_idx] = kps_26[:, h_idx]
    return mp33

def main():
    meta = np.load(CACHE_META, allow_pickle=True)
    clip_names = meta['clip_names']
    subjects = meta['subjects']
    views = meta['views']
    y_golden = meta['y']
    N = len(clip_names)
    y_d3 = y_golden[:, 2]

    # 시점 C(view3) 인덱스
    v3_mask = (views == 'view3')
    v3_clips = clip_names[v3_mask]
    v3_y = y_d3[v3_mask]

    print(f"[*] 시점 C (정면) 총 {len(v3_clips)}건 (정상 {np.sum(v3_y==0)}건, 결함 {np.sum(v3_y==1)}건)")

    # 시점 C에서 후보 피처들 추출
    feat_names = [
        "heel_toe_y_delta_max",       # 기존: 2D 발가락-뒤꿈치 Y차
        "hip_shootup_velocity_ratio", # 신규 1: 골반 상승 속도 / 무릎 신전 속도
        "torso_2d_foreshortening",    # 신규 2: 초기 리프팅 구간 상체 2D 투영 길이 단축비
        "knee_extension_delay",       # 신규 3: 골반 상승 개시 대비 무릎 신전 시작 지연 프레임
        "shank_vertical_velocity"     # 신규 4: 정강이 Y축 수직 속도 RMS
    ]
    feats = []

    for cname in v3_clips:
        rtm_p = os.path.join(RTM_DIR, cname + ".npz")
        data = np.load(rtm_p)
        kps = halpe26_to_mediapipe33(data['kps_26_64f'])
        T = len(kps)
        dt = 1.0 / max(1, T - 1)

        shdr_y = (kps[:, 11, 1] + kps[:, 12, 1]) / 2.0
        hip_y = (kps[:, 23, 1] + kps[:, 24, 1]) / 2.0
        knee_y = (kps[:, 25, 1] + kps[:, 26, 1]) / 2.0
        ank_y = (kps[:, 27, 1] + kps[:, 28, 1]) / 2.0
        heel_y = (kps[:, 29, 1] + kps[:, 30, 1]) / 2.0
        toe_y = (kps[:, 31, 1] + kps[:, 32, 1]) / 2.0

        # 기존 발뒤꿈치 Y차
        f_heel_y = float(np.max(toe_y - heel_y))

        # 골반 및 무릎 수직 속도 (양수: 상승)
        v_hip = -np.gradient(gaussian_filter1d(hip_y, 1.5), dt)
        v_knee = -np.gradient(gaussian_filter1d(knee_y, 1.5), dt)

        # 리프팅 초기 구간 (골반이 올라가기 시작하는 구간)
        lift_start = int(np.argmax(v_hip > 0.05)) if np.any(v_hip > 0.05) else 15
        lift_window = slice(max(0, lift_start), min(T, lift_start + 25))

        # 1. 힙슈트업 속도비: 골반 상승 속도 / 무릎 신전 속도
        hip_speed = float(np.mean(np.maximum(0, v_hip[lift_window])))
        knee_speed = float(np.mean(np.maximum(0, v_knee[lift_window])) + 1e-4)
        f_shootup_ratio = hip_speed / knee_speed

        # 2. 상체 2D 투영 길이
        torso_len_2d = np.abs(hip_y - shdr_y)
        torso_init = float(np.mean(torso_len_2d[:5]))
        torso_min_lift = float(np.min(torso_len_2d[lift_window]))
        f_foreshortening = (torso_init - torso_min_lift) / (torso_init + 1e-4) # 숙여지면 커짐

        # 3. 무릎 신전 지연
        knee_start = int(np.argmax(v_knee > 0.05)) if np.any(v_knee > 0.05) else lift_start
        f_delay = float(knee_start - lift_start)

        # 4. 정강이 수직 속도
        v_shank = np.abs(v_knee - (-np.gradient(ank_y, dt)))
        f_shank_v = float(np.max(v_shank))

        feats.append([f_heel_y, f_shootup_ratio, f_foreshortening, f_delay, f_shank_v])

    feats = np.array(feats, dtype=np.float32)

    print("\n[시점 C (정면) 각 피처의 단독 ROC-AUC 변별력 검증]")
    for i, name in enumerate(feat_names):
        try:
            auc = roc_auc_score(v3_y, feats[:, i])
            if auc < 0.5:
                auc = 1.0 - auc # 역방향 피처
            print(f"  - {name:<30}: ROC-AUC = {auc*100:6.2f}%")
        except Exception as e:
            print(f"  - {name}: error ({e})")

if __name__ == '__main__':
    main()
