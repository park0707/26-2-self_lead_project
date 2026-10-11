# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: benchmark_v6_test.py
Purpose: c2 v6.0 생체역학 15D 피처셋 추출 및 5-Fold StratifiedGroupKFold 실측 평가
         - 체간 로컬 직교 좌표계(Body-Local Frame) 투영
         - Signed 시상면 굴곡/신전 & 관상면 측굴 분해
         - 후방 안면 피처 NaN 결측화 (Zero-imputation 배제)
         - 1) NaN-aware 통합 앙상블 모델 vs 2) 이원화 전문가 모델(Dual-Expert)
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb

RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"

FEATURE_NAMES_15_C2 = [
    "c2_neck_flexion_mean",
    "c2_neck_flexion_p90",
    "c2_neck_flexion_drop_at_nadir",
    "c2_neck_lateral_flexion_p90",
    "c2_neck_angular_velocity_peak",
    "c2_neck_extension_ratio_ascent",
    "c2_neck_ear_shoulder_roll_mismatch",
    "c2_head_pelvis_coordination_lag",
    "c2_face_pitch_drop_bottom",
    "c2_face_pitch_abs_dev_p90",
    "c2_face_yaw_asym_p90",
    "c2_face_heading_instability_std",
    "c2_face_nadir_abs_pitch",
    "c2_view_score_normal_z",
    "c2_face_valid_mask"
]


def extract_c2_v6_15d_features_from_seq(kps_33_64f: np.ndarray) -> np.ndarray:
    T = kps_33_64f.shape[0] # 64
    dt = 3.0 / max(1, T - 1)

    nose = kps_33_64f[:, 0, :]               # #0 Nose
    l_ear = kps_33_64f[:, 7, :]              # #7 Left Ear
    r_ear = kps_33_64f[:, 8, :]              # #8 Right Ear
    mid_ear = (l_ear + r_ear) / 2.0

    l_sh = kps_33_64f[:, 11, :]              # #11 Left Shoulder
    r_sh = kps_33_64f[:, 12, :]              # #12 Right Shoulder
    mid_sh = (l_sh + r_sh) / 2.0

    l_hip = kps_33_64f[:, 23, :]             # #23 Left Hip
    r_hip = kps_33_64f[:, 24, :]             # #24 Right Hip
    mid_hip = (l_hip + r_hip) / 2.0

    # 1. 체간 로컬 정규 직교 기저계 (Body-Local Orthonormal Basis)
    v_torso = mid_sh - mid_hip
    y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)

    v_sh = l_sh - r_sh
    z_raw = np.cross(y_hat, v_sh)
    z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)

    x_raw = np.cross(y_hat, z_hat)
    x_hat = x_raw / (np.linalg.norm(x_raw, axis=1, keepdims=True) + 1e-7)

    # 2. 시점 지표 및 안면 가시성 판정
    # z_hat[:, 2]: 카메라 Z축 성분 (전면/측면은 양수, 후방은 음수)
    normal_z_mean = float(np.mean(z_hat[:, 2]))
    is_face_visible = bool(normal_z_mean > 0.0)

    # 3. 스쿼트 위상 분석 (골반 Y좌표 기반 최하사점 b)
    pelvis_y = mid_hip[:, 1]
    pelvis_y_s = gaussian_filter1d(pelvis_y, sigma=1.2)
    b = int(np.clip(np.argmax(pelvis_y_s), 4, T - 5))
    standing_slice = max(1, T // 8)

    # 4. 체간 로컬 경추 벡터 분해 (시상면 굴곡/신전 & 관상면 측굴)
    v_neck = mid_ear - mid_sh
    v_neck_x = np.sum(v_neck * x_hat, axis=1)
    v_neck_y = np.sum(v_neck * y_hat, axis=1)
    v_neck_z = np.sum(v_neck * z_hat, axis=1)

    # 시상면 Signed 굴곡/신전각 (+ 굴곡, - 과신전)
    sagittal_raw = np.degrees(np.arctan2(v_neck_z, v_neck_y))
    sagittal_s = gaussian_filter1d(sagittal_raw, sigma=1.0)

    # 관상면 Signed 측굴각 (+ 좌측굴, - 우측굴)
    coronal_raw = np.degrees(np.arctan2(v_neck_x, v_neck_y))
    coronal_s = gaussian_filter1d(coronal_raw, sigma=1.0)

    # [f1] neck_flexion_mean
    f1_neck_flex_mean = float(np.mean(sagittal_s))

    # [f2] neck_flexion_p90
    f2_neck_flex_p90 = float(np.percentile(sagittal_s, 90))

    # [f3] neck_flexion_drop_at_nadir
    sagittal_base = float(np.median(sagittal_s[:standing_slice]))
    w_b_start = max(0, b - 3)
    w_b_end = min(T, b + 4)
    sagittal_nadir_win = float(np.median(sagittal_s[w_b_start:w_b_end]))
    f3_neck_flex_drop = float(max(0.0, sagittal_nadir_win - sagittal_base))

    # [f4] neck_lateral_flexion_p90
    f4_neck_lat_flex_p90 = float(np.percentile(np.abs(coronal_s), 90))

    # [f5] neck_angular_velocity_peak (|deg/s|)
    vel_sagittal = np.gradient(sagittal_s, dt)
    vel_win = vel_sagittal[max(0, b - 4):min(T, b + 5)]
    f5_neck_vel_peak = float(np.max(np.abs(vel_win)))

    # [f6] neck_extension_ratio_ascent
    ascent_slice = sagittal_s[b:]
    if len(ascent_slice) > 0:
        f6_neck_ext_ratio = float(np.sum(ascent_slice < (sagittal_base - 5.0)) / len(ascent_slice))
    else:
        f6_neck_ext_ratio = 0.0

    # [f7] neck_ear_shoulder_roll_mismatch
    v_ear_lr = l_ear - r_ear
    cos_roll = np.sum(v_ear_lr * v_sh, axis=1) / (np.linalg.norm(v_ear_lr, axis=1) * np.linalg.norm(v_sh, axis=1) + 1e-7)
    roll_mismatch = np.degrees(np.arccos(np.clip(cos_roll, -1.0, 1.0)))
    f7_roll_mismatch_p90 = float(np.percentile(roll_mismatch, 90))

    # [f8] head_pelvis_coordination_lag (구 VOR 지연)
    p_norm = (pelvis_y_s - np.mean(pelvis_y_s)) / (np.std(pelvis_y_s) + 1e-7)
    s_norm = (sagittal_s - np.mean(sagittal_s)) / (np.std(sagittal_s) + 1e-7)
    xcorr = correlate(s_norm, p_norm, mode='full')
    lags = np.arange(-(T - 1), T)
    f8_coord_lag = float(abs(lags[np.argmax(xcorr)]))

    # 5. 선택적 안면 지향 피처 (Face Heading 5D)
    if is_face_visible:
        v_head = nose - mid_ear
        head_horiz = np.linalg.norm(v_head[:, [0, 2]], axis=-1) + 1e-7
        gaze_pitch_raw = np.degrees(np.arctan2(v_head[:, 1], head_horiz))
        gaze_pitch_s = gaussian_filter1d(gaze_pitch_raw, sigma=1.0)
        gaze_stand = float(np.median(gaze_pitch_s[:standing_slice]))

        f9_face_pitch_drop = float(max(0.0, gaze_stand - gaze_pitch_s[b]))
        f10_face_pitch_dev_p90 = float(np.percentile(np.abs(gaze_pitch_s - gaze_stand), 90))

        dist_l = np.linalg.norm(nose - l_ear, axis=-1)
        dist_r = np.linalg.norm(nose - r_ear, axis=-1)
        yaw_asym = np.abs(dist_l - dist_r) / (dist_l + dist_r + 1e-6)
        f11_face_yaw_asym_p90 = float(np.percentile(yaw_asym, 90))

        f12_face_heading_std = float(np.std(gaze_pitch_s))
        f13_face_nadir_pitch = float(gaze_pitch_s[b])
    else:
        # 후방 결측치는 NaN으로 처리
        f9_face_pitch_drop = np.nan
        f10_face_pitch_dev_p90 = np.nan
        f11_face_yaw_asym_p90 = np.nan
        f12_face_heading_std = np.nan
        f13_face_nadir_pitch = np.nan

    # [f14] view_score_normal_z
    f14_view_score_z = normal_z_mean

    # [f15] face_valid_mask
    f15_face_valid_mask = 1.0 if is_face_visible else 0.0

    return np.array([
        f1_neck_flex_mean, f2_neck_flex_p90, f3_neck_flex_drop, f4_neck_lat_flex_p90,
        f5_neck_vel_peak, f6_neck_ext_ratio, f7_roll_mismatch_p90, f8_coord_lag,
        f9_face_pitch_drop, f10_face_pitch_dev_p90, f11_face_yaw_asym_p90,
        f12_face_heading_std, f13_face_nadir_pitch,
        f14_view_score_z, f15_face_valid_mask
    ], dtype=np.float32)


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c2(고개 정면) v6.0 생체역학 15D 피처 추출 및 5-Fold StratifiedGroupKFold 벤치마크", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    y_c2 = master['Y_defects_golden'][:, 1].astype(int)
    subjects = master['subjects']
    views = master['views']
    N = len(seq_ids)

    print(f"[*] 2,605개 시퀀스 v6 15D 피처 추출 시작...", flush=True)
    t0 = time.time()
    feats = []
    for i, sid in enumerate(seq_ids, 1):
        fp = os.path.join(RAW_DIR, f"{sid}.npz")
        kps = np.load(fp)['kps_33_3d_64f']
        f15 = extract_c2_v6_15d_features_from_seq(kps)
        feats.append(f15)
        if i % 500 == 0 or i == N:
            print(f"    - 진행률: {i:4d}/{N} ({i/N*100:5.1f}%) | 경과: {time.time()-t0:.1f}초", flush=True)

    X_v6 = np.array(feats, dtype=np.float32)
    print(f"[+] v6 15D 피처 추출 완료: 형상 {X_v6.shape} (소요 시간: {time.time()-t0:.1f}초)", flush=True)

    # 1. NaN-aware Unified 모델 평가 (CatBoost nan_mode='Min' + LightGBM native NaN)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_unified = np.zeros(N, dtype=np.float32)

    print("\n[*] 1. v6.0 NaN-aware 통합 앙상블 모델 5-Fold 학습 시작...", flush=True)
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_v6, y_c2, groups=subjects), 1):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
                                nan_mode='Min', random_seed=42 + fold, verbose=0)
        cb.fit(X_v6[trn_idx], y_c2[trn_idx])

        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31,
                                 random_state=42 + fold, verbose=-1, n_jobs=-1)
        clf.fit(X_v6[trn_idx], y_c2[trn_idx])

        p_cb = cb.predict_proba(X_v6[val_idx])[:, 1]
        p_lgb = clf.predict_proba(X_v6[val_idx])[:, 1]
        oof_unified[val_idx] = 0.60 * p_cb + 0.40 * p_lgb

    # 2. 이원화 전문가 모델 (Mixture of Experts) 평가
    # 전방 전문가: views in ['view2', 'view3', 'view4'] -> 15D 학습/추론
    # 후방 전문가: views in ['view1', 'view5'] -> 8D 순수 경추 피처 학습/추론
    print("\n[*] 2. v6.0 이원화 전문가 모델 (Dual-Expert: Front 15D + Rear 8D) 5-Fold 학습 시작...", flush=True)
    oof_expert = np.zeros(N, dtype=np.float32)

    is_front_all = np.isin(views, ['view2', 'view3', 'view4'])
    is_rear_all = np.isin(views, ['view1', 'view5'])

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_v6, y_c2, groups=subjects), 1):
        # Front Expert
        trn_front = trn_idx[is_front_all[trn_idx]]
        val_front = val_idx[is_front_all[val_idx]]

        cb_front = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
                                      random_seed=42 + fold, verbose=0)
        cb_front.fit(X_v6[trn_front], y_c2[trn_front])
        lgb_front = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31,
                                       random_state=42 + fold, verbose=-1, n_jobs=-1)
        lgb_front.fit(X_v6[trn_front], y_c2[trn_front])

        p_f_cb = cb_front.predict_proba(X_v6[val_front])[:, 1]
        p_f_lgb = lgb_front.predict_proba(X_v6[val_front])[:, 1]
        oof_expert[val_front] = 0.60 * p_f_cb + 0.40 * p_f_lgb

        # Rear Expert (공통 경추 8D만 사용: [:8])
        trn_rear = trn_idx[is_rear_all[trn_idx]]
        val_rear = val_idx[is_rear_all[val_idx]]

        cb_rear = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
                                     random_seed=42 + fold, verbose=0)
        cb_rear.fit(X_v6[trn_rear, :8], y_c2[trn_rear])
        lgb_rear = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31,
                                      random_state=42 + fold, verbose=-1, n_jobs=-1)
        lgb_rear.fit(X_v6[trn_rear, :8], y_c2[trn_rear])

        p_r_cb = cb_rear.predict_proba(X_v6[val_rear, :8])[:, 1]
        p_r_lgb = lgb_rear.predict_proba(X_v6[val_rear, :8])[:, 1]
        oof_expert[val_rear] = 0.60 * p_r_cb + 0.40 * p_r_lgb

    # v5.0 기존 성적표 (비교용)
    # v5 gated OOF는 D:\Data\피처_데이터\oof_master_64f_predictions_v2.npz 내의 c2 또는 74.45%
    def calc_metrics(oof):
        p_bin = (oof >= 0.50).astype(int)
        return {
            'f1': f1_score(y_c2, p_bin, zero_division=0) * 100.0,
            'prec': precision_score(y_c2, p_bin, zero_division=0) * 100.0,
            'rec': recall_score(y_c2, p_bin, zero_division=0) * 100.0,
            'auc': roc_auc_score(y_c2, oof) * 100.0,
            'acc': accuracy_score(y_c2, p_bin) * 100.0
        }

    m_uni = calc_metrics(oof_unified)
    m_exp = calc_metrics(oof_expert)

    print("\n" + "=" * 115, flush=True)
    print("  [실측 성적표 1] c2 v5.0 대비 v6.0 통합 NaN 모델 vs v6.0 이원화 전문가 모델", flush=True)
    print("=" * 115, flush=True)
    print(f"{'평가 지표':<20} | {'v5.0 (가드 9D Zero)':<22} | {'★ v6.0 NaN-통합 (15D)':<24} | {'★ v6.0 이원화 전문가':<22}", flush=True)
    print("-" * 115, flush=True)
    print(f"{'F1-Score':<20} | {74.45:>20.2f}% | {m_uni['f1']:>22.2f}% | {m_exp['f1']:>20.2f}%", flush=True)
    print(f"{'ROC-AUC':<20} | {78.23:>20.2f}% | {m_uni['auc']:>22.2f}% | {m_exp['auc']:>20.2f}%", flush=True)
    print(f"{'Precision (정밀도)':<20} | {72.06:>20.2f}% | {m_uni['prec']:>22.2f}% | {m_exp['prec']:>20.2f}%", flush=True)
    print(f"{'Recall (재현율)':<20} | {77.01:>20.2f}% | {m_uni['rec']:>22.2f}% | {m_exp['rec']:>20.2f}%", flush=True)
    print(f"{'Accuracy (정확도)':<20} | {72.21:>20.2f}% | {m_uni['acc']:>22.2f}% | {m_exp['acc']:>20.2f}%", flush=True)

    print("\n" + "=" * 115, flush=True)
    print("  [실측 성적표 2] 5대 시점(view1~view5)별 1:1 세부 F1 대조 감사", flush=True)
    print("=" * 115, flush=True)
    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_names = ['view1 (완전 후방)', 'view2 (우측 45도)', 'view3 (완전 정면)', 'view4 (좌측 45도)', 'view5 (우측 후방 45도)']
    v5_f1s = [69.15, 76.07, 74.69, 79.58, 72.85]

    p_uni_bin = (oof_unified >= 0.50).astype(int)
    p_exp_bin = (oof_expert >= 0.50).astype(int)

    print(f"{'시점 (카메라 앵글)':<25} | {'v5.0 F1':<12} | {'v6.0 NaN F1':<15} | {'v6.0 전문가 F1':<15} | {'특성'}", flush=True)
    print("-" * 115, flush=True)
    for v, vn, v5_f in zip(view_list, view_names, v5_f1s):
        mask = (views == v)
        f_uni = f1_score(y_c2[mask], p_uni_bin[mask], zero_division=0) * 100.0
        f_exp = f1_score(y_c2[mask], p_exp_bin[mask], zero_division=0) * 100.0
        feature_type = "순수 경추 8D 대리판정" if v in ['view1', 'view5'] else "경추 8D + 안면 5D 정밀"
        print(f"{vn:<25} | {v5_f:>10.2f}% | {f_uni:>13.2f}% | {f_exp:>13.2f}% | {feature_type}", flush=True)

    print("\n" + "=" * 115, flush=True)


if __name__ == '__main__':
    main()
