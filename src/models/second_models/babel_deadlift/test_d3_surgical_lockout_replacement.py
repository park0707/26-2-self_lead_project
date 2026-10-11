# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_d3_surgical_lockout_replacement.py
Purpose: 피처 차원을 불필요하게 늘리지 않고(차원의 저주 방지),
         기존 17D 피처 내부의 stand_slice (0~T//8)를
         True-Lockout 적응형 윈도우(stand_idx)로 정밀 치환(In-place Surgical Replacement)하여
         RTMPose 5-Fold 피험자 격리 성능 검증
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = r"D:\src\models\second_models\babel_deadlift"
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d3_model_v5_3 import clean_and_smooth_keypoints, inpaint_lower_body_joints, KEY_JOINT_INDICES
from src.models.second_models.babel_deadlift.d3_model_v5_3_opt import OPTIMAL_17_INDICES

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

def compute_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    v1 = p1 - p2
    v2 = p3 - p2
    cos_ang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-7)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

def extract_d3_surgical_true_lockout_17d(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    기존 17D 정예 피처의 수학적 인터페이스는 완벽히 유지하되,
    'stand_slice'를 골반 최고점(True Lockout)으로 정밀 치환한 수식
    """
    kps = clean_and_smooth_keypoints(kps_33_64f)
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)

    hip_l, hip_r = kps[:, 23], kps[:, 24]
    knee_l, knee_r = kps[:, 25], kps[:, 26]
    ank_l, ank_r = kps[:, 27], kps[:, 28]
    heel_l, heel_r = kps[:, 29], kps[:, 30]
    toe_l, toe_r = kps[:, 31], kps[:, 32]
    wrist_l, wrist_r = kps[:, 15], kps[:, 16]

    hip_mid = (hip_l + hip_r) / 2.0
    knee_mid = (knee_l + knee_r) / 2.0
    ank_mid = (ank_l + ank_r) / 2.0
    heel_mid = (heel_l + heel_r) / 2.0
    toe_mid = (toe_l + toe_r) / 2.0
    wrist_mid = (wrist_l + wrist_r) / 2.0

    # 1. 골반 최고점(True Lockout) 탐색
    hip_y_s = gaussian_filter1d(hip_mid[:, 1], sigma=1.2)
    b = int(np.clip(np.argmax(hip_y_s), 4, T - 5)) # 최하사점
    stand_idx = int(np.argmin(hip_y_s))          # 진짜 기립 시점
    stand_slice = slice(max(0, stand_idx - 3), min(T, stand_idx + 4))

    rom = np.max(hip_y_s) - np.min(hip_y_s)
    if rom > 0.0004:
        active_mask = (hip_y_s - np.min(hip_y_s)) >= (rom * 0.15)
    else:
        active_mask = np.ones(T, dtype=bool)
    if np.sum(active_mask) < 6:
        active_mask = np.ones(T, dtype=bool)

    # 2. 발 피치 및 델타
    delta_l = toe_l[:, 1] - heel_l[:, 1]
    delta_r = toe_r[:, 1] - heel_r[:, 1]
    delta_mean = (delta_l + delta_r) / 2.0
    delta_s = gaussian_filter1d(delta_mean, sigma=1.0)

    vec_foot_l = toe_l - heel_l
    vec_foot_r = toe_r - heel_r
    pitch_l = np.degrees(np.arctan2(vec_foot_l[:, 1], np.hypot(vec_foot_l[:, 0], vec_foot_l[:, 2]) + 1e-7))
    pitch_r = np.degrees(np.arctan2(vec_foot_r[:, 1], np.hypot(vec_foot_r[:, 0], vec_foot_r[:, 2]) + 1e-7))
    pitch_mean = (pitch_l + pitch_r) / 2.0
    pitch_s = gaussian_filter1d(pitch_mean, sigma=1.0)

    # 진짜 기립 기준선
    delta_stand = float(np.median(delta_s[stand_slice]))
    pitch_stand = float(np.median(pitch_s[stand_slice]))

    # v1 계승 9D
    f1 = float(pitch_mean[b])
    f2 = float(pitch_mean[b] - pitch_stand) # 진짜 기립 기준 드롭
    f3 = float(np.max(pitch_mean))
    f4 = float(delta_mean[b])
    f5 = float(np.max(np.abs(pitch_l - pitch_r)))
    f6 = float(max(np.max(np.maximum(0, heel_l[:, 1] - toe_l[:, 1])), np.max(np.maximum(0, heel_r[:, 1] - toe_r[:, 1]))))
    roll_l = np.clip(np.degrees(np.arctan2(vec_foot_l[:, 0], vec_foot_l[:, 1] + 1e-7)), -35.0, 35.0)
    roll_r = np.clip(np.degrees(np.arctan2(vec_foot_r[:, 0], vec_foot_r[:, 1] + 1e-7)), -35.0, 35.0)
    f7 = float((abs(roll_l[b]) + abs(roll_r[b])) / 2.0)
    ang_ank_l = compute_angle_3d(knee_l, ank_l, toe_l)
    ang_ank_r = compute_angle_3d(knee_r, ank_r, toe_r)
    ang_ank = (ang_ank_l + ang_ank_r) / 2.0
    vec_shank = (ank_l + ank_r)/2.0 - (knee_l + knee_r)/2.0
    shank_pitch = np.degrees(np.arctan2(vec_shank[:, 2], vec_shank[:, 1] + 1e-7))
    f8 = float((np.max(ang_ank) - np.min(ang_ank)) / (np.max(shank_pitch) - np.min(shank_pitch) + 1e-6))
    f9 = float(np.std(delta_mean[b:]))
    v1_9d = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    # v1 20D 피처
    delta_nadir = float(delta_s[b])
    delta_drop = float(delta_nadir - delta_stand)
    delta_max_active = float(np.max(delta_s[active_mask]))
    pitch_nadir = float(pitch_s[b])
    pitch_drop = float(pitch_nadir - pitch_stand)
    pitch_max = float(np.max(pitch_s[active_mask]))
    asym_pitch = float(np.max(np.abs(pitch_l - pitch_r)[active_mask]))
    asym_delta = float(np.max(np.abs(delta_l - delta_r)[active_mask]))
    toe_lift_l = np.maximum(0.0, heel_l[:, 1] - toe_l[:, 1])
    toe_lift_r = np.maximum(0.0, heel_r[:, 1] - toe_r[:, 1])
    toe_lift_max = float(max(np.max(toe_lift_l[active_mask]), np.max(toe_lift_r[active_mask])))
    roll_nadir = float((abs(roll_l[b]) + abs(roll_r[b])) / 2.0)
    roll_max = float(max(np.max(np.abs(roll_l)[active_mask]), np.max(np.abs(roll_r)[active_mask])))
    ang_ank_s = gaussian_filter1d(ang_ank, sigma=1.0)
    ank_min = float(np.min(ang_ank_s[active_mask]))
    ank_stand = float(np.median(ang_ank_s[stand_slice])) # 진짜 기립 기준
    ank_drop = float(ank_stand - ank_min)
    norm_delta_nadir = float(delta_nadir / (float(np.linalg.norm(knee_mid - ank_mid, axis=-1).mean()) + 1e-5))

    v1_20d = [
        delta_nadir, delta_drop, delta_max_active,
        pitch_nadir, pitch_drop, pitch_max,
        asym_pitch, asym_delta, toe_lift_max,
        roll_nadir, roll_max, ank_min, ank_stand, ank_drop, norm_delta_nadir
    ]

    # 정강이 정규화 4D
    shank_l = np.linalg.norm(knee_l - ank_l, axis=-1)
    shank_r = np.linalg.norm(knee_r - ank_r, axis=-1)
    shank_len = float(np.maximum(0.1, np.median((shank_l + shank_r) / 2.0)))
    diff_mean = delta_mean
    base_diff = delta_stand # 진짜 기립 기준선
    dyn_lift = diff_mean - base_diff
    f_shank_max = float(np.max(dyn_lift) / shank_len)
    f_shank_mean = float(np.mean(np.maximum(0.0, dyn_lift)) / shank_len)
    f_shank_integ = float(np.sum(np.maximum(0.0, dyn_lift)) / (shank_len * 64.0))
    f_shank_asym = float(np.max(np.abs(delta_l - delta_r)) / shank_len)

    # Q3 6종
    delta_y_all = toe_mid[:, 1] - heel_mid[:, 1]
    horiz_dist = np.linalg.norm(toe_mid[:, [0, 2]] - heel_mid[:, [0, 2]], axis=-1) + 1e-6
    pitch_angle_deg = np.degrees(np.arctan2(delta_y_all, horiz_dist))
    f_pitch_max_q3 = float(np.max(pitch_angle_deg))
    nadir_idx_wrist = int(np.argmax(gaussian_filter1d(wrist_mid[:, 1], sigma=2.0)))
    lifting_slice = slice(max(0, nadir_idx_wrist - 5), min(64, nadir_idx_wrist + 30))
    f_pitch_lift_mean = float(np.mean(pitch_angle_deg[lifting_slice]))
    f_tibia_norm_lift_max = float(np.max(delta_y_all) / shank_len)
    v_wrist_y = -np.gradient(wrist_mid[:, 1], dt)
    v_heel_y = -np.gradient(heel_mid[:, 1], dt)
    f_vel_corr = float(np.corrcoef(v_wrist_y, v_heel_y)[0, 1]) if (np.std(v_wrist_y) > 1e-5 and np.std(v_heel_y) > 1e-5) else 0.0
    if np.isnan(f_vel_corr):
        f_vel_corr = 0.0
    f_always_lift_score = float(np.mean(pitch_angle_deg[:10]))
    f_inpaint_residual = 0.05

    # 32D 종합
    raw_32 = np.array([
        *v1_9d,
        *v1_20d[:15],
        f_shank_max, f_shank_mean,
        f_pitch_max_q3, f_pitch_lift_mean, f_tibia_norm_lift_max, f_vel_corr, f_always_lift_score, f_inpaint_residual
    ], dtype=np.float32)

    # OPTIMAL_17_INDICES 슬라이싱
    return np.nan_to_num(raw_32[OPTIMAL_17_INDICES], nan=0.0)

def main():
    meta = np.load(CACHE_META, allow_pickle=True)
    clip_names = meta['clip_names']
    subjects = meta['subjects']
    views = meta['views']
    y_golden = meta['y']
    N = len(clip_names)
    y_d3 = y_golden[:, 2]

    print("=" * 80)
    print("  [Surgical Test] True-Lockout 기준선 치환 17D 피처 5-Fold 평가")
    print("=" * 80)

    t0 = time.time()
    feats_list = []
    for cname in clip_names:
        rtm_p = os.path.join(RTM_DIR, cname + ".npz")
        data = np.load(rtm_p)
        kps_26 = data['kps_26_64f']
        kps_33 = halpe26_to_mediapipe33(kps_26)
        f17 = extract_d3_surgical_true_lockout_17d(kps_33)
        feats_list.append(f17)
    X = np.array(feats_list, dtype=np.float32)
    print(f"[+] 17D 피처 추출 완료: shape={X.shape} ({time.time()-t0:.2f}초)")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y_d3, groups=subjects))

    oof_probs = np.zeros(N, dtype=np.float32)
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        X_trn, y_trn = X[trn_idx], y_d3[trn_idx]
        X_val = X[val_idx]

        cb = CatBoostClassifier(
            iterations=160, depth=5, learning_rate=0.045,
            auto_class_weights='Balanced', random_seed=42 + fold_idx,
            thread_count=1, allow_writing_files=False, verbose=False
        )
        cb.fit(X_trn, y_trn)
        p_cb = cb.predict_proba(X_val)[:, 1]

        clf = lgb.LGBMClassifier(
            n_estimators=130, max_depth=5, learning_rate=0.045,
            num_leaves=24, class_weight='balanced',
            random_state=42 + fold_idx, verbose=-1, n_jobs=1
        )
        clf.fit(X_trn, y_trn)
        p_lgb = clf.predict_proba(X_val)[:, 1]

        oof_probs[val_idx] = 0.50 * p_cb + 0.50 * p_lgb

    auc = float(roc_auc_score(y_d3, oof_probs) * 100.0)
    print(f"[*] ROC-AUC: {auc:.2f}% (기존 79.03% 대비 {auc - 79.03:+.2f}%p)")

    # 시점별 3-State 최적화 (보류율 <= 15.0%)
    view_list = ['view5', 'view2', 'view3', 'view4', 'view1']
    tot_mask = np.zeros(N, dtype=bool)

    for v in view_list:
        v_idx = (views == v)
        y_v = y_d3[v_idx]
        p_v = oof_probs[v_idx]

        best_f1 = -1.0
        best_pair = (0.5, 0.5)
        best_res = None

        for tn in np.arange(0.20, 0.51, 0.01):
            for td in np.arange(0.50, 0.81, 0.01):
                if tn > td:
                    continue
                mask = (p_v <= tn) | (p_v >= td)
                cov = float(np.sum(mask) / len(y_v) * 100.0)
                retake = 100.0 - cov
                if retake <= 15.0 and np.sum(mask) > 0:
                    yt = y_v[mask]
                    yp = (p_v[mask] >= 0.50).astype(int)
                    f1 = float(f1_score(yt, yp, zero_division=0) * 100.0)
                    prec = float(precision_score(yt, yp, zero_division=0) * 100.0)
                    rec = float(recall_score(yt, yp, zero_division=0) * 100.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_pair = (round(float(tn), 3), round(float(td), 3))
                        best_res = {'f1': f1, 'prec': prec, 'rec': rec, 'cov': cov, 'retake': retake}

        tot_mask[v_idx] = (p_v <= best_pair[0]) | (p_v >= best_pair[1])
        print(f"  - {v}: pair={best_pair} | F1={best_res['f1']:.2f}% | Prec={best_res['prec']:.2f}% | Rec={best_res['rec']:.2f}% | Retake={best_res['retake']:.1f}%")

    yt_d = y_d3[tot_mask]
    yp_d = (oof_probs[tot_mask] >= 0.50).astype(int)
    comb_f1 = float(f1_score(yt_d, yp_d, zero_division=0) * 100.0)
    comb_prec = float(precision_score(yt_d, yp_d, zero_division=0) * 100.0)
    comb_rec = float(recall_score(yt_d, yp_d, zero_division=0) * 100.0)
    comb_cov = float(np.sum(tot_mask) / N * 100.0)
    comb_retake = 100.0 - comb_cov

    print(f"\n>> 원천 데이터 770건 종합 성능 <<")
    print(f"   3-State F1: {comb_f1:.2f}% | Precision: {comb_prec:.2f}% | Recall: {comb_rec:.2f}% | "
          f"AUC: {auc:.2f}% | 보류율: {comb_retake:.2f}%")

if __name__ == '__main__':
    main()
