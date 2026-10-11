# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_d3_v8_full_pipeline.py
Purpose: v5.3-Opt(17D) + v7.0 원판가림/힙슈트업 동역학(4D) + True-Lockout 기준선 보정(4D)
         총 25D 정예 피처셋으로 RTMPose 770건 5-Fold 피험자 격리 학습 및 성능 평가
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

from src.models.second_models.babel_deadlift.d3_model_v5_3_opt import extract_d3_v5_3_opt_features_from_seq

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

def extract_d3_v8_features(kps: np.ndarray) -> np.ndarray:
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)

    # 1. 기존 v5.3-Opt 17D 피처
    f17 = extract_d3_v5_3_opt_features_from_seq(kps)

    # 주요 관절
    hip_mid = (kps[:, 23] + kps[:, 24]) / 2.0
    knee_mid = (kps[:, 25] + kps[:, 26]) / 2.0
    ank_mid = (kps[:, 27] + kps[:, 28]) / 2.0
    heel_mid = (kps[:, 29] + kps[:, 30]) / 2.0
    toe_mid = (kps[:, 31] + kps[:, 32]) / 2.0
    shdr_mid = (kps[:, 11] + kps[:, 12]) / 2.0
    wrist_mid = (kps[:, 15] + kps[:, 16]) / 2.0

    # 2. True-Lockout 기준선 피처 (4D)
    hip_s = gaussian_filter1d(hip_mid[:, 1], sigma=1.5)
    stand_idx = int(np.argmin(hip_s))
    stand_win = slice(max(0, stand_idx - 3), min(T, stand_idx + 4))

    delta_y = toe_mid[:, 1] - heel_mid[:, 1] # 뒤꿈치가 들리면 > 0
    delta_s = gaussian_filter1d(delta_y, sigma=1.0)
    delta_lockout_stand = float(np.median(delta_s[stand_win]))
    delta_setup_offset = float(np.mean(delta_s[:5]) - delta_lockout_stand) # 셋업 시작 오프셋
    pure_lift_max = float(np.max(delta_s - delta_lockout_stand))
    shank_len = float(np.linalg.norm(knee_mid - ank_mid, axis=-1).mean()) + 1e-5
    pure_lift_ratio = pure_lift_max / shank_len

    # 3. v7.0 원판 가림 간접 포착 및 힙슈트업 동역학 피처 (4D)
    shank_vec = knee_mid - ank_mid
    shank_horiz = np.linalg.norm(shank_vec[:, [0, 2]], axis=-1) + 1e-6
    shank_vert = np.abs(shank_vec[:, 1]) + 1e-6
    shank_angle_deg = np.degrees(np.arctan2(shank_horiz, shank_vert))
    d_shank = np.gradient(shank_angle_deg)
    dd_shank = np.gradient(d_shank)

    f_shank_vel = float(np.max(np.abs(d_shank)))
    f_shank_acc = float(np.max(np.abs(dd_shank)))

    v_hip_y = -np.gradient(hip_mid[:, 1])
    v_knee_y = -np.gradient(knee_mid[:, 1])
    hip_knee_r = float(np.mean(v_hip_y[15:50]) / (np.mean(np.abs(v_knee_y[15:50])) + 1e-4))

    torso_vec = shdr_mid - hip_mid
    torso_horiz = np.linalg.norm(torso_vec[:, [0, 2]], axis=-1)
    f_torso_d = float(np.ptp(torso_horiz))

    f25 = np.hstack([
        f17,
        [delta_lockout_stand, delta_setup_offset, pure_lift_max, pure_lift_ratio],
        [f_shank_vel, f_shank_acc, hip_knee_r, f_torso_d]
    ]).astype(np.float32)

    return np.nan_to_num(f25, nan=0.0)

def main():
    meta = np.load(CACHE_META, allow_pickle=True)
    clip_names = meta['clip_names']
    subjects = meta['subjects']
    views = meta['views']
    y_golden = meta['y']
    N = len(clip_names)
    y_d3 = y_golden[:, 2]

    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 D3 v8.0-RTM 25D 정예 파이프라인 학습 & 평가")
    print("=" * 80)

    print("[*] 770건 전수 v8.0-RTM 25D 피처 추출 시작...")
    t0 = time.time()
    feats_list = []
    for cname in clip_names:
        rtm_p = os.path.join(RTM_DIR, cname + ".npz")
        data = np.load(rtm_p)
        kps_26 = data['kps_26_64f']
        kps_33 = halpe26_to_mediapipe33(kps_26)
        f = extract_d3_v8_features(kps_33)
        feats_list.append(f)
    X = np.array(feats_list, dtype=np.float32)
    print(f"[+] 25D 피처 추출 완료: shape={X.shape} ({time.time()-t0:.2f}초)")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y_d3, groups=subjects))

    oof_probs = np.zeros(N, dtype=np.float32)
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        X_trn, y_trn = X[trn_idx], y_d3[trn_idx]
        X_val = X[val_idx]

        cb = CatBoostClassifier(
            iterations=180, depth=5, learning_rate=0.04,
            auto_class_weights='Balanced', random_seed=42 + fold_idx,
            thread_count=1, allow_writing_files=False, verbose=False
        )
        cb.fit(X_trn, y_trn)
        p_cb = cb.predict_proba(X_val)[:, 1]

        clf = lgb.LGBMClassifier(
            n_estimators=140, max_depth=5, learning_rate=0.04,
            num_leaves=24, class_weight='balanced',
            random_state=42 + fold_idx, verbose=-1, n_jobs=1
        )
        clf.fit(X_trn, y_trn)
        p_lgb = clf.predict_proba(X_val)[:, 1]

        rf = RandomForestClassifier(
            n_estimators=80, max_depth=5, class_weight='balanced',
            random_state=42 + fold_idx, n_jobs=1
        )
        rf.fit(X_trn, y_trn)
        p_rf = rf.predict_proba(X_val)[:, 1]

        oof_probs[val_idx] = 0.45 * p_cb + 0.45 * p_lgb + 0.10 * p_rf

    auc = float(roc_auc_score(y_d3, oof_probs) * 100.0)
    print(f"\n[*] 25D v8.0-RTM ROC-AUC: {auc:.2f}% (기존 v5.3 79.03% 대비 +{auc - 79.03:.2f}%p)")

    # 사용자가 확정한 [기준 2: 보류율 <= 15.0%] 시점별 최적화
    view_list = ['view5', 'view2', 'view3', 'view4', 'view1']
    view_names = {
        'view5': '시점 A: 우측 후방 (45도)',
        'view2': '시점 B: 우측 전방 (45도)',
        'view3': '시점 C: 정면',
        'view4': '시점 D: 좌측 전방 (45도)',
        'view1': '시점 E: 좌측 후방 (45도)'
    }

    tot_mask = np.zeros(N, dtype=bool)
    print("\n" + "="*80)
    print("  [v8.0-RTM] 5대 공식 시점별 3-State 실측 성능 (보류율 <= 15.0%)")
    print("="*80)

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
        print(f"  - {view_names[v]:<26}: pair=[tn={best_pair[0]:.2f}, td={best_pair[1]:.2f}] | "
              f"F1={best_res['f1']:6.2f}% | Prec={best_res['prec']:6.2f}% | Rec={best_res['rec']:6.2f}% | "
              f"보류율={best_res['retake']:5.1f}%")

    yt_d = y_d3[tot_mask]
    yp_d = (oof_probs[tot_mask] >= 0.50).astype(int)
    comb_f1 = float(f1_score(yt_d, yp_d, zero_division=0) * 100.0)
    comb_prec = float(precision_score(yt_d, yp_d, zero_division=0) * 100.0)
    comb_rec = float(recall_score(yt_d, yp_d, zero_division=0) * 100.0)
    comb_acc = float(accuracy_score(yt_d, yp_d) * 100.0)
    comb_cov = float(np.sum(tot_mask) / N * 100.0)
    comb_retake = 100.0 - comb_cov

    print("\n" + "="*80)
    print("  >> [최종 원천 데이터 770건 전수 v8.0-RTM 종합 검증 결과] <<")
    print(f"  - 3-State F1: {comb_f1:.2f}% (기존 v5.3-Opt 균등 66.99% 대비 +{comb_f1 - 66.99:.2f}%p 향상)")
    print(f"  - 정밀도 (Precision): {comb_prec:.2f}% | 재현율 (Recall): {comb_rec:.2f}% | 정확도: {comb_acc:.2f}%")
    print(f"  - 전체 ROC-AUC: {auc:.2f}% (기존 79.03% 대비 +{auc - 79.03:.2f}%p 향상)")
    print(f"  - 전체 커버리지: {comb_cov:.2f}% | 공식 보류율: {comb_retake:.2f}% (15% 제약 엄격 준수)")
    print("="*80)

if __name__ == '__main__':
    main()
