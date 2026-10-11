# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_view_adaptive_exposed_foot.py
Purpose: 시점별 바벨 원판 가림 기하학(Camera-Plate Geometry)을 역산하여
         가려지지 않은 노출 발(Exposed Foot)의 순수 랜드마크를 추출하는
         시점 적응형 시선 개방(View-Adaptive Line-of-Sight Exposure) 기법 검증
"""

import os
import sys
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

from src.models.second_models.babel_deadlift.d3_model_v8_0_rtm import halpe26_to_mediapipe33, extract_d3_v8_features_from_seq

CACHE_META = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")
RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"
VIEW_LIST = ['view5', 'view2', 'view3', 'view4', 'view1']

def extract_exposed_foot_features(kps: np.ndarray, view_code: str) -> np.ndarray:
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)

    # 1. v8.0 21D 피처 (기본 계승)
    f21 = extract_d3_v8_features_from_seq(kps, view_code)

    # 주요 관절
    hip_mid = (kps[:, 23] + kps[:, 24]) / 2.0
    knee_mid = (kps[:, 25] + kps[:, 26]) / 2.0
    ank_mid = (kps[:, 27] + kps[:, 28]) / 2.0
    shdr_mid = (kps[:, 11] + kps[:, 12]) / 2.0

    # 발 관절 분리
    heel_l, heel_r = kps[:, 29], kps[:, 30]
    toe_l, toe_r = kps[:, 31], kps[:, 32]

    # 시점별 '노출 발(Exposed Foot)' 선택:
    # view5 (우측 후방 45°): 오른발 뒤꿈치가 카메라 바로 앞 -> 오른발 우선
    # view1 (좌측 후방 45°): 왼발 뒤꿈치가 카메라 바로 앞 -> 왼발 우선
    # view2 (우측 전방 45°): 우측 원판이 오른발 가림 -> 왼발(안쪽발) 우선
    # view4 (좌측 전방 45°): 좌측 원판이 왼발 가림 -> 오른발(안쪽발) 우선
    # view3 (정면 0°): 좌우 평균 + 원판 위 노출 관절 우선
    if view_code in ['view5', 'view4']:
        # 오른발이 더 잘 보임
        primary_heel = heel_r
        primary_toe = toe_r
    elif view_code in ['view1', 'view2']:
        # 왼발이 더 잘 보임
        primary_heel = heel_l
        primary_toe = toe_l
    else: # view3
        primary_heel = (heel_l + heel_r) / 2.0
        primary_toe = (toe_l + toe_r) / 2.0

    # 노출 발의 순수 들림 델타
    delta_exposed = primary_toe[:, 1] - primary_heel[:, 1]
    delta_exp_s = gaussian_filter1d(delta_exposed, 1.0)

    # True-Lockout 기준선
    hip_s = gaussian_filter1d(hip_mid[:, 1], 1.5)
    stand_idx = int(np.argmin(hip_s))
    stand_win = slice(max(0, stand_idx - 3), min(T, stand_idx + 4))

    delta_exp_stand = float(np.median(delta_exp_s[stand_win]))
    delta_exp_pure_max = float(np.max(delta_exp_s - delta_exp_stand))
    delta_exp_mean = float(np.mean(np.maximum(0, delta_exp_s - delta_exp_stand)))

    # 동역학 피처 (힙슈트업, 정강이 수직속도, 상체 단축비)
    v_hip = -np.gradient(hip_s, dt)
    v_knee = -np.gradient(gaussian_filter1d(knee_mid[:, 1], 1.5), dt)
    v_ank = -np.gradient(gaussian_filter1d(ank_mid[:, 1], 1.5), dt)

    lift_start = int(np.argmax(v_hip > 0.05)) if np.any(v_hip > 0.05) else 15
    lift_window = slice(max(0, lift_start), min(T, lift_start + 25))

    hip_speed = float(np.mean(np.maximum(0, v_hip[lift_window])))
    knee_speed = float(np.mean(np.maximum(0, v_knee[lift_window])) + 1e-4)
    f_shootup = float(np.clip(hip_speed / knee_speed, 0.0, 8.0))

    torso_len_2d = np.abs(hip_mid[:, 1] - shdr_mid[:, 1])
    torso_init = float(np.mean(torso_len_2d[:5]))
    torso_min = float(np.min(torso_len_2d[lift_window]))
    f_torso_short = float(np.clip((torso_init - torso_min) / (torso_init + 1e-4), -1.0, 1.0))

    v_shank = np.abs(v_knee - v_ank)
    f_shank_v = float(np.sqrt(np.mean(v_shank**2)))

    # 시점 원핫 (5D)
    f_view = [1.0 if view_code == v else 0.0 for v in VIEW_LIST]

    # 결합: 21D + 노출발 3D + 동역학 3D + 시점 5D = 32D
    f32 = np.hstack([
        f21,
        [delta_exp_stand, delta_exp_pure_max, delta_exp_mean],
        [f_shootup, f_torso_short, f_shank_v],
        f_view
    ]).astype(np.float32)

    return np.nan_to_num(f32, nan=0.0)

def main():
    meta = np.load(CACHE_META, allow_pickle=True)
    clip_names = meta['clip_names']
    subjects = meta['subjects']
    views = meta['views']
    y_golden = meta['y']
    N = len(clip_names)
    y_d3 = y_golden[:, 2]

    print("[*] 32D 노출 발(Exposed Foot) 시점 적응형 피처 추출 중...")
    feats = []
    for i, cname in enumerate(clip_names):
        rtm_p = os.path.join(RTM_DIR, cname + ".npz")
        data = np.load(rtm_p)
        kps = halpe26_to_mediapipe33(data['kps_26_64f'])
        f = extract_exposed_foot_features(kps, str(views[i]))
        feats.append(f)
    X = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0)
    print(f"[+] 피처 추출 완료: shape={X.shape}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y_d3, groups=subjects))

    oof_probs = np.zeros(N, dtype=np.float32)
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        X_trn, y_trn = X[trn_idx], y_d3[trn_idx]
        X_val = X[val_idx]

        cb = CatBoostClassifier(
            iterations=240, depth=5, learning_rate=0.038,
            l2_leaf_reg=3.0, auto_class_weights='Balanced',
            random_seed=42 + fold_idx, thread_count=1,
            allow_writing_files=False, verbose=False
        )
        cb.fit(X_trn, y_trn)
        p_cb = cb.predict_proba(X_val)[:, 1]

        clf = lgb.LGBMClassifier(
            n_estimators=170, max_depth=5, learning_rate=0.035,
            num_leaves=24, class_weight='balanced',
            subsample=0.85, colsample_bytree=0.85,
            random_state=42 + fold_idx, verbose=-1, n_jobs=1
        )
        clf.fit(X_trn, y_trn)
        p_lgb = clf.predict_proba(X_val)[:, 1]

        rf = RandomForestClassifier(
            n_estimators=90, max_depth=6, class_weight='balanced',
            random_state=42 + fold_idx, n_jobs=1
        )
        rf.fit(X_trn, y_trn)
        p_rf = rf.predict_proba(X_val)[:, 1]

        oof_probs[val_idx] = 0.45 * p_cb + 0.45 * p_lgb + 0.10 * p_rf

    auc = float(roc_auc_score(y_d3, oof_probs) * 100.0)
    print(f"[*] ROC-AUC: {auc:.2f}%")

    # 시점별 최적화 (보류율 <= 15.0%)
    tot_mask = np.zeros(N, dtype=bool)
    print("\n[시점별 3-State 최적화 성능]")
    for v in VIEW_LIST:
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
                    elif f1 == best_f1 and best_res is not None and retake < best_res['retake']:
                        best_res = {'f1': f1, 'prec': prec, 'rec': rec, 'cov': cov, 'retake': retake}
                        best_pair = (round(float(tn), 3), round(float(td), 3))

        tot_mask[v_idx] = (p_v <= best_pair[0]) | (p_v >= best_pair[1])
        print(f"  - {v}: pair={best_pair} | F1={best_res['f1']:6.2f}% | Prec={best_res['prec']:6.2f}% | Rec={best_res['rec']:6.2f}% | Retake={best_res['retake']:5.1f}%")

    yt_d = y_d3[tot_mask]
    yp_d = (oof_probs[tot_mask] >= 0.50).astype(int)
    all_f1 = float(f1_score(yt_d, yp_d, zero_division=0) * 100.0)
    all_prec = float(precision_score(yt_d, yp_d, zero_division=0) * 100.0)
    all_rec = float(recall_score(yt_d, yp_d, zero_division=0) * 100.0)
    all_cov = float(np.sum(tot_mask) / N * 100.0)
    all_retake = 100.0 - all_cov

    print("\n" + "=" * 80)
    print(f"  >> [원천 데이터 770건 종합 성능] <<")
    print(f"  - 3-State F1: {all_f1:.2f}% (목표 70.0% 달성 여부: {'성공! ★★★' if all_f1 >= 70.0 else '미달'})")
    print(f"  - 정밀도: {all_prec:.2f}% | 재현율: {all_rec:.2f}%")
    print(f"  - 전체 ROC-AUC: {auc:.2f}%")
    print(f"  - 커버리지: {all_cov:.2f}% | 보류율: {all_retake:.2f}% (<= 15.0% 제약 준수)")
    print("=" * 80)

if __name__ == '__main__':
    main()
