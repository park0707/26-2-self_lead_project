# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_d3_v9_prototype.py
Purpose: v9.0-RTM 30D 시점 적응형 생체역학 피처셋
         (v8.0 21D + 힙슈트업/정강이 동역학 4D + 시점 원핫 5D)
         770건 전수 5-Fold 피험자 격리 학습 및 보류율 <= 15% 제약 하 F1 70%+ 돌파 검증
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

from src.models.second_models.babel_deadlift.d3_model_v8_0_rtm import extract_d3_v8_features_from_seq, halpe26_to_mediapipe33

CACHE_META = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")
RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"

VIEW_LIST = ['view5', 'view2', 'view3', 'view4', 'view1']
VIEW_MAP = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)'
}

def extract_d3_v9_features(kps: np.ndarray, view_code: str = 'view3') -> np.ndarray:
    T = kps.shape[0]
    dt = 1.0 / max(1, T - 1)

    # 1. v8.0 21D 피처 (True-Lockout 포함)
    f21 = extract_d3_v8_features_from_seq(kps, view_code)

    # 관절 좌표
    shdr_y = (kps[:, 11, 1] + kps[:, 12, 1]) / 2.0
    hip_y = (kps[:, 23, 1] + kps[:, 24, 1]) / 2.0
    knee_y = (kps[:, 25, 1] + kps[:, 26, 1]) / 2.0
    ank_y = (kps[:, 27, 1] + kps[:, 28, 1]) / 2.0

    # 속도 계산 (양수: 상승)
    v_hip = -np.gradient(gaussian_filter1d(hip_y, 1.5), dt)
    v_knee = -np.gradient(gaussian_filter1d(knee_y, 1.5), dt)
    v_ank = -np.gradient(gaussian_filter1d(ank_y, 1.5), dt)

    lift_start = int(np.argmax(v_hip > 0.05)) if np.any(v_hip > 0.05) else 15
    lift_window = slice(max(0, lift_start), min(T, lift_start + 25))

    # 2. 정면/전방 가림 극복 4대 동역학 피처
    # (1) 힙슈트업 속도비: 골반 상승 속도 / 무릎 신전 속도
    hip_speed = float(np.mean(np.maximum(0, v_hip[lift_window])))
    knee_speed = float(np.mean(np.maximum(0, v_knee[lift_window])) + 1e-4)
    f_shootup_ratio = float(np.clip(hip_speed / knee_speed, 0.0, 10.0))

    # (2) 상체 2D 투영 길이 단축비 (상체 숙여짐)
    torso_len_2d = np.abs(hip_y - shdr_y)
    torso_init = float(np.mean(torso_len_2d[:5]))
    torso_min_lift = float(np.min(torso_len_2d[lift_window]))
    f_foreshortening = float(np.clip((torso_init - torso_min_lift) / (torso_init + 1e-4), -1.0, 1.0))

    # (3) 무릎 신전 지연 프레임
    knee_start = int(np.argmax(v_knee > 0.05)) if np.any(v_knee > 0.05) else lift_start
    f_delay = float(np.clip(knee_start - lift_start, -10.0, 30.0))

    # (4) 정강이 수직 속도 RMS (원판 위 정강이 가속도)
    v_shank = np.abs(v_knee - v_ank)
    f_shank_v = float(np.sqrt(np.mean(v_shank**2)))

    f4_dyn = [f_shootup_ratio, f_foreshortening, f_delay, f_shank_v]

    # 3. 5대 시점 원핫 인코딩 (5D)
    f5_view = [1.0 if view_code == v else 0.0 for v in VIEW_LIST]

    # 30D 결합
    f30 = np.hstack([f21, f4_dyn, f5_view]).astype(np.float32)
    return np.nan_to_num(f30, nan=0.0)

def main():
    meta = np.load(CACHE_META, allow_pickle=True)
    clip_names = meta['clip_names']
    subjects = meta['subjects']
    views = meta['views']
    y_golden = meta['y']
    N = len(clip_names)
    y_d3 = y_golden[:, 2]

    print("=" * 80)
    print("  [v9.0-RTM 프로토타입] 30D 시점 적응형 생체역학 앙상블 학습 & 평가")
    print("=" * 80)

    t0 = time.time()
    feats = []
    for i, cname in enumerate(clip_names):
        rtm_p = os.path.join(RTM_DIR, cname + ".npz")
        data = np.load(rtm_p)
        kps = halpe26_to_mediapipe33(data['kps_26_64f'])
        f = extract_d3_v9_features(kps, str(views[i]))
        feats.append(f)
    X = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0)
    print(f"[+] 30D 피처 추출 완료: shape={X.shape} ({time.time()-t0:.2f}초)")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y_d3, groups=subjects))

    oof_probs = np.zeros(N, dtype=np.float32)
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        X_trn, y_trn = X[trn_idx], y_d3[trn_idx]
        X_val = X[val_idx]

        cb = CatBoostClassifier(
            iterations=220, depth=5, learning_rate=0.04,
            l2_leaf_reg=3.0, auto_class_weights='Balanced',
            random_seed=42 + fold_idx, thread_count=1,
            allow_writing_files=False, verbose=False
        )
        cb.fit(X_trn, y_trn)
        p_cb = cb.predict_proba(X_val)[:, 1]

        clf = lgb.LGBMClassifier(
            n_estimators=160, max_depth=5, learning_rate=0.038,
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
    print(f"[*] v9.0 30D ROC-AUC: {auc:.2f}% (기존 v8.0 81.51% 대비 {auc - 81.51:+.2f}%p)")

    # 시점별 3-State 그리드 최적화 (보류율 <= 15.0% 하드 제약 준수)
    print("\n[시점별 3-State 최적화 탐색 (보류율 <= 15.0%)]")
    tot_mask = np.zeros(N, dtype=bool)

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
        print(f"  - {VIEW_MAP[v]:<26}: [{best_pair[0]:.2f}, {best_pair[1]:.2f}] | "
              f"F1={best_res['f1']:6.2f}% | Prec={best_res['prec']:6.2f}% | Rec={best_res['rec']:6.2f}% | "
              f"보류율={best_res['retake']:5.1f}%")

    yt_d = y_d3[tot_mask]
    yp_d = (oof_probs[tot_mask] >= 0.50).astype(int)
    all_f1 = float(f1_score(yt_d, yp_d, zero_division=0) * 100.0)
    all_prec = float(precision_score(yt_d, yp_d, zero_division=0) * 100.0)
    all_rec = float(recall_score(yt_d, yp_d, zero_division=0) * 100.0)
    all_acc = float(accuracy_score(yt_d, yp_d) * 100.0)
    all_cov = float(np.sum(tot_mask) / N * 100.0)
    all_retake = 100.0 - all_cov

    print("\n" + "=" * 80)
    print(f"  >> [원천 데이터 770건 전수 v9.0 종합 성능] <<")
    print(f"  - 3-State F1: {all_f1:.2f}% (목표 70.0% 달성 여부: {'성공! ★' if all_f1 >= 70.0 else '미달'})")
    print(f"  - 정밀도: {all_prec:.2f}% | 재현율: {all_rec:.2f}% | 정확도: {all_acc:.2f}%")
    print(f"  - 전체 ROC-AUC: {auc:.2f}%")
    print(f"  - 커버리지: {all_cov:.2f}% | 보류율: {all_retake:.2f}% (<= 15.0% 제약 준수 여부: {'준수' if all_retake <= 15.0 else '초과'})")
    print("=" * 80)

if __name__ == '__main__':
    main()
