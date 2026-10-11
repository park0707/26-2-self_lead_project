# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: verify_sl2_feature_optimization.py
Purpose: sl2 피처 수 과적합 가설 검증 및 피처 최적화(Feature Selection/Pruning) 토너먼트 실험
         - 84D Baseline 대비 피처 축소 시 성능 변화(과적합 완화 vs 정보 유실) 실증 분석
         - 5-Fold StratifiedGroupKFold 피험자 9명 전수 격리 교차검증
         - 평가 지표: Train vs Val AUC Gap (과적합 지표), OOF ROC-AUC, 시점 B F1, 시점 D F1
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, log_loss
from scipy.optimize import minimize

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MP_FRONT_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v8_front_58d_cache.pkl"
RTMP_CACHE_NPZ = r"D:\Data\피처_데이터\side_lunge_rtmp_features.npz"
OUT_JSON = r"D:\src\models\second_models\side_lunge\sl2_feature_optimization_results.json"

VALID_RTMP_INDICES = [
    0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 44, 45, 46, 50
]
DROP_COLS_84D = [25, 27, 28, 29, 58 + 22, 58 + 23, 58 + 24, 58 + 25, 58 + 26]
KEEP_COLS_84D = [i for i in range(58 + 35) if i not in DROP_COLS_84D]


def search_view_thresholds(y_true: np.ndarray, y_prob: np.ndarray, max_retake: float = 10.0) -> dict:
    best_f1 = -1.0
    best_res = {'tau_clean': 0.45, 'tau_defect': 0.60, 'f1': 0.0, 'prec': 0.0, 'rec': 0.0, 'acc': 0.0, 'retake': 0.0}

    tc_cands = np.linspace(0.20, 0.55, 36)
    td_cands = np.linspace(0.40, 0.75, 36)

    for tc in tc_cands:
        for td in td_cands:
            if td <= tc:
                continue
            unc = (y_prob > tc) & (y_prob < td)
            ret = float(np.mean(unc) * 100.0)
            if ret > max_retake + 0.01:
                continue
            dec = ~unc
            if np.sum(dec) == 0:
                continue
            yt = y_true[dec]
            yp = (y_prob[dec] >= td).astype(int)
            f1_val = float(f1_score(yt, yp, zero_division=0) * 100.0)

            if f1_val > best_f1:
                best_f1 = f1_val
                best_res = {
                    'tau_clean': round(float(tc), 3),
                    'tau_defect': round(float(td), 3),
                    'f1': round(f1_val, 2),
                    'prec': round(float(precision_score(yt, yp, zero_division=0) * 100.0), 2),
                    'rec': round(float(recall_score(yt, yp, zero_division=0) * 100.0), 2),
                    'acc': round(float(accuracy_score(yt, yp) * 100.0), 2),
                    'retake': round(ret, 2)
                }
    return best_res


def evaluate_feature_subset(
    X_sub: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    views: np.ndarray,
    config_name: str,
    feature_indices: List[int]
) -> Dict[str, Any]:
    N, D = X_sub.shape
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(N)
    train_aucs = []

    scale_pos = 4.364

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups), start=1):
        X_tr, y_tr = X_sub[trn_idx], y[trn_idx]
        X_val, y_val = X_sub[val_idx], y[val_idx]

        # CatBoost
        cb = CatBoostClassifier(
            iterations=600, learning_rate=0.035, depth=5, scale_pos_weight=scale_pos,
            random_seed=42 + fold, verbose=0, thread_count=-1
        )
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

        # XGBoost
        xm = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.035, max_depth=5, scale_pos_weight=scale_pos,
            reg_alpha=1.5, reg_lambda=3.5, random_state=42 + fold, eval_metric='auc',
            early_stopping_rounds=50, n_jobs=-1
        )
        xm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # LightGBM
        lm = lgb.LGBMClassifier(
            n_estimators=450, learning_rate=0.035, max_depth=5, num_leaves=24,
            scale_pos_weight=scale_pos, reg_alpha=1.5, reg_lambda=3.5,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        lm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # Train prediction for overfitting audit
        p_tr_cb = cb.predict_proba(X_tr)[:, 1]
        p_tr_xm = xm.predict_proba(X_tr)[:, 1]
        p_tr_lm = lm.predict_proba(X_tr)[:, 1]
        p_tr = 0.55 * p_tr_cb + 0.25 * p_tr_xm + 0.20 * p_tr_lm
        train_aucs.append(roc_auc_score(y_tr, p_tr))

        # Val OOF prediction
        p_val_cb = cb.predict_proba(X_val)[:, 1]
        p_val_xm = xm.predict_proba(X_val)[:, 1]
        p_val_lm = lm.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = 0.55 * p_val_cb + 0.25 * p_val_xm + 0.20 * p_val_lm

    # Temperature Scaling
    eps = 1e-7
    p_clip = np.clip(oof_probs, eps, 1.0 - eps)
    z = np.log(p_clip / (1.0 - p_clip))
    def nll(T):
        p_t = 1.0 / (1.0 + np.exp(-z / T[0]))
        p_t = np.clip(p_t, eps, 1.0 - eps)
        return log_loss(y, p_t)
    res_t = minimize(nll, x0=[1.0], bounds=[(0.05, 5.0)], method='Nelder-Mead')
    T_opt = float(res_t.x[0]) if res_t.success else 1.0
    oof_cal = 1.0 / (1.0 + np.exp(-z / T_opt))

    mean_train_auc = float(np.mean(train_aucs) * 100.0)
    val_auc = float(roc_auc_score(y, oof_cal) * 100.0)
    overfit_gap = float(mean_train_auc - val_auc)

    # View metrics
    mask_v2 = (views == 'view2')
    mask_v4 = (views == 'view4')

    auc_v2 = float(roc_auc_score(y[mask_v2], oof_cal[mask_v2]) * 100.0)
    auc_v4 = float(roc_auc_score(y[mask_v4], oof_cal[mask_v4]) * 100.0)

    # Breakthrough profile (retake <= 18%)
    bt_v2 = search_view_thresholds(y[mask_v2], oof_cal[mask_v2], max_retake=18.0)
    bt_v4 = search_view_thresholds(y[mask_v4], oof_cal[mask_v4], max_retake=18.0)

    # Strict profile (retake <= 10%)
    st_v2 = search_view_thresholds(y[mask_v2], oof_cal[mask_v2], max_retake=10.0)
    st_v4 = search_view_thresholds(y[mask_v4], oof_cal[mask_v4], max_retake=10.0)

    result = {
        'config': config_name,
        'dim': D,
        'train_auc': round(mean_train_auc, 2),
        'val_auc': round(val_auc, 2),
        'overfit_gap': round(overfit_gap, 2),
        'auc_v2': round(auc_v2, 2),
        'auc_v4': round(auc_v4, 2),
        'v2_breakthrough': bt_v2,
        'v4_breakthrough': bt_v4,
        'v2_strict': st_v2,
        'v4_strict': st_v4,
        'temperature': round(T_opt, 4)
    }

    print(f"[{config_name:24s} | {D:2d}D] Train AUC: {mean_train_auc:.2f}% | Val AUC: {val_auc:.2f}% (Gap: {overfit_gap:+.2f}%p) | V2 F1(BT): {bt_v2['f1']:.2f}% | V4 F1(BT): {bt_v4['f1']:.2f}%")
    return result


def run_feature_optimization_experiments():
    print("=" * 90)
    print("vidio-health-trainer: sl2 피처 수 최적화 및 과적합 가설 실증 검증 토너먼트")
    print("=" * 90)

    # 1. 데이터 로드
    with open(MP_FRONT_CACHE_PKL, 'rb') as f:
        mp = pickle.load(f)
    rtm = np.load(RTMP_CACHE_NPZ, allow_pickle=True)

    rtm_views = rtm['views']
    rtm_clips = rtm['clip_names']
    mp_clips = mp['clip_names']
    mp_views = mp['views']

    rtm_key_map = {(clip, view): i for i, (clip, view) in enumerate(zip(rtm_clips, rtm_views))}
    rtm_indices = [rtm_key_map[(clip, view)] for clip, view in zip(mp_clips, mp_views)]

    X_rtm_front = rtm['X_c2'][rtm_indices][:, VALID_RTMP_INDICES].astype(np.float32)
    X_mp_front = mp['X'].astype(np.float32)

    X_hyb_93d = np.hstack([X_mp_front, X_rtm_front])
    X_hyb_84d = X_hyb_93d[:, KEEP_COLS_84D]
    y = mp['y'].astype(int)
    groups = mp['groups']
    views = mp['views']
    N, D_full = X_hyb_84d.shape

    print(f"[*] 원본 84D 데이터: N={N}, D={D_full}, 양성률={np.mean(y==1)*100:.2f}%")

    # 2. 84D 전 피처에 대한 GBDT 통합 Feature Importance 계산 (Fold 앙상블)
    print("[*] 84D 전 피처 대상 5-Fold 평균 GBDT 피처 중요도(CatBoost+XGBoost+LightGBM) 산출 중...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    importances = np.zeros(D_full)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_hyb_84d, y, groups), start=1):
        X_tr, y_tr = X_hyb_84d[trn_idx], y[trn_idx]
        X_val, y_val = X_hyb_84d[val_idx], y[val_idx]

        cb = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, scale_pos_weight=4.364, random_seed=42+fold, verbose=0)
        cb.fit(X_tr, y_tr, verbose=False)
        importances += cb.get_feature_importance() / 5.0

    ranked_indices = np.argsort(importances)[::-1]
    print(f"[*] 최고 중요도 상위 5개 인덱스: {ranked_indices[:5]} (중요도: {importances[ranked_indices[:5]]})")
    print(f"[*] 최저 중요도 하위 5개 인덱스: {ranked_indices[-5:]} (중요도: {importances[ranked_indices[-5:]]})")

    results = {}

    # 실험 0: Baseline (84D)
    results['Baseline_84D'] = evaluate_feature_subset(
        X_hyb_84d, y, groups, views, "Baseline_84D", list(range(D_full))
    )

    # 실험 1: Top-60D (하위 24개 노이즈 피처 제거)
    top60_idx = sorted(ranked_indices[:60])
    results['Top60D'] = evaluate_feature_subset(
        X_hyb_84d[:, top60_idx], y, groups, views, "Top60D", top60_idx
    )

    # 실험 2: Top-45D (중요도 상위 45개 정예화)
    top45_idx = sorted(ranked_indices[:45])
    results['Top45D'] = evaluate_feature_subset(
        X_hyb_84d[:, top45_idx], y, groups, views, "Top45D", top45_idx
    )

    # 실험 3: Top-30D (핵심 30개 고밀도 피처)
    top30_idx = sorted(ranked_indices[:30])
    results['Top30D'] = evaluate_feature_subset(
        X_hyb_84d[:, top30_idx], y, groups, views, "Top30D", top30_idx
    )

    # 실험 4: Top-20D (초경량 20개 피처)
    top20_idx = sorted(ranked_indices[:20])
    results['Top20D'] = evaluate_feature_subset(
        X_hyb_84d[:, top20_idx], y, groups, views, "Top20D", top20_idx
    )

    # 실험 5: 다중공선성 프루닝 (|r| > 0.85 제거)
    corr_matrix = np.corrcoef(X_hyb_84d, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    to_drop = set()
    for i in range(D_full):
        if i in to_drop:
            continue
        for j in range(i + 1, D_full):
            if j in to_drop:
                continue
            if abs(corr_matrix[i, j]) > 0.85:
                # 중요도가 더 낮은 피처 제거
                if importances[i] >= importances[j]:
                    to_drop.add(j)
                else:
                    to_drop.add(i)
                    break
    collinear_keep = [i for i in range(D_full) if i not in to_drop]
    results[f'CollinearPruned_{len(collinear_keep)}D'] = evaluate_feature_subset(
        X_hyb_84d[:, collinear_keep], y, groups, views, f"CollinearPruned_{len(collinear_keep)}D", collinear_keep
    )

    # 실험 6: MediaPipe 3D Only (54D)
    mp_indices = list(range(54))
    results['MediaPipe_Only_54D'] = evaluate_feature_subset(
        X_hyb_84d[:, mp_indices], y, groups, views, "MediaPipe_Only_54D", mp_indices
    )

    # 실험 7: RTMPose 2D Only (30D)
    rtm_indices = list(range(54, 84))
    results['RTMPose_Only_30D'] = evaluate_feature_subset(
        X_hyb_84d[:, rtm_indices], y, groups, views, "RTMPose_Only_30D", rtm_indices
    )

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 결과 JSON 저장 완료: {OUT_JSON}")


if __name__ == '__main__':
    run_feature_optimization_experiments()
