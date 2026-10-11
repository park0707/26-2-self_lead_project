# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: experiment_cl2_benchmarks_on_sl2.py
Purpose: 크로스 런지 c2 (cl2)의 성공 요인(3-way 앙상블, 외과적 피처 프루닝, 시점 편향 제거, 불균형 분석)을
         사이드 런지 c2 (sl2) 4,130건 전수 데이터셋에 체계적으로 벤치마킹하여 성능 향상 가능성을 정밀 검증.

실험 구성:
  1. [Baseline]: sl2_hybrid_v1 현행 챔피언 (86D: MP 51D + RTMP 35D, CatBoost + XGBoost 2-way)
  2. [Exp 1 - cl2 아키텍처 이식]: 86D + CatBoost + XGBoost + LightGBM 3-way 앙상블
  3. [Exp 2 - cl2 중복 제거]: 81D (RTMP 시점 중복 5D 제거) + 3-way 앙상블
  4. [Exp 3 - cl2 뷰-프리 일반화]: 76D (MP/RTMP 모든 시점 원핫 10D 완전 제거) + 3-way 앙상블
  5. [Exp 4 - cl2 외과적 정예화]: 72D (시점 10D + 고상관 다중공선성 4D 외과적 제거) + 3-way 앙상블
  6. [Exp 5 - 분포 역설(Prevalence Paradox) 증명]: 50:50 균형 분포 조건에서의 메트릭 시뮬레이션
"""

import os
import sys
import json
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, log_loss, confusion_matrix
)
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from side_lunge_c2_v7_features import FEATURE_NAMES_51D

MP_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v7_all_views_51d_cache.pkl"
RTMP_CACHE_NPZ = r"D:\Data\피처_데이터\side_lunge_rtmp_features.npz"

VIEW_LIST = ['view5', 'view2', 'view3', 'view4', 'view1']
VIEW_ALIAS_MAP = {
    'view5': '시점 A: 우측 후방 (45°)',
    'view2': '시점 B: 우측 전방 (45°)',
    'view3': '시점 C: 정면 (0°)',
    'view4': '시점 D: 좌측 전방 (45°)',
    'view1': '시점 E: 좌측 후방 (45°)'
}

VALID_RTMP_INDICES = [
    0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 44, 45, 46, 50
]

FEATURE_NAMES_86D = (
    [f"mp_{name}" for name in FEATURE_NAMES_51D] +
    [f"rtmp_{FEATURE_NAMES_51D[idx]}" for idx in VALID_RTMP_INDICES]
)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(labels[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece * 100.0)


def optimize_temperature(z: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-7
    def nll_func(T):
        p_t = 1.0 / (1.0 + np.exp(-z / T[0]))
        p_t = np.clip(p_t, eps, 1.0 - eps)
        return log_loss(y, p_t)
    res = minimize(nll_func, x0=[1.0], bounds=[(0.05, 5.0)], method='Nelder-Mead')
    return float(res.x[0]) if res.success else 1.0


def search_view_thresholds(y_true: np.ndarray, y_prob: np.ndarray, min_cov: float = 90.0) -> dict:
    best_f1 = -1.0
    best_cov = 100.0
    best_params = {'tau_norm': 0.40, 'tau_def': 0.60}

    tc_cands = np.linspace(0.20, 0.48, 25)
    td_cands = np.linspace(0.50, 0.85, 25)

    for tc in tc_cands:
        for td in td_cands:
            if td <= tc:
                continue
            unc = (y_prob > tc) & (y_prob < td)
            cov = float(np.mean(~unc) * 100.0)
            if cov < min_cov:
                continue
            yt = y_true[~unc]
            yp = (y_prob[~unc] >= td).astype(int)
            if len(yt) == 0:
                continue
            f1_val = float(f1_score(yt, yp, zero_division=0) * 100.0)

            if f1_val > best_f1:
                best_f1 = f1_val
                best_cov = cov
                best_params = {'tau_norm': round(float(tc), 3), 'tau_def': round(float(td), 3)}
            elif abs(f1_val - best_f1) <= 0.2 and cov > best_cov:
                best_cov = cov
                best_params = {'tau_norm': round(float(tc), 3), 'tau_def': round(float(td), 3)}

    if best_f1 < 0:
        for margin in np.linspace(0.01, 0.10, 20):
            tc = 0.50 - margin
            td = 0.50 + margin
            unc = (y_prob > tc) & (y_prob < td)
            cov = float(np.mean(~unc) * 100.0)
            if cov >= min_cov:
                yt = y_true[~unc]
                yp = (y_prob[~unc] >= td).astype(int)
                best_f1 = float(f1_score(yt, yp, zero_division=0) * 100.0)
                best_params = {'tau_norm': round(float(tc), 3), 'tau_def': round(float(td), 3)}
                break

    return best_params


def evaluate_3state_pipeline(y_prob: np.ndarray, y_true: np.ndarray, views: np.ndarray) -> dict:
    eps = 1e-7
    p_clip = np.clip(y_prob, eps, 1.0 - eps)
    z = np.log(p_clip / (1.0 - p_clip))
    opt_T = optimize_temperature(z, y_true)
    p_cal = 1.0 / (1.0 + np.exp(-z / opt_T))
    post_ece = compute_ece(p_cal, y_true)

    y_pred_2s = (y_prob >= 0.5).astype(int)
    f1_2s = float(f1_score(y_true, y_pred_2s, zero_division=0) * 100.0)
    prec_2s = float(precision_score(y_true, y_pred_2s, zero_division=0) * 100.0)
    rec_2s = float(recall_score(y_true, y_pred_2s, zero_division=0) * 100.0)
    acc_2s = float(accuracy_score(y_true, y_pred_2s) * 100.0)
    auc_2s = float(roc_auc_score(y_true, y_prob) * 100.0)

    # 3-State per-view
    view_metrics = {}
    tot_dec = []
    tot_yp = []
    tot_yt = []

    for vn in VIEW_LIST:
        v_mask = (views == vn)
        yt_v = y_true[v_mask]
        yp_v = p_cal[v_mask]
        th = search_view_thresholds(yt_v, yp_v, min_cov=90.0)

        unc_v = (yp_v > th['tau_norm']) & (yp_v < th['tau_def'])
        dec_v = ~unc_v
        tot_dec.extend(dec_v)
        tot_yp.extend((yp_v[dec_v] >= th['tau_def']).astype(int))
        tot_yt.extend(yt_v[dec_v])

        v_yt_dec = yt_v[dec_v]
        v_yp_dec = (yp_v[dec_v] >= th['tau_def']).astype(int)
        view_metrics[vn] = {
            'alias': VIEW_ALIAS_MAP[vn],
            'tau_clean': th['tau_norm'],
            'tau_defect': th['tau_def'],
            'sel_f1': float(f1_score(v_yt_dec, v_yp_dec, zero_division=0) * 100.0),
            'sel_prec': float(precision_score(v_yt_dec, v_yp_dec, zero_division=0) * 100.0),
            'sel_rec': float(recall_score(v_yt_dec, v_yp_dec, zero_division=0) * 100.0),
            'sel_acc': float(accuracy_score(v_yt_dec, v_yp_dec) * 100.0),
            'coverage': float(np.mean(dec_v) * 100.0),
            'retake': float(np.mean(unc_v) * 100.0)
        }

    tot_dec = np.array(tot_dec)
    tot_yp = np.array(tot_yp)
    tot_yt = np.array(tot_yt)

    sel_f1 = float(f1_score(tot_yt, tot_yp, zero_division=0) * 100.0)
    sel_prec = float(precision_score(tot_yt, tot_yp, zero_division=0) * 100.0)
    sel_rec = float(recall_score(tot_yt, tot_yp, zero_division=0) * 100.0)
    sel_acc = float(accuracy_score(tot_yt, tot_yp) * 100.0)
    coverage = float(np.mean(tot_dec) * 100.0)
    retake = float(100.0 - coverage)

    # Balanced distribution simulation (p = 0.50)
    # Using empirical FPR and TPR:
    tpr = sel_rec / 100.0
    # From precision = p * tpr / (p * tpr + (1-p) * fpr)
    p_orig = np.mean(tot_yt == 1)
    if sel_prec > 0 and tpr > 0:
        denom = (p_orig * tpr) / (sel_prec / 100.0)
        fpr = (denom - p_orig * tpr) / (1.0 - p_orig)
        fpr = max(0.001, min(0.999, fpr))
        prec_balanced = (0.5 * tpr) / (0.5 * tpr + 0.5 * fpr)
        f1_balanced = float(2.0 * (prec_balanced * tpr) / (prec_balanced + tpr) * 100.0)
    else:
        fpr = 0.5
        f1_balanced = 0.0

    return {
        'auc': auc_2s,
        'f1_2s': f1_2s,
        'prec_2s': prec_2s,
        'rec_2s': rec_2s,
        'acc_2s': acc_2s,
        'sel_f1': sel_f1,
        'sel_prec': sel_prec,
        'sel_rec': sel_rec,
        'sel_acc': sel_acc,
        'coverage': coverage,
        'retake': retake,
        'ece': post_ece,
        'temperature': opt_T,
        'f1_balanced_50_50': f1_balanced,
        'fpr': float(fpr * 100.0),
        'view_metrics': view_metrics
    }


def run_experiment(exp_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, views: np.ndarray, use_lgb: bool = False, scale_pos: float = 4.364):
    print(f"\n{'='*70}\n[실험 시작] {exp_name} (X shape: {X.shape})\n{'='*70}")
    N, D = X.shape
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_cb = np.zeros(N)
    oof_xgb = np.zeros(N)
    oof_lgb = np.zeros(N)

    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. CatBoost
        cb = CatBoostClassifier(
            iterations=500, learning_rate=0.038, depth=6, l2_leaf_reg=4.0,
            scale_pos_weight=scale_pos, random_seed=42 + fold, verbose=False, thread_count=-1
        )
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof_cb[val_idx] = cb.predict_proba(X_val)[:, 1]

        # 2. XGBoost
        xgb_m = xgb.XGBClassifier(
            n_estimators=450, learning_rate=0.038, max_depth=5, scale_pos_weight=scale_pos,
            reg_alpha=1.0, reg_lambda=3.0, random_state=42 + fold, eval_metric='auc',
            early_stopping_rounds=45, n_jobs=-1
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_m.predict_proba(X_val)[:, 1]

        # 3. LightGBM (cl2 스타일)
        if use_lgb:
            lgb_m = lgb.LGBMClassifier(
                n_estimators=450, learning_rate=0.038, max_depth=5, num_leaves=24,
                scale_pos_weight=scale_pos, reg_alpha=1.0, reg_lambda=3.0,
                random_state=42 + fold, verbose=-1, n_jobs=-1
            )
            lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(45, verbose=False)])
            oof_lgb[val_idx] = lgb_m.predict_proba(X_val)[:, 1]

    elapsed = time.time() - t0

    if use_lgb:
        # Optimize 3-way weights via Nelder-Mead on AUC
        def loss_func(w):
            w = np.array(w)
            w = np.maximum(w, 0.0)
            if np.sum(w) == 0:
                return 1.0
            w = w / np.sum(w)
            ens = w[0] * oof_cb + w[1] * oof_xgb + w[2] * oof_lgb
            return -roc_auc_score(y, ens)
        opt = minimize(loss_func, x0=[0.45, 0.35, 0.20], method='Nelder-Mead')
        w_opt = np.maximum(opt.x, 0.0)
        w_opt = w_opt / np.sum(w_opt)
        oof_ens = w_opt[0] * oof_cb + w_opt[1] * oof_xgb + w_opt[2] * oof_lgb
        weights = {'catboost': round(float(w_opt[0]), 3), 'xgboost': round(float(w_opt[1]), 3), 'lightgbm': round(float(w_opt[2]), 3)}
    else:
        w_cb, w_xgb = 0.488, 0.512
        oof_ens = w_cb * oof_cb + w_xgb * oof_xgb
        weights = {'catboost': w_cb, 'xgboost': w_xgb}

    metrics = evaluate_3state_pipeline(oof_ens, y, views)
    metrics['weights'] = weights
    metrics['elapsed_sec'] = round(elapsed, 2)
    metrics['n_features'] = D

    print(f"[결과 요약] {exp_name}:")
    print(f"  - 3-State F1:       {metrics['sel_f1']:.2f}% (Prec: {metrics['sel_prec']:.2f}%, Rec: {metrics['sel_rec']:.2f}%)")
    print(f"  - ROC-AUC:          {metrics['auc']:.2f}%")
    print(f"  - FPR (위양성률):   {metrics['fpr']:.2f}%")
    print(f"  - Retake (재촬영):  {metrics['retake']:.2f}% (Coverage: {metrics['coverage']:.2f}%)")
    print(f"  - 가상 균형 F1:     {metrics['f1_balanced_50_50']:.2f}% (cl2와 동일 50:50 환경 시뮬레이션)")
    print(f"  - 소요 시간:        {elapsed:.1f}초")

    return metrics


def main():
    print("=" * 80)
    print("vidio-health-trainer: cl2 벤치마킹 기반 sl2 적용 실험 하네스 시작")
    print("=" * 80)

    # 1. 데이터 로드
    with open(MP_CACHE_PKL, 'rb') as f:
        mp_data = pickle.load(f)
    rtmp_data = np.load(RTMP_CACHE_NPZ, allow_pickle=True)

    X_mp = mp_data['X'].astype(np.float32)
    X_rtmp = rtmp_data['X_c2'][:, VALID_RTMP_INDICES].astype(np.float32)
    X_hybrid_86d = np.hstack([X_mp, X_rtmp])

    y = mp_data['y'].astype(int)
    groups = mp_data['groups']
    views = mp_data['views']

    print(f"[*] 원본 데이터: {len(y)}건 (양성: {np.sum(y==1)}건, 음성: {np.sum(y==0)}건, 양성률: {np.mean(y==1)*100:.2f}%)")

    all_results = {}

    # 실험 1: Baseline (sl2_hybrid_v1 재현: 86D, CatBoost + XGBoost)
    all_results['Baseline_86D_2way'] = run_experiment(
        "Baseline (sl2_hybrid_v1, 86D, 2-way)", X_hybrid_86d, y, groups, views, use_lgb=False
    )

    # 실험 2: cl2 3-way 앙상블 이식 (86D, CatBoost + XGBoost + LightGBM)
    all_results['Exp1_86D_3way'] = run_experiment(
        "Exp 1: cl2 3-way 앙상블 이식 (86D, CB+XGB+LGB)", X_hybrid_86d, y, groups, views, use_lgb=True
    )

    # 실험 3: cl2 피처 프루닝 1단계 - 중복된 RTMP 뷰 원핫 5D 제거 -> 81D
    # RTMP indices 73..77 correspond to VALID_RTMP_INDICES 22..26 (which are view_is_view1..5 in RTMP)
    # In 86D vector, indices 73..77 are exact duplicates of indices 25..29!
    drop_dup_view = [73, 74, 75, 76, 77]
    keep_81d = [i for i in range(86) if i not in drop_dup_view]
    X_81d = X_hybrid_86d[:, keep_81d]
    all_results['Exp2_81D_DropDupView'] = run_experiment(
        "Exp 2: cl2 중복 뷰 원핫 제거 (81D, 3-way)", X_81d, y, groups, views, use_lgb=True
    )

    # 실험 4: cl2 피처 프루닝 2단계 - 모든 뷰 원핫 10D 완전 제거 (순수 생체역학 76D 일반화 모델)
    drop_all_view = [25, 26, 27, 28, 29, 73, 74, 75, 76, 77]
    keep_76d = [i for i in range(86) if i not in drop_all_view]
    X_76d = X_hybrid_86d[:, keep_76d]
    all_results['Exp3_76D_ViewFree'] = run_experiment(
        "Exp 3: cl2 뷰-프리 일반화 모델 (76D, 3-way)", X_76d, y, groups, views, use_lgb=True
    )

    # 실험 5: cl2 피처 프루닝 3단계 - 뷰 10D 제거 + 고상관 다중공선성 4D 외과적 제거 (72D 정예 피처셋)
    # 고상관 피처:
    # 8 (mp_shank_torsion_plane_angle) vs 47 (mp_kinematic_shank_torsion_angle, r=-0.977) -> drop 47
    # 34 (mp_user_geom_torso_foot_parallel_dot) vs 35 (mp_user_geom_torso_foot_angle_deg, r=-0.973) -> drop 35
    # 66 (rtmp_delta_q_angle_standing_to_bottom) vs 71 (rtmp_fppa_pre_bottom_slope, r=-0.957) -> drop 66
    # 2 (mp_pacs_knee_ankle_coronal_tilt) vs 53 (rtmp_pacs_knee_ankle_coronal_tilt, r=0.91) -> keep both
    drop_collinear = [47, 35, 66]
    drop_72d = drop_all_view + drop_collinear
    keep_72d = [i for i in range(86) if i not in drop_72d]
    X_72d = X_hybrid_86d[:, keep_72d]
    all_results['Exp4_72D_SurgicalPruned'] = run_experiment(
        "Exp 4: cl2 외과적 정예화 모델 (72D, 3-way)", X_72d, y, groups, views, use_lgb=True
    )

    # 결과 JSON 저장
    OUT_JSON = os.path.join(CURRENT_DIR, "cl2_benchmarking_on_sl2_comparison_results.json")
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 모든 실험 완료! 결과가 저장되었습니다: {OUT_JSON}")


if __name__ == '__main__':
    main()
