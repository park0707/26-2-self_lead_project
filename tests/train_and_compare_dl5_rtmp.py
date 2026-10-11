# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드, 06_스텝_백워드)
Script: D:\tests\train_and_compare_dl5_rtmp.py
Purpose: 다이나믹 런지 c5 RTMP 기반 모델 비교 훈련 및 전수 교차검증 (StratifiedGroupKFold)
         - Setup 1: Pure RTMP 73D (CatBoost + LightGBM + XGBoost)
         - Setup 2: Hybrid 79D (MediaPipe 3D 36D + RTMP 37D + Cross-modal 6D - bl5_final 아키텍처 계승)
         - Setup 3: MediaPipe 36D (비교군)
         - 전수 1,468개 시퀀스, 피험자 1~9 전원 격리, 5대 시점 전수 감사
"""

import os
import sys
import json
import time
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)

DATASET_PATH = r"D:\Data\피처_데이터\dynamic_lunge_c5_rtmp_dataset.npz"
OUT_JSON_PATH = r"D:\tests\dl5_rtmp_comparison_results.json"

VIEW_INFO = {
    "view5": {"code": "view5", "name": "시점 A (우측 후방 45°)", "angle": "Right Back 45°"},
    "view2": {"code": "view2", "name": "시점 B (우측 전방 45°)", "angle": "Right Front 45°"},
    "view3": {"code": "view3", "name": "시점 C (정면 0°)", "angle": "Front View 0°"},
    "view4": {"code": "view4", "name": "시점 D (좌측 전방 45°)", "angle": "Left Front 45°"},
    "view1": {"code": "view1", "name": "시점 E (좌측 후방 45°)", "angle": "Left Back 45°"},
}

def find_best_3state_thresholds(y_true, y_prob, max_rejection=0.10):
    N = len(y_true)
    if N == 0:
        return 0.5, 0.5, {'f1': 0.0, 'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'cov': 0.0, 'retake': 0.0}
    
    p_min, p_max = float(np.percentile(y_prob, 1)), float(np.percentile(y_prob, 99))
    taus = np.linspace(p_min, p_max, 50)
    best_f1 = -1.0
    best_clean, best_defect = 0.5, 0.5
    best_metrics = {}

    for i in range(len(taus)):
        t_c = taus[i]
        for j in range(i, len(taus)):
            t_d = taus[j]
            pred = np.full(N, 2, dtype=int)
            pred[y_prob < t_c] = 0
            pred[y_prob >= t_d] = 1

            cov_mask = (pred != 2)
            retake = 1.0 - np.mean(cov_mask)
            if retake > max_rejection + 1e-4:
                continue

            if np.sum(cov_mask) > 0:
                y_c = y_true[cov_mask]
                p_c = pred[cov_mask]
                f1 = f1_score(y_c, p_c, zero_division=0)
                if f1 > best_f1 or (abs(f1 - best_f1) < 1e-5 and retake < best_metrics.get('retake', 1.0)):
                    best_f1 = f1
                    best_clean, best_defect = float(t_c), float(t_d)
                    best_metrics = {
                        'f1': float(f1),
                        'acc': float(accuracy_score(y_c, p_c)),
                        'prec': float(precision_score(y_c, p_c, zero_division=0)),
                        'rec': float(recall_score(y_c, p_c, zero_division=0)),
                        'cov': float(np.mean(cov_mask)),
                        'retake': float(retake)
                    }

    if best_f1 < 0:
        pred_bin = (y_prob >= 0.5).astype(int)
        best_clean, best_defect = 0.5, 0.5
        best_metrics = {
            'f1': float(f1_score(y_true, pred_bin, zero_division=0)),
            'acc': float(accuracy_score(y_true, pred_bin)),
            'prec': float(precision_score(y_true, pred_bin, zero_division=0)),
            'rec': float(recall_score(y_true, pred_bin, zero_division=0)),
            'cov': 1.0,
            'retake': 0.0
        }
    return best_clean, best_defect, best_metrics

def evaluate_setup(X, y, groups, views, ex_types, setup_name):
    N = len(y)
    print(f"\n[*] Evaluating [{setup_name}] (Feature shape: {X.shape})...")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_cb = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_xgb = np.zeros(N, dtype=np.float32)

    t0 = time.time()
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups), 1):
        # 1. CatBoost
        cb = CatBoostClassifier(
            iterations=450, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
            scale_pos_weight=1.2, verbose=0, random_seed=42 + fold, thread_count=4
        )
        cb.fit(X[tr], y[tr], eval_set=(X[va], y[va]), early_stopping_rounds=35)
        oof_cb[va] = cb.predict_proba(X[va])[:, 1]

        # 2. LightGBM
        lgbm = lgb.LGBMClassifier(
            n_estimators=350, max_depth=5, num_leaves=25, learning_rate=0.03,
            scale_pos_weight=1.2, subsample=0.8, colsample_bytree=0.8,
            random_state=42 + fold, n_jobs=4, verbose=-1
        )
        lgbm.fit(X[tr], y[tr])
        oof_lgb[va] = lgbm.predict_proba(X[va])[:, 1]

        # 3. XGBoost
        xg = xgb.XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.03,
            scale_pos_weight=1.2, subsample=0.8, colsample_bytree=0.8,
            random_state=42 + fold, n_jobs=4, eval_metric='logloss'
        )
        xg.fit(X[tr], y[tr])
        oof_xgb[va] = xg.predict_proba(X[va])[:, 1]

    train_time = time.time() - t0

    # 3-Way Ensemble (CatBoost 50% + LightGBM 30% + XGBoost 20% - bl5_final weighting)
    oof_ens = 0.5 * oof_cb + 0.3 * oof_lgb + 0.2 * oof_xgb

    # 2-State Metrics
    preds_2s = (oof_ens >= 0.5).astype(int)
    f1_2s = float(f1_score(y, preds_2s))
    auc_total = float(roc_auc_score(y, oof_ens))
    acc_2s = float(accuracy_score(y, preds_2s))
    prec_2s = float(precision_score(y, preds_2s, zero_division=0))
    rec_2s = float(recall_score(y, preds_2s, zero_division=0))

    # 3-State Metrics (Global at 10% and 15% rejection)
    tc10, td10, m3s_10 = find_best_3state_thresholds(y, oof_ens, max_rejection=0.10)
    tc15, td15, m3s_15 = find_best_3state_thresholds(y, oof_ens, max_rejection=0.15)

    # View-Adaptive 3-State (Rejection <= 10% and <= 15% per view)
    view_results_10 = {}
    view_results_15 = {}
    y_pred_adaptive_10 = np.full(N, 2, dtype=int)
    y_pred_adaptive_15 = np.full(N, 2, dtype=int)

    for v_code, v_meta in VIEW_INFO.items():
        v_idx = (views == v_code)
        y_v = y[v_idx]
        prob_v = oof_ens[v_idx]

        # 10%
        tc_v10, td_v10, mv10 = find_best_3state_thresholds(y_v, prob_v, max_rejection=0.10)
        p_v10 = np.where(prob_v >= td_v10, 1, np.where(prob_v < tc_v10, 0, 2))
        y_pred_adaptive_10[v_idx] = p_v10
        auc_v = float(roc_auc_score(y_v, prob_v))
        view_results_10[v_code] = {
            'view_name': v_meta['name'],
            'count': int(np.sum(v_idx)),
            'auc': auc_v,
            'f1_3state': mv10['f1'],
            'accuracy': mv10['acc'],
            'precision': mv10['prec'],
            'recall': mv10['rec'],
            'retake_rate': mv10['retake'],
            'tau_clean': tc_v10,
            'tau_defect': td_v10
        }

        # 15%
        tc_v15, td_v15, mv15 = find_best_3state_thresholds(y_v, prob_v, max_rejection=0.15)
        p_v15 = np.where(prob_v >= td_v15, 1, np.where(prob_v < tc_v15, 0, 2))
        y_pred_adaptive_15[v_idx] = p_v15
        view_results_15[v_code] = {
            'view_name': v_meta['name'],
            'count': int(np.sum(v_idx)),
            'auc': auc_v,
            'f1_3state': mv15['f1'],
            'accuracy': mv15['acc'],
            'precision': mv15['prec'],
            'recall': mv15['rec'],
            'retake_rate': mv15['retake'],
            'tau_clean': tc_v15,
            'tau_defect': td_v15
        }

    # Overall adaptive metrics
    cov10 = (y_pred_adaptive_10 != 2)
    adapt_f1_10 = float(f1_score(y[cov10], y_pred_adaptive_10[cov10]))
    adapt_ret_10 = float(1.0 - np.mean(cov10))
    adapt_acc_10 = float(accuracy_score(y[cov10], y_pred_adaptive_10[cov10]))
    adapt_prec_10 = float(precision_score(y[cov10], y_pred_adaptive_10[cov10], zero_division=0))
    adapt_rec_10 = float(recall_score(y[cov10], y_pred_adaptive_10[cov10], zero_division=0))

    cov15 = (y_pred_adaptive_15 != 2)
    adapt_f1_15 = float(f1_score(y[cov15], y_pred_adaptive_15[cov15]))
    adapt_ret_15 = float(1.0 - np.mean(cov15))
    adapt_acc_15 = float(accuracy_score(y[cov15], y_pred_adaptive_15[cov15]))

    # Exercise type breakdown
    ex_results = {}
    for et in ['05_스텝_포워드', '06_스텝_백워드']:
        e_idx = (ex_types == et)
        y_e = y[e_idx]
        prob_e = oof_ens[e_idx]
        tc_e, td_e, me = find_best_3state_thresholds(y_e, prob_e, max_rejection=0.10)
        ex_results[et] = {
            'count': int(np.sum(e_idx)),
            'auc': float(roc_auc_score(y_e, prob_e)),
            'f1_3state_10': me['f1'],
            'retake_rate': me['retake'],
            'accuracy': me['acc']
        }

    print(f"  [+] 2-State Binary F1: {f1_2s*100:.2f}% | AUC = {auc_total:.4f}")
    print(f"  [+] 3-State Adaptive F1 (Rejection <= 10%): {adapt_f1_10*100:.2f}% (Retake: {adapt_ret_10*100:.2f}%)")
    print(f"  [+] 3-State Adaptive F1 (Rejection <= 15%): {adapt_f1_15*100:.2f}% (Retake: {adapt_ret_15*100:.2f}%)")
    for v_code, stat in view_results_10.items():
        print(f"      - {stat['view_name']:<24}: F1={stat['f1_3state']*100:5.2f}%, AUC={stat['auc']:.4f}, Retake={stat['retake_rate']*100:4.1f}%")

    return {
        'setup_name': setup_name,
        'feature_dim': int(X.shape[1]),
        'train_time_sec': float(train_time),
        '2state': {
            'f1': f1_2s,
            'auc': auc_total,
            'accuracy': acc_2s,
            'precision': prec_2s,
            'recall': rec_2s
        },
        '3state_global_10': m3s_10,
        '3state_global_15': m3s_15,
        '3state_adaptive_10': {
            'f1': adapt_f1_10,
            'retake_rate': adapt_ret_10,
            'accuracy': adapt_acc_10,
            'precision': adapt_prec_10,
            'recall': adapt_rec_10,
            'per_view': view_results_10
        },
        '3state_adaptive_15': {
            'f1': adapt_f1_15,
            'retake_rate': adapt_ret_15,
            'accuracy': adapt_acc_15,
            'per_view': view_results_15
        },
        'per_exercise': ex_results,
        'oof_probs': oof_ens.tolist()
    }

def main():
    print("=" * 85)
    print("  [vidio-health-trainer] 다이나믹 런지 c5 RTMP 기반 모델 비교 훈련 및 전수 교차검증")
    print("=" * 85)

    data = np.load(DATASET_PATH, allow_pickle=True)
    X_rtmp = data['X_rtmp_73d'].astype(np.float32)
    X_hybrid = data['X_hybrid_79d'].astype(np.float32)
    X_mp = data['X_mp_36d'].astype(np.float32)
    y = data['y'].astype(int)
    groups = data['subjects'].astype(int)
    views = data['views'].astype(str)
    ex_types = data['exercise_types'].astype(str)

    results = {}

    # Setup 1: dl5_rtmp_pure (73D RTMPose 2D)
    res_rtmp = evaluate_setup(X_rtmp, y, groups, views, ex_types, "dl5_rtmp_pure (73D Pure RTMPose)")
    results['dl5_rtmp_pure'] = res_rtmp

    # Setup 2: dl5_rtmp_hybrid (79D Hybrid: MP 36D + RTMP 37D + 6D cross-modal inheriting bl5_final)
    res_hybrid = evaluate_setup(X_hybrid, y, groups, views, ex_types, "dl5_rtmp_1 (79D Hybrid - bl5_final 계승)")
    results['dl5_rtmp_hybrid'] = res_hybrid

    # Setup 3: dl5_mp_only (36D MediaPipe 3D)
    res_mp = evaluate_setup(X_mp, y, groups, views, ex_types, "dl5_v5_retrained (36D MediaPipe Only)")
    results['dl5_mp_only'] = res_mp

    # Baseline comparison summary
    print("\n" + "=" * 105)
    print("  [다이나믹 런지 c5 모델 종합 성능 비교 매트릭스]")
    print("=" * 105)
    print(f"{'모델 명칭 및 구조':<36} | {'ROC-AUC':<8} | {'2-State F1':<10} | {'3-State F1 (<=10%)':<19} | {'3-State F1 (<=15%)':<19} | {'정확도':<8}")
    print("-" * 105)

    # 1. bl5_final zero-shot baseline
    print(f"{'bl5_final (바벨런지 제로샷 전이)':<36} | 0.7635   |  64.69%   | 68.80% (보류: 9.8%) | 69.82% (보류: 13.0%)| 68.99%")

    # 2. dl5_v5 previous baseline
    print(f"{'dl5_v5 (이전 MediaPipe 36D)':<36} | 0.8499   |  72.16%   | 76.05% (보류: 9.7%) | 78.20% (보류: 14.5%)| 81.13%")

    # 3. New Models
    for k, r in results.items():
        name = r['setup_name']
        auc = r['2state']['auc']
        f1_2s = r['2state']['f1'] * 100
        f1_3s_10 = r['3state_adaptive_10']['f1'] * 100
        ret_10 = r['3state_adaptive_10']['retake_rate'] * 100
        f1_3s_15 = r['3state_adaptive_15']['f1'] * 100
        ret_15 = r['3state_adaptive_15']['retake_rate'] * 100
        acc = r['3state_adaptive_10']['accuracy'] * 100
        print(f"{name:<36} | {auc:.4f}   |  {f1_2s:6.2f}%   | {f1_3s_10:5.2f}% (보류:{ret_10:4.1f}%) | {f1_3s_15:5.2f}% (보류:{ret_15:4.1f}%)| {acc:6.2f}%")

    print("=" * 105)

    # Per-view comparison table
    print("\n" + "=" * 105)
    print("  [공식 5대 시점(A~E)별 3-State F1 성능 비교 (보류율 <= 10.0%)]")
    print("=" * 105)
    print(f"{'공식 시점 명칭 및 각도':<28} | {'bl5_final (전이)':<15} | {'dl5_rtmp_pure':<15} | {'dl5_rtmp_1 (Hybrid)':<20} | {'개선폭 (ΔF1 vs 전이)':<18}")
    print("-" * 105)

    bl5_transfer_views = {
        'view5': 80.00,
        'view2': 76.99,
        'view3': 58.26,
        'view4': 68.63,
        'view1': 73.33
    }

    for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_name = VIEW_INFO[v]['name']
        f1_bl5 = bl5_transfer_views[v]
        f1_pure = results['dl5_rtmp_pure']['3state_adaptive_10']['per_view'][v]['f1_3state'] * 100
        f1_hyb = results['dl5_rtmp_hybrid']['3state_adaptive_10']['per_view'][v]['f1_3state'] * 100
        diff = f1_hyb - f1_bl5
        print(f"{v_name:<28} | {f1_bl5:6.2f}%         | {f1_pure:6.2f}%         | {f1_hyb:6.2f}%               | +{diff:5.2f}%p" if diff >= 0 else f"{v_name:<28} | {f1_bl5:6.2f}%         | {f1_pure:6.2f}%         | {f1_hyb:6.2f}%               | {diff:5.2f}%p")
    print("=" * 105)

    # Save to JSON
    # Remove large arrays before saving json
    save_results = {}
    for k, v in results.items():
        d = dict(v)
        if 'oof_probs' in d:
            del d['oof_probs']
        save_results[k] = d

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 벤치마크 결과 JSON 저장 완료: {OUT_JSON_PATH}")

if __name__ == '__main__':
    main()
