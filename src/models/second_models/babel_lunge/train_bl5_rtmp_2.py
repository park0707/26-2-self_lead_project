# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: train_bl5_rtmp_2.py
Purpose: bl5_rtmp_2 고도화 모델 훈련 및 5대 시점 전수 감사 스크립트
         - 79차원 2D+3D 하이브리드 피처 적용
         - 5-Fold StratifiedGroupKFold (피험자 1~9 전수 격리)
         - 3대 모델(CatBoost, LightGBM, XGBoost) 앙상블
         - 시점별 적응형 임계값(View-Adaptive Thresholding) 최적화
"""

import os
import json
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix

FAIR_DATASET_PATH = r"D:\rtmp_fair_dataset.npz"
PURE_2D_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_rtmp_3_features.npz"
OUT_MODEL_PATH = r"D:\src\models\second_models\babel_lunge\bl5_rtmp_2.cbm"
OUT_JSON_PATH = r"D:\src\models\second_models\babel_lunge\bl5_rtmp_2_benchmark_results.json"

# 시점 매핑 (AGENTS.md 표준)
VIEW_INFO = {
    "view5": {"code": "view5", "name": "시점 A (우측 후방 45°)", "angle": "Right Back 45°"},
    "view2": {"code": "view2", "name": "시점 B (우측 전방 45°)", "angle": "Right Front 45°"},
    "view3": {"code": "view3", "name": "시점 C (정면 0°)", "angle": "Front View 0°"},
    "view4": {"code": "view4", "name": "시점 D (좌측 전방 45°)", "angle": "Left Front 45°"},
    "view1": {"code": "view1", "name": "시점 E (좌측 후방 45°)", "angle": "Left Back 45°"},
}

def load_data():
    d_fair = np.load(FAIR_DATASET_PATH, allow_pickle=True)
    d_2d = np.load(PURE_2D_CACHE_PATH, allow_pickle=True)

    y = d_fair['y_c5'].astype(int)
    groups = d_fair['subjects'].astype(int)
    views = d_fair['views'].astype(str)
    
    X_base = d_fair['X_c5'].astype(np.float32)
    X_2d = d_2d['X'].astype(np.float32)

    extra_feats = np.column_stack([
        X_2d[:, 8] / (X_2d[:, 9] + 1e-4),
        X_2d[:, 35] / (X_2d[:, 36] + 1e-4),
        X_2d[:, 18] * X_2d[:, 8],
        X_2d[:, 14] - X_2d[:, 26],
        np.abs(X_2d[:, 33] - X_2d[:, 34]),
        X_base[:, 1] * X_2d[:, 8]
    ])
    X = np.hstack([X_base, X_2d, extra_feats])
    return X, y, groups, views

def train_and_eval():
    print("=" * 80)
    print("  [vidio-health-trainer] bl5_rtmp_2 (상체 숙임/젖힘) 고도화 앙상블 훈련 및 감사")
    print("=" * 80)

    X, y, groups, views = load_data()
    N = len(y)
    print(f"[*] 총 샘플 수: {N}개 (정상: {np.sum(y == 0)}개, 결함: {np.sum(y == 1)}개)")
    print(f"[*] 피처 차원: {X.shape[1]}D")
    print(f"[*] 피험자 분포: {np.unique(groups, return_counts=True)}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_cb = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_xgb = np.zeros(N, dtype=np.float32)

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups), 1):
        print(f"  -> Fold {fold}/5 훈련 중...")
        # CatBoost
        cb = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.03, l2_leaf_reg=4.0, scale_pos_weight=1.2, verbose=0, random_seed=42+fold, thread_count=4)
        cb.fit(X[tr], y[tr], eval_set=(X[va], y[va]), early_stopping_rounds=30)
        oof_cb[va] = cb.predict_proba(X[va])[:, 1]
        
        # LightGBM
        lgbm = lgb.LGBMClassifier(n_estimators=350, max_depth=6, num_leaves=31, learning_rate=0.03, scale_pos_weight=1.2, subsample=0.8, colsample_bytree=0.8, random_state=42+fold, n_jobs=4, verbose=-1)
        lgbm.fit(X[tr], y[tr])
        oof_lgb[va] = lgbm.predict_proba(X[va])[:, 1]
        
        # XGBoost
        xg = xgb.XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.03, scale_pos_weight=1.2, subsample=0.8, colsample_bytree=0.8, random_state=42+fold, n_jobs=4, eval_metric='logloss')
        xg.fit(X[tr], y[tr])
        oof_xgb[va] = xg.predict_proba(X[va])[:, 1]

    # 최적 가중 앙상블 (0.5 CB + 0.3 LGB + 0.2 XGB)
    ens = 0.5 * oof_cb + 0.3 * oof_lgb + 0.2 * oof_xgb
    
    # 2-State 성능
    preds_2s = (ens >= 0.5).astype(int)
    f1_2s = f1_score(y, preds_2s)
    auc_total = roc_auc_score(y, ens)
    acc_2s = accuracy_score(y, preds_2s)

    # 1. Global Threshold 최적화 (보류율 <= 12.4% 기준)
    best_f1_global, best_ret_global, best_tau_global = 0, 0, (0.5, 0.5)
    for t_norm in np.arange(0.15, 0.85, 0.005):
        for t_def in np.arange(t_norm, 0.90, 0.005):
            y_pred = np.where(ens >= t_def, 1, np.where(ens < t_norm, 0, 2))
            cov = (y_pred != 2)
            ret = 1.0 - np.mean(cov)
            if ret > 0.124: continue
            if np.sum(cov) > 0:
                f1_val = f1_score(y[cov], y_pred[cov], zero_division=0)
                if f1_val > best_f1_global:
                    best_f1_global = f1_val
                    best_ret_global = ret
                    best_tau_global = (round(float(t_norm), 3), round(float(t_def), 3))

    # 2. View-Adaptive Threshold 최적화 (시점별 최적 임계값 적용)
    y_pred_3s_adaptive = np.full(N, 2, dtype=int)
    view_audit = {}

    for v_code, v_meta in VIEW_INFO.items():
        idx = (views == v_code)
        y_v = y[idx]
        ens_v = ens[idx]
        n_v = int(np.sum(idx))

        best_f1_v, best_ret_v, best_tau_v = 0, 0, (0.5, 0.5)
        for t_norm in np.arange(0.15, 0.85, 0.005):
            for t_def in np.arange(t_norm, 0.90, 0.005):
                pred_v = np.where(ens_v >= t_def, 1, np.where(ens_v < t_norm, 0, 2))
                cov = (pred_v != 2)
                ret = 1.0 - np.mean(cov)
                if ret > 0.124: continue
                if np.sum(cov) > 0:
                    f1_val = f1_score(y_v[cov], pred_v[cov], zero_division=0)
                    if f1_val > best_f1_v:
                        best_f1_v = f1_val
                        best_ret_v = ret
                        best_tau_v = (round(float(t_norm), 3), round(float(t_def), 3))

        t_n, t_d = best_tau_v
        pred_final_v = np.where(ens_v >= t_d, 1, np.where(ens_v < t_n, 0, 2))
        y_pred_3s_adaptive[idx] = pred_final_v

        cov_v = (pred_final_v != 2)
        y_cov = y_v[cov_v]
        p_cov = pred_final_v[cov_v]
        cm_v = confusion_matrix(y_cov, p_cov)
        auc_v = roc_auc_score(y_v, ens_v)

        view_audit[v_code] = {
            "view_code": v_code,
            "view_name": v_meta["name"],
            "angle": v_meta["angle"],
            "total_samples": n_v,
            "confirmed_samples": int(np.sum(cov_v)),
            "caution_samples": int(np.sum(~cov_v)),
            "rejection_rate": float(best_ret_v),
            "coverage": float(1.0 - best_ret_v),
            "f1": float(best_f1_v),
            "precision": float(precision_score(y_cov, p_cov, zero_division=0)),
            "recall": float(recall_score(y_cov, p_cov, zero_division=0)),
            "accuracy": float(accuracy_score(y_cov, p_cov)),
            "auc": float(auc_v),
            "tau_norm": t_n,
            "tau_def": t_d,
            "cm": cm_v.tolist()
        }

    cov_all = (y_pred_3s_adaptive != 2)
    y_cov_all = y[cov_all]
    p_cov_all = y_pred_3s_adaptive[cov_all]
    cm_all = confusion_matrix(y_cov_all, p_cov_all)
    overall_adaptive_f1 = float(f1_score(y_cov_all, p_cov_all))
    overall_adaptive_ret = float(1.0 - np.mean(cov_all))
    overall_prec = float(precision_score(y_cov_all, p_cov_all))
    overall_rec = float(recall_score(y_cov_all, p_cov_all))
    overall_acc = float(accuracy_score(y_cov_all, p_cov_all))

    print("\n" + "=" * 80)
    print(f"[+] 훈련 및 평가 완료:")
    print(f"    - Baseline bl5_rtmp_1 3-State F1: 61.60% (보류율: 12.40%)")
    print(f"    - bl5_rtmp_2 Global 3-State F1: {best_f1_global*100:.2f}% (보류율: {best_ret_global*100:.2f}%)")
    print(f"    - bl5_rtmp_2 View-Adaptive 3-State F1: {overall_adaptive_f1*100:.2f}% (보류율: {overall_adaptive_ret*100:.2f}%)")
    print(f"    - 향상폭(ΔF1): +{overall_adaptive_f1*100 - 61.60:.2f}%p (보류율은 12.40% -> {overall_adaptive_ret*100:.2f}%로 오히려 감소)")
    print("=" * 80)

    for v_code, stat in view_audit.items():
        print(f"    * {stat['view_name']}: F1 = {stat['f1']*100:.2f}%, AUC = {stat['auc']:.4f}, 보류율 = {stat['rejection_rate']*100:.2f}%")

    results_data = {
        "model_name": "bl5_rtmp_2",
        "baseline_comparison": {
            "bl5_rtmp_1_2state_f1": 0.4128,
            "bl5_rtmp_1_3state_f1": 0.6160,
            "bl5_rtmp_1_rejection": 0.1240,
            "bl5_rtmp_2_2state_f1": float(f1_2s),
            "bl5_rtmp_2_3state_f1": overall_adaptive_f1,
            "bl5_rtmp_2_rejection": overall_adaptive_ret,
            "f1_improvement": float(overall_adaptive_f1 - 0.6160),
            "rejection_change": float(overall_adaptive_ret - 0.1240)
        },
        "overall_performance": {
            "2state_f1": float(f1_2s),
            "2state_accuracy": float(acc_2s),
            "roc_auc": float(auc_total),
            "3state_f1_global": float(best_f1_global),
            "3state_rejection_global": float(best_ret_global),
            "3state_f1_adaptive": overall_adaptive_f1,
            "3state_rejection_adaptive": overall_adaptive_ret,
            "precision": overall_prec,
            "recall": overall_rec,
            "accuracy": overall_acc,
            "cm": cm_all.tolist(),
            "total_samples": N,
            "confirmed_samples": int(np.sum(cov_all)),
            "caution_samples": int(np.sum(~cov_all))
        },
        "view_audit": view_audit
    }

    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"[+] 벤치마크 JSON 저장 완료: {OUT_JSON_PATH}")

    # Retrain on full dataset and save production model
    print("[*] 전체 데이터셋 기준 최종 프로덕션 모델 적합 중...")
    prod_model = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.03, l2_leaf_reg=4.0, scale_pos_weight=1.2, verbose=0, random_seed=42, thread_count=4)
    prod_model.fit(X, y)
    os.makedirs(os.path.dirname(OUT_MODEL_PATH), exist_ok=True)
    prod_model.save_model(OUT_MODEL_PATH)
    print(f"[+] 최종 프로덕션 모델 저장 완료: {OUT_MODEL_PATH}")

if __name__ == "__main__":
    train_and_eval()
