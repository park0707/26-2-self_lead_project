# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: train_bl5_hybrid_1.py
Purpose: MediaPipe 3D + RTMPose 2D 하이브리드 모델 (bl5_hybrid_1) 훈련 및 5대 시점 전수 감사
         - 79D 피처셋 (MediaPipe 36D + RTMP 37D + 6D 크로스 모달 상호작용)
         - 5-Fold StratifiedGroupKFold (피험자 1~9 전원 엄격 격리)
         - 3-Way 앙상블 (CatBoost + LightGBM + XGBoost)
         - 공식 5대 시점(시점 A~E) 적응형 3-State 최적화 (보류율 <= 10.0% 엄격 통제)
"""

import os
import sys
if "D:/" not in sys.path:
    sys.path.insert(0, "D:/")
import json
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix

MP_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_v5_36d_features.npz"
RTMP_2D_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_rtmp_3_features.npz"
FAIR_DATASET_PATH = r"D:\rtmp_fair_dataset.npz"
HYBRID_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_hybrid_1_79d_features.npz"

OUT_MODEL_PATH = r"D:\src\models\second_models\babel_lunge\bl5_hybrid_1.cbm"
OUT_JSON_PATH = r"D:\src\models\second_models\babel_lunge\bl5_hybrid_1_benchmark_results.json"

VIEW_INFO = {
    "view5": {"code": "view5", "name": "시점 A (우측 후방 45°)", "angle": "Right Back 45°"},
    "view2": {"code": "view2", "name": "시점 B (우측 전방 45°)", "angle": "Right Front 45°"},
    "view3": {"code": "view3", "name": "시점 C (정면 0°)", "angle": "Front View 0°"},
    "view4": {"code": "view4", "name": "시점 D (좌측 전방 45°)", "angle": "Left Front 45°"},
    "view1": {"code": "view1", "name": "시점 E (좌측 후방 45°)", "angle": "Left Back 45°"},
}

def load_or_build_hybrid_dataset():
    if os.path.exists(HYBRID_CACHE_PATH):
        print(f"[*] 캐시된 하이브리드 피처셋 로드: {HYBRID_CACHE_PATH}")
        c = np.load(HYBRID_CACHE_PATH, allow_pickle=True)
        return (
            c['X'].astype(np.float32),
            c['y'].astype(int),
            c['subjects'].astype(int),
            c['views'].astype(str),
            list(c['feature_names']),
            list(c['filenames'])
        )

    print("[*] MediaPipe(36D) 및 RTMPose(37D) 피처 결합 및 6D 상호작용 피처 생성 중...")
    d_mp = np.load(MP_CACHE_PATH, allow_pickle=True)
    d_rtmp_fair = np.load(FAIR_DATASET_PATH, allow_pickle=True)
    d_rtmp_2d = np.load(RTMP_2D_CACHE_PATH, allow_pickle=True)

    mp_fnames = list(d_mp['filenames'])
    rtmp_fnames = list(d_rtmp_fair['fnames'])
    rtmp_map = {f: i for i, f in enumerate(rtmp_fnames)}
    rtmp_indices = [rtmp_map[f] for f in mp_fnames]

    y = d_mp['y'].astype(int)
    subjects = d_mp['subjects'].astype(int)
    views = d_mp['views'].astype(str)

    X_mp = d_mp['X'].astype(np.float32)
    X_rtmp = d_rtmp_2d['X'][rtmp_indices].astype(np.float32)

    from src.features.lunge_c5_features_hybrid import (
        compute_cross_interactions,
        LUNGE_C5_HYBRID_79D_NAMES
    )

    cross_interactions = compute_cross_interactions(X_mp, X_rtmp)
    X_hybrid = np.hstack([X_mp, X_rtmp, cross_interactions]).astype(np.float32)

    os.makedirs(os.path.dirname(HYBRID_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        HYBRID_CACHE_PATH,
        X=X_hybrid,
        y=y,
        subjects=subjects,
        views=views,
        feature_names=LUNGE_C5_HYBRID_79D_NAMES,
        filenames=mp_fnames
    )
    print(f"[+] 하이브리드 피처셋 캐시 저장 완료 (Shape: {X_hybrid.shape})")
    return X_hybrid, y, subjects, views, LUNGE_C5_HYBRID_79D_NAMES, mp_fnames

def train_and_eval():
    print("=" * 80)
    print("  [vidio-health-trainer] bl5_hybrid_1 하이브리드 융합 모델 훈련 및 감사 (79D)")
    print("=" * 80)

    X, y, groups, views, feature_names, filenames = load_or_build_hybrid_dataset()
    N = len(y)
    print(f"[*] 총 샘플 수: {N}개 (정상: {np.sum(y == 0)}개, 결함: {np.sum(y == 1)}개)")
    print(f"[*] 하이브리드 피처 차원: {X.shape[1]}D")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_cb = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_xgb = np.zeros(N, dtype=np.float32)

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups), 1):
        val_subs = np.unique(groups[va])
        print(f"  -> Fold {fold}/5 훈련 중... (검증 피험자: {val_subs})")

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

    # Weighted Ensemble (0.5 CB + 0.3 LGB + 0.2 XGB)
    ens = 0.5 * oof_cb + 0.3 * oof_lgb + 0.2 * oof_xgb

    preds_2s = (ens >= 0.5).astype(int)
    f1_2s = f1_score(y, preds_2s)
    auc_total = roc_auc_score(y, ens)
    acc_2s = accuracy_score(y, preds_2s)
    prec_2s = precision_score(y, preds_2s, zero_division=0)
    rec_2s = recall_score(y, preds_2s, zero_division=0)

    # Global 3-State Threshold Optimization (Constraint: Retake <= 10.0%)
    best_f1_global, best_ret_global, best_tau_global = 0, 0, (0.5, 0.5)
    for t_norm in np.arange(0.15, 0.85, 0.005):
        for t_def in np.arange(t_norm, 0.90, 0.005):
            y_pred = np.where(ens >= t_def, 1, np.where(ens < t_norm, 0, 2))
            cov = (y_pred != 2)
            ret = 1.0 - np.mean(cov)
            if ret > 0.1001:
                continue
            if np.sum(cov) > 0:
                f1_val = f1_score(y[cov], y_pred[cov], zero_division=0)
                if f1_val > best_f1_global:
                    best_f1_global = f1_val
                    best_ret_global = ret
                    best_tau_global = (round(float(t_norm), 3), round(float(t_def), 3))

    # View-Adaptive 3-State Optimization (Constraint: Retake <= 10.0% per view)
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
                if ret > 0.1001:
                    continue
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
    overall_prec = float(precision_score(y_cov_all, p_cov_all, zero_division=0))
    overall_rec = float(recall_score(y_cov_all, p_cov_all, zero_division=0))
    overall_acc = float(accuracy_score(y_cov_all, p_cov_all))

    print("\n" + "=" * 80)
    print(f"[+] bl5_hybrid_1 훈련 및 평가 최종 결과:")
    print(f"    - MediaPipe 전용 (bl5_v5) 3-State F1: 73.11% (보류율: 9.90%)")
    print(f"    - RTMP 전용 (bl5_rtmp_3) 3-State F1 : 67.86% (보류율: 11.71%)")
    print(f"    - 하이브리드 (bl5_hybrid_1) 3-State F1: {overall_adaptive_f1*100:.2f}% (보류율: {overall_adaptive_ret*100:.2f}%)")
    print(f"    - 2-State Binary F1: {f1_2s*100:.2f}% | AUC = {auc_total:.4f}")
    print(f"    - 순수 성능 향상폭(ΔF1 vs MP v5): +{overall_adaptive_f1*100 - 73.11:.2f}%p")
    print(f"    - 순수 성능 향상폭(ΔF1 vs RTMP rtmp_3): +{overall_adaptive_f1*100 - 67.86:.2f}%p")
    print("=" * 80)

    for v_code, stat in view_audit.items():
        print(f"    * {stat['view_name']}: F1 = {stat['f1']*100:.2f}%, AUC = {stat['auc']:.4f}, 보류율 = {stat['rejection_rate']*100:.2f}%, tau = ({stat['tau_norm']}, {stat['tau_def']})")

    results_data = {
        "model_name": "bl5_hybrid_1",
        "description": "MediaPipe (36D 3D) + RTMPose (37D 2D) + Cross-Modal (6D) Feature-level Fusion",
        "benchmark_comparison": {
            "bl5_v5_mediapipe": {"2state_f1": 0.6980, "3state_f1": 0.7311, "rejection": 0.0990, "auc": 0.8335},
            "bl5_rtmp_3": {"2state_f1": 0.5958, "3state_f1": 0.6786, "rejection": 0.1171, "auc": 0.8004},
            "bl5_hybrid_1": {"2state_f1": float(f1_2s), "3state_f1": overall_adaptive_f1, "rejection": overall_adaptive_ret, "auc": float(auc_total)}
        },
        "overall_performance": {
            "2state_f1": float(f1_2s),
            "2state_precision": float(prec_2s),
            "2state_recall": float(rec_2s),
            "2state_accuracy": float(acc_2s),
            "roc_auc": float(auc_total),
            "3state_f1_global": float(best_f1_global),
            "3state_rejection_global": float(best_ret_global),
            "3state_tau_global": list(best_tau_global),
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
        "view_audit": view_audit,
        "feature_names": feature_names
    }

    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"[+] 벤치마크 JSON 저장 완료: {OUT_JSON_PATH}")

    # Retrain final CatBoost production model
    print("[*] 전체 1,081 데이터셋 기준 최종 bl5_hybrid_1 프로덕션 모델 적합 중...")
    prod_model = CatBoostClassifier(
        iterations=450, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
        scale_pos_weight=1.2, verbose=0, random_seed=42, thread_count=4
    )
    prod_model.fit(X, y)
    os.makedirs(os.path.dirname(OUT_MODEL_PATH), exist_ok=True)
    prod_model.save_model(OUT_MODEL_PATH)
    print(f"[+] 최종 프로덕션 모델 저장 완료: {OUT_MODEL_PATH}")

if __name__ == "__main__":
    train_and_eval()
