# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: audit_d3_v5_3_features.py
Purpose: v5.3 32D 피처셋 전수 감사 및 정밀 분석
         1. 피처 간 상호 상관관계 분석 (|r| >= 0.85 고상관 중복 피처 식별)
         2. CatBoost + LightGBM + Random Forest 모델 피처 중요도(Feature Importance) 전수 산출
         3. 5-Fold 피험자 완전 격리 순열 중요도 (PFI: Permutation Feature Importance) 분석
         4. F1-Score를 저해하는 음수/노이즈 피처(Negative/Zero PFI) 식별
"""

import os
import sys

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from src.models.second_models.babel_deadlift.train_and_eval_d3_v5_3 import prepare_v5_3_clean_dataset
from src.models.second_models.babel_deadlift.d3_model_v5_3 import FEATURE_NAMES_32_D3

OUT_AUDIT_JSON = os.path.join(CURRENT_DIR, "d3_v5_3_feature_audit_results.json")


def audit_correlations(X: np.ndarray, feature_names: List[str], threshold: float = 0.85):
    """피처 간 상관계수 분석 (|r| >= threshold)"""
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr(method='pearson')

    high_corr_pairs = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                high_corr_pairs.append({
                    "feat1_idx": i,
                    "feat1_name": feature_names[i],
                    "feat2_idx": j,
                    "feat2_name": feature_names[j],
                    "correlation": float(val)
                })

    high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return corr_matrix, high_corr_pairs


def compute_tree_feature_importances(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """CatBoost, LightGBM, Random Forest 모델의 피처 중요도 산출"""
    pos_w = float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))

    cb = CatBoostClassifier(iterations=350, depth=4, learning_rate=0.03, auto_class_weights='Balanced', verbose=False, random_seed=42)
    cb.fit(X, y)
    cb_imp = cb.get_feature_importance()

    ds = lgb.Dataset(X, label=y)
    params = {'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42}
    lgb_m = lgb.train(params, ds, num_boost_round=230)
    lgb_imp = lgb_m.feature_importance(importance_type='gain')

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    cb_norm = (cb_imp / np.sum(cb_imp)) * 100.0
    lgb_norm = (lgb_imp / (np.sum(lgb_imp) + 1e-9)) * 100.0
    rf_norm = (rf_imp / np.sum(rf_imp)) * 100.0

    ensemble_imp = 0.50 * cb_norm + 0.30 * lgb_norm + 0.20 * rf_norm

    records = []
    for i, name in enumerate(feature_names):
        records.append({
            "index": i,
            "name": name,
            "catboost_imp": float(cb_norm[i]),
            "lightgbm_imp": float(lgb_norm[i]),
            "rf_imp": float(rf_norm[i]),
            "ensemble_imp": float(ensemble_imp[i])
        })

    records.sort(key=lambda x: x["ensemble_imp"], reverse=True)
    return records


def compute_permutation_importance(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, feature_names: List[str], n_repeats: int = 5):
    """5-Fold 피험자 완전 격리 순열 중요도 (Permutation Feature Importance, PFI)"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_baseline_probs = np.zeros(len(y), dtype=np.float32)
    models = []

    print("[*] 5-Fold 베이스라인 모델 학습 중...")
    for fold, (tr, val) in enumerate(sgkf.split(X, y, groups=subjects)):
        pos_w = float(np.sum(y[tr] == 0)) / max(1, float(np.sum(y[tr] == 1)))

        cb = CatBoostClassifier(iterations=350, depth=4, learning_rate=0.03, auto_class_weights='Balanced', verbose=False, random_seed=42+fold)
        cb.fit(X[tr], y[tr])

        ds = lgb.Dataset(X[tr], label=y[tr])
        params = {'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42+fold}
        lgb_m = lgb.train(params, ds, num_boost_round=230)

        rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42+fold, n_jobs=-1)
        rf.fit(X[tr], y[tr])

        p_cb = cb.predict_proba(X[val])[:, 1]
        p_lgb = lgb_m.predict(X[val])
        p_rf = rf.predict_proba(X[val])[:, 1]

        oof_baseline_probs[val] = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf
        models.append((cb, lgb_m, rf))

    base_f1 = f1_score(y, (oof_baseline_probs >= 0.5).astype(int), zero_division=0)
    base_auc = roc_auc_score(y, oof_baseline_probs)
    print(f"[+] 베이스라인 Tree OOF Raw F1(0.5): {base_f1*100:.2f}%, AUC: {base_auc:.4f}")

    print("[*] 32개 피처별 5-Fold 순열 중요도(PFI) 측정 개시...")
    pfi_results = []
    n_features = X.shape[1]

    for f_idx in range(n_features):
        f_name = feature_names[f_idx]
        f1_drops = []
        auc_drops = []

        for rep in range(n_repeats):
            oof_perm_probs = np.zeros(len(y), dtype=np.float32)

            for fold, (tr, val) in enumerate(sgkf.split(X, y, groups=subjects)):
                cb, lgb_m, rf = models[fold]

                X_val_perm = X[val].copy()
                np.random.seed(42 + f_idx * 100 + rep * 10 + fold)
                X_val_perm[:, f_idx] = np.random.permutation(X_val_perm[:, f_idx])

                p_cb = cb.predict_proba(X_val_perm)[:, 1]
                p_lgb = lgb_m.predict(X_val_perm)
                p_rf = rf.predict_proba(X_val_perm)[:, 1]

                oof_perm_probs[val] = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf

            perm_f1 = f1_score(y, (oof_perm_probs >= 0.5).astype(int), zero_division=0)
            perm_auc = roc_auc_score(y, oof_perm_probs)

            f1_drops.append(base_f1 - perm_f1)
            auc_drops.append(base_auc - perm_auc)

        mean_f1_drop = float(np.mean(f1_drops))
        std_f1_drop = float(np.std(f1_drops))
        mean_auc_drop = float(np.mean(auc_drops))

        pfi_results.append({
            "index": f_idx,
            "name": f_name,
            "pfi_f1_drop_mean": mean_f1_drop,
            "pfi_f1_drop_std": std_f1_drop,
            "pfi_auc_drop_mean": mean_auc_drop,
            "is_negative": mean_f1_drop < -0.002,
            "is_zero": abs(mean_f1_drop) <= 0.002
        })

    pfi_results.sort(key=lambda x: x["pfi_f1_drop_mean"], reverse=True)
    return base_f1, base_auc, pfi_results


def run_full_audit():
    print("=" * 80)
    print("vidio-health-trainer 바벨 데드리프트 d3 v5.3 32D 피처 전수 감사 (Feature Audit)")
    print("목표: F1-Score를 저해하는 노이즈 피처 및 고상관(|r| >= 0.85) 중복 피처 전수 식별")
    print("=" * 80)

    X_seq_57, X_feats_32, y, subjects, views, clip_names = prepare_v5_3_clean_dataset()
    feature_names = FEATURE_NAMES_32_D3

    # 1. 상관관계 분석
    print("\n[1] 피처 간 상호 상관관계 분석 (|r| >= 0.85)...")
    corr_matrix, high_corr_pairs = audit_correlations(X_feats_32, feature_names, threshold=0.85)
    print(f"  - 총 고상관(|r| >= 0.85) 피처 쌍: {len(high_corr_pairs)}쌍 검출")
    for p in high_corr_pairs[:10]:
        print(f"    * [{p['feat1_idx']}] {p['feat1_name']} <-> [{p['feat2_idx']}] {p['feat2_name']} : r = {p['correlation']:.4f}")

    # 2. 트리 모델 피처 중요도 분석
    print("\n[2] 트리 앙상블(CatBoost + LightGBM + RF) 피처 중요도 산출...")
    tree_importances = compute_tree_feature_importances(X_feats_32, y, feature_names)
    print("  - 상위 5대 핵심 피처:")
    for rec in tree_importances[:5]:
        print(f"    * [{rec['index']:02d}] {rec['name']:30s} : 중요도 {rec['ensemble_imp']:.2f}%")
    print("  - 하위 5대 미미 피처:")
    for rec in tree_importances[-5:]:
        print(f"    * [{rec['index']:02d}] {rec['name']:30s} : 중요도 {rec['ensemble_imp']:.2f}%")

    # 3. 5-Fold 순열 중요도(PFI) 분석
    print("\n[3] 5-Fold 피험자 완전 격리 순열 중요도(PFI) 분석...")
    base_f1, base_auc, pfi_results = compute_permutation_importance(X_feats_32, y, subjects, feature_names)

    neg_feats = [r for r in pfi_results if r["is_negative"]]
    zero_feats = [r for r in pfi_results if r["is_zero"]]

    print(f"\n[★ 피처 감사 진단 결과 ★]")
    print(f"  - 긍정 기여 피처: {len(pfi_results) - len(neg_feats) - len(zero_feats)}개")
    print(f"  - F1 저해 음수(-) 노이즈 피처: {len(neg_feats)}개")
    for r in neg_feats:
        print(f"    ! [제거 후보] [{r['index']:02d}] {r['name']:30s} : PFI F1-Drop = {r['pfi_f1_drop_mean']*100:.3f}% (순열 셔플 시 오히려 F1 상승!)")

    print(f"  - 무기여/중복(Zero) 피처: {len(zero_feats)}개")
    for r in zero_feats:
        print(f"    ~ [정리 후보] [{r['index']:02d}] {r['name']:30s} : PFI F1-Drop = {r['pfi_f1_drop_mean']*100:.3f}%")

    audit_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_features": len(feature_names),
        "high_corr_pairs": high_corr_pairs,
        "tree_importances": tree_importances,
        "pfi_results": pfi_results,
        "negative_features": neg_feats,
        "zero_features": zero_feats
    }

    with open(OUT_AUDIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 피처 감사 결과 JSON 저장 완료: {OUT_AUDIT_JSON}")

if __name__ == '__main__':
    run_full_audit()
