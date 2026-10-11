# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: audit_d3_features.py
Purpose: d3 v5.0 모델에 투입되는 40D 피처 전수 감사 (Feature Audit)
         1. 피처 간 상호 상관관계 분석 (|r| >= 0.85 고상관 중복 피처 쌍 식별)
         2. CatBoost + LightGBM + Random Forest 피처 중요도(Feature Importance) 전수 산출
         3. 5-Fold 피험자 격리 순열 중요도 (Permutation Feature Importance, PFI) 분석
         4. 성능 저해/노이즈 후보 피처(Negative/Zero PFI) 식별 및 소거(Ablation) 영향도 실측
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d3_model_v4 import FEATURE_NAMES_40_D3

FEATURE_CACHE_FILE = os.path.join(CURRENT_DIR, "d3_v4_40d_features_cache.npz")
OUT_AUDIT_JSON = os.path.join(CURRENT_DIR, "d3_feature_audit_results.json")


def load_cached_data():
    data = np.load(FEATURE_CACHE_FILE, allow_pickle=True)
    X = data['X']
    y = data['y']
    subjects = data['subjects']
    views = data['views']
    return X, y, subjects, views


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
    """CatBoost, LightGBM, Random Forest 모델의 피처 중요도 전수 산출"""
    # 1. CatBoost
    cb = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, auto_class_weights='Balanced', verbose=False, random_seed=42)
    cb.fit(X, y)
    cb_imp = cb.get_feature_importance()
    
    # 2. LightGBM
    pos_w = float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))
    ds = lgb.Dataset(X, label=y)
    params = {'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42}
    lgb_m = lgb.train(params, ds, num_boost_round=220)
    lgb_imp = lgb_m.feature_importance(importance_type='gain')
    
    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=180, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    
    # 정규화 (합이 100이 되도록)
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


def compute_permutation_importance(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, feature_names: List[str]):
    """5-Fold 피험자 완전 격리 순열 중요도 (Permutation Feature Importance, PFI)"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    # 베이스라인 OOF 예측 및 성능
    oof_baseline_probs = np.zeros(len(y))
    models = []
    val_indices = []
    
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        cb = CatBoostClassifier(iterations=250, depth=4, learning_rate=0.03, auto_class_weights='Balanced', verbose=False, random_seed=42 + fold)
        cb.fit(X[tr_idx], y[tr_idx])
        probs = cb.predict_proba(X[va_idx])[:, 1]
        oof_baseline_probs[va_idx] = probs
        models.append(cb)
        val_indices.append(va_idx)
        
    base_auc = roc_auc_score(y, oof_baseline_probs)
    base_f1 = f1_score(y, np.where(oof_baseline_probs >= 0.50, 1, 0), zero_division=0)
    
    pfi_results = []
    np.random.seed(42)
    
    for i, name in enumerate(feature_names):
        oof_shuffled_probs = np.zeros(len(y))
        
        for fold, va_idx in enumerate(val_indices):
            X_va_shuffled = X[va_idx].copy()
            # 해당 피처만 셔플
            X_va_shuffled[:, i] = np.random.permutation(X_va_shuffled[:, i])
            oof_shuffled_probs[va_idx] = models[fold].predict_proba(X_va_shuffled)[:, 1]
            
        shuff_auc = roc_auc_score(y, oof_shuffled_probs)
        shuff_f1 = f1_score(y, np.where(oof_shuffled_probs >= 0.50, 1, 0), zero_division=0)
        
        # PFI는 베이스라인 성능에서 셔플 후 성능을 뺀 값 (값이 클수록 필수 피처, 음수이면 노이즈/방해 피처)
        delta_auc = float(base_auc - shuff_auc)
        delta_f1 = float(base_f1 - shuff_f1)
        
        pfi_results.append({
            "index": i,
            "name": name,
            "delta_auc": delta_auc,
            "delta_f1": delta_f1,
            "is_harmful_or_noise": bool(delta_auc <= 0.0 and delta_f1 <= 0.0)
        })
        
    pfi_results.sort(key=lambda x: x["delta_auc"], reverse=True)
    return base_auc, base_f1, pfi_results


def run_ablation_experiment(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, harmful_indices: List[int], feature_names: List[str]):
    """방해/노이즈 피처 소거(Ablation) 시 5-Fold 피험자 격리 성능 실측"""
    keep_indices = [i for i in range(X.shape[1]) if i not in harmful_indices]
    X_reduced = X[:, keep_indices]
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs_reduced = np.zeros(len(y))
    
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_reduced, y, groups=subjects)):
        cb = CatBoostClassifier(iterations=250, depth=4, learning_rate=0.03, auto_class_weights='Balanced', verbose=False, random_seed=42 + fold)
        cb.fit(X_reduced[tr_idx], y[tr_idx])
        oof_probs_reduced[va_idx] = cb.predict_proba(X_reduced[va_idx])[:, 1]
        
    auc_red = roc_auc_score(y, oof_probs_reduced)
    f1_red = f1_score(y, np.where(oof_probs_reduced >= 0.50, 1, 0), zero_division=0)
    acc_red = accuracy_score(y, np.where(oof_probs_reduced >= 0.50, 1, 0))
    
    return {
        "original_dim": X.shape[1],
        "reduced_dim": len(keep_indices),
        "removed_indices": harmful_indices,
        "removed_names": [feature_names[i] for i in harmful_indices],
        "reduced_auc": float(auc_red),
        "reduced_f1": float(f1_red),
        "reduced_acc": float(acc_red)
    }


def main():
    X, y, subjects, views = load_cached_data()
    feature_names = FEATURE_NAMES_40_D3
    
    print("\n" + "="*80)
    print(">>> 1. 피처 간 상호 상관관계 분석 (|r| >= 0.85 고상관 중복 쌍)")
    print("="*80)
    corr_matrix, high_corr_pairs = audit_correlations(X, feature_names, threshold=0.85)
    for p in high_corr_pairs:
        print(f"  - [{p['feat1_name']}] <---> [{p['feat2_name']}] : r = {p['correlation']:+.4f}")
        
    print("\n" + "="*80)
    print(">>> 2. 트리 앙상블 피처 중요도 분석 (상위 10개 vs 하위 10개)")
    print("="*80)
    importances = compute_tree_feature_importances(X, y, feature_names)
    print("  [상위 10개 핵심 피처]")
    for r in importances[:10]:
        print(f"    - [{r['index']:02d}] {r['name']:<35} : 중요도 {r['ensemble_imp']:5.2f}% (CB:{r['catboost_imp']:4.1f}%, LGB:{r['lightgbm_imp']:4.1f}%, RF:{r['rf_imp']:4.1f}%)")
        
    print("\n  [하위 10개 비기여 후보 피처]")
    for r in importances[-10:]:
        print(f"    - [{r['index']:02d}] {r['name']:<35} : 중요도 {r['ensemble_imp']:5.2f}% (CB:{r['catboost_imp']:4.1f}%, LGB:{r['lightgbm_imp']:4.1f}%, RF:{r['rf_imp']:4.1f}%)")

    print("\n" + "="*80)
    print(">>> 3. 5-Fold 피험자 완전 격리 순열 중요도 (PFI) 분석")
    print("="*80)
    base_auc, base_f1, pfi_results = compute_permutation_importance(X, y, subjects, feature_names)
    print(f"  * 베이스라인 성능 : AUC = {base_auc:.4f}, F1 = {base_f1*100:.2f}%")
    
    harmful_or_noise = [p for p in pfi_results if p["is_harmful_or_noise"]]
    print(f"\n  [★ 성능 저해 또는 무기여 노이즈 피처 ({len(harmful_or_noise)}개 식별)]")
    for p in harmful_or_noise:
        print(f"    - [{p['index']:02d}] {p['name']:<35} : delta_AUC = {p['delta_auc']:+.4f}, delta_F1 = {p['delta_f1']:+.4f}")

    harmful_indices = [p["index"] for p in harmful_or_noise]
    ablation_res = run_ablation_experiment(X, y, subjects, harmful_indices, feature_names)
    print("\n" + "="*80)
    print(">>> 4. 저해/노이즈 피처 전면 소거(Ablation) 실측 결과")
    print("="*80)
    print(f"  - 원본 피처 차원: {ablation_res['original_dim']}D -> 정예화 차원: {ablation_res['reduced_dim']}D (제거: {len(harmful_indices)}개)")
    print(f"  - 제거 전 베이스라인: AUC = {base_auc:.4f}, F1 = {base_f1*100:.2f}%")
    print(f"  - 제거 후 정예화성능: AUC = {ablation_res['reduced_auc']:.4f} (변화: {ablation_res['reduced_auc']-base_auc:+.4f})")
    print(f"                        F1  = {ablation_res['reduced_f1']*100:.2f}% (변화: {(ablation_res['reduced_f1']-base_f1)*100:+.2f}%p)")
    print(f"                        Acc = {ablation_res['reduced_acc']*100:.2f}%")

    out_data = {
        "high_corr_pairs": high_corr_pairs,
        "feature_importances": importances,
        "base_metrics": {"auc": base_auc, "f1": base_f1},
        "pfi_results": pfi_results,
        "harmful_features": harmful_or_noise,
        "ablation_results": ablation_res
    }
    
    with open(OUT_AUDIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 피처 전수 감사 결과 JSON 저장 완료: {OUT_AUDIT_JSON}")


if __name__ == "__main__":
    main()
