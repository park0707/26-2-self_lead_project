# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier

import sys
CURRENT_DIR = r"D:\src\models\second_models\babel_lunge"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
cache_file = os.path.join(CURRENT_DIR, "lunge_c2_features_hybrid_2_cache.npz")

c = np.load(cache_file, allow_pickle=True)
X_full = c["X"]
y = c["y"]
subjects = c["subjects"]
views = c["views"]

# Load selected top 20 features from bl2_hybrid_pruned
results_pruned = os.path.join(CURRENT_DIR, "bl2_hybrid_pruned_benchmark_results.json")
with open(results_pruned, "r", encoding="utf-8") as f:
    pruned_data = json.load(f)
selected_feature_names = pruned_data["selected_features"]

from lunge_c2_features_v6 import FEATURE_NAMES_18D
from lunge_c2_features_hybrid_2 import FEATURE_NAMES_FRONTAL
FEATURE_NAMES_46D = (
    [f"MP_{f}" for f in FEATURE_NAMES_18D] + 
    [f"RTMP_{f}" for f in FEATURE_NAMES_18D] + 
    [f"MP_{f}" for f in FEATURE_NAMES_FRONTAL] + 
    [f"RTMP_{f}" for f in FEATURE_NAMES_FRONTAL]
)

selected_indices = [FEATURE_NAMES_46D.index(name) for name in selected_feature_names]
X = X_full[:, selected_indices]

print(f"[*] Training 5-Fold SGKF on {len(selected_indices)}D features to get clean OOF predictions...")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(y), dtype=np.float32)

n_pos = np.sum(y == 1)
n_neg = np.sum(y == 0)
cost_weight = (float(n_neg) / float(n_pos)) * 1.35

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=160, max_depth=4, num_leaves=13,
        learning_rate=0.033, min_child_samples=15,
        colsample_bytree=0.75, subsample=0.85,
        reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=cost_weight,
        random_state=42 + fold, verbose=-1, n_jobs=-1
    )
    cat_clf = CatBoostClassifier(
        iterations=220, depth=5, learning_rate=0.030,
        l2_leaf_reg=4.0, auto_class_weights='Balanced',
        random_seed=42 + fold, verbose=0, thread_count=-1
    )
    lgb_clf.fit(X_train, y_train)
    cat_clf.fit(X_train, y_train)

    oof_probs[val_idx] = 0.50 * lgb_clf.predict_proba(X_val)[:, 1] + 0.50 * cat_clf.predict_proba(X_val)[:, 1]

print("\n" + "=" * 80)
print("  [cl2_v5 방식: 시점별 적응형 임계값 최적화 (Per-View Adaptive Optimization)]")
print("=" * 80)

view_order = ["view3", "view2", "view4", "view1", "view5"]
view_alias = {
    "view3": "시점 C (정면 0°)",
    "view2": "시점 B (우측 전방 45°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)",
    "view5": "시점 A (우측 후방 45°)"
}

# 1. Per-View 2-State Optimization
per_view_opt_th = {}
per_view_2s_res = {}
preds_2s_overall = np.zeros(len(y), dtype=int)

for v in view_order:
    mask = (views == v)
    y_v = y[mask]
    p_v = oof_probs[mask]
    
    best_th = 0.5
    best_f1 = 0.0
    for th in np.arange(0.20, 0.70, 0.005):
        s = f1_score(y_v, (p_v >= th).astype(int), zero_division=0)
        if s > best_f1:
            best_f1 = s
            best_th = th
            
    per_view_opt_th[v] = float(best_th)
    p_dec = (p_v >= best_th).astype(int)
    preds_2s_overall[mask] = p_dec
    
    acc = accuracy_score(y_v, p_dec)
    prec = precision_score(y_v, p_dec, zero_division=0)
    rec = recall_score(y_v, p_dec, zero_division=0)
    auc = roc_auc_score(y_v, p_v)
    
    per_view_2s_res[v] = {"f1": best_f1, "acc": acc, "prec": prec, "rec": rec, "auc": auc, "th": best_th}
    print(f"[*] 2-State {view_alias[v]:<25} | 최적 th={best_th:.3f} | F1: {best_f1*100:.2f}% | Acc: {acc*100:.2f}% | AUC: {auc:.4f}")

overall_2s_f1 = f1_score(y, preds_2s_overall, zero_division=0)
overall_2s_acc = accuracy_score(y, preds_2s_overall)
print(f"\n[+] cl2 방식 시점별 2-State 통합: F1 = {overall_2s_f1*100:.2f}%, Acc = {overall_2s_acc*100:.2f}%")

# 2. Per-View 3-State Optimization (cl2_v5 exact method)
print("\n" + "=" * 80)
print("  [cl2_v5 방식: 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10%)]")
print("=" * 80)

mask_3s_overall = np.zeros(len(y), dtype=bool)
preds_3s_overall = np.zeros(len(y), dtype=int)
per_view_3s_res = {}

for v in view_order:
    mask = (views == v)
    y_v = y[mask]
    p_v = oof_probs[mask]
    th_star = per_view_opt_th[v]
    
    best_3s_f1 = best_f1
    best_cfg = (th_star - 0.001, th_star + 0.001, 0.0, 1.0)
    
    for tau_c in np.arange(0.20, 0.60, 0.01):
        for tau_d in np.arange(tau_c + 0.01, 0.70, 0.01):
            m = (p_v <= tau_c) | (p_v >= tau_d)
            unc = 1.0 - np.mean(m)
            if unc > 0.10: # Max 10% retake
                continue
            s = f1_score(y_v[m], (p_v[m] >= tau_d).astype(int), zero_division=0)
            c_f1 = f1_score(1 - y_v[m], (p_v[m] < tau_c).astype(int), zero_division=0)
            if c_f1 >= 0.50 and s > best_3s_f1:
                best_3s_f1 = s
                best_cfg = (float(tau_c), float(tau_d), float(unc), float(np.mean(m)))

                
    m_v = mask & ((oof_probs <= best_cfg[0]) | (oof_probs >= best_cfg[1]))
    mask_3s_overall |= m_v
    preds_3s_overall[m_v] = (oof_probs[m_v] >= best_cfg[1]).astype(int)
    
    per_view_3s_res[v] = {
        "tau_clean": best_cfg[0], "tau_defect": best_cfg[1],
        "effective_f1": best_3s_f1, "coverage": best_cfg[3], "retake": best_cfg[2]
    }
    print(f"[*] 3-State {view_alias[v]:<25} | tau_c={best_cfg[0]:.3f}, tau_d={best_cfg[1]:.3f} | Defect F1: {best_3s_f1*100:.2f}% | Retake: {best_cfg[2]*100:.1f}%")

overall_3s_f1 = f1_score(y[mask_3s_overall], preds_3s_overall[mask_3s_overall], zero_division=0)
overall_3s_clean_f1 = f1_score(1 - y[mask_3s_overall], 1 - preds_3s_overall[mask_3s_overall], zero_division=0)
overall_3s_acc = accuracy_score(y[mask_3s_overall], preds_3s_overall[mask_3s_overall])
overall_3s_cov = np.mean(mask_3s_overall)

print(f"\n[+] cl2_v5 방식 시점별 3-State 통합 결과:")
print(f"    Selective Defect F1 : {overall_3s_f1*100:.2f}%")
print(f"    Selective Clean F1  : {overall_3s_clean_f1*100:.2f}%")
print(f"    판정 정확도 (Acc)    : {overall_3s_acc*100:.2f}%")
print(f"    전체 커버리지 (Cov)  : {overall_3s_cov*100:.2f}% (보류율: {(1-overall_3s_cov)*100:.2f}%)")
