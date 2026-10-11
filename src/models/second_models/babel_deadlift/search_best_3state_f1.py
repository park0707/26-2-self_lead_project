# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: search_best_3state_f1.py
Purpose: d2 v3.0 모델의 3-State 선택적 판정에서 'F1-score'를 극대화하는 최적 윈도우 전수 그리드 서치
"""

import os
import sys
import json
import glob
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/')

from src.models.second_models.babel_deadlift.build_and_eval_d2_v3_integrated import (
    extract_dual_geometry_features, CACHE_DIR, LABEL_JSON_DIR
)

files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))

X_list = []
y_list = []
subs_list = []
views_list = []

for p in files:
    base = os.path.basename(p).replace(".npz", "")
    parts = base.split("_")
    vn = parts[-1]
    folder = "_".join(parts[:-1])

    sub = 1
    for i in range(1, 10):
        if f"피험자{i}" in folder:
            sub = i
            break

    with open(os.path.join(LABEL_JSON_DIR, f"{folder}.json"), "r", encoding="utf-8", errors="ignore") as fp:
        d = json.load(fp)
    conds = {c["condition"]: c["value"] for c in d.get("type_info", {}).get("conditions", [])}
    y_val = 0 if conds.get("발과 무릎의 방향 일치", True) else 1

    kps = np.load(p)["kps_33_3d_64f"]
    f12 = extract_dual_geometry_features(kps)

    X_list.append(f12)
    y_list.append(y_val)
    subs_list.append(sub)
    views_list.append(vn)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
subs = np.array(subs_list, dtype=np.int32)
views = np.array(views_list)

# 1. 1차 OOF 확률 추정 (Cleanlab용)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
init_oof_probs = np.zeros((len(y), 2), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    clf_lgb = lgb.LGBMClassifier(
        n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=14,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=2.2, random_state=42 + fold, verbose=-1, n_jobs=1
    )
    clf_lgb.fit(X_tr, y_tr)
    p_l = clf_lgb.predict_proba(X_val)

    clf_cat = CatBoostClassifier(
        iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
        auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1
    )
    clf_cat.fit(X_tr, y_tr)
    p_c = clf_cat.predict_proba(X_val)

    init_oof_probs[val_idx] = 0.5 * p_l + 0.5 * p_c

# Cleanlab 프루닝
from cleanlab.filter import find_label_issues
label_issues = find_label_issues(
    labels=y, pred_probs=init_oof_probs, return_indices_ranked_by='self_confidence', n_jobs=1
)
prune_k = min(len(label_issues), 60)
clean_mask = np.ones(len(y), dtype=bool)
clean_mask[label_issues[:prune_k]] = False

# 2. 정제 학습셋 기반 5-Fold 최종 앙상블 재학습
final_oof_preds = np.zeros(len(y), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
    clean_tr_idx = [i for i in train_idx if clean_mask[i]]
    X_tr, y_tr = X[clean_tr_idx], y[clean_tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    clf_lgb = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=4, num_leaves=14,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=2.0, random_state=42 + fold, verbose=-1, n_jobs=1
    )
    clf_lgb.fit(X_tr, y_tr)
    p_l = clf_lgb.predict_proba(X_val)[:, 1]

    clf_cat = CatBoostClassifier(
        iterations=170, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
        auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1
    )
    clf_cat.fit(X_tr, y_tr)
    p_c = clf_cat.predict_proba(X_val)[:, 1]

    final_oof_preds[val_idx] = 0.5 * p_l + 0.5 * p_c

# 3. 3-State F1-score 극대화 전수 그리드 서치
# tau_normal in [0.10 ~ 0.35], tau_defect in [0.15 ~ 0.45]
print("=" * 110)
print("  [vidio-health-trainer] d2 v3.0 3-State 'F1-Score 극대화' 전수 그리드 서치 (Coverage >= 70%)")
print("=" * 110)

grid_results = []

for tn in np.arange(0.12, 0.32, 0.01):
    for td in np.arange(tn + 0.01, 0.38, 0.01):
        decided = (final_oof_preds <= tn) | (final_oof_preds >= td)
        cov = float(np.mean(decided) * 100)
        if cov < 70.0:  # 최소 70% 이상 커버리지 보장
            continue

        y_dec = y[decided]
        p_dec = (final_oof_preds[decided] >= td).astype(int)

        if len(np.unique(y_dec)) < 2:
            continue

        f1 = float(f1_score(y_dec, p_dec, zero_division=0) * 100)
        acc = float(accuracy_score(y_dec, p_dec) * 100)
        prec = float(precision_score(y_dec, p_dec, zero_division=0) * 100)
        rec = float(recall_score(y_dec, p_dec, zero_division=0) * 100)
        cm = confusion_matrix(y_dec, p_dec)

        grid_results.append({
            'tn': round(float(tn), 2),
            'td': round(float(td), 2),
            'f1': f1,
            'acc': acc,
            'prec': prec,
            'rec': rec,
            'cov': cov,
            'decline': round(100.0 - cov, 1),
            'tp': int(cm[1, 1]) if len(cm) > 1 else 0,
            'fp': int(cm[0, 1]) if len(cm) > 1 else 0,
            'fn': int(cm[1, 0]) if len(cm) > 1 else 0,
            'tn_cnt': int(cm[0, 0]) if len(cm) > 1 else 0
        })

# F1-Score 기준 내림차순 정렬
grid_results_sorted = sorted(grid_results, key=lambda x: x['f1'], reverse=True)

print(f"{'순위':<4} | {'3-State 윈도우':<18} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Accuracy':<10} | {'Coverage':<10} | {'TP/FP/FN/TN'}")
print("-" * 110)

for rank, r in enumerate(grid_results_sorted[:15], 1):
    win_str = f"[{r['tn']:.2f}, {r['td']:.2f}]"
    cm_str = f"{r['tp']}/{r['fp']}/{r['fn']}/{r['tn_cnt']}"
    print(f" {rank:<3} | {win_str:<18} | {r['f1']:<10.2f}% | {r['prec']:<10.2f}% | {r['rec']:<10.2f}% | {r['acc']:<10.2f}% | {r['cov']:<10.1f}% | {cm_str}")

print("=" * 110)

# 시점별 3-State F1 (최고 F1 윈도우 기준)
best_r = grid_results_sorted[0]
print(f"\n[★ 3-State 최고 F1 윈도우 [{best_r['tn']:.2f}, {best_r['td']:.2f}] 기준 5대 공식 시점별 3-State F1-Score]:")
VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

print(f"{'공식 시점 명칭':<24} | {'코드':<6} | {'3-State F1':<12} | {'Precision':<10} | {'Recall':<10} | {'Coverage'}")
print("-" * 80)
for v_code, v_alias in VIEW_ALIAS_MAP.items():
    v_mask = (views == v_code)
    y_v = y[v_mask]
    prob_v = final_oof_preds[v_mask]

    dec_v = (prob_v <= best_r['tn']) | (prob_v >= best_r['td'])
    y_v_dec = y_v[dec_v]
    p_v_dec = (prob_v[dec_v] >= best_r['td']).astype(int)

    f1_v = f1_score(y_v_dec, p_v_dec, zero_division=0) * 100
    prec_v = precision_score(y_v_dec, p_v_dec, zero_division=0) * 100
    rec_v = recall_score(y_v_dec, p_v_dec, zero_division=0) * 100
    cov_v = np.mean(dec_v) * 100
    print(f"{v_alias:<24} | {v_code:<6} | {f1_v:<12.2f}% | {prec_v:<10.2f}% | {rec_v:<10.2f}% | {cov_v:.1f}%")
print("=" * 80)
