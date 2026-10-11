# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: optimize_diet_feature_selection.py
Purpose: d2 v2.0 공식 파라미터 환경에서 피처 다이어트 세부 조합별 정밀 5-Fold OOF 성능 비교
"""

import os
import sys
import json
import glob
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/')

from src.models.second_models.babel_deadlift.d2_model_v2 import extract_d2_v2_features_from_seq, FEATURE_NAMES_30_D2

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
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
    X_list.append(extract_d2_v2_features_from_seq(kps))
    y_list.append(y_val)
    subs_list.append(sub)
    views_list.append(vn)

X_all = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
subs = np.array(subs_list, dtype=np.int32)
views = np.array(views_list)

combos = {
    "V2.0 Baseline (30D)": list(range(30)),
    "V1.0 Baseline (22D)": list(range(22)),
    "Diet-9D (p<0.05 Top 9)": [24, 3, 4, 1, 8, 7, 25, 26, 6],
    "Diet-10D (Top 9 + ShankTilt)": [24, 3, 4, 1, 8, 7, 25, 26, 6, 28],
    "Diet-11D (Top 10 + StandingKA)": [24, 3, 4, 1, 8, 7, 25, 26, 6, 28, 2],
    "Diet-12D (Top 11 + ChronicValgus)": [24, 3, 4, 1, 8, 7, 25, 26, 6, 28, 2, 5],
    "Diet-13D (Top 12 + NadirKA)": [24, 3, 4, 1, 8, 7, 25, 26, 6, 28, 2, 5, 0],
    "Diet-14D (Top 13 + InwardSync)": [24, 3, 4, 1, 8, 7, 25, 26, 6, 28, 2, 5, 0, 29],
}

def eval_combo(feat_indices):
    X = X_all[:, feat_indices]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(len(y), dtype=np.float32)
    oof_preds_cat = np.zeros(len(y), dtype=np.float32)
    oof_preds_ens = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=130,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=14,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.5,
            scale_pos_weight=2.2,
            random_state=42 + fold,
            verbose=-1
        )
        clf_lgb.fit(X_tr, y_tr)
        oof_preds_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]

        clf_cat = CatBoostClassifier(
            iterations=160,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=3.0,
            auto_class_weights='Balanced',
            random_seed=42 + fold,
            verbose=0
        )
        clf_cat.fit(X_tr, y_tr)
        oof_preds_cat[val_idx] = clf_cat.predict_proba(X_val)[:, 1]

    oof_preds_ens = 0.5 * oof_preds_lgb + 0.5 * oof_preds_cat

    auc = roc_auc_score(y, oof_preds_ens)

    # Optimal threshold search
    best_th, best_f1, best_acc, best_rec, best_prec = 0.5, 0.0, 0.0, 0.0, 0.0
    for th in np.arange(0.20, 0.65, 0.01):
        p_bin = (oof_preds_ens >= th).astype(int)
        f = f1_score(y, p_bin, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
            best_acc = accuracy_score(y, p_bin)
            best_rec = recall_score(y, p_bin, zero_division=0)
            best_prec = precision_score(y, p_bin, zero_division=0)

    # 3-State [0.28, 0.30]
    decided = (oof_preds_ens < 0.28) | (oof_preds_ens >= 0.30)
    cov = float(np.mean(decided) * 100)
    y_dec_true = y[decided]
    y_dec_pred = (oof_preds_ens[decided] >= 0.30).astype(int)
    f1_3s = float(f1_score(y_dec_true, y_dec_pred, zero_division=0) * 100)
    acc_3s = float(accuracy_score(y_dec_true, y_dec_pred) * 100)
    rec_3s = float(recall_score(y_dec_true, y_dec_pred, zero_division=0) * 100)
    prec_3s = float(precision_score(y_dec_true, y_dec_pred, zero_division=0) * 100)

    # View별 F1 (Best threshold)
    view_f1s = {}
    p_opt = (oof_preds_ens >= best_th).astype(int)
    for v_alias, v_code in [("시점 A", "view5"), ("시점 B", "view2"), ("시점 C", "view3"), ("시점 D", "view4"), ("시점 E", "view1")]:
        v_mask = (views == v_code)
        view_f1s[v_alias] = f1_score(y[v_mask], p_opt[v_mask], zero_division=0) * 100

    return {
        "n_feat": len(feat_indices),
        "auc": auc,
        "best_th": best_th,
        "best_f1": best_f1 * 100,
        "best_acc": best_acc * 100,
        "best_prec": best_prec * 100,
        "best_rec": best_rec * 100,
        "cov": cov,
        "f1_3s": f1_3s,
        "acc_3s": acc_3s,
        "rec_3s": rec_3s,
        "prec_3s": prec_3s,
        "views": view_f1s,
        "oof_preds": oof_preds_ens
    }

print("\n" + "=" * 125)
print(f"{'조합 명칭':<25} | {'피처수':<5} | {'AUC':<7} | {'Opt F1(th)':<14} | {'Opt Acc':<8} | {'3S F1':<7} | {'3S Acc':<7} | {'Coverage':<8} | 시점별 F1(A/B/C/D/E)")
print("=" * 125)

combo_results = {}
for name, idxs in combos.items():
    r = eval_combo(idxs)
    combo_results[name] = r
    v_str = f"{r['views']['시점 A']:.1f}/{r['views']['시점 B']:.1f}/{r['views']['시점 C']:.1f}/{r['views']['시점 D']:.1f}/{r['views']['시점 E']:.1f}"
    print(f"{name:<25} | {r['n_feat']:<5} | {r['auc']:.4f}  | {r['best_f1']:.2f}%({r['best_th']:.2f}) | {r['best_acc']:.2f}%  | {r['f1_3s']:.2f}% | {r['acc_3s']:.2f}% | {r['cov']:.1f}%   | {v_str}")
print("=" * 125)
