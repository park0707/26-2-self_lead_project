# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: evaluate_7d_and_3state.py
Purpose: 선발된 7D 정예 피처 기반 d2 모델 상세 성능 평가 및 3-State 윈도우 그리드 서치
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
clips_list = []

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
    clips_list.append(base)

X_all = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
subs = np.array(subs_list, dtype=np.int32)
views = np.array(views_list)

# 7D 정예 피처셋
diet_7_indices = [24, 4, 7, 26, 28, 27, 23]
X_7d = X_all[:, diet_7_indices]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_lgb = np.zeros(len(y), dtype=np.float32)
oof_preds_cat = np.zeros(len(y), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_7d, y, groups=subs), 1):
    X_tr, y_tr = X_7d[train_idx], y[train_idx]
    X_val, y_val = X_7d[val_idx], y[val_idx]

    clf_lgb = lgb.LGBMClassifier(
        n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=14,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=2.2, random_state=42 + fold, verbose=-1
    )
    clf_lgb.fit(X_tr, y_tr)
    oof_preds_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]

    clf_cat = CatBoostClassifier(
        iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
        auto_class_weights='Balanced', random_seed=42 + fold, verbose=0
    )
    clf_cat.fit(X_tr, y_tr)
    oof_preds_cat[val_idx] = clf_cat.predict_proba(X_val)[:, 1]

oof_ens = 0.5 * oof_preds_lgb + 0.5 * oof_preds_cat
auc = roc_auc_score(y, oof_ens)

print(f"[*] 7D 정예 피처 앙상블 5-Fold OOF AUC: {auc:.4f}")

# Optimal 2-State
best_f1, best_th = 0.0, 0.5
for th in np.arange(0.20, 0.65, 0.01):
    f = f1_score(y, (oof_ens >= th).astype(int), zero_division=0)
    if f > best_f1:
        best_f1 = f
        best_th = th

p_opt = (oof_ens >= best_th).astype(int)
acc_opt = accuracy_score(y, p_opt)
prec_opt = precision_score(y, p_opt, zero_division=0)
rec_opt = recall_score(y, p_opt, zero_division=0)
cm_opt = confusion_matrix(y, p_opt)

print(f"[*] 최적 2-State (임계값 {best_th:.2f}): F1={best_f1*100:.2f}%, Acc={acc_opt*100:.2f}%, Prec={prec_opt*100:.2f}%, Rec={rec_opt*100:.2f}%")
print(f"    Confusion Matrix: TN={cm_opt[0,0]}, FP={cm_opt[0,1]}, FN={cm_opt[1,0]}, TP={cm_opt[1,1]}")

# 3-State 그리드 서치
print("\n[*] 3-State [tau_normal, tau_defect] 그리드 서치 (Coverage >= 75% 기준):")
print(f"{'tau_norm':<10} | {'tau_def':<10} | {'Coverage':<10} | {'F1-score':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 80)

best_3s = None
best_3s_score = 0.0

for t_norm in np.arange(0.20, 0.40, 0.02):
    for t_def in np.arange(t_norm + 0.02, 0.50, 0.02):
        decided = (oof_ens <= t_norm) | (oof_ens >= t_def)
        cov = float(np.mean(decided) * 100)
        if cov < 75.0:
            continue
        y_d_true = y[decided]
        y_d_pred = (oof_ens[decided] >= t_def).astype(int)
        f1_3s = f1_score(y_d_true, y_d_pred, zero_division=0) * 100
        acc_3s = accuracy_score(y_d_true, y_d_pred) * 100
        prec_3s = precision_score(y_d_true, y_d_pred, zero_division=0) * 100
        rec_3s = recall_score(y_d_true, y_d_pred, zero_division=0) * 100

        # 종합 평가지수
        score = f1_3s * 0.5 + acc_3s * 0.3 + (cov / 100) * 20
        if score > best_3s_score:
            best_3s_score = score
            best_3s = {
                'tn': round(t_norm, 2),
                'td': round(t_def, 2),
                'cov': cov,
                'f1': f1_3s,
                'acc': acc_3s,
                'prec': prec_3s,
                'rec': rec_3s
            }

        if abs(t_norm - 0.28) < 1e-4 and abs(t_def - 0.30) < 1e-4:
            print(f"{t_norm:<10.2f} | {t_def:<10.2f} | {cov:<10.1f}% | {f1_3s:<10.2f}% | {acc_3s:<10.2f}% | {prec_3s:<10.2f}% | {rec_3s:<10.2f}% (기존 v2.0 공식 기준)")

print("-" * 80)
print(f"[★ 7D 최적 3-State 윈도우]: [{best_3s['tn']}, {best_3s['td']}] -> F1={best_3s['f1']:.2f}%, Acc={best_3s['acc']:.2f}%, Cov={best_3s['cov']:.1f}%")

# 시점별 성능 검증
print("\n[*] 5대 공식 시점별 성능 (7D 정예 모델 최적 임계값):")
for v_alias, v_code in [("시점 A (우측 후방 45°)", "view5"),
                        ("시점 B (우측 전방 45°)", "view2"),
                        ("시점 C (정면)", "view3"),
                        ("시점 D (좌측 전방 45°)", "view4"),
                        ("시점 E (좌측 후방 45°)", "view1")]:
    mask = (views == v_code)
    y_v = y[mask]
    p_v = p_opt[mask]
    prob_v = oof_ens[mask]
    v_auc = roc_auc_score(y_v, prob_v) if len(np.unique(y_v)) > 1 else 0.5
    v_f1 = f1_score(y_v, p_v, zero_division=0) * 100
    v_acc = accuracy_score(y_v, p_v) * 100
    v_rec = recall_score(y_v, p_v, zero_division=0) * 100
    print(f"  {v_alias:<20} | AUC: {v_auc:.4f} | F1: {v_f1:.2f}% | Acc: {v_acc:.2f}% | Recall: {v_rec:.2f}%")

# 피험자별 5-Fold 성능
print("\n[*] 피험자 1~9번별 5-Fold Fold 성과:")
sgkf_check = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(sgkf_check.split(X_7d, y, groups=subs), 1):
    val_subs = np.unique(subs[va])
    f_f1 = f1_score(y[va], p_opt[va], zero_division=0) * 100
    f_auc = roc_auc_score(y[va], oof_ens[va])
    print(f"  Fold {fold} (피험자 {val_subs.tolist()}) -> AUC: {f_auc:.4f}, F1: {f_f1:.2f}%")
