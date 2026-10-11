# -*- coding: utf-8 -*-
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

cache_path = r'D:\Data\피처_데이터\side_lunge_c5_v5_all_relabel_36d_features.npz'
data = np.load(cache_path, allow_pickle=True)
X = data['X']
y = data['y']
keys = data['keys']
views = data['views']
subjects = data['subjects']

bundle_path = r'D:\src\models\second_models\side_lunge\weights\sl5_v5_all_relabel_bundle.pkl'
with open(bundle_path, 'rb') as f:
    bundle = pickle.load(f)

models = bundle['models']
weights = bundle['blend_weights']
th_low = bundle['th_low']
th_high = bundle['th_high']

# 5-fold OOF reproduction
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(y), dtype=np.float32)

for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
    X_val = X[val_idx]
    p_lgb = models['lgb'][fold_idx].predict_proba(X_val)[:, 1]
    p_cat = models['cb'][fold_idx].predict_proba(X_val)[:, 1]
    p_xgb = models['xgb'][fold_idx].predict_proba(X_val)[:, 1]
    p_rf = models['rf'][fold_idx].predict_proba(X_val)[:, 1]
    p_et = models['et'][fold_idx].predict_proba(X_val)[:, 1]
    
    oof_fold = (
        weights[0] * p_lgb +
        weights[1] * p_cat +
        weights[2] * p_xgb +
        weights[3] * p_rf +
        weights[4] * p_et
    )
    oof_probs[val_idx] = oof_fold

# 3-state predictions
# 0: normal (< th_low), 1: defect (>= th_high), -1: caution
preds = np.full(len(y), -1, dtype=np.int32)
preds[oof_probs < th_low] = 0
preds[oof_probs >= th_high] = 1

valid_mask = (preds != -1)
print(f"Overall Coverage: {np.mean(valid_mask)*100:.2f}%")
print(f"Overall 3-State F1: {f1_score(y[valid_mask], preds[valid_mask])*100:.2f}%")

# 1. 시점별 성적
print("\n=== [1. 시점별 3-State F1 및 오분류율] ===")
for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
    vm = (views == v) & valid_mask
    y_v = y[vm]
    p_v = preds[vm]
    f1 = f1_score(y_v, p_v) * 100
    prec = precision_score(y_v, p_v) * 100
    rec = recall_score(y_v, p_v) * 100
    fp = np.sum((y_v == 0) & (p_v == 1))
    fn = np.sum((y_v == 1) & (p_v == 0))
    print(f"  {v}: F1 {f1:5.2f}% | Prec {prec:5.2f}% | Rec {rec:5.2f}% | FP: {fp:3d}, FN: {fn:3d}")

# 2. 피험자별 성적
print("\n=== [2. 피험자별 3-State F1] ===")
for sub in np.unique(subjects):
    sm = (subjects == sub) & valid_mask
    y_s = y[sm]
    p_s = preds[sm]
    f1 = f1_score(y_s, p_s) * 100
    print(f"  피험자 {sub}: F1 {f1:5.2f}% (샘플 {len(y_s)}개)")

# 3. Key별 오분류 분석
print("\n=== [3. Key별 주요 오분류 현황 (FP / FN)] ===")
key_err = {}
for k in np.unique(keys):
    km = (keys == k) & valid_mask
    y_k = y[km]
    p_k = preds[km]
    fp = np.sum((y_k == 0) & (p_k == 1))
    fn = np.sum((y_k == 1) & (p_k == 0))
    corr = np.sum(y_k == p_k)
    key_err[k] = {'total': len(y_k), 'fp': fp, 'fn': fn, 'corr': corr, 'is_defect': bool(np.mean(y_k) > 0.5)}

sorted_keys = sorted(key_err.items(), key=lambda x: (x[1]['fp'] + x[1]['fn']), reverse=True)
for k, info in sorted_keys[:12]:
    kind = "결함Key" if info['is_defect'] else "정상Key"
    print(f"  Key {k:<4} ({kind:<5}): 오류합 {info['fp']+info['fn']:3d} / {info['total']:3d}건 (FP: {info['fp']:2d}, FN: {info['fn']:2d}, 정확도: {info['corr']/info['total']*100:.1f}%)")
