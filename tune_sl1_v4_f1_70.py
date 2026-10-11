# -*- coding: utf-8 -*-
import pickle
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
from scipy.optimize import minimize

cache = pickle.load(open('D:/Data/피처_데이터/sl1_v4_dataset_cache.pkl', 'rb'))
bundle = pickle.load(open('D:/src/models/second_models/side_lunge/sl1_v4_bundle.pkl', 'rb'))

y = cache['y']
views = cache['views']
X, groups = cache['X'], cache['groups']

models_lgb = bundle['models_lgb']
models_cat = bundle['models_cat']
models_xgb = bundle['models_xgb']
w = bundle['ensemble_weights']

sgkf = StratifiedGroupKFold(n_splits=5)
oof_p = np.zeros(len(y), dtype=np.float32)
for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    p_l = models_lgb[fold].predict(X[val_idx])
    p_c = models_cat[fold].predict_proba(X[val_idx])[:, 1]
    p_x = models_xgb[fold].predict(xgb.DMatrix(X[val_idx]))
    oof_p[val_idx] = w[0]*p_l + w[1]*p_c + w[2]*p_x

unique_views = sorted(list(set(views)))
print("[*] 5대 시점 OOF 확률 생성 완료.")

# 시점별 최적화 그리드 탐색
best_th = {}
t_range = np.arange(0.10, 0.90, 0.005)

for v in unique_views:
    vm = (views == v)
    y_v = y[vm]
    p_v = oof_p[vm]
    
    b_f1, b_t1, b_t2 = -1.0, 0.35, 0.60
    for i, t1 in enumerate(t_range):
        for t2 in t_range[i+1:]:
            unc = (p_v >= t1) & (p_v < t2)
            if np.mean(unc) > 0.10: # 보류율 <= 10.0% 제약
                continue
            dec = ~unc
            if np.sum(dec) == 0:
                continue
            f1 = f1_score(y_v[dec], (p_v[dec] >= t2).astype(int), zero_division=0)
            if f1 > b_f1:
                b_f1, b_t1, b_t2 = f1, t1, t2
    best_th[v] = (b_t1, b_t2, b_f1)
    print(f"  시점 {v}: tau=[{b_t1:.3f}, {b_t2:.3f}], F1={b_f1:.4f}")

# 전체 3-State 종합
pred_states = []
for i in range(len(y)):
    v, p = views[i], oof_p[i]
    t1, t2, _ = best_th[v]
    if p < t1: pred_states.append('Clean')
    elif p >= t2: pred_states.append('Defect')
    else: pred_states.append('Retake')

pred_states = np.array(pred_states)
dec = (pred_states != 'Retake')
f1_total = f1_score(y[dec], (pred_states[dec] == 'Defect').astype(int), zero_division=0)
acc_total = accuracy_score(y[dec], (pred_states[dec] == 'Defect').astype(int))
unc_total = np.mean(pred_states == 'Retake')
rec_total = recall_score(y[dec], (pred_states[dec] == 'Defect').astype(int), zero_division=0)
prec_total = precision_score(y[dec], (pred_states[dec] == 'Defect').astype(int), zero_division=0)

print("="*80)
print(f"최종 독립 최적화 전체 3-State: F1={f1_total:.4f}, Acc={acc_total*100:.2f}%, Rec={rec_total*100:.2f}%, Prec={prec_total*100:.2f}%, 보류율={unc_total*100:.2f}%")
print("="*80)
