# -*- coding: utf-8 -*-
"""
Sequential Backward Elimination (SBE) for Side Lunge c1
"""

import pickle
import numpy as np
import sys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb

cache = pickle.load(open('D:/Data/피처_데이터/sl1_v4_dataset_cache.pkl', 'rb'))
X = cache['X']
y = cache['y']
groups = cache['groups']
fnames = list(cache['feature_names'])

sgkf = StratifiedGroupKFold(n_splits=5)
spw = float(np.sum(y == 0) / np.sum(y == 1))

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.04,
    'num_leaves': 31,
    'max_depth': 6,
    'scale_pos_weight': spw,
    'subsample': 0.8,
    'colsample_bytree': 0.75,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

def eval_subset(feat_list):
    sub_indices = [fnames.index(f) for f in feat_list]
    X_sub = X[:, sub_indices]
    oof_pred = np.zeros(len(y), dtype=np.float32)
    for trn_idx, val_idx in sgkf.split(X_sub, y, groups):
        trn_d = lgb.Dataset(X_sub[trn_idx], label=y[trn_idx])
        val_d = lgb.Dataset(X_sub[val_idx], label=y[val_idx], reference=trn_d)
        m = lgb.train(lgb_params, trn_d, num_boost_round=350, valid_sets=[val_d], callbacks=[lgb.early_stopping(35, verbose=False)])
        oof_pred[val_idx] = m.predict(X_sub[val_idx])
    auc = roc_auc_score(y, oof_pred)
    f1 = f1_score(y, (oof_pred >= 0.5).astype(int))
    return auc, f1

# 36개 피처 시작
curr_feats = list(fnames)
best_auc, best_f1 = eval_subset(curr_feats)
print(f"[*] 시작 36D: AUC={best_auc:.4f}, F1={best_f1:.4f}")

history = [(len(curr_feats), list(curr_feats), best_auc, best_f1, "INITIAL")]

# 매 스텝 가장 성능을 방해하는 피처 1개씩 제거 (최대 20개 제거 시도)
for step in range(1, 22):
    worst_feat = None
    best_step_f1 = -1.0
    best_step_auc = -1.0
    
    for cand in curr_feats:
        trial = [f for f in curr_feats if f != cand]
        auc, f1 = eval_subset(trial)
        if f1 > best_step_f1:
            best_step_f1 = f1
            best_step_auc = auc
            worst_feat = cand
            
    diff_f1 = best_step_f1 - history[-1][3]
    curr_feats.remove(worst_feat)
    history.append((len(curr_feats), list(curr_feats), best_step_auc, best_step_f1, worst_feat))
    print(f"  Step {step:2d} | 남은피처: {len(curr_feats):2d}D | 제거피처: {worst_feat:<35s} | F1={best_step_f1:.4f} ({diff_f1:+.4f}) | AUC={best_step_auc:.4f}")

# F1 최고점 선택
history.sort(key=lambda x: x[3], reverse=True)
opt_len, opt_feats, opt_auc, opt_f1, _ = history[0]

print("="*80)
print(f"[+] 최적 피처 수: {opt_len}D (시작 36D 대비 F1 {opt_f1-best_f1:+.4f} 향상)")
print(f"[+] 최적 성능: AUC={opt_auc:.4f}, 2-State F1={opt_f1:.4f}")
print("선택된 정예 피처 목록:")
for i, f in enumerate(opt_feats):
    print(f"  {i+1:2d}. {f}")

with open('D:/Data/피처_데이터/sl1_v5_optimal_features.json', 'w', encoding='utf-8') as fp:
    import json
    json.dump({'optimal_dimension': opt_len, 'features': opt_feats, 'auc': opt_auc, 'f1': opt_f1}, fp, ensure_ascii=False, indent=2)
