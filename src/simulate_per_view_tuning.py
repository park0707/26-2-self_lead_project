# -*- coding: utf-8 -*-
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score

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

# 5-fold OOF
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

print("=== [시점별 독립 최적 임계치 그리드 탐색] ===")
unique_views = ['view1', 'view2', 'view3', 'view4', 'view5']
per_view_res = {}
total_confirmed = 0
total_samples = len(y)
all_confirmed_y = []
all_confirmed_p = []

for v in unique_views:
    vm = (views == v)
    y_v = y[vm]
    p_v = oof_probs[vm]
    
    best_f1 = 0
    best_th_low = 0.4
    best_th_high = 0.5
    best_cov = 0
    best_prec = 0
    best_rec = 0
    
    # Grid search th_low, th_high
    for th_l in np.arange(0.25, 0.55, 0.01):
        for th_h in np.arange(th_l, min(0.75, th_l + 0.25), 0.01):
            rej = np.mean((p_v >= th_l) & (p_v < th_h))
            if rej > 0.10: # 보류율 <= 10.0% 엄격 준수
                continue
            
            p_lbl = np.full(len(y_v), -1)
            p_lbl[p_v < th_l] = 0
            p_lbl[p_v >= th_h] = 1
            
            val_m = (p_lbl != -1)
            if np.sum(val_m) == 0:
                continue
            f1 = f1_score(y_v[val_m], p_lbl[val_m])
            if f1 > best_f1:
                best_f1 = f1
                best_th_low = th_l
                best_th_high = th_h
                best_cov = 1.0 - rej
                best_prec = precision_score(y_v[val_m], p_lbl[val_m])
                best_rec = recall_score(y_v[val_m], p_lbl[val_m])
                
    per_view_res[v] = {
        'th_low': best_th_low,
        'th_high': best_th_high,
        'f1': best_f1,
        'cov': best_cov,
        'prec': best_prec,
        'rec': best_rec
    }
    print(f"  {v}: Best [tau_clean={best_th_low:.3f}, tau_defect={best_th_high:.3f}] | F1: {best_f1*100:5.2f}% | Prec: {best_prec*100:5.2f}% | Rec: {best_rec*100:5.2f}% | Cov: {best_cov*100:5.2f}%")
    
    # Collect predictions with optimal thresholds
    p_final = np.full(len(y_v), -1)
    p_final[p_v < best_th_low] = 0
    p_final[p_v >= best_th_high] = 1
    vm_conf = (p_final != -1)
    all_confirmed_y.extend(y_v[vm_conf])
    all_confirmed_p.extend(p_final[vm_conf])

all_confirmed_y = np.array(all_confirmed_y)
all_confirmed_p = np.array(all_confirmed_p)
overall_f1 = f1_score(all_confirmed_y, all_confirmed_p) * 100
overall_prec = precision_score(all_confirmed_y, all_confirmed_p) * 100
overall_rec = recall_score(all_confirmed_y, all_confirmed_p) * 100
overall_cov = len(all_confirmed_y) / total_samples * 100

print(f"\n[+] 시점별 맞춤 임계치 적용 전체 종합 성적:")
print(f"    - Overall 3-State F1: {overall_f1:.2f}% (기존 63.61% 대비 +{overall_f1-63.61:.2f}%p)")
print(f"    - Precision: {overall_prec:.2f}% | Recall: {overall_rec:.2f}% | Coverage: {overall_cov:.2f}%")
