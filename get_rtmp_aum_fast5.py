import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import builtins
def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

def eval_3s_view(y_true, prob, tc, td):
    m = (prob < tc) | (prob >= td)
    cov = m.mean() * 100.0
    if cov == 0: return 0.0, 0.0, 0.0, 0.0, 0.0
    yt = y_true[m]
    yp = (prob[m] >= td).astype(int)
    
    if len(yt) == 0: return 0.0, cov, 0.0, 0.0, 0.0
    
    f1 = f1_score(yt, yp) * 100.0
    acc = (yt == yp).mean() * 100.0
    
    # Precision & Recall manually
    tp = ((yp == 1) & (yt == 1)).sum()
    fp = ((yp == 1) & (yt == 0)).sum()
    fn = ((yp == 0) & (yt == 1)).sum()
    
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    
    return f1, cov, prec, rec, acc

def main():
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    keys = df['key'].values
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    views = df['view'].values
    subjects = df['subj'].values
    
    print("Loading RTMP features...")
    npz = np.load(r'D:\src\models\second_models\babel_stiff_deadlift\stiff_sd1_rtmp_clean_features.npz')
    X = npz['X_tab']
    
    # 1. AUM filtering
    print("Generating AUM Mask...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    aum_oof = np.zeros(len(y_raw))
    
    for trn_idx, val_idx in sgkf.split(X_scaled, y_raw, groups=subjects):
        mlp_fold = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1, warm_start=True, random_state=42)
        fold_aum = np.zeros(len(val_idx))
        for epoch in range(25):
            mlp_fold.partial_fit(X_scaled[trn_idx], y_raw[trn_idx], classes=np.array([0, 1]))
            probs = mlp_fold.predict_proba(X_scaled[val_idx])
            for j, v_i in enumerate(val_idx):
                l = y_raw[v_i]
                margin = probs[j, l] - probs[j, 1-l]
                fold_aum[j] += margin
        aum_oof[val_idx] = fold_aum
        
    drop_threshold = np.percentile(aum_oof, (339 / len(y_raw)) * 100)
    mask_aum_keep = ~(aum_oof <= drop_threshold)
    
    X_sub = X[mask_aum_keep]
    y_sub = y_raw[mask_aum_keep]
    subj_sub = subjects[mask_aum_keep]
    view_sub = views[mask_aum_keep]
    
    print("Training models on AUM subset...")
    oof_prob = np.zeros(len(y_sub))
    for trn_idx, val_idx in sgkf.split(X_sub, y_sub, groups=subj_sub):
        clf_lgb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.025, max_depth=5, num_leaves=20, class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
        clf_lgb.fit(X_sub[trn_idx], y_sub[trn_idx])
        oof_prob[val_idx] = clf_lgb.predict_proba(X_sub[val_idx])[:, 1]
        
    b_f1, b_cov, b_tc, b_td = 0.0, 100.0, 0, 0
    b_prec, b_rec, b_acc = 0, 0, 0
    
    for tc in np.linspace(0.20, 0.52, 33):
        for td in np.linspace(tc + 0.02, 0.70, 33):
            f1, cov, prec, rec, acc = eval_3s_view(y_sub, oof_prob, tc, td)
            if cov >= 85.0 and f1 > b_f1:
                b_f1, b_cov = f1, cov
                b_tc, b_td = tc, td
                b_prec, b_rec, b_acc = prec, rec, acc
                
    print(f"\n[Overall SD1 - AUM] F1: {b_f1:.2f}% | Prec: {b_prec:.2f}% | Rec: {b_rec:.2f}% | Cov: {b_cov:.2f}% (tc={b_tc:.3f}, td={b_td:.3f})")
    
    with open('d:/rtmp_f1_result.txt', 'w') as f:
        f.write(f"{b_f1:.2f}\n{b_cov:.2f}")

if __name__ == '__main__':
    main()
