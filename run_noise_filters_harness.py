import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import cleanlab
from cleanlab.filter import find_label_issues

import builtins
def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

def eval_3s(y_true, prob, tc, td):
    m = (prob < tc) | (prob >= td)
    cov = m.mean() * 100.0
    if cov == 0: return 0.0, 0.0
    yt = y_true[m]
    yp = (prob[m] >= td).astype(int)
    if len(np.unique(yt)) < 2 and len(np.unique(yp)) < 2: return 0.0, cov
    return f1_score(yt, yp) * 100.0, cov

def train_and_eval(X_sub, y_sub, subj_sub, y_clean_sub, method_name):
    print(f"\n--- Training Model for {method_name} ---")
    print(f"Dataset Size: {len(y_sub)} (Retention Rate: {len(y_sub)/1035*100:.1f}%)")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(y_sub))
    
    for trn_idx, val_idx in sgkf.split(X_sub, y_sub, groups=subj_sub):
        clf_lgb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.025, max_depth=5, num_leaves=20, class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
        clf_cb = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6, auto_class_weights='Balanced', verbose=0, random_seed=42)
        clf_lgb.fit(X_sub[trn_idx], y_sub[trn_idx])
        clf_cb.fit(X_sub[trn_idx], y_sub[trn_idx])
        oof_prob[val_idx] = (clf_lgb.predict_proba(X_sub[val_idx])[:, 1] + clf_cb.predict_proba(X_sub[val_idx])[:, 1]) / 2.0
    
    b_f1, b_cov = 0.0, 100.0
    for tc in np.linspace(0.20, 0.52, 33):
        for td in np.linspace(tc + 0.02, 0.70, 33):
            f1, cov = eval_3s(y_sub, oof_prob, tc, td)
            if cov >= 85.0 and f1 > b_f1:
                b_f1 = f1; b_cov = cov
                
    yp = (oof_prob > 0.5).astype(int)
    true_acc = (yp == y_clean_sub).mean() * 100.0
    print(f"[{method_name}] Result => F1(3-State, >85%cov): {b_f1:.2f}% | True Oracle Accuracy: {true_acc:.2f}%")
    return b_f1

def check_filter_quality(is_filtered, y_raw, y_clean, method_name):
    num_filtered = is_filtered.sum()
    true_noise_mask = (y_raw != y_clean)
    num_true_noise = true_noise_mask.sum()
    
    correctly_caught = (is_filtered & true_noise_mask).sum()
    precision = correctly_caught / num_filtered if num_filtered > 0 else 0
    recall = correctly_caught / num_true_noise if num_true_noise > 0 else 0
    print(f"[{method_name} Filter] Dropped: {num_filtered}/{len(y_raw)}")
    print(f"[{method_name} Filter] Precision (Dropped was actual noise): {precision*100:.1f}%")
    print(f"[{method_name} Filter] Recall (Caught noise / Total noise): {recall*100:.1f}%")

def main():
    print("==========================================================")
    print(" 3-Method Label Noise Filtering & Evaluation Harness")
    print("==========================================================")
    
    CACHE_PKL = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']
    keys = df['key'].values
    y_raw = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    y_clean = np.load(r'D:\src\models\second_models\babel_stiff_deadlift\y_clean.npy')
    subjects = df['subj'].values

    X_PATH = r'D:\src\models\second_models\babel_stiff_deadlift\X_26d.npy'
    import time
    while not os.path.exists(X_PATH):
        print("Waiting for X_26d.npy...")
        time.sleep(2)
    X = np.load(X_PATH)
    
    print(f"Baseline Data Size: {len(y_raw)}, True Noise Count: {(y_raw != y_clean).sum()}")

    # A. Cleanlab
    print("\n[A] Cleanlab Method (sd1_rtmp_3_a)")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs_cl = np.zeros((len(y_raw), 2))
    for trn_idx, val_idx in sgkf.split(X, y_raw, groups=subjects):
        clf = lgb.LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
        clf.fit(X[trn_idx], y_raw[trn_idx])
        oof_probs_cl[val_idx] = clf.predict_proba(X[val_idx])

    issues = find_label_issues(labels=y_raw, pred_probs=oof_probs_cl)
    mask_cl_drop = issues
    check_filter_quality(mask_cl_drop, y_raw, y_clean, "Cleanlab")
    mask_cl_keep = ~mask_cl_drop
    train_and_eval(X[mask_cl_keep], y_raw[mask_cl_keep], subjects[mask_cl_keep], y_clean[mask_cl_keep], "Cleanlab")

    # B. Unsupervised k-NN
    print("\n[B] Unsupervised k-NN Method (sd1_rtmp_3_b)")
    X_scaled = StandardScaler().fit_transform(X)
    knn = NearestNeighbors(n_neighbors=11)
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(X_scaled)

    mask_knn_drop = np.zeros(len(y_raw), dtype=bool)
    for i in range(len(y_raw)):
        neighbor_labels = y_raw[indices[i][1:]]
        self_l = y_raw[i]
        opp_count = (neighbor_labels != self_l).sum()
        if opp_count >= 8:
            mask_knn_drop[i] = True

    check_filter_quality(mask_knn_drop, y_raw, y_clean, "Unsupervised k-NN")
    mask_knn_keep = ~mask_knn_drop
    train_and_eval(X[mask_knn_keep], y_raw[mask_knn_keep], subjects[mask_knn_keep], y_clean[mask_knn_keep], "Unsupervised k-NN")

    # C. AUM
    print("\n[C] AUM Method (sd1_rtmp_3_c)")
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

    num_to_drop = mask_cl_drop.sum()
    drop_threshold = np.percentile(aum_oof, (num_to_drop / len(y_raw)) * 100)
    mask_aum_drop = (aum_oof <= drop_threshold)

    check_filter_quality(mask_aum_drop, y_raw, y_clean, "AUM")
    mask_aum_keep = ~mask_aum_drop
    train_and_eval(X[mask_aum_keep], y_raw[mask_aum_keep], subjects[mask_aum_keep], y_clean[mask_aum_keep], "AUM")
    print("\nEvaluation Harness Complete.")

if __name__ == '__main__':
    main()
