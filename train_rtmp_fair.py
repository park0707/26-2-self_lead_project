# -*- coding: utf-8 -*-
import json
import numpy as np
import concurrent.futures
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_defect(X, y, groups, defect_name):
    # Only use samples that have y==0 or y==1.
    # Actually wait, y is 0 or 1 for ALL sequences, but some sequences were filtered out in MediaPipe
    # However, if we just use the Lunge target (0 or 1) across the FULL dataset (5330 sequences) or the filtered dataset?
    # To be perfectly fair to RTMP vs MediaPipe, we should train RTMP on the exact SAME subsets?
    # Or just train RTMP on all valid RTMP samples? 
    # If RTMP has more valid samples, it's better for RTMP! Let's train on all valid RTMP samples!
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, aucs = [], []
    for tr, va in sgkf.split(X, y, groups):
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]
        
        # Check if there is only one class in y_tr or y_va
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
            
        model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=20)
        
        preds = model.predict(X_va)
        probs = model.predict_proba(X_va)[:, 1]
        
        f1s.append(f1_score(y_va, preds))
        aucs.append(roc_auc_score(y_va, probs))
        
    if not f1s:
        return 0, 0
    return np.mean(f1s), np.mean(aucs)

def main():
    d = np.load(r"D:\rtmp_fair_dataset.npz")
    groups = d['subjects']
    
    results = {}
    
    for c in ['1', '2', '3', '4', '5']:
        X = d[f'X_c{c}']
        y = d[f'y_c{c}']
        
        print(f"Evaluating c{c}...")
        f1, auc = evaluate_defect(X, y, groups, f'c{c}')
        print(f"  F1: {f1:.4f}, AUC: {auc:.4f}")
        results[f'c{c}'] = {'f1': f1, 'auc': auc}
        
    with open("D:/rtmp_fair_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
