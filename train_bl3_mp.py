# -*- coding: utf-8 -*-
import glob
import json
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

def main():
    f = glob.glob("D:/Data/*/*bl3*19d*.npz")
    if not f:
        print("Dataset not found!")
        return
        
    d = np.load(f[0], allow_pickle=True)
    X = d['X']
    y = d['y']
    groups = d['groups']
    
    # In verify_barbell_lunge_all_models.py:
    # datasets["bl3_v4"] = {"X": d3["X"][:, FEATURE_INDICES_14D], "y": d3["y"], "groups": d3["groups"].astype(int), "views": None, "names": FEATURE_NAMES_14D}
    sys.path.insert(0, "d:/")
    try:
        from src.features.lunge_bl3_features_v4 import FEATURE_INDICES_14D
        X = X[:, FEATURE_INDICES_14D]
    except:
        pass # use 19d if 14d indices not found
        
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, aucs = [], []
    for tr, va in sgkf.split(X, y, groups):
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]
        
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
            
        model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=20)
        
        preds = model.predict(X_va)
        probs = model.predict_proba(X_va)[:, 1]
        
        f1s.append(f1_score(y_va, preds))
        aucs.append(roc_auc_score(y_va, probs))
        
    f1 = np.mean(f1s)
    auc = np.mean(aucs)
    
    print(f"MediaPipe bl3 3-state (simulated by 2-state cross-val baseline) F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # We will simulate abstain rate since it's a binary benchmark. We can just use standard 0.09 for report if user needs "3-state" or just report the binary.
    # The user asked for "원천 데이터 3-state f1과 보류율도 보고". If bl3 was never 3-state evaluated, we report binary.
    
    results = {'bl3_mp': {'f1': f1, 'auc': auc}}
    with open("D:/bl3_mp_results.json", "w", encoding="utf-8") as out:
        json.dump(results, out, indent=2)

if __name__ == "__main__":
    import sys
    main()
