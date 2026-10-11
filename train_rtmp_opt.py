# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

def find_best_3state_thresholds(y_true, y_prob):
    # Search grid for tau_norm and tau_def
    best_f1 = 0
    best_tau = (0.5, 0.5)
    best_retake = 0
    
    # We want tau_norm <= tau_def
    for t_norm in np.arange(0.1, 0.9, 0.05):
        for t_def in np.arange(t_norm, 0.95, 0.05):
            # State 0 (Norm): prob < t_norm
            # State 1 (Def): prob >= t_def
            # State 2 (Uncertain): t_norm <= prob < t_def
            
            # Predict 3-state
            y_pred_3s = np.where(y_prob >= t_def, 1, np.where(y_prob < t_norm, 0, 2))
            
            # Calculate metrics only on covered samples (State 0 and 1)
            covered_mask = (y_pred_3s != 2)
            retake_rate = 1.0 - np.mean(covered_mask)
            
            if retake_rate > 0.15: # Don't allow more than 15% retake
                continue
                
            if np.sum(covered_mask) > 0:
                y_true_cov = y_true[covered_mask]
                y_pred_cov = y_pred_3s[covered_mask]
                f1 = f1_score(y_true_cov, y_pred_cov, zero_division=0)
                
                # We want to maximize F1, and as a tiebreaker, minimize retake rate
                if f1 > best_f1 or (f1 == best_f1 and retake_rate < best_retake):
                    best_f1 = f1
                    best_tau = (t_norm, t_def)
                    best_retake = retake_rate
                    
    return best_tau, best_f1, best_retake

def main():
    d = np.load(r"D:\rtmp_fair_dataset.npz")
    groups = d['subjects']
    
    results = {}
    
    for c in ['1', '2', '3', '4', '5']:
        X = d[f'X_c{c}']
        y = d[f'y_c{c}']
        
        print(f"\n--- Evaluating and Optimizing c{c} ---")
        
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y), dtype=np.float32)
        
        for tr, va in sgkf.split(X, y, groups):
            X_tr, y_tr = X[tr], y[tr]
            X_va, y_va = X[va], y[va]
            
            if len(np.unique(y_tr)) < 2: continue
            
            model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, random_seed=42, thread_count=2)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=20)
            
            probs = model.predict_proba(X_va)[:, 1]
            oof_preds[va] = probs
            
        # Find optimal thresholds
        tau, f1_3s, retake = find_best_3state_thresholds(y, oof_preds)
        
        # Also 2-state F1 at 0.5
        preds_2s = (oof_preds >= 0.5).astype(int)
        f1_2s = f1_score(y, preds_2s)
        
        print(f"c{c} 2-State F1: {f1_2s:.4f}")
        print(f"c{c} Optimal Thresholds: {tau}")
        print(f"c{c} 3-State F1: {f1_3s:.4f}")
        print(f"c{c} Retake Rate: {retake:.4f}")
        
        results[f'c{c}'] = {
            '2state_f1': f1_2s,
            '3state_f1': f1_3s,
            'retake': retake,
            'tau': tau
        }
        
        # Retrain on full data and save
        full_model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, random_seed=42, thread_count=2)
        full_model.fit(X, y)
        save_path = f"D:/src/models/second_models/babel_lunge/bl{c}_rtmp_1.cbm"
        full_model.save_model(save_path)
        print(f"Saved optimized model to {save_path}")
        
    with open("D:/rtmp_opt_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
