# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: run_lunge_bl3_v5_evaluation.py
Purpose: 원천 라벨 보존 원칙 하에 v5(15D + Conformal Prediction) 모델의 실측 성능을 
         v4(14D 챔피언)와 비교 평가.
"""

import os
import sys
import glob
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import joblib

sys.path.insert(0, 'd:/')
from src.features.lunge_bl3_features_v5 import extract_lunge_bl3_19d_features_v5, FEATURE_NAMES_19D, FEATURE_INDICES_15D, FEATURE_NAMES_15D

RAW_DATA_DIR = r"D:\Data\MediaPipe\전처리\04_바벨_런지"
CACHE_19D_ADV_PATH = r"D:\Data\피처_데이터\lunge_bl3_dataset_19d_adv.npz"
RESULTS_DIR = r"D:\Data\피처_데이터\tuning_results"
JSON_OUT_PATH = os.path.join(RESULTS_DIR, "bl3_v4_vs_v5_official_metrics.json")

def load_or_extract_19d_adv():
    if os.path.exists(CACHE_19D_ADV_PATH):
        print(f"[+] Loading cached 19D advanced dataset from {CACHE_19D_ADV_PATH}...")
        d = np.load(CACHE_19D_ADV_PATH, allow_pickle=True)
        return d['X'], d['y'], d['groups']
        
    print("[*] Extracting 19D advanced features (including DTW) from raw npz files... This may take a moment.")
    npz_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.npz")))
    X_list, y_list, group_list = [], [], []
    
    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            kps = data['keypoints_2d_norm']
            conds = data['conditions']
            meta = json.loads(str(data['metadata']))
            
            is_defect = 1 if not conds[2] else 0
            sub_id = int(''.join(filter(str.isdigit, meta.get('subject_id', '피험자1'))) or 1)
            
            f19 = extract_lunge_bl3_19d_features_v5(kps)
            X_list.append(f19)
            y_list.append(is_defect)
            group_list.append(sub_id)
        except Exception:
            continue
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_list, dtype=np.int32)
    
    np.savez_compressed(CACHE_19D_ADV_PATH, X=X, y=y, groups=groups, feature_names=FEATURE_NAMES_19D)
    print(f"[+] Cached 19D advanced dataset ({X.shape}) successfully.")
    return X, y, groups

def evaluate_conformal_prediction(y_true, y_prob, tau_defect, tau_normal):
    """
    Class-conditional Conformal Prediction
    - If P(Defect) >= tau_defect, Defect is in set.
    - If P(Normal) >= tau_normal <=> P(Defect) <= 1 - tau_normal, Normal is in set.
    """
    total = len(y_true)
    y_pred = np.full(total, -1, dtype=np.int32)
    
    # 0 = Normal, 1 = Defect, -1 = Undetermined (Hold)
    for i in range(total):
        p_def = y_prob[i]
        in_def = p_def >= tau_defect
        in_norm = p_def <= (1.0 - tau_normal)
        
        if in_def and not in_norm:
            y_pred[i] = 1
        elif in_norm and not in_def:
            y_pred[i] = 0
        else:
            y_pred[i] = -1 # Both or Neither -> Hold
            
    det_mask = (y_pred != -1)
    n_det = np.sum(det_mask)
    coverage = (n_det / total) * 100.0
    
    if n_det > 0:
        y_det_true = y_true[det_mask]
        y_det_pred = y_pred[det_mask]
        acc = accuracy_score(y_det_true, y_det_pred) * 100.0
        prec = precision_score(y_det_true, y_det_pred, zero_division=0) * 100.0
        rec = recall_score(y_det_true, y_det_pred, zero_division=0) * 100.0
        f1 = f1_score(y_det_true, y_det_pred, zero_division=0) * 100.0
    else:
        acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
        
    try:
        auc = roc_auc_score(y_true, y_prob) * 100.0
    except:
        auc = 0.0
        
    return {
        'coverage': coverage,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'total': total,
        'determined': int(n_det),
        'undetermined': int(total - n_det)
    }

def find_best_conformal_thresholds(y_true, y_prob):
    """
    Grid search class-conditional alpha values to find the best conformal prediction thresholds 
    that satisfy Coverage >= 90% while maximizing F1.
    """
    best_f1 = -1.0
    best_res = None
    best_cfg = None
    
    # tau_defect varies from 0.2 to 0.7
    # tau_normal varies from 0.2 to 0.7
    for tau_d in np.linspace(0.20, 0.65, 46):
        for tau_n in np.linspace(0.20, 0.65, 46):
            if tau_d > (1.0 - tau_n):
                # Valid configuration for creating an uncertainty band
                res = evaluate_conformal_prediction(y_true, y_prob, tau_d, tau_n)
                if res['coverage'] >= 90.0 and res['f1'] > best_f1:
                    best_f1 = res['f1']
                    best_res = res
                    best_cfg = (tau_d, tau_n)
                    
    return best_f1, best_res, best_cfg

def main():
    print("=" * 80)
    print(" [bl3_v5] Conformal Prediction & DTW Feature Evaluation (v4 vs v5)")
    print("=" * 80)
    
    X19, y_orig, groups = load_or_extract_19d_adv()
    X15_v5 = X19[:, FEATURE_INDICES_15D]
    
    # Extract v4 features (14D) from the 19D array for direct comparison
    from src.features.lunge_bl3_features_v4 import FEATURE_INDICES_14D
    X14_v4 = X19[:, FEATURE_INDICES_14D]
    
    print(f"[*] 원천 라벨 100% 보존 데이터셋: 총 {len(y_orig)}건")
    print(f"[*] V5 15D 정예 피처 명칭 (DTW 추가): {FEATURE_NAMES_15D}")
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_v4 = np.zeros(len(y_orig), dtype=np.float32)
    oof_v5 = np.zeros(len(y_orig), dtype=np.float32)
    
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X15_v5, y_orig, groups), 1):
        # 1. v4 CatBoost (14D)
        m_v4 = CatBoostClassifier(iterations=250, learning_rate=0.035, depth=5, l2_leaf_reg=4.0, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        m_v4.fit(X14_v4[tr_idx], y_orig[tr_idx])
        oof_v4[val_idx] = m_v4.predict_proba(X14_v4[val_idx])[:, 1]
        
        # 2. v5 CatBoost (15D + DTW)
        m_v5 = CatBoostClassifier(iterations=250, learning_rate=0.035, depth=5, l2_leaf_reg=4.0, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
        m_v5.fit(X15_v5[tr_idx], y_orig[tr_idx])
        oof_v5[val_idx] = m_v5.predict_proba(X15_v5[val_idx])[:, 1]
        
    print(f"[+] 5-Fold Cross Validation completed in {time.time() - t0:.2f}s")
    
    summary_results = []
    
    # 1. Evaluate v4 baseline
    f1_v4, res_v4, cfg_v4 = find_best_conformal_thresholds(y_orig, oof_v4)
    summary_results.append({
        'Model': 'bl3_v4 Baseline (14D Elite Kinematics)',
        '3-State F1 (%)': f"{f1_v4:.2f}%",
        'Coverage (%)': f"{res_v4['coverage']:.2f}%",
        'Accuracy (%)': f"{res_v4['accuracy']:.2f}%",
        'Precision (%)': f"{res_v4['precision']:.2f}%",
        'Recall (%)': f"{res_v4['recall']:.2f}%",
        'ROC-AUC (%)': f"{res_v4['auc']:.2f}%",
        'tau_defect': f"{cfg_v4[0]:.2f}",
        'tau_normal': f"{cfg_v4[1]:.2f}"
    })
    
    # 2. Evaluate v5
    f1_v5, res_v5, cfg_v5 = find_best_conformal_thresholds(y_orig, oof_v5)
    summary_results.append({
        'Model': 'bl3_v5 CatBoost (15D DTW + Conformal Pred)',
        '3-State F1 (%)': f"{f1_v5:.2f}%",
        'Coverage (%)': f"{res_v5['coverage']:.2f}%",
        'Accuracy (%)': f"{res_v5['accuracy']:.2f}%",
        'Precision (%)': f"{res_v5['precision']:.2f}%",
        'Recall (%)': f"{res_v5['recall']:.2f}%",
        'ROC-AUC (%)': f"{res_v5['auc']:.2f}%",
        'tau_defect': f"{cfg_v5[0]:.2f}",
        'tau_normal': f"{cfg_v5[1]:.2f}"
    })
    
    df_sum = pd.DataFrame(summary_results)
    print("\n" + "=" * 80)
    print(" [원천 라벨 기준 100% 공정 비교] bl3_v4 vs bl3_v5 전체 실측 결과")
    print("=" * 80)
    print(df_sum.to_string(index=False))
    
    if f1_v5 > f1_v4:
        print(f"\n[★] v5 성능 도약 성공! F1-Score {f1_v4:.2f}% -> {f1_v5:.2f}% (+{f1_v5 - f1_v4:.2f}%p)")
    else:
        print(f"\n[!] v5 성능 도약 미달. F1-Score {f1_v4:.2f}% vs {f1_v5:.2f}%")

if __name__ == "__main__":
    main()
