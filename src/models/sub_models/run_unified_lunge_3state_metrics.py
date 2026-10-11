# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
try:
    from src.features.lunge_bl3_features_v4 import extract_lunge_bl3_18d_features_v4 as feature_extractor
except ImportError:
    pass

DIRS = {
    '04': r"D:\Data\MediaPipe\전처리\04_바벨_런지",
    '05': r"D:\Data\MediaPipe\전처리\05_스텝_포워드_다이나믹_런지",
    '06': r"D:\Data\MediaPipe\전처리\06_스텝_백워드_다이나믹_런지"
}

def load_data():
    X_list, y_list, group_list, src_list = [], [], [], []
    for src_id, dpath in DIRS.items():
        npz_files = sorted(glob.glob(os.path.join(dpath, "*.npz")))
        for f in npz_files:
            try:
                data = np.load(f, allow_pickle=True)
                kps = data['keypoints_2d_norm']
                conds = data['conditions']
                try:
                    meta = json.loads(str(data['metadata'].item()))
                except:
                    meta = json.loads(str(data['metadata']))
                
                is_defect = 1 if not conds[2] else 0
                sub_id = int(''.join(filter(str.isdigit, meta.get('subject_id', '피험자1'))) or 1)
                
                f_vec = feature_extractor(kps)
                X_list.append(f_vec)
                y_list.append(is_defect)
                group_list.append(sub_id)
                src_list.append(src_id)
            except Exception:
                continue
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), np.array(group_list, dtype=np.int32), np.array(src_list)

def evaluate_predictions(y_true, y_prob, tau_low, tau_high):
    total = len(y_true)
    y_pred = np.full(total, -1, dtype=np.int32)
    y_pred[y_prob < tau_low] = 0
    y_pred[y_prob > tau_high] = 1
    
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
        
    return {'coverage': coverage, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'n_det': n_det}

def find_best_3state_thresholds(y_true, y_prob, min_coverage=0.94):
    best_f1 = -1
    best_tau_low, best_tau_high = 0.5, 0.5
    best_res = None
    
    for t_low in np.arange(0.1, 0.50, 0.05):
        for t_high in np.arange(0.50, 0.90, 0.05):
            res = evaluate_predictions(y_true, y_prob, t_low, t_high)
            if res['coverage'] >= min_coverage * 100.0:
                if res['f1'] > best_f1:
                    best_f1 = res['f1']
                    best_tau_low = t_low
                    best_tau_high = t_high
                    best_res = res
    return best_tau_low, best_tau_high, best_res

def main():
    X, y, groups, sources = load_data()
    
    mask_04 = sources == '04'
    X_04, y_04, groups_04 = X[mask_04], y[mask_04], groups[mask_04]
    
    mask_dyn = (sources == '05') | (sources == '06')
    X_dyn, y_dyn = X[mask_dyn], y[mask_dyn]
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X_04, y_04, groups_04))
    tr_idx, te_idx = splits[0]
    
    X_04_train, y_04_train = X_04[tr_idx], y_04[tr_idx]
    X_04_test, y_04_test = X_04[te_idx], y_04[te_idx]
    
    print("[*] 1. Baseline Model (Train on 04 only) - Probabilities")
    cb_baseline = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
    cb_baseline.fit(X_04_train, y_04_train)
    probs_base = cb_baseline.predict_proba(X_04_test)[:, 1]
    
    t_low_b, t_high_b, res_b = find_best_3state_thresholds(y_04_test, probs_base, min_coverage=0.94)
    
    print("\n[*] 2. Unified Model (Train on 04 + 05 + 06) - Probabilities")
    X_uni_train = np.vstack([X_04_train, X_dyn])
    y_uni_train = np.hstack([y_04_train, y_dyn])
    
    cb_unified = CatBoostClassifier(iterations=180, learning_rate=0.035, depth=5, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=4)
    cb_unified.fit(X_uni_train, y_uni_train)
    probs_uni = cb_unified.predict_proba(X_04_test)[:, 1]
    
    t_low_u, t_high_u, res_u = find_best_3state_thresholds(y_04_test, probs_uni, min_coverage=0.94)
    
    print("\n================== 3-State F1 Results (Coverage >= 94%) ==================")
    print(f"[A. Baseline (04 Only)]")
    print(f" - Best Thresholds : tau_low={t_low_b:.2f}, tau_high={t_high_b:.2f}")
    if res_b:
        print(f" - 3-State F1      : {res_b['f1']:.2f}%")
        print(f" - Coverage        : {res_b['coverage']:.2f}% ({res_b['n_det']}/{len(y_04_test)})")
        print(f" - 3-State Accuracy: {res_b['acc']:.2f}%")
    else:
        print(" - No valid threshold found for >= 94% coverage.")
        
    print(f"\n[B. Unified (04+05+06)]")
    print(f" - Best Thresholds : tau_low={t_low_u:.2f}, tau_high={t_high_u:.2f}")
    if res_u:
        print(f" - 3-State F1      : {res_u['f1']:.2f}%")
        print(f" - Coverage        : {res_u['coverage']:.2f}% ({res_u['n_det']}/{len(y_04_test)})")
        print(f" - 3-State Accuracy: {res_u['acc']:.2f}%")
    else:
        print(" - No valid threshold found for >= 94% coverage.")
    print("==========================================================================")
    
if __name__ == "__main__":
    main()
