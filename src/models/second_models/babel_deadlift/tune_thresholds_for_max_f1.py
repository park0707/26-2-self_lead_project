# -*- coding: utf-8 -*-
import os
import sys

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

from src.models.second_models.babel_deadlift.train_and_eval_d3_v5_3 import (
    prepare_v5_3_clean_dataset,
    VIEW_CODE_LIST,
    VIEW_ALIAS_MAP
)

# Load predictions from v5.3 training
def run_threshold_optimization():
    # We can load data and run evaluation
    data = np.load(r'd:\src\models\second_models\babel_deadlift\d3_v5_dataset_cache.npz', allow_pickle=True)
    y = data['y']
    views = data['views']
    
    with open(r'd:\src\models\second_models\babel_deadlift\d3_model_v5_2_evaluation_results.json', 'r', encoding='utf-8') as f:
        d2 = json.load(f)
    p_v5_2 = np.array(d2['oof_probs'])
    
    print("=== v5.2 OOF Threshold Grid Test ===")
    for vcode in VIEW_CODE_LIST:
        v_idx = np.where(views == vcode)[0]
        y_v = y[v_idx]
        p_v = p_v5_2[v_idx]
        
        best_f1 = 0
        best_cfg = None
        for tn in np.arange(0.25, 0.45, 0.005):
            for td in np.arange(0.42, 0.65, 0.005):
                if tn >= td:
                    continue
                determ = (p_v <= tn) | (p_v >= td)
                cov = np.mean(determ)
                if cov < 0.90:
                    continue
                pred = np.where(p_v[determ] >= td, 1, 0)
                if len(np.unique(y_v[determ])) < 2 or np.sum(pred == 1) == 0:
                    continue
                f1 = f1_score(y_v[determ], pred, zero_division=0)
                prec = precision_score(y_v[determ], pred, zero_division=0)
                rec = recall_score(y_v[determ], pred, zero_division=0)
                if f1 > best_f1 and prec >= 0.50:
                    best_f1 = f1
                    best_cfg = (tn, td, f1, prec, rec, cov)
        if best_cfg:
            print(f"[{VIEW_ALIAS_MAP[vcode]}] F1: {best_cfg[2]*100:.2f}%, Prec: {best_cfg[3]*100:.2f}%, Rec: {best_cfg[4]*100:.2f}%, Cov: {best_cfg[5]*100:.2f}%, tn={best_cfg[0]:.3f}, td={best_cfg[1]:.3f}")

if __name__ == '__main__':
    run_threshold_optimization()
