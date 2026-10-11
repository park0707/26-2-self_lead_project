# -*- coding: utf-8 -*-
"""
[vidio-health-trainer] 저해 피처 및 중복 피처 제거 시뮬레이션 (Feature Pruning Simulation)
"""

import os
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

DATA_PATH = r"D:\Data\피처_데이터\lunge_c5_v5_36d_features.npz"

def eval_3state(probs, y_true, max_rej=0.10):
    best = {'f1': 0.0, 'th_low': 0.35, 'th_high': 0.45, 'rejection_rate': 0.0, 'recall': 0.0, 'cm': None}
    for th_l in np.arange(0.28, 0.45, 0.01):
        for th_h in np.arange(th_l + 0.02, 0.55, 0.01):
            mask = (probs < th_l) | (probs >= th_h)
            rej = 1.0 - np.mean(mask)
            if rej > max_rej:
                continue
            y_c = y_true[mask]
            p_c = (probs[mask] >= th_h).astype(int)
            if len(np.unique(p_c)) < 2:
                continue
            f1 = f1_score(y_c, p_c, zero_division=0)
            rec = np.sum((y_c == 1) & (p_c == 1)) / (np.sum(y_c == 1) + 1e-6)
            if f1 > best['f1']:
                cm = confusion_matrix(y_c, p_c)
                best = {
                    'f1': float(f1),
                    'th_low': float(th_l),
                    'th_high': float(th_h),
                    'rejection_rate': float(rej),
                    'recall': float(rec),
                    'cm': cm.tolist()
                }
    return best

def simulate_pruning():
    data = np.load(DATA_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    feat_names = list(data['feature_names'])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_featureset(selected_indices, name=""):
        X_sub = X[:, selected_indices]
        N = X.shape[0]
        oof_matrix = np.zeros((N, 5))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups=subjects), 1):
            X_tr, y_tr = X_sub[train_idx], y[train_idx]
            X_va, y_va = X_sub[val_idx], y[val_idx]
            sw = np.where(y_tr == 1, 1.7789, 1.0)

            # 1. LGB
            m_lgb = lgb.LGBMClassifier(n_estimators=160, max_depth=5, num_leaves=24, learning_rate=0.035, random_state=42+fold, verbose=-1, n_jobs=1)
            m_lgb.fit(X_tr, y_tr, sample_weight=sw)
            oof_matrix[val_idx, 0] = m_lgb.predict_proba(X_va)[:, 1]

            # 2. CB
            m_cb = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.035, auto_class_weights='Balanced', random_seed=42+fold, verbose=0, thread_count=1)
            m_cb.fit(X_tr, y_tr)
            oof_matrix[val_idx, 1] = m_cb.predict_proba(X_va)[:, 1]

            # 3. XGB
            m_xgb = xgb.XGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.035, scale_pos_weight=1.7789, random_state=42+fold, n_jobs=1, eval_metric='logloss')
            m_xgb.fit(X_tr, y_tr, sample_weight=sw)
            oof_matrix[val_idx, 2] = m_xgb.predict_proba(X_va)[:, 1]

            # 4. RF
            m_rf = RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=42+fold, n_jobs=1)
            m_rf.fit(X_tr, y_tr, sample_weight=sw)
            oof_matrix[val_idx, 3] = m_rf.predict_proba(X_va)[:, 1]

            # 5. ET
            m_et = ExtraTreesClassifier(n_estimators=180, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=42+fold, n_jobs=1)
            m_et.fit(X_tr, y_tr, sample_weight=sw)
            oof_matrix[val_idx, 4] = m_et.predict_proba(X_va)[:, 1]

        blended = (0.20 * oof_matrix[:, 0] + 0.30 * oof_matrix[:, 1] + 0.30 * oof_matrix[:, 2] + 0.10 * oof_matrix[:, 3] + 0.10 * oof_matrix[:, 4])
        eps = 1e-6
        cp = np.clip(blended, eps, 1.0 - eps)
        logits = np.log(cp / (1.0 - cp))
        final_probs = 1.0 / (1.0 + np.exp(-logits / 1.02))

        auc = roc_auc_score(y, final_probs)
        res = eval_3state(final_probs, y, max_rej=0.10)
        print(f"[*] {name:<35} (D={len(selected_indices)}): 3-State F1={res['f1']*100:.2f}%, AUC={auc:.4f}, Rec={res['recall']*100:.2f}%, Rej={res['rejection_rate']*100:.2f}%")
        return res, auc

    print("=" * 75)
    print("  피처 정제 및 저해 피처 제거 시뮬레이션 (Feature Pruning Simulation)")
    print("=" * 75)

    all_indices = list(range(36))
    res_base, auc_base = evaluate_featureset(all_indices, "Baseline (전체 36D)")

    # 1. 성능 저해 피처(34번 spectral_energy_ratio, 28번 baseline_subtracted_peak) 제거 -> 34D
    prune_1 = [i for i in all_indices if i not in [33, 27]]  # 0-indexed: 34-1=33, 28-1=27
    res_p1, auc_p1 = evaluate_featureset(prune_1, "Step 1: 저해 피처 2개 제거 (34D)")

    # 2. 극단적 고상관 중복 피처 추가 제거:
    # 22번(ascent_jerk, 32번 burst_jerk와 r=0.983 -> 22번 제거)
    # 15번(calibrated_delta_z, 14번 foreshortening과 r=0.970 -> 15번 제거)
    # 7번(ldc_phase_weighted, 1번 pitch_bottom과 r=0.938, VIF=144.6 -> 7번 제거)
    prune_2 = [i for i in prune_1 if i not in [21, 14, 6]]  # 22-1=21, 15-1=14, 7-1=6 -> 31D
    res_p2, auc_p2 = evaluate_featureset(prune_2, "Step 2: 저해+극단적 중복 5개 제거 (31D)")

    # 3. 중요도 최하위 무기여 피처(4번 farrokhi_flexion, 18번 frontal_compression) 추가 제거 -> 29D
    prune_3 = [i for i in prune_2 if i not in [3, 17]]  # 4-1=3, 18-1=17 -> 29D
    res_p3, auc_p3 = evaluate_featureset(prune_3, "Step 3: 저해+중복+무기여 7개 제거 (29D)")

    sim_results = {
        'baseline_36d': {'f1': res_base['f1']*100, 'auc': auc_base, 'dim': 36},
        'prune_step1_34d': {'f1': res_p1['f1']*100, 'auc': auc_p1, 'dim': 34, 'gain': (res_p1['f1'] - res_base['f1'])*100},
        'prune_step2_31d': {'f1': res_p2['f1']*100, 'auc': auc_p2, 'dim': 31, 'gain': (res_p2['f1'] - res_base['f1'])*100},
        'prune_step3_29d': {'f1': res_p3['f1']*100, 'auc': auc_p3, 'dim': 29, 'gain': (res_p3['f1'] - res_base['f1'])*100},
    }

    with open(r"D:\src\features\lunge_c5_pruning_simulation_results.json", 'w', encoding='utf-8') as f:
        json.dump(sim_results, f, indent=2)

if __name__ == '__main__':
    simulate_pruning()
