"""
experiment_v4_perfect_frontier.py
v4 SOTA Frontier: 24D Meta Features + View-Specific Submodel Decision Thresholds + View-Adaptive 3-State Gating
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CACHE_OOF_V2_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\integrated\integrated_stiff_oof_cache_v2.npz"

def build_meta_features_24d(oof_sd1, oof_sd2, oof_sd3, views):
    N = len(oof_sd1)
    p1 = oof_sd1
    p2 = np.maximum(oof_sd2, oof_sd1)
    p3 = oof_sd3

    cert1 = np.abs(p1 - 0.5) * 2.0
    cert2 = np.abs(p2 - 0.5) * 2.0
    cert3 = np.abs(p3 - 0.5) * 2.0
    margin_min = np.minimum(np.minimum(cert1, cert2), cert3)
    margin_mean = (cert1 + cert2 + cert3) / 3.0
    margin_std = np.std(np.stack([cert1, cert2, cert3], axis=1), axis=1)

    p_inter_12 = p1 * p2
    p_inter_23 = p2 * p3
    p_inter_13 = p1 * p3
    p_inter_123 = p1 * p2 * p3

    p_clean_joint = (1.0 - p1) * (1.0 - p2) * (1.0 - p3)

    eps = 1e-7
    h1 = - (p1 * np.log2(p1 + eps) + (1.0 - p1) * np.log2(1.0 - p1 + eps))
    h2 = - (p2 * np.log2(p2 + eps) + (1.0 - p2) * np.log2(1.0 - p2 + eps))
    h3 = - (p3 * np.log2(p3 + eps) + (1.0 - p3) * np.log2(1.0 - p3 + eps))
    h_total = h1 + h2 + h3

    v_idx_map = {'view1': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view5': 4}
    v_mat = np.zeros((N, 5), dtype=np.float32)
    for i, v in enumerate(views):
        v_mat[i, v_idx_map.get(str(v), 3)] = 1.0

    p_defect_max = np.maximum(np.maximum(p1, p2), p3)

    meta_X = np.column_stack([
        p1, p2, p3, p_defect_max,
        cert1, cert2, cert3, margin_min, margin_mean, margin_std,
        p_inter_12, p_inter_23, p_inter_13, p_inter_123,
        p_clean_joint,
        h1, h2, h3, h_total,
        v_mat
    ]).astype(np.float32)

    return meta_X, p2


def main():
    cache = np.load(CACHE_OOF_V2_PATH, allow_pickle=True)
    oof_sd1 = cache['oof_sd1']
    oof_sd2 = cache['oof_sd2']
    oof_sd3 = cache['oof_sd3']
    y_sd1 = cache['y_sd1']
    y_sd2 = cache['y_sd2']
    y_sd3 = cache['y_sd3']
    y_clean = cache['y_clean']
    views = cache['views']
    subjects = cache['subjects']
    N = len(oof_sd1)

    # 1. 24D Meta Features
    meta_24d, p_sd2_corr = build_meta_features_24d(oof_sd1, oof_sd2, oof_sd3, views)

    # 2. 5-Fold StratifiedGroupKFold on 24D
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_clean_prob = np.zeros(N, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(meta_24d, y_clean, groups=subjects)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(meta_24d[trn_idx])
        X_va = scaler.transform(meta_24d[val_idx])

        y_tr = y_clean[trn_idx]
        y_va = y_clean[val_idx]

        cb = CatBoostClassifier(iterations=250, learning_rate=0.03, depth=4, l2_leaf_reg=5.0,
                                auto_class_weights='Balanced', random_seed=42+fold, verbose=0, thread_count=1)
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
        oof_clean_prob[val_idx] = cb.predict_proba(X_va)[:, 1]

    auc_24d = roc_auc_score(y_clean, oof_clean_prob) * 100
    print(f"24D CatBoost Meta-Learner AUC: {auc_24d:.2f}%")

    # 3. View-Specific Submodel Decision Thresholds [th1(v), th2(v), th3(v)]
    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_th_map = {}
    pred_s1_v = np.zeros(N, dtype=int)
    pred_s2_v = np.zeros(N, dtype=int)
    pred_s3_v = np.zeros(N, dtype=int)

    for v in view_list:
        v_idx = np.where(views == v)[0]
        b_th = (0.55, 0.60, 0.55)
        b_score = 0.0

        for th1 in np.linspace(0.48, 0.62, 8):
            for th2 in np.linspace(0.52, 0.68, 9):
                for th3 in np.linspace(0.48, 0.62, 8):
                    p1 = (oof_sd1[v_idx] >= th1).astype(int)
                    p2 = (p_sd2_corr[v_idx] >= th2).astype(int)
                    p3 = (oof_sd3[v_idx] >= th3).astype(int)

                    emr = ((p1 == y_sd1[v_idx]) & (p2 == y_sd2[v_idx]) & (p3 == y_sd3[v_idx])).mean() * 100
                    p_c = ((p1 == 0) & (p2 == 0) & (p3 == 0)).astype(int)
                    cf1 = f1_score(y_clean[v_idx], p_c, zero_division=0) * 100

                    # Balanced Objective: 1.0 * EMR + 1.0 * Clean F1
                    score = emr + cf1
                    if score > b_score:
                        b_score = score
                        b_th = (round(float(th1), 3), round(float(th2), 3), round(float(th3), 3))

        view_th_map[v] = b_th
        pred_s1_v[v_idx] = (oof_sd1[v_idx] >= b_th[0]).astype(int)
        pred_s2_v[v_idx] = (p_sd2_corr[v_idx] >= b_th[1]).astype(int)
        pred_s3_v[v_idx] = (oof_sd3[v_idx] >= b_th[2]).astype(int)
        print(f"  {v}: optimal submodel th = {b_th}")

    emr_2s = ((pred_s1_v == y_sd1) & (pred_s2_v == y_sd2) & (pred_s3_v == y_sd3)).mean() * 100
    pred_c_2s = (oof_clean_prob >= 0.5).astype(int)
    cf1_2s = f1_score(y_clean, pred_c_2s, zero_division=0) * 100

    print(f"\n[v4 2-State (View-Specific Submodel Thresholds)]")
    print(f"  - 다중라벨 EMR:   {emr_2s:.2f}% (v3: 38.94%, v2: 39.03%)")
    print(f"  - Clean F1-Score: {cf1_2s:.2f}% (v3: 38.04%, v2: 30.38%)")

    # 4. View-Adaptive 3-State Gating (Strict Coverage >= 90.0% per view)
    print("\n[v4 View-Adaptive 3-State Gating 최적화]")
    v4_gating = {}
    v_mask_v4 = np.zeros(N, dtype=bool)
    pred_c_3s = np.zeros(N, dtype=int)

    for v in view_list:
        v_idx = np.where(views == v)[0]
        y_c_v = y_clean[v_idx]
        p_c_v = oof_clean_prob[v_idx]

        b_tc, b_td = 0.50, 0.50
        b_emr, b_cf1, b_cov = 0.0, 0.0, 100.0

        for tc in np.linspace(0.38, 0.65, 81):
            for td in np.linspace(0.25, tc, 81):
                m_v = (p_c_v >= tc) | (p_c_v < td)
                cov_v = float(m_v.mean()) * 100
                if cov_v < 90.0:  # 전 시점 90% 보장
                    continue

                cf1_v = f1_score(y_c_v[m_v], (p_c_v[m_v] >= tc).astype(int), zero_division=0) * 100
                emr_v = ((pred_s1_v[v_idx][m_v] == y_sd1[v_idx][m_v]) &
                         (pred_s2_v[v_idx][m_v] == y_sd2[v_idx][m_v]) &
                         (pred_s3_v[v_idx][m_v] == y_sd3[v_idx][m_v])).mean() * 100

                combined = emr_v + cf1_v
                if combined > (b_emr + b_cf1):
                    b_emr = emr_v
                    b_cf1 = cf1_v
                    b_tc = float(tc)
                    b_td = float(td)
                    b_cov = float(cov_v)

        v4_gating[v] = {'tau_c': b_tc, 'tau_d': b_td, 'cov': b_cov, 'emr': b_emr, 'cf1': b_cf1}
        print(f"  {v}: tau_c={b_tc:.4f}, tau_d={b_td:.4f}, Cov={b_cov:.2f}%, EMR={b_emr:.2f}%, Clean F1={b_cf1:.2f}%")

        m_opt = (p_c_v >= b_tc) | (p_c_v < b_td)
        v_mask_v4[v_idx[m_opt]] = True
        pred_c_3s[v_idx[m_opt]] = (p_c_v[m_opt] >= b_tc).astype(int)

    total_cov = float(v_mask_v4.mean()) * 100
    total_emr = ((pred_s1_v[v_mask_v4] == y_sd1[v_mask_v4]) &
                 (pred_s2_v[v_mask_v4] == y_sd2[v_mask_v4]) &
                 (pred_s3_v[v_mask_v4] == y_sd3[v_mask_v4])).mean() * 100
    total_cf1 = f1_score(y_clean[v_mask_v4], pred_c_3s[v_mask_v4], zero_division=0) * 100
    tp_clean = int(((pred_c_3s[v_mask_v4] == 1) & (y_clean[v_mask_v4] == 1)).sum())

    print("\n" + "="*80)
    print("  [v4 SOTA Frontier 종합 성능 결과]")
    print(f"  - 전체 커버리지:   {total_cov:.2f}% (판단 보류율: {100-total_cov:.2f}%)")
    print(f"  - 3-State EMR:     {total_emr:.2f}%")
    print(f"  - 3-State Clean F1:{total_cf1:.2f}% (정상 맞춤: {tp_clean} / {y_clean.sum()}건)")
    print("="*80)

if __name__ == "__main__":
    main()
