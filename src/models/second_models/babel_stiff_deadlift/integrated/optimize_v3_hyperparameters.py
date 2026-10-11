"""
optimize_v3_hyperparameters.py
v3 통합 진단 모델 최적화 탐색:
- 24D 메타 피처 기반 Meta-Learner 블렌딩 (CatBoost 가중 증대)
- 엄격한 커버리지 제약 (전체 Coverage >= 90.0%, 즉 보류율 <= 10.0%)
- Submodel Thresholds 정밀 최적화
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CACHE_OOF_V2_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\integrated\integrated_stiff_oof_cache_v2.npz"

class StiffMetaNN(nn.Module):
    def __init__(self, in_features: int = 24, hidden_dim: int = 32):
        super(StiffMetaNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 16),
            nn.BatchNorm1d(16),
            nn.Mish(),
            nn.Dropout(0.15),
            nn.Linear(16, 8),
            nn.Mish(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_meta_features_v3(oof_sd1, oof_sd2, oof_sd3, views):
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

    meta_X, p_sd2_corr = build_meta_features_v3(oof_sd1, oof_sd2, oof_sd3, views)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_nn = np.zeros(N, dtype=np.float32)
    oof_cb = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(meta_X, y_clean, groups=subjects)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(meta_X[trn_idx])
        X_va = scaler.transform(meta_X[val_idx])

        y_tr = y_clean[trn_idx]
        y_va = y_clean[val_idx]

        # CatBoost
        cb = CatBoostClassifier(iterations=250, learning_rate=0.03, depth=4, l2_leaf_reg=5.0,
                                auto_class_weights='Balanced', random_seed=42+fold, verbose=0, thread_count=1)
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
        oof_cb[val_idx] = cb.predict_proba(X_va)[:, 1]

        # LightGBM
        lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=4, num_leaves=12,
                                  class_weight='balanced', random_state=42+fold, verbose=-1, n_jobs=1)
        lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_lgb[val_idx] = lgbm.predict_proba(X_va)[:, 1]

        # StiffMetaNN
        torch.manual_seed(42 + fold)
        model = StiffMetaNN(in_features=24, hidden_dim=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], dtype=torch.float32))

        ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(35):
            for bx, by in loader_tr:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(torch.tensor(X_va, dtype=torch.float32))
            oof_nn[val_idx] = torch.sigmoid(va_logits).numpy()

    # Blends
    blends = {
        'pure_cb': oof_cb,
        'cb60_nn25_lgb15': 0.60 * oof_cb + 0.25 * oof_nn + 0.15 * oof_lgb,
        'cb70_nn30': 0.70 * oof_cb + 0.30 * oof_nn,
        'cb50_nn30_lgb20': 0.50 * oof_cb + 0.30 * oof_nn + 0.20 * oof_lgb
    }

    for name, prob in blends.items():
        print(f"[{name}] AUC: {roc_auc_score(y_clean, prob)*100:.2f}%")

    # Pick the best blend
    best_blend_name = max(blends.keys(), key=lambda k: roc_auc_score(y_clean, blends[k]))
    best_p_clean = blends[best_blend_name]
    print(f"\n>> Selected Best Meta Probability: {best_blend_name} (AUC: {roc_auc_score(y_clean, best_p_clean)*100:.2f}%)")

    # Submodel threshold fine grid
    th1_candidates = [0.50, 0.52, 0.55, 0.57]
    th2_candidates = [0.55, 0.58, 0.60, 0.62]
    th3_candidates = [0.50, 0.52, 0.55, 0.57]

    best_th = (0.55, 0.60, 0.55)
    best_emr = 0.0
    best_cf1 = 0.0

    for th1 in th1_candidates:
        for th2 in th2_candidates:
            for th3 in th3_candidates:
                p_s1 = (oof_sd1 >= th1).astype(int)
                p_s2 = (p_sd2_corr >= th2).astype(int)
                p_s3 = (oof_sd3 >= th3).astype(int)

                emr = ((p_s1 == y_sd1) & (p_s2 == y_sd2) & (p_s3 == y_sd3)).mean() * 100
                pred_c = ((p_s1 == 0) & (p_s2 == 0) & (p_s3 == 0)).astype(int)
                cf1 = f1_score(y_clean, pred_c, zero_division=0) * 100

                if (emr + cf1) > (best_emr + best_cf1):
                    best_emr = emr
                    best_cf1 = cf1
                    best_th = (th1, th2, th3)

    print(f"Optimal Submodel Thresholds: th1={best_th[0]}, th2={best_th[1]}, th3={best_th[2]}")
    print(f"  -> Submodel-based EMR: {best_emr:.2f}%, Clean F1: {best_cf1:.2f}%")

    opt_s1 = (oof_sd1 >= best_th[0]).astype(int)
    opt_s2 = (p_sd2_corr >= best_th[1]).astype(int)
    opt_s3 = (oof_sd3 >= best_th[2]).astype(int)

    # 1. Global 3-State Gating (Strict Coverage >= 90.0%)
    print("\n--- 1. Global 3-State Gating Optimization (Coverage >= 90.0%) ---")
    g_best_tc, g_best_td = 0.50, 0.50
    g_best_emr, g_best_cf1, g_best_cov = 0.0, 0.0, 100.0

    for tc in np.linspace(0.38, 0.58, 51):
        for td in np.linspace(0.28, tc, 51):
            mask = (best_p_clean >= tc) | (best_p_clean < td)
            cov = float(mask.mean()) * 100
            if cov < 90.0:
                continue

            y_c_eval = y_clean[mask]
            pred_c_eval = (best_p_clean[mask] >= tc).astype(int)
            cf1 = f1_score(y_c_eval, pred_c_eval, zero_division=0) * 100

            emr = ((opt_s1[mask] == y_sd1[mask]) & (opt_s2[mask] == y_sd2[mask]) & (opt_s3[mask] == y_sd3[mask])).mean() * 100

            if (cf1 + emr) > (g_best_cf1 + g_best_emr):
                g_best_cf1 = cf1
                g_best_emr = emr
                g_best_tc = float(tc)
                g_best_td = float(td)
                g_best_cov = float(cov)

    print(f"Global Gating: tau_c={g_best_tc:.4f}, tau_d={g_best_td:.4f}, Coverage={g_best_cov:.2f}% (보류율: {100-g_best_cov:.2f}%)")
    print(f"  -> 3-State EMR: {g_best_emr:.2f}%, Clean F1: {g_best_cf1:.2f}%")

    # 2. View-Adaptive 3-State Gating (Strict Coverage >= 90.0% per view)
    print("\n--- 2. View-Adaptive 3-State Gating (Each View Coverage >= 90.0%) ---")
    view_gating_strict = {}
    v_mask_strict = np.zeros(N, dtype=bool)
    pred_c_strict = np.zeros(N, dtype=int)

    for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
        v_idx = np.where(views == v)[0]
        y_c_v = y_clean[v_idx]
        p_v = best_p_clean[v_idx]

        b_tc, b_td = 0.50, 0.50
        b_emr, b_cf1, b_cov = 0.0, 0.0, 100.0

        for tc in np.linspace(0.38, 0.62, 61):
            for td in np.linspace(0.25, tc, 61):
                m_v = (p_v >= tc) | (p_v < td)
                cov_v = float(m_v.mean()) * 100
                if cov_v < 90.0:  # Strict 90% per view!
                    continue

                cf1_v = f1_score(y_c_v[m_v], (p_v[m_v] >= tc).astype(int), zero_division=0) * 100
                emr_v = ((opt_s1[v_idx][m_v] == y_sd1[v_idx][m_v]) &
                         (opt_s2[v_idx][m_v] == y_sd2[v_idx][m_v]) &
                         (opt_s3[v_idx][m_v] == y_sd3[v_idx][m_v])).mean() * 100

                if (cf1_v + emr_v) > (b_cf1 + b_emr):
                    b_cf1 = cf1_v
                    b_emr = emr_v
                    b_tc = float(tc)
                    b_td = float(td)
                    b_cov = float(cov_v)

        view_gating_strict[v] = {'tc': b_tc, 'td': b_td, 'cov': b_cov, 'emr': b_emr, 'cf1': b_cf1}
        print(f"  {v}: tau_c={b_tc:.4f}, tau_d={b_td:.4f}, Cov={b_cov:.2f}%, EMR={b_emr:.2f}%, Clean F1={b_cf1:.2f}%")

        m_v_best = (p_v >= b_tc) | (p_v < b_td)
        v_mask_strict[v_idx[m_v_best]] = True
        pred_c_strict[v_idx[m_v_best]] = (p_v[m_v_best] >= b_tc).astype(int)

    total_cov_strict = float(v_mask_strict.mean()) * 100
    total_emr_strict = ((opt_s1[v_mask_strict] == y_sd1[v_mask_strict]) &
                        (opt_s2[v_mask_strict] == y_sd2[v_mask_strict]) &
                        (opt_s3[v_mask_strict] == y_sd3[v_mask_strict])).mean() * 100
    total_cf1_strict = f1_score(y_clean[v_mask_strict], pred_c_strict[v_mask_strict], zero_division=0) * 100

    print(f"\n[View-Adaptive Strict 90% 종합]")
    print(f"  전체 Coverage: {total_cov_strict:.2f}% (보류율: {100-total_cov_strict:.2f}%)")
    print(f"  3-State EMR:   {total_emr_strict:.2f}%")
    print(f"  3-State Clean F1: {total_cf1_strict:.2f}%")

if __name__ == "__main__":
    main()
