"""
experiment_v3_architectures.py
바벨 스티프 데드리프트 통합 진단 모델 v3 개발을 위한 아키텍처 실험 스크립트
1. 24D 메타 피처 (엔트로피, 교차 결합 상호작용, 마진 통계)
2. Tri-Ensemble Meta-Learner (StiffMetaNN + CatBoost + LightGBM)
3. 서브모델 적응형 임계치 튜닝 (Submodel Adaptive Thresholding)
4. 시점 적응형 3-State 완충망 (View-Adaptive Selective Gating)
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

# 환경 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CACHE_OOF_V2_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\integrated\integrated_stiff_oof_cache_v2.npz"

class StiffMetaNN(nn.Module):
    """v3 24D 심층 메타 신경망 (24D -> 32 -> 16 -> 8 -> 1)"""
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
    """24차원 고도화 메타 피처 행렬 구축"""
    N = len(oof_sd1)
    
    # 생체역학 전제: sd1=1 -> sd2=1 수반 보정
    p1 = oof_sd1
    p2 = np.maximum(oof_sd2, oof_sd1)
    p3 = oof_sd3

    # 1. 확신성 마진 (Certainty Margin)
    cert1 = np.abs(p1 - 0.5) * 2.0
    cert2 = np.abs(p2 - 0.5) * 2.0
    cert3 = np.abs(p3 - 0.5) * 2.0
    margin_min = np.minimum(np.minimum(cert1, cert2), cert3)
    margin_mean = (cert1 + cert2 + cert3) / 3.0
    margin_std = np.std(np.stack([cert1, cert2, cert3], axis=1), axis=1)

    # 2. 교차 결함 상호작용 (Cross-Defect Interactions)
    p_inter_12 = p1 * p2
    p_inter_23 = p2 * p3
    p_inter_13 = p1 * p3
    p_inter_123 = p1 * p2 * p3

    # 3. 결합 정상 확률곱 (Clean Joint Probability)
    p_clean_joint = (1.0 - p1) * (1.0 - p2) * (1.0 - p3)

    # 4. 샤논 정보 엔트로피 (Shannon Entropy) - 불확실성 정량화
    eps = 1e-7
    h1 = - (p1 * np.log2(p1 + eps) + (1.0 - p1) * np.log2(1.0 - p1 + eps))
    h2 = - (p2 * np.log2(p2 + eps) + (1.0 - p2) * np.log2(1.0 - p2 + eps))
    h3 = - (p3 * np.log2(p3 + eps) + (1.0 - p3) * np.log2(1.0 - p3 + eps))
    h_total = h1 + h2 + h3

    # 5. 시점 원-핫 (view1~view5)
    v_idx_map = {'view1': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view5': 4}
    v_mat = np.zeros((N, 5), dtype=np.float32)
    for i, v in enumerate(views):
        v_mat[i, v_idx_map.get(str(v), 3)] = 1.0

    # 24차원 메타 피처 결합:
    # 3 (p1, p2, p3) + 5 (cert1, cert2, cert3, min, mean, std) - wait, cert1..3 + min + mean + std = 6
    # + 4 interactions (12, 23, 13, 123)
    # + 1 clean_joint
    # + 4 entropy (h1, h2, h3, h_total)
    # + 5 views
    # Total = 3 + 6 + 4 + 1 + 4 + 5 = 23 + 1 (최대 결함 확률 max(p1,p2,p3)) = 24D
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


def run_experiment():
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

    print(f"Total samples: {N}, Clean count: {y_clean.sum()} ({y_clean.sum()/N*100:.2f}%)")

    # 1. 24D 메타 피처 구축
    meta_X, p_sd2_corr = build_meta_features_v3(oof_sd1, oof_sd2, oof_sd3, views)
    print(f"Meta feature shape: {meta_X.shape}")

    # 2. 피험자 격리 5-Fold StratifiedGroupKFold
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

        # Model A: CatBoost Meta-Learner
        cb = CatBoostClassifier(iterations=250, learning_rate=0.03, depth=4, l2_leaf_reg=5.0,
                                auto_class_weights='Balanced', random_seed=42+fold, verbose=0, thread_count=1)
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
        oof_cb[val_idx] = cb.predict_proba(X_va)[:, 1]

        # Model B: LightGBM Meta-Learner
        lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=4, num_leaves=12,
                                  class_weight='balanced', random_state=42+fold, verbose=-1, n_jobs=1)
        lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_lgb[val_idx] = lgbm.predict_proba(X_va)[:, 1]

        # Model C: StiffMetaNN v3 (24D)
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

    # Model evaluation
    print("\n--- Meta-Learner AUC Comparison ---")
    print(f"CatBoost Meta AUC: {roc_auc_score(y_clean, oof_cb)*100:.2f}%")
    print(f"LightGBM Meta AUC: {roc_auc_score(y_clean, oof_lgb)*100:.2f}%")
    print(f"StiffMetaNN Meta AUC: {roc_auc_score(y_clean, oof_nn)*100:.2f}%")

    # 3-Way Ensemble (NN 0.4, CB 0.35, LGB 0.25)
    oof_tri = 0.40 * oof_nn + 0.35 * oof_cb + 0.25 * oof_lgb
    print(f"Tri-Ensemble Meta AUC: {roc_auc_score(y_clean, oof_tri)*100:.2f}%")

    # 3. Submodel Threshold Optimization
    print("\n--- Submodel Threshold Optimization ---")
    best_th = (0.5, 0.5, 0.5)
    best_emr = 0.0
    best_clean_f1 = 0.0

    # Grid search for submodel thresholds
    th_range = [0.45, 0.50, 0.55, 0.60]
    for th1 in th_range:
        for th2 in th_range:
            for th3 in th_range:
                p_s1 = (oof_sd1 >= th1).astype(int)
                p_s2 = (p_sd2_corr >= th2).astype(int)
                p_s3 = (oof_sd3 >= th3).astype(int)

                emr = ((p_s1 == y_sd1) & (p_s2 == y_sd2) & (p_s3 == y_sd3)).mean() * 100
                pred_c = ((p_s1 == 0) & (p_s2 == 0) & (p_s3 == 0)).astype(int)
                cf1 = f1_score(y_clean, pred_c, zero_division=0) * 100

                if (emr + cf1) > (best_emr + best_clean_f1):
                    best_emr = emr
                    best_clean_f1 = cf1
                    best_th = (th1, th2, th3)

    print(f"Optimal Submodel Thresholds: th1={best_th[0]}, th2={best_th[1]}, th3={best_th[2]}")
    print(f"  -> Independent EMR: {best_emr:.2f}%, Clean F1: {best_clean_f1:.2f}%")

    # Using optimal submodel thresholds
    opt_pred_s1 = (oof_sd1 >= best_th[0]).astype(int)
    opt_pred_s2 = (p_sd2_corr >= best_th[1]).astype(int)
    opt_pred_s3 = (oof_sd3 >= best_th[2]).astype(int)

    # 4. View-Adaptive Selective Gating vs Global Gating
    print("\n--- 3-State Gating Optimization ---")

    # A: Global Gating with Tri-Ensemble
    best_gtc, best_gtd = 0.50, 0.50
    best_g_emr = 0.0
    best_g_cf1 = 0.0
    best_g_cov = 100.0

    for tc in np.linspace(0.40, 0.60, 41):
        for td in np.linspace(0.30, tc, 41):
            mask = (oof_tri >= tc) | (oof_tri < td)
            cov = float(mask.mean()) * 100
            if cov < 90.0:
                continue

            y_c_eval = y_clean[mask]
            pred_c_eval = (oof_tri[mask] >= tc).astype(int)
            cf1 = f1_score(y_c_eval, pred_c_eval, zero_division=0) * 100

            emr = ((opt_pred_s1[mask] == y_sd1[mask]) & (opt_pred_s2[mask] == y_sd2[mask]) & (opt_pred_s3[mask] == y_sd3[mask])).mean() * 100

            if (cf1 + emr) > (best_g_cf1 + best_g_emr):
                best_g_cf1 = cf1
                best_g_emr = emr
                best_gtc = float(tc)
                best_gtd = float(td)
                best_g_cov = float(cov)

    print(f"[Global Gating (Tri-Ensemble)]")
    print(f"  tau_clean: {best_gtc:.4f}, tau_defect: {best_gtd:.4f}, Coverage: {best_g_cov:.2f}%")
    print(f"  3-State EMR: {best_g_emr:.2f}%, Clean F1: {best_g_cf1:.2f}%")

    # B: View-Adaptive Gating
    print("\n[View-Adaptive Gating Optimization]")
    view_gating = {}
    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']

    overall_mask = np.zeros(N, dtype=bool)
    pred_clean_adaptive = np.zeros(N, dtype=int)

    for v in view_list:
        v_idx = np.where(views == v)[0]
        y_c_v = y_clean[v_idx]
        p_tri_v = oof_tri[v_idx]

        b_tc, b_td = 0.50, 0.50
        b_emr = 0.0
        b_cf1 = 0.0
        b_cov = 100.0

        for tc in np.linspace(0.38, 0.62, 49):
            for td in np.linspace(0.28, tc, 49):
                m_v = (p_tri_v >= tc) | (p_tri_v < td)
                cov_v = float(m_v.mean()) * 100
                if cov_v < 85.0: # 시점별 최소 85% 보장
                    continue

                cf1_v = f1_score(y_c_v[m_v], (p_tri_v[m_v] >= tc).astype(int), zero_division=0) * 100
                emr_v = ((opt_pred_s1[v_idx][m_v] == y_sd1[v_idx][m_v]) &
                         (opt_pred_s2[v_idx][m_v] == y_sd2[v_idx][m_v]) &
                         (opt_pred_s3[v_idx][m_v] == y_sd3[v_idx][m_v])).mean() * 100

                if (cf1_v + emr_v) > (b_cf1 + b_emr):
                    b_cf1 = cf1_v
                    b_emr = emr_v
                    b_tc = float(tc)
                    b_td = float(td)
                    b_cov = float(cov_v)

        view_gating[v] = {'tc': b_tc, 'td': b_td, 'cov': b_cov, 'emr': b_emr, 'cf1': b_cf1}
        print(f"  {v}: tau_c={b_tc:.4f}, tau_d={b_td:.4f}, Coverage={b_cov:.2f}%, EMR={b_emr:.2f}%, Clean F1={b_cf1:.2f}%")

        # Global array update
        m_v_best = (p_tri_v >= b_tc) | (p_tri_v < b_td)
        overall_mask[v_idx[m_v_best]] = True
        pred_clean_adaptive[v_idx[m_v_best]] = (p_tri_v[m_v_best] >= b_tc).astype(int)

    total_cov = float(overall_mask.mean()) * 100
    total_emr_v3 = ((opt_pred_s1[overall_mask] == y_sd1[overall_mask]) &
                    (opt_pred_s2[overall_mask] == y_sd2[overall_mask]) &
                    (opt_pred_s3[overall_mask] == y_sd3[overall_mask])).mean() * 100
    total_cf1_v3 = f1_score(y_clean[overall_mask], pred_clean_adaptive[overall_mask], zero_division=0) * 100

    print("\n" + "="*80)
    print("  [v3 View-Adaptive Selective Gating 종합 결과]")
    print(f"  - 전체 커버리지: {total_cov:.2f}% (보류율: {100.0 - total_cov:.2f}%)")
    print(f"  - 3-State EMR:   {total_emr_v3:.2f}%")
    print(f"  - 3-State Clean F1: {total_cf1_v3:.2f}%")
    print("="*80)

if __name__ == "__main__":
    run_experiment()
