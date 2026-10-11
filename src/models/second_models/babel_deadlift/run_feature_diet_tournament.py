# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반
Script: run_feature_diet_tournament.py
Description: 피처 다이어트(Feature Diet) 5단계 토너먼트
             43D -> 31D -> 26D -> 22D -> 16D
             Two-Stream Meta Fusion (GBDT + 1D-TCN) 5-Fold StratifiedGroupKFold OOF 평가
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CURRENT_DIR, 'd4_v4_features_cache.npz')

from d4_model_v5 import FEATURE_NAMES_43_D4

# 12채널 1D-TCN 아키텍처
class KinematicTCNLocal(nn.Module):
    def __init__(self, in_channels=12, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)

def train_tcn_quick(X_tr, y_tr, X_va, y_va, epochs=35, lr=0.003, batch_size=32, device='cpu'):
    model = KinematicTCNLocal(in_channels=12, num_classes=1).to(device)
    pos_weight = torch.tensor([(len(y_tr) - sum(y_tr)) / max(sum(y_tr), 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_weights = None
    
    model.train()
    for ep in range(epochs):
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        vx = torch.tensor(X_va, dtype=torch.float32).to(device)
        val_logits = model(vx)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        
    return val_probs

def run_tournament():
    print("=" * 90)
    print(" [vidio-health-trainer] 바벨 데드리프트 d4 결함 모델 피처 다이어트 토너먼트")
    print("=" * 90)
    
    data = np.load(CACHE_PATH, allow_pickle=True)
    X_gbdt_all = data['X_gbdt']   # (770, 43)
    X_tcn_all = data['X_tcn']     # (770, 12, 64)
    y_all = data['y']             # (770,)
    subjects = data['subjects']   # (770,)
    views = data['views']         # (770,)
    
    N, D_all = X_gbdt_all.shape
    
    # 순열 중요도 결과에서 확인된 음수 피처 12개
    negative_features = [
        'd4_hip_bar_decoupling',         # [41] -0.85%
        'd4_hip_shoot_up_speed_norm',    # [9]  -0.78%
        'd4_crp_std_ascent_deg',         # [30] -0.77%
        'd4_hip_shoulder_vert_dist_min', # [12] -0.63%
        'd4_crp_max_drift_deg',          # [31] -0.33%
        'd4_mvc_knee_dominant_ratio',    # [32] -0.30%
        'd4_bilateral_knee_asym_deg',    # [19] -0.21%
        'd4_ratio_std_ascent',           # [26] -0.18%
        'd4_hip_rom_deg',                # [17] -0.12%
        'd4_morph_torso_thigh_ratio',    # [36] -0.10%
        'd4_hs_vvr_first_pull_max',      # [37] -0.09%
        'd4_crp_mean_first_pull_deg'     # [29] -0.05%
    ]
    
    # 5단계 피처셋 정의
    tier1_indices = list(range(43)) # 43D 베이스라인
    
    # Tier 2: 음수 피처 12개 완전 제거 (31D)
    tier2_indices = [i for i, name in enumerate(FEATURE_NAMES_43_D4) if name not in negative_features]
    
    # Tier 3: 26D (Tier 2에서 고상관 중복 5개 추가 압축)
    # 고상관/낮은기여: d4_ratio_max_ascent, d4_ratio_delta_first_pull, d4_ascent_duration_ratio, d4_morph_femur_tibia_ratio, d4_pelvis_y_rise_rate
    tier3_removals = [
        'd4_ratio_max_ascent',           # r=0.96 with ratio_deriv_max
        'd4_ratio_delta_first_pull',     # r=0.90 with ratio_deriv_max
        'd4_ascent_duration_ratio',      # ΔF1 +0.15% (미미)
        'd4_morph_femur_tibia_ratio',    # ΔF1 +0.19% (체형 비율 단독으론 노이즈)
        'd4_view_robust_coupling'        # ΔF1 +0.12% (미미)
    ]
    tier3_indices = [i for i in tier2_indices if FEATURE_NAMES_43_D4[i] not in tier3_removals]
    
    # Tier 4: 21D (Tier 3에서 추가 경량화 - 상위 21개 알짜 피처)
    tier4_removals = [
        'd4_accel_lead_norm',
        'd4_quasi_static_trunk_moment',
        'd4_pelvis_y_rise_rate',
        'd4_vel_ratio_full_ascent',
        'd4_bilateral_hip_asym_deg'
    ]
    tier4_indices = [i for i in tier3_indices if FEATURE_NAMES_43_D4[i] not in tier4_removals]
    
    # Tier 5: 15D (극초경량 Top 15 피처)
    tier5_indices = [
        14, # d4_hip_lock_time_norm (+4.19%)
        0,  # d4_knee_hip_corr_full (+2.98%)
        6,  # d4_angular_accel_mismatch (+2.89%)
        33, # d4_mvc_co_extension_ratio (+2.71%)
        13, # d4_knee_lock_time_norm (+1.76%)
        28, # d4_norm_progress_diff_max (+1.60%)
        38, # d4_onset_latency_norm (+1.60%)
        18, # d4_knee_hip_rom_ratio (+1.50%)
        8,  # d4_trunk_drop_ascent_max_deg (+1.43%)
        15, # d4_lockout_phase_lag_norm (+1.41%)
        3,  # d4_vel_ratio_second_half (+1.19%)
        10, # d4_torso_horizontal_ratio (+1.14%)
        5,  # d4_extension_rate_diff_max (+1.05%)
        1,  # d4_knee_hip_corr_ascent (+1.02%)
        7   # d4_trunk_angle_nadir_deg (+0.57%)
    ]
    
    candidates = [
        ("Tier 1: 43D (전체 피처 베이스라인)", tier1_indices),
        ("Tier 2: 31D (음수/노이즈 12개 제거)", tier2_indices),
        ("Tier 3: 26D (고상관 중복 압축)", tier3_indices),
        ("Tier 4: 21D (정예 핵심 피처셋)", tier4_indices),
        ("Tier 5: 15D (초경량 코어 15 피처)", tier5_indices),
    ]
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # TCN OOF 사전 산출 (TCN은 피처 선택과 무관하므로 1회 계산 후 재사용)
    print("\n[Step 0] 1D-TCN 5-Fold OOF 예측 사전 산출...")
    tcn_oof = np.zeros(N)
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_gbdt_all, y_all, groups=subjects)):
        tcn_pred = train_tcn_quick(X_tcn_all[tr_idx], y_all[tr_idx], X_tcn_all[va_idx], y_all[va_idx], epochs=35)
        tcn_oof[va_idx] = tcn_pred
        
    tcn_auc = roc_auc_score(y_all, tcn_oof)
    print(f"  -> 1D-TCN 단독 OOF AUC: {tcn_auc:.4f}")
    
    results = []
    
    for name, indices in candidates:
        dim = len(indices)
        X_sub = X_gbdt_all[:, indices]
        
        gbdt_oof = np.zeros(N)
        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_sub, y_all, groups=subjects)):
            X_tr, y_tr = X_sub[tr_idx], y_all[tr_idx]
            X_va, y_va = X_sub[va_idx], y_all[va_idx]
            
            mlgb = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.05,
                class_weight='balanced', random_state=42 + fold, verbose=-1
            )
            mcat = CatBoostClassifier(
                iterations=100, depth=4, learning_rate=0.05,
                auto_class_weights='Balanced', random_seed=42 + fold, verbose=0
            )
            mrf = RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight='balanced',
                random_state=42 + fold, n_jobs=-1
            )
            mlgb.fit(X_tr, y_tr)
            mcat.fit(X_tr, y_tr)
            mrf.fit(X_tr, y_tr)
            
            p_lgb = mlgb.predict_proba(X_va)[:, 1]
            p_cat = mcat.predict_proba(X_va)[:, 1]
            p_rf = mrf.predict_proba(X_va)[:, 1]
            gbdt_oof[va_idx] = 0.45 * p_lgb + 0.35 * p_cat + 0.20 * p_rf
            
        gbdt_auc = roc_auc_score(y_all, gbdt_oof)
        
        # Two-Stream Meta Fusion: 0.65 * GBDT + 0.35 * TCN
        fusion_oof = 0.65 * gbdt_oof + 0.35 * tcn_oof
        fusion_auc = roc_auc_score(y_all, fusion_oof)
        
        # 1. 2-State (tau = 0.70 고정 및 최적 탐색)
        f1_tau70 = f1_score(y_all, (fusion_oof >= 0.70).astype(int), zero_division=0)
        prec_tau70 = precision_score(y_all, (fusion_oof >= 0.70).astype(int), zero_division=0)
        rec_tau70 = recall_score(y_all, (fusion_oof >= 0.70).astype(int), zero_division=0)
        
        best_f1, best_tau = 0, 0.5
        for tau in np.arange(0.40, 0.85, 0.02):
            f1_val = f1_score(y_all, (fusion_oof >= tau).astype(int), zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_tau = tau
                
        # 2. 고도화 3-State (tau_defect = 0.70, tau_normal = 0.535)
        # Gray Zone [0.535, 0.700) 제외
        mask_cov = (fusion_oof < 0.535) | (fusion_oof >= 0.700)
        coverage = np.mean(mask_cov) * 100.0
        y_det = y_all[mask_cov]
        p_det = (fusion_oof[mask_cov] >= 0.700).astype(int)
        f1_3state = f1_score(y_det, p_det, zero_division=0)
        prec_3state = precision_score(y_det, p_det, zero_division=0)
        rec_3state = recall_score(y_det, p_det, zero_division=0)
        
        # 시점별 F1 (tau=0.70)
        view_f1s = {}
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            mask_v = (views == v)
            if np.sum(mask_v) > 0 and np.sum(y_all[mask_v]) > 0:
                vf1 = f1_score(y_all[mask_v], (fusion_oof[mask_v] >= 0.70).astype(int), zero_division=0)
                view_f1s[v] = vf1
            else:
                view_f1s[v] = 0.0
                
        results.append({
            'name': name,
            'dim': dim,
            'gbdt_auc': gbdt_auc,
            'fusion_auc': fusion_auc,
            'f1_tau70': f1_tau70,
            'prec_tau70': prec_tau70,
            'rec_tau70': rec_tau70,
            'best_f1': best_f1,
            'best_tau': best_tau,
            'f1_3state': f1_3state,
            'prec_3state': prec_3state,
            'rec_3state': rec_3state,
            'coverage': coverage,
            'view_f1s': view_f1s,
            'indices': indices
        })
        
        print(f"\n[{name}] Dimension: {dim}")
        print(f"  - GBDT AUC: {gbdt_auc:.4f} | Fusion AUC: {fusion_auc:.4f}")
        print(f"  - 2-State (tau=0.70): F1={f1_tau70*100:.2f}%, Prec={prec_tau70*100:.2f}%, Rec={rec_tau70*100:.2f}%")
        print(f"  - 2-State (Best tau={best_tau:.2f}): F1={best_f1*100:.2f}%")
        print(f"  - 고도화 3-State: F1={f1_3state*100:.2f}%, Prec={prec_3state*100:.2f}%, Rec={rec_3state*100:.2f}%, Coverage={coverage:.1f}%")
        print(f"  - 시점별 F1(tau=0.70): A(v5)={view_f1s['view5']*100:.1f}%, B(v2)={view_f1s['view2']*100:.1f}%, C(v3)={view_f1s['view3']*100:.1f}%, D(v4)={view_f1s['view4']*100:.1f}%, E(v1)={view_f1s['view1']*100:.1f}%")

    print("\n" + "=" * 90)
    print(" [최종 토너먼트 종합 순위표]")
    print("=" * 90)
    print(f"{'순위':^4} | {'모델 피처셋':<28} | {'차원':^5} | {'GBDT AUC':^9} | {'Fusion AUC':^10} | {'F1(τ=0.70)':^10} | {'3-State F1':^10} | {'Coverage':^8}")
    print("-" * 90)
    
    # 3-State F1 및 tau=0.70 F1 기준 정렬
    results_sorted = sorted(results, key=lambda x: (x['f1_3state'] + x['f1_tau70']), reverse=True)
    for rank, r in enumerate(results_sorted, 1):
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"{rank}위"))
        print(f"{medal:^4} | {r['name']:<28} | {r['dim']:^5} | {r['gbdt_auc']:^9.4f} | {r['fusion_auc']:^10.4f} | {r['f1_tau70']*100:^9.2f}% | {r['f1_3state']*100:^9.2f}% | {r['coverage']:^7.1f}%")
        
    # 최우수 피처셋 선정 및 저장
    best_candidate = results_sorted[0]
    print("\n" + "=" * 90)
    print(f" ★ 챔피언 선정: {best_candidate['name']}")
    print(f"  - 선정 피처 개수: {best_candidate['dim']}개 (기존 43개 대비 {43 - best_candidate['dim']}개 제거, {((43-best_candidate['dim'])/43)*100:.1f}% 경량화!)")
    print(f"  - 2-State F1: {best_candidate['f1_tau70']*100:.2f}% (기존 {results[0]['f1_tau70']*100:.2f}%)")
    print(f"  - 고도화 3-State F1: {best_candidate['f1_3state']*100:.2f}% (기존 {results[0]['f1_3state']*100:.2f}%)")
    print(f"  - Fusion ROC-AUC: {best_candidate['fusion_auc']:.4f} (기존 {results[0]['fusion_auc']:.4f})")
    print("=" * 90)
    
    # 최적 인덱스 및 피처명 저장
    best_feature_names = [FEATURE_NAMES_43_D4[i] for i in best_candidate['indices']]
    np.savez_compressed(
        os.path.join(CURRENT_DIR, 'd4_optimal_diet_features.npz'),
        indices=np.array(best_candidate['indices']),
        feature_names=np.array(best_feature_names),
        best_name=best_candidate['name'],
        dim=best_candidate['dim']
    )
    print("최적 피처셋 메타데이터 저장 완료: d4_optimal_diet_features.npz")

if __name__ == '__main__':
    run_tournament()
