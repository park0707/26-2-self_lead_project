# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반
Script: evaluate_official_two_stream_diet.py
Description: 정식 KinematicTCN(Dilation TCN) + 슬림 GBDT 앙상블 정밀 토너먼트
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

from d4_model_v5 import FEATURE_NAMES_43_D4, TemporalBlock, KinematicTCN

# 정식 KinematicTCN 학습 함수 (v5.0 동일 스펙)
def train_official_tcn(X_tr, y_tr, X_va, y_va, epochs=45, lr=0.002, batch_size=32, device='cpu'):
    torch.manual_seed(42)
    np.random.seed(42)
    model = KinematicTCN(in_features=12, hidden_dim=32).to(device)
    
    pos_cnt = sum(y_tr)
    neg_cnt = len(y_tr) - pos_cnt
    pos_weight = torch.tensor([neg_cnt / max(pos_cnt, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()
            
    model.eval()
    with torch.no_grad():
        vx = torch.tensor(X_va, dtype=torch.float32).to(device)
        val_logits = model(vx).squeeze(-1)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        
    return val_probs

def run_evaluation():
    print("=" * 90)
    print(" [vidio-health-trainer] 바벨 데드리프트 d4 정식 Two-Stream 피처 다이어트 토너먼트")
    print("=" * 90)
    
    data = np.load(CACHE_PATH, allow_pickle=True)
    X_gbdt_all = data['X_gbdt']   # (770, 43)
    X_tcn_all = data['X_tcn']     # (770, 12, 64)
    y_all = data['y']             # (770,)
    subjects = data['subjects']   # (770,)
    views = data['views']         # (770,)
    
    N, D_all = X_gbdt_all.shape
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. 정식 Dilation TCN 5-Fold OOF 예측 산출
    print("\n[Step 1] 정식 Dilation TCN 5-Fold OOF 사전 산출...")
    tcn_oof = np.zeros(N)
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_gbdt_all, y_all, groups=subjects)):
        tcn_pred = train_official_tcn(X_tcn_all[tr_idx], y_all[tr_idx], X_tcn_all[va_idx], y_all[va_idx], epochs=45)
        tcn_oof[va_idx] = tcn_pred
        
    tcn_auc = roc_auc_score(y_all, tcn_oof)
    print(f"  -> 정식 KinematicTCN OOF ROC-AUC: {tcn_auc:.4f}")
    
    # 2. 피처 후보군 정의
    # 음수/노이즈 12개 피처: [41, 9, 30, 12, 31, 32, 19, 26, 17, 36, 37, 29]
    # 고상관/중복 5개: [27, 24, 21, 35, 23]
    
    tier_candidates = {
        "v5.0 원본 (43D Full Baseline)": list(range(43)),
        "Diet-A (31D: 음수/노이즈 12개 제거)": [
            i for i in range(43) if i not in [41, 9, 30, 12, 31, 32, 19, 26, 17, 36, 37, 29]
        ],
        "Diet-B (26D: 고상관 중복 압축)": [
            i for i in range(43) if i not in [41, 9, 30, 12, 31, 32, 19, 26, 17, 36, 37, 29, 27, 24, 21, 35, 23]
        ],
        "Diet-C (19D: 중요도 0.5% 이상 엄선)": [
            14, 0, 6, 33, 13, 28, 38, 18, 8, 15, 3, 10, 5, 1, 34, 25, 16, 7, 2
        ],
        "Diet-D (15D: 초경량 코어 15 피처)": [
            14, 0, 6, 33, 13, 28, 38, 18, 8, 15, 3, 10, 5, 1, 7
        ],
        "Diet-E (12D: 극단적 압축 절대필수 12 피처)": [
            14, 0, 6, 33, 13, 28, 38, 18, 8, 15, 3, 10
        ]
    }
    
    results = []
    
    print("\n[Step 2] 각 피처 다이어트 후보군 5-Fold 피험자 격리 학습 및 검증...")
    for name, indices in tier_candidates.items():
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
        
        # Two-Stream Meta Fusion: 0.60 * GBDT + 0.40 * TCN (v5.0 최적 앙상블 비중)
        fusion_oof = 0.60 * gbdt_oof + 0.40 * tcn_oof
        fusion_auc = roc_auc_score(y_all, fusion_oof)
        
        # 1. 2-State 최적 F1 탐색
        best_f1, best_tau, best_prec, best_rec = 0, 0.5, 0, 0
        for tau in np.arange(0.40, 0.85, 0.01):
            p = (fusion_oof >= tau).astype(int)
            f1 = f1_score(y_all, p, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
                best_prec = precision_score(y_all, p, zero_division=0)
                best_rec = recall_score(y_all, p, zero_division=0)
                
        # tau = 0.70 고정 F1
        f1_70 = f1_score(y_all, (fusion_oof >= 0.70).astype(int), zero_division=0)
        prec_70 = precision_score(y_all, (fusion_oof >= 0.70).astype(int), zero_division=0)
        rec_70 = recall_score(y_all, (fusion_oof >= 0.70).astype(int), zero_division=0)
        
        # 2. 고도화 3-State (Coverage >= 90% 유지선 탐색)
        best_3s_f1 = 0
        best_tn, best_td = 0.5, 0.7
        best_cov = 0
        best_3s_prec, best_3s_rec = 0, 0
        
        # tau_defect를 0.65~0.75, tau_normal을 0.50~0.58 범위에서 그리드 탐색 (Coverage >= 88%)
        for td in np.arange(0.66, 0.75, 0.02):
            for tn in np.arange(0.50, td - 0.08, 0.02):
                mask = (fusion_oof < tn) | (fusion_oof >= td)
                cov = np.mean(mask) * 100.0
                if cov >= 88.0:
                    y_det = y_all[mask]
                    p_det = (fusion_oof[mask] >= td).astype(int)
                    f1_3s = f1_score(y_det, p_det, zero_division=0)
                    if f1_3s > best_3s_f1:
                        best_3s_f1 = f1_3s
                        best_tn = tn
                        best_td = td
                        best_cov = cov
                        best_3s_prec = precision_score(y_det, p_det, zero_division=0)
                        best_3s_rec = recall_score(y_det, p_det, zero_division=0)
                        
        # 5대 시점별 F1 (Best tau 기준)
        view_f1s = {}
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            mask_v = (views == v)
            if np.sum(mask_v) > 0 and np.sum(y_all[mask_v]) > 0:
                vf1 = f1_score(y_all[mask_v], (fusion_oof[mask_v] >= best_tau).astype(int), zero_division=0)
                view_f1s[v] = vf1
            else:
                view_f1s[v] = 0.0
                
        results.append({
            'name': name,
            'dim': dim,
            'gbdt_auc': gbdt_auc,
            'fusion_auc': fusion_auc,
            'best_f1': best_f1,
            'best_tau': best_tau,
            'best_prec': best_prec,
            'best_rec': best_rec,
            'f1_70': f1_70,
            'prec_70': prec_70,
            'rec_70': rec_70,
            'best_3s_f1': best_3s_f1,
            'best_3s_prec': best_3s_prec,
            'best_3s_rec': best_3s_rec,
            'best_cov': best_cov,
            'best_tn': best_tn,
            'best_td': best_td,
            'view_f1s': view_f1s,
            'indices': indices,
            'oof_preds': fusion_oof
        })
        
        print(f"\n[{name}] 피처수: {dim}개")
        print(f"  - GBDT AUC: {gbdt_auc:.4f} | Two-Stream Fusion AUC: {fusion_auc:.4f}")
        print(f"  - 2-State (tau={best_tau:.2f}): F1={best_f1*100:.2f}%, Prec={best_prec*100:.2f}%, Rec={best_rec*100:.2f}%")
        print(f"  - 2-State (tau=0.70): F1={f1_70*100:.2f}%, Prec={prec_70*100:.2f}%, Rec={rec_70*100:.2f}%")
        print(f"  - 고도화 3-State (tn={best_tn:.2f}, td={best_td:.2f}): F1={best_3s_f1*100:.2f}%, Coverage={best_cov:.1f}%")
        print(f"  - 시점별 F1: A(우측후방)={view_f1s['view5']*100:.1f}%, B(우측전방)={view_f1s['view2']*100:.1f}%, C(정면)={view_f1s['view3']*100:.1f}%, D(좌측전방)={view_f1s['view4']*100:.1f}%, E(좌측후방)={view_f1s['view1']*100:.1f}%")

    print("\n" + "=" * 95)
    print(" [최종 피처 다이어트 토너먼트 종합 결과표]")
    print("=" * 95)
    print(f"{'순위':<6} | {'모델 피처셋':<36} | {'차원':^5} | {'GBDT AUC':^9} | {'Fusion AUC':^10} | {'2-State F1':^11} | {'3-State F1':^11} | {'Coverage':^8}")
    print("-" * 95)
    
    # 순위 기준: 3-State F1 + 2-State F1 + Fusion AUC
    sorted_res = sorted(results, key=lambda x: (x['best_3s_f1'] + x['best_f1'] + x['fusion_auc']), reverse=True)
    for rank, r in enumerate(sorted_res, 1):
        rank_str = f"[{rank}위]"
        print(f"{rank_str:<6} | {r['name']:<36} | {r['dim']:^5} | {r['gbdt_auc']:^9.4f} | {r['fusion_auc']:^10.4f} | {r['best_f1']*100:^10.2f}% | {r['best_3s_f1']*100:^10.2f}% | {r['best_cov']:^7.1f}%")
        
    best_one = sorted_res[0]
    print("\n" + "=" * 95)
    print(f" [최종 우승 챔피언] {best_one['name']}")
    print(f"  - 최적 피처 수: {best_one['dim']}개 (기존 43개 중 {43 - best_one['dim']}개 제거, {(43 - best_one['dim'])/43*100:.1f}% 경량화 달성!)")
    print(f"  - GBDT AUC 변화: {results[0]['gbdt_auc']:.4f} -> {best_one['gbdt_auc']:.4f} ({best_one['gbdt_auc'] - results[0]['gbdt_auc']:+.4f})")
    print(f"  - Two-Stream Fusion AUC: {results[0]['fusion_auc']:.4f} -> {best_one['fusion_auc']:.4f} ({best_one['fusion_auc'] - results[0]['fusion_auc']:+.4f})")
    print(f"  - 2-State F1: {results[0]['best_f1']*100:.2f}% -> {best_one['best_f1']*100:.2f}% ({best_one['best_f1']*100 - results[0]['best_f1']*100:+.2f}%p)")
    print(f"  - 고도화 3-State F1: {results[0]['best_3s_f1']*100:.2f}% -> {best_one['best_3s_f1']*100:.2f}% ({best_one['best_3s_f1']*100 - results[0]['best_3s_f1']*100:+.2f}%p)")
    print(f"  - 3-State 설정: tau_normal={best_one['best_tn']:.3f}, tau_defect={best_one['best_td']:.3f}, Coverage={best_one['best_cov']:.1f}%")
    print("=" * 95)
    
    # 챔피언 피처 인덱스 및 목록 저장
    best_names = [FEATURE_NAMES_43_D4[i] for i in best_one['indices']]
    np.savez_compressed(
        os.path.join(CURRENT_DIR, 'd4_v5_1_diet_champion.npz'),
        indices=np.array(best_one['indices']),
        feature_names=np.array(best_names),
        best_name=best_one['name'],
        dim=best_one['dim'],
        best_tau=best_one['best_tau'],
        best_tn=best_one['best_tn'],
        best_td=best_one['best_td'],
        oof_preds=best_one['oof_preds'],
        y=y_all,
        views=views,
        subjects=subjects
    )
    print("챔피언 메타데이터 저장 완료: d4_v5_1_diet_champion.npz")

if __name__ == '__main__':
    run_evaluation()
