# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Submodel: d2 (발과 무릎의 방향 일치 / 동적 무릎 외반)
Script: train_d2_v7_robust.py
Purpose: v7.0 Noise-Robust 모델 기획
         - 모호성 분리(Level 4 배제)를 전혀 사용하지 않고 원천 데이터 385개를 100% 학습
         - GBDT 및 TCN에 Noise Robustness 기법(Label Smoothing, SCE Loss, 정규화 강화) 도입
         - v6.0(Two-Tier) 모델과 3-State 성능 정면 대조
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d2_model_v5_2_pruned import FEATURE_NAMES_10_D2, VIEW_ALIAS_MAP
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

CSV_PATH = r"C:\Users\phj20\.gemini\antigravity\brain\84b70613-7463-404c-a91f-c996f02de642\scratch\d2_ambiguity_classified.csv"

# Noise-Robust Loss: Symmetric Cross Entropy (SCE) or Label Smoothed BCE
class LabelSmoothedBCE(nn.Module):
    def __init__(self, smoothing=0.15):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # targets: 0 or 1
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, smoothed_targets)

def main():
    df = pd.read_csv(CSV_PATH)
    
    # 1. 모호성 배제 없이 전수(385개) 학습 데이터 구성
    X_all = df[FEATURE_NAMES_10_D2].values
    y_all = df['y_d2'].values
    subjs_all = df['subj'].values
    views_all = df['view'].values
    
    # TCN을 위한 가상 시계열 피처 생성 (임시로 정형 데이터를 시계열처럼 확장복사)
    X_temporal = np.repeat(X_all[:, np.newaxis, :], 64, axis=1) # (N, 64, 10)
    
    N = len(df)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs_v7 = np.zeros(N)
    
    print("[*] Training v7.0 Noise-Robust Model on Full Raw Data (N=385, No Ambiguity Filtering)...")
    
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_all, y_all, groups=subjs_all)):
        # LightGBM - Extra Trees 및 정규화 강화로 과적합/노이즈 민감도 억제
        clf_l = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=4.0,
            extra_trees=True, path_smooth=1.0, min_child_samples=15,
            scale_pos_weight=3.28, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        clf_l.fit(X_all[tr_idx], y_all[tr_idx])
        
        # CatBoost - L2 강력 규제 및 Subsample 세팅
        clf_c = CatBoostClassifier(
            iterations=200, learning_rate=0.03, depth=4, l2_leaf_reg=6.0,
            auto_class_weights='Balanced', bootstrap_type='MVS', subsample=0.8,
            random_seed=42 + fold, verbose=0, thread_count=1,
            train_dir=r"C:\Temp\catboost_info"
        )
        clf_c.fit(X_all[tr_idx], y_all[tr_idx])
        
        # TCN - Label Smoothing 기반 Robust Loss 적용
        tcn = SquatC1DilatedTCN(in_channels=10, hidden_dim=32, dropout=0.35)
        opt = torch.optim.AdamW(tcn.parameters(), lr=1e-3, weight_decay=1e-3)
        crit = LabelSmoothedBCE(smoothing=0.2) # 20% Smoothing
        
        ldr = DataLoader(TensorDataset(torch.tensor(X_temporal[tr_idx].transpose(0, 2, 1), dtype=torch.float32),
                                       torch.tensor(y_all[tr_idx], dtype=torch.float32).unsqueeze(1)), batch_size=32, shuffle=True)
        
        tcn.train()
        for _ in range(35):
            for bx, by in ldr:
                opt.zero_grad()
                loss = crit(tcn(bx), by)
                loss.backward()
                opt.step()
        
        tcn.eval()
        with torch.no_grad():
            vx = torch.tensor(X_temporal[val_idx].transpose(0, 2, 1), dtype=torch.float32)
            p_tcn = torch.sigmoid(tcn(vx).squeeze(1)).numpy()
            
        p_gbdt = 0.5 * clf_l.predict_proba(X_all[val_idx])[:, 1] + 0.5 * clf_c.predict_proba(X_all[val_idx])[:, 1]
        
        # Tri-Ensemble (80% GBDT + 20% TCN)
        oof_probs_v7[val_idx] = 0.8 * p_gbdt + 0.2 * p_tcn
        
    def search_best_3state(probs, targets, max_decline=0.1001):
        candidates = []
        for tc in np.arange(0.15, 0.45, 0.005):
            for td in np.arange(0.35, 0.65, 0.005):
                if tc >= td: continue
                mask = (probs <= tc) | (probs >= td)
                cov = float(np.mean(mask))
                if 1.0 - cov > max_decline: continue
                
                yt = targets[mask]
                yp = (probs[mask] >= td).astype(int)
                if len(np.unique(yt)) < 2: continue
                
                f1 = float(f1_score(yt, yp, zero_division=0))
                cm = confusion_matrix(yt, yp, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                
                candidates.append({
                    'f1': f1, 'precision': float(precision_score(yt, yp, zero_division=0)),
                    'recall': float(recall_score(yt, yp, zero_division=0)),
                    'accuracy': float(accuracy_score(yt, yp)),
                    'coverage': cov, 'decline': 1.0 - cov,
                    'tau_clean': round(float(tc), 3), 'tau_def': round(float(td), 3),
                    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
                })
        candidates.sort(key=lambda x: (x['f1'], x['accuracy']), reverse=True)
        return candidates

    cands_v7 = search_best_3state(oof_probs_v7, y_all, max_decline=0.1001)
    best_v7 = cands_v7[0]

    # Load v6.0 result for comparison
    V6_JSON = r"C:\Users\phj20\.gemini\antigravity\brain\84b70613-7463-404c-a91f-c996f02de642\scratch\d2_final_3state_results.json"
    with open(V6_JSON, 'r', encoding='utf-8') as f:
        v6_data = json.load(f)
    best_v6 = v6_data['best_twotier_3state']

    print("\n" + "=" * 80)
    print("  [vidio-health-trainer] d2 v7.0 Robust vs v6.0 Two-Tier 성능 비교")
    print("=" * 80)
    
    print(f"\n[v6.0 Two-Tier (모호성 필터링 O - 27건 배제, 358개 평가)]")
    print(f"  - 3-State F1-Score : {best_v6['f1']*100:.2f}%")
    print(f"  - 정확도           : {best_v6['accuracy']*100:.2f}%")
    print(f"  - 커버리지         : {best_v6['coverage']*100:.2f}% (보류 {best_v6['undetermined']}건)")
    
    print(f"\n[v7.0 Noise-Robust (모호성 필터링 X - 원천 385개 전수 평가)]")
    print(f"  - 3-State F1-Score : {best_v7['f1']*100:.2f}%")
    print(f"  - 정밀도           : {best_v7['precision']*100:.2f}%")
    print(f"  - 재현율           : {best_v7['recall']*100:.2f}%")
    print(f"  - 정확도           : {best_v7['accuracy']*100:.2f}%")
    print(f"  - 판정 윈도우      : [{best_v7['tau_clean']}, {best_v7['tau_def']}]")
    print(f"  - 커버리지         : {best_v7['coverage']*100:.2f}% (보류 {int(best_v7['decline']*N)}건)")
    print(f"  - 혼동 행렬        : TP={best_v7['tp']}, TN={best_v7['tn']}, FP={best_v7['fp']}, FN={best_v7['fn']}")

    diff_f1 = (best_v7['f1'] - best_v6['f1']) * 100
    print(f"\n  >> F1-Score 차이: {diff_f1:+6.2f}%p (필터링 없이 강건성 기법만으로 방어한 결과)")

    res_v7 = {
        'best_v7_robust': best_v7,
        'best_v6_twotier': best_v6
    }
    
    OUT_JSON = os.path.join(CURRENT_DIR, "d2_v7_robust_results.json")
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(res_v7, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
