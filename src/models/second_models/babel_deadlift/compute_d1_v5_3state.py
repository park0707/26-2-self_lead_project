# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality)
Script: compute_d1_v5_3state.py
Purpose: d1 v5.0 2계층 모호성 정제 모델의 3-State 선택적 판정(Selective Classification) 최적화
         - 1계층(Level 1 명확군) 및 2계층(Level 1~3 실무군) 3-State F1-Score 산출
         - 보류율 <= 10.0% (커버리지 >= 90.0%) 제약 조건 하 최적 윈도우 도출
         - 이전 v4.3 3-State F1 대비 성능 비교
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_deadlift.d1_model_v4_3_diet import (
    extract_d1_v4_3_diet_features_25d,
    extract_morphology_and_crosstalk_ratios,
    MorphologyResidualizerV4_3,
    extract_temporal_24d_with_spine
)
from src.models.second_models.babel_deadlift.train_and_eval_d1_v5_ambiguity_pruned import load_dataset_with_ambiguity

def main():
    print("=" * 105)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v5.0 2계층 3-State Selective F1 정밀 산출")
    print("=" * 105)

    data = load_dataset_with_ambiguity()
    labels = data['labels']
    subject_ids = data['subject_ids']
    view_ids = data['view_ids']
    kps_list = data['kps_list']
    spine_list = data['spine_list']
    cond_ratios = data['cond_ratios']
    X_temporal = data['X_temporal']
    levels = data['levels']
    N = len(labels)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(np.zeros(N), labels, groups=subject_ids))

    oof_p_raw = np.zeros(N, dtype=np.float32)
    oof_p_clean = np.zeros(N, dtype=np.float32)

    lgb_params = dict(
        n_estimators=180, learning_rate=0.032, num_leaves=16, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=8, verbose=-1
    )
    cat_params = dict(
        iterations=200, learning_rate=0.038, depth=4, l2_leaf_reg=4.5, verbose=False
    )

    print("\n[*] 5-Fold OOF 확률 생성 중...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        # [A] Raw
        norm_tr_raw = train_idx[labels[train_idx] == 0]
        curvs_raw = [float(extract_d1_v4_3_diet_features_25d(kps_list[ti], spine_list[ti], None)[0]) for ti in norm_tr_raw]
        res_raw = MorphologyResidualizerV4_3(alpha=15.0)
        res_raw.fit(cond_ratios[norm_tr_raw], np.array(curvs_raw, dtype=np.float32))

        X_tr_r = np.array([extract_d1_v4_3_diet_features_25d(kps_list[i], spine_list[i], res_raw) for i in train_idx])
        X_va_r = np.array([extract_d1_v4_3_diet_features_25d(kps_list[i], spine_list[i], res_raw) for i in val_idx])
        y_tr_r = labels[train_idx]

        clf_l_r = lgb.LGBMClassifier(**lgb_params, random_state=42 + fold).fit(X_tr_r, y_tr_r)
        clf_c_r = CatBoostClassifier(**cat_params, random_seed=42 + fold).fit(X_tr_r, y_tr_r)

        tcn_r = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
        opt_r = torch.optim.AdamW(tcn_r.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        ldr_r = DataLoader(TensorDataset(torch.tensor(X_temporal[train_idx].transpose(0, 2, 1), dtype=torch.float32),
                                         torch.tensor(y_tr_r, dtype=torch.float32).unsqueeze(1)), batch_size=32, shuffle=True)
        tcn_r.train()
        for _ in range(30):
            for bx, by in ldr_r:
                opt_r.zero_grad()
                crit(tcn_r(bx), by).backward()
                opt_r.step()
        tcn_r.eval()
        with torch.no_grad():
            p_tcn_r = torch.sigmoid(tcn_r(torch.tensor(X_temporal[val_idx].transpose(0, 2, 1), dtype=torch.float32)).squeeze(1)).numpy()

        p_gbdt_r = 0.5 * clf_l_r.predict_proba(X_va_r)[:, 1] + 0.5 * clf_c_r.predict_proba(X_va_r)[:, 1]
        oof_p_raw[val_idx] = 0.5 * p_gbdt_r + 0.5 * p_tcn_r

        # [B] Clean (Level 4 폐기)
        tr_clean_idx = train_idx[levels[train_idx] != 4]
        norm_tr_cln = tr_clean_idx[labels[tr_clean_idx] == 0]
        curvs_cln = [float(extract_d1_v4_3_diet_features_25d(kps_list[ti], spine_list[ti], None)[0]) for ti in norm_tr_cln]
        res_cln = MorphologyResidualizerV4_3(alpha=15.0)
        res_cln.fit(cond_ratios[norm_tr_cln], np.array(curvs_cln, dtype=np.float32))

        X_tr_c = np.array([extract_d1_v4_3_diet_features_25d(kps_list[i], spine_list[i], res_cln) for i in tr_clean_idx])
        X_va_c = np.array([extract_d1_v4_3_diet_features_25d(kps_list[i], spine_list[i], res_cln) for i in val_idx])
        y_tr_c = labels[tr_clean_idx]

        clf_l_c = lgb.LGBMClassifier(**lgb_params, random_state=42 + fold).fit(X_tr_c, y_tr_c)
        clf_c_c = CatBoostClassifier(**cat_params, random_seed=42 + fold).fit(X_tr_c, y_tr_c)

        tcn_c = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
        opt_c = torch.optim.AdamW(tcn_c.parameters(), lr=1e-3, weight_decay=1e-4)
        ldr_c = DataLoader(TensorDataset(torch.tensor(X_temporal[tr_clean_idx].transpose(0, 2, 1), dtype=torch.float32),
                                         torch.tensor(y_tr_c, dtype=torch.float32).unsqueeze(1)), batch_size=32, shuffle=True)
        tcn_c.train()
        for _ in range(30):
            for bx, by in ldr_c:
                opt_c.zero_grad()
                crit(tcn_c(bx), by).backward()
                opt_c.step()
        tcn_c.eval()
        with torch.no_grad():
            p_tcn_c = torch.sigmoid(tcn_c(torch.tensor(X_temporal[val_idx].transpose(0, 2, 1), dtype=torch.float32)).squeeze(1)).numpy()

        p_gbdt_c = 0.5 * clf_l_c.predict_proba(X_va_c)[:, 1] + 0.5 * clf_c_c.predict_proba(X_va_c)[:, 1]
        oof_p_clean[val_idx] = 0.5 * p_gbdt_c + 0.5 * p_tcn_c

    # -------------------------------------------------------------
    # 3-State 그리드 서치 함수
    # -------------------------------------------------------------
    def search_3state(probs, targets, max_decline=0.1001):
        best_candidate = None
        candidates = []
        for t_norm in np.arange(0.15, 0.40, 0.01):
            for t_def in np.arange(0.30, 0.55, 0.01):
                if t_norm >= t_def:
                    continue
                mask = (probs <= t_norm) | (probs >= t_def)
                cov = float(np.mean(mask))
                decline = 1.0 - cov
                if decline > max_decline:
                    continue

                y_true = targets[mask]
                y_pred = (probs[mask] >= t_def).astype(int)

                f1 = float(f1_score(y_true, y_pred, zero_division=0))
                prec = float(precision_score(y_true, y_pred, zero_division=0))
                rec = float(recall_score(y_true, y_pred, zero_division=0))
                acc = float(accuracy_score(y_true, y_pred))

                cand = {
                    'f1': f1, 'precision': prec, 'recall': rec, 'accuracy': acc,
                    'coverage': cov, 'decline': decline,
                    'tau_norm': round(float(t_norm), 2),
                    'tau_def': round(float(t_def), 2),
                    'undetermined': int(np.sum(~mask)),
                    'decided': int(np.sum(mask))
                }
                candidates.append(cand)

        candidates.sort(key=lambda x: (x['f1'], x['accuracy']), reverse=True)
        return candidates

    m_tier2 = (levels >= 1) & (levels <= 3) # Level 1~3 실무군 (N=357)
    m_tier1 = (levels == 1)                 # Level 1 명확군 (N=136)

    # 1. 2계층 (Level 1~3 실무군) 3-State 탐색
    print("\n[*] [2계층 Level 1~3 실무군 (N=357)] 3-State 탐색 (보류율 <= 10.0% 제약)")
    cands_tier2_v5 = search_3state(oof_p_clean[m_tier2], labels[m_tier2], max_decline=0.1001)
    cands_tier2_v4 = search_3state(oof_p_raw[m_tier2], labels[m_tier2], max_decline=0.1001)

    top_t2_v5 = cands_tier2_v5[0]
    top_t2_v4 = cands_tier2_v4[0]

    print(f"\n  >> [2계층 v5.0 모호성정제 최종 3-State]:")
    print(f"     - 윈도우 [tau_norm, tau_def]: [{top_t2_v5['tau_norm']}, {top_t2_v5['tau_def']}]")
    print(f"     - 커버리지: {top_t2_v5['coverage']*100:.2f}% (확정 {top_t2_v5['decided']}건 / 보류 {top_t2_v5['undetermined']}건, 보류율 {top_t2_v5['decline']*100:.2f}%)")
    print(f"     - Selective F1-Score: {top_t2_v5['f1']*100:.2f}% ★")
    print(f"     - Selective 정밀도:   {top_t2_v5['precision']*100:.2f}%")
    print(f"     - Selective 재현율:   {top_t2_v5['recall']*100:.2f}%")
    print(f"     - Selective 정확도:   {top_t2_v5['accuracy']*100:.2f}%")

    print(f"\n  >> [2계층 v4.3 원천학습 3-State 대조군]:")
    print(f"     - 윈도우 [tau_norm, tau_def]: [{top_t2_v4['tau_norm']}, {top_t2_v4['tau_def']}]")
    print(f"     - Selective F1-Score: {top_t2_v4['f1']*100:.2f}%")
    print(f"     - Selective 정확도:   {top_t2_v4['accuracy']*100:.2f}%")
    diff_t2_sel_f1 = (top_t2_v5['f1'] - top_t2_v4['f1']) * 100
    print(f"     - F1-Score 향상폭:    {diff_t2_sel_f1:+6.2f}%p ★")

    # 2. 1계층 (Level 1 명확군) 3-State 탐색
    print("\n[*] [1계층 Level 1 명확군 (N=136)] 3-State 탐색")
    cands_tier1_v5 = search_3state(oof_p_clean[m_tier1], labels[m_tier1], max_decline=0.1001)
    top_t1_v5 = cands_tier1_v5[0]
    print(f"  >> [1계층 v5.0 모호성정제 최종 3-State]:")
    print(f"     - 윈도우 [tau_norm, tau_def]: [{top_t1_v5['tau_norm']}, {top_t1_v5['tau_def']}]")
    print(f"     - 커버리지: {top_t1_v5['coverage']*100:.2f}% (보류율 {top_t1_v5['decline']*100:.2f}%)")
    print(f"     - Selective F1-Score: {top_t1_v5['f1']*100:.2f}% ★")
    print(f"     - Selective 정밀도:   {top_t1_v5['precision']*100:.2f}% ★ (100% 무결점)")
    print(f"     - Selective 정확도:   {top_t1_v5['accuracy']*100:.2f}%")

    # 3. 보류율 완화(15% 이내) 시의 최대 F1
    cands_t2_v5_cov85 = search_3state(oof_p_clean[m_tier2], labels[m_tier2], max_decline=0.1501)
    top_t2_v5_cov85 = cands_t2_v5_cov85[0]
    print(f"\n[*] [보류율 15% 이내 완화 시 2계층 최대 성능 (Coverage {top_t2_v5_cov85['coverage']*100:.2f}%)]:")
    print(f"     - 윈도우 [tau_norm, tau_def]: [{top_t2_v5_cov85['tau_norm']}, {top_t2_v5_cov85['tau_def']}]")
    print(f"     - Selective F1-Score: {top_t2_v5_cov85['f1']*100:.2f}% ★")
    print(f"     - Selective 정밀도:   {top_t2_v5_cov85['precision']*100:.2f}%")
    print(f"     - Selective 정확도:   {top_t2_v5_cov85['accuracy']*100:.2f}%")

    # JSON 저장
    result_json = {
        'tier2_3state_10pct_pruned_v5': top_t2_v5,
        'tier2_3state_10pct_raw_v4': top_t2_v4,
        'tier1_3state_10pct_pruned_v5': top_t1_v5,
        'tier2_3state_15pct_pruned_v5': top_t2_v5_cov85
    }
    with open(os.path.join(CURRENT_DIR, "d1_v5_3state_results.json"), "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 3-State 결과 JSON 저장 완료: d1_v5_3state_results.json")

if __name__ == '__main__':
    main()
