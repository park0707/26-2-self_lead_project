# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Submodel: d2 (발과 무릎의 방향 일치 / 동적 무릎 외반)
Script: compute_d2_v6_3state.py
Purpose: d2 v6.0 Two-Tier 최종 모델의 최적 3-State Selective F1-Score 정밀 산출
         - 보류율 <= 10.0% (커버리지 >= 90.0%) 제약 조건 하 그리드 서치
         - 이전 v5.2 베이스라인 대비 최종 3-State F1-Score 및 지표 대조
         - 5대 시점별 최종 3-State F1-Score 전수 산출
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d2_model_v5_2_pruned import FEATURE_NAMES_10_D2, VIEW_ALIAS_MAP

CSV_PATH = r"C:\Users\phj20\.gemini\antigravity\brain\84b70613-7463-404c-a91f-c996f02de642\scratch\d2_ambiguity_classified.csv"
df = pd.read_csv(CSV_PATH)

# Level 4 폐기된 실무 데이터셋 (358개)
df_clean = df[df['ambiguity_level'] != 4].copy().reset_index(drop=True)
X_clean = df_clean[FEATURE_NAMES_10_D2].values
y_clean = df_clean['y_d2'].values
subjs_clean = df_clean['subj'].values
views_clean = df_clean['view'].values

# 이전 베이스라인 (원천 385개)
X_all = df[FEATURE_NAMES_10_D2].values
y_all = df['y_d2'].values
subjs_all = df['subj'].values
views_all = df['view'].values

# 5-Fold StratifiedGroupKFold OOF 확률 생성
sgkf_clean = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs_clean = np.zeros(len(df_clean))

for fold, (tr_idx, val_idx) in enumerate(sgkf_clean.split(X_clean, y_clean, groups=subjs_clean)):
    clf_l = lgb.LGBMClassifier(
        n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=12,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.15, reg_lambda=2.0,
        scale_pos_weight=3.28, random_state=42 + fold, verbose=-1, n_jobs=1
    )
    clf_l.fit(X_clean[tr_idx], y_clean[tr_idx])
    
    clf_c = CatBoostClassifier(
        iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=4.0,
        auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1,
        train_dir=r"C:\Temp\catboost_info"
    )
    clf_c.fit(X_clean[tr_idx], y_clean[tr_idx])
    
    oof_probs_clean[val_idx] = 0.5 * clf_l.predict_proba(X_clean[val_idx])[:, 1] + 0.5 * clf_c.predict_proba(X_clean[val_idx])[:, 1]

# 원천 베이스라인 OOF 확률
sgkf_all = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs_all = np.zeros(len(df))
for fold, (tr_idx, val_idx) in enumerate(sgkf_all.split(X_all, y_all, groups=subjs_all)):
    clf_l = lgb.LGBMClassifier(
        n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=12,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.15, reg_lambda=2.0,
        scale_pos_weight=3.28, random_state=42 + fold, verbose=-1, n_jobs=1
    )
    clf_l.fit(X_all[tr_idx], y_all[tr_idx])
    
    clf_c = CatBoostClassifier(
        iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=4.0,
        auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1,
        train_dir=r"C:\Temp\catboost_info"
    )
    clf_c.fit(X_all[tr_idx], y_all[tr_idx])
    
    oof_probs_all[val_idx] = 0.5 * clf_l.predict_proba(X_all[val_idx])[:, 1] + 0.5 * clf_c.predict_proba(X_all[val_idx])[:, 1]


def search_best_3state(probs, targets, max_decline=0.1001):
    candidates = []
    for t_clean in np.arange(0.15, 0.45, 0.005):
        for t_def in np.arange(0.35, 0.65, 0.005):
            if t_clean >= t_def:
                continue
            mask = (probs <= t_clean) | (probs >= t_def)
            cov = float(np.mean(mask))
            decline = 1.0 - cov
            if decline > max_decline:
                continue
            
            y_true = targets[mask]
            y_pred = (probs[mask] >= t_def).astype(int)
            if len(np.unique(y_true)) < 2:
                continue
            
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            acc = float(accuracy_score(y_true, y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            candidates.append({
                'f1': f1, 'precision': prec, 'recall': rec, 'accuracy': acc,
                'coverage': cov, 'decline': decline,
                'tau_clean': round(float(t_clean), 3),
                'tau_def': round(float(t_def), 3),
                'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
                'decided': int(np.sum(mask)), 'undetermined': int(np.sum(~mask))
            })
    candidates.sort(key=lambda x: (x['f1'], x['accuracy']), reverse=True)
    return candidates

# 1. 2계층 최종 모델 최적화 (보류율 <= 10.0%)
cands_twotier = search_best_3state(oof_probs_clean, y_clean, max_decline=0.1001)
best_twotier = cands_twotier[0]

# 2. 베이스라인 원천 모델 최적화 (보류율 <= 10.0%)
cands_base = search_best_3state(oof_probs_all, y_all, max_decline=0.1001)
best_base = cands_base[0]

print("=" * 80)
print("  [vidio-health-trainer] 바벨 데드리프트 d2 2계층 최종 3-State F1-Score 실측 결과")
print("=" * 80)
print(f"\n[★ 2계층 적용 최종 모델 (Level 4 폐기, 358개 실무 데이터셋)]")
print(f"  - 최종 3-State F1-Score : {best_twotier['f1']*100:.2f}% ★")
print(f"  - 3-State 정밀도 (Precision): {best_twotier['precision']*100:.2f}%")
print(f"  - 3-State 재현율 (Recall)   : {best_twotier['recall']*100:.2f}%")
print(f"  - 3-State 정확도 (Accuracy) : {best_twotier['accuracy']*100:.2f}%")
print(f"  - 판정 커버리지 / 보류율    : {best_twotier['coverage']*100:.2f}% / {best_twotier['decline']*100:.2f}% (보류 {best_twotier['undetermined']}건)")
print(f"  - 최적 판정 윈도우          : [Tau Clean: {best_twotier['tau_clean']}, Tau Defect: {best_twotier['tau_def']}]")
print(f"  - 혼동 행렬 (Confusion)     : TP={best_twotier['tp']}, TN={best_twotier['tn']}, FP={best_twotier['fp']}, FN={best_twotier['fn']}")

print(f"\n[이전 베이스라인 v5.2 (원천 385개, Level 4 오염 포함)]")
print(f"  - 3-State F1-Score          : {best_base['f1']*100:.2f}%")
print(f"  - 3-State 정밀도            : {best_base['precision']*100:.2f}%")
print(f"  - 3-State 재현율            : {best_base['recall']*100:.2f}%")
print(f"  - 3-State 정확도            : {best_base['accuracy']*100:.2f}%")
print(f"  - 최적 판정 윈도우          : [Tau Clean: {best_base['tau_clean']}, Tau Defect: {best_base['tau_def']}]")
print(f"  - 혼동 행렬                 : TP={best_base['tp']}, TN={best_base['tn']}, FP={best_base['fp']}, FN={best_base['fn']}")

diff_f1 = (best_twotier['f1'] - best_base['f1']) * 100
diff_fp = best_twotier['fp'] - best_base['fp']
print(f"\n[최종 성능 비교 및 향상폭]")
print(f"  ★ 최종 3-State F1-Score 향상 : {best_base['f1']*100:.2f}% -> {best_twotier['f1']*100:.2f}% ({diff_f1:+6.2f}%p 상승)")
print(f"  ★ 오탐 (FP, False Positive) : {best_base['fp']}건 -> {best_twotier['fp']}건 ({diff_fp:+d}건, {(diff_fp/best_base['fp'])*100:.1f}%)")

# 5대 시점별 3-State F1-Score (동일 최적 윈도우 적용)
print("\n" + "=" * 80)
print("  공식 5대 시점별 최종 3-State F1-Score 실측 결과 (AGENTS.md 표준)")
print("=" * 80)

tc = best_twotier['tau_clean']
td = best_twotier['tau_def']
view_metrics = {}

for v in ["view3", "view2", "view4", "view5", "view1"]:
    v_alias = VIEW_ALIAS_MAP[v]
    m_v = (views_clean == v)
    y_v = y_clean[m_v]
    p_v = oof_probs_clean[m_v]
    
    mask_v = (p_v <= tc) | (p_v >= td)
    y_true_v = y_v[mask_v]
    y_pred_v = (p_v[mask_v] >= td).astype(int)
    
    if len(np.unique(y_true_v)) > 1:
        f1_v = f1_score(y_true_v, y_pred_v, zero_division=0) * 100.0
        prec_v = precision_score(y_true_v, y_pred_v, zero_division=0) * 100.0
        rec_v = recall_score(y_true_v, y_pred_v, zero_division=0) * 100.0
        acc_v = accuracy_score(y_true_v, y_pred_v) * 100.0
        cm_v = confusion_matrix(y_true_v, y_pred_v, labels=[0, 1])
        tn_v, fp_v, fn_v, tp_v = cm_v.ravel()
    else:
        f1_v, prec_v, rec_v, acc_v, tn_v, fp_v, fn_v, tp_v = 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0
    
    cov_v = np.mean(mask_v) * 100.0
    view_metrics[v] = {
        'alias': v_alias, 'f1': f1_v, 'precision': prec_v, 'recall': rec_v,
        'accuracy': acc_v, 'coverage': cov_v, 'tp': int(tp_v), 'tn': int(tn_v), 'fp': int(fp_v), 'fn': int(fn_v)
    }
    print(f"[{v_alias}]")
    print(f"  3-State F1: {f1_v:5.2f}% | 정밀도: {prec_v:5.2f}% | 재현율: {rec_v:5.2f}% | 정확도: {acc_v:5.2f}% | FP: {fp_v} | FN: {fn_v} | 커버리지: {cov_v:.1f}%")

# Save final 3state json
res_3state_final = {
    'best_twotier_3state': best_twotier,
    'best_baseline_3state': best_base,
    'view_metrics': view_metrics
}
OUT_JSON = r"C:\Users\phj20\.gemini\antigravity\brain\84b70613-7463-404c-a91f-c996f02de642\scratch\d2_final_3state_results.json"
with open(OUT_JSON, 'w', encoding='utf-8') as fp:
    json.dump(res_3state_final, fp, indent=2, ensure_ascii=False)
print(f"\nSaved final 3-state results to {OUT_JSON}")

