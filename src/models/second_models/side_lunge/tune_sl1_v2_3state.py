# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: tune_sl1_v2_3state.py
Purpose: sl1_v2 3-State 초정밀 임계값 튜닝 (보류율 <= 10.0% 제약)
"""

import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)

# 1. 캐시 및 모델 로드
cache = pickle.load(open('D:/Data/피처_데이터/sl1_v2_dataset_cache.pkl', 'rb'))
bundle = pickle.load(open('D:/src/models/second_models/side_lunge/sl1_v2_bundle.pkl', 'rb'))

X = cache['X']
y = cache['y']
groups = cache['groups']
views = np.array(cache['views'])

models_lgb = bundle['models_lgb']
models_cat = bundle['models_cat']
models_xgb = bundle['models_xgb']
weights = bundle['ensemble_weights']

# 2. OOF 확률 계산
sgkf = StratifiedGroupKFold(n_splits=5)
oof_lgb = np.zeros(len(y), dtype=np.float32)
oof_cat = np.zeros(len(y), dtype=np.float32)
oof_xgb = np.zeros(len(y), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    X_val = X[val_idx]
    oof_lgb[val_idx] = models_lgb[fold].predict(X_val)
    oof_cat[val_idx] = models_cat[fold].predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] = models_xgb[fold].predict(xgb.DMatrix(X_val))

oof_ens = weights[0]*oof_lgb + weights[1]*oof_cat + weights[2]*oof_xgb

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}

def evaluate_thresholds(y_true, y_prob, t1, t2):
    unc = (y_prob >= t1) & (y_prob < t2)
    dec = ~unc
    unc_rate = float(np.mean(unc))
    if np.sum(dec) == 0:
        return None
    y_t_dec = y_true[dec]
    y_p_dec = (y_prob[dec] >= t2).astype(int)
    f1 = float(f1_score(y_t_dec, y_p_dec, zero_division=0))
    acc = float(accuracy_score(y_t_dec, y_p_dec))
    prec = float(precision_score(y_t_dec, y_p_dec, zero_division=0))
    rec = float(recall_score(y_t_dec, y_p_dec, zero_division=0))
    return {
        'tau_clean': t1,
        'tau_defect': t2,
        'uncertainty_rate': unc_rate,
        'coverage': 1.0 - unc_rate,
        'f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec
    }

print("="*80)
print("1. [전역(Global) 단일 임계값] 보류율 <= 10.0% 초정밀 스캔")
print("="*80)

best_g_f1 = {'f1': -1}
best_g_acc = {'accuracy': -1}
best_g_bal = {'score': -1}

t_range = np.arange(0.15, 0.85, 0.005)
for i, t1 in enumerate(t_range):
    for t2 in t_range[i+1:]:
        unc = (oof_ens >= t1) & (oof_ens < t2)
        unc_rate = np.mean(unc)
        if unc_rate > 0.10:
            continue
        res = evaluate_thresholds(y, oof_ens, t1, t2)
        if res is None:
            continue
        if res['f1'] > best_g_f1.get('f1', -1):
            best_g_f1 = res
        if res['accuracy'] > best_g_acc.get('accuracy', -1):
            best_g_acc = res
        # Balanced score: 2 * (F1 * Acc) / (F1 + Acc)
        bal_score = 2 * res['f1'] * res['accuracy'] / (res['f1'] + res['accuracy'] + 1e-9)
        if bal_score > best_g_bal.get('score', -1):
            best_g_bal = {**res, 'score': bal_score}

print(f"[전역 F1 극대화]   : tau=[{best_g_f1['tau_clean']:.3f}, {best_g_f1['tau_defect']:.3f}] | F1={best_g_f1['f1']:.4f}, Acc={best_g_f1['accuracy']*100:.2f}%, Rec={best_g_f1['recall']*100:.2f}%, 보류율={best_g_f1['uncertainty_rate']*100:.2f}%")
print(f"[전역 정확도 극대화]: tau=[{best_g_acc['tau_clean']:.3f}, {best_g_acc['tau_defect']:.3f}] | F1={best_g_acc['f1']:.4f}, Acc={best_g_acc['accuracy']*100:.2f}%, Rec={best_g_acc['recall']*100:.2f}%, 보류율={best_g_acc['uncertainty_rate']*100:.2f}%")
print(f"[전역 Balanced]    : tau=[{best_g_bal['tau_clean']:.3f}, {best_g_bal['tau_defect']:.3f}] | F1={best_g_bal['f1']:.4f}, Acc={best_g_bal['accuracy']*100:.2f}%, Rec={best_g_bal['recall']*100:.2f}%, 보류율={best_g_bal['uncertainty_rate']*100:.2f}%")


print("\n" + "="*80)
print("2. [시점별(Per-view) 적응형 임계값] 보류율 <= 10.0% 초정밀 스캔")
print("="*80)

unique_views = sorted(list(set(views)))

# 시점별 F1 극대화
best_pv_f1 = {}
best_pv_acc = {}

for v in unique_views:
    vm = (views == v)
    y_v = y[vm]
    p_v = oof_ens[vm]
    
    b_f1 = {'f1': -1}
    b_acc = {'accuracy': -1}
    
    for i, t1 in enumerate(t_range):
        for t2 in t_range[i+1:]:
            unc = (p_v >= t1) & (p_v < t2)
            if np.mean(unc) > 0.10:
                continue
            res = evaluate_thresholds(y_v, p_v, t1, t2)
            if res is None:
                continue
            if res['f1'] > b_f1.get('f1', -1):
                b_f1 = res
            if res['accuracy'] > b_acc.get('accuracy', -1):
                b_acc = res
    best_pv_f1[v] = b_f1
    best_pv_acc[v] = b_acc

def calc_overall_3state(view_th_dict):
    pred_states = []
    for i in range(len(y)):
        v = views[i]
        p = oof_ens[i]
        th = view_th_dict[v]
        if p < th['tau_clean']:
            pred_states.append('Clean')
        elif p >= th['tau_defect']:
            pred_states.append('Defect')
        else:
            pred_states.append('Retake')
    pred_states = np.array(pred_states)
    is_decided = (pred_states != 'Retake')
    y_t_dec = y[is_decided]
    y_p_dec = (pred_states[is_decided] == 'Defect').astype(int)
    
    res = {
        'uncertainty_rate': float(np.mean(pred_states == 'Retake')),
        'f1': float(f1_score(y_t_dec, y_p_dec, zero_division=0)),
        'accuracy': float(accuracy_score(y_t_dec, y_p_dec)),
        'precision': float(precision_score(y_t_dec, y_p_dec, zero_division=0)),
        'recall': float(recall_score(y_t_dec, y_p_dec, zero_division=0)),
        'view_results': {}
    }
    for v in unique_views:
        vm = (views == v)
        v_states = pred_states[vm]
        v_dec = (v_states != 'Retake')
        res['view_results'][v] = {
            'alias': VIEW_ALIAS_MAP.get(v, v),
            'tau_clean': view_th_dict[v]['tau_clean'],
            'tau_defect': view_th_dict[v]['tau_defect'],
            'uncertainty_rate': float(np.mean(v_states == 'Retake')),
            'f1': float(f1_score(y[vm][v_dec], (v_states[v_dec] == 'Defect').astype(int), zero_division=0)),
            'accuracy': float(accuracy_score(y[vm][v_dec], (v_states[v_dec] == 'Defect').astype(int))),
            'precision': float(precision_score(y[vm][v_dec], (v_states[v_dec] == 'Defect').astype(int), zero_division=0)),
            'recall': float(recall_score(y[vm][v_dec], (v_states[v_dec] == 'Defect').astype(int), zero_division=0))
        }
    return res

overall_f1_opt = calc_overall_3state(best_pv_f1)
overall_acc_opt = calc_overall_3state(best_pv_acc)

print(f"[시점별 F1 극대화]   전체: F1={overall_f1_opt['f1']:.4f}, Acc={overall_f1_opt['accuracy']*100:.2f}%, Rec={overall_f1_opt['recall']*100:.2f}%, 보류율={overall_f1_opt['uncertainty_rate']*100:.2f}%")
for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
    vr = overall_f1_opt['view_results'][v]
    print(f"  - [{v}] {vr['alias']}: tau=[{vr['tau_clean']:.3f}, {vr['tau_defect']:.3f}] | F1={vr['f1']:.4f}, Acc={vr['accuracy']*100:.2f}%, Rec={vr['recall']*100:.2f}%, 보류율={vr['uncertainty_rate']*100:.2f}%")

print(f"\n[시점별 정확도 극대화] 전체: F1={overall_acc_opt['f1']:.4f}, Acc={overall_acc_opt['accuracy']*100:.2f}%, Rec={overall_acc_opt['recall']*100:.2f}%, 보류율={overall_acc_opt['uncertainty_rate']*100:.2f}%")
for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
    vr = overall_acc_opt['view_results'][v]
    print(f"  - [{v}] {vr['alias']}: tau=[{vr['tau_clean']:.3f}, {vr['tau_defect']:.3f}] | F1={vr['f1']:.4f}, Acc={vr['accuracy']*100:.2f}%, Rec={vr['recall']*100:.2f}%, 보류율={vr['uncertainty_rate']*100:.2f}%")
