# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_evaluate_c1_v11_final.py
Purpose: c1(척추 중립 위반) v11_final (v11_a + v11_a_diet + v11_b TCN Tri-Ensemble) 전수 실측 및 가중치 저장
"""

import os
import sys
import time
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c1_model_v11_final import BabelSquatC1ModelV11Final
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_38D_PATH = r"D:\Data\피처_데이터\squat_c1_v11_a_38d_feats.npz"
FEATS_DIET_PATH = r"D:\Data\피처_데이터\squat_c1_v11_a_diet_27d_feats.npz"
TENSOR_PATH = r"D:\Data\피처_데이터\squat_c1_temporal_24d_64f.npz"
WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c1_model_v11_final_weights.pkl"
RESULTS_JSON_PATH = r"d:\src\models\second_models\babel_squat\c1_v11_final_evaluation_results.json"


def evaluate_metrics(y_true, y_prob, threshold=0.50):
    preds = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    acc = accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, y_prob) * 100.0
    except Exception:
        auc = 50.0
    return {
        'f1': round(f1, 2),
        'prec': round(prec, 2),
        'rec': round(rec, 2),
        'acc': round(acc, 2),
        'auc': round(auc, 2)
    }


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c1 v11_final (E0 + E1 궁극의 융합 챔피언) 5-Fold 전수 실측 평가", flush=True)
    print("  ★ v11_a (38D GBDT 10%) + v11_a_diet (27D GBDT 40%) + v11_b (24D TCN 50%) Tri-Ensemble", flush=True)
    print("  ★ 2,605개 마스터 데이터셋 5-Fold StratifiedGroupKFold 피험자 완전 격리 실측", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    views = master['views']
    subjects = master['subjects']
    y = master['Y_defects_golden'][:, 0].astype(int)
    N = len(seq_ids)

    X_38d = np.load(FEATS_38D_PATH, allow_pickle=True)['X_c1_v11_a']
    X_27d = np.load(FEATS_DIET_PATH, allow_pickle=True)['X_c1_v11_a_diet']
    X_temporal = np.load(TENSOR_PATH, allow_pickle=True)['X_temporal']

    with open(r'd:\src\models\second_models\babel_squat\c1_model_v11_a_weights.pkl', 'rb') as f:
        w_a = pickle.load(f)
    with open(r'd:\src\models\second_models\babel_squat\c1_model_v11_a_diet_weights.pkl', 'rb') as f:
        w_diet = pickle.load(f)
    with open(r'd:\src\models\second_models\babel_squat\c1_model_v11_b_weights.pkl', 'rb') as f:
        w_b = pickle.load(f)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    p_a = np.zeros(N, dtype=np.float32)
    p_d = np.zeros(N, dtype=np.float32)
    p_t = np.zeros(N, dtype=np.float32)
    tcn_models = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_38d, y, groups=subjects), 0):
        cb_a = w_a['cb_models'][fold]
        lgb_a = w_a['lgb_models'][fold]
        p_a[val_idx] = 0.60 * cb_a.predict_proba(X_38d[val_idx])[:, 1] + 0.40 * lgb_a.predict_proba(X_38d[val_idx])[:, 1]

        cb_d = w_diet['cb_models'][fold]
        lgb_d = w_diet['lgb_models'][fold]
        p_d[val_idx] = 0.60 * cb_d.predict_proba(X_27d[val_idx])[:, 1] + 0.40 * lgb_d.predict_proba(X_27d[val_idx])[:, 1]

        tcn = SquatC1DilatedTCN()
        tcn.load_state_dict(w_b['tcn_state_dicts'][fold])
        tcn.eval()
        tcn_models.append(tcn)
        with torch.no_grad():
            x_val_t = torch.tensor(X_temporal[val_idx].transpose(0, 2, 1), dtype=torch.float32)
            p_t[val_idx] = torch.sigmoid(tcn(x_val_t)).cpu().numpy().squeeze(1)

    p_final = 0.10 * p_a + 0.40 * p_d + 0.50 * p_t

    # 가중치 저장
    model_final = BabelSquatC1ModelV11Final()
    model_final.model_a.cb_models = w_a['cb_models']
    model_final.model_a.lgb_models = w_a['lgb_models']
    model_final.model_a.is_fitted = True

    model_final.model_diet.cb_models = w_diet['cb_models']
    model_final.model_diet.lgb_models = w_diet['lgb_models']
    model_final.model_diet.is_fitted = True

    model_final.tcn_models = tcn_models
    model_final.is_fitted = True
    model_final.save_weights(WEIGHTS_PATH)

    m_final = evaluate_metrics(y, p_final)

    print("\n" + "=" * 115, flush=True)
    print("  [★ c1 모델 세대별 실측 성능 종합 비교표 (N=2,605 전수)]", flush=True)
    print("=" * 115, flush=True)
    print(f"{'지표 (Metric)':<18} | {'v10 (28D 베이스)':<18} | {'v11_a_diet (27D)':<18} | {'v11_b (TCN Fusion)':<20} | {'★ v11_final':<18} | {'v10 대비 증감'}", flush=True)
    print("-" * 115, flush=True)
    for name, v10_v, diet_v, b_v, final_v in [
        ("F1-Score", 79.73, 80.52, 84.24, m_final['f1']),
        ("ROC-AUC", 88.33, 88.13, 91.59, m_final['auc']),
        ("Precision", 80.00, 81.88, 84.81, m_final['prec']),
        ("Recall", 79.46, 79.20, 83.67, m_final['rec']),
        ("Accuracy", 80.38, 81.54, 84.91, m_final['acc'])
    ]:
        delta = final_v - v10_v
        delta_str = f"+{delta:.2f}%p" if delta >= 0 else f"{delta:.2f}%p"
        print(f"{name:<18} | {v10_v:>16.2f}% | {diet_v:>16.2f}% | {b_v:>18.2f}% | {final_v:>16.2f}% | {delta_str:>12}", flush=True)

    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_names = ['view1 (후방 180°)', 'view2 (우측 45°)', 'view3 (정면 0°)', 'view4 (좌측 45°)', 'view5 (우측 후방 45°)']
    v10_vf1s = {'view1': 76.12, 'view2': 83.21, 'view3': 74.30, 'view4': 81.85, 'view5': 83.18}

    print("\n" + "=" * 115, flush=True)
    print("  [5대 시점별 세부 성능 감사 (Multi-View Audit)]", flush=True)
    print("=" * 115, flush=True)
    print(f"{'시점 (카메라)':<22} | {'v10 F1':<10} | {'★ v11_final F1':<16} | {'v10 대비 증감':<14} | {'final AUC':<12} | {'final Acc'}", flush=True)
    print("-" * 115, flush=True)

    view_results = {}
    for v, vn in zip(view_list, view_names):
        mask = (views == v)
        mv = evaluate_metrics(y[mask], p_final[mask])
        v10_f = v10_vf1s[v]
        d_f1 = mv['f1'] - v10_f
        d_f1_str = f"+{d_f1:.2f}%p" if d_f1 >= 0 else f"{d_f1:.2f}%p"
        print(f"{vn:<22} | {v10_f:>8.2f}% | {mv['f1']:>14.2f}% | {d_f1_str:>12} | {mv['auc']:>10.2f}% | {mv['acc']:>8.2f}%", flush=True)
        view_results[v] = {
            'view_name': vn, 'count': int(np.sum(mask)), 'v10_f1': v10_f,
            'final_f1': mv['f1'], 'delta_f1': round(d_f1, 2),
            'final_auc': mv['auc'], 'final_acc': mv['acc']
        }

    output_summary = {
        'model_name': 'c1_v11_final',
        'experiment_code': 'E0_E1_Tri_Ensemble',
        'description': 'v11_a (10%) + v11_a_diet (40%) + v11_b TCN (50%) 3-Way Tri-Ensemble',
        'overall_metrics': m_final,
        'baseline_v10_metrics': {'f1': 79.73, 'auc': 88.33, 'prec': 80.00, 'rec': 79.46, 'acc': 80.38},
        'view_metrics': view_results
    }
    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[*] v11_final 결과 JSON 저장 완료: {RESULTS_JSON_PATH}", flush=True)


if __name__ == '__main__':
    main()
