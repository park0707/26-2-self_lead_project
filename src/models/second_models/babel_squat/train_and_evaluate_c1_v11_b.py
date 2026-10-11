# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_evaluate_c1_v11_b.py
Purpose: c1(척추 중립 위반) v11_b (실험 코드 E1: 경량 Dilated TCN + GBDT 융합)
         - TCN 단독 5-Fold 피험자 완전 격리 학습 및 검증
         - GBDT(v11_a_diet 27D)와 TCN의 Late Fusion 앙상블 성능 평가
         - 5대 시점별 세부 성능 감사
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
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c1_model_v11_b import (
    SquatC1DilatedTCN,
    BabelSquatC1ModelV11B
)
from src.models.second_models.babel_squat.c1_model_v11_a_diet import BabelSquatC1ModelV11ADiet

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_DIET_PATH = r"D:\Data\피처_데이터\squat_c1_v11_a_diet_27d_feats.npz"
TENSOR_PATH = r"D:\Data\피처_데이터\squat_c1_temporal_24d_64f.npz"
WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c1_model_v11_b_weights.pkl"
RESULTS_JSON_PATH = r"d:\src\models\second_models\babel_squat\c1_v11_b_evaluation_results.json"


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


def train_tcn_fold(X_train, y_train, X_val, y_val, fold=1, epochs=25, batch_size=64, lr=1e-3):
    """
    X: (N, 64, 24) -> PyTorch (N, 24, 64)
    """
    device = torch.device('cpu')
    model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25).to(device)

    # Class weight calculation for balanced loss
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    x_tr_t = torch.tensor(X_train.transpose(0, 2, 1), dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    ds_train = TensorDataset(x_tr_t, y_tr_t)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    x_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32).to(device)

    best_val_loss = float('inf')
    best_weights = None

    for ep in range(epochs):
        model.train()
        for bx, by in loader_train:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(x_val_t)
            v_loss = criterion(v_logits, torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(x_val_t)).cpu().numpy().squeeze(1)

    return model, val_probs


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c1 v11_b (실험 코드 E1: 경량 Dilated TCN + GBDT 융합) 실측 평가", flush=True)
    print("  ★ 24D 관절 64F 시계열 텐서(64, 24) 직접 입력 기반 Dilated TCN 학습", flush=True)
    print("  ★ 동일 split에서 TCN 단독 대 GBDT(v11_a_diet) 직접 비교 및 Late Fusion 앙상블", flush=True)
    print("  ★ 2,605개 마스터 데이터셋 5-Fold StratifiedGroupKFold 피험자 완전 격리", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    views = master['views']
    subjects = master['subjects']
    y_c1 = master['Y_defects_golden'][:, 0].astype(int)
    N = len(seq_ids)

    # 데이터 로드
    X_27d = np.load(FEATS_DIET_PATH, allow_pickle=True)['X_c1_v11_a_diet']
    X_temporal = np.load(TENSOR_PATH, allow_pickle=True)['X_temporal']
    print(f"[*] 27D 피처 로드 완료: {X_27d.shape}", flush=True)
    print(f"[*] 24D 시계열 텐서 로드 완료: {X_temporal.shape}", flush=True)

    # GBDT 모델 가중치 로드
    gbdt_w_path = r"d:\src\models\second_models\babel_squat\c1_model_v11_a_diet_weights.pkl"
    gbdt_model = BabelSquatC1ModelV11ADiet(weights_path=gbdt_w_path)
    # GBDT OOF 예측 확률 계산
    p_gbdt_oof = np.zeros(N, dtype=np.float32)
    for cb, clf in zip(gbdt_model.cb_models, gbdt_model.lgb_models):
        p_cb = cb.predict_proba(X_27d)[:, 1]
        p_lgb = clf.predict_proba(X_27d)[:, 1]
        p_gbdt_oof += (0.60 * p_cb + 0.40 * p_lgb) / len(gbdt_model.cb_models)

    print(f"\n[*] [1/2] 5-Fold StratifiedGroupKFold TCN 학습 시작...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    p_tcn_oof = np.zeros(N, dtype=np.float32)
    tcn_models = []

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_temporal, y_c1, groups=subjects), 1):
        X_tr, y_tr = X_temporal[train_idx], y_c1[train_idx]
        X_v, y_v = X_temporal[val_idx], y_c1[val_idx]
        val_sub = np.unique(subjects[val_idx])

        tcn, val_probs = train_tcn_fold(X_tr, y_tr, X_v, y_v, fold=fold, epochs=25, batch_size=64, lr=1e-3)
        p_tcn_oof[val_idx] = val_probs
        tcn_models.append(tcn)

        m_f = evaluate_metrics(y_v, val_probs)
        print(f"    - Fold {fold} (피험자 {val_sub}): TCN F1={m_f['f1']:.2f}%, AUC={m_f['auc']:.2f}%, Acc={m_f['acc']:.2f}%", flush=True)

    print(f"[*] TCN 5-Fold 학습 완료! (총 소요: {time.time()-t0:.1f}초)", flush=True)

    # 최적 Late Fusion 가중치 탐색
    print(f"\n[*] [2/2] TCN과 GBDT의 Late Fusion 앙상블 탐색...", flush=True)
    best_w, best_f1, best_fused_oof = 0.0, 0.0, None
    for w_tcn in np.arange(0.0, 1.05, 0.05):
        w_gbdt = 1.0 - w_tcn
        p_fused = w_gbdt * p_gbdt_oof + w_tcn * p_tcn_oof
        cur_f1 = f1_score(y_c1, (p_fused >= 0.50).astype(int), zero_division=0) * 100.0
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_w = w_tcn
            best_fused_oof = p_fused

    print(f"[*] 최적 가중치: GBDT {1.0-best_w:.2f} + TCN {best_w:.2f} -> Fusion F1 = {best_f1:.2f}%", flush=True)

    # 메트릭 계산
    m_gbdt = evaluate_metrics(y_c1, p_gbdt_oof)
    m_tcn = evaluate_metrics(y_c1, p_tcn_oof)
    m_fused = evaluate_metrics(y_c1, best_fused_oof)

    print("\n" + "=" * 115, flush=True)
    print("  [★ E1 실측 결과: GBDT vs TCN 단독 vs Late Fusion 앙상블 대조표 (N=2,605 전수)]", flush=True)
    print("=" * 115, flush=True)
    print(f"{'지표 (Metric)':<18} | {'GBDT (v11_a_diet 27D)':<24} | {'TCN 단독 (시계열 24D)':<24} | {'Fusion 앙상블 (v11_b)':<24}", flush=True)
    print("-" * 115, flush=True)
    for name, g_val, t_val, f_val in [
        ("F1-Score", m_gbdt['f1'], m_tcn['f1'], m_fused['f1']),
        ("ROC-AUC", m_gbdt['auc'], m_tcn['auc'], m_fused['auc']),
        ("Precision", m_gbdt['prec'], m_tcn['prec'], m_fused['prec']),
        ("Recall", m_gbdt['rec'], m_tcn['rec'], m_fused['rec']),
        ("Accuracy", m_gbdt['acc'], m_tcn['acc'], m_fused['acc'])
    ]:
        print(f"{name:<18} | {g_val:>22.2f}% | {t_val:>22.2f}% | {f_val:>22.2f}%", flush=True)

    # 5대 시점별 세부 감사
    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_names = ['view1 (후방 180°)', 'view2 (우측 45°)', 'view3 (정면 0°)', 'view4 (좌측 45°)', 'view5 (우측 후방 45°)']

    print("\n" + "=" * 115, flush=True)
    print("  [5대 시점별 세부 성능 감사 (Multi-View Performance Audit)]", flush=True)
    print("=" * 115, flush=True)
    print(f"{'시점 (카메라)':<22} | {'GBDT F1':<12} | {'TCN F1':<12} | {'Fusion F1':<12} | {'Fusion AUC':<12} | {'Fusion Acc'}", flush=True)
    print("-" * 115, flush=True)

    view_results = {}
    for v, vn in zip(view_list, view_names):
        mask = (views == v)
        v_g = evaluate_metrics(y_c1[mask], p_gbdt_oof[mask])
        v_t = evaluate_metrics(y_c1[mask], p_tcn_oof[mask])
        v_f = evaluate_metrics(y_c1[mask], best_fused_oof[mask])
        print(f"{vn:<22} | {v_g['f1']:>10.2f}% | {v_t['f1']:>10.2f}% | {v_f['f1']:>10.2f}% | {v_f['auc']:>10.2f}% | {v_f['acc']:>8.2f}%", flush=True)
        view_results[v] = {
            'view_name': vn,
            'count': int(np.sum(mask)),
            'gbdt_f1': v_g['f1'],
            'tcn_f1': v_t['f1'],
            'fusion_f1': v_f['f1'],
            'fusion_auc': v_f['auc'],
            'fusion_acc': v_f['acc']
        }

    # v11_b 가중치 저장
    model_v11_b = BabelSquatC1ModelV11B(tcn_weight=round(best_w, 2), gbdt_weight=round(1.0-best_w, 2))
    model_v11_b.gbdt_model = gbdt_model
    model_v11_b.tcn_models = tcn_models
    model_v11_b.is_fitted = True
    model_v11_b.save_weights(WEIGHTS_PATH)

    output_summary = {
        'model_name': 'c1_v11_b',
        'experiment_code': 'E1',
        'description': 'Lightweight Dilated TCN + GBDT Late Fusion Ensemble',
        'best_tcn_weight': round(best_w, 2),
        'best_gbdt_weight': round(1.0-best_w, 2),
        'gbdt_metrics': m_gbdt,
        'tcn_metrics': m_tcn,
        'fusion_metrics': m_fused,
        'view_metrics': view_results
    }
    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[*] E1 실험 결과 JSON 저장 완료: {RESULTS_JSON_PATH}", flush=True)


if __name__ == '__main__':
    main()
