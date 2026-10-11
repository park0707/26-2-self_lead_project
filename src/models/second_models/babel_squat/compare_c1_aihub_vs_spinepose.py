# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: compare_c1_aihub_vs_spinepose.py
Purpose: 바벨 스쿼트 c1 결함 모델의
         Model-A (기존: AI-Hub 3D JSON 라벨 보간 척추 기반 공식 챔피언) vs
         Model-B (신규: 원본 영상 SpinePose 순수 비전 추론 척추 기반 신규 모델)
         동일 조건 하 5-Fold StratifiedGroupKFold 실측 비교 평가
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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

# 경로 정의
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"

# Model-A (기존 AI-Hub 척추 보간 기반 피처 및 공식 확정 가중치)
A_FEATS_38D = r"D:\Data\피처_데이터\squat_c1_v11_a_38d_feats.npz"
A_FEATS_27D = r"D:\Data\피처_데이터\squat_c1_v11_a_diet_27d_feats.npz"
A_TENSOR_24D = r"D:\Data\피처_데이터\squat_c1_temporal_24d_64f.npz"

A_WEIGHTS_38 = r"d:\src\models\second_models\babel_squat\c1_model_v11_a_weights.pkl"
A_WEIGHTS_27 = r"d:\src\models\second_models\babel_squat\c1_model_v11_a_diet_weights.pkl"
A_WEIGHTS_TCN = r"d:\src\models\second_models\babel_squat\c1_model_v11_b_weights.pkl"

# Model-B (신규 SpinePose 순수 비전 척추 기반 피처)
B_FEATS_ALL = r"D:\Data\피처_데이터\squat_c1_v11_vision_spinepose_feats.npz"

OUT_JSON = r"d:\src\models\second_models\babel_squat\c1_aihub_vs_spinepose_comparison_results.json"
B_WEIGHTS_PATH = r"d:\src\models\second_models\babel_squat\c1_model_v11_spinepose_vision_weights.pkl"

VIEW_NAMES = ['view1', 'view2', 'view3', 'view4', 'view5']
VIEW_ALIASES = {
    'view1': '시점 E (좌측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view5': '시점 A (우측 후방 45°)'
}


def compute_metrics(y_true, y_prob, threshold=0.50):
    preds = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, preds, zero_division=0) * 100.0)
    prec = float(precision_score(y_true, preds, zero_division=0) * 100.0)
    rec = float(recall_score(y_true, preds, zero_division=0) * 100.0)
    acc = float(accuracy_score(y_true, preds) * 100.0)
    try:
        auc = float(roc_auc_score(y_true, y_prob) * 100.0)
    except Exception:
        auc = 50.0
    cm = confusion_matrix(y_true, preds).tolist()
    return {'f1': round(f1, 2), 'prec': round(prec, 2), 'rec': round(rec, 2), 'acc': round(acc, 2), 'auc': round(auc, 2), 'cm': cm}


def train_tcn_fast(X_tr, y_tr, X_val, in_channels=24, epochs=15, batch_size=64, lr=1.5e-3):
    """Dilated TCN 고속 학습 (CPU 최적화)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SquatC1DilatedTCN(in_channels=in_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr.transpose(0, 2, 1), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.transpose(0, 2, 1), dtype=torch.float32)

    ds_tr = TensorDataset(X_tr_t, y_tr_t)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t.to(device)).squeeze(-1).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    return model, val_probs


def run_pipeline():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c1 모델 정합성 감사: AI-Hub 3D JSON(Model-A) vs SpinePose 순수 비전(Model-B)", flush=True)
    print("  - 엄격한 5-Fold StratifiedGroupKFold 피험자 완전 격리 (Subject 1~9 전수)", flush=True)
    print("  - 5대 전 시점(시점 A ~ 시점 E / view1~view5) 전수 대조 감사", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    y = master['Y_defects_golden'][:, 0].astype(int)
    subjects = master['subjects']
    views = master['views']
    N = len(y)

    print(f"[*] 총 시퀀스 수: {N}, 양성(결함) 비율: {np.mean(y)*100:.1f}%", flush=True)

    # 1. Model-A 피처 및 기존 확정 가중치 로드
    print("[*] [1/3] Model-A (AI-Hub JSON 척추 기반 공식 확정 모델) 로드 중...", flush=True)
    A_38d = np.load(A_FEATS_38D, allow_pickle=True)['X_c1_v11_a']
    A_27d = np.load(A_FEATS_27D, allow_pickle=True)['X_c1_v11_a_diet']
    A_t24 = np.load(A_TENSOR_24D, allow_pickle=True)['X_temporal']

    with open(A_WEIGHTS_38, 'rb') as f:
        w_a_38 = pickle.load(f)
    with open(A_WEIGHTS_27, 'rb') as f:
        w_a_27 = pickle.load(f)
    with open(A_WEIGHTS_TCN, 'rb') as f:
        w_a_tcn = pickle.load(f)

    # 2. Model-B 피처 로드
    print("[*] [2/3] Model-B (SpinePose 순수 비전 척추 기반 신규 모델) 피처 로드 중...", flush=True)
    b_data = np.load(B_FEATS_ALL, allow_pickle=True)
    B_38d = b_data['X_38d_vision']
    B_27d = b_data['X_27d_vision']
    B_t24 = b_data['X_temporal_vision']

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_p_A = np.zeros(N, dtype=np.float32)
    oof_p_B = np.zeros(N, dtype=np.float32)

    b_models = {
        'cb_38d': [], 'lgb_38d': [],
        'cb_27d': [], 'lgb_27d': [],
        'tcn_state_dicts': []
    }

    t0 = time.time()

    print("\n[*] [3/3] 5-Fold StratifiedGroupKFold 실측 학습 및 검증 시작...", flush=True)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(A_38d, y, groups=subjects), 0):
        t_fold = time.time()
        # -------------------------------------------------------------
        # [Model-A] 기존 공식 확정 가중치로 100% 동일한 OOF 예측 산출
        # -------------------------------------------------------------
        cb_a_38 = w_a_38['cb_models'][fold]
        lgb_a_38 = w_a_38['lgb_models'][fold]
        p_a_38 = 0.60 * cb_a_38.predict_proba(A_38d[val_idx])[:, 1] + 0.40 * lgb_a_38.predict_proba(A_38d[val_idx])[:, 1]

        cb_a_27 = w_a_27['cb_models'][fold]
        lgb_a_27 = w_a_27['lgb_models'][fold]
        p_a_27 = 0.60 * cb_a_27.predict_proba(A_27d[val_idx])[:, 1] + 0.40 * lgb_a_27.predict_proba(A_27d[val_idx])[:, 1]

        tcn_a = SquatC1DilatedTCN(in_channels=24)
        tcn_a.load_state_dict(w_a_tcn['tcn_state_dicts'][fold])
        tcn_a.eval()
        with torch.no_grad():
            x_val_t = torch.tensor(A_t24[val_idx].transpose(0, 2, 1), dtype=torch.float32)
            logits_a = tcn_a(x_val_t).squeeze(-1).numpy()
            p_a_tcn = 1.0 / (1.0 + np.exp(-logits_a))

        oof_p_A[val_idx] = 0.10 * p_a_38 + 0.40 * p_a_27 + 0.50 * p_a_tcn

        # -------------------------------------------------------------
        # [Model-B] SpinePose 순수 비전 기반 3-Way Tri-Ensemble 신규 훈련
        # -------------------------------------------------------------
        # 1) Branch 1: 38D GBDT (CB 60% + LGB 40%)
        cb_b_38 = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.06, verbose=0, random_seed=42)
        cb_b_38.fit(B_38d[tr_idx], y[tr_idx])
        lgb_b_38 = lgb.LGBMClassifier(n_estimators=160, max_depth=5, learning_rate=0.06, verbose=-1, random_state=42)
        lgb_b_38.fit(B_38d[tr_idx], y[tr_idx])
        p_b_38 = 0.60 * cb_b_38.predict_proba(B_38d[val_idx])[:, 1] + 0.40 * lgb_b_38.predict_proba(B_38d[val_idx])[:, 1]

        # 2) Branch 2: 27D Diet GBDT
        cb_b_27 = CatBoostClassifier(iterations=220, depth=5, learning_rate=0.06, verbose=0, random_seed=42)
        cb_b_27.fit(B_27d[tr_idx], y[tr_idx])
        lgb_b_27 = lgb.LGBMClassifier(n_estimators=160, max_depth=5, learning_rate=0.06, verbose=-1, random_state=42)
        lgb_b_27.fit(B_27d[tr_idx], y[tr_idx])
        p_b_27 = 0.60 * cb_b_27.predict_proba(B_27d[val_idx])[:, 1] + 0.40 * lgb_b_27.predict_proba(B_27d[val_idx])[:, 1]

        # 3) Branch 3: 24D Dilated TCN
        tcn_b_model, p_b_tcn = train_tcn_fast(B_t24[tr_idx], y[tr_idx], B_t24[val_idx], in_channels=24, epochs=15, batch_size=64)

        oof_p_B[val_idx] = 0.10 * p_b_38 + 0.40 * p_b_27 + 0.50 * p_b_tcn

        b_models['cb_38d'].append(cb_b_38)
        b_models['lgb_38d'].append(lgb_b_38)
        b_models['cb_27d'].append(cb_b_27)
        b_models['lgb_27d'].append(lgb_b_27)
        b_models['tcn_state_dicts'].append(tcn_b_model.state_dict())

        m_a_fold = compute_metrics(y[val_idx], oof_p_A[val_idx])
        m_b_fold = compute_metrics(y[val_idx], oof_p_B[val_idx])
        print(f"  >>> [Fold {fold+1}/5] 완료 ({time.time()-t_fold:.1f}초) | "
              f"Model-A F1={m_a_fold['f1']:.2f}% vs Model-B F1={m_b_fold['f1']:.2f}% "
              f"(격차: {m_b_fold['f1']-m_a_fold['f1']:+.2f}%p)", flush=True)

    print(f"\n[*] 5-Fold 전체 교차검증 완수! 총 소요 시간: {time.time()-t0:.1f}초", flush=True)

    # -----------------------------------------------------------------
    # 종합 성능 평가 및 5대 시점별 세부 분석
    # -----------------------------------------------------------------
    metrics_A_total = compute_metrics(y, oof_p_A)
    metrics_B_total = compute_metrics(y, oof_p_B)

    # 5대 시점별 성능 집계
    view_results = {}
    for vn in VIEW_NAMES:
        mask_v = (views == vn)
        m_a_v = compute_metrics(y[mask_v], oof_p_A[mask_v])
        m_b_v = compute_metrics(y[mask_v], oof_p_B[mask_v])
        alias = VIEW_ALIASES.get(vn, vn)
        view_results[vn] = {
            'alias': alias,
            'count': int(np.sum(mask_v)),
            'model_a': m_a_v,
            'model_b': m_b_v,
            'delta_f1': round(m_b_v['f1'] - m_a_v['f1'], 2),
            'delta_auc': round(m_b_v['auc'] - m_a_v['auc'], 2),
            'delta_acc': round(m_b_v['acc'] - m_a_v['acc'], 2)
        }

    # 3-State 선택적 거부(Selective Abstention) 비교
    def evaluate_selective(p_arr, tau_norm=0.40, tau_def=0.60):
        certain_mask = (p_arr <= tau_norm) | (p_arr >= tau_def)
        cov = float(np.mean(certain_mask) * 100.0)
        sel_y = y[certain_mask]
        sel_p = p_arr[certain_mask]
        sel_preds = (sel_p >= 0.50).astype(int)
        sel_f1 = float(f1_score(sel_y, sel_preds, zero_division=0) * 100.0)
        sel_acc = float(accuracy_score(sel_y, sel_preds) * 100.0)
        return {'coverage': round(cov, 2), 'selective_f1': round(sel_f1, 2), 'selective_acc': round(sel_acc, 2)}

    sel_A = evaluate_selective(oof_p_A)
    sel_B = evaluate_selective(oof_p_B)

    final_report = {
        'total_samples': N,
        'model_a_oracle_aihub': {
            'description': 'AI-Hub 3D JSON 라벨 보간 척추 기반 공식 확정 모델 (Oracle Baseline)',
            'overall': metrics_A_total,
            'selective_abstention': sel_A
        },
        'model_b_vision_spinepose': {
            'description': 'SpinePose 순수 비전 추론 척추 기반 신규 모델 (Pure Vision Real-World)',
            'overall': metrics_B_total,
            'selective_abstention': sel_B
        },
        'delta_overall': {
            'delta_f1': round(metrics_B_total['f1'] - metrics_A_total['f1'], 2),
            'delta_auc': round(metrics_B_total['auc'] - metrics_A_total['auc'], 2),
            'delta_acc': round(metrics_B_total['acc'] - metrics_A_total['acc'], 2),
            'delta_prec': round(metrics_B_total['prec'] - metrics_A_total['prec'], 2),
            'delta_rec': round(metrics_B_total['rec'] - metrics_A_total['rec'], 2)
        },
        'view_breakdown': view_results
    }

    # 결과 JSON 저장
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 최종 비교 분석 결과 JSON 저장 완료: {OUT_JSON}", flush=True)

    # Model-B 가중치 저장
    with open(B_WEIGHTS_PATH, 'wb') as f:
        pickle.dump(b_models, f)
    print(f"[OK] Model-B 모델 공식 가중치 저장 완료: {B_WEIGHTS_PATH}", flush=True)

    # 요약 출력
    print("\n" + "=" * 115, flush=True)
    print("  ★ [바벨 스쿼트 c1 모델] Model-A(기존 AI-Hub 보간) vs Model-B(신규 SpinePose 순수 비전) 비교 총괄", flush=True)
    print("=" * 115, flush=True)
    print(f"  • [전체 이진 F1]   Model-A: {metrics_A_total['f1']:.2f}%  ──>  Model-B: {metrics_B_total['f1']:.2f}%  (격차: {final_report['delta_overall']['delta_f1']:+.2f}%p)", flush=True)
    print(f"  • [전체 정확도]   Model-A: {metrics_A_total['acc']:.2f}%  ──>  Model-B: {metrics_B_total['acc']:.2f}%  (격차: {final_report['delta_overall']['delta_acc']:+.2f}%p)", flush=True)
    print(f"  • [전체 ROC-AUC]  Model-A: {metrics_A_total['auc']:.2f}%  ──>  Model-B: {metrics_B_total['auc']:.2f}%  (격차: {final_report['delta_overall']['delta_auc']:+.2f}%p)", flush=True)
    print(f"  • [3-State Coverage] Model-A: {sel_A['coverage']:.1f}% (Sel-F1: {sel_A['selective_f1']:.1f}%) ──> Model-B: {sel_B['coverage']:.1f}% (Sel-F1: {sel_B['selective_f1']:.1f}%)", flush=True)
    print("-" * 115, flush=True)
    print("  [5대 시점별 세부 F1 비교]", flush=True)
    for vn, r in view_results.items():
        print(f"    - {r['alias']} (N={r['count']}): Model-A {r['model_a']['f1']:.2f}% vs Model-B {r['model_b']['f1']:.2f}% ({r['delta_f1']:+.2f}%p)", flush=True)
    print("=" * 115, flush=True)


if __name__ == '__main__':
    run_pipeline()
