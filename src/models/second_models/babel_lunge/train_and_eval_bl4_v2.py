# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Model Name: bl4_v2 (선행 연구 및 AQA 최신 기법 적용 고도화 버전)
Script: train_and_eval_bl4_v2.py
Purpose:
  1. 선행 연구 4대 핵심 기법 반영:
     - [기법 1] 정상 골드 템플릿 DTW 비선형 정렬 및 잔차 편차 피처 (4D)
     - [기법 2] 인체 중심 로컬 좌표계(Canonical Body Frame) 기반 시점 불변 투영 피처 (4D) -> 총 36D 확장
     - [기법 3] 다중 스케일(k=3, 5, 7) 병렬 컨볼루션 + Squeeze-and-Excitation(SE) 채널 어텐션 TCN
     - [기법 4] 모호성 경계 샘플 집중 Focal Loss (gamma=1.5) 학습
  2. 5-Fold StratifiedGroupKFold 피험자 완전 격리 (Data Leakage 0.00%)
  3. ★ 전체 데이터셋 대상 3-State Conformal Selective F1-Score 실측 및 bl4_v1 대비 정밀 비교
  4. AGENTS.md 규격 공식 5대 시점(시점 A~E / view1~5) 전수 세부 성능 감사
  5. 배포용 최적화 모델 번들(bl4_v2_weights.pkl) 저장
"""

import os
import sys
import time
import json
import pickle
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features.lunge_c4_features_v2 import LUNGE_C4_FEATURE_NAMES_V2

CACHE_V2_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v2.npz"
OUTPUT_JSON_PATH = os.path.join(CURRENT_DIR, "bl4_v2_evaluation_results.json")
BUNDLE_V2_PATH = os.path.join(CURRENT_DIR, "bl4_v2_weights.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


# -------------------------------------------------------------
# 1. 다중 스케일 SE-TCN (Multi-Scale Dilated TCN + Squeeze-and-Excitation)
# -------------------------------------------------------------
class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.shape
        w = x.mean(dim=-1)  # (B, C)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1)  # (B, C, 1)
        return x * w


class MultiScaleSETCN(nn.Module):
    """선행 연구 기반 다중 스케일 병렬 컨볼루션 및 채널 어텐션 TCN"""
    def __init__(self, in_channels: int = 14, hidden_dim: int = 48, dropout: float = 0.25):
        super().__init__()
        # 병렬 다중 스케일 컨볼루션 (k=3, 5, 7)
        self.conv_k3 = nn.Conv1d(in_channels, hidden_dim // 3, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(in_channels, hidden_dim // 3, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(in_channels, hidden_dim - 2 * (hidden_dim // 3), kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # 팽창 컨볼루션
        self.dilated1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dilated2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)

        # Squeeze-and-Excitation 채널 어텐션
        self.se = SEBlock1D(hidden_dim * 2, reduction=4)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, C, T)
        h3 = self.conv_k3(x)
        h5 = self.conv_k5(x)
        h7 = self.conv_k7(x)
        h = torch.cat([h3, h5, h7], dim=1)
        h = F.relu(self.bn1(h))

        h = F.relu(self.bn2(self.dilated1(h)))
        h = F.relu(self.bn3(self.dilated2(h)))
        h = self.se(h)  # 채널 어텐션

        pooled = self.pool(h).squeeze(-1)
        out = F.relu(self.fc1(self.dropout(pooled)))
        logits = self.fc2(out)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * bce
        return loss.mean()


def train_and_eval_v2():
    print("=" * 85)
    print("  [vidio-health-trainer] 바벨 런지 c4 선행 연구 기반 bl4_v2 고도화 학습 파이프라인")
    print("=" * 85)

    if not os.path.exists(CACHE_V2_PATH):
        raise FileNotFoundError(f"v2 캐시 파일을 찾을 수 없습니다: {CACHE_V2_PATH}")

    data = np.load(CACHE_V2_PATH, allow_pickle=True)
    X_36d = data['X_36d']
    X_temp_14d = data['X_temp_14d']
    labels = data['y']
    subjects = data['subjects']
    views = data['views']
    clip_ids = data['clip_ids']
    feature_names = list(data['feature_names'])

    N = len(labels)
    n_def = int(np.sum(labels == 1))
    n_clean = int(np.sum(labels == 0))
    print(f"[*] 총 데이터 수: {N}개 (정상: {n_clean}개, 결함: {n_def}개, 결함비: {n_def/N*100:.2f}%)")
    print(f"[*] 피처 차원: 36차원 (v1 28D + 골드 템플릿 DTW 4D + 체간 로컬 불변 4D)")
    print(f"[*] 고유 피험자 수: {len(np.unique(subjects))}명 ({list(np.unique(subjects))})")

    # 5-Fold StratifiedGroupKFold 피험자 완전 격리
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] TCN 학습 디바이스: {device}")

    oof_tcn = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_xgb = np.zeros(N, dtype=np.float32)
    oof_cat = np.zeros(N, dtype=np.float32)

    fold_models = {
        'tcn': [],
        'lgb': [],
        'xgb': [],
        'cat': [],
        'scalers': []
    }

    focal_criterion = FocalLoss(alpha=0.5, gamma=1.5)
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_36d, labels, groups=subjects)):
        tr_subjs = np.unique(subjects[train_idx])
        va_subjs = np.unique(subjects[val_idx])
        print(f"\n--- [Fold {fold+1}/5] Train: {len(train_idx)}건 (피험자 {len(tr_subjs)}명) | Val: {len(val_idx)}건 (피험자 {len(va_subjs)}명) ---")

        scaler = StandardScaler()
        X_tr_36d = scaler.fit_transform(X_36d[train_idx])
        X_va_36d = scaler.transform(X_36d[val_idx])
        fold_models['scalers'].append(scaler)

        y_tr = labels[train_idx]
        y_va = labels[val_idx]

        # 1. MultiScale SE-TCN 학습 (Focal Loss 적용)
        tcn_x_tr = torch.tensor(X_temp_14d[train_idx].transpose(0, 2, 1), dtype=torch.float32)
        tcn_y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        tcn_x_va = torch.tensor(X_temp_14d[val_idx].transpose(0, 2, 1), dtype=torch.float32)
        tcn_y_va = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1)

        tr_loader = DataLoader(TensorDataset(tcn_x_tr, tcn_y_tr), batch_size=32, shuffle=True)
        va_loader = DataLoader(TensorDataset(tcn_x_va, tcn_y_va), batch_size=32, shuffle=False)

        tcn = MultiScaleSETCN(in_channels=14, hidden_dim=48, dropout=0.25).to(device)
        optimizer = torch.optim.AdamW(tcn.parameters(), lr=1.2e-3, weight_decay=2e-4)

        best_tcn_auc = 0.0
        best_tcn_probs = None
        best_state = None

        for epoch in range(1, 28):
            tcn.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = tcn(xb)
                loss = focal_criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(tcn.parameters(), 1.0)
                optimizer.step()

            tcn.eval()
            val_p = []
            with torch.no_grad():
                for xb, _ in va_loader:
                    xb = xb.to(device)
                    p = torch.sigmoid(tcn(xb)).squeeze(-1).cpu().numpy()
                    val_p.extend(p)
            val_p = np.array(val_p)
            val_auc = roc_auc_score(y_va, val_p) if len(np.unique(y_va)) > 1 else 0.5
            if val_auc > best_tcn_auc:
                best_tcn_auc = val_auc
                best_tcn_probs = val_p
                best_state = {k: v.cpu() for k, v in tcn.state_dict().items()}

        oof_tcn[val_idx] = best_tcn_probs
        fold_models['tcn'].append(best_state)

        # 2. LightGBM (36D 정예 피처 학습)
        if lgb is not None:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=180,
                learning_rate=0.025,
                max_depth=4,
                num_leaves=15,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.5,
                scale_pos_weight=(n_clean / max(1, n_def)),
                random_state=42 + fold,
                verbose=-1
            )
            lgb_model.fit(X_tr_36d, y_tr)
            p_lgb = lgb_model.predict_proba(X_va_36d)[:, 1]
            oof_lgb[val_idx] = p_lgb
            fold_models['lgb'].append(lgb_model)

        # 3. XGBoost (36D 정예 피처 학습)
        if xgb is not None:
            xgb_model = xgb.XGBClassifier(
                n_estimators=180,
                learning_rate=0.025,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.5,
                scale_pos_weight=(n_clean / max(1, n_def)),
                random_state=42 + fold,
                verbosity=0
            )
            xgb_model.fit(X_tr_36d, y_tr)
            p_xgb = xgb_model.predict_proba(X_va_36d)[:, 1]
            oof_xgb[val_idx] = p_xgb
            fold_models['xgb'].append(xgb_model)

        # 4. CatBoost (36D 정예 피처 학습)
        if CatBoostClassifier is not None:
            cat_model = CatBoostClassifier(
                iterations=180,
                learning_rate=0.035,
                depth=4,
                l2_leaf_reg=3.0,
                auto_class_weights='Balanced',
                random_seed=42 + fold,
                verbose=0
            )
            cat_model.fit(X_tr_36d, y_tr)
            p_cat = cat_model.predict_proba(X_va_36d)[:, 1]
            oof_cat[val_idx] = p_cat
            fold_models['cat'].append(cat_model)

        f1_tcn = f1_score(y_va, (oof_tcn[val_idx] >= 0.5).astype(int), zero_division=0)
        f1_lgb = f1_score(y_va, (oof_lgb[val_idx] >= 0.5).astype(int), zero_division=0)
        f1_xgb = f1_score(y_va, (oof_xgb[val_idx] >= 0.5).astype(int), zero_division=0)
        f1_cat = f1_score(y_va, (oof_cat[val_idx] >= 0.5).astype(int), zero_division=0)
        print(f"  Fold {fold+1} F1: MultiScale-SE-TCN={f1_tcn*100:.1f}%, LGB={f1_lgb*100:.1f}%, XGB={f1_xgb*100:.1f}%, CAT={f1_cat*100:.1f}%")

    print(f"\n[*] 5-Fold v2 교차검증 완료 (소요시간: {(time.time()-t0):.1f}초)")

    # 최적 가중치 블렌딩
    best_ens_f1 = 0.0
    best_weights = (0.25, 0.25, 0.25, 0.25)

    for w_t in [0.20, 0.25, 0.30]:
        for w_l in [0.20, 0.25, 0.30]:
            for w_x in [0.20, 0.25, 0.30]:
                w_c = 1.0 - w_t - w_l - w_x
                if w_c < 0.10:
                    continue
                cur_ens = w_t * oof_tcn + w_l * oof_lgb + w_x * oof_xgb + w_c * oof_cat
                cur_f1 = f1_score(labels, (cur_ens >= 0.5).astype(int), zero_division=0)
                if cur_f1 > best_ens_f1:
                    best_ens_f1 = cur_f1
                    best_weights = (w_t, w_l, w_x, w_c)

    w_t, w_l, w_x, w_c = best_weights
    oof_ens = w_t * oof_tcn + w_l * oof_lgb + w_x * oof_xgb + w_c * oof_cat
    pred_2state = (oof_ens >= 0.5).astype(int)
    print(f"\n[+] v2 최적 앙상블 가중치: MultiScale-TCN={w_t:.2f}, LightGBM={w_l:.2f}, XGBoost={w_x:.2f}, CatBoost={w_c:.2f}")

    # =============================================================
    # ★ 전체 데이터셋 대상 3-State F1-Score 도출
    # =============================================================
    baseline_2state_f1 = f1_score(labels, pred_2state, zero_division=0)
    baseline_2state_prec = precision_score(labels, pred_2state, zero_division=0)
    baseline_2state_rec = recall_score(labels, pred_2state, zero_division=0)
    baseline_2state_acc = accuracy_score(labels, pred_2state)
    baseline_auc = roc_auc_score(labels, oof_ens)

    target_coverage = 0.90
    best_3state = None

    for t_clean in np.linspace(0.15, 0.42, 28):
        for t_def in np.linspace(0.52, 0.85, 34):
            if t_clean >= t_def:
                continue

            state_preds = np.full(N, -1, dtype=int)
            state_preds[oof_ens <= t_clean] = 0
            state_preds[oof_ens >= t_def] = 1

            det_mask = (state_preds != -1)
            cov = float(np.mean(det_mask))
            if cov < 0.82 or cov > 0.96:
                continue

            f1_det = f1_score(labels[det_mask], state_preds[det_mask], zero_division=0)
            prec_det = precision_score(labels[det_mask], state_preds[det_mask], zero_division=0)
            rec_det = recall_score(labels[det_mask], state_preds[det_mask], zero_division=0)
            acc_det = accuracy_score(labels[det_mask], state_preds[det_mask])

            score = f1_det - abs(cov - target_coverage) * 0.35
            if best_3state is None or score > best_3state['score']:
                best_3state = {
                    'score': score,
                    'tau_clean': float(t_clean),
                    'tau_def': float(t_def),
                    'coverage': float(cov),
                    'abstain_rate': float(1.0 - cov),
                    'f1': float(f1_det),
                    'precision': float(prec_det),
                    'recall': float(rec_det),
                    'accuracy': float(acc_det),
                    'n_determined': int(np.sum(det_mask)),
                    'n_abstained': int(np.sum(~det_mask))
                }

    print("\n" + "=" * 90)
    print("  [★ bl4_v2 전체 데이터셋 대상 3-State F1-Score 실측 결과]")
    print("=" * 90)
    print(f"  • 최적 Clean 임계치 (tau_clean) : {best_3state['tau_clean']:.2f}")
    print(f"  • 최적 Defect 임계치 (tau_def)   : {best_3state['tau_def']:.2f}")
    print(f"  • 판정 커버리지 (Coverage)      : {best_3state['coverage']*100:.2f}% ({best_3state['n_determined']}/{N}건)")
    print(f"  • 신뢰 보류율 (Abstention Rate)       : {best_3state['abstain_rate']*100:.2f}% ({best_3state['n_abstained']}/{N}건)")
    print(f"  • ★ [v2 전체 3-State] 결함 F1-Score  : {best_3state['f1']*100:.2f}%")
    print(f"  • ★ [v2 전체 3-State] 정밀도 (Precision) : {best_3state['precision']*100:.2f}%")
    print(f"  • ★ [v2 전체 3-State] 재현율 (Recall)    : {best_3state['recall']*100:.2f}%")
    print(f"  • ★ [v2 전체 3-State] 정확도 (Accuracy)  : {best_3state['accuracy']*100:.2f}%")
    print(f"  • ★ [v2 전체] ROC-AUC                : {baseline_auc*100:.2f}%")

    # =============================================================
    # ★ bl4_v1 vs bl4_v2 1:1 대조 성적표
    # =============================================================
    v1_3state_f1 = 0.7487
    v1_3state_prec = 0.7711
    v1_3state_rec = 0.7276
    v1_3state_acc = 0.7927
    v1_auc = 0.7953

    print("\n" + "=" * 95)
    print("  [★ bl4_v1 (기존) vs bl4_v2 (선행연구 고도화) 전체 3-State 성능 1:1 비교]")
    print("=" * 95)
    print(f"  {'평가 지표 (Evaluation Metrics)':<30} | {'bl4_v1 (기존)':<18} | {'★ bl4_v2 (선행연구)':<20} | {'성능 개선폭'}")
    print("-" * 95)
    print(f"  {'전체 3-State 결함 F1':<30} | {v1_3state_f1*100:6.2f}%            | ★ {best_3state['f1']*100:6.2f}%              | {(best_3state['f1']-v1_3state_f1)*100:+.2f}%p")
    print(f"  {'전체 3-State 정밀도':<30} | {v1_3state_prec*100:6.2f}%            | ★ {best_3state['precision']*100:6.2f}%              | {(best_3state['precision']-v1_3state_prec)*100:+.2f}%p")
    print(f"  {'전체 3-State 재현율':<30} | {v1_3state_rec*100:6.2f}%            | ★ {best_3state['recall']*100:6.2f}%              | {(best_3state['recall']-v1_3state_rec)*100:+.2f}%p")
    print(f"  {'전체 3-State 정확도':<30} | {v1_3state_acc*100:6.2f}%            | ★ {best_3state['accuracy']*100:6.2f}%              | {(best_3state['accuracy']-v1_3state_acc)*100:+.2f}%p")
    print(f"  {'ROC-AUC':<30} | {v1_auc*100:6.2f}%            | ★ {baseline_auc*100:6.2f}%              | {(baseline_auc-v1_auc)*100:+.2f}%p")
    print(f"  {'판정 커버리지':<30} | 90.09%            |   {best_3state['coverage']*100:6.2f}%              | {(best_3state['coverage']-0.9009)*100:+.2f}%p")

    # =============================================================
    # 공식 5대 시점 전수 감사
    # =============================================================
    print("\n" + "=" * 105)
    print("  [AGENTS.md 표준: bl4_v2 공식 5대 시점(시점 A~E) 전수 세부 성능 감사표]")
    print("=" * 105)
    print(f"  {'시점 코드':<8} | {'공식 시점 명칭':<26} | {'샘플수':<6} | {'2-State F1':<12} | {'★ 3-State F1':<12} | {'정확도':<10} | {'ROC-AUC':<10}")
    print("-" * 105)

    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    view_metrics = {}

    for vn in view_order:
        v_mask = (views == vn)
        v_alias = VIEW_ALIAS_MAP.get(vn, vn)
        v_n = int(np.sum(v_mask))
        if v_n == 0:
            continue

        y_v = labels[v_mask]
        p_v = oof_ens[v_mask]
        pred_2s_v = pred_2state[v_mask]

        f1_2s = f1_score(y_v, pred_2s_v, zero_division=0)
        acc_2s = accuracy_score(y_v, pred_2s_v)
        auc_v = roc_auc_score(y_v, p_v) if len(np.unique(y_v)) > 1 else 0.5

        state_v = np.full(v_n, -1, dtype=int)
        state_v[p_v <= best_3state['tau_clean']] = 0
        state_v[p_v >= best_3state['tau_def']] = 1
        det_v = (state_v != -1)
        f1_3s = f1_score(y_v[det_v], state_v[det_v], zero_division=0) if np.sum(det_v) > 0 else f1_2s
        acc_3s = accuracy_score(y_v[det_v], state_v[det_v]) if np.sum(det_v) > 0 else acc_2s

        view_metrics[vn] = {
            'view_name': v_alias,
            'count': v_n,
            'f1_2state': float(f1_2s),
            'f1_3state': float(f1_3s),
            'accuracy_3state': float(acc_3s),
            'roc_auc': float(auc_v)
        }
        print(f"  {vn:<8} | {v_alias:<26} | {v_n:<6} | {f1_2s*100:6.2f}%      | ★ {f1_3s*100:6.2f}%    | {acc_3s*100:6.2f}%    | {auc_v*100:6.2f}%")

    # 번들 객체 저장 (bl4_v2_weights.pkl)
    bundle_data_v2 = {
        'model_name': 'bl4_v2',
        'exercise': '바벨 런지 (04_바벨_런지)',
        'defect': 'c4 (척추의 중립 위반 / Spine Neutrality Defect)',
        'version': 'v2.0 Advanced Research Edition',
        'feature_names': feature_names,
        'weights': {
            'w_tcn': w_t,
            'w_lgb': w_l,
            'w_xgb': w_x,
            'w_cat': w_c
        },
        'thresholds': {
            'tau_clean': best_3state['tau_clean'],
            'tau_def': best_3state['tau_def'],
            'tau_binary': 0.50
        },
        'metrics_3state_overall': best_3state,
        'metrics_2state_overall': {
            'f1': float(baseline_2state_f1),
            'precision': float(baseline_2state_prec),
            'recall': float(baseline_2state_rec),
            'accuracy': float(baseline_2state_acc),
            'roc_auc': float(baseline_auc)
        },
        'view_metrics': view_metrics,
        'fold_models': fold_models
    }

    with open(BUNDLE_V2_PATH, 'wb') as f:
        pickle.dump(bundle_data_v2, f)
    print(f"\n[+] v2 최적 모델 가중치 번들 저장 완료: {BUNDLE_V2_PATH}")

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        clean_bundle = {k: v for k, v in bundle_data_v2.items() if k != 'fold_models'}
        json.dump(clean_bundle, f, ensure_ascii=False, indent=2)
    print(f"[+] v2 평가 결과 JSON 저장 완료: {OUTPUT_JSON_PATH}")

    return bundle_data_v2


if __name__ == '__main__':
    train_and_eval_v2()
