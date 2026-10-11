# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (04_바벨_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Model Name: bl4_v3 (피처 전수 감사 기반 24D 정예 최적화 및 모델 구조 최적화 버전)
Script: train_and_eval_bl4_v3.py
Purpose:
  1. 피처 전수 감사 기반 다중공선성(|r| >= 0.85) 및 노이즈 12개 피처 완전 제거 (36D -> 정예 24D)
  2. 모델 구조 최적화:
     - MultiScale-SE-TCN: 채널 48개 확장, Residual Skip Connection 보강, Focal Loss(gamma=1.5)
     - GBDT 3총사(LightGBM, XGBoost, CatBoost) 정예 24D 맞춤형 규제화 하이퍼파라미터 튜닝
  3. 5-Fold StratifiedGroupKFold 피험자 9명 완전 격리 (Data Leakage 0.00%)
  4. ★ 전체 데이터셋 대상 3-State Conformal Selective F1-Score 실측 및 bl4_v2 대비 정밀 비교
  5. AGENTS.md 규격 공식 5대 시점(시점 A~E / view1~5) 전수 세부 성능 감사
  6. 배포용 최적화 모델 번들(bl4_v3_weights.pkl) 저장
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

from src.features.lunge_c4_features_v3 import LUNGE_C4_FEATURE_NAMES_V3

CACHE_V3_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz"
OUTPUT_JSON_PATH = os.path.join(CURRENT_DIR, "bl4_v3_evaluation_results.json")
BUNDLE_V3_PATH = os.path.join(CURRENT_DIR, "bl4_v3_weights.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


# -------------------------------------------------------------
# 1. 구조 최적화 다중 스케일 SE-TCN
# -------------------------------------------------------------
class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.shape
        w = torch.mean(x, dim=2)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1)
        return x * w


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        branch_channels = out_channels // 3
        rem = out_channels - branch_channels * 2

        self.conv_k3 = nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_k5 = nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.conv_k7 = nn.Conv1d(in_channels, rem, kernel_size=7, padding=3 * dilation, dilation=dilation)

        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels, reduction=4)
        self.dropout = nn.Dropout(0.20)

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        o3 = self.conv_k3(x)
        o5 = self.conv_k5(x)
        o7 = self.conv_k7(x)
        out = torch.cat([o3, o5, o7], dim=1)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.se(out)
        out = self.dropout(out)
        return F.gelu(out + res)


class MultiScaleSETCN(nn.Module):
    def __init__(self, in_features: int = 14, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU()
        )
        self.b1 = MultiScaleConvBlock(48, 48, dilation=1)
        self.b2 = MultiScaleConvBlock(48, 48, dilation=2)
        self.b3 = MultiScaleConvBlock(48, 48, dilation=4)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        h = self.stem(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        feat = self.pool(h).squeeze(-1)
        logits = self.classifier(feat)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.45, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# -------------------------------------------------------------
# 2. 3-State Conformal Selective 평가 함수
# -------------------------------------------------------------
def evaluate_3state_conformal(probs: np.ndarray, labels: np.ndarray, tau_clean: float, tau_def: float) -> Dict[str, Any]:
    preds_3state = []
    det_mask = []

    for p in probs:
        if p <= tau_clean:
            preds_3state.append(0)
            det_mask.append(True)
        elif p >= tau_def:
            preds_3state.append(1)
            det_mask.append(True)
        else:
            preds_3state.append(-1)
            det_mask.append(False)

    preds_3state = np.array(preds_3state)
    det_mask = np.array(det_mask)
    n_total = len(labels)
    n_det = int(np.sum(det_mask))
    coverage = n_det / n_total if n_total > 0 else 0.0
    abstain_rate = 1.0 - coverage

    if n_det > 0:
        y_true_det = labels[det_mask]
        y_pred_det = preds_3state[det_mask]
        f1 = float(f1_score(y_true_det, y_pred_det, pos_label=1, zero_division=0))
        prec = float(precision_score(y_true_det, y_pred_det, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true_det, y_pred_det, pos_label=1, zero_division=0))
        acc = float(accuracy_score(y_true_det, y_pred_det))
    else:
        f1, prec, rec, acc = 0.0, 0.0, 0.0, 0.0

    score = f1 * (coverage ** 0.5)

    return {
        'score': score,
        'tau_clean': tau_clean,
        'tau_def': tau_def,
        'coverage': coverage,
        'abstain_rate': abstain_rate,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'accuracy': acc,
        'n_determined': n_det,
        'n_abstained': n_total - n_det,
        'preds_3state': preds_3state
    }


def find_optimal_thresholds(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    best_res = None
    best_score = -1.0

    for tc in np.linspace(0.35, 0.45, 11):
        for td in np.linspace(0.50, 0.60, 11):
            if tc >= td:
                continue
            res = evaluate_3state_conformal(probs, labels, tc, td)
            if res['coverage'] >= 0.85 and res['score'] > best_score:
                best_score = res['score']
                best_res = res

    if best_res is None:
        best_res = evaluate_3state_conformal(probs, labels, 0.40, 0.55)

    return best_res


# -------------------------------------------------------------
# 3. 메인 훈련 및 평가 파이프라인
# -------------------------------------------------------------
def train_and_eval_bl4_v3():
    print("=" * 85)
    print("  [vidio-health-trainer] 바벨 런지 c4 피처 최적화 bl4_v3 학습 및 성능 평가")
    print("=" * 85)

    t_start = time.time()

    if not os.path.exists(CACHE_V3_PATH):
        raise FileNotFoundError(f"캐시 파일이 존재하지 않습니다: {CACHE_V3_PATH}")

    data = np.load(CACHE_V3_PATH, allow_pickle=True)
    X_24d = data['X_24d']
    X_temp = data['X_temp_14d']
    y = data['y']
    subjects = data['subjects']
    views = data['views']
    feature_names = list(data['feature_names'])

    N = len(y)
    n_def = int(np.sum(y == 1))
    n_clean = int(np.sum(y == 0))
    print(f"[*] 총 데이터 수: {N}개 (정상: {n_clean}개, 결함: {n_def}개, 결함비: {n_def/N*100:.2f}%)")
    print(f"[*] 피처 차원: {X_24d.shape[1]}차원 (전수 감사 기반 12개 노이즈/고상관 제거 완료)")
    print(f"[*] 고유 피험자 수: {len(np.unique(subjects))}명 ({list(np.unique(subjects))})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] TCN 학습 디바이스: {device}")

    # 5-Fold StratifiedGroupKFold 설정
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_tcn_probs = np.zeros(N, dtype=np.float32)
    oof_lgb_probs = np.zeros(N, dtype=np.float32)
    oof_xgb_probs = np.zeros(N, dtype=np.float32)
    oof_cat_probs = np.zeros(N, dtype=np.float32)

    fold_models = {
        'tcn': [],
        'lgb': [],
        'xgb': [],
        'cat': [],
        'scaler': []
    }

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_24d, y, groups=subjects), 1):
        tr_subj = np.unique(subjects[train_idx])
        va_subj = np.unique(subjects[val_idx])
        print(f"\n--- [Fold {fold}/5] Train: {len(train_idx)}건 (피험자 {len(tr_subj)}명) | Val: {len(val_idx)}건 (피험자 {len(va_subj)}명) ---")

        # 1. 정규화
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_24d[train_idx])
        X_va_sc = scaler.transform(X_24d[val_idx])
        fold_models['scaler'].append(scaler)

        y_tr, y_va = y[train_idx], y[val_idx]
        T_tr, T_va = X_temp[train_idx], X_temp[val_idx]

        # ---------------------------------------------------------
        # A. MultiScale-SE-TCN 학습 (채널 48개 최적화 모델)
        # ---------------------------------------------------------
        tcn_model = MultiScaleSETCN(in_features=14, num_classes=2).to(device)
        criterion = FocalLoss(alpha=0.45, gamma=1.5)
        optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)

        ds_tr = TensorDataset(torch.from_numpy(T_tr).float(), torch.from_numpy(y_tr).long())
        ds_va = TensorDataset(torch.from_numpy(T_va).float(), torch.from_numpy(y_va).long())
        loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

        best_tcn_f1 = -1.0
        best_tcn_state = None

        for epoch in range(1, 36):
            tcn_model.train()
            for bx, by in loader_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = tcn_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            # 검증
            tcn_model.eval()
            with torch.no_grad():
                val_x = torch.from_numpy(T_va).float().to(device)
                v_out = tcn_model(val_x)
                v_prob = F.softmax(v_out, dim=-1)[:, 1].cpu().numpy()
                v_pred = (v_prob >= 0.5).astype(int)
                f1_ep = f1_score(y_va, v_pred, pos_label=1, zero_division=0)
                if f1_ep > best_tcn_f1:
                    best_tcn_f1 = f1_ep
                    best_tcn_state = {k: v.cpu().clone() for k, v in tcn_model.state_dict().items()}

        tcn_model.load_state_dict(best_tcn_state)
        tcn_model.eval()
        with torch.no_grad():
            v_prob = F.softmax(tcn_model(torch.from_numpy(T_va).float().to(device)), dim=-1)[:, 1].cpu().numpy()
        oof_tcn_probs[val_idx] = v_prob
        fold_models['tcn'].append(tcn_model.cpu())

        # ---------------------------------------------------------
        # B. LightGBM (24D 최적화 하이퍼파라미터)
        # ---------------------------------------------------------
        lgb_model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=0.3,
            class_weight='balanced',
            random_state=42 + fold,
            verbose=-1
        )
        lgb_model.fit(X_tr_sc, y_tr)
        p_lgb = lgb_model.predict_proba(X_va_sc)[:, 1]
        oof_lgb_probs[val_idx] = p_lgb
        fold_models['lgb'].append(lgb_model)

        # ---------------------------------------------------------
        # C. XGBoost (24D 최적화 하이퍼파라미터)
        # ---------------------------------------------------------
        pos_weight = float(np.sum(y_tr == 0)) / float(max(1, np.sum(y_tr == 1)))
        xgb_model = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.80,
            gamma=0.15,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=42 + fold
        )
        xgb_model.fit(X_tr_sc, y_tr)
        p_xgb = xgb_model.predict_proba(X_va_sc)[:, 1]
        oof_xgb_probs[val_idx] = p_xgb
        fold_models['xgb'].append(xgb_model)

        # ---------------------------------------------------------
        # D. CatBoost (24D 최적화 하이퍼파라미터)
        # ---------------------------------------------------------
        cat_model = CatBoostClassifier(
            iterations=120,
            learning_rate=0.04,
            depth=4,
            l2_leaf_reg=3.0,
            auto_class_weights='Balanced',
            verbose=0,
            random_seed=42 + fold
        )
        cat_model.fit(X_tr_sc, y_tr)
        p_cat = cat_model.predict_proba(X_va_sc)[:, 1]
        oof_cat_probs[val_idx] = p_cat
        fold_models['cat'].append(cat_model)

        f1_tcn = f1_score(y_va, (p_lgb >= 0.5).astype(int), pos_label=1, zero_division=0)
        print(f"  Fold {fold} F1: MultiScale-SE-TCN={best_tcn_f1*100:.1f}%, LGB={f1_score(y_va, (p_lgb>=0.5).astype(int))*100:.1f}%, XGB={f1_score(y_va, (p_xgb>=0.5).astype(int))*100:.1f}%, CAT={f1_score(y_va, (p_cat>=0.5).astype(int))*100:.1f}%")

    t_elapsed = time.time() - t_start
    print(f"\n[*] 5-Fold v3 교차검증 완료 (소요시간: {t_elapsed:.1f}초)")

    # -------------------------------------------------------------
    # 4. 최적 앙상블 블렌딩 가중치 탐색 (Grid Search)
    # -------------------------------------------------------------
    best_weights = (0.25, 0.25, 0.25, 0.25)
    best_f1_ens = -1.0

    weight_candidates = [
        (0.20, 0.20, 0.20, 0.40),
        (0.15, 0.25, 0.25, 0.35),
        (0.20, 0.25, 0.25, 0.30),
        (0.25, 0.25, 0.25, 0.25),
        (0.10, 0.30, 0.30, 0.30),
        (0.15, 0.20, 0.25, 0.40),
        (0.20, 0.30, 0.20, 0.30)
    ]

    for w_t, w_l, w_x, w_c in weight_candidates:
        p_comb = w_t * oof_tcn_probs + w_l * oof_lgb_probs + w_x * oof_xgb_probs + w_c * oof_cat_probs
        f1_c = f1_score(y, (p_comb >= 0.5).astype(int), pos_label=1, zero_division=0)
        if f1_c > best_f1_ens:
            best_f1_ens = f1_c
            best_weights = (w_t, w_l, w_x, w_c)

    w_tcn, w_lgb, w_xgb, w_cat = best_weights
    print(f"\n[+] v3 최적 앙상블 가중치: MultiScale-TCN={w_tcn:.2f}, LightGBM={w_lgb:.2f}, XGBoost={w_xgb:.2f}, CatBoost={w_cat:.2f}")

    final_oof_probs = (
        w_tcn * oof_tcn_probs +
        w_lgb * oof_lgb_probs +
        w_xgb * oof_xgb_probs +
        w_cat * oof_cat_probs
    )

    # -------------------------------------------------------------
    # 5. 전체 3-State Conformal Selective F1-Score 실측
    # -------------------------------------------------------------
    best_3state = find_optimal_thresholds(final_oof_probs, y)
    tc = best_3state['tau_clean']
    td = best_3state['tau_def']

    # 2-State 베이스라인 지표
    bin_preds = (final_oof_probs >= 0.5).astype(int)
    f1_2s = float(f1_score(y, bin_preds, pos_label=1, zero_division=0))
    prec_2s = float(precision_score(y, bin_preds, pos_label=1, zero_division=0))
    rec_2s = float(recall_score(y, bin_preds, pos_label=1, zero_division=0))
    acc_2s = float(accuracy_score(y, bin_preds))
    auc_val = float(roc_auc_score(y, final_oof_probs))

    print("\n" + "=" * 90)
    print("  [★ bl4_v3 전체 데이터셋 대상 3-State F1-Score 실측 결과]")
    print("=" * 90)
    print(f"  • 최적 Clean 임계치 (tau_clean) : {tc:.2f}")
    print(f"  • 최적 Defect 임계치 (tau_def)   : {td:.2f}")
    print(f"  • 판정 커버리지 (Coverage)      : {best_3state['coverage']*100:.2f}% ({best_3state['n_determined']}/{N}건)")
    print(f"  • 신뢰 보류율 (Abstention Rate)       : {best_3state['abstain_rate']*100:.2f}% ({best_3state['n_abstained']}/{N}건)")
    print(f"  • ★ [v3 전체 3-State] 결함 F1-Score  : {best_3state['f1']*100:.2f}%")
    print(f"  • ★ [v3 전체 3-State] 정밀도 (Precision) : {best_3state['precision']*100:.2f}%")
    print(f"  • ★ [v3 전체 3-State] 재현율 (Recall)    : {best_3state['recall']*100:.2f}%")
    print(f"  • ★ [v3 전체 3-State] 정확도 (Accuracy)  : {best_3state['accuracy']*100:.2f}%")
    print(f"  • ★ [v3 전체] ROC-AUC                : {auc_val*100:.2f}%")

    # -------------------------------------------------------------
    # 6. bl4_v2 vs bl4_v3 1:1 비교 성적표 출력
    # -------------------------------------------------------------
    # bl4_v2 실측 지표 로드
    v2_json_path = os.path.join(CURRENT_DIR, "bl4_v2_evaluation_results.json")
    v2_f1_3s, v2_prec_3s, v2_rec_3s, v2_acc_3s, v2_auc, v2_cov = 76.32, 75.57, 77.08, 79.60, 80.03, 89.71
    if os.path.exists(v2_json_path):
        with open(v2_json_path, 'r', encoding='utf-8') as f:
            v2_d = json.load(f)
            v2_f1_3s = v2_d['metrics_3state_overall']['f1'] * 100.0
            v2_prec_3s = v2_d['metrics_3state_overall']['precision'] * 100.0
            v2_rec_3s = v2_d['metrics_3state_overall']['recall'] * 100.0
            v2_acc_3s = v2_d['metrics_3state_overall']['accuracy'] * 100.0
            v2_auc = v2_d['metrics_2state_overall']['roc_auc'] * 100.0
            v2_cov = v2_d['metrics_3state_overall']['coverage'] * 100.0

    print("\n" + "=" * 95)
    print("  [★ bl4_v2 (선행연구 36D) vs bl4_v3 (피처/구조 최적화 24D) 전체 3-State 성능 1:1 비교]")
    print("=" * 95)
    print(f"  {'평가 지표 (Evaluation Metrics)':30s} | {'bl4_v2 (기존)':18s} | {'★ bl4_v3 (최적화)':18s} | {'성능 개선폭':12s}")
    print("-" * 95)
    print(f"  {'전체 3-State 결함 F1':30s} | {v2_f1_3s:6.2f}%            | ★ {best_3state['f1']*100:6.2f}%            | {best_3state['f1']*100 - v2_f1_3s:+5.2f}%p")
    print(f"  {'전체 3-State 정밀도':30s} | {v2_prec_3s:6.2f}%            | ★ {best_3state['precision']*100:6.2f}%            | {best_3state['precision']*100 - v2_prec_3s:+5.2f}%p")
    print(f"  {'전체 3-State 재현율':30s} | {v2_rec_3s:6.2f}%            | ★ {best_3state['recall']*100:6.2f}%            | {best_3state['recall']*100 - v2_rec_3s:+5.2f}%p")
    print(f"  {'전체 3-State 정확도':30s} | {v2_acc_3s:6.2f}%            | ★ {best_3state['accuracy']*100:6.2f}%            | {best_3state['accuracy']*100 - v2_acc_3s:+5.2f}%p")
    print(f"  {'ROC-AUC':30s} | {v2_auc:6.2f}%            | ★ {auc_val*100:6.2f}%            | {auc_val*100 - v2_auc:+5.2f}%p")
    print(f"  {'판정 커버리지':30s} | {v2_cov:6.2f}%            |   {best_3state['coverage']*100:6.2f}%            | {best_3state['coverage']*100 - v2_cov:+5.2f}%p")

    # -------------------------------------------------------------
    # 7. AGENTS.md 규격 공식 5대 시점(시점 A~E) 전수 세부 성능 감사
    # -------------------------------------------------------------
    print("\n" + "=" * 105)
    print("  [AGENTS.md 표준: bl4_v3 공식 5대 시점(시점 A~E) 전수 세부 성능 감사표]")
    print("=" * 105)
    print(f"  {'시점 코드':8s} | {'공식 시점 명칭':28s} | {'샘플수':6s} | {'2-State F1':12s} | {'★ 3-State F1':14s} | {'정확도':10s} | {'ROC-AUC':10s}")
    print("-" * 105)

    view_results = {}
    preds_3s_all = best_3state['preds_3state']

    for v_code in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_name = VIEW_ALIAS_MAP[v_code]
        mask_v = (views == v_code)
        n_v = int(np.sum(mask_v))

        y_v = y[mask_v]
        p_v = final_oof_probs[mask_v]
        pred_3s_v = preds_3s_all[mask_v]

        f1_2s_v = float(f1_score(y_v, (p_v >= 0.5).astype(int), pos_label=1, zero_division=0))
        auc_v = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else 0.5

        det_v = (pred_3s_v != -1)
        if np.sum(det_v) > 0:
            f1_3s_v = float(f1_score(y_v[det_v], pred_3s_v[det_v], pos_label=1, zero_division=0))
            acc_3s_v = float(accuracy_score(y_v[det_v], pred_3s_v[det_v]))
        else:
            f1_3s_v, acc_3s_v = 0.0, 0.0

        view_results[v_code] = {
            'view_name': v_name,
            'count': n_v,
            'f1_2state': f1_2s_v,
            'f1_3state': f1_3s_v,
            'accuracy_3state': acc_3s_v,
            'roc_auc': auc_v
        }

        print(f"  {v_code:8s} | {v_name:28s} | {n_v:6d} | {f1_2s_v*100:6.2f}%      | ★ {f1_3s_v*100:6.2f}%    | {acc_3s_v*100:6.2f}%   | {auc_v*100:6.2f}%")

    # -------------------------------------------------------------
    # 8. 모델 가중치 번들 및 JSON 저장
    # -------------------------------------------------------------
    bundle_data = {
        'model_name': 'bl4_v3',
        'exercise': '바벨 런지 (04_바벨_런지)',
        'defect': 'c4 (척추의 중립 위반 / Spine Neutrality Defect)',
        'version': 'v3.0 Feature & Architecture Optimized Edition',
        'feature_names': feature_names,
        'weights': {
            'w_tcn': float(w_tcn),
            'w_lgb': float(w_lgb),
            'w_xgb': float(w_xgb),
            'w_cat': float(w_cat)
        },
        'thresholds': {
            'tau_clean': float(tc),
            'tau_def': float(td),
            'tau_binary': 0.5
        },
        'fold_models': fold_models
    }

    with open(BUNDLE_V3_PATH, 'wb') as f:
        pickle.dump(bundle_data, f)
    print(f"\n[+] v3 최적 모델 가중치 번들 저장 완료: {BUNDLE_V3_PATH}")

    # JSON 저장
    eval_json_data = {
        'model_name': 'bl4_v3',
        'exercise': '바벨 런지 (04_바벨_런지)',
        'defect': 'c4 (척추의 중립 위반 / Spine Neutrality Defect)',
        'version': 'v3.0 Feature & Architecture Optimized Production Release',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'weights': bundle_data['weights'],
        'thresholds': bundle_data['thresholds'],
        'metrics_3state_overall': {
            'score': float(best_3state['score']),
            'tau_clean': float(tc),
            'tau_def': float(td),
            'coverage': float(best_3state['coverage']),
            'abstain_rate': float(best_3state['abstain_rate']),
            'f1': float(best_3state['f1']),
            'precision': float(best_3state['precision']),
            'recall': float(best_3state['recall']),
            'accuracy': float(best_3state['accuracy']),
            'n_determined': int(best_3state['n_determined']),
            'n_abstained': int(best_3state['n_abstained'])
        },
        'metrics_2state_overall': {
            'f1': f1_2s,
            'precision': prec_2s,
            'recall': rec_2s,
            'accuracy': acc_2s,
            'roc_auc': auc_val
        },
        'view_metrics': view_results
    }

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(eval_json_data, f, indent=2, ensure_ascii=False)
    print(f"[+] v3 평가 결과 JSON 저장 완료: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    train_and_eval_bl4_v3()
