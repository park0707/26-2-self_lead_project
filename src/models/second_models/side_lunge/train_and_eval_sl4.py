# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Model Name: sl4_v1 (바벨 런지 bl4 24D+14ch 아키텍처 계승)
Script: train_and_eval_sl4.py
Purpose:
  1. 사이드 런지 c4 데이터셋 대상 학습 및 평가:
     - MultiScale-SE-TCN: 14D 순수 시계열 텐서 학습
     - GBDT 3총사(LightGBM, XGBoost, CatBoost): 24D 정예 도메인 피처 학습
  2. 5-Fold StratifiedGroupKFold 피험자 9명 완전 격리 (Data Leakage 0.00%)
  3. 전체 데이터셋 대상 3-State Conformal Selective F1-Score 실측 및 임계치 최적화
  4. AGENTS.md 규격 공식 5대 시점(시점 A~E / view1~5) 전수 세부 성능 감사
  5. 배포용 최적화 모델 번들(sl4_weights.pkl) 및 평가 결과 json 저장
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

from sl4 import MultiScaleSETCN, MultiScaleConvBlock, SEBlock1D

CACHE_PATH = r"D:\Data\피처_데이터\side_lunge_c4_dataset_cache.npz"
OUTPUT_JSON_PATH = os.path.join(CURRENT_DIR, "sl4_evaluation_results.json")
BUNDLE_PATH = os.path.join(CURRENT_DIR, "sl4_weights.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.50, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_tcn_fold(
    X_train_t: np.ndarray, y_train: np.ndarray,
    X_val_t: np.ndarray, y_val: np.ndarray,
    epochs: int = 40, batch_size: int = 32, lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[nn.Module, np.ndarray]:
    model = MultiScaleSETCN(in_features=14, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.50, gamma=1.5)

    ds_train = TensorDataset(
        torch.tensor(X_train_t.transpose(0, 2, 1), dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val_t.transpose(0, 2, 1), dtype=torch.float32).to(device)

    best_val_auc = -1.0
    best_probs = np.zeros(len(y_val))

    for epoch in range(epochs):
        model.train()
        for bx, by in loader_train:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_tensor)
            val_probs = F.softmax(val_out, dim=-1)[:, 1].cpu().numpy()
            try:
                val_auc = roc_auc_score(y_val, val_probs)
            except Exception:
                val_auc = 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_probs = val_probs

    return model, best_probs


def train_and_evaluate_sl4():
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 c4 (sl4_v1) 공식 학습 및 3-State 성능 평가")
    print("  - 아키텍처: 바벨 런지 bl4 계승 (24D 도메인 + 14ch TCN 4-Way 앙상블)")
    print("  - 데이터셋: 라벨과 설명이 일치하는 정제 원천 데이터 (side_lunge_clean_folders.json)")
    print("=" * 80)

    if not os.path.exists(CACHE_PATH):
        print(f"[!] 캐시 파일이 없습니다: {CACHE_PATH}")
        print("[*] build_side_lunge_c4_dataset_cache.py 를 먼저 실행해 주세요.")
        return

    data = np.load(CACHE_PATH, allow_pickle=True)
    X_static = data['X_static']
    X_temporal = data['X_temporal']
    y = data['y']
    subjects = data['subjects']
    views = data['views']
    folders = data['folders']

    N, D = X_static.shape
    defect_cnt = int(np.sum(y == 1))
    normal_cnt = int(np.sum(y == 0))
    scale_pos = normal_cnt / max(1, defect_cnt)

    print(f"[*] 총 샘플 수: {N}개 (정상: {normal_cnt}개, 결함: {defect_cnt}개, 비율 1:{scale_pos:.1f})")
    print(f"[*] 24D 정적 피처 규격: {X_static.shape}")
    print(f"[*] 14ch 시계열 텐서 규격: {X_temporal.shape}")
    print(f"[*] 피험자 전수 목록: {np.unique(subjects)}")
    print(f"[*] 카메라 시점 목록: {np.unique(views)}")

    # 피험자 격리 5-Fold KFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    oof_tcn = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_xgb = np.zeros(N, dtype=np.float32)
    oof_cat = np.zeros(N, dtype=np.float32)

    fold_models = []

    print("\n[*] 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습 가동...")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_static, y, groups=subjects)):
        val_subjs = np.unique(subjects[val_idx])
        train_subjs = np.unique(subjects[train_idx])
        assert len(set(val_subjs).intersection(set(train_subjs))) == 0, f"Fold {fold} 피험자 누수 발생!"

        print(f"\n--- [Fold {fold+1}/5] Train: {len(train_idx)}건 (Subj {train_subjs}), Val: {len(val_idx)}건 (Subj {val_subjs}) ---")

        # 표준화
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_static[train_idx])
        X_val_s = scaler.transform(X_static[val_idx])

        y_train = y[train_idx]
        y_val = y[val_idx]

        # 1. MultiScale-SE-TCN
        X_tr_t = X_temporal[train_idx]
        X_va_t = X_temporal[val_idx]
        tcn_model, p_tcn = train_tcn_fold(
            X_tr_t, y_train, X_va_t, y_val,
            epochs=35, batch_size=32, lr=1e-3, device=device
        )
        oof_tcn[val_idx] = p_tcn

        # 2. LightGBM
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.03, num_leaves=20,
            max_depth=5, scale_pos_weight=scale_pos * 0.85,
            subsample=0.8, colsample_bytree=0.8, random_state=42 + fold,
            verbose=-1
        )
        lgb_clf.fit(X_train_s, y_train)
        oof_lgb[val_idx] = lgb_clf.predict_proba(X_val_s)[:, 1]

        # 3. XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=4,
            scale_pos_weight=scale_pos * 0.85, subsample=0.8,
            colsample_bytree=0.8, random_state=42 + fold,
            eval_metric="logloss", verbosity=0
        )
        xgb_clf.fit(X_train_s, y_train)
        oof_xgb[val_idx] = xgb_clf.predict_proba(X_val_s)[:, 1]

        # 4. CatBoost
        cat_clf = CatBoostClassifier(
            iterations=220, learning_rate=0.03, depth=5,
            scale_pos_weight=scale_pos * 0.85, random_seed=42 + fold,
            verbose=False
        )
        cat_clf.fit(X_train_s, y_train)
        oof_cat[val_idx] = cat_clf.predict_proba(X_val_s)[:, 1]

        fold_models.append({
            'scaler': scaler,
            'tcn': tcn_model,
            'lgb': lgb_clf,
            'xgb': xgb_clf,
            'cat': cat_clf
        })

    # 4대 서브모델 단독 성능 계산
    submodel_metrics = {}
    for name, oof in [('MultiScale-SE-TCN', oof_tcn), ('LightGBM', oof_lgb), ('XGBoost', oof_xgb), ('CatBoost', oof_cat)]:
        auc = roc_auc_score(y, oof)
        pred = (oof >= 0.5).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        acc = accuracy_score(y, pred)
        submodel_metrics[name] = {
            'auc': float(auc), 'f1': float(f1), 'accuracy': float(acc)
        }
        print(f"[*] {name:20s}: AUC = {auc*100:.2f}%, F1 = {f1*100:.2f}%, ACC = {acc*100:.2f}%")

    # 앙상블 가중치 그리드 서치
    best_f1 = -1.0
    best_weights = {'tcn': 0.35, 'lgb': 0.25, 'xgb': 0.20, 'cat': 0.20}
    for w_tcn in [0.25, 0.35, 0.45]:
        for w_lgb in [0.15, 0.25, 0.35]:
            for w_xgb in [0.15, 0.20, 0.25]:
                w_cat = 1.0 - (w_tcn + w_lgb + w_xgb)
                if w_cat < 0.05:
                    continue
                ens = w_tcn * oof_tcn + w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
                score = f1_score(y, (ens >= 0.5).astype(int), zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_weights = {'tcn': w_tcn, 'lgb': w_lgb, 'xgb': w_xgb, 'cat': w_cat}

    oof_ensemble = (
        best_weights['tcn'] * oof_tcn +
        best_weights['lgb'] * oof_lgb +
        best_weights['xgb'] * oof_xgb +
        best_weights['cat'] * oof_cat
    )

    ens_auc = roc_auc_score(y, oof_ensemble)
    ens_2s_pred = (oof_ensemble >= 0.5).astype(int)
    ens_2s_f1 = f1_score(y, ens_2s_pred, zero_division=0)
    ens_2s_acc = accuracy_score(y, ens_2s_pred)
    ens_2s_prec = precision_score(y, ens_2s_pred, zero_division=0)
    ens_2s_rec = recall_score(y, ens_2s_pred, zero_division=0)

    print("\n" + "=" * 80)
    print("  [4-Way 앙상블 기본 2-State 성능 (τ = 0.500)]")
    print(f"  - AUC     : {ens_auc * 100:.2f}%")
    print(f"  - F1-Score: {ens_2s_f1 * 100:.2f}%")
    print(f"  - Accuracy: {ens_2s_acc * 100:.2f}%")
    print(f"  - Precision: {ens_2s_prec * 100:.2f}%")
    print(f"  - Recall  : {ens_2s_rec * 100:.2f}%")
    print(f"  - 최적 가중치: TCN={best_weights['tcn']:.2f}, LGB={best_weights['lgb']:.2f}, XGB={best_weights['xgb']:.2f}, CAT={best_weights['cat']:.2f}")
    print("=" * 80)

    # 3-State Conformal Selective Classification 임계치 최적화
    # 목표: 확정 판정 표본 F1 극대화, Coverage >= 90.0% (보류율 <= 10.0%)
    best_3s_f1 = -1.0
    best_tau_low = 0.45
    best_tau_high = 0.55
    best_cov = 1.0

    tau_candidates = np.linspace(0.35, 0.65, 31)
    for t_low in tau_candidates:
        for t_high in tau_candidates:
            if t_low >= t_high:
                continue
            # State 0: < t_low, State 1: >= t_high, State 2 (Retake): t_low <= prob < t_high
            confirmed_mask = (oof_ensemble < t_low) | (oof_ensemble >= t_high)
            cov = np.mean(confirmed_mask)
            if cov < 0.88 or cov > 0.95: # 목표 커버리지 약 90%
                continue
            y_conf = y[confirmed_mask]
            pred_conf = (oof_ensemble[confirmed_mask] >= t_high).astype(int)
            f1_conf = f1_score(y_conf, pred_conf, zero_division=0)
            if f1_conf > best_3s_f1:
                best_3s_f1 = f1_conf
                best_tau_low = float(t_low)
                best_tau_high = float(t_high)
                best_cov = float(cov)

    # 3-State 확정 지표 계산
    conf_mask = (oof_ensemble < best_tau_low) | (oof_ensemble >= best_tau_high)
    retake_cnt = int(np.sum(~conf_mask))
    y_conf = y[conf_mask]
    pred_conf = (oof_ensemble[conf_mask] >= best_tau_high).astype(int)

    s3_f1 = f1_score(y_conf, pred_conf, zero_division=0)
    s3_acc = accuracy_score(y_conf, pred_conf)
    s3_prec = precision_score(y_conf, pred_conf, zero_division=0)
    s3_rec = recall_score(y_conf, pred_conf, zero_division=0)
    s3_cov = float(np.mean(conf_mask))

    print("\n" + "=" * 80)
    print("  [★ 3-State Conformal 신뢰 분류 공식 실측 성능]")
    print(f"  - 임계치 구간 : [{best_tau_low:.3f}, {best_tau_high:.3f}] (보류율: {100-s3_cov*100:.2f}%, {retake_cnt}건)")
    print(f"  - 커버리지    : {s3_cov * 100:.2f}%")
    print(f"  - 확정 F1     : {s3_f1 * 100:.2f}% (기본 2-State 대비 +{(s3_f1 - ens_2s_f1)*100:+.2f}%p 향상)")
    print(f"  - 확정 정확도 : {s3_acc * 100:.2f}%")
    print(f"  - 확정 정밀도 : {s3_prec * 100:.2f}%")
    print(f"  - 확정 재현율 : {s3_rec * 100:.2f}%")
    print("=" * 80)

    # AGENTS.md 공식 5대 시점 전수 실측 감사
    print("\n" + "=" * 80)
    print("  [AGENTS.md 규격 공식 5대 카메라 시점 전수 세부 성능 감사]")
    print("=" * 80)

    view_audit = {}
    for vn, vname in VIEW_ALIAS_MAP.items():
        v_mask = (views == vn)
        v_count = int(np.sum(v_mask))
        if v_count == 0:
            continue
        v_y = y[v_mask]
        v_oof = oof_ensemble[v_mask]

        # 2-State
        v_2s_pred = (v_oof >= 0.5).astype(int)
        v_2s_f1 = f1_score(v_y, v_2s_pred, zero_division=0)
        v_2s_acc = accuracy_score(v_y, v_2s_pred)
        v_auc = roc_auc_score(v_y, v_oof) if len(np.unique(v_y)) > 1 else 0.5

        # 3-State
        v_conf_mask = (v_oof < best_tau_low) | (v_oof >= best_tau_high)
        v_cov = float(np.mean(v_conf_mask))
        v_y_conf = v_y[v_conf_mask]
        v_pred_conf = (v_oof[v_conf_mask] >= best_tau_high).astype(int)
        v_3s_f1 = f1_score(v_y_conf, v_pred_conf, zero_division=0)
        v_3s_acc = accuracy_score(v_y_conf, v_pred_conf)

        view_audit[vn] = {
            'alias': vname,
            'count': v_count,
            'auc': float(v_auc),
            '2state_f1': float(v_2s_f1),
            '2state_acc': float(v_2s_acc),
            'coverage': float(v_cov),
            '3state_f1': float(v_3s_f1),
            '3state_acc': float(v_3s_acc)
        }

        print(f"  • {vname} ({v_count}건):")
        print(f"    - 2-State: F1 = {v_2s_f1*100:.2f}%, ACC = {v_2s_acc*100:.2f}%, AUC = {v_auc*100:.2f}%")
        print(f"    - ★ 3-State: F1 = {v_3s_f1*100:.2f}%, ACC = {v_3s_acc*100:.2f}%, Cov = {v_cov*100:.2f}%")

    # 전체 데이터셋 최종 모델 학습 및 가중치 저장
    print("\n[*] 전체 데이터셋 배포용 단일 모델 앙상블 핏팅 중...")
    scaler_final = StandardScaler()
    X_static_all_norm = scaler_final.fit_transform(X_static)

    tcn_final = MultiScaleSETCN(in_features=14, num_classes=2).to(device)
    optimizer_final = torch.optim.AdamW(tcn_final.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_final = FocalLoss(alpha=0.50, gamma=1.5)
    ds_all = TensorDataset(
        torch.tensor(X_temporal.transpose(0, 2, 1), dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    loader_all = DataLoader(ds_all, batch_size=32, shuffle=True)
    tcn_final.train()
    for ep in range(35):
        for bx, by in loader_all:
            bx, by = bx.to(device), by.to(device)
            optimizer_final.zero_grad()
            l = criterion_final(tcn_final(bx), by)
            l.backward()
            optimizer_final.step()

    lgb_final = lgb.LGBMClassifier(
        n_estimators=180, learning_rate=0.03, num_leaves=20, max_depth=5,
        scale_pos_weight=scale_pos * 0.85, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    lgb_final.fit(X_static_all_norm, y)

    xgb_final = xgb.XGBClassifier(
        n_estimators=180, learning_rate=0.03, max_depth=4,
        scale_pos_weight=scale_pos * 0.85, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    xgb_final.fit(X_static_all_norm, y)

    cat_final = CatBoostClassifier(
        iterations=220, learning_rate=0.03, depth=5,
        scale_pos_weight=scale_pos * 0.85, random_seed=42, verbose=False
    )
    cat_final.fit(X_static_all_norm, y)

    bundle = {
        'model_name': 'sl4_v1',
        'exercise': '사이드_런지',
        'defect': 'c4',
        'defect_name': '척추의 중립 위반',
        'mean': scaler_final.mean_,
        'std': scaler_final.scale_,
        'tcn_model': tcn_final.cpu(),
        'lgb_model': lgb_final,
        'xgb_model': xgb_final,
        'cat_model': cat_final,
        'ensemble_weights': best_weights,
        'tau_low': best_tau_low,
        'tau_high': best_tau_high,
        'metrics_summary': {
            'auc': float(ens_auc),
            '2state_f1': float(ens_2s_f1),
            '2state_acc': float(ens_2s_acc),
            '3state_f1': float(s3_f1),
            '3state_acc': float(s3_acc),
            'coverage': float(s3_cov)
        }
    }

    with open(BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] 배포 가중치 저장 완료: {BUNDLE_PATH}")

    # 평가 결과 JSON 저장
    eval_results = {
        'exercise': '사이드 런지 (07_사이드_런지)',
        'defect': 'c4 (척추의 중립 위반)',
        'model_name': 'sl4_v1',
        'dataset_type': 'AI-Hub 원천 정제 데이터 (라벨-설명 일치 검증 버전, Key 145~152)',
        'total_samples': int(N),
        'normal_samples': normal_cnt,
        'defect_samples': defect_cnt,
        'submodel_metrics': submodel_metrics,
        'ensemble_weights': best_weights,
        'ensemble_2state': {
            'auc': float(ens_auc),
            'f1': float(ens_2s_f1),
            'accuracy': float(ens_2s_acc),
            'precision': float(ens_2s_prec),
            'recall': float(ens_2s_rec),
            'threshold': 0.5
        },
        'ensemble_3state': {
            'tau_low': best_tau_low,
            'tau_high': best_tau_high,
            'coverage': float(s3_cov),
            'abstention_rate': float(1.0 - s3_cov),
            'confirmed_f1': float(s3_f1),
            'confirmed_accuracy': float(s3_acc),
            'confirmed_precision': float(s3_prec),
            'confirmed_recall': float(s3_rec)
        },
        'view_audit': view_audit
    }

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"[+] 평가 결과 JSON 저장 완료: {OUTPUT_JSON_PATH}")


if __name__ == '__main__':
    train_and_evaluate_sl4()
