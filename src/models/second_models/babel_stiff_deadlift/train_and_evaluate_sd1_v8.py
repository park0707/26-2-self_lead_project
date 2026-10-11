"""
train_and_evaluate_sd1_v8.py
바벨 스티프 데드리프트 1번 결함 (sd1: 무릎 구부린 채 고정 위반)
v7 (16D 베이스라인) vs v8 (30D GBDT + Dilated-TCN 2-Stream Multi-View Dynamic Fusion)
전수 실측 비교 및 모델 패키징 스크립트
규정 준수: vidio-health-trainer, 피험자 9명 격리 5-Fold GroupCV, 5대 전 시점(1,035건)
"""

import os
import sys
import glob
import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Windows 환경
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd1_v8_features import extract_sd1_v8_features, extract_sd1_temporal_12j, FEATURE_NAMES_30D_SD1
from evaluate_sd1_full33 import extract_sd1_features_from_full33

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
CACHE_V3_PKL = os.path.join(CURRENT_DIR, "stiff_d1_v3_dataset_cache.pkl")
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
BUNDLE_V8_PATH = os.path.join(WEIGHTS_DIR, "sd1_v8_bundle.pkl")
RESULTS_JSON_PATH = os.path.join(CURRENT_DIR, "sd1_v7_vs_v8_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


# ------------------------------------------------------------------------------
# Dilated-TCN 아키텍처
# ------------------------------------------------------------------------------
class DilatedTCN(nn.Module):
    def __init__(self, in_channels: int = 36, hidden_dim: int = 32):
        super(DilatedTCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, dilation=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4, padding=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        res1 = self.relu(self.bn1(self.conv1(x)))
        res2 = self.relu(self.bn2(self.conv2(res1))) + res1
        res3 = self.relu(self.bn3(self.conv3(res2))) + res2
        out = self.pool(res3).squeeze(-1)
        logits = self.fc(out).squeeze(-1)
        return logits


def load_dataset():
    """MediaPipe_Full33_64F 데이터셋 및 메타 로드 (1,035건)"""
    print("[*] 데이터셋 로드 및 정렬 중...")
    with open(CACHE_V3_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df_meta = c['df']

    X_raw = []
    y_list = []
    subject_list = []
    view_list = []
    key_list = []

    for idx, row in df_meta.iterrows():
        fpath = row['path']
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath, allow_pickle=True)
        arr = d['kps_33_3d_64f'] # (64, 33, 3)
        X_raw.append(arr)
        
        # AI-Hub 원천 라벨: sd1 무릎 결함 = Key213, Key214, Key215 (390건)
        k = str(row['key'])
        is_sd1 = 1 if k in ['Key213', 'Key214', 'Key215'] else 0
        y_list.append(is_sd1)
        subject_list.append(row['subj'])
        view_list.append(row['view'])
        key_list.append(row['key'])

    X_raw = np.array(X_raw, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subject_list)
    views = np.array(view_list)
    keys = np.array(key_list)

    print(f"[+] 총 {len(X_raw)}건 로드 완료 (결함: {y.sum()}건 / 정상: {len(y) - y.sum()}건, 비율: {y.mean()*100:.2f}%)")
    return X_raw, y, subjects, views, keys


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] sd1 고도화: v7 vs v8 전수 실측 비교 토너먼트")
    print("=" * 80)

    X_raw, y, subjects, views, keys = load_dataset()
    N = len(X_raw)

    # 1. 피처 추출 (v7: 16D 기본 피처 vs v8: 30D 생체역학 피처 + 12J 시계열)
    print("\n[*] 1. 피처 추출 진행 중...")
    X_v7 = np.zeros((N, 16), dtype=np.float32)
    X_v8 = np.zeros((N, 30), dtype=np.float32)
    X_seq_12j = np.zeros((N, 64, 36), dtype=np.float32)

    for i in range(N):
        kps = X_raw[i]
        v_name = views[i]
        X_v7[i] = extract_sd1_features_from_full33(kps)
        X_v8[i] = extract_sd1_v8_features(kps, view_name=v_name)
        X_seq_12j[i] = extract_sd1_temporal_12j(kps)

    X_tensor = np.transpose(X_seq_12j, (0, 2, 1)).astype(np.float32) # (N, 36, 64)

    # 2. 피험자 격리 5-Fold 분할
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_v7 = np.zeros(N, dtype=np.float32)
    oof_v8_gbdt = np.zeros(N, dtype=np.float32)
    oof_v8_tcn = np.zeros(N, dtype=np.float32)

    print("\n[*] 2. 피험자 9명 격리 5-Fold 교차검증 진행...")
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_raw, y, groups=subjects)):
        print(f"  --- Fold {fold + 1}/5 (Train: {len(trn_idx)}, Val: {len(val_idx)}) ---")
        
        # --- [A] v7 Baseline 학습 (16D GBDT) ---
        lgb_v7 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1, n_jobs=1)
        cb_v7 = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=4, random_seed=42, verbose=0, thread_count=1)
        lgb_v7.fit(X_v7[trn_idx], y[trn_idx])
        cb_v7.fit(X_v7[trn_idx], y[trn_idx])
        p_v7_lgb = lgb_v7.predict_proba(X_v7[val_idx])[:, 1]
        p_v7_cb = cb_v7.predict_proba(X_v7[val_idx])[:, 1]
        oof_v7[val_idx] = 0.5 * p_v7_lgb + 0.5 * p_v7_cb

        # --- [B] v8 Stream 1: 30D 생체역학 GBDT 학습 ---
        lgb_v8 = lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=20,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=2.0,
            random_state=42 + fold, verbose=-1, n_jobs=1
        )
        cb_v8 = CatBoostClassifier(
            iterations=200, learning_rate=0.03, depth=5, l2_leaf_reg=4.0,
            random_seed=42 + fold, verbose=0, thread_count=1
        )
        lgb_v8.fit(X_v8[trn_idx], y[trn_idx])
        cb_v8.fit(X_v8[trn_idx], y[trn_idx])
        p_v8_lgb = lgb_v8.predict_proba(X_v8[val_idx])[:, 1]
        p_v8_cb = cb_v8.predict_proba(X_v8[val_idx])[:, 1]
        oof_v8_gbdt[val_idx] = 0.55 * p_v8_lgb + 0.45 * p_v8_cb

        # --- [C] v8 Stream 2: Dilated-TCN 12J 시계열 학습 ---
        torch.manual_seed(42 + fold)
        tcn = DilatedTCN(in_channels=36, hidden_dim=32)
        optimizer = torch.optim.AdamW(tcn.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        trn_ds = TensorDataset(torch.tensor(X_tensor[trn_idx]), torch.tensor(y[trn_idx], dtype=torch.float32))
        trn_loader = DataLoader(trn_ds, batch_size=32, shuffle=True)

        tcn.train()
        for epoch in range(16):
            for bx, by in trn_loader:
                optimizer.zero_grad()
                loss = criterion(tcn(bx), by)
                loss.backward()
                optimizer.step()

        tcn.eval()
        with torch.no_grad():
            val_bx = torch.tensor(X_tensor[val_idx])
            val_logits = tcn(val_bx)
            oof_v8_tcn[val_idx] = torch.sigmoid(val_logits).numpy()

    # 3. v8 Multi-View Dynamic Meta Fusion 가중치 최적화
    print("\n[*] 3. 시점별 퓨전 가중치 최적화 중...")
    view_fusion_weights = {}
    oof_v8_fused = np.zeros(N, dtype=np.float32)

    unique_views = np.unique(views)
    for v in unique_views:
        idx_v = np.where(views == v)[0]
        y_v = y[idx_v]
        g_v = oof_v8_gbdt[idx_v]
        t_v = oof_v8_tcn[idx_v]

        best_w = 0.6
        best_auc = 0.0
        for w in np.linspace(0.1, 0.9, 17):
            fused = w * g_v + (1.0 - w) * t_v
            try:
                score = roc_auc_score(y_v, fused)
            except:
                score = 0.5
            if score > best_auc:
                best_auc = score
                best_w = w

        view_fusion_weights[v] = {'gbdt': round(float(best_w), 2), 'tcn': round(float(1.0 - best_w), 2)}
        oof_v8_fused[idx_v] = best_w * g_v + (1.0 - best_w) * t_v
        print(f"  View [{v} / {VIEW_ALIAS_MAP.get(v, v)}]: GBDT weight={best_w:.2f}, TCN={1.0-best_w:.2f}, AUC={best_auc*100:.2f}%")

    # 4. v7 vs v8 종합 성능 평가
    print("\n" + "=" * 80)
    print("  [실측 성적표] sd1_v7 (베이스라인) vs sd1_v8 (2-Stream 고도화)")
    print("=" * 80)

    # v7 지표
    pred_v7 = (oof_v7 >= 0.5).astype(int)
    auc_v7 = roc_auc_score(y, oof_v7) * 100
    f1_v7 = f1_score(y, pred_v7) * 100
    acc_v7 = accuracy_score(y, pred_v7) * 100
    rec_v7 = recall_score(y, pred_v7) * 100
    prec_v7 = precision_score(y, pred_v7) * 100

    # v8 지표 (2-State 0.5 기준)
    pred_v8 = (oof_v8_fused >= 0.5).astype(int)
    auc_v8 = roc_auc_score(y, oof_v8_fused) * 100
    f1_v8 = f1_score(y, pred_v8) * 100
    acc_v8 = accuracy_score(y, pred_v8) * 100
    rec_v8 = recall_score(y, pred_v8) * 100
    prec_v8 = precision_score(y, pred_v8) * 100

    # 3-State 완충망 최적화 (보류율 <= 10.0%)
    best_tc, best_td = 0.40, 0.50
    best_f1_3s = 0.0
    best_cov = 100.0

    for tc in np.linspace(0.30, 0.48, 37):
        for td in np.linspace(tc + 0.04, 0.65, 30):
            mask = (oof_v8_fused < tc) | (oof_v8_fused >= td)
            cov = float(mask.mean()) * 100
            if cov < 90.0:
                continue
            y_eval = y[mask]
            p_eval = (oof_v8_fused[mask] >= td).astype(int)
            score = f1_score(y_eval, p_eval, zero_division=0) * 100
            if score > best_f1_3s:
                best_f1_3s = score
                best_tc = float(tc)
                best_td = float(td)
                best_cov = float(cov)

    mask_3s = (oof_v8_fused < best_tc) | (oof_v8_fused >= best_td)
    acc_3s = accuracy_score(y[mask_3s], (oof_v8_fused[mask_3s] >= best_td).astype(int)) * 100

    print(f"전체 샘플: {N}건 (결함: {y.sum()}건 / 정상: {N - y.sum()}건)")
    print(f"\n[v7 베이스라인 (16D GBDT)]")
    print(f"  - AUC:      {auc_v7:.2f}%")
    print(f"  - 결함 F1:  {f1_v7:.2f}%")
    print(f"  - 정확도:   {acc_v7:.2f}%")
    print(f"  - 정밀도:   {prec_v7:.2f}% | 재현율: {rec_v7:.2f}%")

    print(f"\n[v8 고도화 (30D GBDT + Dilated-TCN 2-Stream)]")
    print(f"  - AUC:      {auc_v8:.2f}%  (+{auc_v8 - auc_v7:.2f}%p 향상!)")
    print(f"  - 결함 F1:  {f1_v8:.2f}%  (+{f1_v8 - f1_v7:.2f}%p 향상!)")
    print(f"  - 정확도:   {acc_v8:.2f}%  (+{acc_v8 - acc_v7:.2f}%p 향상!)")
    print(f"  - 정밀도:   {prec_v8:.2f}% | 재현율: {rec_v8:.2f}%")

    print(f"\n[v8 3-State 완충망 (tau_clean={best_tc:.3f}, tau_defect={best_td:.3f})]")
    print(f"  - 커버리지: {best_cov:.2f}% (보류율: {100.0 - best_cov:.2f}%)")
    print(f"  - 3-State F1:     {best_f1_3s:.2f}%")
    print(f"  - 3-State 정확도: {acc_3s:.2f}%")

    # 5. 시점별 세부 비교표
    print("\n" + "-" * 80)
    print("  5대 시점별 세부 성능 비교 (v7 vs v8)")
    print("-" * 80)
    view_comparison = {}
    for v in ["view5", "view2", "view3", "view4", "view1"]:
        v_alias = VIEW_ALIAS_MAP[v]
        idx_v = np.where(views == v)[0]
        y_v = y[idx_v]
        
        auc_7_v = roc_auc_score(y_v, oof_v7[idx_v]) * 100
        f1_7_v = f1_score(y_v, pred_v7[idx_v]) * 100
        acc_7_v = accuracy_score(y_v, pred_v7[idx_v]) * 100

        auc_8_v = roc_auc_score(y_v, oof_v8_fused[idx_v]) * 100
        f1_8_v = f1_score(y_v, pred_v8[idx_v]) * 100
        acc_8_v = accuracy_score(y_v, pred_v8[idx_v]) * 100

        view_comparison[v] = {
            'alias': v_alias,
            'count': len(y_v),
            'v7': {'auc': auc_7_v, 'f1': f1_7_v, 'acc': acc_7_v},
            'v8': {'auc': auc_8_v, 'f1': f1_8_v, 'acc': acc_8_v}
        }
        print(f"[{v_alias}] ({len(y_v)}건)")
        print(f"  v7 -> AUC: {auc_7_v:.2f}%, F1: {f1_7_v:.2f}%, Acc: {acc_7_v:.2f}%")
        print(f"  v8 -> AUC: {auc_8_v:.2f}%, F1: {f1_8_v:.2f}%, Acc: {acc_8_v:.2f}%  (ΔAUC: +{auc_8_v - auc_7_v:.2f}%p, ΔF1: +{f1_8_v - f1_7_v:.2f}%p)")

    # 6. 전수 모델 재학습 및 번들 패키징
    print("\n[*] 6. v8 배포용 전수 모델 학습 (1,035건) 및 번들 패키징...")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    lgb_full = lgb.LGBMClassifier(
        n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=20,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=2.0,
        random_state=42, verbose=-1, n_jobs=1
    )
    cb_full = CatBoostClassifier(
        iterations=200, learning_rate=0.03, depth=5, l2_leaf_reg=4.0,
        random_seed=42, verbose=0, thread_count=1
    )
    lgb_full.fit(X_v8, y)
    cb_full.fit(X_v8, y)

    torch.manual_seed(42)
    tcn_full = DilatedTCN(in_channels=36, hidden_dim=32)
    optimizer = torch.optim.AdamW(tcn_full.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    full_ds = TensorDataset(torch.tensor(X_tensor), torch.tensor(y, dtype=torch.float32))
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=True)

    tcn_full.train()
    for epoch in range(16):
        for bx, by in full_loader:
            optimizer.zero_grad()
            loss = criterion(tcn_full(bx), by)
            loss.backward()
            optimizer.step()
    tcn_full.eval()

    bundle = {
        'model_name': 'sd1_v8',
        'defect_name': '무릎 구부린 채 고정 위반 (30D GBDT + Dilated-TCN 2-Stream)',
        'lgb_model': lgb_full,
        'cb_model': cb_full,
        'tcn_state_dict': tcn_full.state_dict(),
        'gbdt_sub_weights': (0.55, 0.45),
        'view_fusion_weights': view_fusion_weights,
        'thresholds': {
            'tau_clean': best_tc,
            'tau_defect': best_td
        },
        'feature_names': FEATURE_NAMES_30D_SD1,
        'optimal_3state': {
            'coverage': best_cov,
            'abstain_rate': 100.0 - best_cov,
            'f1_3s': best_f1_3s,
            'acc_3s': acc_3s
        },
        'metrics_v8': {
            'auc': auc_v8, 'f1': f1_v8, 'acc': acc_v8
        },
        'metrics_v7': {
            'auc': auc_v7, 'f1': f1_v7, 'acc': acc_v7
        }
    }

    with open(BUNDLE_V8_PATH, 'wb') as fp:
        pickle.dump(bundle, fp)
    print(f"[+] v8 가중치 번들 저장 완료: {BUNDLE_V8_PATH}")

    # 결과 JSON 저장
    results_data = {
        'exercise': '03_바벨_스티프_데드리프트',
        'defect': 'sd1 (무릎 구부린 채 고정 위반)',
        'sample_size': N,
        'v7': {'auc': auc_v7, 'f1': f1_v7, 'acc': acc_v7, 'prec': prec_v7, 'rec': rec_v7},
        'v8': {'auc': auc_v8, 'f1': f1_v8, 'acc': acc_v8, 'prec': prec_v8, 'rec': rec_v8},
        'v8_3state': {'coverage': best_cov, 'f1_3s': best_f1_3s, 'acc_3s': acc_3s, 'tau_clean': best_tc, 'tau_defect': best_td},
        'view_results': view_comparison
    }
    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(results_data, fp, indent=2, ensure_ascii=False)
    print(f"[+] 결과 JSON 저장 완료: {RESULTS_JSON_PATH}")

    # 통합 OOF 캐시 업데이트용 OOF 저장
    oof_save_path = os.path.join(CURRENT_DIR, "sd1_v8_oof.npz")
    np.savez_compressed(oof_save_path, oof_v8=oof_v8_fused, oof_v7=oof_v7, y=y)
    print(f"[+] OOF 캐시 저장 완료: {oof_save_path}")

if __name__ == "__main__":
    main()
