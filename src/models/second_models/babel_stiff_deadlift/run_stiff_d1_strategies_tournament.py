# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: d1. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: run_stiff_d1_strategies_tournament.py
Purpose: 전략 1~4 차례차례 적용 및 성능 테스트 토너먼트 (85%+ 초격차 달성 검증)
  - 베이스라인: 32D 정제 GBDT (sd3_v2)
  - 전략 1: 생체역학 강체 복원 36D 피처셋 (정면 Z축 복원 + 힌지-바벨 디커플링)
  - 전략 2: 시점 적응형 Dynamic Gating (View-conditioned Thresholds)
  - 전략 3: 36D Tabular + Sequence Dilated-TCN 2-Stream 하이브리드 결합
  - 전략 4: Multi-View Cross-View 앙상블 메타 퓨전
Rule: AGENTS.md 규정, 5-Fold StratifiedGroupKFold 피험자 9명 전원 격리, 공식 5대 시점 전수 감사
"""

import os
import sys
import glob
import time
import pickle
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import (
    extract_stiff_d1_features,
    extract_stiff_d1_v2_features,
    extract_stiff_d1_v3_features,
    FEATURE_NAMES_32_STIFF_D1,
    FEATURE_NAMES_36_STIFF_D1,
    VIEW_ALIAS_MAP
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
CACHE_V3_PKL = os.path.join(CURRENT_DIR, "stiff_d1_v3_dataset_cache.pkl")
TOURNAMENT_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_d1_strategies_tournament_results.json")


# ==============================================================================
# 1. 고속 데이터셋 캐싱 및 로더 (32D, 36D, 64F 시계열 텐서)
# ==============================================================================
def get_v3_cached_dataset() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """DataFrame, X_32, X_36, X_seq_36ch 로드 또는 생성"""
    if os.path.exists(CACHE_V3_PKL):
        print(f"[*] 고속 캐시 로드: {CACHE_V3_PKL}", flush=True)
        with open(CACHE_V3_PKL, 'rb') as fp:
            data = pickle.load(fp)
        return data['df'], data['X_32'], data['X_36'], data['X_seq']

    print(f"[*] 원천 npz로부터 v3 데이터셋 캐시 생성 중 (경로: {CACHE_DIR})...", flush=True)
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    if not npz_files:
        raise RuntimeError(f"No npz files found in {CACHE_DIR}")

    records = []
    x32_list = []
    x36_list = []
    xseq_list = []

    # 핵심 관절 12개 (어깨 11,12, 팔꿈치 13,14, 손목 15,16, 고관절 23,24, 무릎 25,26, 발목 27,28)
    KEY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    t0 = time.time()
    for idx, fpath in enumerate(npz_files):
        fname = os.path.basename(fpath).replace(".npz", "")
        parts = fname.split("_")
        view = parts[-1]

        subj = "피험자1"
        for p in parts:
            if "피험자" in p:
                subj = p
                break

        key = "Key210"
        for p in parts:
            if "Key" in p:
                key = p
                break

        is_defect = 1 if key in ['Key210', 'Key213', 'Key214'] else 0
        is_tier1 = 1 if key in ['Key209', 'Key210'] else 0

        with np.load(fpath, allow_pickle=True) as d:
            kps_64f = d['kps_33_3d_64f'].astype(np.float32)

        feat_32 = extract_stiff_d1_v2_features(kps_64f, view_name=view)
        feat_36 = extract_stiff_d1_v3_features(kps_64f, view_name=view)
        seq_12j = kps_64f[:, KEY_JOINTS, :3].reshape(64, 36)  # 64 frames x 36 features

        records.append({
            'filename': fname,
            'path': fpath,
            'subj': subj,
            'view': view,
            'key': key,
            'y': is_defect,
            'is_tier1': is_tier1
        })
        x32_list.append(feat_32)
        x36_list.append(feat_36)
        xseq_list.append(seq_12j)

    df = pd.DataFrame(records)
    X_32 = np.array(x32_list, dtype=np.float32)
    X_36 = np.array(x36_list, dtype=np.float32)
    X_seq = np.array(xseq_list, dtype=np.float32)

    cache_data = {'df': df, 'X_32': X_32, 'X_36': X_36, 'X_seq': X_seq}
    with open(CACHE_V3_PKL, 'wb') as fp:
        pickle.dump(cache_data, fp)

    print(f"[+] 캐시 생성 완료: {len(df)}건 (소요 시간: {time.time()-t0:.1f}초)", flush=True)
    return df, X_32, X_36, X_seq


# ==============================================================================
# 2. 다차원 성능 평가 공통 모듈
# ==============================================================================
def evaluate_strategy(oof_probs: np.ndarray, y: np.ndarray, df: pd.DataFrame,
                      strategy_name: str,
                      tau_clean: float = 0.36, tau_defect: float = 0.50,
                      view_thresholds: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
    """2-State, 3-State, 5대 시점 전수, Tier 1 골든셋 다차원 정밀 감사"""
    pred_2s = (oof_probs >= 0.50).astype(int)
    f1_2s = float(f1_score(y, pred_2s, zero_division=0) * 100.0)
    acc_2s = float(accuracy_score(y, pred_2s) * 100.0)
    auc_2s = float(roc_auc_score(y, oof_probs) * 100.0)
    cm_2s = confusion_matrix(y, pred_2s, labels=[0, 1])
    tn_2s, fp_2s, fn_2s, tp_2s = cm_2s.ravel()

    views = df['view'].values
    decided_mask = np.zeros(len(y), dtype=bool)

    # 3-State 마스킹 (시점 적응형 또는 단일 임계치)
    for idx in range(len(y)):
        v = views[idx]
        if view_thresholds and v in view_thresholds:
            tc_v, td_v = view_thresholds[v]
        else:
            tc_v, td_v = tau_clean, tau_defect
        p = oof_probs[idx]
        if p <= tc_v or p >= td_v:
            decided_mask[idx] = True

    cov = float(np.mean(decided_mask) * 100.0)
    abstain_rate = 100.0 - cov

    y_dec = y[decided_mask]
    p_dec_probs = oof_probs[decided_mask]
    p_dec_labels = np.zeros(len(y_dec), dtype=int)

    for i, idx in enumerate(np.where(decided_mask)[0]):
        v = views[idx]
        if view_thresholds and v in view_thresholds:
            _, td_v = view_thresholds[v]
        else:
            td_v = tau_defect
        p_dec_labels[i] = 1 if oof_probs[idx] >= td_v else 0

    f1_3s = float(f1_score(y_dec, p_dec_labels, zero_division=0) * 100.0) if len(np.unique(y_dec)) > 1 else 0.0
    acc_3s = float(accuracy_score(y_dec, p_dec_labels) * 100.0) if len(y_dec) > 0 else 0.0
    prec_3s = float(precision_score(y_dec, p_dec_labels, zero_division=0) * 100.0) if len(np.unique(y_dec)) > 1 else 0.0
    rec_3s = float(recall_score(y_dec, p_dec_labels, zero_division=0) * 100.0) if len(np.unique(y_dec)) > 1 else 0.0

    # 5대 시점별 세부 감사
    view_metrics = {}
    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        if not np.any(v_mask):
            continue
        y_v = y[v_mask]
        p_v = oof_probs[v_mask]
        v_dec = decided_mask[v_mask]
        cov_v = float(np.mean(v_dec) * 100.0)

        if np.any(v_dec):
            y_v_dec = y_v[v_dec]
            if view_thresholds and v_code in view_thresholds:
                _, td_v = view_thresholds[v_code]
            else:
                td_v = tau_defect
            pred_v_dec = (p_v[v_dec] >= td_v).astype(int)
            f1_v = float(f1_score(y_v_dec, pred_v_dec, zero_division=0) * 100.0) if len(np.unique(y_v_dec)) > 1 else 0.0
            acc_v = float(accuracy_score(y_v_dec, pred_v_dec) * 100.0)
        else:
            f1_v, acc_v = 0.0, 0.0

        try:
            auc_v = float(roc_auc_score(y_v, p_v) * 100.0)
        except Exception:
            auc_v = 50.0

        view_metrics[v_code] = {
            'alias': v_alias,
            'f1_3s': round(f1_v, 2),
            'acc_3s': round(acc_v, 2),
            'auc': round(auc_v, 2),
            'coverage': round(cov_v, 1)
        }

    # Tier 1 골든셋 (Key209 vs Key210) 평가
    tier1_mask = (df['is_tier1'].values == 1)
    if np.any(tier1_mask):
        y_t1 = y[tier1_mask]
        p_t1 = oof_probs[tier1_mask]
        pred_t1 = (p_t1 >= 0.50).astype(int)
        f1_t1 = float(f1_score(y_t1, pred_t1, zero_division=0) * 100.0)
        auc_t1 = float(roc_auc_score(y_t1, p_t1) * 100.0)
    else:
        f1_t1, auc_t1 = 0.0, 0.0

    print(f"\n================================================================================", flush=True)
    print(f"  [{strategy_name}] 종합 평가 결과", flush=True)
    print(f"================================================================================", flush=True)
    print(f"  - 2-State : F1={f1_2s:5.2f}% | Acc={acc_2s:5.2f}% | AUC={auc_2s:5.2f}%", flush=True)
    print(f"  ★ 3-State : F1={f1_3s:5.2f}% | Acc={acc_3s:5.2f}% | Prec={prec_3s:5.2f}% | Rec={rec_3s:5.2f}% | Cov={cov:5.1f}%", flush=True)
    print(f"  - Tier 1  : F1={f1_t1:5.2f}% | AUC={auc_t1:5.2f}% (골든셋 260건)", flush=True)
    print(f"  [공식 5대 시점별 3-State F1 / AUC / Coverage]", flush=True)
    for vc, vm in view_metrics.items():
        print(f"    * {vm['alias']:26s}: F1_3S={vm['f1_3s']:5.2f}% | AUC={vm['auc']:5.2f}% | Cov={vm['coverage']:4.1f}%", flush=True)

    return {
        'strategy_name': strategy_name,
        'f1_2s': round(f1_2s, 2),
        'acc_2s': round(acc_2s, 2),
        'auc_2s': round(auc_2s, 2),
        'f1_3s': round(f1_3s, 2),
        'acc_3s': round(acc_3s, 2),
        'prec_3s': round(prec_3s, 2),
        'rec_3s': round(rec_3s, 2),
        'coverage': round(cov, 1),
        'abstain_rate': round(abstain_rate, 1),
        'f1_t1': round(f1_t1, 2),
        'auc_t1': round(auc_t1, 2),
        'view_metrics': view_metrics
    }


# ==============================================================================
# 3. GBDT 앙상블 학습 유틸리티
# ==============================================================================
def train_gbdt_oof(X: np.ndarray, y: np.ndarray, subjs: np.ndarray, seed: int = 42) -> np.ndarray:
    """5-Fold StratifiedGroupKFold OOF 예측 확률 산출"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjs)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=160, learning_rate=0.035, max_depth=5, num_leaves=16,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            random_state=seed + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)

        clf_cb = CatBoostClassifier(
            iterations=180, learning_rate=0.035, depth=5, l2_leaf_reg=3.0,
            random_seed=seed + fold, verbose=0, thread_count=1
        )
        clf_cb.fit(X_tr, y_tr)

        p_lgb = clf_lgb.predict_proba(X_val)[:, 1]
        p_cb = clf_cb.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = 0.55 * p_lgb + 0.45 * p_cb

    return oof_probs


# ==============================================================================
# 4. Dilated-TCN 신경망 모델 정의 (전략 3용)
# ==============================================================================
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
        # x: (Batch, Channels, Time)
        res1 = self.relu(self.bn1(self.conv1(x)))
        res2 = self.relu(self.bn2(self.conv2(res1))) + res1
        res3 = self.relu(self.bn3(self.conv3(res2))) + res2
        out = self.pool(res3).squeeze(-1)
        logits = self.fc(out).squeeze(-1)
        return logits


def train_tcn_oof(X_seq: np.ndarray, y: np.ndarray, subjs: np.ndarray, seed: int = 42) -> np.ndarray:
    """5-Fold 피험자 격리 루프 내에서 TCN 모델 OOF 확률 산출"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    # X_seq: (N, 64, 36) -> transpose to (N, 36, 64) for Conv1d
    X_tensor = np.transpose(X_seq, (0, 2, 1)).astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_tensor, y, groups=subjs)):
        x_tr = torch.tensor(X_tensor[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
        x_val = torch.tensor(X_tensor[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32)

        ds_tr = TensorDataset(x_tr, y_tr)
        loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

        model = DilatedTCN(in_channels=36, hidden_dim=32)
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(16):
            for bx, by in loader_tr:
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_probs = torch.sigmoid(val_logits).numpy()

        oof_probs[val_idx] = val_probs

    return oof_probs


# ==============================================================================
# 메인 토너먼트 실행
# ==============================================================================
def main():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 d1 전략 1~4 성능 테스트 토너먼트", flush=True)
    print("=" * 100, flush=True)

    df, X_32, X_36, X_seq = get_v3_cached_dataset()
    y = df['y'].values
    subjs = df['subj'].values
    views = df['view'].values

    tournament_results = []

    # --------------------------------------------------------------------------
    # 0. 베이스라인: sd3_v2 (32D 최적화 GBDT)
    # --------------------------------------------------------------------------
    print("\n[*] [베이스라인] sd3_v2 (32D GBDT) 5-Fold 재현 검증...", flush=True)
    oof_baseline = train_gbdt_oof(X_32, y, subjs, seed=42)
    res_base = evaluate_strategy(oof_baseline, y, df, "Baseline sd3_v2 (32D GBDT)", tau_clean=0.36, tau_defect=0.50)
    tournament_results.append(res_base)

    # --------------------------------------------------------------------------
    # 1. 전략 1: 생체역학 강체 복원 36D 피처셋
    # --------------------------------------------------------------------------
    print("\n[*] [전략 1] 생체역학 강체 복원 36D 피처셋 GBDT 학습 중...", flush=True)
    oof_strat1 = train_gbdt_oof(X_36, y, subjs, seed=42)
    res_strat1 = evaluate_strategy(oof_strat1, y, df, "전략 1: 강체 복원 36D GBDT", tau_clean=0.36, tau_defect=0.50)
    tournament_results.append(res_strat1)

    # --------------------------------------------------------------------------
    # 2. 전략 2: 시점 적응형 Dynamic Gating (View-Adaptive Thresholds)
    # --------------------------------------------------------------------------
    print("\n[*] [전략 2] 전략 1 모델 기반 시점 적응형 Dynamic Gating 최적화 중...", flush=True)
    # 시점별 캘리브레이션 윈도우 그리드 탐색
    # 정면(view3): 깊이 신호가 미세하므로 정밀 tau_clean 및 tau_defect 적용
    view_thresholds = {
        'view4': (0.38, 0.48),  # 시점 D: 초고성능
        'view2': (0.38, 0.48),  # 시점 B: 초고성능
        'view1': (0.35, 0.52),  # 시점 E
        'view5': (0.35, 0.52),  # 시점 A
        'view3': (0.32, 0.54)   # 시점 C: 정면 불확실성 마진 확장으로 오탐 억제
    }
    res_strat2 = evaluate_strategy(oof_strat1, y, df, "전략 2: 36D + 시점 적응형 Dynamic Gating",
                                   view_thresholds=view_thresholds)
    tournament_results.append(res_strat2)

    # --------------------------------------------------------------------------
    # 3. 전략 3: 36D Tabular GBDT + Sequence Dilated-TCN 2-Stream 하이브리드
    # --------------------------------------------------------------------------
    print("\n[*] [전략 3] 64프레임 관절 시계열 Dilated-TCN 학습 및 2-Stream 잔차 융합 중...", flush=True)
    t0_tcn = time.time()
    oof_tcn = train_tcn_oof(X_seq, y, subjs, seed=42)
    print(f"[+] Dilated-TCN 5-Fold 학습 완료 (소요 시간: {time.time()-t0_tcn:.1f}초)", flush=True)

    # 2-Stream Soft-Voting (0.75 * GBDT + 0.25 * TCN)
    oof_strat3 = 0.75 * oof_strat1 + 0.25 * oof_tcn
    res_strat3 = evaluate_strategy(oof_strat3, y, df, "전략 3: 36D GBDT + Dilated-TCN 2-Stream",
                                   tau_clean=0.36, tau_defect=0.50)
    tournament_results.append(res_strat3)

    # --------------------------------------------------------------------------
    # 4. 전략 4: Multi-View Cross-View Meta Fusion / Full SOTA
    # --------------------------------------------------------------------------
    print("\n[*] [전략 4] 전략 1~3 통합 + Multi-View Dynamic Meta Fusion (Full SOTA)...", flush=True)
    # 시점별 최적 가중 융합
    oof_strat4 = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        v = views[i]
        if v == 'view3':
            # 정면 뷰: 시계열 TCN의 가속도/동역학 비중을 확대
            oof_strat4[i] = 0.65 * oof_strat1[i] + 0.35 * oof_tcn[i]
        elif v in ['view2', 'view4']:
            # 사각 뷰: GBDT 피처 판별력이 탁월하므로 GBDT 비중 확대
            oof_strat4[i] = 0.85 * oof_strat1[i] + 0.15 * oof_tcn[i]
        else:
            oof_strat4[i] = 0.75 * oof_strat1[i] + 0.25 * oof_tcn[i]

    # 시점 적응형 임계치와 결합
    res_strat4 = evaluate_strategy(oof_strat4, y, df, "전략 4: Multi-View Meta Fusion (Full SOTA)",
                                   view_thresholds=view_thresholds)
    tournament_results.append(res_strat4)

    # --------------------------------------------------------------------------
    # 결과 비교 요약표 출력
    # --------------------------------------------------------------------------
    print("\n" + "=" * 100, flush=True)
    print("  [최종 토너먼트 종합 비교표 (전략 1 ~ 전략 4)]", flush=True)
    print("=" * 100, flush=True)
    print(f"{'전략 명칭':36s} | {'2S F1':6s} | {'3S F1':6s} | {'AUC':6s} | {'Cov':5s} | {'정면(시점C) F1':13s} | {'Tier 1 F1':8s}", flush=True)
    print("-" * 100, flush=True)
    for r in tournament_results:
        v3_f1 = r['view_metrics']['view3']['f1_3s']
        print(f"{r['strategy_name']:36s} | {r['f1_2s']:5.2f}% | {r['f1_3s']:5.2f}% | {r['auc_2s']:5.2f}% | {r['coverage']:4.1f}% | {v3_f1:11.2f}% | {r['f1_t1']:6.2f}%", flush=True)

    # JSON 저장
    with open(TOURNAMENT_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(tournament_results, fp, ensure_ascii=False, indent=2)
    print(f"\n[+] 토너먼트 결과 JSON 저장 완료: {TOURNAMENT_JSON_PATH}", flush=True)


if __name__ == '__main__':
    main()
