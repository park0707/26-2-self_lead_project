# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반 (Knee-Hip Simultaneous Extension Synchrony)
Script: tournament_d4_optimal_combination.py
Purpose: 과적합 방지 및 기술 간 상호 간섭 배제를 위한 최적 기술 조합 탐색 토너먼트 (Ablation Tournament)
         - 2대 데이터 소스 (미보정 vs 4단계 기구학 보정)
         - 4대 피처 공간 (37D vs 43D vs Top 25 vs Top 15)
         - 6대 모델 아키텍처 (Tri-GBDT vs Regularized LGBM vs CatBoost vs 1D-TCN vs Two-Stream Meta vs Weighted Avg)
         - 3대 의사결정 전략 (2-State 0.50 vs Tuned Tau vs Calibrated 3-State)
         - 5-Fold StratifiedGroupKFold 피험자 엄격 격리
"""

import os
import sys

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import traceback
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(2)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from d4_model_v4 import (
    KinematicTCN,
    AsymmetricFocalLoss,
    BabelDeadliftD4ModelV4GBDT,
    FEATURE_NAMES_43_D4
)

CACHE_V4 = os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz")
CACHE_V5 = os.path.join(CURRENT_DIR, "d4_v5_features_cache.npz")
OUT_JSON = os.path.join(CURRENT_DIR, "d4_optimal_combination_results.json")
OUT_LOG = os.path.join(CURRENT_DIR, "tournament_live.log")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def log(msg: str):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def train_tcn_quick(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, epochs: int = 20) -> np.ndarray:
    device = torch.device('cpu')
    model = KinematicTCN(in_features=12, hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=3.0, lambda_neg=0.5)

    dataset_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
        probs = torch.sigmoid(model(val_x).squeeze(-1)).cpu().numpy()
    return probs


def evaluate_3state_risk(y_true: np.ndarray, p_final: np.ndarray,
                         p_aux1: np.ndarray, p_aux2: np.ndarray,
                         tau_n: float, tau_d: float,
                         risk_threshold_pct: float = 10.0) -> Dict[str, Any]:
    entropy = - (p_final * np.log2(np.clip(p_final, 1e-6, 1.0)) +
                 (1 - p_final) * np.log2(np.clip(1 - p_final, 1e-6, 1.0)))
    disagreement = np.abs(p_aux1 - p_aux2)
    risk_score = entropy + 1.2 * disagreement

    basic_undecided = (p_final > tau_n) & (p_final < tau_d)
    top_risk_val = np.percentile(risk_score, 100.0 - risk_threshold_pct)
    retake_mask = basic_undecided & (risk_score >= top_risk_val)

    if np.mean(retake_mask) > 0.100:
        retake_mask = (risk_score >= np.percentile(risk_score, 90.0))

    decided_mask = ~retake_mask
    coverage = float(np.mean(decided_mask))
    decline_rate = float(1.0 - coverage)

    y_decided_true = y_true[decided_mask]
    y_decided_pred = (p_final[decided_mask] >= 0.50).astype(int)

    if len(y_decided_true) > 0 and len(np.unique(y_decided_true)) > 1:
        f1 = float(f1_score(y_decided_true, y_decided_pred))
        acc = float(accuracy_score(y_decided_true, y_decided_pred))
        prec = float(precision_score(y_decided_true, y_decided_pred, zero_division=0))
        rec = float(recall_score(y_decided_true, y_decided_pred, zero_division=0))
    else:
        acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0

    return {
        "coverage": coverage,
        "decline_rate": decline_rate,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    }


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    best_tau = 0.50
    best_f1 = 0.0
    for tau in np.linspace(0.15, 0.70, 56):
        pred = (probs >= tau).astype(int)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)
    return best_tau, best_f1


def main():
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("=== vidio-health-trainer d4 Optimal Combination Tournament Live Log ===\n")

    log("================================================================================")
    log("  [vidio-health-trainer] 바벨 데드리프트 d4 최적 기술 조합 탐색 토너먼트")
    log("  - 목표: 과적합 방지, 기술 간 간섭 제거, 단일/융합 최적 조합 도출")
    log("  - 검증 규격: StratifiedGroupKFold (5-Fold 피험자 엄격 격리, 770 시퀀스 전수)")
    log("================================================================================")

    data_v4 = np.load(CACHE_V4, allow_pickle=True)
    data_v5 = np.load(CACHE_V5, allow_pickle=True)

    y = data_v4["y"]
    subjects = data_v4["subjects"]
    views = data_v4["views"]

    # 피처 중요도 기반 상위 인덱스 계산 (v4 및 v5 공통)
    # v4 LightGBM 기준 중요도
    n_pos = np.sum(y == 1)
    scale_pos = (len(y) - n_pos) / n_pos
    quick_lgb = lgb.LGBMClassifier(n_estimators=80, learning_rate=0.05, num_leaves=15,
                                   scale_pos_weight=scale_pos, random_state=42, verbose=-1)
    quick_lgb.fit(data_v4["X_gbdt"], y)
    imp_order = np.argsort(quick_lgb.feature_importances_)[::-1]
    top25_idx = imp_order[:25]
    top15_idx = imp_order[:15]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 실험 조합 정의
    CANDIDATES = [
        # [Group 1: 데이터 소스 및 피처 공간 단독 GBDT 비교]
        {
            "id": "C01",
            "name": "원본 MediaPipe + 37D GBDT (v3 베이스라인)",
            "data": "v4", "feat_mode": "37D", "model_type": "tri_gbdt", "use_tcn": False
        },
        {
            "id": "C02",
            "name": "원본 MediaPipe + 43D GBDT (v4 Phase 1)",
            "data": "v4", "feat_mode": "43D", "model_type": "tri_gbdt", "use_tcn": False
        },
        {
            "id": "C03",
            "name": "4단계 보정 + 43D GBDT",
            "data": "v5", "feat_mode": "43D", "model_type": "tri_gbdt", "use_tcn": False
        },
        {
            "id": "C04",
            "name": "4단계 보정 + Top25 피처선별 GBDT (과적합 억제)",
            "data": "v5", "feat_mode": "top25", "model_type": "tri_gbdt", "use_tcn": False
        },
        {
            "id": "C05",
            "name": "4단계 보정 + Top15 초경량 피처선별 GBDT (초단순화)",
            "data": "v5", "feat_mode": "top15", "model_type": "tri_gbdt", "use_tcn": False
        },

        # [Group 2: 정규화 단일 모델 (앙상블 과적합 방지)]
        {
            "id": "C06",
            "name": "4단계 보정 + L2 강규제 LightGBM (depth=3, colsample=0.7)",
            "data": "v5", "feat_mode": "43D", "model_type": "lgb_reg", "use_tcn": False
        },
        {
            "id": "C07",
            "name": "4단계 보정 + 얕은 CatBoost 단독 (depth=3, l2=5)",
            "data": "v5", "feat_mode": "43D", "model_type": "cat_reg", "use_tcn": False
        },

        # [Group 3: 시계열 1D-TCN 단독 비교]
        {
            "id": "C08",
            "name": "원본 MediaPipe + 64F 1D-TCN 단독",
            "data": "v4", "feat_mode": "tcn", "model_type": "tcn_only", "use_tcn": True
        },
        {
            "id": "C09",
            "name": "4단계 보정 + 64F 1D-TCN 단독 (관절 복원 시계열)",
            "data": "v5", "feat_mode": "tcn", "model_type": "tcn_only", "use_tcn": True
        },

        # [Group 4: Two-Stream 융합 전략 비교]
        {
            "id": "C10",
            "name": "원본 MediaPipe + Two-Stream Meta Fusion (v4.0 완성형)",
            "data": "v4", "feat_mode": "43D", "model_type": "fusion_meta", "use_tcn": True
        },
        {
            "id": "C11",
            "name": "4단계 보정 + Two-Stream Meta Fusion (v5.0 완성형)",
            "data": "v5", "feat_mode": "43D", "model_type": "fusion_meta", "use_tcn": True
        },
        {
            "id": "C12",
            "name": "4단계 보정 + Two-Stream 가중 평균 (0.5 GBDT + 0.5 TCN, 메타과적합 배제)",
            "data": "v5", "feat_mode": "43D", "model_type": "fusion_avg", "use_tcn": True
        },
        {
            "id": "C13",
            "name": "4단계 보정 + Top25 GBDT & 1D-TCN 결합 (피처선별 + 시계열)",
            "data": "v5", "feat_mode": "top25", "model_type": "fusion_avg", "use_tcn": True
        },
        {
            "id": "C14",
            "name": "★ 하이브리드: 원본 GBDT + 4단계 보정 1D-TCN 가중 결합",
            "data": "hybrid", "feat_mode": "43D", "model_type": "fusion_avg", "use_tcn": True
        }
    ]

    log(f"[*] 총 {len(CANDIDATES)}개 최적화 후보 조합 토너먼트 시작...")
    results = []

    # 1D-TCN OOF 사전 산출 (v4, v5 각각 1회씩만 계산하여 연산 효율 극대화)
    log("[*] 1D-TCN 5-Fold OOF 확률 사전 계산 중 (v4 & v5)...")
    oof_tcn_v4 = np.zeros(len(y), dtype=np.float32)
    oof_tcn_v5 = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(data_v4["X_gbdt"], y, groups=subjects), 1):
        torch.manual_seed(42 + fold)
        p_v4 = train_tcn_quick(data_v4["X_tcn"][train_idx], y[train_idx], data_v4["X_tcn"][val_idx], epochs=22)
        oof_tcn_v4[val_idx] = p_v4

        torch.manual_seed(42 + fold)
        p_v5 = train_tcn_quick(data_v5["X_tcn"][train_idx], y[train_idx], data_v5["X_tcn"][val_idx], epochs=22)
        oof_tcn_v5[val_idx] = p_v5

    log(f"[+] 1D-TCN OOF 계산 완료: v4 AUC={roc_auc_score(y, oof_tcn_v4):.4f}, v5 AUC={roc_auc_score(y, oof_tcn_v5):.4f}")

    # 각 조합별 5-Fold 평가
    for idx, c in enumerate(CANDIDATES, 1):
        t0 = time.time()
        c_id = c["id"]
        c_name = c["name"]
        d_source = c["data"]
        f_mode = c["feat_mode"]
        m_type = c["model_type"]

        # 피처 공간 선택
        if d_source == "v4":
            X_raw = data_v4["X_gbdt"]
            tcn_oof = oof_tcn_v4
        elif d_source == "v5":
            X_raw = data_v5["X_gbdt"]
            tcn_oof = oof_tcn_v5
        else: # hybrid
            X_raw = data_v4["X_gbdt"]
            tcn_oof = oof_tcn_v5

        if f_mode == "37D":
            X_feat = X_raw[:, :37]
        elif f_mode == "43D":
            X_feat = X_raw
        elif f_mode == "top25":
            X_feat = X_raw[:, top25_idx]
        elif f_mode == "top15":
            X_feat = X_raw[:, top15_idx]
        else:
            X_feat = X_raw

        oof_probs = np.zeros(len(y), dtype=np.float32)
        oof_gbdt_aux = np.zeros(len(y), dtype=np.float32)

        if m_type == "tcn_only":
            oof_probs = tcn_oof.copy()
            oof_gbdt_aux = tcn_oof.copy()
        else:
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_feat, y, groups=subjects), 1):
                X_tr, y_tr = X_feat[train_idx], y[train_idx]
                X_va = X_feat[val_idx]

                n_p = max(np.sum(y_tr == 1), 1)
                n_n = max(np.sum(y_tr == 0), 1)
                spw = n_n / n_p

                if m_type == "tri_gbdt" or "fusion" in m_type:
                    # Tri-Ensemble
                    model = BabelDeadliftD4ModelV4GBDT()
                    model.fit(X_tr, y_tr, random_state=42 + fold)
                    p_val_g = model.predict_proba(X_va)
                elif m_type == "lgb_reg":
                    # 강력한 규제 LightGBM (트리 깊이 3, 컬럼 샘플링 0.7)
                    clf = lgb.LGBMClassifier(
                        n_estimators=70, learning_rate=0.04, max_depth=3, num_leaves=7,
                        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
                        scale_pos_weight=spw, n_jobs=1, random_state=42 + fold, verbose=-1
                    )
                    clf.fit(X_tr, y_tr)
                    p_val_g = clf.predict_proba(X_va)[:, 1]
                elif m_type == "cat_reg":
                    clf = CatBoostClassifier(
                        iterations=70, learning_rate=0.05, depth=3, l2_leaf_reg=5.0,
                        scale_pos_weight=spw, random_seed=42 + fold, verbose=0, thread_count=1
                    )
                    clf.fit(X_tr, y_tr)
                    p_val_g = clf.predict_proba(X_va)[:, 1]
                else:
                    p_val_g = np.zeros(len(val_idx))

                oof_gbdt_aux[val_idx] = p_val_g

            # 융합 처리
            if m_type == "fusion_meta":
                stack_X = np.column_stack([oof_gbdt_aux, tcn_oof])
                lr = LogisticRegression(class_weight='balanced', C=1.0, random_state=42)
                lr.fit(stack_X, y)
                oof_probs = lr.predict_proba(stack_X)[:, 1]
            elif m_type == "fusion_avg":
                # 순수 가중 평균 (과적합 제로)
                oof_probs = 0.5 * oof_gbdt_aux + 0.5 * tcn_oof
            else:
                oof_probs = oof_gbdt_aux.copy()

        # 성능 지표 산출
        # 1. 고정 임계치 (0.50)
        p_50 = (oof_probs >= 0.50).astype(int)
        f1_50 = float(f1_score(y, p_50, zero_division=0))
        rec_50 = float(recall_score(y, p_50, zero_division=0))
        prec_50 = float(precision_score(y, p_50, zero_division=0))
        acc_50 = float(accuracy_score(y, p_50))
        auc = float(roc_auc_score(y, oof_probs))

        # 2. 최적 임계치 튜닝
        best_tau, tuned_f1 = find_best_threshold(y, oof_probs)
        p_tuned = (oof_probs >= best_tau).astype(int)
        tuned_rec = float(recall_score(y, p_tuned, zero_division=0))
        tuned_prec = float(precision_score(y, p_tuned, zero_division=0))

        # 3. 3-State 신뢰도 보정 (Coverage >= 90%)
        best_3s_f1 = 0.0
        best_3s_cov = 1.0
        best_3s_rec = 0.0
        best_3s_prec = 0.0
        for tn in np.linspace(0.20, 0.45, 16):
            for td in np.linspace(0.40, 0.65, 16):
                if tn >= td:
                    continue
                m3 = evaluate_3state_risk(y, oof_probs, oof_gbdt_aux, tcn_oof, tn, td)
                if m3["coverage"] >= 0.900 and m3["f1"] > best_3s_f1:
                    best_3s_f1 = m3["f1"]
                    best_3s_cov = m3["coverage"]
                    best_3s_rec = m3["recall"]
                    best_3s_prec = m3["precision"]

        # 시점별 F1 산출 (시점 E, B, C, D, A)
        v_f1_map = {}
        for vn in ["view1", "view2", "view3", "view4", "view5"]:
            m_v = (views == vn)
            v_f1_map[vn] = float(f1_score(y[m_v], (oof_probs[m_v] >= 0.50).astype(int), zero_division=0))

        item = {
            "rank": 0,
            "id": c_id,
            "name": c_name,
            "f1_2state": f1_50,
            "recall_2state": rec_50,
            "prec_2state": prec_50,
            "accuracy": acc_50,
            "auc": auc,
            "tuned_tau": best_tau,
            "tuned_f1": tuned_f1,
            "tuned_recall": tuned_rec,
            "3state_f1": best_3s_f1,
            "3state_coverage": best_3s_cov,
            "3state_recall": best_3s_rec,
            "view_f1": v_f1_map,
            "time_sec": round(time.time() - t0, 1)
        }
        results.append(item)

        log(f"[{c_id}] {c_name:<45} | 2S-F1: {f1_50*100:5.2f}% (Rec: {rec_50*100:5.2f}%) | Tuned-F1: {tuned_f1*100:5.2f}% (τ={best_tau:.2f}) | 3S-F1: {best_3s_f1*100:5.2f}% | AUC: {auc:.4f} ({item['time_sec']}s)")

    # 3-State F1 및 Tuned F1 기준으로 정렬
    results.sort(key=lambda x: max(x["3state_f1"], x["tuned_f1"], x["f1_2state"]), reverse=True)
    for r, it in enumerate(results, 1):
        it["rank"] = r

    log("================================================================================")
    log("  [토너먼트 최종 결과 상위 순위 (Top Ranked Combinations)]")
    log("================================================================================")
    for it in results[:7]:
        log(f"  Rank {it['rank']:2d} | [{it['id']}] {it['name']:<48} | Max-F1: {max(it['3state_f1'], it['tuned_f1'], it['f1_2state'])*100:5.2f}% (3S: {it['3state_f1']*100:5.2f}%, 2S: {it['f1_2state']*100:5.2f}%) | AUC: {it['auc']:.4f}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[+] 토너먼트 결과 JSON 저장 완료: {OUT_JSON}")


if __name__ == "__main__":
    main()
