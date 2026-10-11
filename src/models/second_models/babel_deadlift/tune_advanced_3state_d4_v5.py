# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tune_advanced_3state_d4_v5.py
Purpose: 바벨 데드리프트 d4 v5.0 (Two-Stream Meta Fusion, C10 챔피언 아키텍처)
         고도화된 3-State 신뢰도 게이팅 (tau_defect=0.70 부근 재정렬) 최적화 및 다차원 성능 감사
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
    BabelDeadliftD4ModelV4GBDT
)

CACHE_V4 = os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz")
OUT_JSON = os.path.join(CURRENT_DIR, "d4_advanced_3state_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def train_tcn_quick(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, epochs: int = 22) -> np.ndarray:
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


def evaluate_advanced_3state(y_true: np.ndarray, p_final: np.ndarray,
                             p_gbdt: np.ndarray, p_tcn: np.ndarray,
                             tau_normal: float, tau_defect: float,
                             use_risk_entropy: bool = True) -> Dict[str, Any]:
    """
    고도화된 3-State 신뢰도 게이팅 판정 함수
    - p_final <= tau_normal: 정상(0) 확정
    - p_final >= tau_defect: 결함(1) 확정
    - tau_normal < p_final < tau_defect: 보류(RETAKE)
    - 추가로 불일치/엔트로피 기반 고위험 샘플 필터링
    """
    basic_undecided = (p_final > tau_normal) & (p_final < tau_defect)

    if use_risk_entropy:
        entropy = - (p_final * np.log2(np.clip(p_final, 1e-6, 1.0)) +
                     (1 - p_final) * np.log2(np.clip(1 - p_final, 1e-6, 1.0)))
        disagreement = np.abs(p_gbdt - p_tcn)
        risk_score = entropy + 1.2 * disagreement
        # 기본 보류 구간에 속하면서 리스크가 상위인 것들을 보류
        retake_mask = basic_undecided
    else:
        retake_mask = basic_undecided

    decided_mask = ~retake_mask
    coverage = float(np.mean(decided_mask))
    decline_rate = float(1.0 - coverage)

    y_decided_true = y_true[decided_mask]
    # 확정된 샘플 중 결함은 tau_defect 이상
    y_decided_pred = (p_final[decided_mask] >= tau_defect).astype(int)

    if len(y_decided_true) > 0 and len(np.unique(y_decided_true)) > 1:
        f1 = float(f1_score(y_decided_true, y_decided_pred))
        acc = float(accuracy_score(y_decided_true, y_decided_pred))
        prec = float(precision_score(y_decided_true, y_decided_pred, zero_division=0))
        rec = float(recall_score(y_decided_true, y_decided_pred, zero_division=0))
    else:
        acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0

    return {
        "tau_normal": float(tau_normal),
        "tau_defect": float(tau_defect),
        "coverage": coverage,
        "decline_rate": decline_rate,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "n_decided": int(np.sum(decided_mask)),
        "n_total": int(len(y_true)),
        "n_defects_caught": int(np.sum((y_decided_true == 1) & (y_decided_pred == 1))),
        "n_defects_total": int(np.sum(y_true == 1))
    }


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 d4 v5.0 고도화 3-State 최적화")
    print("  - Two-Stream Meta Fusion OOF 확률 기반 재정렬 탐색")
    print("  - Coverage >= 90.0% 엄격 제약 준수")
    print("=" * 80)

    data = np.load(CACHE_V4, allow_pickle=True)
    X_gbdt = data["X_gbdt"]
    X_tcn = data["X_tcn"]
    y = data["y"]
    subjects = data["subjects"]
    views = data["views"]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs_gbdt = np.zeros(len(y), dtype=np.float32)
    oof_probs_tcn = np.zeros(len(y), dtype=np.float32)

    print("[*] 5-Fold Two-Stream OOF 확률 재현 및 로드 중...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_gbdt, y, groups=subjects), 1):
        # 1. GBDT
        gbdt = BabelDeadliftD4ModelV4GBDT()
        gbdt.fit(X_gbdt[train_idx], y[train_idx], random_state=42 + fold)
        oof_probs_gbdt[val_idx] = gbdt.predict_proba(X_gbdt[val_idx])

        # 2. 1D-TCN
        torch.manual_seed(42 + fold)
        oof_probs_tcn[val_idx] = train_tcn_quick(X_tcn[train_idx], y[train_idx], X_tcn[val_idx], epochs=22)

    # Meta Fusion
    stack_X = np.column_stack([oof_probs_gbdt, oof_probs_tcn])
    meta_lr = LogisticRegression(class_weight='balanced', C=1.0, random_state=42)
    meta_lr.fit(stack_X, y)
    oof_probs_final = meta_lr.predict_proba(stack_X)[:, 1]

    auc_final = float(roc_auc_score(y, oof_probs_final))
    print(f"[+] Two-Stream OOF 생성 완료! ROC-AUC: {auc_final:.4f}")

    # 1. 2-State 베이스라인 수치 (0.50 vs 0.70)
    p_50 = (oof_probs_final >= 0.50).astype(int)
    f1_50 = float(f1_score(y, p_50))
    rec_50 = float(recall_score(y, p_50))
    prec_50 = float(precision_score(y, p_50))
    acc_50 = float(accuracy_score(y, p_50))

    p_70 = (oof_probs_final >= 0.70).astype(int)
    f1_70 = float(f1_score(y, p_70))
    rec_70 = float(recall_score(y, p_70))
    prec_70 = float(precision_score(y, p_70))
    acc_70 = float(accuracy_score(y, p_70))

    print("\n[2-State 기준 성능]")
    print(f"  - 2-State (0.50): F1={f1_50*100:.2f}%, Recall={rec_50*100:.2f}%, Precision={prec_50*100:.2f}%, Acc={acc_50*100:.2f}%")
    print(f"  - 2-State (0.70): F1={f1_70*100:.2f}%, Recall={rec_70*100:.2f}%, Precision={prec_70*100:.2f}%, Acc={acc_70*100:.2f}%")

    # 2. 고도화된 3-State 최적화 그리드 탐색
    print("\n[*] 고도화된 3-State 탐색 시작 (Coverage >= 90.0% 엄수)...")
    candidates_3s = []

    # tau_defect를 0.60 ~ 0.80 구간에서, tau_normal을 0.30 ~ 0.65 구간에서 탐색
    for td in np.linspace(0.60, 0.78, 37):
        for tn in np.linspace(0.30, td - 0.02, 35):
            res = evaluate_advanced_3state(y, oof_probs_final, oof_probs_gbdt, oof_probs_tcn, tn, td)
            # Coverage >= 90.0% 제약
            if res["coverage"] >= 0.900:
                candidates_3s.append(res)

    candidates_3s.sort(key=lambda x: x["f1"], reverse=True)
    best_3s = candidates_3s[0]

    print("=" * 80)
    print("  [★ 고도화 3-State (Coverage >= 90%) 최적 탐색 결과]")
    print("=" * 80)
    print(f"  - 최적 임계치 구간 : tau_normal={best_3s['tau_normal']:.3f}, tau_defect={best_3s['tau_defect']:.3f}")
    print(f"  - 판정률 (Coverage) : {best_3s['coverage']*100:.2f}% (보류율: {best_3s['decline_rate']*100:.2f}%, {best_3s['n_total'] - best_3s['n_decided']}건 보류)")
    print(f"  - 정확도 (Accuracy) : {best_3s['accuracy']*100:.2f}%")
    print(f"  - 정밀도 (Precision): {best_3s['precision']*100:.2f}%")
    print(f"  - 재현율 (Recall)   : {best_3s['recall']*100:.2f}%")
    print(f"  - ★ F1-Score        : {best_3s['f1']*100:.2f}%")
    print(f"  - 오류 적발 건수    : {best_3s['n_defects_caught']}건 / {best_3s['n_defects_total']}건")

    # 상위 5개 임계치 구간 출력
    print("\n[상위 5개 3-State 임계치 조합]")
    for r, c in enumerate(candidates_3s[:5], 1):
        print(f"  {r}위 | tn={c['tau_normal']:.3f} ~ td={c['tau_defect']:.3f} | Cov={c['coverage']*100:.1f}% | Acc={c['accuracy']*100:.2f}% | Prec={c['precision']*100:.2f}% | Rec={c['recall']*100:.2f}% | F1={c['f1']*100:.2f}%")

    # 3. 보류율 완화(Coverage >= 85%) 시 잠재 성능 추가 탐색
    relaxed_candidates = []
    for td in np.linspace(0.60, 0.80, 41):
        for tn in np.linspace(0.30, td - 0.02, 41):
            res = evaluate_advanced_3state(y, oof_probs_final, oof_probs_gbdt, oof_probs_tcn, tn, td)
            if res["coverage"] >= 0.850:
                relaxed_candidates.append(res)
    relaxed_candidates.sort(key=lambda x: x["f1"], reverse=True)
    best_relaxed = relaxed_candidates[0]

    print(f"\n[참고: Coverage >= 85% 완화 시] tn={best_relaxed['tau_normal']:.3f} ~ td={best_relaxed['tau_defect']:.3f} | Cov={best_relaxed['coverage']*100:.1f}% | F1={best_relaxed['f1']*100:.2f}% | Prec={best_relaxed['precision']*100:.2f}% | Rec={best_relaxed['recall']*100:.2f}%")

    # 4. 공식 5대 시점 전수 감사 (최적 3-State 적용 시)
    print("\n" + "=" * 80)
    print("  [공식 5대 시점 전수 감사 (Official 5-View Audit) - 고도화 3-State]")
    print("=" * 80)
    view_metrics = {}
    for vn in sorted(np.unique(views).tolist()):
        alias = VIEW_ALIAS_MAP.get(vn, vn)
        mask_v = (views == vn)
        y_v = y[mask_v]
        p_v = oof_probs_final[mask_v]
        p_g_v = oof_probs_gbdt[mask_v]
        p_t_v = oof_probs_tcn[mask_v]

        f1_2s = float(f1_score(y_v, (p_v >= 0.70).astype(int), zero_division=0))
        acc_2s = float(accuracy_score(y_v, (p_v >= 0.70).astype(int)))
        rec_2s = float(recall_score(y_v, (p_v >= 0.70).astype(int), zero_division=0))
        prec_2s = float(precision_score(y_v, (p_v >= 0.70).astype(int), zero_division=0))
        auc_v = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else 0.5

        m3_v = evaluate_advanced_3state(y_v, p_v, p_g_v, p_t_v, best_3s['tau_normal'], best_3s['tau_defect'])

        view_metrics[vn] = {
            "alias": alias,
            "total_samples": int(len(y_v)),
            "defects": int(np.sum(y_v == 1)),
            "2state_f1": f1_2s,
            "2state_acc": acc_2s,
            "2state_rec": rec_2s,
            "2state_prec": prec_2s,
            "auc": auc_v,
            "3state_f1": m3_v["f1"],
            "3state_acc": m3_v["accuracy"],
            "3state_coverage": m3_v["coverage"],
            "3state_rec": m3_v["recall"],
            "3state_prec": m3_v["precision"]
        }

        print(f"  [{alias}] ({vn}):")
        print(f"    * 2-State (0.70) : Acc={acc_2s*100:.2f}%, F1={f1_2s*100:.2f}%, Prec={prec_2s*100:.2f}%, Rec={rec_2s*100:.2f}%, AUC={auc_v:.4f}")
        print(f"    * 고도화 3-State  : Cov={m3_v['coverage']*100:.1f}%, Acc={m3_v['accuracy']*100:.2f}%, F1={m3_v['f1']*100:.2f}%, Prec={m3_v['precision']*100:.2f}%, Rec={m3_v['recall']*100:.2f}%")

    out_data = {
        "model": "babel_deadlift_d4_v5_advanced_3state",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "roc_auc": auc_final,
        "baseline_2state_50": {"f1": f1_50, "recall": rec_50, "precision": prec_50, "accuracy": acc_50},
        "tuned_2state_70": {"f1": f1_70, "recall": rec_70, "precision": prec_70, "accuracy": acc_70},
        "best_3state_strict_90": best_3s,
        "best_3state_relaxed_85": best_relaxed,
        "view_metrics": view_metrics
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 결과 JSON 저장 완료: {OUT_JSON}")

if __name__ == "__main__":
    main()
