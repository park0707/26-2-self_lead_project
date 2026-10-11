# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반 (Knee-Hip Simultaneous Extension Synchrony)
Script: run_phase2_tcn.py
Purpose: Phase 2: 64프레임 12채널 경량 1D-TCN 시계열 딥러닝 스트림
         - Class-Balanced Asymmetric Focal Loss (γ+=1.0, γ-=3.0, λ-=0.5)
         - 5-Fold 피험자 엄격 격리 OOF 학습 및 확률 저장 (d4_oof_probs_tcn.npy)
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_NPZ = os.path.join(CURRENT_DIR, "d4_v4_features_cache.npz")
OUT_OOF_NPY = os.path.join(CURRENT_DIR, "d4_oof_probs_tcn.npy")
OUT_JSON = os.path.join(CURRENT_DIR, "d4_phase2_tcn_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=3.0, lambda_neg=0.5, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.lambda_neg = lambda_neg
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()
        loss_pos = -targets * torch.pow(1.0 - probs, self.gamma_pos) * torch.log(probs + self.eps)
        loss_neg = -(1.0 - targets) * self.lambda_neg * torch.pow(probs, self.gamma_neg) * torch.log(1.0 - probs + self.eps)
        return torch.mean(loss_pos + loss_neg)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.10):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        return out + res


class FastKinematicTCN(nn.Module):
    def __init__(self, in_features: int = 12, hidden_dim: int = 16):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(in_features, hidden_dim, kernel_size=3, dilation=1),
            TemporalBlock(hidden_dim, hidden_dim * 2, kernel_size=3, dilation=2),
            TemporalBlock(hidden_dim * 2, hidden_dim * 2, kernel_size=3, dilation=4)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.tcn(x)
        pooled = F.adaptive_max_pool1d(feat, 1).squeeze(-1)
        logits = self.head(pooled)
        return logits


def train_single_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                      X_va: np.ndarray, y_va: np.ndarray,
                      seed: int = 42, epochs: int = 15) -> np.ndarray:
    torch.manual_seed(seed)
    model = FastKinematicTCN(in_features=12, hidden_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=3.0, lambda_neg=0.5)

    tx = torch.tensor(X_tr, dtype=torch.float32)
    ty = torch.tensor(y_tr, dtype=torch.float32)
    vx = torch.tensor(X_va, dtype=torch.float32)

    n_samples = len(tx)
    batch_size = 64

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = tx[idx], ty[idx]
            optimizer.zero_grad()
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        v_logits = model(vx).squeeze(-1)
        v_probs = torch.sigmoid(v_logits).numpy()

    return v_probs


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 d4 Phase 2: 64프레임 Kinematic 1D-TCN")
    print("  - 데이터셋: 770개 시퀀스 (피험자 1~9번 전원, 5대 전 시점 전수)")
    print("  - 검증 규격: StratifiedGroupKFold (5-Fold 피험자 엄격 격리)")
    print("=" * 80, flush=True)

    data = np.load(CACHE_NPZ, allow_pickle=True)
    X = data["X_tcn"]  # (770, 12, 64)
    y = data["y"]
    subjects = data["subjects"]
    views = data["views"]

    print(f"[*] 캐시 데이터 로드 완료: X={X.shape}, y 결함={np.sum(y==1)}개, 정상={np.sum(y==0)}개", flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        f_t = time.time()
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        va_subs = np.unique(subjects[val_idx]).tolist()

        val_p = train_single_fold(X_tr, y_tr, X_va, y_va, seed=42 + fold, epochs=15)
        oof_probs[val_idx] = val_p

        val_pred = (val_p >= 0.50).astype(int)
        f1_fold = f1_score(y_va, val_pred, zero_division=0)
        acc_fold = accuracy_score(y_va, val_pred)
        auc_fold = roc_auc_score(y_va, val_p) if len(np.unique(y_va)) > 1 else 0.5
        print(f"  - Fold {fold}/5 (검증 피험자={va_subs}, N={len(y_va)}, 소요 {time.time()-f_t:.1f}s): Acc={acc_fold*100:.2f}%, F1={f1_fold*100:.2f}%, AUC={auc_fold:.4f}", flush=True)

    print(f"\n[+] 1D-TCN 5-Fold 학습 완료! 총 소요 시간: {time.time()-t0:.1f}초", flush=True)
    np.save(OUT_OOF_NPY, oof_probs)
    print(f"[+] TCN OOF 확률 저장 완료: {OUT_OOF_NPY}", flush=True)

    # 전체 OOF 성능 (0.50)
    pred_50 = (oof_probs >= 0.50).astype(int)
    f1_50 = float(f1_score(y, pred_50, zero_division=0))
    acc_50 = float(accuracy_score(y, pred_50))
    prec_50 = float(precision_score(y, pred_50, zero_division=0))
    rec_50 = float(recall_score(y, pred_50, zero_division=0))
    auc_total = float(roc_auc_score(y, oof_probs))
    cm = confusion_matrix(y, pred_50)

    print("\n" + "=" * 80)
    print("  [Phase 2 완료: 64프레임 Kinematic 1D-TCN OOF 종합 성능 (임계치 0.50)]")
    print(f"  - Accuracy  : {acc_50*100:.2f}%")
    print(f"  - Precision : {prec_50*100:.2f}%")
    print(f"  - Recall    : {rec_50*100:.2f}%")
    print(f"  - ★ F1-Score : {f1_50*100:.2f}%")
    print(f"  - ★ ROC-AUC  : {auc_total:.4f}")
    print(f"  - 혼동 행렬 (Confusion Matrix):\n{cm}")
    print("=" * 80, flush=True)

    # 공식 5대 시점 전수 감사
    print("\n  [공식 5대 시점 전수 감사]")
    view_res = {}
    for vn in sorted(np.unique(views).tolist()):
        alias = VIEW_ALIAS_MAP.get(vn, vn)
        mask = (views == vn)
        y_v = y[mask]
        p_v = oof_probs[mask]
        pred_v = (p_v >= 0.50).astype(int)

        f1_v = float(f1_score(y_v, pred_v, zero_division=0))
        acc_v = float(accuracy_score(y_v, pred_v))
        auc_v = float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else 0.5
        view_res[vn] = {"alias": alias, "f1": f1_v, "acc": acc_v, "auc": auc_v, "n": len(y_v)}
        print(f"    * [{alias}] ({vn}): Acc={acc_v*100:.2f}%, F1={f1_v*100:.2f}%, AUC={auc_v:.4f}")

    results = {
        "phase": 2,
        "model": "64F_Kinematic_1D_TCN",
        "accuracy": acc_50,
        "precision": prec_50,
        "recall": rec_50,
        "f1": f1_50,
        "roc_auc": auc_total,
        "view_metrics": view_res
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
