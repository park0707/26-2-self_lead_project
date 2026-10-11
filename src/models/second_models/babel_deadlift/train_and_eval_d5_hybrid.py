# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: train_and_eval_d5_hybrid.py
Purpose: d5 v4.0 - 32D 생체역학 GBDT 앙상블 + 64F 시계열 PyTorch TCN(Temporal Convolutional Network)
         하이브리드 메타 앙상블 구축 및 5-Fold 피험자 완전 격리 실측 평가
"""

import os
import sys
import json
import glob
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d5_model_v3 import (
    BabelDeadliftD5ModelV3,
    extract_d5_v3_features_from_seq
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d5_model_v4_hybrid_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs=99, num_channels=[64, 64, 32], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (B, C, T) -> (B, 99, 64)
        out = self.network(x)
        pooled = self.pool(out).squeeze(-1)
        logits = self.fc(pooled).squeeze(-1)
        return logits


def load_raw_and_tabular_dataset():
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 시퀀스 로드: {len(npz_files)}개")

    raw_seqs = []
    tab_feats = []
    y_list = []
    groups_list = []
    views_list = []

    for npz_path in npz_files:
        bn = os.path.basename(npz_path).replace(".npz", "")
        parts = bn.split("_")
        vn = parts[-1]
        folder = "_".join(parts[:-1])

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder}.json")
        if not os.path.exists(label_p):
            continue

        with open(label_p, "r", encoding="utf-8", errors="ignore") as f:
            d_json = json.load(f)
        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}
        y_d5 = 0 if conds.get("바벨 궤적과 몸 밀착", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder:
                sub = d
                break

        data = np.load(npz_path)
        kps_64f = data["kps_33_3d_64f"][:, :, :3]  # (64, 33, 3)

        # 32D GBDT 피처
        feats_32 = extract_d5_v3_features_from_seq(data["kps_33_3d_64f"], view_name=vn)

        # 64F 시계열 (T=64, C=99)
        seq_flat = kps_64f.reshape(64, -1).astype(np.float32)

        raw_seqs.append(seq_flat)
        tab_feats.append(feats_32)
        y_list.append(y_d5)
        groups_list.append(sub)
        views_list.append(vn)

    X_raw = np.array(raw_seqs, dtype=np.float32)  # (770, 64, 99)
    X_tab = np.array(tab_feats, dtype=np.float32) # (770, 32)
    y = np.array(y_list, dtype=int)
    groups = np.array(groups_list, dtype=int)
    views = np.array(views_list)

    print(f"[+] 로드 완료: Raw Shape {X_raw.shape}, Tabular Shape {X_tab.shape}, y={y.shape}")
    return X_raw, X_tab, y, groups, views


def train_tcn_model(X_train_raw, y_train, X_val_raw, epochs=12, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalConvNet(num_inputs=99, num_channels=[48, 48, 24], kernel_size=3, dropout=0.2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    # (B, T, C) -> (B, C, T)
    x_tr_t = torch.tensor(X_train_raw.transpose(0, 2, 1), dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    ds = TensorDataset(x_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    x_val_t = torch.tensor(X_val_raw.transpose(0, 2, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        val_logits = model(x_val_t).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    return val_probs


def run_hybrid_eval():
    print("=" * 90)
    print("  [Step 1] d5 v4.0 GBDT(32D) + 64F 시계열 TCN 하이브리드 메타 앙상블 학습")
    print("=" * 90)

    X_raw, X_tab, y, groups, views = load_raw_and_tabular_dataset()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    gbdt_oof_probs = np.zeros(len(y), dtype=np.float32)
    tcn_oof_probs = np.zeros(len(y), dtype=np.float32)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tab, y, groups)):
        val_subs = [int(s) for s in sorted(list(set(groups[val_idx])))]
        print(f"\n--- [Fold {fold+1}/5] 피험자 {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건) ---")

        # 1. GBDT 학습 및 예측
        gbdt = BabelDeadliftD5ModelV3()
        gbdt.fit(X_tab[train_idx], y[train_idx])
        gbdt_oof_probs[val_idx] = gbdt.predict_proba(X_tab[val_idx])[:, 1]

        # 2. TCN 학습 및 예측
        tcn_probs = train_tcn_model(X_raw[train_idx], y[train_idx], X_raw[val_idx], epochs=14)
        tcn_oof_probs[val_idx] = tcn_probs

        print(f"  > Fold {fold+1}: GBDT AUC={roc_auc_score(y[val_idx], gbdt_oof_probs[val_idx]):.4f}, TCN AUC={roc_auc_score(y[val_idx], tcn_oof_probs[val_idx]):.4f}")

    print(f"\n[+] 5-Fold 전체 OOF 수집 완료! (경과시간: {time.time()-t0:.1f}초)")

    # 3. 하이브리드 소프트 보팅 가중치 최적화 (GBDT w_g + TCN w_t)
    best_w = 0.70
    best_hybrid_f1 = -1.0
    for w_g in np.arange(0.50, 0.95, 0.05):
        w_t = 1.0 - w_g
        comb_probs = gbdt_oof_probs * w_g + tcn_oof_probs * w_t
        comb_preds = (comb_probs >= 0.50).astype(int)
        f1 = f1_score(y, comb_preds)
        if f1 > best_hybrid_f1:
            best_hybrid_f1 = f1
            best_w = float(w_g)

    hybrid_probs = gbdt_oof_probs * best_w + tcn_oof_probs * (1.0 - best_w)
    hybrid_preds = (hybrid_probs >= 0.50).astype(int)

    acc = accuracy_score(y, hybrid_preds)
    prec = precision_score(y, hybrid_preds, zero_division=0)
    rec = recall_score(y, hybrid_preds, zero_division=0)
    f1 = f1_score(y, hybrid_preds, zero_division=0)
    auc = roc_auc_score(y, hybrid_probs)
    cm = confusion_matrix(y, hybrid_preds).tolist()

    print("\n" + "=" * 90)
    print(f"  [d5 v4.0 하이브리드 전체 OOF 종합 결과 (GBDT {best_w*100:.0f}% + TCN {(1.0-best_w)*100:.0f}%)]")
    print(f"  - Accuracy : {acc*100:.2f}% (v1.0 대비 +{(acc - 0.7714)*100:.2f}%p)")
    print(f"  - Precision: {prec*100:.2f}%")
    print(f"  - Recall   : {rec*100:.2f}%")
    print(f"  - F1-Score : {f1*100:.2f}% (v1.0 대비 +{(f1 - 0.7615)*100:.2f}%p)")
    print(f"  - ROC-AUC  : {auc:.4f} (v1.0 대비 +{(auc - 0.8638):.4f})")
    print(f"  - 혼동 행렬: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print("=" * 90)

    # 공식 5대 시점 전수 감사
    print("\n" + "=" * 90)
    print("  [Step 2] 공식 5대 카메라 시점 전수 성능 감사 (Multi-View Audit)")
    print("=" * 90)
    view_audit = {}
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        alias = VIEW_ALIAS_MAP[vn]
        v_mask = (views == vn)
        v_y = y[v_mask]
        v_probs = hybrid_probs[v_mask]
        v_preds = (v_probs >= 0.50).astype(int)

        v_acc = accuracy_score(v_y, v_preds)
        v_prec = precision_score(v_y, v_preds, zero_division=0)
        v_rec = recall_score(v_y, v_preds, zero_division=0)
        v_f1 = f1_score(v_y, v_preds, zero_division=0)
        v_auc = roc_auc_score(v_y, v_probs)

        print(f"  [{alias}] 샘플={len(v_y)}개 | Acc={v_acc*100:.2f}% | F1={v_f1*100:.2f}% | Prec={v_prec*100:.2f}% | Rec={v_rec*100:.2f}% | AUC={v_auc:.4f}")
        view_audit[vn] = {
            "alias": alias,
            "sample_count": int(np.sum(v_mask)),
            "accuracy": float(v_acc),
            "precision": float(v_prec),
            "recall": float(v_rec),
            "f1_score": float(v_f1),
            "roc_auc": float(v_auc)
        }

    # 3-State 선택적 판정 파레토 분석
    print("\n" + "=" * 90)
    print("  [Step 3] 3-State 선택적 판정 파레토 분석")
    print("=" * 90)
    pareto_results = []
    target_coverages = [0.90, 0.88, 0.85, 0.80, 0.75]

    for target_cov in target_coverages:
        best_cand = None
        best_cand_f1 = -1.0
        for tau_norm in np.arange(0.20, 0.48, 0.01):
            for tau_def in np.arange(0.50, 0.80, 0.01):
                if tau_norm >= tau_def:
                    continue
                is_retained = (hybrid_probs < tau_norm) | (hybrid_probs >= tau_def)
                cov = np.mean(is_retained)
                if cov < target_cov:
                    continue
                retained_y = y[is_retained]
                retained_pred = (hybrid_probs[is_retained] >= tau_def).astype(int)
                if len(retained_y) == 0:
                    continue
                c_f1 = f1_score(retained_y, retained_pred, zero_division=0)
                if c_f1 > best_cand_f1:
                    best_cand_f1 = c_f1
                    c_acc = accuracy_score(retained_y, retained_pred)
                    c_prec = precision_score(retained_y, retained_pred, zero_division=0)
                    c_rec = recall_score(retained_y, retained_pred, zero_division=0)
                    best_cand = {
                        "target_coverage": target_cov,
                        "actual_coverage": float(cov),
                        "tau_norm": float(tau_norm),
                        "tau_def": float(tau_def),
                        "retained_count": int(np.sum(is_retained)),
                        "retake_count": int(np.sum(~is_retained)),
                        "accuracy": float(c_acc),
                        "precision": float(c_prec),
                        "recall": float(c_rec),
                        "f1_score": float(c_f1)
                    }
        if best_cand:
            pareto_results.append(best_cand)
            print(f"  [커버리지 >={target_cov*100:.0f}%] 실제={best_cand['actual_coverage']*100:.2f}% | 보류={best_cand['retake_count']}건 | [tau_norm={best_cand['tau_norm']:.2f}, tau_def={best_cand['tau_def']:.2f}] -> Acc={best_cand['accuracy']*100:.2f}%, F1={best_cand['f1_score']*100:.2f}%, Prec={best_cand['precision']*100:.2f}%, Rec={best_cand['recall']*100:.2f}%")

    full_results = {
        "model_name": "d5_model_v4_hybrid.py",
        "best_gbdt_weight": best_w,
        "best_tcn_weight": 1.0 - best_w,
        "total_samples": len(y),
        "oof_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm
        },
        "view_audit": view_audit,
        "pareto_results": pareto_results
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] d5 v4.0 하이브리드 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")
    return full_results


if __name__ == "__main__":
    run_hybrid_eval()
