# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_stgcn_transformer_residual.py
Purpose: v10.5 [해결책 5 (Phase 2 최종 딥러닝 확장)]
         14D 관절 채널 경량 시공간 신경망 (ST-GCN + Temporal Transformer) Residual Late Fusion
         - 5-Fold StratifiedGroupKFold (9인 피험자 엄격 분리) OOF 잔차 로짓(Δz_16, Δz_4) 산출
         - v10.4 챔피언 베이스라인 로짓에 부착하는 Residual 학습
         - 최적 게이팅 파라미터 g* OOF 그리드 탐색
         - 전수 2,605건 프로덕션 가중치 학습 및 저장 (stgcn_transformer_residual_weights.pth)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from stgcn_transformer_net import STGCNTransformerResidualNet
from joint_meta_16way_moe_net import FalseCleanAsymmetricLoss
from joint_meta_16way_net import decode_hybrid, STATE_TO_DEFECTS_MAP

TENSOR_PATH = r"D:\Data\피처_데이터\squat_14d_spatiotemporal_tensor_2605.npz"
BASE_LOGITS_PATH = os.path.join(CURRENT_DIR, "squat_v10_4_baseline_logits_2605.npz")
SAVE_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "stgcn_transformer_residual_weights.pth")
SAVE_OOF_PATH = os.path.join(CURRENT_DIR, "stgcn_transformer_oof_residuals.npz")
SAVE_META_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_5_meta.json")

print("=" * 115)
print("  [v10.5 Step 1] 14D 관절 시공간 텐서 및 v10.4 베이스라인 로짓 로드")
print("=" * 115)

tensor_d = np.load(TENSOR_PATH, allow_pickle=True)
base_d   = np.load(BASE_LOGITS_PATH, allow_pickle=True)

X_14d    = tensor_d['X_14d'].astype(np.float32)  # (2605, 14, 64, 33)
z_v10_4  = base_d['z_v10_4'].astype(np.float32) # (2605, 16)
p4_pred  = base_d['p4_pred'].astype(np.float32)  # (2605, 4)
views    = tensor_d['views']
subjects = tensor_d['subjects']
y_all    = tensor_d['y_all']
true_clean = tensor_d['true_clean']
N = len(true_clean)

y_16 = (y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1).astype(np.int64)
y_clean_np = (y_16 == 0).astype(np.float32)

class_counts = np.bincount(y_16, minlength=16)
smoothed_weights = (N / (16.0 * class_counts)) ** 0.35
smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32)

print(f"  • 총 표본 수: {N}건 | 14D 텐서 크기: {X_14d.shape} | 베이스라인 로짓: {z_v10_4.shape}")
print(f"  • 피험자 수: {len(np.unique(subjects))}명 (1~9번 전수) | 5대 전 시점 전수 반영")

# ============================================================
# [Step 2] 5-Fold StratifiedGroupKFold OOF 잔차 로짓 산출
# ============================================================
print("\n" + "=" * 115)
print("  [v10.5 Step 2] 5-Fold StratifiedGroupKFold (9인 피험자 격리) OOF 잔차 학습 및 수집")
print("=" * 115)

device = torch.device('cpu')
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_delta_z16 = np.zeros((N, 16), dtype=np.float32)
oof_delta_z4  = np.zeros((N, 4), dtype=np.float32)

M_matrix = torch.from_numpy(STATE_TO_DEFECTS_MAP).float()
fc_loss_fn = FalseCleanAsymmetricLoss(lambda_fc=3.0, gamma_pos=1.0, gamma_neg=2.0)
ce_loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
bce_loss_fn = nn.BCEWithLogitsLoss()

EPOCHS = 10
BATCH_SIZE = 128
LR = 0.003

t_start = time.time()

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_14d, y_16, groups=subjects)):
    val_subs = np.unique(subjects[val_idx])
    print(f"  --- Fold {fold + 1}/5 (검증 피험자: {val_subs}, val_N={len(val_idx)}) ---")

    # DataLoader 구성
    train_dataset = TensorDataset(
        torch.from_numpy(X_14d[train_idx]),
        torch.from_numpy(z_v10_4[train_idx]),
        torch.from_numpy(y_16[train_idx]),
        torch.from_numpy(y_all[train_idx]).float(),
        torch.from_numpy(y_clean_np[train_idx])
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    fold_model = STGCNTransformerResidualNet().to(device)
    optimizer = optim.AdamW(fold_model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

    fold_model.train()
    for ep in range(1, EPOCHS + 1):
        ep_loss = 0.0
        for batch_x, batch_zbase, batch_y16, batch_y4, batch_ycl in train_loader:
            optimizer.zero_grad()
            dz16, dz4 = fold_model(batch_x)

            # Residual Fused Logits
            z_fused_16 = batch_zbase + dz16

            # 1. 16-State Weighted CE
            l_ce = ce_loss_fn(z_fused_16, batch_y16)

            # 2. False-Clean Reverse-ASL
            l_fc = fc_loss_fn(z_fused_16[:, 0], batch_ycl)

            # 3. 4-Defect Marginal BCE
            l_bce = bce_loss_fn(dz4, batch_y4)

            # 4. Consistency MSE
            p16 = torch.softmax(z_fused_16, dim=-1)
            p4_from_16 = torch.matmul(p16, M_matrix)
            p4_from_bce = torch.sigmoid(dz4)
            l_cons = F.mse_loss(p4_from_16, p4_from_bce)

            # 5. Residual Shrinkage Regularization (L2 penalty on delta_z)
            l_reg = 0.05 * (torch.mean(dz16.pow(2)) + torch.mean(dz4.pow(2)))

            loss = 0.50 * l_ce + 0.20 * l_fc + 0.15 * l_bce + 0.10 * l_cons + l_reg
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(batch_x)

        scheduler.step()
        if ep % 5 == 0 or ep == EPOCHS:
            print(f"       Epoch {ep:2d}/{EPOCHS} | Loss: {ep_loss / len(train_idx):.4f}")

    # Val OOF 잔차 추론
    fold_model.eval()
    val_dataset = TensorDataset(torch.from_numpy(X_14d[val_idx]))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_dz16_list = []
    val_dz4_list  = []
    with torch.no_grad():
        for (vx,) in val_loader:
            vdz16, vdz4 = fold_model(vx)
            val_dz16_list.append(vdz16.numpy())
            val_dz4_list.append(vdz4.numpy())

    oof_delta_z16[val_idx] = np.vstack(val_dz16_list)
    oof_delta_z4[val_idx]  = np.vstack(val_dz4_list)

print(f"  [5-Fold OOF 잔차 수집 완료] 소요 시간: {time.time() - t_start:.2f}초")

# OOF 잔차 파일 저장
np.savez_compressed(
    SAVE_OOF_PATH,
    oof_delta_z16=oof_delta_z16,
    oof_delta_z4=oof_delta_z4,
    y_16=y_16,
    y_all=y_all,
    true_clean=true_clean
)
print(f"  [OOF 캐시 저장] -> {SAVE_OOF_PATH}")

# ============================================================
# [Step 3] 최적 잔차 게이팅 가중치 g* OOF 그리드 탐색
# ============================================================
print("\n" + "=" * 115)
print("  [v10.5 Step 3] 최적 잔차 게이트 가중치 g* OOF 그리드 탐색 (g in [0.00, 0.35])")
print("=" * 115)

# v10.4 판정 기준 파라미터 로드
QUALITY_PATH = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
C4_PATH      = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
q_data = np.load(QUALITY_PATH, allow_pickle=True)
c4_data = np.load(C4_PATH, allow_pickle=True)

q_overall_vec = q_data['X_quality_12d'][:, 2]
X_c4 = c4_data['X_c4_v5_25d']
wobble = X_c4[:, 3]; jerk = X_c4[:, 15]; f0_rate = X_c4[:, 24]
p_cal = base_d['p_cal']

H_max = np.log(16.0)
RISK_THRESH = 0.45
ETA = 0.05
TAU_CLEAN = 0.36

def evaluate_g(g_val):
    z_fused = z_v10_4 + g_val * oof_delta_z16
    p16_fused = np.exp(z_fused - np.max(z_fused, axis=-1, keepdims=True))
    p16_fused = p16_fused / np.sum(p16_fused, axis=-1, keepdims=True)

    # 동적 위험도 계산
    log_p = np.log(np.clip(p16_fused, 1e-8, 1.0))
    ent = -np.sum(p16_fused * log_p, axis=-1)
    sp = np.sort(p16_fused, axis=-1)
    margin = sp[:, -1] - sp[:, -2]
    risk = (ent / H_max) * (1.0 - margin)

    # 전수 판정 및 선택적 판정 실행
    pred_full = np.zeros(N, dtype=int)
    defs_full = np.zeros((N, 4), dtype=int)
    is_retake = np.zeros(N, dtype=bool)
    pred_sel = np.zeros(N, dtype=int)
    defs_sel = np.zeros((N, 4), dtype=int)

    for i in range(N):
        s, defs, is_cln, p_cln = decode_hybrid(p16_fused[i], p4_pred[i], eta=ETA, tau_clean=TAU_CLEAN)
        if is_cln:
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.38):
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        else:
            defs_full[i] = defs

    for i in range(N):
        if pred_full[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        if pred_full[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_full[i] = 0

    for i in range(N):
        if q_overall_vec[i] < 0.05:
            is_retake[i] = True; continue
        s, defs, is_cln, p_cln = decode_hybrid(p16_fused[i], p4_pred[i], eta=ETA, tau_clean=TAU_CLEAN)
        if risk[i] > RISK_THRESH and not is_cln:
            is_retake[i] = True; continue
        if is_cln:
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.38):
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        else:
            defs_sel[i] = defs

    for i in range(N):
        if is_retake[i]: continue
        if pred_sel[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        if pred_sel[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_sel[i] = 0

    # 지표 계산
    mask_sel = ~is_retake
    f1_sel = f1_score(true_clean[mask_sel], pred_sel[mask_sel], zero_division=0) * 100.0
    prec_sel = precision_score(true_clean[mask_sel], pred_sel[mask_sel], zero_division=0) * 100.0
    rec_sel = recall_score(true_clean[mask_sel], pred_sel[mask_sel], zero_division=0) * 100.0

    pred_16_sel = defs_sel[mask_sel, 0]*8 + defs_sel[mask_sel, 1]*4 + defs_sel[mask_sel, 2]*2 + defs_sel[mask_sel, 3]*1
    true_16_sel = y_all[mask_sel, 0]*8 + y_all[mask_sel, 1]*4 + y_all[mask_sel, 2]*2 + y_all[mask_sel, 3]*1
    emr_sel = float(np.mean(pred_16_sel == true_16_sel)) * 100.0

    ret_rate = float(is_retake.sum()) / N * 100.0
    fp_full = int(np.sum((true_clean == 0) & (pred_full == 1)))

    return f1_sel, prec_sel, rec_sel, emr_sel, ret_rate, fp_full

print(f"  {'g_val':>6s} | {'Clean F1':>9s} | {'Precision':>10s} | {'Recall':>8s} | {'4-EMR':>7s} | {'RETAKE%':>8s} | {'False Clean':>12s}")
print("  " + "-" * 75)

g_candidates = [0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
best_g = 0.00
best_emr = -1.0
best_f1 = -1.0
best_metrics = None

for g in g_candidates:
    f1, prec, rec, emr, ret, fp = evaluate_g(g)
    marker = ""
    # 최적 기준: Precision >= 70% 제약 하에서 4-Defect EMR 우선 최대화
    if prec >= 70.0 and emr > best_emr:
        best_emr = emr
        best_f1 = f1
        best_g = g
        best_metrics = (f1, prec, rec, emr, ret, fp)
        marker = " ★"
    print(f"  {g:6.2f} | {f1:8.2f}% | {prec:9.2f}% | {rec:7.2f}% | {emr:6.2f}% | {ret:7.2f}% | {fp:8d}건{marker}")

print("\n" + "=" * 115)
print(f"  ★ 최적 잔차 게이트 가중치: g* = {best_g:.2f}")
print(f"    → Clean F1:        {best_metrics[0]:.2f}%")
print(f"    → Clean Precision: {best_metrics[1]:.2f}% (70% 하드 제약 완벽 준수)")
print(f"    → Clean Recall:    {best_metrics[2]:.2f}%")
print(f"    → 4-Defect EMR:    {best_metrics[3]:.2f}%")
print(f"    → RETAKE 비율:     {best_metrics[4]:.2f}%")
print(f"    → False Clean:     {best_metrics[5]}건")
print("=" * 115)

# ============================================================
# [Step 4] 전체 2,605건 프로덕션 가중치 학습 및 저장
# ============================================================
print("\n" + "=" * 115)
print("  [v10.5 Step 4] 전체 2,605건 프로덕션 가중치 학습 및 저장")
print("=" * 115)

full_dataset = TensorDataset(
    torch.from_numpy(X_14d),
    torch.from_numpy(z_v10_4),
    torch.from_numpy(y_16),
    torch.from_numpy(y_all).float(),
    torch.from_numpy(y_clean_np)
)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

prod_model = STGCNTransformerResidualNet().to(device)
prod_opt = optim.AdamW(prod_model.parameters(), lr=LR, weight_decay=1e-3)
prod_sched = optim.lr_scheduler.CosineAnnealingLR(prod_opt, T_max=EPOCHS, eta_min=1e-4)

prod_model.train()
for ep in range(1, EPOCHS + 1):
    ep_loss = 0.0
    for batch_x, batch_zbase, batch_y16, batch_y4, batch_ycl in full_loader:
        prod_opt.zero_grad()
        dz16, dz4 = prod_model(batch_x)
        z_fused_16 = batch_zbase + dz16

        l_ce = ce_loss_fn(z_fused_16, batch_y16)
        l_fc = fc_loss_fn(z_fused_16[:, 0], batch_ycl)
        l_bce = bce_loss_fn(dz4, batch_y4)
        p16 = torch.softmax(z_fused_16, dim=-1)
        l_cons = F.mse_loss(torch.matmul(p16, M_matrix), torch.sigmoid(dz4))
        l_reg = 0.05 * (torch.mean(dz16.pow(2)) + torch.mean(dz4.pow(2)))

        loss = 0.50 * l_ce + 0.20 * l_fc + 0.15 * l_bce + 0.10 * l_cons + l_reg
        loss.backward()
        prod_opt.step()
        ep_loss += loss.item() * len(batch_x)

    prod_sched.step()
    if ep % 5 == 0 or ep == EPOCHS:
        print(f"  Production Epoch {ep:2d}/{EPOCHS} | Loss: {ep_loss / N:.4f}")

torch.save(prod_model.state_dict(), SAVE_WEIGHTS_PATH)
print(f"  [프로덕션 가중치 저장 완료] -> {SAVE_WEIGHTS_PATH}")

# 메타데이터 생성 및 저장
meta_v10_5 = {
    "version": "v10.5",
    "status": "CANDIDATE_EVALUATION",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "architecture": "ST-GCN (2-Layer) + Temporal Transformer (2-Layer) Residual Late Fusion on v10.4",
    "in_channels": 14,
    "num_joints": 33,
    "num_frames": 64,
    "hidden_dim": 64,
    "num_tf_layers": 2,
    "nhead": 4,
    "dropout": 0.25,
    "total_parameters": 133749,
    "g_residual_weight": float(best_g),
    "eta": ETA,
    "tau_clean": TAU_CLEAN,
    "risk_threshold": RISK_THRESH,
    "oof_metrics": {
        "clean_f1": float(best_metrics[0]),
        "clean_precision": float(best_metrics[1]),
        "clean_recall": float(best_metrics[2]),
        "four_defect_emr": float(best_metrics[3]),
        "retake_rate": float(best_metrics[4]),
        "coverage": float(100.0 - best_metrics[4]),
        "false_clean_count": int(best_metrics[5])
    }
}

with open(SAVE_META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta_v10_5, f, indent=2, ensure_ascii=False)
print(f"  [메타데이터 저장 완료] -> {SAVE_META_PATH}")
print("=" * 115)
