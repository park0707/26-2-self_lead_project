# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_joint_meta_v10_0.py
Purpose: v10.0 [해결책 1] False-Clean Asymmetric Loss + OOF Precision-Constrained Threshold
         - 기반: train_joint_meta_16way_v8_6.py (v9.0 베이스라인) 100% 계승
         - 변경 1: MoEConsistencyLossV10 (Reverse-ASL 추가)
         - 변경 2: 5-Fold OOF p_clean 수집 완전 구현
         - 변경 3: OOF Precision>=70% 제약 하 Recall 최대화로 tau_clean 탐색
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from joint_meta_16way_moe_net import JointMeta16WayMoENet, MoEConsistencyLossV10
from joint_meta_16way_net import decode_hybrid, STATE_TO_DEFECTS_MAP, idx_to_defect_tuple

META_PATH    = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH      = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")

SAVE_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_0_weights.pth")
SAVE_META_PATH    = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_0_meta.json")
SAVE_OOF_PATH     = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_0_oof_clean_probs.npy")

print("=" * 115)
print("  [v10.0 Step 1] Level 1 온도 보정 및 39D MoE 메타 특성 행렬 구성")
print("=" * 115)

data    = np.load(META_PATH,    allow_pickle=True)
c4_data = np.load(C4_PATH,     allow_pickle=True)
q_data  = np.load(QUALITY_PATH, allow_pickle=True)
k_data  = np.load(KINETIC_PATH, allow_pickle=True)

probs_defect = data['probs_defect'].copy()
views    = data['views']
subjects = data['subjects']
y_all    = data['y_all']
true_clean = data['true_clean']
N = len(true_clean)

with open(TEMP_PARAM_PATH, 'r', encoding='utf-8') as f:
    temp_params = json.load(f)

T_c1 = temp_params['c1']['global_T']
T_c3 = temp_params['c3']['global_T']
T_c4 = temp_params['c4']['global_T']
view_T_c2 = temp_params['c2']['view_T']

eps = 1e-7
p_clip = np.clip(probs_defect, eps, 1.0 - eps)
logits = np.log(p_clip / (1.0 - p_clip))

p_calibrated = np.zeros_like(p_clip)
p_calibrated[:, 0] = 1.0 / (1.0 + np.exp(-logits[:, 0] / T_c1))
p_calibrated[:, 2] = 1.0 / (1.0 + np.exp(-logits[:, 2] / T_c3))
p_calibrated[:, 3] = 1.0 / (1.0 + np.exp(-logits[:, 3] / T_c4))
for v, T_v in view_T_c2.items():
    mask = (views == v)
    p_calibrated[mask, 1] = 1.0 / (1.0 + np.exp(-logits[mask, 1] / T_v))

X_c4 = c4_data['X_c4_v5_25d']
jerk_norm   = np.clip(X_c4[:, 15:16] / 200.0,  0.0, 2.5)
wobble_norm = np.clip(X_c4[:, 3:4]  / 0.015,   0.0, 2.5)
ydelta_norm = np.clip(X_c4[:, 6:7]  / 0.020,  -1.0, 2.5)
toe_norm    = np.clip(X_c4[:, 1:2]  / 0.020,   0.0, 2.5)
f0_norm     = np.clip(X_c4[:, 24:25]/ 0.50,    0.0, 2.0)
phys_feats  = np.hstack([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm])

margins_cal = np.abs(p_calibrated - 0.5)

view_map = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
view_indices = np.array([view_map[v] for v in views])
view_onehot  = np.eye(5, dtype=np.float32)[view_indices]

X_quality_raw = q_data['X_quality_12d']
q_bone      = X_quality_raw[:, 0:1]
q_jitter    = X_quality_raw[:, 1:2]
q_overall_feat = X_quality_raw[:, 2:3]
bone_instab = np.clip(X_quality_raw[:, 3:4]  / 0.20,          0.0, 2.5)
jitter_s    = np.clip(X_quality_raw[:, 4:5]  / 0.001,          0.0, 2.5)
nadir_s     = X_quality_raw[:, 5:6]
rom_s       = np.clip(X_quality_raw[:, 6:7]  / 0.015,          0.0, 2.5)
sh_w_s      = np.clip((X_quality_raw[:, 7:8]  - 0.65) / 0.15, -2.5, 2.5)
hip_w_s     = np.clip((X_quality_raw[:, 8:9]  - 0.42) / 0.10, -2.5, 2.5)
z_knee_s    = np.clip(X_quality_raw[:, 9:10]  / 0.40,          0.0, 2.5)
z_ank_s     = np.clip(X_quality_raw[:, 10:11] / 0.45,          0.0, 2.5)
head_vis_s  = np.clip((X_quality_raw[:, 11:12] - 0.27) / 0.05, -2.5, 2.5)

quality_feats = np.hstack([
    q_bone, q_jitter, q_overall_feat, bone_instab, jitter_s,
    nadir_s, rom_s, sh_w_s, hip_w_s, z_knee_s, z_ank_s, head_vis_s
]).astype(np.float32)

X_kinetic_raw = k_data['X_kinetic_9d']
c_fk   = X_kinetic_raw[:, 0:1]
c_kt   = X_kinetic_raw[:, 1:2]
c_tn   = X_kinetic_raw[:, 2:3]
c_ft   = X_kinetic_raw[:, 3:4]
d_valg = np.clip(X_kinetic_raw[:, 4:5]  / 0.40,          -2.5, 2.5)
n_tilt = np.clip(X_kinetic_raw[:, 5:6]  / 0.50,           0.0, 2.5)
n_valg = np.clip((X_kinetic_raw[:, 6:7] - 1.40) / 0.35,  -2.5, 2.5)
a_heel = np.clip(X_kinetic_raw[:, 7:8]  / 0.60,           0.0, 2.5)
p_lag  = np.clip(X_kinetic_raw[:, 8:9]  / 0.40,          -2.5, 2.5)

kinetic_feats = np.hstack([
    c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag
]).astype(np.float32)

X_meta_39d = np.hstack([
    p_calibrated, margins_cal, view_onehot, phys_feats, quality_feats, kinetic_feats
]).astype(np.float32)

y_16 = (y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1).astype(np.int64)
y_clean_np = (y_16 == 0).astype(np.float32)  # [v10.0 신규] Clean 여부 라벨

class_counts = np.bincount(y_16, minlength=16)
smoothed_weights = (N / (16.0 * class_counts)) ** 0.35
smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32)

print(f"  총 표본: {N}건 | Clean: {int(y_clean_np.sum())}건 ({y_clean_np.mean()*100:.1f}%) | 39D MoE 특성 행렬 구축 완료")

# ============================================================
# [v10.0 Step 2] 5-Fold OOF 완전 구현: p_clean 수집 + Precision 제약 tau 탐색
# ============================================================
print("\n" + "=" * 115)
print("  [v10.0 Step 2] 5-Fold StratifiedGroupKFold OOF p_clean 수집 및 tau_clean 탐색")
print("=" * 115)

sgkf   = StratifiedGroupKFold(n_splits=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

oof_p_clean = np.zeros(N, dtype=np.float32)  # 전체 OOF Clean 확률 수집 배열

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_meta_39d, y_16, groups=subjects)):
    val_subs = np.unique(subjects[val_idx])
    print(f"  --- Fold {fold + 1}/5 (검증 피험자: {val_subs}, val_N={len(val_idx)}) ---")

    X_tr = torch.tensor(X_meta_39d[train_idx], device=device)
    X_vl = torch.tensor(X_meta_39d[val_idx],   device=device)
    y16_tr = torch.tensor(y_16[train_idx],      device=device)
    y4_tr  = torch.tensor(y_all[train_idx],     dtype=torch.float32, device=device)
    y_cl_tr = torch.tensor(y_clean_np[train_idx], device=device)  # [v10.0 신규]

    fold_model = JointMeta16WayMoENet(in_features=39, hidden_dim=48, dropout_rate=0.20).to(device)
    fold_loss_fn = MoEConsistencyLossV10(
        weight_ce=0.50, weight_clean=0.20,
        weight_bce=0.20, weight_consistency=0.05, weight_gate_balance=0.05,
        lambda_fc=3.0, gamma_pos=1.0, gamma_neg=2.0,
        class_weights_16=weights_tensor.to(device)
    ).to(device)
    fold_opt   = optim.AdamW(fold_model.parameters(), lr=0.005, weight_decay=1e-3)
    fold_sched = optim.lr_scheduler.CosineAnnealingLR(fold_opt, T_max=120, eta_min=1e-4)

    fold_model.train()
    for epoch in range(1, 121):
        fold_opt.zero_grad()
        l16, l4, gate_w = fold_model(X_tr)
        loss, _ = fold_loss_fn(l16, l4, y16_tr, y4_tr, gate_w, y_cl_tr)  # [v10.0: y_cl_tr 추가]
        loss.backward()
        fold_opt.step()
        fold_sched.step()

    # Val fold OOF p_clean 수집
    fold_model.eval()
    with torch.no_grad():
        l16_val, _, _ = fold_model(X_vl)
        p16_val = torch.softmax(l16_val / 1.20, dim=-1).cpu().numpy()  # T_meta=1.20 동일 적용
        oof_p_clean[val_idx] = p16_val[:, 0]  # state=0(Clean) 확률 수집

    val_clean_f1 = f1_score(y_clean_np[val_idx], (oof_p_clean[val_idx] >= 0.38).astype(int), zero_division=0)
    print(f"       Fold {fold+1} Val Clean F1 (tau=0.38): {val_clean_f1*100:.2f}%")

# OOF p_clean 저장
np.save(SAVE_OOF_PATH, oof_p_clean)
print(f"\n  [OOF p_clean 저장] -> {SAVE_OOF_PATH}")

# ============================================================
# [v10.0 Step 3] Precision >= 70% 제약 하 tau_clean 탐색
# ============================================================
print("\n" + "=" * 115)
print("  [v10.0 Step 3] OOF Precision-Constrained tau_clean 탐색 (Precision >= 70% 제약)")
print("=" * 115)

best_tau_clean = 0.38  # fallback 기본값
best_recall    = -1.0
best_precision = 0.0
best_f1        = 0.0

results_table = []
for tau in np.linspace(0.20, 0.95, 151):
    pred_clean = (oof_p_clean >= tau).astype(int)
    prec = precision_score(y_clean_np.astype(int), pred_clean, zero_division=0)
    rec  = recall_score(y_clean_np.astype(int), pred_clean, zero_division=0)
    f1   = f1_score(y_clean_np.astype(int), pred_clean, zero_division=0)
    results_table.append((tau, prec, rec, f1))

    if prec >= 0.70:  # 하드 제약: Precision 70% 이상
        if rec > best_recall:  # 그 조건에서 Recall 최대화
            best_recall    = rec
            best_tau_clean = tau
            best_precision = prec
            best_f1        = f1

print(f"  * 최적 tau_clean: {best_tau_clean:.3f}")
print(f"    -> OOF Clean Precision: {best_precision*100:.2f}%")
print(f"    -> OOF Clean Recall:    {best_recall*100:.2f}%")
print(f"    -> OOF Clean F1:        {best_f1*100:.2f}%")
print(f"  기존 v9.0 tau_clean: 0.380 (비교 기준)")

# 주요 tau 구간 출력
print("\n  [tau별 OOF Precision/Recall/F1 주요 구간]")
print(f"  {'tau':>6s} | {'Precision':>10s} | {'Recall':>8s} | {'F1':>7s}")
print("  " + "-" * 40)
for tau, prec, rec, f1 in results_table:
    if abs(tau - round(tau, 1)) < 0.01 or (prec >= 0.68 and prec <= 0.75):  # 주요 구간만
        marker = "  *" if abs(tau - best_tau_clean) < 0.005 else ""
        print(f"  {tau:6.3f} | {prec*100:9.2f}% | {rec*100:7.2f}% | {f1*100:6.2f}%{marker}")

# ============================================================
# [v10.0 Step 4] 전체 2,605건 프로덕션 가중치 학습 및 저장
# ============================================================
print("\n" + "=" * 115)
print("  [v10.0 Step 4] 전체 2,605건 v10.0 프로덕션 가중치 학습 및 저장")
print("=" * 115)

prod_model = JointMeta16WayMoENet(in_features=39, hidden_dim=48, dropout_rate=0.20).to(device)
prod_loss_fn = MoEConsistencyLossV10(
    weight_ce=0.50, weight_clean=0.20,
    weight_bce=0.20, weight_consistency=0.05, weight_gate_balance=0.05,
    lambda_fc=3.0, gamma_pos=1.0, gamma_neg=2.0,
    class_weights_16=weights_tensor.to(device)
).to(device)

prod_opt   = optim.AdamW(prod_model.parameters(), lr=0.005, weight_decay=1e-3)
prod_sched = optim.lr_scheduler.CosineAnnealingLR(prod_opt, T_max=120, eta_min=1e-4)

x_full     = torch.tensor(X_meta_39d, device=device)
y16_full   = torch.tensor(y_16, device=device)
y4_full    = torch.tensor(y_all, dtype=torch.float32, device=device)
y_cl_full  = torch.tensor(y_clean_np, device=device)  # [v10.0 신규]

prod_model.train()
for epoch in range(1, 121):
    prod_opt.zero_grad()
    l16, l4, gate_w = prod_model(x_full)
    loss, loss_dict = prod_loss_fn(l16, l4, y16_full, y4_full, gate_w, y_cl_full)
    loss.backward()
    prod_opt.step()
    prod_sched.step()
    if epoch % 40 == 0:
        print(f"  Epoch {epoch:3d}/120 | Loss: {loss_dict['loss_total']:.4f} "
              f"(CE:{loss_dict['loss_ce']:.4f} FC:{loss_dict['loss_fc']:.4f} "
              f"BCE:{loss_dict['loss_bce']:.4f})")

torch.save(prod_model.state_dict(), SAVE_WEIGHTS_PATH)
print(f"  [가중치 저장 완료] -> {SAVE_WEIGHTS_PATH}")

cfg_v10 = {
    'version': 'v10.0',
    'architecture': 'View-Conditioned Mixture-of-Experts (MoE) + False-Clean Asymmetric Loss',
    'in_features': 39,
    'num_experts': 3,
    'T_meta': 1.20,
    'eta': 0.10,
    'tau_clean': float(best_tau_clean),  # [v10.0: OOF Precision-Constrained 탐색 결과]
    'tau_clean_v9_baseline': 0.38,       # 비교용 기존값 보관
    'q_min': 0.05,
    'deadband_low': 0.32,
    'deadband_high': float(best_tau_clean),  # deadband_high도 tau_clean과 일치
    'loss': {
        'weight_ce': 0.50,
        'weight_clean': 0.20,
        'weight_bce': 0.20,
        'weight_consistency': 0.05,
        'weight_gate_balance': 0.05,
        'lambda_fc': 3.0,
        'gamma_pos': 1.0,
        'gamma_neg': 2.0
    }
}

with open(SAVE_META_PATH, 'w', encoding='utf-8') as f:
    json.dump(cfg_v10, f, indent=2, ensure_ascii=False)
print(f"  [메타데이터 저장 완료] -> {SAVE_META_PATH}")
print("=" * 115)
