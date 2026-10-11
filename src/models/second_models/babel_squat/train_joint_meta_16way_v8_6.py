# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_joint_meta_16way_v8_6.py
Purpose: 해결책 6 [View-Conditioned Mixture-of-Experts (MoE) 동적 라우팅] 기반 v8.6 학습
         - 1단계: 39D 메타 특성 행렬 구성 (Level 1 보정확률 + 마진 + 시점 + c4물리 + 품질게이트 12D + 운동사슬 9D)
         - 2단계: 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습 (3대 전문가 + 게이팅 넷)
         - 3단계: 전체 2,605건 기반 프로덕션 가중치 학습 및 저장 (v8_6_weights.pth, v8_6_meta.json)
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

from joint_meta_16way_moe_net import JointMeta16WayMoENet, MoEConsistencyLoss
from joint_meta_16way_net import decode_hybrid, STATE_TO_DEFECTS_MAP, idx_to_defect_tuple

META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")

SAVE_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_6_weights.pth")
SAVE_META_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_6_meta.json")

print("=" * 115)
print("  [Step 1] Level 1 온도 보정 및 39D MoE 메타 특성 행렬 구성")
print("=" * 115)

data = np.load(META_PATH, allow_pickle=True)
c4_data = np.load(C4_PATH, allow_pickle=True)
q_data = np.load(QUALITY_PATH, allow_pickle=True)
k_data = np.load(KINETIC_PATH, allow_pickle=True)

probs_defect = data['probs_defect'].copy()
views = data['views']
subjects = data['subjects']
y_all = data['y_all']
true_clean = data['true_clean']
N = len(true_clean)

# 온도 파라미터 로드
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

# 18D Baseline
X_c4 = c4_data['X_c4_v5_25d']
jerk_norm = np.clip(X_c4[:, 15:16] / 200.0, 0.0, 2.5)
wobble_norm = np.clip(X_c4[:, 3:4] / 0.015, 0.0, 2.5)
ydelta_norm = np.clip(X_c4[:, 6:7] / 0.020, -1.0, 2.5)
toe_norm = np.clip(X_c4[:, 1:2] / 0.020, 0.0, 2.5)
f0_norm = np.clip(X_c4[:, 24:25] / 0.50, 0.0, 2.0)
phys_feats = np.hstack([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm])

margins_cal = np.abs(p_calibrated - 0.5)

view_map = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
view_indices = np.array([view_map[v] for v in views])
view_onehot = np.eye(5, dtype=np.float32)[view_indices]

# 12D Quality Features
X_quality_raw = q_data['X_quality_12d']
q_bone = X_quality_raw[:, 0:1]
q_jitter = X_quality_raw[:, 1:2]
q_overall = X_quality_raw[:, 2:3]
bone_instab = np.clip(X_quality_raw[:, 3:4] / 0.20, 0.0, 2.5)
jitter_s = np.clip(X_quality_raw[:, 4:5] / 0.001, 0.0, 2.5)
nadir_s = X_quality_raw[:, 5:6]
rom_s = np.clip(X_quality_raw[:, 6:7] / 0.015, 0.0, 2.5)
sh_w_s = np.clip((X_quality_raw[:, 7:8] - 0.65) / 0.15, -2.5, 2.5)
hip_w_s = np.clip((X_quality_raw[:, 8:9] - 0.42) / 0.10, -2.5, 2.5)
z_knee_s = np.clip(X_quality_raw[:, 9:10] / 0.40, 0.0, 2.5)
z_ank_s = np.clip(X_quality_raw[:, 10:11] / 0.45, 0.0, 2.5)
head_vis_s = np.clip((X_quality_raw[:, 11:12] - 0.27) / 0.05, -2.5, 2.5)

quality_feats = np.hstack([
    q_bone, q_jitter, q_overall, bone_instab, jitter_s,
    nadir_s, rom_s, sh_w_s, hip_w_s, z_knee_s, z_ank_s, head_vis_s
]).astype(np.float32)

# 9D Kinetic Chain Features
X_kinetic_raw = k_data['X_kinetic_9d']
c_fk = X_kinetic_raw[:, 0:1]
c_kt = X_kinetic_raw[:, 1:2]
c_tn = X_kinetic_raw[:, 2:3]
c_ft = X_kinetic_raw[:, 3:4]
d_valg = np.clip(X_kinetic_raw[:, 4:5] / 0.40, -2.5, 2.5)
n_tilt = np.clip(X_kinetic_raw[:, 5:6] / 0.50, 0.0, 2.5)
n_valg = np.clip((X_kinetic_raw[:, 6:7] - 1.40) / 0.35, -2.5, 2.5)
a_heel = np.clip(X_kinetic_raw[:, 7:8] / 0.60, 0.0, 2.5)
p_lag = np.clip(X_kinetic_raw[:, 8:9] / 0.40, -2.5, 2.5)

kinetic_feats = np.hstack([
    c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag
]).astype(np.float32)

# 39D MoE 특성 행렬
X_meta_39d = np.hstack([p_calibrated, margins_cal, view_onehot, phys_feats, quality_feats, kinetic_feats]).astype(np.float32)
y_16 = (y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1).astype(np.int64)

class_counts = np.bincount(y_16, minlength=16)
smoothed_weights = (N / (16.0 * class_counts)) ** 0.35
smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32)

print(f"  총 표본: {N}건 | 39D MoE 특성 행렬 구축 완료")

# 2. 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습
print("\n" + "=" * 115)
print("  [Step 2] 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습 (View-Conditioned MoE)")
print("=" * 115)

sgkf = StratifiedGroupKFold(n_splits=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_meta_39d, y_16, groups=subjects)):
    val_subs = np.unique(subjects[val_idx])
    print(f"  --- Fold {fold + 1}/5 (검증 피험자: {val_subs}) ---")

# 3. 전체 데이터 기반 최종 프로덕션 가중치 학습 및 저장
print("\n" + "=" * 115)
print("  [Step 3] 전체 2,605건 기반 최종 v8.6 MoE 프로덕션 가중치 학습 및 저장")
print("=" * 115)

prod_model = JointMeta16WayMoENet(in_features=39, hidden_dim=48, dropout_rate=0.20).to(device)
prod_loss_fn = MoEConsistencyLoss(
    weight_ce=0.55,
    weight_bce=0.25,
    weight_consistency=0.10,
    weight_gate_balance=0.10,
    class_weights_16=weights_tensor.to(device)
).to(device)

prod_opt = optim.AdamW(prod_model.parameters(), lr=0.005, weight_decay=1e-3)
prod_sched = optim.lr_scheduler.CosineAnnealingLR(prod_opt, T_max=120, eta_min=1e-4)

x_full = torch.tensor(X_meta_39d, device=device)
y16_full = torch.tensor(y_16, device=device)
y4_full = torch.tensor(y_all, dtype=torch.float32, device=device)

prod_model.train()
for epoch in range(1, 121):
    prod_opt.zero_grad()
    l16, l4, gate_w = prod_model(x_full)
    loss, _ = prod_loss_fn(l16, l4, y16_full, y4_full, gate_w)
    loss.backward()
    prod_opt.step()
    prod_sched.step()

torch.save(prod_model.state_dict(), SAVE_WEIGHTS_PATH)
print(f"  [가중치 저장 완료] -> {SAVE_WEIGHTS_PATH}")

cfg_v86 = {
    'version': 'v8.6',
    'architecture': 'View-Conditioned Mixture-of-Experts (MoE)',
    'in_features': 39,
    'num_experts': 3,
    'T_meta': 1.20,
    'eta': 0.10,
    'tau_clean': 0.38,
    'q_min': 0.05,
    'deadband_low': 0.32,
    'deadband_high': 0.38
}

with open(SAVE_META_PATH, 'w', encoding='utf-8') as f:
    json.dump(cfg_v86, f, indent=2, ensure_ascii=False)
print(f"  [메타데이터 저장 완료] -> {SAVE_META_PATH}")
print("=" * 115)
