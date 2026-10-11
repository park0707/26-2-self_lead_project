# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_joint_meta_v12_3.py
Purpose: v12.3 [전체 해결책 통합 마스터] 학습 및 OOF 제약 최적화
         - [해결책 1]: Factorized Pairwise Energy Head
         - [해결책 2]: Soft-EMR + Soft-Clean F1 지표 정렬 복합 손실
         - [해결책 3]: 39D 도메인 토큰화 & FiLM Gated Residual 중간 융합
         - [사후 보정]: v12.3 전용 5대 시점별 16D 벡터 스케일러 피팅
         - [해결책 4]: OOF 기반 제약 최적화 (Coverage >= 90%, RETAKE <= 10%, Prec >= 75%)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from scipy.optimize import minimize
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from joint_meta_16way_v12_3_net import JointMeta16WayMoENetV12_3, CompositeMetricAlignedLossV12_3
from joint_meta_16way_net import decode_hybrid, STATE_TO_DEFECTS_MAP, idx_to_defect_tuple

META_PATH    = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH      = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
C2_FEATS_PATH= r"D:\Data\피처_데이터\squat_c2_v7_19d_feats.npz"
NORM_PRED_PATH = os.path.join(CURRENT_DIR, "squat_skeletal_normalized_predictions_2605.npz")
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")

SAVE_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_v12_3_weights.pth")
SAVE_META_PATH    = os.path.join(CURRENT_DIR, "joint_meta_16way_v12_3_meta.json")
SAVE_OOF_PATH     = os.path.join(CURRENT_DIR, "joint_meta_v12_3_oof_logits16.npy")
SAVE_SCALER_PATH  = os.path.join(CURRENT_DIR, "view_vector_scaler_params_v12_3.json")
SAVE_GATE_PATH    = os.path.join(CURRENT_DIR, "optimal_gate_params_v12_3.json")

print("=" * 115)
print("  [v12.3 Step 1] 39D MoE 메타 특성 행렬 구성 (FiLM + Energy Head + Soft-EMR)")
print("=" * 115)

data    = np.load(META_PATH,    allow_pickle=True)
c4_data = np.load(C4_PATH,     allow_pickle=True)
q_data  = np.load(QUALITY_PATH, allow_pickle=True)
k_data  = np.load(KINETIC_PATH, allow_pickle=True)
c2_data = np.load(C2_FEATS_PATH,allow_pickle=True)
norm_d  = np.load(NORM_PRED_PATH, allow_pickle=True)

probs_defect = data['probs_defect'].copy()
views    = data['views']
subjects = data['subjects']
y_all    = data['y_all']
true_clean = data['true_clean']
N = len(true_clean)

# c2 사후 수축 및 c3 정규화
p_c2_feats = c2_data['X_c2_v7_19d']
face_valid = (p_c2_feats[:, 14] > 0.5)
is_rear = (views == 'view1') | (views == 'view5')
sigma2_c2 = np.where(is_rear & (~face_valid), 0.85, 0.05)
kappa_c2 = 1.0 / np.sqrt(1.0 + np.pi * sigma2_c2 / 8.0)
eps = 1e-7
p2_clip = np.clip(probs_defect[:, 1], eps, 1.0 - eps)
logit_c2 = np.log(p2_clip / (1.0 - p2_clip))
p2_cal_v8 = 1.0 / (1.0 + np.exp(-logit_c2 * kappa_c2))
probs_defect[:, 1] = p2_cal_v8
probs_defect[:, 2] = 0.70 * probs_defect[:, 2] + 0.30 * norm_d['p_c3_norm']

with open(TEMP_PARAM_PATH, 'r', encoding='utf-8') as f:
    temp_params = json.load(f)

T_c1 = temp_params['c1']['global_T']
T_c3 = temp_params['c3']['global_T']
T_c4 = temp_params['c4']['global_T']
view_T_c2 = temp_params['c2']['view_T']

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
y_clean_np = (y_16 == 0).astype(np.float32)

class_counts = np.bincount(y_16, minlength=16)
smoothed_weights = (N / (16.0 * np.maximum(class_counts, 1))) ** 0.35
smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32)

print(f"  총 표본: {N}건 | Clean: {int(y_clean_np.sum())}건 ({y_clean_np.mean()*100:.1f}%) | 39D MoE 특성 행렬 준비 완료")

# ============================================================
# [v12.3 Step 2] 5-Fold StratifiedGroupKFold 피험자 격리 학습 (FiLM + Soft-EMR)
# ============================================================
print("\n" + "=" * 115)
print("  [v12.3 Step 2] 5-Fold StratifiedGroupKFold 피험자 격리 학습 (FiLM 융합 + Soft-EMR 복합 손실)")
print("=" * 115)

sgkf = StratifiedGroupKFold(n_splits=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

oof_logits16 = np.zeros((N, 16), dtype=np.float32)
oof_logits4  = np.zeros((N, 4), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_meta_39d, y_16, groups=subjects)):
    val_subjs = np.unique(subjects[val_idx])
    
    X_tr = torch.from_numpy(X_meta_39d[train_idx]).to(device)
    y16_tr = torch.from_numpy(y_16[train_idx]).to(device)
    y4_tr = torch.from_numpy(y_all[train_idx]).float().to(device)
    yc_tr = torch.from_numpy(y_clean_np[train_idx]).to(device)

    X_va = torch.from_numpy(X_meta_39d[val_idx]).to(device)
    
    net = JointMeta16WayMoENetV12_3(in_features=39, hidden_dim=48, dropout_rate=0.20).to(device)
    criterion = CompositeMetricAlignedLossV12_3(
        weight_ce=0.40, weight_clean=0.15, weight_bce=0.15,
        weight_soft_emr=0.15, weight_soft_clean_f1=0.10,
        weight_consistency=0.025, weight_gate_balance=0.025,
        lambda_fc=3.0, class_weights_16=weights_tensor.to(device)
    )
    optimizer = optim.AdamW(net.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-5)

    for epoch in range(120):
        net.train()
        optimizer.zero_grad()
        l16, l4, g = net(X_tr)
        loss, _ = criterion(l16, l4, y16_tr, y4_tr, g, yc_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.5)
        optimizer.step()
        scheduler.step()

    net.eval()
    with torch.no_grad():
        v16, v4, _ = net(X_va)
        oof_logits16[val_idx] = v16.cpu().numpy()
        oof_logits4[val_idx]  = v4.cpu().numpy()

    p16_val = torch.softmax(v16, dim=-1).cpu().numpy()
    pred16_val = np.argmax(p16_val, axis=-1)
    fold_emr = np.mean(pred16_val == y_16[val_idx]) * 100.0
    print(f"  Fold {fold+1} (격리 피험자: {list(val_subjs)}) -> OOF 16-상태 일치율: {fold_emr:.2f}%")

np.save(SAVE_OOF_PATH, oof_logits16)
print(f"  [SUCCESS] OOF 로짓 저장 완료: {SAVE_OOF_PATH}")

# ============================================================
# [v12.3 Step 3] v12.3 전용 5대 시점별 16D 벡터 스케일러 피팅
# ============================================================
print("\n" + "=" * 115)
print("  [v12.3 Step 3] v12.3 전용 5대 시점별 16D 벡터 스케일러 피팅")
print("=" * 115)

scaler_params_v12_3 = {}
view_names = ['view5', 'view2', 'view3', 'view4', 'view1']

for v_name in view_names:
    m = (views == v_name)
    z_v = oof_logits16[m]
    y_v = y_16[m]
    N_v = len(y_v)

    def nll_objective(params):
        s = params[:16]
        b = params[16:]
        z_scaled = z_v * s + b
        z_max = np.max(z_scaled, axis=-1, keepdims=True)
        exp_z = np.exp(z_scaled - z_max)
        p = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        log_p = np.log(np.clip(p, 1e-8, 1.0))
        nll = -np.mean(log_p[np.arange(N_v), y_v])
        reg = 0.01 * np.sum((s - 1.0)**2) + 0.005 * np.sum(b**2)
        return nll + reg

    init_params = np.concatenate([np.ones(16, dtype=np.float32), np.zeros(16, dtype=np.float32)])
    bounds = [(0.2, 3.0)] * 16 + [(-3.0, 3.0)] * 16
    res = minimize(nll_objective, init_params, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
    
    scaler_params_v12_3[v_name] = {
        'scale': [round(x, 4) for x in res.x[:16].tolist()],
        'bias': [round(x, 4) for x in res.x[16:].tolist()]
    }
    print(f"  시점 {v_name} ({N_v}건) -> 피팅 완료 (NLL: {res.fun:.4f})")

with open(SAVE_SCALER_PATH, 'w', encoding='utf-8') as f:
    json.dump(scaler_params_v12_3, f, indent=2)
print(f"  [SUCCESS] v12.3 벡터 스케일러 파라미터 저장 완료: {SAVE_SCALER_PATH}")

# ============================================================
# [v12.3 Step 4] 전수 2,605건 프로덕션 챔피언 모델 학습 및 저장
# ============================================================
print("\n" + "=" * 115)
print("  [v12.3 Step 4] 전수 2,605건 프로덕션 챔피언 모델 학습 및 저장")
print("=" * 115)

prod_net = JointMeta16WayMoENetV12_3(in_features=39, hidden_dim=48, dropout_rate=0.20).to(device)
optimizer = optim.AdamW(prod_net.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=130, eta_min=1e-5)
criterion = CompositeMetricAlignedLossV12_3(
    weight_ce=0.40, weight_clean=0.15, weight_bce=0.15,
    weight_soft_emr=0.15, weight_soft_clean_f1=0.10,
    weight_consistency=0.025, weight_gate_balance=0.025,
    lambda_fc=3.0, class_weights_16=weights_tensor.to(device)
)

X_all_t = torch.from_numpy(X_meta_39d).to(device)
y16_all_t = torch.from_numpy(y_16).to(device)
y4_all_t = torch.from_numpy(y_all).float().to(device)
yc_all_t = torch.from_numpy(y_clean_np).to(device)

torch.manual_seed(42)
for epoch in range(130):
    prod_net.train()
    optimizer.zero_grad()
    l16, l4, g = prod_net(X_all_t)
    loss, _ = criterion(l16, l4, y16_all_t, y4_all_t, g, yc_all_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(prod_net.parameters(), max_norm=1.5)
    optimizer.step()
    scheduler.step()

torch.save(prod_net.state_dict(), SAVE_WEIGHTS_PATH)
print(f"  [SUCCESS] v12.3 프로덕션 가중치 저장 완료: {SAVE_WEIGHTS_PATH}")
