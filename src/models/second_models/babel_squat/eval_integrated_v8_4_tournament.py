# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_integrated_v8_4_tournament.py
Purpose: Step 4 - 통합 진단 엔진 v8.4 공식 토너먼트 전수 대조 평가
         (v6.0 -> v8.0 -> v8.1 -> v8.2 -> v8.3 -> v8.4 관측성 및 품질 게이트)
         공식 5대 시점: 시점 A(우후방 45도), 시점 B(우전방 45도), 시점 C(정면 0도), 시점 D(좌전방 45도), 시점 E(좌후방 45도)
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from joint_meta_16way_net import JointMeta16WayNet, decode_hybrid, idx_to_defect_tuple

META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")

WEIGHTS_PATH_V8_1 = os.path.join(CURRENT_DIR, "joint_meta_16way_weights.pth")
WEIGHTS_PATH_V8_2 = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_2_weights.pth")
WEIGHTS_PATH_V8_4 = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_4_weights.pth")

data = np.load(META_PATH, allow_pickle=True)
c4_data = np.load(C4_PATH, allow_pickle=True)
q_data = np.load(QUALITY_PATH, allow_pickle=True)

probs_defect = data['probs_defect']
c1_statuses = data['c1_statuses']
views = data['views']
subjects = data['subjects']
y_all = data['y_all']
true_clean = data['true_clean']
N = len(true_clean)

X_c4 = c4_data['X_c4_v5_25d']
wobble = X_c4[:, 3]
ydelta = X_c4[:, 6]
toe_lift_norm = X_c4[:, 1]
jerk = X_c4[:, 15]
f0_rate = X_c4[:, 24]

p_c1, p_c2, p_c3, p_c4 = probs_defect[:, 0], probs_defect[:, 1], probs_defect[:, 2], probs_defect[:, 3]
y_16 = (y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1).astype(np.int64)

print("=" * 125)
print("  [Step 4] 통합 진단 엔진 공식 토너먼트 평가 (v6.0 -> v8.0 -> v8.1 -> v8.2 -> v8.3 -> v8.4)")
print("  공식 5대 시점: 시점 A(우후방), 시점 B(우전방), 시점 C(정면), 시점 D(좌전방), 시점 E(좌후방)")
print("=" * 125)

# 1. v6.0 Baseline
st_c1_base = np.where(c1_statuses == 'NORMAL', 0, np.where(c1_statuses == 'DEFECT', 1, -1))
st_c2_base = []
for i in range(N):
    if views[i] in ['view2', 'view3', 'view4']:
        st_c2_base.append(0 if p_c2[i] <= 0.40 else (1 if p_c2[i] >= 0.60 else -1))
    else:
        st_c2_base.append(0 if p_c2[i] <= 0.30 else (1 if p_c2[i] >= 0.70 else -1))
st_c2_base = np.array(st_c2_base)
st_c3_base = np.where(p_c3 <= 0.40, 0, np.where(p_c3 >= 0.60, 1, -1))
st_c4_base = np.where(p_c4 <= 0.40, 0, np.where(p_c4 >= 0.60, 1, -1))

pred_v6 = []
for i in range(N):
    c1, c2, c3, c4 = st_c1_base[i], st_c2_base[i], st_c3_base[i], st_c4_base[i]
    if views[i] in ['view2', 'view3', 'view4']:
        pred_v6.append(1 if (c1 == 0 and c2 == 0 and c3 == 0 and c4 == 0) else 0)
    else:
        pred_v6.append(1 if (c1 == 0 and c3 == 0 and c4 == 0) else 0)
pred_v6 = np.array(pred_v6)

# 2. v8.0 Master
pred_v8 = np.zeros(N, dtype=int)
for i in range(N):
    cond_clean = (p_c1[i] <= 0.38 and p_c2[i] <= 0.40 and p_c3[i] <= 0.40 and p_c4[i] <= 0.40)
    cond_rear = (views[i] in ['view1', 'view5']) and (p_c1[i] <= 0.38 and p_c3[i] <= 0.40 and p_c4[i] <= 0.40 and p_c2[i] < 0.70)
    if cond_clean or cond_rear:
        pred_v8[i] = 1
    if pred_v8[i] == 0 and max(p_c1[i], p_c2[i], p_c3[i], p_c4[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        pred_v8[i] = 1
    if pred_v8[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        pred_v8[i] = 0

# 3. Level 1 온도 보정
with open(TEMP_PARAM_PATH, 'r', encoding='utf-8') as f:
    t_cfg = json.load(f)
T_c1, T_c3, T_c4 = t_cfg['c1']['global_T'], t_cfg['c3']['global_T'], t_cfg['c4']['global_T']
view_T_c2 = t_cfg['c2']['view_T']

eps = 1e-7
p_clip = np.clip(probs_defect, eps, 1.0 - eps)
logits_raw = np.log(p_clip / (1.0 - p_clip))
p_cal = np.zeros_like(p_clip)
p_cal[:, 0] = 1.0 / (1.0 + np.exp(-logits_raw[:, 0] / T_c1))
p_cal[:, 2] = 1.0 / (1.0 + np.exp(-logits_raw[:, 2] / T_c3))
p_cal[:, 3] = 1.0 / (1.0 + np.exp(-logits_raw[:, 3] / T_c4))
for v, T_v in view_T_c2.items():
    m = (views == v)
    p_cal[m, 1] = 1.0 / (1.0 + np.exp(-logits_raw[m, 1] / T_v))

margins_cal = np.abs(p_cal - 0.5)
jerk_norm = np.clip(jerk / 200.0, 0.0, 2.5)[:, None]
wobble_norm = np.clip(wobble / 0.015, 0.0, 2.5)[:, None]
ydelta_norm = np.clip(ydelta / 0.020, -1.0, 2.5)[:, None]
toe_norm = np.clip(toe_lift_norm / 0.020, 0.0, 2.5)[:, None]
f0_norm = np.clip(f0_rate / 0.50, 0.0, 2.0)[:, None]
phys_feats = np.hstack([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm])

view_map = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
view_indices = np.array([view_map[v] for v in views])
view_onehot = np.eye(5, dtype=np.float32)[view_indices]
X_v82 = np.hstack([p_cal, margins_cal, view_onehot, phys_feats]).astype(np.float32)

# 4. v8.2 Baseline Run
joint_model_v82 = JointMeta16WayNet(in_features=18, hidden_dim=64, dropout_rate=0.20)
joint_model_v82.load_state_dict(torch.load(WEIGHTS_PATH_V8_2, map_location='cpu', weights_only=True))
joint_model_v82.eval()

with torch.no_grad():
    l16_v82, l4_v82 = joint_model_v82(torch.from_numpy(X_v82).float())
    p16_v82_scaled = torch.softmax(l16_v82 / 1.20, dim=-1).numpy()
    p4_v82_sig = torch.sigmoid(l4_v82).numpy()

pred_v8_2 = np.zeros(N, dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    s, defs, is_cln, _ = decode_hybrid(p16_v82_scaled[i], p4_v82_sig[i], eta=0.10, tau_clean=0.38)
    if is_cln:
        pred_v8_2[i] = 1
    elif is_rear and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.70) and ((1.0 - p_cal[i, 0]) * (1.0 - p_cal[i, 2]) * (1.0 - p_cal[i, 3]) >= 0.55) and (p_cal[i, 0] < 0.50 and p_cal[i, 2] < 0.50 and p_cal[i, 3] < 0.50):
        pred_v8_2[i] = 1
    if pred_v8_2[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        pred_v8_2[i] = 1
    if pred_v8_2[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        pred_v8_2[i] = 0

# 5. v8.3 Run (Deadband [0.28, 0.42])
is_retake_v83 = np.zeros(N, dtype=bool)
pred_v83 = np.zeros(N, dtype=int)
pred_defs_v83 = np.zeros((N, 4), dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    s, defs, is_cln, conf = decode_hybrid(p16_v82_scaled[i], p4_v82_sig[i], eta=0.10, tau_clean=0.38)
    p_cln_node = p16_v82_scaled[i, 0]
    in_deadband = (p_cln_node >= 0.28) and (p_cln_node < 0.42)
    if in_deadband and not is_cln:
        is_retake_v83[i] = True
        continue
    if is_cln:
        pred_v83[i] = 1
        pred_defs_v83[i] = [0, 0, 0, 0]
    elif is_rear and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.70) and ((1.0 - p_cal[i, 0]) * (1.0 - p_cal[i, 2]) * (1.0 - p_cal[i, 3]) >= 0.55) and (p_cal[i, 0] < 0.50 and p_cal[i, 2] < 0.50 and p_cal[i, 3] < 0.50):
        pred_v83[i] = 1
        pred_defs_v83[i] = [0, 0, 0, 0]
    else:
        pred_defs_v83[i] = defs

for i in range(N):
    if is_retake_v83[i]:
        continue
    if pred_v83[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        pred_v83[i] = 1
        pred_defs_v83[i] = [0, 0, 0, 0]
    if pred_v83[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        pred_v83[i] = 0

# 6. v8.4 Run (30D 관측성 & 품질 게이트)
X_quality_raw = q_data['X_quality_12d']
q_overall_vec = X_quality_raw[:, 2]

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

X_meta_30d = np.hstack([p_cal, margins_cal, view_onehot, phys_feats, quality_feats]).astype(np.float32)

joint_model_v84 = JointMeta16WayNet(in_features=30, hidden_dim=64, dropout_rate=0.20)
joint_model_v84.load_state_dict(torch.load(WEIGHTS_PATH_V8_4, map_location='cpu', weights_only=True))
joint_model_v84.eval()

with torch.no_grad():
    l16_v84, l4_v84 = joint_model_v84(torch.from_numpy(X_meta_30d).float())
    p16_v84_scaled = torch.softmax(l16_v84 / 1.20, dim=-1).numpy()
    p4_v84_sig = torch.sigmoid(l4_v84).numpy()

# 6-A. v8.4 Full Force (Coverage 100%)
pred_v84_full = np.zeros(N, dtype=int)
pred_defs_v84_full = np.zeros((N, 4), dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    s, defs, is_cln, _ = decode_hybrid(p16_v84_scaled[i], p4_v84_sig[i], eta=0.10, tau_clean=0.38)
    if is_cln:
        pred_v84_full[i] = 1
        pred_defs_v84_full[i] = [0, 0, 0, 0]
    elif is_rear and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.70) and ((1.0 - p_cal[i, 0]) * (1.0 - p_cal[i, 2]) * (1.0 - p_cal[i, 3]) >= 0.55) and (p_cal[i, 0] < 0.50 and p_cal[i, 2] < 0.50 and p_cal[i, 3] < 0.50):
        pred_v84_full[i] = 1
        pred_defs_v84_full[i] = [0, 0, 0, 0]
    else:
        pred_defs_v84_full[i] = defs

for i in range(N):
    if pred_v84_full[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        pred_v84_full[i] = 1
        pred_defs_v84_full[i] = [0, 0, 0, 0]
    if pred_v84_full[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        pred_v84_full[i] = 0

# 6-B. v8.4 Selective Master (q_min=0.05, DB=[0.32, 0.38])
is_retake_v84 = np.zeros(N, dtype=bool)
pred_v84_sel = np.zeros(N, dtype=int)
pred_defs_v84_sel = np.zeros((N, 4), dtype=int)
for i in range(N):
    if q_overall_vec[i] < 0.05:
        is_retake_v84[i] = True
        continue
    is_rear = views[i] in ['view1', 'view5']
    s, defs, is_cln, p_cln = decode_hybrid(p16_v84_scaled[i], p4_v84_sig[i], eta=0.10, tau_clean=0.38)
    if 0.32 <= p_cln < 0.38 and not is_cln:
        is_retake_v84[i] = True
        continue
    if is_cln:
        pred_v84_sel[i] = 1
        pred_defs_v84_sel[i] = [0, 0, 0, 0]
    elif is_rear and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.70) and ((1.0 - p_cal[i, 0]) * (1.0 - p_cal[i, 2]) * (1.0 - p_cal[i, 3]) >= 0.55) and (p_cal[i, 0] < 0.50 and p_cal[i, 2] < 0.50 and p_cal[i, 3] < 0.50):
        pred_v84_sel[i] = 1
        pred_defs_v84_sel[i] = [0, 0, 0, 0]
    else:
        pred_defs_v84_sel[i] = defs

for i in range(N):
    if is_retake_v84[i]:
        continue
    if pred_v84_sel[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        pred_v84_sel[i] = 1
        pred_defs_v84_sel[i] = [0, 0, 0, 0]
    if pred_v84_sel[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        pred_v84_sel[i] = 0

# 성능 집계
def calc_metrics(y_true, y_pred):
    return (
        f1_score(y_true, y_pred, zero_division=0) * 100.0,
        precision_score(y_true, y_pred, zero_division=0) * 100.0,
        recall_score(y_true, y_pred, zero_division=0) * 100.0,
        int(np.sum((y_true == 1) & (y_pred == 1))),
        int(np.sum((y_true == 0) & (y_pred == 1)))
    )

f1_v6, pre_v6, rec_v6, tp_v6, fp_v6 = calc_metrics(true_clean, pred_v6)
f1_v8, pre_v8, rec_v8, tp_v8, fp_v8 = calc_metrics(true_clean, pred_v8)
f1_v82, pre_v82, rec_v82, tp_v82, fp_v82 = calc_metrics(true_clean, pred_v8_2)

# v8.3
mask_v83 = ~is_retake_v83
f1_v83, pre_v83, rec_v83, tp_v83, fp_v83 = calc_metrics(true_clean[mask_v83], pred_v83[mask_v83])
emr_v83 = ((pred_defs_v83[mask_v83, 0]*8 + pred_defs_v83[mask_v83, 1]*4 + pred_defs_v83[mask_v83, 2]*2 + pred_defs_v83[mask_v83, 3]*1) == y_16[mask_v83]).mean() * 100.0

# v8.4 Full
f1_v84_f, pre_v84_f, rec_v84_f, tp_v84_f, fp_v84_f = calc_metrics(true_clean, pred_v84_full)
emr_v84_f = ((pred_defs_v84_full[:, 0]*8 + pred_defs_v84_full[:, 1]*4 + pred_defs_v84_full[:, 2]*2 + pred_defs_v84_full[:, 3]*1) == y_16).mean() * 100.0

# v8.4 Sel
mask_v84 = ~is_retake_v84
f1_v84_s, pre_v84_s, rec_v84_s, tp_v84_s, fp_v84_s = calc_metrics(true_clean[mask_v84], pred_v84_sel[mask_v84])
emr_v84_s = ((pred_defs_v84_sel[mask_v84, 0]*8 + pred_defs_v84_sel[mask_v84, 1]*4 + pred_defs_v84_sel[mask_v84, 2]*2 + pred_defs_v84_sel[mask_v84, 3]*1) == y_16[mask_v84]).mean() * 100.0

print(f"{'버전':<32s} | {'Clean F1':<10s} | {'Precision':<10s} | {'Recall':<10s} | {'4-EMR':<10s} | {'Coverage':<10s} | {'TP (정상적중)':<12s} | {'FP (정상오인)'}")
print("-" * 125)
print(f"{'v6.0 프로덕션 베이스라인':<32s} | {f1_v6:8.2f}% | {pre_v6:8.2f}% | {rec_v6:8.2f}% | {'15.90%':<10s} | {'100.0%':<10s} | {tp_v6:4d} / 135건    | {fp_v6:4d}건")
print(f"{'v8.0 마스터 챔피언':<32s} | {f1_v8:8.2f}% | {pre_v8:8.2f}% | {rec_v8:8.2f}% | {'48.02%':<10s} | {'100.0%':<10s} | {tp_v8:4d} / 135건    | {fp_v8:4d}건")
print(f"{'v8.2 해결책 2 (계층적온도보정)':<32s} | {f1_v82:8.2f}% | {pre_v82:8.2f}% | {rec_v82:8.2f}% | {'53.36%':<10s} | {'100.0%':<10s} | {tp_v82:4d} / 135건    | {fp_v82:4d}건")
print(f"{'v8.3 해결책 3 (선택적판정/데드밴드)':<32s} | {f1_v83:8.2f}% | {pre_v83:8.2f}% | {rec_v83:8.2f}% | {emr_v83:8.2f}% | {'98.9%':<10s} | {tp_v83:4d} / 133건    | {fp_v83:4d}건")
print(f"{'v8.4 해결책 4 (전수 100% 강제판정)':<32s} | {f1_v84_f:8.2f}% | {pre_v84_f:8.2f}% | {rec_v84_f:8.2f}% | {emr_v84_f:8.2f}% | {'100.0%':<10s} | {tp_v84_f:4d} / 135건    | {fp_v84_f:4d}건")
print(f"{'v8.4 해결책 4 (품질게이트 선택적판정)':<32s} | {f1_v84_s:8.2f}% | {pre_v84_s:8.2f}% | {rec_v84_s:8.2f}% | {emr_v84_s:8.2f}% | {np.sum(mask_v84)/N*100:8.1f}% | {tp_v84_s:4d} / {int(np.sum(true_clean[mask_v84]))}건    | {fp_v84_s:4d}건")
print("=" * 125)

# 4대 세부 결함 성능 대조
print("\n  [v8.4 4대 세부 결함(c1~c4) 최종 성능 (전수 100% 강제 판정 N=2,605)]")
defect_labels = ['c1 (척추 중립 위반)', 'c2 (고개 정면/시선 위반)', 'c3 (무릎 외반/모임 위반)', 'c4 (발바닥 지면 고정 위반)']
for k in range(4):
    y_t = y_all[:, k]
    y_p = pred_defs_v84_full[:, k]
    acc = accuracy_score(y_t, y_p) * 100.0
    pre = precision_score(y_t, y_p) * 100.0
    rec = recall_score(y_t, y_p) * 100.0
    f1 = f1_score(y_t, y_p) * 100.0
    print(f"    {defect_labels[k]:<24s} | F1: {f1:5.2f}% | Precision: {pre:5.2f}% | Recall: {rec:5.2f}% | Accuracy: {acc:5.2f}%")

# 5대 공식 시점별 성능 대조
print("\n  [v8.4 공식 5대 시점별 Clean F1 및 Recall (v8.0 vs v8.4)]")
view_order = [
    ('시점 A (우측 후방 45도)', 'view5'),
    ('시점 B (우측 전방 45도)', 'view2'),
    ('시점 C (정면 0도)', 'view3'),
    ('시점 D (좌측 전방 45도)', 'view4'),
    ('시점 E (좌측 후방 45도)', 'view1')
]
for v_name, v_code in view_order:
    m = (views == v_code)
    f1_v8_view = f1_score(true_clean[m], pred_v8[m], zero_division=0) * 100.0
    rec_v8_view = recall_score(true_clean[m], pred_v8[m], zero_division=0) * 100.0
    f1_v84_view = f1_score(true_clean[m], pred_v84_full[m], zero_division=0) * 100.0
    rec_v84_view = recall_score(true_clean[m], pred_v84_full[m], zero_division=0) * 100.0
    tp_v84_view = int(np.sum((true_clean[m] == 1) & (pred_v84_full[m] == 1)))
    print(f"    {v_name:<24s} | v8.0 F1: {f1_v8_view:5.2f}% (Rec: {rec_v8_view:5.2f}%) -> v8.4 F1: {f1_v84_view:5.2f}% (Rec: {rec_v84_view:5.2f}% [{tp_v84_view}/27건])")
print("=" * 125)
