# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_integrated_v8_5_tournament.py
Purpose: Step 4 - 통합 진단 엔진 v8.5 공식 토너먼트 전수 대조 평가
         (v6.0 -> v8.0 -> v8.2 -> v8.3 -> v8.4 -> v8.5 운동사슬 39D 융합 & 품질게이트 계승)
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
KINETIC_PATH = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")

WEIGHTS_PATH_V8_2 = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_2_weights.pth")
WEIGHTS_PATH_V8_4 = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_4_weights.pth")
WEIGHTS_PATH_V8_5 = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_5_weights.pth")

data = np.load(META_PATH, allow_pickle=True)
c4_data = np.load(C4_PATH, allow_pickle=True)
q_data = np.load(QUALITY_PATH, allow_pickle=True)
k_data = np.load(KINETIC_PATH, allow_pickle=True)

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

# 1. Level 1 온도 보정
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

# 2. 12D Quality Features
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

# 3. 9D Kinetic Chain Features
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

# 4. v8.5 39D Run
X_meta_39d = np.hstack([p_cal, margins_cal, view_onehot, phys_feats, quality_feats, kinetic_feats]).astype(np.float32)

joint_model_v85 = JointMeta16WayNet(in_features=39, hidden_dim=64, dropout_rate=0.20)
joint_model_v85.load_state_dict(torch.load(WEIGHTS_PATH_V8_5, map_location='cpu', weights_only=True))
joint_model_v85.eval()

with torch.no_grad():
    l16_v85, l4_v85 = joint_model_v85(torch.from_numpy(X_meta_39d).float())
    p16_v85_scaled = torch.softmax(l16_v85 / 1.20, dim=-1).numpy()
    p4_v85_sig = torch.sigmoid(l4_v85).numpy()

# 4-A. v8.5 Full Force (100% Coverage)
pred_v85_full = np.zeros(N, dtype=int)
pred_defs_v85_full = np.zeros((N, 4), dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    s, defs, is_cln, _ = decode_hybrid(p16_v85_scaled[i], p4_v85_sig[i], eta=0.10, tau_clean=0.38)
    if is_cln:
        pred_v85_full[i] = 1
        pred_defs_v85_full[i] = [0, 0, 0, 0]
    elif is_rear and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.70) and ((1.0 - p_cal[i, 0]) * (1.0 - p_cal[i, 2]) * (1.0 - p_cal[i, 3]) >= 0.55) and (p_cal[i, 0] < 0.50 and p_cal[i, 2] < 0.50 and p_cal[i, 3] < 0.50):
        pred_v85_full[i] = 1
        pred_defs_v85_full[i] = [0, 0, 0, 0]
    else:
        pred_defs_v85_full[i] = defs

for i in range(N):
    if pred_v85_full[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        pred_v85_full[i] = 1
        pred_defs_v85_full[i] = [0, 0, 0, 0]
    if pred_v85_full[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        pred_v85_full[i] = 0

# 4-B. v8.5 [v8.4 계승] 품질 게이트 선택적 판정 (Selective Master)
is_retake_v85 = np.zeros(N, dtype=bool)
pred_v85_sel = np.zeros(N, dtype=int)
pred_defs_v85_sel = np.zeros((N, 4), dtype=int)
for i in range(N):
    if q_overall_vec[i] < 0.05:
        is_retake_v85[i] = True
        continue
    is_rear = views[i] in ['view1', 'view5']
    s, defs, is_cln, p_cln = decode_hybrid(p16_v85_scaled[i], p4_v85_sig[i], eta=0.10, tau_clean=0.38)
    if 0.32 <= p_cln < 0.38 and not is_cln:
        is_retake_v85[i] = True
        continue
    if is_cln:
        pred_v85_sel[i] = 1
        pred_defs_v85_sel[i] = [0, 0, 0, 0]
    elif is_rear and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.70) and ((1.0 - p_cal[i, 0]) * (1.0 - p_cal[i, 2]) * (1.0 - p_cal[i, 3]) >= 0.55) and (p_cal[i, 0] < 0.50 and p_cal[i, 2] < 0.50 and p_cal[i, 3] < 0.50):
        pred_v85_sel[i] = 1
        pred_defs_v85_sel[i] = [0, 0, 0, 0]
    else:
        pred_defs_v85_sel[i] = defs

for i in range(N):
    if is_retake_v85[i]:
        continue
    if pred_v85_sel[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        pred_v85_sel[i] = 1
        pred_defs_v85_sel[i] = [0, 0, 0, 0]
    if pred_v85_sel[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        pred_v85_sel[i] = 0

# v8.5 성능 집계
def calc_metrics(y_true, y_pred):
    return (
        f1_score(y_true, y_pred, zero_division=0) * 100.0,
        precision_score(y_true, y_pred, zero_division=0) * 100.0,
        recall_score(y_true, y_pred, zero_division=0) * 100.0,
        int(np.sum((y_true == 1) & (y_pred == 1))),
        int(np.sum((y_true == 0) & (y_pred == 1)))
    )

f1_v85_f, pre_v85_f, rec_v85_f, tp_v85_f, fp_v85_f = calc_metrics(true_clean, pred_v85_full)
emr_v85_f = ((pred_defs_v85_full[:, 0]*8 + pred_defs_v85_full[:, 1]*4 + pred_defs_v85_full[:, 2]*2 + pred_defs_v85_full[:, 3]*1) == y_16).mean() * 100.0

mask_v85 = ~is_retake_v85
f1_v85_s, pre_v85_s, rec_v85_s, tp_v85_s, fp_v85_s = calc_metrics(true_clean[mask_v85], pred_v85_sel[mask_v85])
emr_v85_s = ((pred_defs_v85_sel[mask_v85, 0]*8 + pred_defs_v85_sel[mask_v85, 1]*4 + pred_defs_v85_sel[mask_v85, 2]*2 + pred_defs_v85_sel[mask_v85, 3]*1) == y_16[mask_v85]).mean() * 100.0

print("=" * 125)
print("  [공식 토너먼트 종합 대조표 (v6.0 -> v8.0 -> v8.2 -> v8.3 -> v8.4 -> v8.5)]")
print("=" * 125)
print(f"{'버전':<34s} | {'Clean F1':<10s} | {'Precision':<10s} | {'Recall':<10s} | {'4-EMR':<10s} | {'Coverage':<10s} | {'TP (정상적중)':<12s} | {'FP (정상오인)'}")
print("-" * 125)
print(f"{'v6.0 프로덕션 베이스라인':<34s} |    34.43% |    34.06% |    34.81% | 15.90%     | 100.0%     |   47 / 135건    |   91건")
print(f"{'v8.0 마스터 챔피언':<34s} |    55.40% |    53.85% |    57.04% | 48.02%     | 100.0%     |   77 / 135건    |   66건")
print(f"{'v8.2 해결책 2 (계층적온도보정)':<34s} |    60.50% |    58.22% |    62.96% | 53.36%     | 100.0%     |   85 / 135건    |   61건")
print(f"{'v8.3 해결책 3 (선택적판정/데드밴드)':<34s} |    62.88% |    62.61% |    63.16% | 53.51%     |  98.9%     |   72 / 133건    |   43건")
print(f"{'v8.4 해결책 4 (전수 100% 강제판정)':<34s} |    61.81% |    58.17% |    65.93% | 57.62%     | 100.0%     |   89 / 135건    |   64건")
print(f"{'v8.4 해결책 4 (품질게이트 선택적판정)':<34s} |    65.13% |    60.28% |    70.83% | 59.94%     |  86.3%     |   85 / 120건    |   56건")
print(f"{'v8.5 해결책 5 (전수 100% 강제판정)':<34s} | {f1_v85_f:8.2f}% | {pre_v85_f:8.2f}% | {rec_v85_f:8.2f}% | {emr_v85_f:8.2f}% | {'100.0%':<10s} | {tp_v85_f:4d} / 135건    | {fp_v85_f:4d}건")
print(f"{'v8.5 해결책 5 (운동사슬 선택적판정)':<34s} | {f1_v85_s:8.2f}% | {pre_v85_s:8.2f}% | {rec_v85_s:8.2f}% | {emr_v85_s:8.2f}% | {np.sum(mask_v85)/N*100:8.1f}% | {tp_v85_s:4d} / {int(np.sum(true_clean[mask_v85]))}건    | {fp_v85_s:4d}건")
print("=" * 125)

print("\n  [v8.5 4대 세부 결함(c1~c4) 최종 성능 (전수 100% 강제 판정 N=2,605)]")
defect_labels = ['c1 (척추 중립 위반)', 'c2 (고개 정면/시선 위반)', 'c3 (무릎 외반/모임 위반)', 'c4 (발바닥 지면 고정 위반)']
for k in range(4):
    y_t = y_all[:, k]
    y_p = pred_defs_v85_full[:, k]
    acc = accuracy_score(y_t, y_p) * 100.0
    pre = precision_score(y_t, y_p) * 100.0
    rec = recall_score(y_t, y_p) * 100.0
    f1 = f1_score(y_t, y_p) * 100.0
    print(f"    {defect_labels[k]:<24s} | F1: {f1:5.2f}% | Precision: {pre:5.2f}% | Recall: {rec:5.2f}% | Accuracy: {acc:5.2f}%")

print("\n  [v8.5 공식 5대 시점별 Clean F1 및 Recall (전수 100% 강제 판정)]")
view_order = [
    ('시점 A (우측 후방 45도)', 'view5'),
    ('시점 B (우측 전방 45도)', 'view2'),
    ('시점 C (정면 0도)', 'view3'),
    ('시점 D (좌측 전방 45도)', 'view4'),
    ('시점 E (좌측 후방 45도)', 'view1')
]
for v_name, v_code in view_order:
    m = (views == v_code)
    f1_view = f1_score(true_clean[m], pred_v85_full[m], zero_division=0) * 100.0
    rec_view = recall_score(true_clean[m], pred_v85_full[m], zero_division=0) * 100.0
    tp_view = int(np.sum((true_clean[m] == 1) & (pred_v85_full[m] == 1)))
    print(f"    {v_name:<24s} | F1: {f1_view:5.2f}% (Recall: {rec_view:5.2f}%, {tp_view}/27건 적중)")
print("=" * 125)
