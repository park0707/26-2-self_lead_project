# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_v10_vs_v10_1_comparison.py
Purpose: v9.0 vs v10.0 vs v10.1 (해결책 4: Exact 16-State Structured Energy Head) 공식 성능 대조 평가
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

from joint_meta_16way_moe_net import JointMeta16WayMoENet, JointMeta16WayMoENetV10_1
from joint_meta_16way_net import decode_hybrid, idx_to_defect_tuple

META_PATH    = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH      = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")

WEIGHTS_V9   = os.path.join(CURRENT_DIR, "joint_meta_16way_v9_0_weights.pth")
WEIGHTS_V10  = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_0_weights.pth")
WEIGHTS_V10_1= os.path.join(CURRENT_DIR, "joint_meta_16way_v10_1_weights.pth")

data    = np.load(META_PATH,    allow_pickle=True)
c4_data = np.load(C4_PATH,     allow_pickle=True)
q_data  = np.load(QUALITY_PATH, allow_pickle=True)
k_data  = np.load(KINETIC_PATH, allow_pickle=True)

probs_defect = data['probs_defect']
views    = data['views']
subjects = data['subjects']
y_all    = data['y_all']
true_clean = data['true_clean']
N = len(true_clean)

X_c4     = c4_data['X_c4_v5_25d']
wobble   = X_c4[:, 3]
jerk     = X_c4[:, 15]
f0_rate  = X_c4[:, 24]
y_16 = (y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1).astype(np.int64)

# 공통 39D 전처리
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
jerk_norm   = np.clip(jerk   / 200.0, 0.0, 2.5)[:, None]
wobble_norm = np.clip(wobble / 0.015, 0.0, 2.5)[:, None]
ydelta_norm = np.clip(X_c4[:, 6] / 0.020, -1.0, 2.5)[:, None]
toe_norm    = np.clip(X_c4[:, 1] / 0.020,  0.0, 2.5)[:, None]
f0_norm     = np.clip(f0_rate / 0.50,       0.0, 2.0)[:, None]
phys_feats  = np.hstack([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm])
view_map     = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
view_onehot  = np.eye(5, dtype=np.float32)[[view_map[v] for v in views]]

X_quality_raw = q_data['X_quality_12d']
q_overall_vec = X_quality_raw[:, 2]
q_bone     = X_quality_raw[:, 0:1]
q_jitter   = X_quality_raw[:, 1:2]
q_overall_f= X_quality_raw[:, 2:3]
bone_instab= np.clip(X_quality_raw[:, 3:4]   / 0.20,          0.0, 2.5)
jitter_s   = np.clip(X_quality_raw[:, 4:5]   / 0.001,          0.0, 2.5)
nadir_s    = X_quality_raw[:, 5:6]
rom_s      = np.clip(X_quality_raw[:, 6:7]   / 0.015,          0.0, 2.5)
sh_w_s     = np.clip((X_quality_raw[:, 7:8]  - 0.65) / 0.15,  -2.5, 2.5)
hip_w_s    = np.clip((X_quality_raw[:, 8:9]  - 0.42) / 0.10,  -2.5, 2.5)
z_knee_s   = np.clip(X_quality_raw[:, 9:10]  / 0.40,           0.0, 2.5)
z_ank_s    = np.clip(X_quality_raw[:, 10:11] / 0.45,           0.0, 2.5)
head_vis_s = np.clip((X_quality_raw[:, 11:12]- 0.27) / 0.05,  -2.5, 2.5)
quality_feats = np.hstack([q_bone, q_jitter, q_overall_f, bone_instab, jitter_s,
                           nadir_s, rom_s, sh_w_s, hip_w_s, z_knee_s, z_ank_s, head_vis_s]).astype(np.float32)

X_kinetic_raw = k_data['X_kinetic_9d']
c_fk   = X_kinetic_raw[:, 0:1]; c_kt = X_kinetic_raw[:, 1:2]
c_tn   = X_kinetic_raw[:, 2:3]; c_ft = X_kinetic_raw[:, 3:4]
d_valg = np.clip(X_kinetic_raw[:, 4:5]  / 0.40,          -2.5, 2.5)
n_tilt = np.clip(X_kinetic_raw[:, 5:6]  / 0.50,           0.0, 2.5)
n_valg = np.clip((X_kinetic_raw[:, 6:7] - 1.40) / 0.35,  -2.5, 2.5)
a_heel = np.clip(X_kinetic_raw[:, 7:8]  / 0.60,           0.0, 2.5)
p_lag  = np.clip(X_kinetic_raw[:, 8:9]  / 0.40,          -2.5, 2.5)
kinetic_feats = np.hstack([c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag]).astype(np.float32)

X_meta_39d = np.hstack([p_cal, margins_cal, view_onehot, phys_feats, quality_feats, kinetic_feats]).astype(np.float32)

def predict_model(model_cls, weights_path, tau=0.38, db_low=0.32):
    m = model_cls(in_features=39, hidden_dim=48, dropout_rate=0.20)
    m.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    m.eval()
    with torch.no_grad():
        l16, l4, gate_w = m(torch.from_numpy(X_meta_39d).float())
        p16 = torch.softmax(l16 / 1.20, dim=-1).numpy()
        p4  = torch.sigmoid(l4).numpy()

    pred_f = np.zeros(N, dtype=int)
    defs_f = np.zeros((N, 4), dtype=int)
    is_ret = np.zeros(N, dtype=bool)
    pred_s = np.zeros(N, dtype=int)
    defs_s = np.zeros((N, 4), dtype=int)

    for i in range(N):
        s, d, is_cln, p_cln = decode_hybrid(p16[i], p4[i], eta=0.10, tau_clean=tau)
        if is_cln:
            pred_f[i] = 1
            defs_f[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.35):
            pred_f[i] = 1
            defs_f[i] = [0, 0, 0, 0]
        else:
            defs_f[i] = d

    for i in range(N):
        if pred_f[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_f[i] = 1
            defs_f[i] = [0, 0, 0, 0]
        if pred_f[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_f[i] = 0

    for i in range(N):
        if q_overall_vec[i] < 0.05:
            is_ret[i] = True
            continue
        s, d, is_cln, p_cln = decode_hybrid(p16[i], p4[i], eta=0.10, tau_clean=tau)
        if db_low <= p_cln < tau and not is_cln:
            is_ret[i] = True
            continue
        if is_cln:
            pred_s[i] = 1
            defs_s[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.35):
            pred_s[i] = 1
            defs_s[i] = [0, 0, 0, 0]
        else:
            defs_s[i] = d

    for i in range(N):
        if is_ret[i]: continue
        if pred_s[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_s[i] = 1
            defs_s[i] = [0, 0, 0, 0]
        if pred_s[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_s[i] = 0

    return pred_f, defs_f, is_ret, pred_s, defs_s

# Evaluate v9.0, v10.0, v10.1
p_v9_f,  d_v9_f,  r_v9,  p_v9_s,  d_v9_s  = predict_model(JointMeta16WayMoENet,      WEIGHTS_V9,    0.38, 0.32)
p_v10_f, d_v10_f, r_v10, p_v10_s, d_v10_s = predict_model(JointMeta16WayMoENet,      WEIGHTS_V10,   0.38, 0.32)
p_v11_f, d_v11_f, r_v11, p_v11_s, d_v11_s = predict_model(JointMeta16WayMoENetV10_1, WEIGHTS_V10_1, 0.38, 0.32)

def calc_metrics(y_true, y_pred):
    f1   = f1_score(y_true, y_pred, zero_division=0) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec  = recall_score(y_true, y_pred, zero_division=0) * 100.0
    tp   = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp   = int(np.sum((y_true == 0) & (y_pred == 1)))
    return f1, prec, rec, tp, fp

def calc_emr(y_all_true, pred_defs):
    pred_16 = pred_defs[:, 0]*8 + pred_defs[:, 1]*4 + pred_defs[:, 2]*2 + pred_defs[:, 3]*1
    true_16 = y_all_true[:, 0]*8 + y_all_true[:, 1]*4 + y_all_true[:, 2]*2 + y_all_true[:, 3]*1
    return float(np.mean(pred_16 == true_16)) * 100.0

# Full Force
f1_9f,  p_9f,  r_9f,  tp_9f,  fp_9f  = calc_metrics(true_clean, p_v9_f)
f1_10f, p_10f, r_10f, tp_10f, fp_10f = calc_metrics(true_clean, p_v10_f)
f1_11f, p_11f, r_11f, tp_11f, fp_11f = calc_metrics(true_clean, p_v11_f)
emr_9f  = calc_emr(y_all, d_v9_f)
emr_10f = calc_emr(y_all, d_v10_f)
emr_11f = calc_emr(y_all, d_v11_f)

# Selective
m9  = ~r_v9
m10 = ~r_v10
m11 = ~r_v11
f1_9s,  p_9s,  r_9s,  tp_9s,  fp_9s  = calc_metrics(true_clean[m9],  p_v9_s[m9])
f1_10s, p_10s, r_10s, tp_10s, fp_10s = calc_metrics(true_clean[m10], p_v10_s[m10])
f1_11s, p_11s, r_11s, tp_11s, fp_11s = calc_metrics(true_clean[m11], p_v11_s[m11])
emr_9s  = calc_emr(y_all[m9],  d_v9_s[m9])
emr_10s = calc_emr(y_all[m10], d_v10_s[m10])
emr_11s = calc_emr(y_all[m11], d_v11_s[m11])

print("\n" + "=" * 135)
print("  [v9.0 vs v10.0 vs v10.1 (해결책 4: Exact 16-State Structured Energy Head) 종합 성능 비교표]")
print("=" * 135)
print(f"  {'버전 / 조건':<38s} | {'Clean F1':>9s} | {'Precision':>10s} | {'Recall':>8s} | {'4-EMR':>7s} | {'False Clean':>12s} | {'RETAKE%':>8s}")
print("  " + "-" * 135)
print(f"  {'v9.0 베이스라인 (선택적 판정)':<38s} | {f1_9s:8.2f}% | {p_9s:9.2f}% | {r_9s:7.2f}% | {emr_9s:6.2f}% | {'-':>12s} | {float(r_v9.sum())/N*100:7.2f}%")
print(f"  {'v10.0 확정본 (선택적 판정)':<38s} | {f1_10s:8.2f}% | {p_10s:9.2f}% | {r_10s:7.2f}% | {emr_10s:6.2f}% | {'-':>12s} | {float(r_v10.sum())/N*100:7.2f}%")
print(f"  {'v10.1 해결책 4 (선택적 판정)':<38s} | {f1_11s:8.2f}% | {p_11s:9.2f}% | {r_11s:7.2f}% | {emr_11s:6.2f}% | {'-':>12s} | {float(r_v11.sum())/N*100:7.2f}%")
print("  " + "-" * 135)
print(f"  {'v9.0 베이스라인 (전수 강제판정)':<38s} | {f1_9f:8.2f}% | {p_9f:9.2f}% | {r_9f:7.2f}% | {emr_9f:6.2f}% | {fp_9f:10d}건 | {'0.00%':>8s}")
print(f"  {'v10.0 확정본 (전수 강제판정)':<38s} | {f1_10f:8.2f}% | {p_10f:9.2f}% | {r_10f:7.2f}% | {emr_10f:6.2f}% | {fp_10f:10d}건 | {'0.00%':>8s}")
print(f"  {'v10.1 해결책 4 (전수 강제판정)':<38s} | {f1_11f:8.2f}% | {p_11f:9.2f}% | {r_11f:7.2f}% | {emr_11f:6.2f}% | {fp_11f:10d}건 | {'0.00%':>8s}")
print("=" * 135)

print("\n  [5대 시점별 Clean F1 (전수 강제판정)]")
view_order = [
    ('시점 A (우측 후방 45°)', 'view5'),
    ('시점 B (우측 전방 45°)', 'view2'),
    ('시점 C (정면 0°)',       'view3'),
    ('시점 D (좌측 전방 45°)', 'view4'),
    ('시점 E (좌측 후방 45°)', 'view1'),
]
for v_name, v_code in view_order:
    m = (views == v_code)
    f9  = f1_score(true_clean[m], p_v9_f[m],  zero_division=0)*100
    f10 = f1_score(true_clean[m], p_v10_f[m], zero_division=0)*100
    f11 = f1_score(true_clean[m], p_v11_f[m], zero_division=0)*100
    print(f"    {v_name:<24s} | v9.0: {f9:5.2f}% | v10.0: {f10:5.2f}% | v10.1: {f11:5.2f}% (Δ vs v10: {f11-f10:+.2f}%)")

print("\n  [4대 세부 결함(c1~c4) F1 (전수 강제판정)]")
defect_labels = ['c1 (척추 중립)', 'c2 (고개 정면)', 'c3 (무릎 외반)', 'c4 (발바닥 고정)']
for k in range(4):
    yt = y_all[:, k]
    f9  = f1_score(yt, d_v9_f[:, k],  zero_division=0)*100
    f10 = f1_score(yt, d_v10_f[:, k], zero_division=0)*100
    f11 = f1_score(yt, d_v11_f[:, k], zero_division=0)*100
    print(f"    {defect_labels[k]:<20s} | v9.0: {f9:5.2f}% | v10.0: {f10:5.2f}% | v10.1: {f11:5.2f}% (Δ vs v10: {f11-f10:+.2f}%)")
