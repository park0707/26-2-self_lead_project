# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_v10_vs_v10_2_comparison.py
Purpose: v10.0 vs v10.1 vs v10.2 (c2 v8 후방 불확실성 인식 모델 탑재) 공식 성능 비교 평가
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
from joint_meta_16way_net import decode_hybrid

META_PATH    = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH      = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
C2_FEATS_PATH= r"D:\Data\피처_데이터\squat_c2_v7_19d_feats.npz"
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")

WEIGHTS_V10  = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_0_weights.pth")
WEIGHTS_V10_1= os.path.join(CURRENT_DIR, "joint_meta_16way_v10_1_weights.pth")

data    = np.load(META_PATH,    allow_pickle=True)
c4_data = np.load(C4_PATH,     allow_pickle=True)
q_data  = np.load(QUALITY_PATH, allow_pickle=True)
k_data  = np.load(KINETIC_PATH, allow_pickle=True)
c2_data = np.load(C2_FEATS_PATH,allow_pickle=True)

probs_defect = data['probs_defect'].copy()
views    = data['views']
subjects = data['subjects']
y_all    = data['y_all']
true_clean = data['true_clean']
N = len(true_clean)

# Load c2 features for uncertainty calculation
p_c2_feats = c2_data['X_c2_v7_19d']
face_valid = (p_c2_feats[:, 14] > 0.5)
is_rear = (views == 'view1') | (views == 'view5')

with open(TEMP_PARAM_PATH, 'r', encoding='utf-8') as f:
    t_cfg = json.load(f)
T_c1, T_c3, T_c4 = t_cfg['c1']['global_T'], t_cfg['c3']['global_T'], t_cfg['c4']['global_T']
view_T_c2 = t_cfg['c2']['view_T']

eps = 1e-7

# Function to build 39D meta features
def build_39d(probs_in):
    p_clip = np.clip(probs_in, eps, 1.0 - eps)
    logits_raw = np.log(p_clip / (1.0 - p_clip))
    p_cal = np.zeros_like(p_clip)
    p_cal[:, 0] = 1.0 / (1.0 + np.exp(-logits_raw[:, 0] / T_c1))
    p_cal[:, 2] = 1.0 / (1.0 + np.exp(-logits_raw[:, 2] / T_c3))
    p_cal[:, 3] = 1.0 / (1.0 + np.exp(-logits_raw[:, 3] / T_c4))
    for v, T_v in view_T_c2.items():
        m = (views == v)
        p_cal[m, 1] = 1.0 / (1.0 + np.exp(-logits_raw[m, 1] / T_v))

    margins_cal = np.abs(p_cal - 0.5)

    X_c4 = c4_data['X_c4_v5_25d']
    wobble = X_c4[:, 3]; jerk = X_c4[:, 15]; f0_rate = X_c4[:, 24]
    jerk_norm   = np.clip(jerk   / 200.0, 0.0, 2.5)[:, None]
    wobble_norm = np.clip(wobble / 0.015, 0.0, 2.5)[:, None]
    ydelta_norm = np.clip(X_c4[:, 6] / 0.020, -1.0, 2.5)[:, None]
    toe_norm    = np.clip(X_c4[:, 1] / 0.020,  0.0, 2.5)[:, None]
    f0_norm     = np.clip(f0_rate / 0.50,       0.0, 2.0)[:, None]
    phys_feats  = np.hstack([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm])
    view_map     = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
    view_onehot  = np.eye(5, dtype=np.float32)[[view_map[v] for v in views]]

    X_quality_raw = q_data['X_quality_12d']
    q_bone     = X_quality_raw[:, 0:1]; q_jitter = X_quality_raw[:, 1:2]; q_overall_f = X_quality_raw[:, 2:3]
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
    c_fk   = X_kinetic_raw[:, 0:1]; c_kt = X_kinetic_raw[:, 1:2]; c_tn = X_kinetic_raw[:, 2:3]; c_ft = X_kinetic_raw[:, 3:4]
    d_valg = np.clip(X_kinetic_raw[:, 4:5]  / 0.40,          -2.5, 2.5)
    n_tilt = np.clip(X_kinetic_raw[:, 5:6]  / 0.50,           0.0, 2.5)
    n_valg = np.clip((X_kinetic_raw[:, 6:7] - 1.40) / 0.35,  -2.5, 2.5)
    a_heel = np.clip(X_kinetic_raw[:, 7:8]  / 0.60,           0.0, 2.5)
    p_lag  = np.clip(X_kinetic_raw[:, 8:9]  / 0.40,          -2.5, 2.5)
    kinetic_feats = np.hstack([c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag]).astype(np.float32)

    X_39d = np.hstack([p_cal, margins_cal, view_onehot, phys_feats, quality_feats, kinetic_feats]).astype(np.float32)
    return X_39d, p_cal

# Standard v10/v10.1 39D
X_meta_std, p_cal_std = build_39d(probs_defect)

# Uncertainty-calibrated c2 v8 39D
probs_defect_v8 = probs_defect.copy()
sigma2_c2 = np.where(is_rear, 0.85, np.where(face_valid, 0.05, 0.40))
logit_c2 = np.log(np.clip(probs_defect_v8[:, 1], eps, 1-eps) / (1.0 - np.clip(probs_defect_v8[:, 1], eps, 1-eps)))
probs_defect_v8[:, 1] = 1.0 / (1.0 + np.exp(-logit_c2 / np.sqrt(1.0 + np.pi * sigma2_c2 / 8.0)))
X_meta_v8, p_cal_v8 = build_39d(probs_defect_v8)

q_overall_vec = q_data['X_quality_12d'][:, 2]
wobble = c4_data['X_c4_v5_25d'][:, 3]
jerk   = c4_data['X_c4_v5_25d'][:, 15]
f0_rate= c4_data['X_c4_v5_25d'][:, 24]

def run_eval(model_cls, weights_path, X_39d, p_cal_matrix, tau=0.36, eta=0.05, db_low=0.30, rear_p2_th=0.35):
    m = model_cls(in_features=39, hidden_dim=48, dropout_rate=0.20)
    m.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    m.eval()
    with torch.no_grad():
        l16, l4, _ = m(torch.from_numpy(X_39d).float())
        p16 = torch.softmax(l16 / 1.20, dim=-1).numpy()
        p4  = torch.sigmoid(l4).numpy()

    pred_f = np.zeros(N, dtype=int)
    defs_f = np.zeros((N, 4), dtype=int)
    is_ret = np.zeros(N, dtype=bool)
    pred_s = np.zeros(N, dtype=int)
    defs_s = np.zeros((N, 4), dtype=int)

    for i in range(N):
        s, d, is_cln, p_cln = decode_hybrid(p16[i], p4[i], eta=eta, tau_clean=tau)
        if is_cln:
            pred_f[i] = 1
            defs_f[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal_matrix[i, 0] <= 0.40) and (p_cal_matrix[i, 2] <= 0.40) and (p_cal_matrix[i, 3] <= 0.40) and (p_cal_matrix[i, 1] < rear_p2_th):
            pred_f[i] = 1
            defs_f[i] = [0, 0, 0, 0]
        else:
            defs_f[i] = d

    for i in range(N):
        if pred_f[i] == 0 and max(p_cal_matrix[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_f[i] = 1
            defs_f[i] = [0, 0, 0, 0]
        if pred_f[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_f[i] = 0

    for i in range(N):
        if q_overall_vec[i] < 0.05:
            is_ret[i] = True
            continue
        s, d, is_cln, p_cln = decode_hybrid(p16[i], p4[i], eta=eta, tau_clean=tau)
        if db_low <= p_cln < tau and not is_cln:
            is_ret[i] = True
            continue
        if is_cln:
            pred_s[i] = 1
            defs_s[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal_matrix[i, 0] <= 0.40) and (p_cal_matrix[i, 2] <= 0.40) and (p_cal_matrix[i, 3] <= 0.40) and (p_cal_matrix[i, 1] < rear_p2_th):
            pred_s[i] = 1
            defs_s[i] = [0, 0, 0, 0]
        else:
            defs_s[i] = d

    for i in range(N):
        if is_ret[i]: continue
        if pred_s[i] == 0 and max(p_cal_matrix[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_s[i] = 1
            defs_s[i] = [0, 0, 0, 0]
        if pred_s[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_s[i] = 0

    return pred_f, defs_f, is_ret, pred_s, defs_s

# Run 3 versions
pred_10f, d_10f, r_10, pred_10s, d_10s = run_eval(JointMeta16WayMoENet,      WEIGHTS_V10,   X_meta_std, p_cal_std, tau=0.38, eta=0.10, db_low=0.32, rear_p2_th=0.35)
pred_11f, d_11f, r_11, pred_11s, d_11s = run_eval(JointMeta16WayMoENetV10_1, WEIGHTS_V10_1, X_meta_std, p_cal_std, tau=0.36, eta=0.05, db_low=0.30, rear_p2_th=0.35)
pred_12f, d_12f, r_12, pred_12s, d_12s = run_eval(JointMeta16WayMoENetV10_1, WEIGHTS_V10_1, X_meta_v8,  p_cal_v8,  tau=0.36, eta=0.05, db_low=0.30, rear_p2_th=0.38)

def metrics(yt, yp):
    return (
        f1_score(yt, yp, zero_division=0)*100,
        precision_score(yt, yp, zero_division=0)*100,
        recall_score(yt, yp, zero_division=0)*100
    )

def emr(yall, d):
    p16 = d[:,0]*8 + d[:,1]*4 + d[:,2]*2 + d[:,3]*1
    t16 = yall[:,0]*8 + yall[:,1]*4 + yall[:,2]*2 + yall[:,3]*1
    return float(np.mean(p16 == t16))*100

m10 = ~r_10; m11 = ~r_11; m12 = ~r_12

f1_10s, pre_10s, rec_10s = metrics(true_clean[m10], pred_10s[m10]); emr_10s = emr(y_all[m10], d_10s[m10])
f1_11s, pre_11s, rec_11s = metrics(true_clean[m11], pred_11s[m11]); emr_11s = emr(y_all[m11], d_11s[m11])
f1_12s, pre_12s, rec_12s = metrics(true_clean[m12], pred_12s[m12]); emr_12s = emr(y_all[m12], d_12s[m12])

f1_10f, pre_10f, rec_10f = metrics(true_clean, pred_10f); emr_10f = emr(y_all, d_10f)
f1_11f, pre_11f, rec_11f = metrics(true_clean, pred_11f); emr_11f = emr(y_all, d_11f)
f1_12f, pre_12f, rec_12f = metrics(true_clean, pred_12f); emr_12f = emr(y_all, d_12f)

ret_10 = float(r_10.sum())/N*100
ret_11 = float(r_11.sum())/N*100
ret_12 = float(r_12.sum())/N*100

print("\n" + "=" * 135)
print("  [v10.0 vs v10.1 vs v10.2 (c2 v8 후방 불확실성 인식 모델 탑재) 종합 성능 비교표]")
print("=" * 135)
print(f"  {'버전 / 조건':<38s} | {'Clean F1':>9s} | {'Precision':>10s} | {'Recall':>8s} | {'4-EMR':>7s} | {'RETAKE%':>8s}")
print("  " + "-" * 135)
print(f"  {'v10.0 마스터 (선택적 판정)':<38s} | {f1_10s:8.2f}% | {pre_10s:9.2f}% | {rec_10s:7.2f}% | {emr_10s:6.2f}% | {ret_10:7.2f}%")
print(f"  {'v10.1 구조화 에너지 헤드 (선택적 판정)':<38s} | {f1_11s:8.2f}% | {pre_11s:9.2f}% | {rec_11s:7.2f}% | {emr_11s:6.2f}% | {ret_11:7.2f}%")
print(f"  {'v10.2 c2 v8 탑재 (선택적 판정)':<38s} | {f1_12s:8.2f}% | {pre_12s:9.2f}% | {rec_12s:7.2f}% | {emr_12s:6.2f}% | {ret_12:7.2f}%")
print("  " + "-" * 135)
print(f"  {'v10.0 마스터 (전수 100% 강제)':<38s} | {f1_10f:8.2f}% | {pre_10f:9.2f}% | {rec_10f:7.2f}% | {emr_10f:6.2f}% | {'0.00%':>8s}")
print(f"  {'v10.1 구조화 에너지 헤드 (전수 100% 강제)':<38s} | {f1_11f:8.2f}% | {pre_11f:9.2f}% | {rec_11f:7.2f}% | {emr_11f:6.2f}% | {'0.00%':>8s}")
print(f"  {'v10.2 c2 v8 탑재 (전수 100% 강제)':<38s} | {f1_12f:8.2f}% | {pre_12f:9.2f}% | {rec_12f:7.2f}% | {emr_12f:6.2f}% | {'0.00%':>8s}")
print("=" * 135)

print("\n  [4대 세부 결함(c1~c4) Selective F1 비교]")
defect_labels = ['c1 (척추 중립)', 'c2 (고개 정면)', 'c3 (무릎 외반)', 'c4 (발바닥 고정)']
for k in range(4):
    f10 = f1_score(y_all[m10, k], d_10s[m10, k])*100
    f11 = f1_score(y_all[m11, k], d_11s[m11, k])*100
    f12 = f1_score(y_all[m12, k], d_12s[m12, k])*100
    print(f"    {defect_labels[k]:<20s} | v10.0: {f10:5.2f}% | v10.1: {f11:5.2f}% | v10.2: {f12:5.2f}% (Δ vs v10.1: {f12-f11:+.2f}%)")

print("\n  [5대 시점별 Clean F1 비교 (Full Force)]")
view_order = [('시점 A (우측 후방)', 'view5'), ('시점 B (우측 전방)', 'view2'), ('시점 C (정면 0°)', 'view3'), ('시점 D (좌측 전방)', 'view4'), ('시점 E (좌측 후방)', 'view1')]
for v_name, v_code in view_order:
    mv = (views == v_code)
    f10 = f1_score(true_clean[mv], pred_10f[mv], zero_division=0)*100
    f11 = f1_score(true_clean[mv], pred_11f[mv], zero_division=0)*100
    f12 = f1_score(true_clean[mv], pred_12f[mv], zero_division=0)*100
    print(f"    {v_name:<20s} | v10.0: {f10:5.2f}% | v10.1: {f11:5.2f}% | v10.2: {f12:5.2f}% (Δ vs v10.1: {f12-f11:+.2f}%)")
