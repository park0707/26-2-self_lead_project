# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_v9_vs_v10_comparison.py
Purpose: v9.0 (기존 베이스라인) vs v10.0 (해결책 1: False-Clean Asymmetric Loss) 공식 성능 비교
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

from joint_meta_16way_moe_net import JointMeta16WayMoENet
from joint_meta_16way_net import decode_hybrid, idx_to_defect_tuple

META_PATH    = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH      = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")

WEIGHTS_V9  = os.path.join(CURRENT_DIR, "joint_meta_16way_v9_0_weights.pth")
WEIGHTS_V10 = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_0_weights.pth")
META_V10    = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_0_meta.json")

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

# 공통 전처리 (v9.0 동일 방식)
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

# v10.0 meta 로드 (tau_clean 동적 결정)
with open(META_V10, 'r', encoding='utf-8') as f:
    meta_v10 = json.load(f)
TAU_V9  = 0.38
TAU_V10 = float(meta_v10['tau_clean'])

print(f"  v9.0 tau_clean: {TAU_V9:.3f}")
print(f"  v10.0 tau_clean: {TAU_V10:.3f} (OOF Precision>=70% 제약 탐색 결과)")

# 모델 추론 공통 함수
def run_model_inference(weights_path, tau_clean, label):
    model = JointMeta16WayMoENet(in_features=39, hidden_dim=48, dropout_rate=0.20)
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        l16, l4, gate_w = model(torch.from_numpy(X_meta_39d).float())
        p16 = torch.softmax(l16 / 1.20, dim=-1).numpy()
        p4  = torch.sigmoid(l4).numpy()
        gate = gate_w.numpy()
    print(f"\n  [{label}] 모델 추론 완료 | tau_clean={tau_clean:.3f}")
    return p16, p4, gate, tau_clean

p16_v9,  p4_v9,  gate_v9,  tau_v9  = run_model_inference(WEIGHTS_V9,  TAU_V9,  "v9.0")
p16_v10, p4_v10, gate_v10, tau_v10 = run_model_inference(WEIGHTS_V10, TAU_V10, "v10.0")

# 판정 로직 공통 함수 (v9.0 / v10.0 동일한 로직 사용)
def run_predictions(p16, p4, tau_clean, label):
    pred_full = np.zeros(N, dtype=int)
    pred_defs_full = np.zeros((N, 4), dtype=int)
    is_retake = np.zeros(N, dtype=bool)
    pred_sel = np.zeros(N, dtype=int)
    pred_defs_sel = np.zeros((N, 4), dtype=int)

    for i in range(N):
        s, defs, is_cln, p_cln = decode_hybrid(p16[i], p4[i], eta=0.10, tau_clean=tau_clean)
        if is_cln:
            pred_full[i] = 1
            pred_defs_full[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.35):
            pred_full[i] = 1
            pred_defs_full[i] = [0, 0, 0, 0]
        else:
            pred_defs_full[i] = defs

    for i in range(N):
        if pred_full[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_full[i] = 1
            pred_defs_full[i] = [0, 0, 0, 0]
        if pred_full[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_full[i] = 0

    for i in range(N):
        if q_overall_vec[i] < 0.05:
            is_retake[i] = True
            continue
        s, defs, is_cln, p_cln = decode_hybrid(p16[i], p4[i], eta=0.10, tau_clean=tau_clean)
        if 0.32 <= p_cln < tau_clean and not is_cln:
            is_retake[i] = True
            continue
        if is_cln:
            pred_sel[i] = 1
            pred_defs_sel[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.35):
            pred_sel[i] = 1
            pred_defs_sel[i] = [0, 0, 0, 0]
        else:
            pred_defs_sel[i] = defs

    for i in range(N):
        if is_retake[i]: continue
        if pred_sel[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_sel[i] = 1
            pred_defs_sel[i] = [0, 0, 0, 0]
        if pred_sel[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_sel[i] = 0

    return pred_full, pred_defs_full, is_retake, pred_sel, pred_defs_sel

pred_v9_f,  defs_v9_f,  ret_v9,  pred_v9_s,  defs_v9_s  = run_predictions(p16_v9,  p4_v9,  tau_v9,  "v9.0")
pred_v10_f, defs_v10_f, ret_v10, pred_v10_s, defs_v10_s = run_predictions(p16_v10, p4_v10, tau_v10, "v10.0")

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

f1_v9_f,  pre_v9_f,  rec_v9_f,  tp_v9_f,  fp_v9_f  = calc_metrics(true_clean, pred_v9_f)
f1_v10_f, pre_v10_f, rec_v10_f, tp_v10_f, fp_v10_f = calc_metrics(true_clean, pred_v10_f)
emr_v9_f  = calc_emr(y_all, defs_v9_f)
emr_v10_f = calc_emr(y_all, defs_v10_f)

mask_v9  = ~ret_v9
mask_v10 = ~ret_v10
f1_v9_s,  pre_v9_s,  rec_v9_s,  tp_v9_s,  fp_v9_s  = calc_metrics(true_clean[mask_v9],  pred_v9_s[mask_v9])
f1_v10_s, pre_v10_s, rec_v10_s, tp_v10_s, fp_v10_s = calc_metrics(true_clean[mask_v10], pred_v10_s[mask_v10])
emr_v9_s  = calc_emr(y_all[mask_v9],  defs_v9_s[mask_v9])
emr_v10_s = calc_emr(y_all[mask_v10], defs_v10_s[mask_v10])
ret_rate_v9  = float(ret_v9.sum()) / N * 100.0
ret_rate_v10 = float(ret_v10.sum()) / N * 100.0
cov_v9  = (1 - ret_v9.mean()) * 100.0
cov_v10 = (1 - ret_v10.mean()) * 100.0

FP_v9  = int(np.sum((true_clean == 0) & (pred_v9_f == 1)))
FP_v10 = int(np.sum((true_clean == 0) & (pred_v10_f == 1)))

print("\n" + "=" * 135)
print("  [v9.0 vs v10.0 공식 성능 비교표 (해결책 1: False-Clean Asymmetric Loss + OOF Precision 제약 임계값)]")
print("=" * 135)
print(f"  {'버전 / 조건':<36s} | {'Clean F1':>9s} | {'Precision':>10s} | {'Recall':>8s} | {'4-EMR':>7s} | {'Coverage':>9s} | {'RETAKE%':>8s} | {'False Clean 건수':>16s}")
print("  " + "-" * 135)
print(f"  {'v9.0 베이스라인 (전수 100% 강제판정)':<36s} | {f1_v9_f:8.2f}% | {pre_v9_f:9.2f}% | {rec_v9_f:7.2f}% | {emr_v9_f:6.2f}% | {'100.0%':>9s} | {'  0.00%':>8s} | {FP_v9:>16d}건")
print(f"  {'v10.0 해결책1 (전수 100% 강제판정)':<36s} | {f1_v10_f:8.2f}% | {pre_v10_f:9.2f}% | {rec_v10_f:7.2f}% | {emr_v10_f:6.2f}% | {'100.0%':>9s} | {'  0.00%':>8s} | {FP_v10:>16d}건")
print("  " + "-" * 135)
print(f"  {'v9.0 베이스라인 (선택적 판정)':<36s} | {f1_v9_s:8.2f}% | {pre_v9_s:9.2f}% | {rec_v9_s:7.2f}% | {emr_v9_s:6.2f}% | {cov_v9:8.1f}% | {ret_rate_v9:7.2f}% | -")
print(f"  {'v10.0 해결책1 (선택적 판정)':<36s} | {f1_v10_s:8.2f}% | {pre_v10_s:9.2f}% | {rec_v10_s:7.2f}% | {emr_v10_s:6.2f}% | {cov_v10:8.1f}% | {ret_rate_v10:7.2f}% | -")
print("=" * 135)

# Clean Precision 개선량 출력
print(f"\n  [핵심 지표 변화량]")
print(f"    Clean Precision: {pre_v9_f:.2f}% -> {pre_v10_f:.2f}% (Delta {pre_v10_f - pre_v9_f:+.2f}%)")
print(f"    Clean Recall:    {rec_v9_f:.2f}% -> {rec_v10_f:.2f}% (Delta {rec_v10_f - rec_v9_f:+.2f}%)")
print(f"    Clean F1:        {f1_v9_f:.2f}% -> {f1_v10_f:.2f}% (Delta {f1_v10_f - f1_v9_f:+.2f}%)")
print(f"    4-Defect EMR:    {emr_v9_f:.2f}% -> {emr_v10_f:.2f}% (Delta {emr_v10_f - emr_v9_f:+.2f}%)")
print(f"    False Clean 건수: {FP_v9}건 -> {FP_v10}건 (Delta {FP_v10 - FP_v9:+d}건)")
print(f"    tau_clean:       0.380 -> {TAU_V10:.3f}")

# 4대 세부 결함별 성능
print("\n  [4대 세부 결함(c1~c4) 전수 강제 판정 성능 비교]")
defect_labels = ['c1 (척추 중립)', 'c2 (고개 정면)', 'c3 (무릎 외반)', 'c4 (발바닥 고정)']
for k in range(4):
    y_t = y_all[:, k]
    f1_v9_k  = f1_score(y_t, defs_v9_f[:, k], zero_division=0) * 100.0
    f1_v10_k = f1_score(y_t, defs_v10_f[:, k], zero_division=0) * 100.0
    pre_v9_k  = precision_score(y_t, defs_v9_f[:, k], zero_division=0) * 100.0
    pre_v10_k = precision_score(y_t, defs_v10_f[:, k], zero_division=0) * 100.0
    print(f"    {defect_labels[k]:<18s} | v9.0 F1: {f1_v9_k:5.2f}% (P:{pre_v9_k:5.2f}%) | v10.0 F1: {f1_v10_k:5.2f}% (P:{pre_v10_k:5.2f}%) | Delta{f1_v10_k-f1_v9_k:+.2f}%")

# 5대 시점별 성능
print("\n  [5대 시점별 Clean F1 및 Recall 비교 (전수 강제 판정)]")
view_order = [
    ('시점 A (우측 후방)', 'view5'),
    ('시점 B (우측 전방)', 'view2'),
    ('시점 C (정면)',       'view3'),
    ('시점 D (좌측 전방)', 'view4'),
    ('시점 E (좌측 후방)', 'view1'),
]
for v_name, v_code in view_order:
    m = (views == v_code)
    f1_v9_v  = f1_score(true_clean[m], pred_v9_f[m],  zero_division=0) * 100.0
    f1_v10_v = f1_score(true_clean[m], pred_v10_f[m], zero_division=0) * 100.0
    pre_v9_v  = precision_score(true_clean[m], pred_v9_f[m],  zero_division=0) * 100.0
    pre_v10_v = precision_score(true_clean[m], pred_v10_f[m], zero_division=0) * 100.0
    rec_v9_v  = recall_score(true_clean[m], pred_v9_f[m],  zero_division=0) * 100.0
    rec_v10_v = recall_score(true_clean[m], pred_v10_f[m], zero_division=0) * 100.0
    print(f"    {v_name:<20s} | v9.0  F1:{f1_v9_v:5.2f}% P:{pre_v9_v:5.2f}% R:{rec_v9_v:5.2f}% | v10.0 F1:{f1_v10_v:5.2f}% P:{pre_v10_v:5.2f}% R:{rec_v10_v:5.2f}% | DeltaF1:{f1_v10_v-f1_v9_v:+.2f}%")

print("\n" + "=" * 135)
print("  [평가 완료] v9.0 vs v10.0 성능 비교 종료")
print("=" * 135)
