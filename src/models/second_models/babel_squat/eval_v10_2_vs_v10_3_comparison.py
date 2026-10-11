# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_v10_2_vs_v10_3_comparison.py
Purpose: v10.2 (베이스라인) vs v10.3 (Skeletal Length Normalization Dual-Stream) 공식 성능 비교 평가
         - AI-Hub 2,605건 전수(9명 피험자 x 5대 전 시점) 대상
         - 전수 100% 강제 판정 및 선택적 판정(Selective) 동시 평가
         - 4대 결함(c1~c4) 및 공식 5대 시점(시점 A~E) 감사
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

from joint_meta_16way_moe_net import JointMeta16WayMoENetV10_1
from joint_meta_16way_net import decode_hybrid

META_PATH      = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH        = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH   = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH   = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
C2_FEATS_PATH  = r"D:\Data\피처_데이터\squat_c2_v7_19d_feats.npz"
NORM_PRED_PATH = os.path.join(CURRENT_DIR, "squat_skeletal_normalized_predictions_2605.npz")
TEMP_PARAM_PATH= os.path.join(CURRENT_DIR, "temperature_params.json")

WEIGHTS_V10_1  = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_1_weights.pth")
META_V10_2     = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_2_meta.json")

# 1. 데이터 로드
data    = np.load(META_PATH,      allow_pickle=True)
c4_data = np.load(C4_PATH,       allow_pickle=True)
q_data  = np.load(QUALITY_PATH,   allow_pickle=True)
k_data  = np.load(KINETIC_PATH,   allow_pickle=True)
c2_data = np.load(C2_FEATS_PATH,  allow_pickle=True)
norm_d  = np.load(NORM_PRED_PATH, allow_pickle=True)

views    = data['views']
subjects = data['subjects']
y_all    = data['y_all']
true_clean = data['true_clean']
N = len(true_clean)

# v10.2 확률 매트릭스 구성 (c2 v8 후방 불확실성 로직 적용)
probs_v10_2 = data['probs_defect'].copy()
p_c2_feats = c2_data['X_c2_v7_19d']
face_valid = (p_c2_feats[:, 14] > 0.5)
is_rear = (views == 'view1') | (views == 'view5')

# c2 v8.0 불확실성 사후 수축 적용
sigma2_c2 = np.where(is_rear & (~face_valid), 0.85, 0.05)
kappa_c2 = 1.0 / np.sqrt(1.0 + np.pi * sigma2_c2 / 8.0)
eps = 1e-7
p2_clip = np.clip(probs_v10_2[:, 1], eps, 1.0 - eps)
logit_c2 = np.log(p2_clip / (1.0 - p2_clip))
p2_cal_v8 = 1.0 / (1.0 + np.exp(-logit_c2 * kappa_c2))
probs_v10_2[:, 1] = p2_cal_v8

# v10.3 확률 매트릭스 구성 (하체 뼈 정규화 Dual-Stream 최적 융합)
probs_v10_3 = probs_v10_2.copy()
# c3 (무릎): 원천 70% + 하지 뼈 정규화 30% 최적 융합
probs_v10_3[:, 2] = 0.70 * probs_v10_2[:, 2] + 0.30 * norm_d['p_c3_norm']

# 운동사슬 피처 (v10.2 동일 유지로 불필요한 스케일 왜곡 차단)
kinetic_raw = k_data['X_kinetic_9d']
kinetic_raw = k_data['X_kinetic_9d']
kinetic_norm = norm_d['kinetic_9d_norm']

with open(TEMP_PARAM_PATH, 'r', encoding='utf-8') as f:
    t_cfg = json.load(f)
T_c1, T_c3, T_c4 = t_cfg['c1']['global_T'], t_cfg['c3']['global_T'], t_cfg['c4']['global_T']
view_T_c2 = t_cfg['c2']['view_T']

def build_39d(probs_in, kinetic_feats_in):
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

    c_fk   = kinetic_feats_in[:, 0:1]; c_kt = kinetic_feats_in[:, 1:2]
    c_tn   = kinetic_feats_in[:, 2:3]; c_ft = kinetic_feats_in[:, 3:4]
    d_valg = np.clip(kinetic_feats_in[:, 4:5]  / 0.40,          -2.5, 2.5)
    n_tilt = np.clip(kinetic_feats_in[:, 5:6]  / 0.50,           0.0, 2.5)
    n_valg = np.clip((kinetic_feats_in[:, 6:7] - 1.40) / 0.35,  -2.5, 2.5)
    a_heel = np.clip(kinetic_feats_in[:, 7:8]  / 0.60,           0.0, 2.5)
    p_lag  = np.clip(kinetic_feats_in[:, 8:9]  / 0.40,          -2.5, 2.5)
    kinetic_feats = np.hstack([c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag]).astype(np.float32)

    X_meta = np.hstack([p_cal, margins_cal, view_onehot, phys_feats, quality_feats, kinetic_feats]).astype(np.float32)
    return X_meta, p_cal

X_39d_v10_2, p_cal_v10_2 = build_39d(probs_v10_2, kinetic_raw)
X_39d_v10_3, p_cal_v10_3 = build_39d(probs_v10_3, kinetic_raw)

# 신경망 추론
model = JointMeta16WayMoENetV10_1(in_features=39, hidden_dim=48, dropout_rate=0.20)
model.load_state_dict(torch.load(WEIGHTS_V10_1, map_location='cpu', weights_only=True))
model.eval()

with torch.no_grad():
    l16_v10_2, l4_v10_2, _ = model(torch.from_numpy(X_39d_v10_2).float())
    p16_v10_2 = torch.softmax(l16_v10_2 / 1.20, dim=-1).numpy()
    p4_v10_2  = torch.sigmoid(l4_v10_2).numpy()

    l16_v10_3, l4_v10_3, _ = model(torch.from_numpy(X_39d_v10_3).float())
    p16_v10_3 = torch.softmax(l16_v10_3 / 1.20, dim=-1).numpy()
    p4_v10_3  = torch.sigmoid(l4_v10_3).numpy()

# 판정 엔진 함수
with open(META_V10_2, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
TAU_CLEAN = float(cfg.get('tau_clean', 0.36))
ETA = float(cfg.get('eta', 0.05))

X_c4 = c4_data['X_c4_v5_25d']
wobble = X_c4[:, 3]; jerk = X_c4[:, 15]; f0_rate = X_c4[:, 24]
q_overall_vec = q_data['X_quality_12d'][:, 2]

def run_predictions(p16, p4, p_cal):
    pred_full = np.zeros(N, dtype=int)
    defs_full = np.zeros((N, 4), dtype=int)
    is_retake = np.zeros(N, dtype=bool)
    pred_sel = np.zeros(N, dtype=int)
    defs_sel = np.zeros((N, 4), dtype=int)

    for i in range(N):
        s, defs, is_cln, p_cln = decode_hybrid(p16[i], p4[i], eta=ETA, tau_clean=TAU_CLEAN)
        if is_cln:
            pred_full[i] = 1
            defs_full[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.38):
            pred_full[i] = 1
            defs_full[i] = [0, 0, 0, 0]
        else:
            defs_full[i] = defs

    for i in range(N):
        if pred_full[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_full[i] = 1
            defs_full[i] = [0, 0, 0, 0]
        if pred_full[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_full[i] = 0

    for i in range(N):
        if q_overall_vec[i] < 0.05:
            is_retake[i] = True
            continue
        s, defs, is_cln, p_cln = decode_hybrid(p16[i], p4[i], eta=ETA, tau_clean=TAU_CLEAN)
        if 0.30 <= p_cln < TAU_CLEAN and not is_cln:
            is_retake[i] = True
            continue
        if is_cln:
            pred_sel[i] = 1
            defs_sel[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.38):
            pred_sel[i] = 1
            defs_sel[i] = [0, 0, 0, 0]
        else:
            defs_sel[i] = defs

    for i in range(N):
        if is_retake[i]: continue
        if pred_sel[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_sel[i] = 1
            defs_sel[i] = [0, 0, 0, 0]
        if pred_sel[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_sel[i] = 0

    return pred_full, defs_full, is_retake, pred_sel, defs_sel

pf_10_2, df_10_2, ret_10_2, ps_10_2, ds_10_2 = run_predictions(p16_v10_2, p4_v10_2, p_cal_v10_2)
pf_10_3, df_10_3, ret_10_3, ps_10_3, ds_10_3 = run_predictions(p16_v10_3, p4_v10_3, p_cal_v10_3)

def calc_metrics(y_true, y_pred):
    f1   = f1_score(y_true, y_pred, zero_division=0) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec  = recall_score(y_true, y_pred, zero_division=0) * 100.0
    return f1, prec, rec

def calc_emr(y_all_true, pred_defs):
    pred_16 = pred_defs[:, 0]*8 + pred_defs[:, 1]*4 + pred_defs[:, 2]*2 + pred_defs[:, 3]*1
    true_16 = y_all_true[:, 0]*8 + y_all_true[:, 1]*4 + y_all_true[:, 2]*2 + y_all_true[:, 3]*1
    return float(np.mean(pred_16 == true_16)) * 100.0

f1_10_2_f, p_10_2_f, r_10_2_f = calc_metrics(true_clean, pf_10_2)
f1_10_3_f, p_10_3_f, r_10_3_f = calc_metrics(true_clean, pf_10_3)
emr_10_2_f = calc_emr(y_all, df_10_2)
emr_10_3_f = calc_emr(y_all, df_10_3)

m2 = ~ret_10_2
m3 = ~ret_10_3
f1_10_2_s, p_10_2_s, r_10_2_s = calc_metrics(true_clean[m2], ps_10_2[m2])
f1_10_3_s, p_10_3_s, r_10_3_s = calc_metrics(true_clean[m3], ps_10_3[m3])
emr_10_2_s = calc_emr(y_all[m2], ds_10_2[m2])
emr_10_3_s = calc_emr(y_all[m3], ds_10_3[m3])

ret_rate_2 = float(ret_10_2.sum()) / N * 100.0
ret_rate_3 = float(ret_10_3.sum()) / N * 100.0
cov_2 = (1 - ret_10_2.mean()) * 100.0
cov_3 = (1 - ret_10_3.mean()) * 100.0

FP_10_2 = int(np.sum((true_clean == 0) & (pf_10_2 == 1)))
FP_10_3 = int(np.sum((true_clean == 0) & (pf_10_3 == 1)))

print("\n" + "=" * 135)
print("  [v10.2 베이스라인 vs v10.3 Skeletal Length Normalization Dual-Stream 공식 성능 비교표]")
print("=" * 135)
print(f"  {'버전 / 조건':<36s} | {'Clean F1':>9s} | {'Precision':>10s} | {'Recall':>8s} | {'4-EMR':>7s} | {'Coverage':>9s} | {'RETAKE%':>8s} | {'False Clean':>14s}")
print("  " + "-" * 135)
print(f"  {'v10.2 (전수 100% 강제 판정)':<36s} | {f1_10_2_f:8.2f}% | {p_10_2_f:9.2f}% | {r_10_2_f:7.2f}% | {emr_10_2_f:6.2f}% | {'100.0%':>9s} | {'  0.00%':>8s} | {FP_10_2:>12d}건")
print(f"  {'v10.3 (전수 100% 강제 판정)':<36s} | {f1_10_3_f:8.2f}% | {p_10_3_f:9.2f}% | {r_10_3_f:7.2f}% | {emr_10_3_f:6.2f}% | {'100.0%':>9s} | {'  0.00%':>8s} | {FP_10_3:>12d}건")
print("  " + "-" * 135)
print(f"  {'v10.2 (선택적 판정 / c2 v8 탑재)':<36s} | {f1_10_2_s:8.2f}% | {p_10_2_s:9.2f}% | {r_10_2_s:7.2f}% | {emr_10_2_s:6.2f}% | {cov_2:8.1f}% | {ret_rate_2:7.2f}% | -")
print(f"  {'v10.3 (선택적 판정 / 뼈 정규화 Dual)':<36s} | {f1_10_3_s:8.2f}% | {p_10_3_s:9.2f}% | {r_10_3_s:7.2f}% | {emr_10_3_s:6.2f}% | {cov_3:8.1f}% | {ret_rate_3:7.2f}% | -")
print("=" * 135)

print(f"\n  [핵심 지표 변화량 (선택적 판정 기준)]")
print(f"    Clean F1-Score: {f1_10_2_s:.2f}% -> {f1_10_3_s:.2f}% (Δ{f1_10_3_s - f1_10_2_s:+.2f}%p)")
print(f"    Clean Precision:{p_10_2_s:.2f}% -> {p_10_3_s:.2f}% (Δ{p_10_3_s - p_10_2_s:+.2f}%p)")
print(f"    Clean Recall:   {r_10_2_s:.2f}% -> {r_10_3_s:.2f}% (Δ{r_10_3_s - r_10_2_s:+.2f}%p)")
print(f"    4-Defect EMR:   {emr_10_2_s:.2f}% -> {emr_10_3_s:.2f}% (Δ{emr_10_3_s - emr_10_2_s:+.2f}%p)")
print(f"    RETAKE 보류율:  {ret_rate_2:.2f}% -> {ret_rate_3:.2f}% (Δ{ret_rate_3 - ret_rate_2:+.2f}%p)")

# 4대 세부 결함 F1 비교
print("\n  [4대 세부 결함(c1~c4) 선택적 판정 F1 비교]")
defect_labels = ['c1 (척추 중립)', 'c2 (고개 정면)', 'c3 (무릎 외반)', 'c4 (발바닥 고정)']
for k in range(4):
    y_t2 = y_all[m2, k]; y_t3 = y_all[m3, k]
    f1_k2 = f1_score(y_t2, ds_10_2[m2, k], zero_division=0) * 100.0
    f1_k3 = f1_score(y_t3, ds_10_3[m3, k], zero_division=0) * 100.0
    print(f"    {defect_labels[k]:<18s} | v10.2: {f1_k2:5.2f}% | v10.3: {f1_k3:5.2f}% | Δ{f1_k3 - f1_k2:+.2f}%p")

# 5대 카메라 시점별 비교
print("\n  [5대 카메라 시점별 Clean F1 비교 (선택적 판정)]")
view_order = [
    ('시점 A (우측 후방 45°)', 'view5'),
    ('시점 B (우측 전방 45°)', 'view2'),
    ('시점 C (정면 0°)',       'view3'),
    ('시점 D (좌측 전방 45°)', 'view4'),
    ('시점 E (좌측 후방 45°)', 'view1'),
]
for v_name, v_code in view_order:
    mv2 = m2 & (views == v_code)
    mv3 = m3 & (views == v_code)
    f1_v2 = f1_score(true_clean[mv2], ps_10_2[mv2], zero_division=0) * 100.0
    f1_v3 = f1_score(true_clean[mv3], ps_10_3[mv3], zero_division=0) * 100.0
    print(f"    {v_name:<24s} | v10.2: {f1_v2:5.2f}% | v10.3: {f1_v3:5.2f}% | Δ{f1_v3 - f1_v2:+.2f}%p")

print("\n" + "=" * 135)
