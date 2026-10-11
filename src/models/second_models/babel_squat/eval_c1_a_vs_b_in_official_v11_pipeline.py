# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_c1_a_vs_b_in_official_v11_pipeline.py
Purpose: 공식 v11.0 챔피언 파이프라인에서
         Model-A (기존 AI-Hub 보간 c1) vs Model-B (신규 SpinePose 순수 비전 c1)
         완전 동일 파이프라인 하에서 엄밀 대조 평가
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

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from joint_meta_16way_moe_net import JointMeta16WayMoENetV10_1
from joint_meta_16way_net import decode_hybrid

META_PATH        = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH          = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH     = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH     = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
C2_FEATS_PATH    = r"D:\Data\피처_데이터\squat_c2_v7_19d_feats.npz"
NORM_PRED_PATH   = os.path.join(CURRENT_DIR, "squat_skeletal_normalized_predictions_2605.npz")
TEMP_PARAM_PATH  = os.path.join(CURRENT_DIR, "temperature_params.json")
SCALER_PATH_V11  = os.path.join(CURRENT_DIR, "view_vector_scaler_params.json")
OOF_RES_PATH     = os.path.join(CURRENT_DIR, "stgcn_transformer_oof_residuals.npz")
OOF_B_PATH       = r"D:\Data\피처_데이터\squat_c1_oof_predictions_a_and_b.npz"

WEIGHTS_V11   = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_1_weights.pth")
OUT_JSON_PATH = os.path.join(CURRENT_DIR, "squat_clean_classification_a_vs_b_official.json")

print("=" * 115)
print("  [공식 v11.0 통합 파이프라인] Model-A(기존 AI-Hub) vs Model-B(신규 SpinePose) 성능 정밀 비교 평가")
print("  - 대상 데이터: AI-Hub 바벨 스쿼트 2,605건 전수 (9인 전 피험자, 5대 전 시점)")
print("=" * 115)

# 1. 데이터 로드
data    = np.load(META_PATH,      allow_pickle=True)
c4_data = np.load(C4_PATH,       allow_pickle=True)
q_data  = np.load(QUALITY_PATH,   allow_pickle=True)
k_data  = np.load(KINETIC_PATH,   allow_pickle=True)
c2_data = np.load(C2_FEATS_PATH,  allow_pickle=True)
norm_d  = np.load(NORM_PRED_PATH, allow_pickle=True)
oof_res = np.load(OOF_RES_PATH,   allow_pickle=True)
b_oof_d = np.load(OOF_B_PATH,     allow_pickle=True)

views    = data['views']
subjects = data['subjects']
y_all    = data['y_all']
true_clean = data['true_clean']
N = len(true_clean)

# Model-A 베이스라인 확률 (기존 공식 마스터 챔피언)
probs_base_A = data['probs_defect'].copy()

# c2, c3 공식 보정 적용
p_c2_feats = c2_data['X_c2_v7_19d']
face_valid = (p_c2_feats[:, 14] > 0.5)
is_rear = (views == 'view1') | (views == 'view5')
sigma2_c2 = np.where(is_rear & (~face_valid), 0.85, 0.05)
kappa_c2 = 1.0 / np.sqrt(1.0 + np.pi * sigma2_c2 / 8.0)
eps = 1e-7
p2_clip = np.clip(probs_base_A[:, 1], eps, 1.0 - eps)
logit_c2 = np.log(p2_clip / (1.0 - p2_clip))
p2_cal_v8 = 1.0 / (1.0 + np.exp(-logit_c2 * kappa_c2))
probs_base_A[:, 1] = p2_cal_v8
probs_base_A[:, 2] = 0.70 * probs_base_A[:, 2] + 0.30 * norm_d['p_c3_norm']

# Model-B 베이스라인 확률 (c1만 SpinePose 순수 비전 OOF로 교체, 나머지는 100% 동일)
probs_base_B = probs_base_A.copy()
probs_base_B[:, 0] = b_oof_d['oof_p_B']

# 온도 보정 파라미터 로드
with open(TEMP_PARAM_PATH, 'r', encoding='utf-8') as f:
    t_cfg = json.load(f)
T_c1, T_c3, T_c4 = t_cfg['c1']['global_T'], t_cfg['c3']['global_T'], t_cfg['c4']['global_T']
view_T_c2 = t_cfg['c2']['view_T']

# 공통 피처 사전 추출
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

kinetic_raw = k_data['X_kinetic_9d']
c_fk   = kinetic_raw[:, 0:1]; c_kt = kinetic_raw[:, 1:2]
c_tn   = kinetic_raw[:, 2:3]; c_ft = kinetic_raw[:, 3:4]
d_valg = np.clip(kinetic_raw[:, 4:5]  / 0.40,          -2.5, 2.5)
n_tilt = np.clip(kinetic_raw[:, 5:6]  / 0.50,           0.0, 2.5)
n_valg = np.clip((kinetic_raw[:, 6:7] - 1.40) / 0.35,  -2.5, 2.5)
a_heel = np.clip(kinetic_raw[:, 7:8]  / 0.60,           0.0, 2.5)
p_lag  = np.clip(kinetic_raw[:, 8:9]  / 0.40,          -2.5, 2.5)
kinetic_feats = np.hstack([c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag]).astype(np.float32)

q_overall_vec = q_data['X_quality_12d'][:, 2]

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
    X_meta = np.hstack([p_cal, margins_cal, view_onehot, phys_feats, quality_feats, kinetic_feats]).astype(np.float32)
    return X_meta, p_cal

# v11 모델 로드
model_v11 = JointMeta16WayMoENetV10_1(in_features=39, hidden_dim=48, dropout_rate=0.20)
model_v11.load_state_dict(torch.load(WEIGHTS_V11, map_location='cpu', weights_only=True))
model_v11.eval()

# 시점별 벡터 스케일러 파라미터 로드
with open(SCALER_PATH_V11, 'r', encoding='utf-8') as f:
    scaler_v11 = json.load(f)

oof_dz16 = oof_res['oof_delta_z16']
is_front = (views == 'view2') | (views == 'view3') | (views == 'view4')
g_vec = np.where(is_front, 0.08, 0.00)[:, None]

def apply_scaling(z_raw, scaler_params):
    z_cal = np.zeros_like(z_raw)
    for v_name in ['view5', 'view2', 'view3', 'view4', 'view1']:
        m = (views == v_name)
        s_v = np.array(scaler_params[v_name]['scale'], dtype=np.float32)
        b_v = np.array(scaler_params[v_name]['bias'], dtype=np.float32)
        z_cal[m] = z_raw[m] * s_v + b_v
    z_fused = z_cal + g_vec * oof_dz16
    p16 = np.exp(z_fused - np.max(z_fused, axis=-1, keepdims=True))
    p16 = p16 / np.sum(p16, axis=-1, keepdims=True)
    return p16

H_max = np.log(16.0)
def compute_risk(p_matrix):
    log_p = np.log(np.clip(p_matrix, 1e-8, 1.0))
    ent = -np.sum(p_matrix * log_p, axis=-1)
    sp = np.sort(p_matrix, axis=-1)
    margin = sp[:, -1] - sp[:, -2]
    return (ent / H_max) * (1.0 - margin)

def run_predictions(p16_in, p4_in, risk_in, p_cal_in, tau_cln=0.36, r_thresh=0.45, eta=0.05):
    pred_full = np.zeros(N, dtype=int)
    defs_full = np.zeros((N, 4), dtype=int)
    is_retake = np.zeros(N, dtype=bool)
    pred_sel = np.zeros(N, dtype=int)
    defs_sel = np.zeros((N, 4), dtype=int)

    for i in range(N):
        s, defs, is_cln, p_cln = decode_hybrid(p16_in[i], p4_in[i], eta=eta, tau_clean=tau_cln)
        if is_cln:
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal_in[i, 0] <= 0.40) and (p_cal_in[i, 2] <= 0.40) and (p_cal_in[i, 3] <= 0.40) and (p_cal_in[i, 1] < 0.38):
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        else:
            defs_full[i] = defs

    for i in range(N):
        if pred_full[i] == 0 and max(p_cal_in[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        if pred_full[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_full[i] = 0

    for i in range(N):
        if q_overall_vec[i] < 0.05:
            is_retake[i] = True; continue
        s, defs, is_cln, p_cln = decode_hybrid(p16_in[i], p4_in[i], eta=eta, tau_clean=tau_cln)
        if risk_in[i] > r_thresh and not is_cln:
            is_retake[i] = True; continue
        if is_cln:
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal_in[i, 0] <= 0.40) and (p_cal_in[i, 2] <= 0.40) and (p_cal_in[i, 3] <= 0.40) and (p_cal_in[i, 1] < 0.38):
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        else:
            defs_sel[i] = defs

    for i in range(N):
        if is_retake[i]: continue
        if pred_sel[i] == 0 and max(p_cal_in[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        if pred_sel[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_sel[i] = 0

    return pred_full, defs_full, is_retake, pred_sel, defs_sel

def calc_metrics(y_true, y_pred):
    f1   = f1_score(y_true, y_pred, zero_division=0) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec  = recall_score(y_true, y_pred, zero_division=0) * 100.0
    return f1, prec, rec

def calc_emr(y_all_true, pred_defs):
    pred_16 = pred_defs[:, 0]*8 + pred_defs[:, 1]*4 + pred_defs[:, 2]*2 + pred_defs[:, 3]*1
    true_16 = y_all_true[:, 0]*8 + y_all_true[:, 1]*4 + y_all_true[:, 2]*2 + y_all_true[:, 3]*1
    return float(np.mean(pred_16 == true_16)) * 100.0

# -------------------------------------------------------------
# [1] Model-A 평가 (기존 공식 챔피언)
# -------------------------------------------------------------
X_39d_A, p_cal_A = build_39d(probs_base_A)
with torch.no_grad():
    raw_l16_A, raw_l4_A, _ = model_v11(torch.from_numpy(X_39d_A).float())
    z_raw_16_A = raw_l16_A.numpy()
    p4_pred_A  = torch.sigmoid(raw_l4_A).numpy()
p16_A = apply_scaling(z_raw_16_A, scaler_v11)
risk_A = compute_risk(p16_A)
pf_A, df_A, ret_A, ps_A, ds_A = run_predictions(p16_A, p4_pred_A, risk_A, p_cal_A, tau_cln=0.36, r_thresh=0.45, eta=0.05)

# -------------------------------------------------------------
# [2] Model-B 평가 (신규 SpinePose 순수 비전 c1 투입)
# -------------------------------------------------------------
X_39d_B, p_cal_B = build_39d(probs_base_B)
with torch.no_grad():
    raw_l16_B, raw_l4_B, _ = model_v11(torch.from_numpy(X_39d_B).float())
    z_raw_16_B = raw_l16_B.numpy()
    p4_pred_B  = torch.sigmoid(raw_l4_B).numpy()
p16_B = apply_scaling(z_raw_16_B, scaler_v11)
risk_B = compute_risk(p16_B)
pf_B, df_B, ret_B, ps_B, ds_B = run_predictions(p16_B, p4_pred_B, risk_B, p_cal_B, tau_cln=0.36, r_thresh=0.45, eta=0.05)

# -------------------------------------------------------------
# 메트릭 계산
# -------------------------------------------------------------
# 강제 판정
f1_A_f, p_A_f, r_A_f = calc_metrics(true_clean, pf_A)
f1_B_f, p_B_f, r_B_f = calc_metrics(true_clean, pf_B)
emr_A_f = calc_emr(y_all, df_A)
emr_B_f = calc_emr(y_all, df_B)
fp_A_f = int(np.sum((true_clean == 0) & (pf_A == 1)))
fp_B_f = int(np.sum((true_clean == 0) & (pf_B == 1)))

# 선택적 판정
mA = ~ret_A
mB = ~ret_B
f1_A_s, p_A_s, r_A_s = calc_metrics(true_clean[mA], ps_A[mA])
f1_B_s, p_B_s, r_B_s = calc_metrics(true_clean[mB], ps_B[mB])
emr_A_s = calc_emr(y_all[mA], ds_A[mA])
emr_B_s = calc_emr(y_all[mB], ds_B[mB])
cov_A = float(mA.sum()) / N * 100.0
cov_B = float(mB.sum()) / N * 100.0
ret_A_rate = 100.0 - cov_A
ret_B_rate = 100.0 - cov_B
fp_A_s = int(np.sum((true_clean[mA] == 0) & (ps_A[mA] == 1)))
fp_B_s = int(np.sum((true_clean[mB] == 0) & (ps_B[mB] == 1)))

# 서브모델 F1
c_names = ['c1 (척추 중립)', 'c2 (고개 정면)', 'c3 (무릎 외반)', 'c4 (발바닥 고정)']
c_f1_A = [f1_score(y_all[mA, k], ds_A[mA, k], zero_division=0)*100.0 for k in range(4)]
c_f1_B = [f1_score(y_all[mB, k], ds_B[mB, k], zero_division=0)*100.0 for k in range(4)]

# 시점별 분석
view_order = [
    ('시점 A (우측 후방)', 'view5'),
    ('시점 B (우측 전방)', 'view2'),
    ('시점 C (정면)',       'view3'),
    ('시점 D (좌측 전방)', 'view4'),
    ('시점 E (좌측 후방)', 'view1'),
]
views_detail = {}
for vn, vk in view_order:
    m_vA = mA & (views == vk)
    m_vB = mB & (views == vk)
    f1_vA = f1_score(true_clean[m_vA], ps_A[m_vA], zero_division=0) * 100.0
    f1_vB = f1_score(true_clean[m_vB], ps_B[m_vB], zero_division=0) * 100.0
    emr_vA = calc_emr(y_all[m_vA], ds_A[m_vA])
    emr_vB = calc_emr(y_all[m_vB], ds_B[m_vB])
    fp_vA = int(np.sum((true_clean[m_vA] == 0) & (ps_A[m_vA] == 1)))
    fp_vB = int(np.sum((true_clean[m_vB] == 0) & (ps_B[m_vB] == 1)))
    views_detail[vk] = {
        'name': vn,
        'f1_A': f1_vA, 'f1_B': f1_vB, 'delta_f1': f1_vB - f1_vA,
        'emr_A': emr_vA, 'emr_B': emr_vB, 'delta_emr': emr_vB - emr_vA,
        'fp_A': fp_vA, 'fp_B': fp_vB
    }

print("\n" + "=" * 135)
print("  [공식 v11.0 챔피언 파이프라인 하 Model-A(기존 AI-Hub) vs Model-B(신규 SpinePose) 성능 정밀 비교표]")
print("=" * 135)
print(f"  {'모델 버전 / 조건':<34s} | {'Clean F1':>9s} | {'Precision':>10s} | {'Recall':>8s} | {'4-EMR':>7s} | {'Coverage':>9s} | {'RETAKE%':>8s} | {'False Clean':>12s}")
print("  " + "-" * 135)
print(f"  {'Model-A (기존 전수 강제판정)':<34s} | {f1_A_f:8.2f}% | {p_A_f:9.2f}% | {r_A_f:7.2f}% | {emr_A_f:6.2f}% | {'100.0%':>9s} | {'0.00%':>8s} | {fp_A_f:>10d}건")
print(f"  {'Model-B (SpinePose 전수 강제판정)':<34s} | {f1_B_f:8.2f}% | {p_B_f:9.2f}% | {r_B_f:7.2f}% | {emr_B_f:6.2f}% | {'100.0%':>9s} | {'0.00%':>8s} | {fp_B_f:>10d}건")
print("  " + "-" * 135)
print(f"  {'Model-A (기존 선택적 판정 챔피언)':<34s} | {f1_A_s:8.2f}% | {p_A_s:9.2f}% | {r_A_s:7.2f}% | {emr_A_s:6.2f}% | {cov_A:8.1f}% | {ret_A_rate:7.2f}% | {fp_A_s:>10d}건")
print(f"  {'Model-B (SpinePose 선택적 판정)':<34s} | {f1_B_s:8.2f}% | {p_B_s:9.2f}% | {r_B_s:7.2f}% | {emr_B_s:6.2f}% | {cov_B:8.1f}% | {ret_B_rate:7.2f}% | {fp_B_s:>10d}건")
print("=" * 135)

print("\n  [주요 성능 지표 변화 (선택적 판정)]")
print(f"    • Clean F1-Score:            {f1_A_s:.2f}% → {f1_B_s:.2f}% (Δ{f1_B_s - f1_A_s:+.2f}%p {'▲' if f1_B_s >= f1_A_s else '▼'})")
print(f"    • 4-Defect EMR (완전 일치율): {emr_A_s:.2f}% → {emr_B_s:.2f}% (Δ{emr_B_s - emr_A_s:+.2f}%p {'▲' if emr_B_s >= emr_A_s else '▼'})")
print(f"    • Clean Precision:           {p_A_s:.2f}% → {p_B_s:.2f}% (Δ{p_B_s - p_A_s:+.2f}%p {'▲' if p_B_s >= p_A_s else '▼'})")
print(f"    • Clean Recall:              {r_A_s:.2f}% → {r_B_s:.2f}% (Δ{r_B_s - r_A_s:+.2f}%p)")
print(f"    • Coverage (판정률):         {cov_A:.2f}% → {cov_B:.2f}% (Δ{cov_B - cov_A:+.2f}%p)")
print(f"    • False Clean 건수:          {fp_A_s}건 → {fp_B_s}건 (Δ{fp_B_s - fp_A_s:+d}건)")

print("\n  [4대 세부 결함(c1~c4) 선택적 판정 F1 비교]")
for k in range(4):
    print(f"    • {c_names[k]:<18s} | Model-A: {c_f1_A[k]:5.2f}% → Model-B: {c_f1_B[k]:5.2f}% (Δ{c_f1_B[k] - c_f1_A[k]:+.2f}%p)")

print("\n  [5대 시점별 Clean F1 및 4-EMR 비교 (선택적 판정)]")
for vk, vd in views_detail.items():
    print(f"    • {vd['name']:<16s} ({vk}) | Clean F1: {vd['f1_A']:5.2f}% → {vd['f1_B']:5.2f}% (Δ{vd['delta_f1']:+.2f}%p) | 4-EMR: {vd['emr_A']:5.2f}% → {vd['emr_B']:5.2f}% (Δ{vd['delta_emr']:+.2f}%p) | FP: {vd['fp_A']}→{vd['fp_B']}건")

# JSON 저장
out_data = {
    'force': {
        'model_a': {'f1': f1_A_f, 'precision': p_A_f, 'recall': r_A_f, 'emr': emr_A_f, 'fp': fp_A_f},
        'model_b': {'f1': f1_B_f, 'precision': p_B_f, 'recall': r_B_f, 'emr': emr_B_f, 'fp': fp_B_f},
        'delta': {'f1': f1_B_f - f1_A_f, 'emr': emr_B_f - emr_A_f, 'fp': fp_B_f - fp_A_f}
    },
    'selective': {
        'model_a': {'f1': f1_A_s, 'precision': p_A_s, 'recall': r_A_s, 'emr': emr_A_s, 'cov': cov_A, 'fp': fp_A_s},
        'model_b': {'f1': f1_B_s, 'precision': p_B_s, 'recall': r_B_s, 'emr': emr_B_s, 'cov': cov_B, 'fp': fp_B_s},
        'delta': {'f1': f1_B_s - f1_A_s, 'emr': emr_B_s - emr_A_s, 'cov': cov_B - cov_A, 'fp': fp_B_s - fp_A_s}
    },
    'submodels': {
        'model_a': {c_names[k]: c_f1_A[k] for k in range(4)},
        'model_b': {c_names[k]: c_f1_B[k] for k in range(4)}
    },
    'views': views_detail
}

with open(OUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)
print(f"\n[OK] 결과 JSON 저장 완료: {OUT_JSON_PATH}")
