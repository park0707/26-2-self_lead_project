# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_v11_vs_v12_3_comparison.py
Purpose: v11.0 (공식 마스터 챔피언) vs v12.3 (전체 해결책 통합 마스터)
         - [해결책 1]: Factorized Pairwise Energy Head
         - [해결책 2]: Soft-EMR + Soft-Clean F1 지표 정렬 복합 손실
         - [해결책 3]: 39D 도메인 토큰화 & FiLM Gated Residual 중간 융합
         - [해결책 4]: OOF 기반 위험 제어형 3-State 동적 게이트
         - [사후 보정]: v12.3 전용 5대 시점별 16D 벡터 스케일러
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
from joint_meta_16way_v12_3_net import JointMeta16WayMoENetV12_3
from joint_meta_16way_net import decode_hybrid

META_PATH        = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH          = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH     = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH     = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
C2_FEATS_PATH    = r"D:\Data\피처_데이터\squat_c2_v7_19d_feats.npz"
NORM_PRED_PATH   = os.path.join(CURRENT_DIR, "squat_skeletal_normalized_predictions_2605.npz")
TEMP_PARAM_PATH  = os.path.join(CURRENT_DIR, "temperature_params.json")
SCALER_PATH_V11  = os.path.join(CURRENT_DIR, "view_vector_scaler_params.json")
SCALER_PATH_V12  = os.path.join(CURRENT_DIR, "view_vector_scaler_params_v12_3.json")
GATE_PARAMS_PATH = os.path.join(CURRENT_DIR, "optimal_gate_params_v12_3.json")
OOF_RES_PATH     = os.path.join(CURRENT_DIR, "stgcn_transformer_oof_residuals.npz")

WEIGHTS_V11   = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_1_weights.pth")
WEIGHTS_V12_3 = os.path.join(CURRENT_DIR, "joint_meta_16way_v12_3_weights.pth")
OUT_JSON_PATH = os.path.join(CURRENT_DIR, "v11_vs_v12_3_comparison_results.json")

print("=" * 115)
print("  [v11.0 vs v12.3] 전체 해결책 통합 마스터 공식 성능 비교 평가 파이프라인 가동")
print("  - 대상 데이터: AI-Hub 바벨 스쿼트 2,605건 전수 (9인 전 피험자, 5대 전 시점)")
print("  - 해결책 1(에너지 헤드) + 해결책 2(Soft-EMR) + 해결책 3(FiLM 융합) + 해결책 4(동적 게이트)")
print("=" * 115)

# 1. 원천 데이터 로드
data    = np.load(META_PATH,      allow_pickle=True)
c4_data = np.load(C4_PATH,       allow_pickle=True)
q_data  = np.load(QUALITY_PATH,   allow_pickle=True)
k_data  = np.load(KINETIC_PATH,   allow_pickle=True)
c2_data = np.load(C2_FEATS_PATH,  allow_pickle=True)
norm_d  = np.load(NORM_PRED_PATH, allow_pickle=True)
oof_res = np.load(OOF_RES_PATH,   allow_pickle=True)

views    = data['views']
subjects = data['subjects']
y_all    = data['y_all']
true_clean = data['true_clean']
N = len(true_clean)

# 베이스라인 확률 구성
probs_base = data['probs_defect'].copy()
p_c2_feats = c2_data['X_c2_v7_19d']
face_valid = (p_c2_feats[:, 14] > 0.5)
is_rear = (views == 'view1') | (views == 'view5')

sigma2_c2 = np.where(is_rear & (~face_valid), 0.85, 0.05)
kappa_c2 = 1.0 / np.sqrt(1.0 + np.pi * sigma2_c2 / 8.0)
eps = 1e-7
p2_clip = np.clip(probs_base[:, 1], eps, 1.0 - eps)
logit_c2 = np.log(p2_clip / (1.0 - p2_clip))
p2_cal_v8 = 1.0 / (1.0 + np.exp(-logit_c2 * kappa_c2))
probs_base[:, 1] = p2_cal_v8
probs_base[:, 2] = 0.70 * probs_base[:, 2] + 0.30 * norm_d['p_c3_norm']

with open(TEMP_PARAM_PATH, 'r', encoding='utf-8') as f:
    t_cfg = json.load(f)
T_c1, T_c3, T_c4 = t_cfg['c1']['global_T'], t_cfg['c3']['global_T'], t_cfg['c4']['global_T']
view_T_c2 = t_cfg['c2']['view_T']

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

    kinetic_raw = k_data['X_kinetic_9d']
    c_fk   = kinetic_raw[:, 0:1]; c_kt = kinetic_raw[:, 1:2]
    c_tn   = kinetic_raw[:, 2:3]; c_ft = kinetic_raw[:, 3:4]
    d_valg = np.clip(kinetic_raw[:, 4:5]  / 0.40,          -2.5, 2.5)
    n_tilt = np.clip(kinetic_raw[:, 5:6]  / 0.50,           0.0, 2.5)
    n_valg = np.clip((kinetic_raw[:, 6:7] - 1.40) / 0.35,  -2.5, 2.5)
    a_heel = np.clip(kinetic_raw[:, 7:8]  / 0.60,           0.0, 2.5)
    p_lag  = np.clip(kinetic_raw[:, 8:9]  / 0.40,          -2.5, 2.5)
    kinetic_feats = np.hstack([c_fk, c_kt, c_tn, c_ft, d_valg, n_tilt, n_valg, a_heel, p_lag]).astype(np.float32)

    X_meta = np.hstack([p_cal, margins_cal, view_onehot, phys_feats, quality_feats, kinetic_feats]).astype(np.float32)
    return X_meta, p_cal

X_39d, p_cal = build_39d(probs_base)

# 2. v11.0 추론
model_v11 = JointMeta16WayMoENetV10_1(in_features=39, hidden_dim=48, dropout_rate=0.20)
model_v11.load_state_dict(torch.load(WEIGHTS_V11, map_location='cpu', weights_only=True))
model_v11.eval()

with torch.no_grad():
    raw_l16_v11, raw_l4_v11, _ = model_v11(torch.from_numpy(X_39d).float())
    z_raw_16_v11 = raw_l16_v11.numpy()
    p4_pred_v11  = torch.sigmoid(raw_l4_v11).numpy()

# 3. v12.3 추론 (FiLM + Energy Head)
model_v12_3 = JointMeta16WayMoENetV12_3(in_features=39, hidden_dim=48, dropout_rate=0.20)
model_v12_3.load_state_dict(torch.load(WEIGHTS_V12_3, map_location='cpu', weights_only=True))
model_v12_3.eval()

with torch.no_grad():
    raw_l16_v12, raw_l4_v12, _ = model_v12_3(torch.from_numpy(X_39d).float())
    z_raw_16_v12 = raw_l16_v12.numpy()
    p4_pred_v12  = torch.sigmoid(raw_l4_v12).numpy()

# 4. 시점별 16D 벡터 스케일러 적용
with open(SCALER_PATH_V11, 'r', encoding='utf-8') as f:
    scaler_v11 = json.load(f)

with open(SCALER_PATH_V12, 'r', encoding='utf-8') as f:
    scaler_v12 = json.load(f)

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

p16_v11   = apply_scaling(z_raw_16_v11, scaler_v11)
p16_v12_3 = apply_scaling(z_raw_16_v12, scaler_v12)

# 위험도 계산
H_max = np.log(16.0)
def compute_risk(p_matrix):
    log_p = np.log(np.clip(p_matrix, 1e-8, 1.0))
    ent = -np.sum(p_matrix * log_p, axis=-1)
    sp = np.sort(p_matrix, axis=-1)
    margin = sp[:, -1] - sp[:, -2]
    return (ent / H_max) * (1.0 - margin)

risk_v11   = compute_risk(p16_v11)
risk_v12_3 = compute_risk(p16_v12_3)

# 5. [해결책 4] 최적 게이트 파라미터 로드
tau_clean_v12 = 0.36
risk_thresh_v12 = 0.70
eta_v12 = 0.05
if os.path.exists(GATE_PARAMS_PATH):
    with open(GATE_PARAMS_PATH, 'r', encoding='utf-8') as f:
        gp = json.load(f)
        tau_clean_v12 = gp.get('tau_clean', 0.36)
        risk_thresh_v12 = gp.get('risk_thresh', 0.70)
        eta_v12 = gp.get('eta', 0.05)

X_c4 = c4_data['X_c4_v5_25d']
wobble = X_c4[:, 3]; jerk = X_c4[:, 15]; f0_rate = X_c4[:, 24]
q_overall_vec = q_data['X_quality_12d'][:, 2]

def run_predictions(p16_in, p4_in, risk_in, tau_cln=0.36, r_thresh=0.45, eta=0.05):
    pred_full = np.zeros(N, dtype=int)
    defs_full = np.zeros((N, 4), dtype=int)
    is_retake = np.zeros(N, dtype=bool)
    pred_sel = np.zeros(N, dtype=int)
    defs_sel = np.zeros((N, 4), dtype=int)

    for i in range(N):
        s, defs, is_cln, p_cln = decode_hybrid(p16_in[i], p4_in[i], eta=eta, tau_clean=tau_cln)
        if is_cln:
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.38):
            pred_full[i] = 1; defs_full[i] = [0, 0, 0, 0]
        else:
            defs_full[i] = defs

    for i in range(N):
        if pred_full[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
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
        elif views[i] in ['view1', 'view5'] and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.38):
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        else:
            defs_sel[i] = defs

    for i in range(N):
        if is_retake[i]: continue
        if pred_sel[i] == 0 and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_sel[i] = 1; defs_sel[i] = [0, 0, 0, 0]
        if pred_sel[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_sel[i] = 0

    return pred_full, defs_full, is_retake, pred_sel, defs_sel

pf_v11, df_v11, ret_v11, ps_v11, ds_v11 = run_predictions(p16_v11, p4_pred_v11, risk_v11, tau_cln=0.36, r_thresh=0.45, eta=0.05)
pf_v12, df_v12, ret_v12, ps_v12, ds_v12 = run_predictions(p16_v12_3, p4_pred_v12, risk_v12_3, tau_cln=tau_clean_v12, r_thresh=risk_thresh_v12, eta=eta_v12)

def calc_metrics(y_true, y_pred):
    f1   = f1_score(y_true, y_pred, zero_division=0) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec  = recall_score(y_true, y_pred, zero_division=0) * 100.0
    return f1, prec, rec

def calc_emr(y_all_true, pred_defs):
    pred_16 = pred_defs[:, 0]*8 + pred_defs[:, 1]*4 + pred_defs[:, 2]*2 + pred_defs[:, 3]*1
    true_16 = y_all_true[:, 0]*8 + y_all_true[:, 1]*4 + y_all_true[:, 2]*2 + y_all_true[:, 3]*1
    return float(np.mean(pred_16 == true_16)) * 100.0

# 1. 전수 강제 판정
f1_11_f, p_11_f, r_11_f = calc_metrics(true_clean, pf_v11)
f1_12_f, p_12_f, r_12_f = calc_metrics(true_clean, pf_v12)
emr_11_f = calc_emr(y_all, df_v11)
emr_12_f = calc_emr(y_all, df_v12)
fp_11_f = int(np.sum((true_clean == 0) & (pf_v11 == 1)))
fp_12_f = int(np.sum((true_clean == 0) & (pf_v12 == 1)))

# 2. 선택적 판정
m11 = ~ret_v11
m12 = ~ret_v12
f1_11_s, p_11_s, r_11_s = calc_metrics(true_clean[m11], ps_v11[m11])
f1_12_s, p_12_s, r_12_s = calc_metrics(true_clean[m12], ps_v12[m12])
emr_11_s = calc_emr(y_all[m11], ds_v11[m11])
emr_12_s = calc_emr(y_all[m12], ds_v12[m12])
ret_11_rate = float(ret_v11.sum()) / N * 100.0
ret_12_rate = float(ret_v12.sum()) / N * 100.0
cov_11 = 100.0 - ret_11_rate
cov_12 = 100.0 - ret_12_rate
fp_11_s = int(np.sum((true_clean[m11] == 0) & (ps_v11[m11] == 1)))
fp_12_s = int(np.sum((true_clean[m12] == 0) & (ps_v12[m12] == 1)))

# 세부 모델 F1
c_names = ['c1 (척추 중립)', 'c2 (고개 정면)', 'c3 (무릎 외반)', 'c4 (발바닥 고정)']
c_f1_11 = [f1_score(y_all[m11, k], ds_v11[m11, k], zero_division=0)*100.0 for k in range(4)]
c_f1_12 = [f1_score(y_all[m12, k], ds_v12[m12, k], zero_division=0)*100.0 for k in range(4)]

print("\n" + "=" * 135)
print("  [v11.0 vs v12.3 전체 해결책 통합 종합 성능 비교표]")
print("=" * 135)
print(f"  {'모델 버전 / 조건':<34s} | {'Clean F1':>9s} | {'Precision':>10s} | {'Recall':>8s} | {'4-EMR':>7s} | {'Coverage':>9s} | {'RETAKE%':>8s} | {'False Clean':>12s}")
print("  " + "-" * 135)
print(f"  {'v11.0 마스터 (전수 100% 강제판정)':<34s} | {f1_11_f:8.2f}% | {p_11_f:9.2f}% | {r_11_f:7.2f}% | {emr_11_f:6.2f}% | {'100.0%':>9s} | {'0.00%':>8s} | {fp_11_f:>10d}건")
print(f"  {'v12.3 통합 마스터 (전수 강제판정)':<34s} | {f1_12_f:8.2f}% | {p_12_f:9.2f}% | {r_12_f:7.2f}% | {emr_12_f:6.2f}% | {'100.0%':>9s} | {'0.00%':>8s} | {fp_12_f:>10d}건")
print("  " + "-" * 135)
print(f"  {'v11.0 마스터 (선택적 판정 챔피언)':<34s} | {f1_11_s:8.2f}% | {p_11_s:9.2f}% | {r_11_s:7.2f}% | {emr_11_s:6.2f}% | {cov_11:8.1f}% | {ret_11_rate:7.2f}% | {fp_11_s:>10d}건")
print(f"  {'v12.3 통합 마스터 (선택적 챔피언)':<34s} | {f1_12_s:8.2f}% | {p_12_s:9.2f}% | {r_12_s:7.2f}% | {emr_12_s:6.2f}% | {cov_12:8.1f}% | {ret_12_rate:7.2f}% | {fp_12_s:>10d}건")
print("=" * 135)

print("\n  [주요 성능 지표 변화 (선택적 판정)]")
print(f"    • 4-Defect EMR (완전 일치율): {emr_11_s:.2f}% → {emr_12_s:.2f}% (Δ{emr_12_s - emr_11_s:+.2f}%p {'▲' if emr_12_s >= emr_11_s else '▼'})")
print(f"    • Clean F1-Score:            {f1_11_s:.2f}% → {f1_12_s:.2f}% (Δ{f1_12_s - f1_11_s:+.2f}%p {'▲' if f1_12_s >= f1_11_s else '▼'})")
print(f"    • Clean Precision:           {p_11_s:.2f}% → {p_12_s:.2f}% (Δ{p_12_s - p_11_s:+.2f}%p {'▲' if p_12_s >= p_11_s else '▼'})")
print(f"    • Clean Recall:              {r_11_s:.2f}% → {r_12_s:.2f}% (Δ{r_12_s - r_11_s:+.2f}%p)")
print(f"    • Coverage (판정률):         {cov_11:.2f}% → {cov_12:.2f}% (Δ{cov_12 - cov_11:+.2f}%p)")
print(f"    • RETAKE (재촬영 권고율):    {ret_11_rate:.2f}% → {ret_12_rate:.2f}% (Δ{ret_12_rate - ret_11_rate:+.2f}%p)")
print(f"    • False Clean 건수:          {fp_11_s}건 → {fp_12_s}건 (Δ{fp_12_s - fp_11_s:+d}건)")

print("\n  [4대 세부 결함(c1~c4) 선택적 판정 F1 비교]")
for k in range(4):
    print(f"    • {c_names[k]:<18s} | v11.0: {c_f1_11[k]:5.2f}% → v12.3: {c_f1_12[k]:5.2f}% (Δ{c_f1_12[k] - c_f1_11[k]:+.2f}%p)")

print("\n  [5대 시점별 Clean F1 및 4-EMR 비교 (선택적 판정)]")
view_order = [
    ('시점 A (우측 후방)', 'view5'),
    ('시점 B (우측 전방)', 'view2'),
    ('시점 C (정면)',       'view3'),
    ('시점 D (좌측 전방)', 'view4'),
    ('시점 E (좌측 후방)', 'view1'),
]
views_detail = {}
for vn, vk in view_order:
    m_v11 = m11 & (views == vk)
    m_v12 = m12 & (views == vk)

    f1_v11 = f1_score(true_clean[m_v11], ps_v11[m_v11], zero_division=0) * 100.0
    f1_v12 = f1_score(true_clean[m_v12], ps_v12[m_v12], zero_division=0) * 100.0
    emr_v11 = calc_emr(y_all[m_v11], ds_v11[m_v11])
    emr_v12 = calc_emr(y_all[m_v12], ds_v12[m_v12])

    views_detail[vk] = {
        'name': vn,
        'f1_v11': f1_v11, 'f1_v12': f1_v12, 'delta_f1': f1_v12 - f1_v11,
        'emr_v11': emr_v11, 'emr_v12': emr_v12, 'delta_emr': emr_v12 - emr_v11
    }
    print(f"    • {vn:<16s} ({vk}) | Clean F1: {f1_v11:5.2f}% → {f1_v12:5.2f}% (Δ{f1_v12 - f1_v11:+.2f}%p) | 4-EMR: {emr_v11:5.2f}% → {emr_v12:5.2f}% (Δ{emr_v12 - emr_v11:+.2f}%p)")

# 결과 저장
out_payload = {
    'force': {
        'v11': {'f1': f1_11_f, 'precision': p_11_f, 'recall': r_11_f, 'emr': emr_11_f, 'fp': fp_11_f},
        'v12_3': {'f1': f1_12_f, 'precision': p_12_f, 'recall': r_12_f, 'emr': emr_12_f, 'fp': fp_12_f},
    },
    'selective': {
        'v11': {'f1': f1_11_s, 'precision': p_11_s, 'recall': r_11_s, 'emr': emr_11_s, 'cov': cov_11, 'retake': ret_11_rate, 'fp': fp_11_s},
        'v12_3': {'f1': f1_12_s, 'precision': p_12_s, 'recall': r_12_s, 'emr': emr_12_s, 'cov': cov_12, 'retake': ret_12_rate, 'fp': fp_12_s},
    },
    'submodels': {
        'v11': {c_names[k]: c_f1_11[k] for k in range(4)},
        'v12_3': {c_names[k]: c_f1_12[k] for k in range(4)}
    },
    'views': views_detail
}

with open(OUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(out_payload, f, indent=2, ensure_ascii=False)
print(f"\n  [SUCCESS] 공식 비교 결과 JSON 저장 완료: {OUT_JSON_PATH}")
