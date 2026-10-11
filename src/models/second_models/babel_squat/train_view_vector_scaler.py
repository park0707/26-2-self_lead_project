# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_view_vector_scaler.py
Purpose: View-Conditioned Vector Scaling (시점별 16-상태 벡터 온도 보정 및 동적 위험도 판정)
         - 5대 카메라 시점(view1~view5)별 16차원 아핀 파라미터 (a_v, b_v) 학습
         - 5-Fold StratifiedGroupKFold 피험자 격리 교차검증 기반
         - 엔트로피 & 마진 기반 동적 위험도(Risk-Aware) RETAKE 게이트 구축
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

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from joint_meta_16way_moe_net import JointMeta16WayMoENetV10_1
from joint_meta_16way_net import decode_hybrid, STATE_TO_DEFECTS_MAP

META_PATH      = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH        = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH   = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
KINETIC_PATH   = r"D:\Data\피처_데이터\squat_kinetic_chain_9d_feats.npz"
C2_FEATS_PATH  = r"D:\Data\피처_데이터\squat_c2_v7_19d_feats.npz"
NORM_PRED_PATH = os.path.join(CURRENT_DIR, "squat_skeletal_normalized_predictions_2605.npz")
TEMP_PARAM_PATH= os.path.join(CURRENT_DIR, "temperature_params.json")
WEIGHTS_V10_1  = os.path.join(CURRENT_DIR, "joint_meta_16way_v10_1_weights.pth")
SAVE_SCALER_PATH = os.path.join(CURRENT_DIR, "view_vector_scaler_params.json")

print("=" * 110)
print("  [vidio-health-trainer v10.4] View-Conditioned Vector Scaling 학습 및 검증")
print("=" * 110)

# 1. 데이터 로드 및 39D 피처 구성
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

# v10.3 확정 확률 매트릭스 구성 (c2 v8 후방 불확실성 + c3 30% 뼈 정규화 Dual-Blend)
probs_v10_3 = data['probs_defect'].copy()
p_c2_feats = c2_data['X_c2_v7_19d']
face_valid = (p_c2_feats[:, 14] > 0.5)
is_rear = (views == 'view1') | (views == 'view5')

sigma2_c2 = np.where(is_rear & (~face_valid), 0.85, 0.05)
kappa_c2 = 1.0 / np.sqrt(1.0 + np.pi * sigma2_c2 / 8.0)
eps = 1e-7
p2_clip = np.clip(probs_v10_3[:, 1], eps, 1.0 - eps)
logit_c2 = np.log(p2_clip / (1.0 - p2_clip))
p2_cal_v8 = 1.0 / (1.0 + np.exp(-logit_c2 * kappa_c2))
probs_v10_3[:, 1] = p2_cal_v8
probs_v10_3[:, 2] = 0.70 * probs_v10_3[:, 2] + 0.30 * norm_d['p_c3_norm']

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

X_39d, p_cal_v10_3 = build_39d(probs_v10_3)

# 2. 16-상태 MoE 신경망에서 raw 로짓 추출
model = JointMeta16WayMoENetV10_1(in_features=39, hidden_dim=48, dropout_rate=0.20)
model.load_state_dict(torch.load(WEIGHTS_V10_1, map_location='cpu', weights_only=True))
model.eval()

with torch.no_grad():
    raw_logits_16, raw_logits_4, _ = model(torch.from_numpy(X_39d).float())
    z_raw_16 = raw_logits_16.numpy()
    p4_v10_3 = torch.sigmoid(raw_logits_4).numpy()

y_16 = (y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1).astype(np.int64)

# 3. View-Conditioned Vector Scaler 학습 모듈 정의
class ViewVectorScalerModule(nn.Module):
    """
    시점 v별 16차원 로짓 아핀 스케일링: z_cal = exp(w_v) * z + b_v
    """
    def __init__(self, init_temp: float = 1.20):
        super().__init__()
        # w_v = log(1 / init_temp)
        init_w = np.log(1.0 / init_temp)
        self.w = nn.Parameter(torch.full((16,), float(init_w), dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(16, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.w)
        return z * scale + self.b

# 4. 시점별 5-Fold OOF 및 전수 파라미터 학습
view_names = ['view5', 'view2', 'view3', 'view4', 'view1']
view_scaler_params = {}
z_cal_oof = np.zeros_like(z_raw_16)

sgkf = StratifiedGroupKFold(n_splits=5)

for v in view_names:
    v_mask = (views == v)
    sub_indices = np.where(v_mask)[0]
    z_sub = z_raw_16[v_mask]
    y_sub = y_16[v_mask]
    subs_sub = subjects[v_mask]

    print(f"\n[*] 시점 [{v}] 벡터 스케일러 최적화 (N={len(z_sub)})...")

    # (1) 5-Fold OOF 교차검증으로 OOF 캘리브레이션 로짓 수집
    for fold, (trn_rel, val_rel) in enumerate(sgkf.split(z_sub, y_sub, groups=subs_sub)):
        scaler_fold = ViewVectorScalerModule(init_temp=1.20)
        optimizer = optim.LBFGS(scaler_fold.parameters(), lr=0.1, max_iter=50, line_search_fn='strong_wolfe')

        z_trn = torch.from_numpy(z_sub[trn_rel]).float()
        y_trn = torch.from_numpy(y_sub[trn_rel]).long()
        z_val = torch.from_numpy(z_sub[val_rel]).float()

        ce_loss = nn.CrossEntropyLoss()
        init_w = float(np.log(1.0 / 1.20))

        def closure():
            optimizer.zero_grad()
            z_cal = scaler_fold(z_trn)
            loss = ce_loss(z_cal, y_trn) + 0.005 * torch.sum((scaler_fold.w - init_w)**2) + 0.002 * torch.sum(scaler_fold.b**2)
            loss.backward()
            return loss

        optimizer.step(closure)

        scaler_fold.eval()
        with torch.no_grad():
            z_val_cal = scaler_fold(z_val).numpy()
            z_cal_oof[sub_indices[val_rel]] = z_val_cal

    # (2) 프로덕션용 전수 뷰별 파라미터 학습
    prod_scaler = ViewVectorScalerModule(init_temp=1.20)
    opt_prod = optim.LBFGS(prod_scaler.parameters(), lr=0.1, max_iter=60, line_search_fn='strong_wolfe')
    z_all_v = torch.from_numpy(z_sub).float()
    y_all_v = torch.from_numpy(y_sub).long()

    def closure_prod():
        opt_prod.zero_grad()
        z_cal = prod_scaler(z_all_v)
        loss = ce_loss(z_cal, y_all_v) + 0.005 * torch.sum((prod_scaler.w - init_w)**2) + 0.002 * torch.sum(prod_scaler.b**2)
        loss.backward()
        return loss

    opt_prod.step(closure_prod)

    scale_np = torch.exp(prod_scaler.w).detach().numpy()
    bias_np  = prod_scaler.b.detach().numpy()

    view_scaler_params[v] = {
        'scale': [round(float(s), 4) for s in scale_np],
        'bias':  [round(float(b), 4) for b in bias_np],
        'effective_temperature_clean': round(float(1.0 / scale_np[0]), 3),
        'bias_clean': round(float(bias_np[0]), 3)
    }
    print(f"    -> {v} Clean Temp: {1.0/scale_np[0]:.3f} | Bias: {bias_np[0]:.3f} | Mean Temp: {1.0/np.mean(scale_np):.3f}")

# JSON 저장
with open(SAVE_SCALER_PATH, 'w', encoding='utf-8') as f:
    json.dump(view_scaler_params, f, indent=2, ensure_ascii=False)
print(f"\n[★ 완료] 시점별 벡터 스케일러 파라미터 저장 -> {SAVE_SCALER_PATH}")

# 5. 확률 변환 및 엔트로피/마진 계산
p16_cal = np.exp(z_cal_oof - np.max(z_cal_oof, axis=-1, keepdims=True))
p16_cal = p16_cal / np.sum(p16_cal, axis=-1, keepdims=True)

# 엔트로피 및 마진 계산
log_p16 = np.log(np.clip(p16_cal, 1e-8, 1.0))
entropy = -np.sum(p16_cal * log_p16, axis=-1)  # (N,)
sorted_p = np.sort(p16_cal, axis=-1)
margin_gap = sorted_p[:, -1] - sorted_p[:, -2]  # top 1 - top 2 gap
q_overall_vec = q_data['X_quality_12d'][:, 2]

# 동적 위험도 점수 R(x) = (Entropy / log(16)) * (1 - Margin) * (1 / (q_overall + 0.1))
H_max = np.log(16.0)
risk_score = (entropy / H_max) * (1.0 - margin_gap)

# 6. v10.4 판정 엔진 검증
ETA = 0.05
TAU_CLEAN = 0.36
X_c4 = c4_data['X_c4_v5_25d']
wobble = X_c4[:, 3]; jerk = X_c4[:, 15]; f0_rate = X_c4[:, 24]

# 다양한 risk_threshold에 따른 성능 탐색
print("\n" + "=" * 110)
print("  [v10.4 동적 위험도(Risk-Aware) RETAKE 임계치 탐색]")
print("=" * 110)

best_emr = 0.0
best_f1 = 0.0
best_th = 0.40

for r_th in np.linspace(0.20, 0.50, 31):
    pred_sel = np.zeros(N, dtype=int)
    defs_sel = np.zeros((N, 4), dtype=int)
    is_retake = np.zeros(N, dtype=bool)

    for i in range(N):
        if q_overall_vec[i] < 0.05:
            is_retake[i] = True
            continue
        s, defs, is_cln, p_cln = decode_hybrid(p16_cal[i], p4_v10_3[i], eta=ETA, tau_clean=TAU_CLEAN)
        # 동적 위험도 게이트
        if risk_score[i] > r_th and not is_cln:
            is_retake[i] = True
            continue
        if is_cln:
            pred_sel[i] = 1
            defs_sel[i] = [0, 0, 0, 0]
        elif views[i] in ['view1', 'view5'] and (p_cal_v10_3[i, 0] <= 0.40) and (p_cal_v10_3[i, 2] <= 0.40) and (p_cal_v10_3[i, 3] <= 0.40) and (p_cal_v10_3[i, 1] < 0.38):
            pred_sel[i] = 1
            defs_sel[i] = [0, 0, 0, 0]
        else:
            defs_sel[i] = defs

    for i in range(N):
        if is_retake[i]: continue
        if pred_sel[i] == 0 and max(p_cal_v10_3[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
            pred_sel[i] = 1
            defs_sel[i] = [0, 0, 0, 0]
        if pred_sel[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
            pred_sel[i] = 0

    m = ~is_retake
    f1 = f1_score(true_clean[m], pred_sel[m], zero_division=0) * 100.0
    prec = precision_score(true_clean[m], pred_sel[m], zero_division=0) * 100.0
    rec = recall_score(true_clean[m], pred_sel[m], zero_division=0) * 100.0
    pred_16 = defs_sel[m, 0]*8 + defs_sel[m, 1]*4 + defs_sel[m, 2]*2 + defs_sel[m, 3]*1
    true_16 = y_all[m, 0]*8 + y_all[m, 1]*4 + y_all[m, 2]*2 + y_all[m, 3]*1
    emr = float(np.mean(pred_16 == true_16)) * 100.0
    ret_rate = float(is_retake.sum()) / N * 100.0
    cov = 100.0 - ret_rate

    if prec >= 70.0:
        marker = " ★" if emr > 74.0 or f1 > 77.5 else ""
        if r_th in [0.25, 0.30, 0.35, 0.40, 0.45] or marker:
            print(f"  Risk Th={r_th:.3f} | Clean F1: {f1:5.2f}% | Prec: {prec:5.2f}% | Rec: {rec:5.2f}% | 4-EMR: {emr:5.2f}% | Cov: {cov:4.1f}% | RETAKE: {ret_rate:4.2f}%{marker}")

print("=" * 110)
