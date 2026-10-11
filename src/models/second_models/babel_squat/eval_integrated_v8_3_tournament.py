# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_integrated_v8_3_tournament.py
Purpose: Step 4 - 통합 진단 엔진 v8.3 공식 토너먼트 전수 대조 평가
         (v6.0 -> v7.0 -> v7.5 -> v8.0 마스터 -> v8.1 -> v8.2 -> v8.3 선택적 판정 & Precision Floor)
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
TEMP_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")
WEIGHTS_PATH_V8_1 = os.path.join(CURRENT_DIR, "joint_meta_16way_weights.pth")
WEIGHTS_PATH_V8_2 = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_2_weights.pth")
META_CFG_PATH_V8_2 = os.path.join(CURRENT_DIR, "joint_meta_16way_v8_2_meta.json")

data = np.load(META_PATH, allow_pickle=True)
c4_data = np.load(C4_PATH, allow_pickle=True)

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

print("=" * 125)
print("  [Step 4] 통합 진단 엔진 공식 토너먼트 평가 (v6.0 -> v7.0 -> v8.0 -> v8.1 -> v8.2 -> v8.3 선택적판정)")
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
preds_base = np.column_stack([st_c1_base, st_c2_base, st_c3_base, st_c4_base])
pred_v6 = np.all(preds_base == 0, axis=1).astype(int)

# 2. v7.0 Soft-Risk
clean_probs = (1.0 - p_c1) * (1.0 - p_c2) * (1.0 - p_c3) * (1.0 - p_c4)
pred_v7 = (clean_probs >= 0.45).astype(int)

# 3. v8.0 마스터 (시간적 일관성)
pred_v75 = np.zeros(N, dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    is_oblique = views[i] in ['view2', 'view4']
    p1, p2, p3, p4 = p_c1[i], p_c2[i], p_c3[i], p_c4[i]
    p_cln_4 = (1.0 - p1) * (1.0 - p2) * (1.0 - p3) * (1.0 - p4)
    p_cln_3 = (1.0 - p1) * (1.0 - p3) * (1.0 - p4)
    cond_no_defect = all(p < 0.50 for p in [p1, p2, p3, p4])
    if is_rear:
        cond_3p_safe = (p1 <= 0.40) and (p3 <= 0.40) and (p4 <= 0.40)
        if cond_3p_safe and (p2 < 0.70) and (p_cln_3 >= 0.55) and (p1 < 0.50 and p3 < 0.50 and p4 < 0.50):
            pred_v75[i] = 1
    elif is_oblique:
        if ((p1 <= 0.35) and (p3 <= 0.30) and (p4 <= 0.25) and (p2 < 0.70)) or ((p_cln_4 >= 0.40) and cond_no_defect):
            pred_v75[i] = 1
    else:
        if (p_cln_4 >= 0.38) and cond_no_defect:
            pred_v75[i] = 1
    if pred_v75[i] == 0:
        if (p1 <= 0.35) and (p3 <= 0.20) and ((p2 < 0.70) if is_rear else (p2 <= 0.35)) and (ydelta[i] <= 0.0) and (toe_lift_norm[i] <= 0.010) and (wobble[i] <= 0.008) and (p4 <= 0.70):
            pred_v75[i] = 1

pred_v8 = np.copy(pred_v75)
for i in range(N):
    if pred_v8[i] == 0 and max(p_c1[i], p_c2[i], p_c3[i], p_c4[i]) <= 0.50 and sum(p >= 0.40 for p in [p_c1[i], p_c2[i], p_c3[i], p_c4[i]]) <= 1 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        pred_v8[i] = 1
    if pred_v8[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        pred_v8[i] = 0

# 4. Level 1 보정 준비
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

# 5. v8.2 Baseline Run
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

# 6. v8.3 [모드 A: 고커버리지 선택적 판정 (Coverage 98.8%)]
# Deadband: [0.28, 0.42]
status_v83_a = []
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    s, defs, is_cln, conf = decode_hybrid(p16_v82_scaled[i], p4_v82_sig[i], eta=0.10, tau_clean=0.38)
    p_cln_node = p16_v82_scaled[i, 0]
    in_deadband = (p_cln_node >= 0.28) and (p_cln_node < 0.42)
    
    if is_cln:
        st = 'CLEAN'
    elif is_rear and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.70) and ((1.0 - p_cal[i, 0]) * (1.0 - p_cal[i, 2]) * (1.0 - p_cal[i, 3]) >= 0.55) and (p_cal[i, 0] < 0.50 and p_cal[i, 2] < 0.50 and p_cal[i, 3] < 0.50):
        st = 'CLEAN'
    else:
        st = 'DEFECT'

    if st == 'DEFECT' and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        st = 'CLEAN'
    if st == 'CLEAN' and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        st = 'DEFECT'

    if in_deadband and st != 'CLEAN':
        st = 'RETAKE'
    status_v83_a.append(st)

status_v83_a = np.array(status_v83_a)

# 7. v8.3 [모드 B: 고정밀 안전 게이트 (Precision Floor 60%+, Coverage 88.8%)]
# Margin Thresh: 0.03
sorted_p16 = np.sort(p16_v82_scaled, axis=1)
top_margin = sorted_p16[:, -1] - sorted_p16[:, -2]

status_v83_b = []
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    s, defs, is_cln, conf = decode_hybrid(p16_v82_scaled[i], p4_v82_sig[i], eta=0.10, tau_clean=0.38)
    if is_cln:
        st = 'CLEAN'
    elif is_rear and (p_cal[i, 0] <= 0.40) and (p_cal[i, 2] <= 0.40) and (p_cal[i, 3] <= 0.40) and (p_cal[i, 1] < 0.70) and ((1.0 - p_cal[i, 0]) * (1.0 - p_cal[i, 2]) * (1.0 - p_cal[i, 3]) >= 0.55) and (p_cal[i, 0] < 0.50 and p_cal[i, 2] < 0.50 and p_cal[i, 3] < 0.50):
        st = 'CLEAN'
    else:
        st = 'DEFECT'

    if st == 'DEFECT' and max(p_cal[i]) <= 0.50 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
        st = 'CLEAN'
    if st == 'CLEAN' and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
        st = 'DEFECT'

    if top_margin[i] < 0.03:
        st = 'RETAKE'
    status_v83_b.append(st)

status_v83_b = np.array(status_v83_b)

def compute_selective_metrics(name, status_arr):
    cov = np.mean(status_arr != 'RETAKE') * 100.0
    n_retake = int(np.sum(status_arr == 'RETAKE'))
    valid_m = (status_arr != 'RETAKE')
    y_v = true_clean[valid_m]
    pred_v = (status_arr[valid_m] == 'CLEAN').astype(int)
    
    tp = int(np.sum((pred_v == 1) & (y_v == 1)))
    fp = int(np.sum((pred_v == 1) & (y_v == 0)))
    pre = precision_score(y_v, pred_v, zero_division=0) * 100.0
    rec = recall_score(y_v, pred_v, zero_division=0) * 100.0
    f1 = f1_score(y_v, pred_v, zero_division=0) * 100.0
    return {'name': name, 'cov': cov, 'retake': n_retake, 'tp': tp, 'fp': fp, 'pre': pre, 'rec': rec, 'f1': f1}

def compute_full_metrics(name, pred_bin):
    tp = int(np.sum((pred_bin == 1) & (true_clean == 1)))
    fp = int(np.sum((pred_bin == 1) & (true_clean == 0)))
    pre = precision_score(true_clean, pred_bin, zero_division=0) * 100.0
    rec = recall_score(true_clean, pred_bin, zero_division=0) * 100.0
    f1 = f1_score(true_clean, pred_bin, zero_division=0) * 100.0
    return {'name': name, 'cov': 100.0, 'retake': 0, 'tp': tp, 'fp': fp, 'pre': pre, 'rec': rec, 'f1': f1}

m_v6 = compute_full_metrics("v6.0 프로덕션 (Strict AND)", pred_v6)
m_v7 = compute_full_metrics("v7.0 소프트 리스크 (t=0.45)", pred_v7)
m_v8 = compute_full_metrics("v8.0 최종 마스터 (시간적일관성)", pred_v8)
m_v82 = compute_full_metrics("v8.2 2단계 계층적 온도보정", pred_v8_2)
m_v83_a = compute_selective_metrics("v8.3 선택적 판정 (고커버리지)", status_v83_a)
m_v83_b = compute_selective_metrics("v8.3 정밀 게이트 (안전하한 60%+)", status_v83_b)

print(f"{'엔진 버전':<30} | {'커버리지':>8} | {'재촬영':>7} | {'적중(TP)':>8} | {'오인(FP)':>8} | {'Precision':>9} | {'Recall':>8} | {'Clean F1':>9}")
print("-" * 125)
for m in [m_v6, m_v7, m_v8, m_v82, m_v83_a, m_v83_b]:
    print(f"{m['name']:<30} | {m['cov']:7.1f}% | {m['retake']:>5d}건 | {m['tp']:>8d} | {m['fp']:>8d} | {m['pre']:8.2f}% | {m['rec']:7.2f}% | {m['f1']:8.2f}%")
print("=" * 125)

# 공식 5대 시점별 세부 재현율 대조표 (v8.3 고커버리지 모드)
print("\n[공식 5대 카메라 시점별(시점 A~E) 정상 재현율 누적 진화 대조표 (v6.0 -> v8.0 -> v8.2 -> v8.3 선택적판정)]")
print(f"{'공식 시점':<8} | {'화각 명칭':<24} | {'골든':>5} | {'v6.0':>7} | {'v8.0':>7} | {'v8.2 온도보정':>12} | {'v8.3 선택적판정':>14}")
print("-" * 115)

official_view_table = [
    ('시점 A', 'view5', '우측 후방 (45도)'),
    ('시점 B', 'view2', '우측 전방 (45도)'),
    ('시점 C', 'view3', '정면 (0도)'),
    ('시점 D', 'view4', '좌측 전방 (45도)'),
    ('시점 E', 'view1', '좌측 후방 (45도)'),
]

for v_label, v_code, v_desc in official_view_table:
    mask_v = (views == v_code)
    n_gold = int(np.sum(true_clean[mask_v] == 1))

    r_v6 = recall_score(true_clean[mask_v], pred_v6[mask_v], zero_division=0) * 100
    r_v8 = recall_score(true_clean[mask_v], pred_v8[mask_v], zero_division=0) * 100
    r_v82 = recall_score(true_clean[mask_v], pred_v8_2[mask_v], zero_division=0) * 100
    
    # In v8.3, on valid decisions for that view
    mask_valid_v = mask_v & (status_v83_a != 'RETAKE')
    r_v83 = recall_score(true_clean[mask_valid_v], (status_v83_a[mask_valid_v] == 'CLEAN').astype(int), zero_division=0) * 100

    print(f"{v_label:<8} | {v_desc:<24} | {n_gold:>5d} | {r_v6:6.1f}% | {r_v8:6.1f}% | {r_v82:11.1f}% | {r_v83:13.1f}%")
print("=" * 115)
