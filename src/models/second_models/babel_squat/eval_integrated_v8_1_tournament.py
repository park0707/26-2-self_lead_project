# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_integrated_v8_1_tournament.py
Purpose: Step 4 - 통합 진단 엔진 v8.1 공식 토너먼트 전수 대조 평가
         (v6.0 -> v7.0 -> v7.4 -> v7.5 -> v8.0 마스터 -> v8.1 16-Way Joint Meta Head)
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
WEIGHTS_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_weights.pth")
META_CFG_PATH = os.path.join(CURRENT_DIR, "joint_meta_16way_meta.json")

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

print("=" * 115)
print("  [Step 4] 통합 진단 엔진 공식 토너먼트 평가 (v6.0 -> v7.0 -> v7.4 -> v7.5 -> v8.0 -> v8.1 16-Way)")
print("  공식 5대 시점: 시점 A(우후방), 시점 B(우전방), 시점 C(정면), 시점 D(좌전방), 시점 E(좌후방)")
print("=" * 115)

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

# 3. v7.4 c4 지면보정
pred_v74 = np.zeros(N, dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    is_oblique = views[i] in ['view2', 'view4']
    p1, p2, p3, p4 = p_c1[i], p_c2[i], p_c3[i], p_c4[i]
    p_cln_4 = (1.0 - p1) * (1.0 - p2) * (1.0 - p3) * (1.0 - p4)
    p_cln_3 = (1.0 - p1) * (1.0 - p3) * (1.0 - p4)
    if is_rear:
        if (p1 <= 0.40) and (p3 <= 0.40) and (p4 <= 0.40) and (p2 < 0.60) and (p_cln_3 >= 0.50):
            pred_v74[i] = 1
    elif is_oblique:
        if ((p1 <= 0.35) and (p3 <= 0.30) and (p4 <= 0.25) and (p2 < 0.70)) or ((p_cln_4 >= 0.40) and all(p < 0.50 for p in [p1, p2, p3, p4])):
            pred_v74[i] = 1
    else:
        if (p_cln_4 >= 0.40) and all(p < 0.50 for p in [p1, p2, p3, p4]):
            pred_v74[i] = 1

    if pred_v74[i] == 0:
        upper_body_clean = (p1 <= 0.35) and (p3 <= 0.20)
        gaze_clean = (p2 < 0.50) if is_rear else (p2 <= 0.35)
        foot_ground_ok = (ydelta[i] <= 0.0) and (toe_lift_norm[i] <= 0.010) and (wobble[i] <= 0.008) and (p4 <= 0.70)
        if upper_body_clean and gaze_clean and foot_ground_ok:
            pred_v74[i] = 1

# 4. v7.5 시점 A~E 동적임계치
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
        cond_3p_vsafe = (p1 <= 0.35) and (p3 <= 0.30) and (p4 <= 0.25)
        if cond_3p_vsafe and (p2 < 0.70):
            pred_v75[i] = 1
        elif (p_cln_4 >= 0.40) and cond_no_defect:
            pred_v75[i] = 1
    else:
        if (p_cln_4 >= 0.38) and cond_no_defect:
            pred_v75[i] = 1

    if pred_v75[i] == 0:
        upper_body_clean = (p1 <= 0.35) and (p3 <= 0.20)
        gaze_clean = (p2 < 0.70) if is_rear else (p2 <= 0.35)
        foot_ground_ok = (ydelta[i] <= 0.0) and (toe_lift_norm[i] <= 0.010) and (wobble[i] <= 0.008) and (p4 <= 0.70)
        if upper_body_clean and gaze_clean and foot_ground_ok:
            pred_v75[i] = 1

# 5. v8.0 마스터 (시간적 일관성)
pred_v8 = np.copy(pred_v75)
for i in range(N):
    if pred_v8[i] == 0:
        max_p = max(p_c1[i], p_c2[i], p_c3[i], p_c4[i])
        n_above_40 = sum(p >= 0.40 for p in [p_c1[i], p_c2[i], p_c3[i], p_c4[i]])
        is_smooth = (wobble[i] <= 0.008) and (jerk[i] <= 180.0)
        if max_p <= 0.50 and n_above_40 <= 1 and is_smooth:
            pred_v8[i] = 1

for i in range(N):
    if pred_v8[i] == 1:
        if (jerk[i] > 220.0 and wobble[i] > 0.008) or (f0_rate[i] > 0.35):
            pred_v8[i] = 0

# 6. v8.1 (16-Way Joint Meta Head 추론)
jerk_norm = np.clip(jerk / 200.0, 0.0, 2.5)[:, None]
wobble_norm = np.clip(wobble / 0.015, 0.0, 2.5)[:, None]
ydelta_norm = np.clip(ydelta / 0.020, -1.0, 2.5)[:, None]
toe_norm = np.clip(toe_lift_norm / 0.020, 0.0, 2.5)[:, None]
f0_norm = np.clip(f0_rate / 0.50, 0.0, 2.0)[:, None]
phys_feats = np.hstack([toe_norm, wobble_norm, ydelta_norm, jerk_norm, f0_norm])

margins = np.abs(probs_defect - 0.5)
view_map = {'view5': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view1': 4}
view_indices = np.array([view_map[v] for v in views])
view_onehot = np.eye(5, dtype=np.float32)[view_indices]

X_meta_18d = np.hstack([probs_defect, margins, view_onehot, phys_feats]).astype(np.float32)

joint_model = JointMeta16WayNet(in_features=18, hidden_dim=64, dropout_rate=0.20)
if os.path.exists(WEIGHTS_PATH):
    joint_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True))
joint_model.eval()

with torch.no_grad():
    p16_all, p4_all = joint_model.predict_probabilities(torch.from_numpy(X_meta_18d).float())
    p16_np = p16_all.numpy()
    p4_np = p4_all.numpy()

eta = 0.20
tau_clean = 0.32

pred_v8_1 = np.zeros(N, dtype=int)
pred_v8_1_defects = np.zeros((N, 4), dtype=int)
for i in range(N):
    is_rear = views[i] in ['view1', 'view5']
    s, defs, is_cln, _ = decode_hybrid(p16_np[i], p4_np[i], eta=eta, tau_clean=tau_clean)
    if is_cln:
        pred_v8_1[i] = 1
        pred_v8_1_defects[i] = defs
    elif is_rear:
        cond_3p_safe = (p_c1[i] <= 0.40) and (p_c3[i] <= 0.40) and (p_c4[i] <= 0.40) and (p_c2[i] < 0.70)
        p_cln_3 = (1.0 - p_c1[i]) * (1.0 - p_c3[i]) * (1.0 - p_c4[i])
        if cond_3p_safe and p_cln_3 >= 0.55 and (p_c1[i] < 0.50 and p_c3[i] < 0.50 and p_c4[i] < 0.50):
            pred_v8_1[i] = 1
            pred_v8_1_defects[i] = [0, 0, 0, 0]
        else:
            pred_v8_1_defects[i] = defs
    else:
        pred_v8_1_defects[i] = defs

# Layer 3 Temporal Filtering 적용 (v8.0 계승)
for i in range(N):
    if pred_v8_1[i] == 0:
        max_p = max(p_c1[i], p_c2[i], p_c3[i], p_c4[i])
        n_above_40 = sum(p >= 0.40 for p in [p_c1[i], p_c2[i], p_c3[i], p_c4[i]])
        is_smooth = (wobble[i] <= 0.008) and (jerk[i] <= 180.0)
        if max_p <= 0.50 and n_above_40 <= 1 and is_smooth:
            pred_v8_1[i] = 1
            pred_v8_1_defects[i] = [0, 0, 0, 0]

    if pred_v8_1[i] == 1:
        if (jerk[i] > 220.0 and wobble[i] > 0.008) or (f0_rate[i] > 0.35):
            pred_v8_1[i] = 0

# 지표 계산 함수
def compute_metrics(name, pred_bin):
    tp = int(np.sum((pred_bin == 1) & (true_clean == 1)))
    fp = int(np.sum((pred_bin == 1) & (true_clean == 0)))
    pre = precision_score(true_clean, pred_bin, zero_division=0) * 100
    rec = recall_score(true_clean, pred_bin, zero_division=0) * 100
    f1 = f1_score(true_clean, pred_bin, zero_division=0) * 100
    fp_rate = (fp / (N - 135)) * 100
    return {'name': name, 'tp': tp, 'fp': fp, 'pre': pre, 'rec': rec, 'f1': f1, 'fp_rate': fp_rate}

m_v6 = compute_metrics("v6.0 프로덕션 (Strict AND)", pred_v6)
m_v7 = compute_metrics("v7.0 소프트 리스크 (t=0.45)", pred_v7)
m_v74 = compute_metrics("v7.4 c4 지면보정", pred_v74)
m_v75 = compute_metrics("v7.5 시점 A~E 동적임계치", pred_v75)
m_v8 = compute_metrics("v8.0 최종 마스터 (시간적일관성)", pred_v8)
m_v8_1 = compute_metrics("v8.1 16-Way Joint Head (신규)", pred_v8_1)

print(f"{'엔진 버전':<32} | {'적중(TP)':>8} | {'오인(FP)':>8} | {'Precision':>9} | {'Recall':>8} | {'Clean F1':>9} | {'Defect 오승인':>12}")
print("-" * 115)
for m in [m_v6, m_v7, m_v74, m_v75, m_v8, m_v8_1]:
    print(f"{m['name']:<32} | {m['tp']:>8d} | {m['fp']:>8d} | {m['pre']:8.2f}% | {m['rec']:7.2f}% | {m['f1']:8.2f}% | {m['fp']:>3d}건 ({m['fp_rate']:4.2f}%)")
print("=" * 115)

# 4-Defect Exact Match Ratio (EMR)
y_16 = (y_all[:, 0]*8 + y_all[:, 1]*4 + y_all[:, 2]*2 + y_all[:, 3]*1).astype(int)
pred_v8_1_states = (pred_v8_1_defects[:, 0]*8 + pred_v8_1_defects[:, 1]*4 + pred_v8_1_defects[:, 2]*2 + pred_v8_1_defects[:, 3]*1).astype(int)
emr_v8_1 = (pred_v8_1_states == y_16).mean() * 100

print(f"\n[4대 세부 결함 완전 일치율(EMR) 및 개별 부위 F1 감사]")
print(f"  - v8.1 4-Defect EMR: {emr_v8_1:.2f}%")
for k_idx, c_name in enumerate(['c1 (척추 중립)', 'c2 (시선 정면)', 'c3 (무릎 방향)', 'c4 (발바닥 고정)']):
    f1_k = f1_score(y_all[:, k_idx], pred_v8_1_defects[:, k_idx]) * 100
    print(f"  - {c_name} F1: {f1_k:.2f}%")

# 공식 5대 시점별 세부 재현율 대조표
print("\n[공식 5대 카메라 시점별(시점 A~E) 정상 재현율 누적 진화 대조표 (v6.0 -> v7.5 -> v8.0 -> v8.1)]")
print(f"{'공식 시점':<8} | {'화각 명칭':<24} | {'골든':>5} | {'v6.0':>7} | {'v7.5':>7} | {'v8.0 마스터':>12} | {'v8.1 16-Way':>12}")
print("-" * 105)

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
    r_v75 = recall_score(true_clean[mask_v], pred_v75[mask_v], zero_division=0) * 100
    r_v8 = recall_score(true_clean[mask_v], pred_v8[mask_v], zero_division=0) * 100
    r_v8_1 = recall_score(true_clean[mask_v], pred_v8_1[mask_v], zero_division=0) * 100

    print(f"{v_label:<8} | {v_desc:<24} | {n_gold:>5d} | {r_v6:6.1f}% | {r_v75:6.1f}% | {r_v8:11.1f}% | {r_v8_1:11.1f}%")
print("=" * 105)
