# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: calibrate_submodels_temperature.py
Purpose: 해결책 2 [Level 1 보정] - 4대 서브모델 로짓 및 5대 시점별 최적 온도 스케일링 파라미터 산출
         (Guo et al., ICML 2017 기반 NLL 최소화 및 ECE / Brier Score 실측 최적화)
"""

import os
import sys
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
SAVE_PARAM_PATH = os.path.join(CURRENT_DIR, "temperature_params.json")

data = np.load(META_PATH, allow_pickle=True)
probs_defect = data['probs_defect'] # (N, 4)
y_all = data['y_all']               # (N, 4)
views = data['views']               # (N,)
N = len(y_all)

eps = 1e-7
p_c = np.clip(probs_defect, eps, 1.0 - eps)
logits = np.log(p_c / (1.0 - p_c))

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (probs == bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

print("=" * 95)
print("  [Step 1] 4대 서브모델 Level 1 온도 스케일링(Temperature Scaling) 최적화")
print("=" * 95)

temp_params = {
    'c1': {'global_T': 1.0},
    'c2': {'global_T': 1.0, 'view_T': {}},
    'c3': {'global_T': 1.0},
    'c4': {'global_T': 1.0}
}

calibrated_probs = np.zeros_like(p_c)

# 1. c1, c3, c4 글로벌 온도 스케일링
for idx, c_name in [(0, 'c1'), (2, 'c3'), (3, 'c4')]:
    z = logits[:, idx]
    y = y_all[:, idx]
    raw_loss = log_loss(y, p_c[:, idx])
    raw_ece = compute_ece(p_c[:, idx], y)
    raw_brier = np.mean((p_c[:, idx] - y) ** 2)

    def nll_func(T):
        p_t = 1.0 / (1.0 + np.exp(-z / T[0]))
        return log_loss(y, p_t)

    res = minimize(nll_func, [1.0], bounds=[(0.05, 5.0)], method='L-BFGS-B')
    opt_T = float(res.x[0])
    temp_params[c_name]['global_T'] = opt_T

    p_cal = 1.0 / (1.0 + np.exp(-z / opt_T))
    calibrated_probs[:, idx] = p_cal
    cal_loss = log_loss(y, p_cal)
    cal_ece = compute_ece(p_cal, y)
    cal_brier = np.mean((p_cal - y) ** 2)

    print(f"  [{c_name}] T={opt_T:.3f} | NLL: {raw_loss:.4f} -> {cal_loss:.4f} | ECE: {raw_ece*100:5.2f}% -> {cal_ece*100:5.2f}% | Brier: {raw_brier:.4f} -> {cal_brier:.4f}")

# 2. c2 시점별(View-Dependent) 비대칭 온도 스케일링
z2 = logits[:, 1]
y2 = y_all[:, 1]
raw_loss_c2 = log_loss(y2, p_c[:, 1])
raw_ece_c2 = compute_ece(p_c[:, 1], y2)
raw_brier_c2 = np.mean((p_c[:, 1] - y2) ** 2)

def nll_c2_global(T):
    p_t = 1.0 / (1.0 + np.exp(-z2 / T[0]))
    return log_loss(y2, p_t)

res_c2_glob = minimize(nll_c2_global, [1.0], bounds=[(0.05, 5.0)], method='L-BFGS-B')
temp_params['c2']['global_T'] = float(res_c2_glob.x[0])

# 시점별 최적화
view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
for v in view_list:
    mask = (views == v)
    z_v = z2[mask]
    y_v = y2[mask]

    def nll_v(T):
        p_t = 1.0 / (1.0 + np.exp(-z_v / T[0]))
        return log_loss(y_v, p_t)

    res_v = minimize(nll_v, [temp_params['c2']['global_T']], bounds=[(0.05, 5.0)], method='L-BFGS-B')
    opt_v_T = float(res_v.x[0])
    temp_params['c2']['view_T'][v] = opt_v_T
    calibrated_probs[mask, 1] = 1.0 / (1.0 + np.exp(-z_v / opt_v_T))

cal_loss_c2 = log_loss(y2, calibrated_probs[:, 1])
cal_ece_c2 = compute_ece(calibrated_probs[:, 1], y2)
cal_brier_c2 = np.mean((calibrated_probs[:, 1] - y2) ** 2)

print(f"  [c2 (5-View)] NLL: {raw_loss_c2:.4f} -> {cal_loss_c2:.4f} | ECE: {raw_ece_c2*100:5.2f}% -> {cal_ece_c2*100:5.2f}% | Brier: {raw_brier_c2:.4f} -> {cal_brier_c2:.4f}")
print("    - 시점별 최적 온도: " + ", ".join([f"{v}: {temp_params['c2']['view_T'][v]:.3f}" for v in view_list]))

# 저장
with open(SAVE_PARAM_PATH, 'w', encoding='utf-8') as f:
    json.dump(temp_params, f, indent=4, ensure_ascii=False)
print(f"\n  ★ 최적 온도 파라미터 저장 완료: {SAVE_PARAM_PATH}")
print("=" * 95)
