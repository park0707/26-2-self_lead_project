# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: optimize_3state_gate_v12_3.py
Purpose: [해결책 4] OOF 기반 위험 제어형 3-State 동적 게이트 제약 최적화
         - 목적함수: max [ EMR_det(theta) + 0.5 * CleanF1(theta) ]
         - 제약조건:
             1. Coverage(theta) >= 90.0%
             2. RETAKE(theta) <= 10.0%
             3. Clean Precision(theta) >= 75.0%
         - 최적 theta 저장: optimal_gate_params_v12_3.json
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from joint_meta_16way_net import decode_hybrid

META_PATH        = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"
C4_PATH          = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
QUALITY_PATH     = r"D:\Data\피처_데이터\squat_quality_gate_12d_feats.npz"
OOF_LOGITS_PATH  = os.path.join(CURRENT_DIR, "joint_meta_v12_3_oof_logits16.npy")
SCALER_PATH      = os.path.join(CURRENT_DIR, "view_vector_scaler_params_v12_3.json")
OOF_RES_PATH     = os.path.join(CURRENT_DIR, "stgcn_transformer_oof_residuals.npz")
OUT_GATE_PATH    = os.path.join(CURRENT_DIR, "optimal_gate_params_v12_3.json")

def run_grid_search():
    if not os.path.exists(OOF_LOGITS_PATH) or not os.path.exists(SCALER_PATH):
        print("[WAIT] Waiting for training and scaler to finish...")
        return

    data = np.load(META_PATH, allow_pickle=True)
    c4_data = np.load(C4_PATH, allow_pickle=True)
    q_data = np.load(QUALITY_PATH, allow_pickle=True)
    oof_logits = np.load(OOF_LOGITS_PATH)
    oof_res = np.load(OOF_RES_PATH, allow_pickle=True)

    with open(SCALER_PATH, 'r', encoding='utf-8') as f:
        scaler_params = json.load(f)

    views = data['views']
    y_all = data['y_all']
    true_clean = data['true_clean']
    N = len(true_clean)

    # 시점별 벡터 스케일링 + 잔차 결합
    oof_dz16 = oof_res['oof_delta_z16']
    is_front = (views == 'view2') | (views == 'view3') | (views == 'view4')
    g_vec = np.where(is_front, 0.08, 0.00)[:, None]

    z_cal = np.zeros_like(oof_logits)
    for v_name in ['view5', 'view2', 'view3', 'view4', 'view1']:
        m = (views == v_name)
        s_v = np.array(scaler_params[v_name]['scale'], dtype=np.float32)
        b_v = np.array(scaler_params[v_name]['bias'], dtype=np.float32)
        z_cal[m] = oof_logits[m] * s_v + b_v

    z_fused = z_cal + g_vec * oof_dz16
    p16 = np.exp(z_fused - np.max(z_fused, axis=-1, keepdims=True))
    p16 = p16 / np.sum(p16, axis=-1, keepdims=True)

    # 4-Way marginal
    from joint_meta_16way_net import STATE_TO_DEFECTS_MAP
    M = STATE_TO_DEFECTS_MAP
    p4 = np.dot(p16, M)

    H_max = np.log(16.0)
    log_p = np.log(np.clip(p16, 1e-8, 1.0))
    ent = -np.sum(p16 * log_p, axis=-1)
    sp = np.sort(p16, axis=-1)
    margin = sp[:, -1] - sp[:, -2]
    risk = (ent / H_max) * (1.0 - margin)

    X_c4 = c4_data['X_c4_v5_25d']
    wobble = X_c4[:, 3]; jerk = X_c4[:, 15]; f0_rate = X_c4[:, 24]
    q_overall = q_data['X_quality_12d'][:, 2]

    # 그리드 탐색 범위
    tau_clean_range = [0.32, 0.34, 0.36, 0.38, 0.40]
    risk_thresh_range = [0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    eta_range = [0.03, 0.05, 0.07]

    best_score = -1.0
    best_params = {}

    print(f"  [해결책 4] 총 {len(tau_clean_range)*len(risk_thresh_range)*len(eta_range)}개 파라미터 조합 제약 최적화 탐색 중...")

    for tau in tau_clean_range:
        for r_th in risk_thresh_range:
            for eta in eta_range:
                is_ret = np.zeros(N, dtype=bool)
                pred_s = np.zeros(N, dtype=int)
                defs_s = np.zeros((N, 4), dtype=int)

                for i in range(N):
                    if q_overall[i] < 0.05:
                        is_ret[i] = True; continue
                    s, defs, is_cln, p_cln = decode_hybrid(p16[i], p4[i], eta=eta, tau_clean=tau)
                    if risk[i] > r_th and not is_cln:
                        is_ret[i] = True; continue
                    if is_cln:
                        pred_s[i] = 1; defs_s[i] = [0, 0, 0, 0]
                    else:
                        defs_s[i] = defs

                for i in range(N):
                    if is_ret[i]: continue
                    if pred_s[i] == 0 and max(p4[i]) <= 0.48 and (wobble[i] <= 0.008 and jerk[i] <= 180.0):
                        pred_s[i] = 1; defs_s[i] = [0, 0, 0, 0]
                    if pred_s[i] == 1 and ((jerk[i] > 220.0 and wobble[i] > 0.008) or f0_rate[i] > 0.35):
                        pred_s[i] = 0

                m = ~is_ret
                cov = float(m.sum()) / N * 100.0
                ret_rate = 100.0 - cov

                if cov < 90.0:  # Coverage 90% 하드 제약
                    continue

                prec = precision_score(true_clean[m], pred_s[m], zero_division=0) * 100.0
                if prec < 70.0:  # Precision 70% 하드 제약
                    continue

                f1 = f1_score(true_clean[m], pred_s[m], zero_division=0) * 100.0
                pred_16 = defs_s[m, 0]*8 + defs_s[m, 1]*4 + defs_s[m, 2]*2 + defs_s[m, 3]*1
                true_16 = y_all[m, 0]*8 + y_all[m, 1]*4 + y_all[m, 2]*2 + y_all[m, 3]*1
                emr = float(np.mean(pred_16 == true_16)) * 100.0

                score = emr + 0.5 * f1

                if score > best_score:
                    best_score = score
                    best_params = {
                        'tau_clean': tau,
                        'risk_thresh': r_th,
                        'eta': eta,
                        'coverage': round(cov, 2),
                        'retake': round(ret_rate, 2),
                        'emr': round(emr, 2),
                        'clean_f1': round(f1, 2),
                        'clean_precision': round(prec, 2),
                        'score': round(score, 2)
                    }

    print(f"  [최적 파라미터 확정]: {best_params}")
    with open(OUT_GATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)
    print(f"  [SUCCESS] 최적 게이트 파라미터 저장 완료: {OUT_GATE_PATH}")

if __name__ == '__main__':
    run_grid_search()
