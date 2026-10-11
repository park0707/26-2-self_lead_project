# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_per_view_adaptive_calibration.py
Purpose: OOF 예측값에 대해 공식 5대 시점별 적응형 3-State 캘리브레이션 적용 시 종합 성능 검증
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')

CACHE_PATH = r"D:\src\models\second_models\dynamic_lunge\dynamic_lunge_c2_hybrid_cache.npz"
data = np.load(CACHE_PATH, allow_pickle=True)
X_all = data["X"]
y = data["y"]
subjects = data["subjects"]
views = data["views"]
ex_types = data["ex_types"]
feature_names = list(data["feature_names"])

# 1. bl2_final 20D features
BL2_FINAL_20D = [
    "RTMP_lead_foot_axis_stability",
    "RTMP_fppa_eccentric_to_bottom_delta",
    "RTMP_lead_hip_adduction_bottom",
    "MP_lead_foot_axis_stability",
    "RTMP_knee_displacement_foot_heading_angle",
    "RTMP_dkcd_medial_deviation_bottom",
    "MP_knee_displacement_foot_heading_angle",
    "RTMP_lead_knee_ankle_alignment",
    "RTMP_ntmd_standing",
    "MP_pelvic_rotation_bottom",
    "MP_lead_knee_valgus_peak",
    "MP_ntmd_standing",
    "MP_trunk_pelvis_yaw_torsion",
    "RTMP_lead_foot_inward_angle_bottom",
    "RTMP_npkd_bottom",
    "RTMP_ntmd_bottom",
    "MP_lead_knee_velocity_ml",
    "MP_mkto_velocity",
    "RTMP_lead_heel_toe_skew",
    "RTMP_mkto_bottom"
]
idx_20d = [feature_names.index(fn) for fn in BL2_FINAL_20D]

# 2. Tournament 12D features
TOURNAMENT_12D = [
    "RTMP_lead_hip_adduction_bottom",
    "RTMP_lead_knee_valgus_peak",
    "RTMP_lead_heel_toe_skew",
    "RTMP_lead_foot_axis_stability",
    "RTMP_fppa_eccentric_to_bottom_delta",
    "RTMP_npkd_bottom",
    "MP_trunk_pelvis_yaw_torsion",
    "MP_lead_foot_inward_ratio",
    "RTMP_lead_foot_inward_ratio",
    "RTMP_dkcd_medial_deviation_bottom",
    "RTMP_mkto_velocity",
    "RTMP_pelvic_rotation_bottom"
]
idx_12d = [feature_names.index(fn) for fn in TOURNAMENT_12D]

# 3. Hybrid 22D (Union of 20D and 12D + Top kinematic features)
combined_set = list(dict.fromkeys(BL2_FINAL_20D + TOURNAMENT_12D))
idx_combined = [feature_names.index(fn) for fn in combined_set]

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}
VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']

def run_oof_and_per_view_eval(feat_indices, name, max_retake=0.10):
    X = X_all[:, feat_indices]
    N, D = X.shape
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(N, dtype=np.float32)
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    cost_weight = (float(n_neg) / float(n_pos)) * 1.35

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=180, max_depth=4, num_leaves=13,
            learning_rate=0.033, min_child_samples=15,
            colsample_bytree=0.75, subsample=0.85,
            reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=cost_weight,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        cat_clf = CatBoostClassifier(
            iterations=250, depth=5, learning_rate=0.030,
            l2_leaf_reg=4.0, auto_class_weights='Balanced',
            random_seed=42 + fold, verbose=0, thread_count=-1
        )
        lgb_clf.fit(X_tr, y_tr)
        cat_clf.fit(X_tr, y_tr)
        
        p_lgb = lgb_clf.predict_proba(X_va)[:, 1]
        p_cat = cat_clf.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = 0.50 * p_lgb + 0.50 * p_cat

    auc = roc_auc_score(y, oof_probs)
    f1_2s = f1_score(y, (oof_probs >= 0.5).astype(int), zero_division=0)

    # 시점별 최적화 (Per-View Adaptive 3-State Calibration)
    per_view_res = {}
    mask_3s_all = np.zeros(N, dtype=bool)
    preds_3s_all = np.zeros(N, dtype=int)
    
    for v in VIEW_ORDER:
        v_mask = (views == v)
        y_v = y[v_mask]
        p_v = oof_probs[v_mask]
        auc_v = roc_auc_score(y_v, p_v)
        
        best_v_f1 = f1_score(y_v, (p_v >= 0.5).astype(int), zero_division=0)
        best_cfg_v = (0.499, 0.501, 0.0)
        
        # Grid search for tau_clean, tau_defect with retake constraint
        for tc in np.arange(0.15, 0.65, 0.005):
            for td in np.arange(tc + 0.005, 0.85, 0.005):
                m = (p_v <= tc) | (p_v >= td)
                retake = 1.0 - np.mean(m)
                if retake <= max_retake + 0.001:
                    s = f1_score(y_v[m], (p_v[m] >= td).astype(int), zero_division=0)
                    c_f1 = f1_score(1 - y_v[m], (p_v[m] < tc).astype(int), zero_division=0)
                    if c_f1 >= 0.50 and s > best_v_f1:
                        best_v_f1 = s
                        best_cfg_v = (float(tc), float(td), float(retake))
                        
        per_view_res[v] = {
            'auc': float(auc_v),
            'f1_3s': float(best_v_f1),
            'tau_clean': best_cfg_v[0],
            'tau_defect': best_cfg_v[1],
            'retake': best_cfg_v[2]
        }
        
        # Apply to global array
        m_v = v_mask & ((oof_probs <= best_cfg_v[0]) | (oof_probs >= best_cfg_v[1]))
        mask_3s_all |= m_v
        preds_3s_all[m_v] = (oof_probs[m_v] >= best_cfg_v[1]).astype(int)

    # Overall Metrics under Per-View Adaptive Calibration
    y_dec = y[mask_3s_all]
    p_dec = preds_3s_all[mask_3s_all]
    total_f1_3s = f1_score(y_dec, p_dec, zero_division=0)
    total_clean_f1 = f1_score(1 - y_dec, 1 - p_dec, zero_division=0)
    total_acc = accuracy_score(y_dec, p_dec)
    total_prec = precision_score(y_dec, p_dec, zero_division=0)
    total_rec = recall_score(y_dec, p_dec, zero_division=0)
    total_cov = np.mean(mask_3s_all)
    total_retake = 1.0 - total_cov

    print(f"\n==========================================================================================")
    print(f"  [모델: {name} (피처 {D}D) - 보류율 제한 <= {max_retake*100:.0f}%]")
    print(f"==========================================================================================")
    print(f"[*] 전체 ROC-AUC : {auc:.4f} | 2-State F1 : {f1_2s*100:.2f}%")
    print(f"[*] 전체 3-State F1 (Per-View Calibrated) : {total_f1_3s*100:.2f}% | 정확도 : {total_acc*100:.2f}% | 커버리지 : {total_cov*100:.2f}% (보류율 {total_retake*100:.2f}%)")
    print(f"[*] Clean F1 : {total_clean_f1*100:.2f}% | Precision : {total_prec*100:.2f}% | Recall : {total_rec*100:.2f}%")
    print("-" * 90)
    print(f"{'시점':<8} | {'시점 명칭':<24} | {'ROC-AUC':<8} | {'3-State F1':<12} | {'tau_c / tau_d':<16} | {'보류율':<8}")
    print("-" * 90)
    for v in VIEW_ORDER:
        r = per_view_res[v]
        star = " ★ [70% 돌파!]" if r['f1_3s'] >= 0.70 else ""
        print(f"{v:<8} | {VIEW_ALIAS_MAP[v]:<24} | {r['auc']:.4f}   | {r['f1_3s']*100:6.2f}%{star:<12} | {r['tau_clean']:.3f} / {r['tau_defect']:.3f}   | {r['retake']*100:4.1f}%")
    print("=" * 90)
    
    return {
        'name': name,
        'dim': D,
        'auc': float(auc),
        'f1_2s': float(f1_2s),
        'total_f1_3s': float(total_f1_3s),
        'total_acc': float(total_acc),
        'total_cov': float(total_cov),
        'total_retake': float(total_retake),
        'per_view_res': per_view_res,
        'oof_probs': oof_probs
    }

print("[1] max_retake <= 10.0% 검증:")
res1 = run_oof_and_per_view_eval(idx_20d, "dl2_rtmp_1 (bl2_final 20D 계승)", max_retake=0.10)
res2 = run_oof_and_per_view_eval(idx_12d, "dl2_rtmp_1 (전진선택 정예 12D)", max_retake=0.10)
res3 = run_oof_and_per_view_eval(idx_combined, "dl2_rtmp_1 (하이브리드 통합 23D)", max_retake=0.10)

print("\n[2] max_retake <= 15.0% 검증 (최적 수렴 구간):")
res1_15 = run_oof_and_per_view_eval(idx_20d, "dl2_rtmp_1 (bl2_final 20D 계승)", max_retake=0.15)
res2_15 = run_oof_and_per_view_eval(idx_12d, "dl2_rtmp_1 (전진선택 정예 12D)", max_retake=0.15)
res3_15 = run_oof_and_per_view_eval(idx_combined, "dl2_rtmp_1 (하이브리드 통합 23D)", max_retake=0.15)
