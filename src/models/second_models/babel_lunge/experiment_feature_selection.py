# -*- coding: utf-8 -*-
"""
피처 최적화 실험기:
v2의 30개 피처 중 죽은 피처, 다중공선성 중복 피처, 저성능 잡음 피처를 단계적으로 제거하며
5-Fold StratifiedGroupKFold 기준 OOF ROC-AUC 및 Full F1, 3-State F1(Coverage <= 90%)을 시뮬레이션
"""

import sys
import os
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

sys.stdout.reconfigure(encoding='utf-8')

def evaluate_feature_subset(X_sub, y, subjects, feature_list, tau_cov_max=0.90):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups=subjects)):
        X_tr, y_tr = X_sub[train_idx], y[train_idx]
        X_va, y_va = X_sub[val_idx], y[val_idx]

        # Tier 1 모델
        lgb = LGBMClassifier(
            n_estimators=120, max_depth=4, num_leaves=15, learning_rate=0.04,
            min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
            scale_pos_weight=2.5 * 1.35, random_state=42 + fold, verbose=-1
        )
        lgb.fit(X_tr, y_tr)

        cat = CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.04,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0
        )
        cat.fit(X_tr, y_tr)

        p_val = 0.5 * lgb.predict_proba(X_va)[:, 1] + 0.5 * cat.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = p_val

    full_auc = roc_auc_score(y, oof_probs)
    full_pred = (oof_probs >= 0.5).astype(int)
    full_f1 = f1_score(y, full_pred, zero_division=0)
    full_clean_f1 = f1_score(1 - y, 1 - full_pred, zero_division=0)
    full_acc = accuracy_score(y, full_pred)

    # 3-State 그리드 탐색: Coverage <= tau_cov_max (최대 90%)
    best_sel_f1 = 0.0
    best_cov = 0.0
    best_tn, best_td = 0.35, 0.65

    tn_list = np.linspace(0.25, 0.45, 21)
    td_list = np.linspace(0.48, 0.75, 28)

    for tn in tn_list:
        for td in td_list:
            if tn >= td:
                continue
            dec_mask = (oof_probs <= tn) | (oof_probs >= td)
            cov = np.mean(dec_mask)
            if 0.75 <= cov <= tau_cov_max:  # 커버리지 75% ~ 90%
                y_dec = y[dec_mask]
                p_dec = (oof_probs[dec_mask] >= td).astype(int)
                sel_f1 = f1_score(y_dec, p_dec, zero_division=0)
                if sel_f1 > best_sel_f1:
                    best_sel_f1 = sel_f1
                    best_cov = cov
                    best_tn, best_td = tn, td

    return {
        "n_features": len(feature_list),
        "auc": full_auc,
        "full_f1": full_f1,
        "full_clean_f1": full_clean_f1,
        "full_acc": full_acc,
        "best_sel_f1": best_sel_f1,
        "best_cov": best_cov,
        "tau_n": best_tn,
        "tau_d": best_td
    }

def main():
    cache_file = r"D:\src\models\second_models\babel_lunge\lunge_c2_features_v2_cache.npz"
    data = np.load(cache_file, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    from lunge_c2_features_v2 import FEATURE_NAMES_30D
    names = list(FEATURE_NAMES_30D)

    print("[-] Base (All 30D Features):")
    res_base = evaluate_feature_subset(X, y, subjects, names)
    print(f"    Features={res_base['n_features']}, AUC={res_base['auc']:.4f}, Full Defect F1={res_base['full_f1']:.4f}, Sel Defect F1={res_base['best_sel_f1']:.4f} (Cov={res_base['best_cov']*100:.1f}%)")

    # 제거 후보군 목록
    # 1. 완전 중복 및 0분산 피처
    dead_and_dup = ["fpa_inward_angle_body", "trunk_lateral_lean_deg_loc"]
    idx_step1 = [i for i, f in enumerate(names) if f not in dead_and_dup]
    names_step1 = [names[i] for i in idx_step1]
    res_step1 = evaluate_feature_subset(X[:, idx_step1], y, subjects, names_step1)
    print(f"\n[Step 1] Dead & Exact Duplicate 제거 (28D):")
    print(f"    Features={res_step1['n_features']}, AUC={res_step1['auc']:.4f}, Full Defect F1={res_step1['full_f1']:.4f}, Sel Defect F1={res_step1['best_sel_f1']:.4f} (Cov={res_step1['best_cov']*100:.1f}%)")

    # 2. 초고상관 다중공선성 피처 추가 제거
    collinear_to_remove = [
        "lead_femur_coronal_tilt",       # lead_hip_adduction_bottom(r=0.93)과 중복
        "lead_hip_adduction_angle_loc",  # lead_hip_adduction_bottom(r=0.93)과 중복
        "lead_ankle_toe_proj_ml",        # lead_heel_toe_skew(r=0.92)와 중복
        "lead_foot_yaw_rom",             # lead_foot_axis_stability(r=0.92)와 중복
        "pelvic_obliquity_bottom"        # pelvic_rotation_bottom(r=0.91)과 중복
    ]
    idx_step2 = [i for i, f in enumerate(names) if f not in dead_and_dup and f not in collinear_to_remove]
    names_step2 = [names[i] for i in idx_step2]
    res_step2 = evaluate_feature_subset(X[:, idx_step2], y, subjects, names_step2)
    print(f"\n[Step 2] 초고상관 다중공선성 추가 제거 (23D):")
    print(f"    Features={res_step2['n_features']}, AUC={res_step2['auc']:.4f}, Full Defect F1={res_step2['full_f1']:.4f}, Sel Defect F1={res_step2['best_sel_f1']:.4f} (Cov={res_step2['best_cov']*100:.1f}%)")

    # 3. 단일 AUC < 0.52 및 변별력 제로 잡음 피처 추가 제거
    noise_to_remove = [
        "lead_knee_fppa_bottom",         # AUC 0.5014
        "lead_knee_fppa_body_loc",       # AUC 0.5074
        "trunk_lateral_lean_bottom",     # AUC 0.5068
        "kinetic_chain_triplanar_score", # AUC 0.5109
        "knee_foot_torsion_angle",       # AUC 0.5136
        "lead_knee_medial_drift"         # AUC 0.5144
    ]
    idx_step3 = [i for i, f in enumerate(names) if f not in dead_and_dup and f not in collinear_to_remove and f not in noise_to_remove]
    names_step3 = [names[i] for i in idx_step3]
    res_step3 = evaluate_feature_subset(X[:, idx_step3], y, subjects, names_step3)
    print(f"\n[Step 3] 저변별력 잡음 피처 추가 제거 (17D):")
    print(f"    Features={res_step3['n_features']}, AUC={res_step3['auc']:.4f}, Full Defect F1={res_step3['full_f1']:.4f}, Sel Defect F1={res_step3['best_sel_f1']:.4f} (Cov={res_step3['best_cov']*100:.1f}%)")

    # 4. 남은 17개 피처 출력
    print("\n[+] Step 3 정예 17개 피처 목록:")
    for i, fn in enumerate(names_step3, 1):
        print(f"    {i:2d}. {fn}")

if __name__ == "__main__":
    main()
