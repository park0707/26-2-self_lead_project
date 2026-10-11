# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: optimize_lunge_c1_v3.py
Purpose: v3 피처 최적화 및 모델 구조 최적화 탐색
         1. 고상관(|r| >= 0.85) 및 성능 저해 피처 제거 단계별 실험 (Ablation Selection)
         2. GBDT 3종(LightGBM, CatBoost, XGBoost) 구조 및 정규화 하이퍼파라미터 탐색
         3. 최적 앙상블 가중치 및 3-State 선택적 분류(보류율 <= 10.0%) 최적화
         4. 최종 v3 피처셋 및 최적 모델 구성안 확정
"""

import os
import sys
import pickle
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

CACHE_PATH = r"D:\Data\피처_데이터\bl1_v2_dataset_cache.pkl"
ANALYSIS_JSON = r"D:\src\models\second_models\babel_lunge\lunge_c1_feature_analysis.json"
V3_CONFIG_OUT = r"D:\src\models\second_models\babel_lunge\bl1_v3_optimal_config.json"


def evaluate_feature_subset_and_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_indices: List[int],
    lgb_params: Dict[str, Any],
    xgb_params: Dict[str, Any],
    cat_params: Dict[str, Any],
    max_uncertainty: float = 0.10
) -> Dict[str, Any]:
    """주어진 피처 서브셋과 모델 구조에 대해 5-Fold StratifiedGroupKFold 평가"""
    X_sub = X[:, feature_indices]
    n_samples = len(y)
    sgkf = StratifiedGroupKFold(n_splits=5)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_sub, y, groups=groups)):
        X_train, y_train = X_sub[train_idx], y[train_idx]
        X_val, y_val = X_sub[val_idx], y[val_idx]

        # 1. LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model_lgb = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=350,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)]
        )
        oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)

        # 2. CatBoost
        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]

        # 3. XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model_xgb = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=350,
            evals=[(dval, "val")],
            early_stopping_rounds=35,
            verbose_eval=False
        )
        oof_xgb[val_idx] = model_xgb.predict(dval)

    # 앙상블 최적 가중치 탐색 (Loss: Logloss / AUC)
    def loss_fn(weights):
        w = np.array(weights)
        w = np.maximum(w, 0)
        if np.sum(w) == 0:
            return 10.0
        w /= np.sum(w)
        pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        # negative ROC AUC
        return -roc_auc_score(y, pred)

    res = minimize(loss_fn, [0.4, 0.4, 0.2], method="Nelder-Mead")
    opt_weights = np.maximum(res.x, 0)
    opt_weights /= np.sum(opt_weights)

    oof_ens = opt_weights[0] * oof_lgb + opt_weights[1] * oof_cat + opt_weights[2] * oof_xgb
    binary_auc = roc_auc_score(y, oof_ens)
    binary_f1 = f1_score(y, (oof_ens >= 0.5).astype(int))

    # 3-State 최적 임계치 탐색 (Uncertainty Rate <= 10.0%)
    best_3state = None
    best_f1_3s = -1.0

    # Grid search for tau_clean (0.25 ~ 0.45), tau_defect (0.45 ~ 0.65)
    for tau_c in np.linspace(0.25, 0.45, 21):
        for tau_d in np.linspace(tau_c + 0.05, 0.65, 21):
            mask_clean = oof_ens < tau_c
            mask_defect = oof_ens >= tau_d
            mask_uncertain = (~mask_clean) & (~mask_defect)

            unc_rate = np.mean(mask_uncertain)
            if unc_rate > max_uncertainty:
                continue

            decided_mask = ~mask_uncertain
            if np.sum(decided_mask) == 0:
                continue

            y_true_dec = y[decided_mask]
            y_pred_dec = np.where(oof_ens[decided_mask] >= tau_d, 1, 0)

            f1_3s = f1_score(y_true_dec, y_pred_dec, zero_division=0)
            acc_3s = accuracy_score(y_true_dec, y_pred_dec)

            if f1_3s > best_f1_3s:
                best_f1_3s = f1_3s
                best_3state = {
                    "3state_f1": float(f1_3s),
                    "3state_acc": float(acc_3s),
                    "uncertainty_rate": float(unc_rate),
                    "coverage": float(1.0 - unc_rate),
                    "tau_clean": float(tau_c),
                    "tau_defect": float(tau_d),
                    "prec": float(precision_score(y_true_dec, y_pred_dec, zero_division=0)),
                    "recall": float(recall_score(y_true_dec, y_pred_dec, zero_division=0))
                }

    return {
        "binary_auc": float(binary_auc),
        "binary_f1": float(binary_f1),
        "opt_weights": [float(w) for w in opt_weights],
        "best_3state": best_3state,
        "n_features": len(feature_indices)
    }


def run_optimization():
    print("=" * 80)
    print(" [vidio-health-trainer] 바벨 런지 c1 피처 최적화 및 모델 구조 탐색")
    print("=" * 80)

    with open(CACHE_PATH, "rb") as f:
        cache_data = pickle.load(f)

    X = np.array(cache_data["X"], dtype=np.float32)
    y = np.array(cache_data["y"], dtype=np.int32)
    groups = np.array(cache_data.get("groups", cache_data.get("subjects")))
    views = np.array(cache_data["views"])
    all_feature_names = cache_data.get("feature_names")

    # 분석 결과 JSON 로드
    with open(ANALYSIS_JSON, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    harmful_features = [item["removed_feature"] for item in analysis.get("harmful_features", [])]
    high_corr_pairs = analysis.get("high_corr_pairs", [])

    print(f"[*] 원천 38D 피처 수: {len(all_feature_names)}")
    print(f"[*] 제거 시 F1 개선된 후보 피처: {harmful_features[:10]}")

    # 기본 하이퍼파라미터 설정
    neg_cnt = np.sum(y == 0)
    pos_cnt = np.sum(y == 1)
    spw = float(neg_cnt / pos_cnt)

    # 1. 다양한 피처 서브셋 후보 설계
    # 후보 A: 완전 중복(view_invariant_flexion_3d) + 극단적 상관(|r| >= 0.90) 제거 (32D)
    remove_cand_A = {
        "view_invariant_flexion_3d",     # r=1.0000 with lead_knee_angle_min
        "lead_knee_rebound_jitter",      # r=0.925 with rom, r=0.905 with angle_min
        "sagittal_knee_angle_bottom",    # r=0.929 with sagittal_90_err, r=0.88 with angle_mean
        "trunk_forward_lean_lead",       # r=0.937 with lead_femur_horizontal_tilt
        "tibia_torso_parallelism_error", # r=0.889 with lead_tibia_vertical_tilt
        "lead_knee_flexion_rom",         # r=0.945 with lead_knee_angle_min
    }

    # 후보 B: 후보 A + c1 90도와 무관하여 노이즈를 유발하는 피처 제거 (26D)
    # (KOT 돌출, 외반/내반, 카메라 추정치, 골반 비틀림, 템포 비율)
    remove_cand_B = remove_cand_A.union({
        "kot_normalized_displacement",   # c1 무릎 90도와 무관, Ablation +2.90%p
        "camera_view_azimuth_est",       # 체형 왜곡 노이즈, Ablation +2.56%p
        "lead_knee_valgus_varus",        # 관상면 외반내반, Ablation +2.49%p
        "descent_ascent_tempo_ratio",    # 템포 노이즈, Ablation +1.34%p
        "hip_knee_flexion_coupling",     # 체형 편차 노이즈, Ablation +3.58%p
        "pelvic_axial_rotation",         # 골반 회전 노이즈, Ablation +1.35%p
    })

    # 후보 C: 후보 B + 미세 기여도 하위 및 추가 노이즈 피처 제거 (22D)
    remove_cand_C = remove_cand_B.union({
        "lead_knee_under_90_ratio",      # 런지 체류시간 짧아 노이즈, Ablation +2.03%p
        "bottom_phase_dwell_ratio",      # Ablation +2.13%p
        "pelvic_drop_angle",             # Ablation +1.81%p
        "step_width_normalized",         # Ablation +0.68%p
    })

    # 후보 D: 후보 C + 체간 정규화 길이 비율 등 정적 체형 피처 제거 (18D 핵심 도메인셋)
    remove_cand_D = remove_cand_C.union({
        "lead_femur_length_ratio",
        "lead_tibia_length_ratio",
        "stride_length_normalized",
        "rear_knee_ground_clearance",
    })

    feature_subsets = {
        "v2_38D_All": list(range(len(all_feature_names))),
        "Sub_A_32D": [i for i, name in enumerate(all_feature_names) if name not in remove_cand_A],
        "Sub_B_26D": [i for i, name in enumerate(all_feature_names) if name not in remove_cand_B],
        "Sub_C_22D": [i for i, name in enumerate(all_feature_names) if name not in remove_cand_C],
        "Sub_D_18D": [i for i, name in enumerate(all_feature_names) if name not in remove_cand_D],
    }

    # 기본 모델 구조 파라미터
    base_lgb = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "scale_pos_weight": spw,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }
    base_xgb = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 5,
        "scale_pos_weight": spw,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }
    base_cat = {
        "iterations": 350,
        "learning_rate": 0.05,
        "depth": 5,
        "scale_pos_weight": spw,
        "random_seed": 42,
        "verbose": False,
        "early_stopping_rounds": 35
    }

    print("\n[*] 1단계: 피처 서브셋별 비교 실험 시작...")
    subset_results = {}
    for subset_name, feat_indices in feature_subsets.items():
        print(f"\n---> [실험] 피처셋: {subset_name} ({len(feat_indices)}D)")
        res = evaluate_feature_subset_and_model(
            X, y, groups, feat_indices, base_lgb, base_xgb, base_cat
        )
        b3 = res["best_3state"]
        print(f"     3-State F1: {b3['3state_f1']*100:.2f}% | ACC: {b3['3state_acc']*100:.2f}% | AUC: {res['binary_auc']*100:.2f}% | 보류율: {b3['uncertainty_rate']*100:.2f}%")
        subset_results[subset_name] = res

    # 1단계 최적 피처셋 선정
    best_subset_name = max(subset_results, key=lambda k: subset_results[k]["best_3state"]["3state_f1"])
    print(f"\n[+] 1단계 최우수 피처셋 선정: {best_subset_name} (3-State F1 = {subset_results[best_subset_name]['best_3state']['3state_f1']*100:.2f}%)")
    best_feat_indices = feature_subsets[best_subset_name]
    best_feat_names = [all_feature_names[i] for i in best_feat_indices]

    # 2단계: 모델 구조 및 하이퍼파라미터 최적화
    print("\n[*] 2단계: 모델 아키텍처 및 정규화(Regularization) 튜닝 실험...")
    arch_configs = [
        {
            "name": "Arch_1_Conservative (depth=4, L2=5.0)",
            "lgb": {**base_lgb, "max_depth": 4, "num_leaves": 15, "reg_lambda": 5.0, "reg_alpha": 1.0, "min_child_samples": 25},
            "xgb": {**base_xgb, "max_depth": 4, "reg_lambda": 5.0, "reg_alpha": 1.0, "min_child_weight": 3},
            "cat": {**base_cat, "depth": 4, "l2_leaf_reg": 5.0}
        },
        {
            "name": "Arch_2_Balanced (depth=5, L2=3.0, colsample=0.7)",
            "lgb": {**base_lgb, "max_depth": 5, "num_leaves": 23, "reg_lambda": 3.0, "reg_alpha": 0.5, "colsample_bytree": 0.7, "min_child_samples": 20},
            "xgb": {**base_xgb, "max_depth": 5, "reg_lambda": 3.0, "reg_alpha": 0.5, "colsample_bytree": 0.7, "min_child_weight": 2},
            "cat": {**base_cat, "depth": 5, "l2_leaf_reg": 3.0, "rsm": 0.7}
        },
        {
            "name": "Arch_3_Robust (depth=6, L1=2.0, L2=10.0, colsample=0.75)",
            "lgb": {**base_lgb, "max_depth": 6, "num_leaves": 31, "reg_lambda": 10.0, "reg_alpha": 2.0, "colsample_bytree": 0.75, "min_child_samples": 30},
            "xgb": {**base_xgb, "max_depth": 6, "reg_lambda": 10.0, "reg_alpha": 2.0, "colsample_bytree": 0.75, "min_child_weight": 4},
            "cat": {**base_cat, "depth": 6, "l2_leaf_reg": 10.0, "rsm": 0.75}
        }
    ]

    arch_results = {}
    for arch in arch_configs:
        print(f"\n---> [실험] 모델 구조: {arch['name']}")
        res = evaluate_feature_subset_and_model(
            X, y, groups, best_feat_indices, arch["lgb"], arch["xgb"], arch["cat"]
        )
        b3 = res["best_3state"]
        print(f"     3-State F1: {b3['3state_f1']*100:.2f}% | ACC: {b3['3state_acc']*100:.2f}% | AUC: {res['binary_auc']*100:.2f}% | 보류율: {b3['uncertainty_rate']*100:.2f}%")
        arch_results[arch["name"]] = {
            "config": arch,
            "result": res
        }

    best_arch_name = max(arch_results, key=lambda k: arch_results[k]["result"]["best_3state"]["3state_f1"])
    print(f"\n[+] 2단계 최우수 모델 구조 선정: {best_arch_name}")
    final_best_res = arch_results[best_arch_name]["result"]

    # 최종 v3 최적 구성 파일 저장
    v3_optimal_config = {
        "best_feature_subset": {
            "subset_name": best_subset_name,
            "feature_dim": len(best_feat_names),
            "feature_names": best_feat_names,
            "removed_features": [f for f in all_feature_names if f not in best_feat_names]
        },
        "best_architecture": {
            "arch_name": best_arch_name,
            "lgb_params": arch_results[best_arch_name]["config"]["lgb"],
            "xgb_params": arch_results[best_arch_name]["config"]["xgb"],
            "cat_params": arch_results[best_arch_name]["config"]["cat"]
        },
        "performance": {
            "binary_auc": final_best_res["binary_auc"],
            "binary_f1": final_best_res["binary_f1"],
            "ensemble_weights": final_best_res["opt_weights"],
            "best_3state": final_best_res["best_3state"]
        },
        "all_subset_experiments": {
            k: {
                "n_features": v["n_features"],
                "binary_auc": v["binary_auc"],
                "binary_f1": v["binary_f1"],
                "3state_f1": v["best_3state"]["3state_f1"],
                "3state_acc": v["best_3state"]["3state_acc"],
                "uncertainty_rate": v["best_3state"]["uncertainty_rate"]
            }
            for k, v in subset_results.items()
        }
    }

    os.makedirs(os.path.dirname(V3_CONFIG_OUT), exist_ok=True)
    with open(V3_CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(v3_optimal_config, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] v3 최적 구성 JSON 저장 완료: {V3_CONFIG_OUT}")


if __name__ == "__main__":
    run_optimization()
