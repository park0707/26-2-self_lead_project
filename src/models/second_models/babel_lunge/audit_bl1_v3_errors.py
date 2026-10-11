# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: audit_bl1_v3_errors.py
Purpose: bl1_v3 5-Fold 교차검증 OOF 예측 결과의 오분류(FP, FN, Uncertain) 데이터 전수 정밀 감사
         1. 전체 오분류 건수, 정확도, 오분류율 산출
         2. 시점별(view1~view5) 및 피험자별(피험자1~9) 오분류 집중도 분석
         3. 오분류 원인 분석:
            - 경계값(Boundary Margin: 85°~95° 부근) 오류 비율
            - 앞다리/뒷다리 오인식 가능성 (Lead Leg Disambiguation)
            - 키포인트 가림/노이즈(Visibility, Rigid consistency)
            - 정면 시점(view3) 깊이 압축 왜곡
         4. 전수 오분류 샘플 리스트 및 세부 지표 JSON 저장
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

CACHE_PATH = r"D:\Data\피처_데이터\bl1_v3_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\babel_lunge\bl1_v3_evaluation_results.json"
AUDIT_OUT_JSON = r"D:\src\models\second_models\babel_lunge\bl1_v3_error_audit.json"

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def run_error_audit():
    print("=" * 80)
    print(" [vidio-health-trainer] bl1_v3 오분류 데이터 전수 정밀 감사")
    print("=" * 80)

    with open(CACHE_PATH, "rb") as f:
        cache_data = pickle.load(f)

    X = np.array(cache_data["X"], dtype=np.float32)
    y = np.array(cache_data["y"], dtype=np.int32)
    groups = np.array(cache_data.get("groups", cache_data.get("subjects")))
    views = np.array(cache_data["views"])
    clip_names = np.array(cache_data["clip_names"])
    feature_names = cache_data["feature_names"]

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    best_3s = eval_data["best_3state"]
    tau_clean = best_3s["tau_clean"]
    tau_defect = best_3s["tau_defect"]

    print(f"[*] 데이터셋: N={len(y)}, D={X.shape[1]}")
    print(f"[*] 3-State 임계치: tau_clean={tau_clean:.3f}, tau_defect={tau_defect:.3f}")

    # 5-Fold OOF 재현
    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)
    oof_ens = np.zeros(n_samples, dtype=np.float32)

    neg_cnt = np.sum(y == 0)
    pos_cnt = np.sum(y == 1)
    spw = float(neg_cnt / pos_cnt)

    lgb_params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'learning_rate': 0.04, 'num_leaves': 23, 'max_depth': 5,
        'scale_pos_weight': spw, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_lambda': 3.0, 'reg_alpha': 0.5, 'min_child_samples': 20,
        'random_state': 42, 'verbose': -1, 'n_jobs': -1
    }
    cat_params = {
        'iterations': 400, 'learning_rate': 0.04, 'depth': 5,
        'scale_pos_weight': spw, 'l2_leaf_reg': 3.0, 'rsm': 0.7,
        'random_seed': 42, 'verbose': False, 'early_stopping_rounds': 40
    }
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'learning_rate': 0.04, 'max_depth': 5, 'scale_pos_weight': spw,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_lambda': 3.0,
        'reg_alpha': 0.5, 'min_child_weight': 2, 'random_state': 42, 'n_jobs': -1
    }

    w_lgb, w_cat, w_xgb = 0.165, 0.779, 0.056

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model_lgb = lgb.train(
            lgb_params, train_data, num_boost_round=350, valid_sets=[val_data],
            callbacks=[lgb.early_stopping(35, verbose=False)]
        )
        p_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)

        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        p_cat = model_cat.predict_proba(X_val)[:, 1]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=350, evals=[(dval, 'val')], early_stopping_rounds=35, verbose_eval=False)
        p_xgb = model_xgb.predict(dval)

        oof_ens[val_idx] = w_lgb * p_lgb + w_cat * p_cat + w_xgb * p_xgb

    # 분류 상태 계산
    # 0: CLEAN, 1: DEFECT, 2: UNCERTAIN
    pred_states = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        if oof_ens[i] < tau_clean:
            pred_states[i] = 0
        elif oof_ens[i] >= tau_defect:
            pred_states[i] = 1
        else:
            pred_states[i] = 2

    # 확정 판정 마스크
    decided_mask = pred_states != 2
    uncertain_mask = pred_states == 2

    # 확정 판정 중 오분류 식별
    # FP: true=0 (정상), pred=1 (결함)
    # FN: true=1 (결함), pred=0 (정상)
    # TP: true=1, pred=1
    # TN: true=0, pred=0
    fp_mask = (y == 0) & (pred_states == 1)
    fn_mask = (y == 1) & (pred_states == 0)
    tp_mask = (y == 1) & (pred_states == 1)
    tn_mask = (y == 0) & (pred_states == 0)

    n_decided = int(np.sum(decided_mask))
    n_uncertain = int(np.sum(uncertain_mask))
    n_fp = int(np.sum(fp_mask))
    n_fn = int(np.sum(fn_mask))
    n_tp = int(np.sum(tp_mask))
    n_tn = int(np.sum(tn_mask))
    n_misclassified = n_fp + n_fn

    print(f"\n[+] 전체 데이터 현황: N={n_samples}")
    print(f"  - 확정 판정 샘플: {n_decided}건 ({n_decided/n_samples*100:.2f}%)")
    print(f"  - 판단 보류 샘플: {n_uncertain}건 ({n_uncertain/n_samples*100:.2f}%)")
    print(f"  - 확정 중 정답(TP+TN): {n_tp + n_tn}건 (정확도: {(n_tp+n_tn)/n_decided*100:.2f}%)")
    print(f"  - 확정 중 오분류(FP+FN): {n_misclassified}건 ({n_misclassified/n_decided*100:.2f}%)")
    print(f"    * False Positive (정상->결함 오판): {n_fp}건")
    print(f"    * False Negative (결함->정상 오판): {n_fn}건")

    # 피처 인덱스 맵
    idx_knee_min = feature_names.index("lead_knee_angle_min")
    idx_knee_mean = feature_names.index("lead_knee_angle_mean")
    idx_knee_90_err = feature_names.index("lead_knee_90_error_abs")
    idx_sag_90_err = feature_names.index("sagittal_knee_90_error")
    idx_femur_tilt = feature_names.index("lead_femur_horizontal_tilt")
    idx_tibia_tilt = feature_names.index("lead_tibia_vertical_tilt")
    idx_bone_cons = feature_names.index("rigid_bone_length_consistency")
    idx_vis_mean = feature_names.index("lead_knee_visibility_mean")

    # 오분류 샘플 상세 정보 수집
    error_records = []
    for i in range(n_samples):
        if fp_mask[i] or fn_mask[i]:
            error_type = "FP" if fp_mask[i] else "FN"
            error_records.append({
                "clip_name": str(clip_names[i]),
                "subject": str(groups[i]),
                "view": str(views[i]),
                "view_name": VIEW_ALIAS_MAP.get(str(views[i]), str(views[i])),
                "true_label": int(y[i]),
                "true_label_str": "결함(DEFECT)" if y[i] == 1 else "정상(CLEAN)",
                "pred_label": int(pred_states[i]),
                "pred_label_str": "결함(DEFECT)" if pred_states[i] == 1 else "정상(CLEAN)",
                "error_type": error_type,
                "pred_prob": float(oof_ens[i]),
                "knee_angle_min": float(X[i, idx_knee_min]),
                "knee_angle_mean": float(X[i, idx_knee_mean]),
                "knee_90_error": float(X[i, idx_knee_90_err]),
                "sagittal_90_error": float(X[i, idx_sag_90_err]),
                "femur_horizontal_tilt": float(X[i, idx_femur_tilt]),
                "tibia_vertical_tilt": float(X[i, idx_tibia_tilt]),
                "bone_consistency": float(X[i, idx_bone_cons]),
                "knee_visibility": float(X[i, idx_vis_mean])
            })

    df_err = pd.DataFrame(error_records)

    # 1. 시점별 오분류 분포
    print("\n[*] 시점별 오분류 발생 건수 및 오분류율:")
    view_err_summary = {}
    for v_code, alias in VIEW_ALIAS_MAP.items():
        v_total = int(np.sum(views == v_code))
        v_decided = int(np.sum((views == v_code) & decided_mask))
        v_err = int(np.sum((views == v_code) & (fp_mask | fn_mask)))
        v_fp = int(np.sum((views == v_code) & fp_mask))
        v_fn = int(np.sum((views == v_code) & fn_mask))
        v_err_rate = (v_err / v_decided * 100) if v_decided > 0 else 0.0
        view_err_summary[v_code] = {
            "alias": alias, "total": v_total, "decided": v_decided,
            "error_cnt": v_err, "fp": v_fp, "fn": v_fn, "error_rate": round(v_err_rate, 2)
        }
        print(f"  - {alias} ({v_code}): 총 {v_total}건 중 확정 {v_decided}건 -> 오분류 {v_err}건 (FP={v_fp}, FN={v_fn}, 오분류율={v_err_rate:.2f}%)")

    # 2. 피험자별 오분류 분포
    print("\n[*] 피험자별 오분류 발생 건수:")
    subj_err_summary = {}
    for subj in sorted(list(set(groups))):
        s_total = int(np.sum(groups == subj))
        s_decided = int(np.sum((groups == subj) & decided_mask))
        s_err = int(np.sum((groups == subj) & (fp_mask | fn_mask)))
        s_err_rate = (s_err / s_decided * 100) if s_decided > 0 else 0.0
        subj_err_summary[str(subj)] = {
            "total": s_total, "decided": s_decided,
            "error_cnt": s_err, "error_rate": round(s_err_rate, 2)
        }
        print(f"  - {subj}: 확정 {s_decided}건 중 오분류 {s_err}건 ({s_err_rate:.2f}%)")

    # 3. 오분류 원인 카테고리화 분석
    # A) 경계값 모호성 (Boundary Ambiguity): 무릎 각도가 85° ~ 95° 사이여서 라벨러 기준에 따라 애매한 샘플
    boundary_errors = df_err[(df_err["knee_angle_min"] >= 83.0) & (df_err["knee_angle_min"] <= 97.0)]
    n_boundary = len(boundary_errors)

    # B) 심각한 가림/지터 노이즈 (Occlusion / Jitter): bone_consistency > 0.08 또는 knee_visibility < 0.85
    occlusion_errors = df_err[(df_err["bone_consistency"] > 0.08) | (df_err["knee_visibility"] < 0.85)]
    n_occlusion = len(occlusion_errors)

    # C) 깊이 왜곡 / 극단 사잇각 착시: 시상면 각도 오차와 3D 각도 오차의 차이가 20도 이상 벌어진 경우
    depth_distortion_errors = df_err[np.abs(df_err["knee_90_error"] - df_err["sagittal_90_error"]) > 20.0]
    n_depth_distortion = len(depth_distortion_errors)

    # D) 대퇴 수평도-무릎 각도 불일치: 대퇴골은 수평(사잇각 < 15도)인데 무릎 각도는 결함으로 찍히거나 그 반대
    femur_mismatch_errors = df_err[
        ((df_err["femur_horizontal_tilt"] < 15.0) & (df_err["true_label"] == 1)) |
        ((df_err["femur_horizontal_tilt"] > 25.0) & (df_err["true_label"] == 0))
    ]
    n_femur_mismatch = len(femur_mismatch_errors)

    print("\n[*] 오분류 원인 분석 카테고리 (복수 해당 가능):")
    print(f"  1) 경계값 판정 모호성 (무릎 각도 83°~97° 근접): {n_boundary}건 ({n_boundary/n_misclassified*100:.1f}%)")
    print(f"  2) 키포인트 가림 및 뼈대 길이 불안정(노이즈): {n_occlusion}건 ({n_occlusion/n_misclassified*100:.1f}%)")
    print(f"  3) 3D vs 시상면 깊이 왜곡 괴리(|Err_3D - Err_Sag| > 20°): {n_depth_distortion}건 ({n_depth_distortion/n_misclassified*100:.1f}%)")
    print(f"  4) 대퇴골 수평도와 무릎 각도 간 도메인 불일치: {n_femur_mismatch}건 ({n_femur_mismatch/n_misclassified*100:.1f}%)")

    # 결과 종합 저장
    audit_report = {
        "summary": {
            "total_samples": n_samples,
            "decided_samples": n_decided,
            "uncertain_samples": n_uncertain,
            "total_errors": n_misclassified,
            "fp_count": n_fp,
            "fn_count": n_fn,
            "accuracy": float((n_tp + n_tn) / n_decided),
            "error_rate": float(n_misclassified / n_decided)
        },
        "view_breakdown": view_err_summary,
        "subject_breakdown": subj_err_summary,
        "cause_breakdown": {
            "boundary_ambiguity": {"count": n_boundary, "ratio": float(n_boundary / n_misclassified)},
            "occlusion_and_noise": {"count": n_occlusion, "ratio": float(n_occlusion / n_misclassified)},
            "depth_sagittal_distortion": {"count": n_depth_distortion, "ratio": float(n_depth_distortion / n_misclassified)},
            "femur_knee_mismatch": {"count": n_femur_mismatch, "ratio": float(n_femur_mismatch / n_misclassified)}
        },
        "error_records": error_records
    }

    os.makedirs(os.path.dirname(AUDIT_OUT_JSON), exist_ok=True)
    with open(AUDIT_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    print(f"\n[+] 오분류 전수 감사 결과 JSON 저장 완료: {AUDIT_OUT_JSON}")


if __name__ == "__main__":
    run_error_audit()
