# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: analyze_d5_misclassifications.py
Purpose: d5 v3.0 모델의 5-Fold OOF 예측에서 발생한 오분류 데이터(FP, FN)의 
         시점별, 피험자별, 생체역학 피처별, 복합 결함 조건별 심층 통계 분석 수행
"""

import os
import sys
import json
import glob
import time
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d5_model_v3 import (
    BabelDeadliftD5ModelV3,
    extract_d5_v3_features_from_seq,
    FEATURE_NAMES_32_D5
)
from src.models.second_models.babel_deadlift.train_and_eval_d5_v3 import load_dataset, VIEW_ALIAS_MAP

LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
OUT_ANALYSIS_JSON = os.path.join(CURRENT_DIR, "d5_misclassification_analysis.json")
FEATURE_CACHE_FILE = os.path.join(CURRENT_DIR, "d5_features_32_cache.npz")


def run_error_analysis():
    print("=" * 90)
    print("  [vidio-health-trainer] 바벨 데드리프트 d5 v3.0 오분류 데이터 심층 특성 분석", flush=True)
    print("=" * 90)

    # 1. 데이터셋 로드 (캐시 우선)
    if os.path.exists(FEATURE_CACHE_FILE):
        print(f"[*] 피처 캐시 파일 로드: {FEATURE_CACHE_FILE}", flush=True)
        cd = np.load(FEATURE_CACHE_FILE, allow_pickle=True)
        X = cd["X"]
        y = cd["y"]
        groups = cd["groups"]
        views = cd["views"]
        clip_names = cd["clip_names"].tolist()
    else:
        print("[*] 피처 추출 시작...", flush=True)
        X, y, groups, views, clip_names = load_dataset()
        np.savez_compressed(FEATURE_CACHE_FILE, X=X, y=y, groups=groups, views=views, clip_names=np.array(clip_names))
        print(f"[+] 피처 캐시 저장 완료: {FEATURE_CACHE_FILE}", flush=True)
    N = len(y)

    # 2. 5-Fold 모델 가중치로 OOF 확률 복원
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(N, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        wp = os.path.join(WEIGHTS_DIR, f"d5_v3_fold{fold+1}.pkl")
        model = BabelDeadliftD5ModelV3.load(wp)
        oof_probs[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    oof_preds = (oof_probs >= 0.50).astype(int)

    # 마스크 분리
    tp_mask = (y == 1) & (oof_preds == 1)
    tn_mask = (y == 0) & (oof_preds == 0)
    fp_mask = (y == 0) & (oof_preds == 1)  # 정상인데 결함으로 오판
    fn_mask = (y == 1) & (oof_preds == 0)  # 결함인데 정상으로 놓침
    err_mask = fp_mask | fn_mask

    print(f"\n[*] 전체 샘플 수: {N}건")
    print(f"    - True Positive (TP, 결함 적중): {np.sum(tp_mask)}건")
    print(f"    - True Negative (TN, 정상 적중): {np.sum(tn_mask)}건")
    print(f"    - False Positive (FP, 과잉 결함 오판): {np.sum(fp_mask)}건 ({np.sum(fp_mask)/N*100:.2f}%)")
    print(f"    - False Negative (FN, 결함 미탐 누락): {np.sum(fn_mask)}건 ({np.sum(fn_mask)/N*100:.2f}%)")
    print(f"    - 총 오분류(Total Errors): {np.sum(err_mask)}건 ({np.sum(err_mask)/N*100:.2f}%)")

    # 3. 분석 1: 카메라 5대 시점별 오분류 집중도 분석
    print("\n" + "=" * 80)
    print("  [분석 1] 카메라 5대 시점별 오분류 집중도 및 에러율")
    print("=" * 80)
    view_stats = {}
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        alias = VIEW_ALIAS_MAP[vn]
        vm = (views == vn)
        v_total = np.sum(vm)
        v_fp = np.sum(fp_mask & vm)
        v_fn = np.sum(fn_mask & vm)
        v_err = v_fp + v_fn
        v_acc = accuracy_score(y[vm], oof_preds[vm])
        v_f1 = f1_score(y[vm], oof_preds[vm], zero_division=0)

        err_share = (v_err / np.sum(err_mask)) * 100
        print(f"  [{alias}] 총 {v_total}건 중 에러 {v_err}건 (FP {v_fp}, FN {v_fn}) | 에러율 {v_err/v_total*100:.1f}% | 전체 에러 점유율: {err_share:.1f}% | F1: {v_f1*100:.2f}%")
        view_stats[vn] = {
            "alias": alias,
            "total_count": int(v_total),
            "fp_count": int(v_fp),
            "fn_count": int(v_fn),
            "error_count": int(v_err),
            "error_rate": float(v_err / v_total),
            "error_share_pct": float(err_share),
            "accuracy": float(v_acc),
            "f1_score": float(v_f1)
        }

    # 4. 분석 2: 피험자별 오분류 편차 및 체형 이상치 분석
    print("\n" + "=" * 80)
    print("  [분석 2] 피험자 1~9번별 오분류 편차 분석")
    print("=" * 80)
    sub_stats = {}
    for sub in range(1, 10):
        sm = (groups == sub)
        s_total = np.sum(sm)
        s_fp = np.sum(fp_mask & sm)
        s_fn = np.sum(fn_mask & sm)
        s_err = s_fp + s_fn
        s_acc = accuracy_score(y[sm], oof_preds[sm])
        s_f1 = f1_score(y[sm], oof_preds[sm], zero_division=0)
        err_share = (s_err / np.sum(err_mask)) * 100
        print(f"  피험자 {sub}: 총 {s_total}건 중 에러 {s_err}건 (FP {s_fp}, FN {s_fn}) | 에러율 {s_err/s_total*100:.1f}% | 에러 점유율: {err_share:.1f}% | F1: {s_f1*100:.2f}%")
        sub_stats[sub] = {
            "total_count": int(s_total),
            "fp_count": int(s_fp),
            "fn_count": int(s_fn),
            "error_count": int(s_err),
            "error_rate": float(s_err / s_total),
            "error_share_pct": float(err_share),
            "accuracy": float(s_acc),
            "f1_score": float(s_f1)
        }

    # 5. 분석 3: 복합 결함(Multi-Defect Co-occurrence) 상관관계 분석
    print("\n" + "=" * 80)
    print("  [분석 3] 다른 결함(척추 d1, 무릎 d2, 발바닥 d3 등)과의 복합 발생 상관관계")
    print("=" * 80)

    # 각 클립별 라벨 메타데이터 수집
    clip_conds_map = {}
    for clip in set(clip_names):
        lp = os.path.join(LABEL_JSON_DIR, f"{clip}.json")
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                d_json = json.load(f)
            conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}
            clip_conds_map[clip] = conds
        else:
            clip_conds_map[clip] = {}

    d1_defects = np.array([0 if clip_conds_map.get(cn, {}).get("척추의 중립", True) else 1 for cn in clip_names])
    d2_defects = np.array([0 if clip_conds_map.get(cn, {}).get("발과 무릎의 방향 일치", True) else 1 for cn in clip_names])
    d3_defects = np.array([0 if clip_conds_map.get(cn, {}).get("발바닥 지면 고정", True) else 1 for cn in clip_names])

    # 순수 단일 결함 vs 복합 결함 오분류율
    num_other_defects = d1_defects + d2_defects + d3_defects

    defect_corr = {}
    for count in range(4):
        c_mask = (num_other_defects == count)
        if np.sum(c_mask) == 0:
            continue
        c_err = np.sum(err_mask & c_mask)
        c_total = np.sum(c_mask)
        print(f"  동반 결함 수 {count}개인 시퀀스: 총 {c_total}건 중 에러 {c_err}건 (에러율 {c_err/c_total*100:.2f}%)")
        defect_corr[f"other_defects_{count}"] = {
            "total_count": int(c_total),
            "error_count": int(c_err),
            "error_rate": float(c_err / c_total)
        }

    # 6. 분석 4: 생체역학적 핵심 피처의 그룹별 평균 비교 (정상 vs FP vs FN vs 결함)
    print("\n" + "=" * 80)
    print("  [분석 4] 그룹별 핵심 생체역학 피처 평균치 대조 (TN vs FP vs FN vs TP)")
    print("=" * 80)

    key_feature_indices = [
        0,  # d5_bar_shank_dist_max
        1,  # d5_bar_shank_dist_mean
        2,  # d5_bar_knee_dist_pass
        5,  # d5_bar_midfoot_init_offset
        8,  # d5_bar_drift_horizontal_range
        10, # d5_bar_path_curvature
        14, # d5_shoulder_wrist_angle_max_deg
        28, # d5_arm_foreshortening_min_ratio
        29  # d5_virtual_sagittal_angle_max_deg
    ]

    feature_comparison = {}
    for idx in key_feature_indices:
        fn_name = FEATURE_NAMES_32_D5[idx]
        tn_val = float(np.mean(X[tn_mask, idx]))
        fp_val = float(np.mean(X[fp_mask, idx]))
        fn_val = float(np.mean(X[fn_mask, idx]))
        tp_val = float(np.mean(X[tp_mask, idx]))

        print(f"  [{fn_name}]")
        print(f"    - TN(정상 적중) : {tn_val:.4f}")
        print(f"    - FP(과잉 결함) : {fp_val:.4f}  <-- 정상인데 결함만큼 높은 수치를 보임!")
        print(f"    - FN(미탐 결함) : {fn_val:.4f}  <-- 결함인데 정상처럼 낮은 수치를 보임!")
        print(f"    - TP(결함 적중) : {tp_val:.4f}")

        feature_comparison[fn_name] = {
            "TN_clean_hit": tn_val,
            "FP_overkill_error": fp_val,
            "FN_missed_defect": fn_val,
            "TP_defect_hit": tp_val
        }

    # 7. 분석 5: 확률값 분포 및 경계선(Borderline) 오분류 비율
    print("\n" + "=" * 80)
    print("  [분석 5] 경계선(Borderline, 0.40 <= Prob <= 0.60) 모호 샘플 분석")
    print("=" * 80)
    border_mask = (oof_probs >= 0.40) & (oof_probs <= 0.60)
    border_total = np.sum(border_mask)
    border_err = np.sum(border_mask & err_mask)
    err_in_border_pct = (border_err / np.sum(err_mask)) * 100

    print(f"  - 모호 구간 [0.40, 0.60] 샘플 수: {border_total}건 (전체의 {border_total/N*100:.1f}%)")
    print(f"  - 모호 구간 내 오분류 샘플 수: {border_err}건")
    print(f"  - 전체 오분류 중 모호 구간에 속한 비율: {err_in_border_pct:.1f}%")
    print(f"  - 확신형 오분류 (Prob < 0.30 or Prob > 0.70): {np.sum(err_mask & ((oof_probs < 0.30) | (oof_probs > 0.70)))}건")

    borderline_stats = {
        "border_range": "[0.40, 0.60]",
        "border_sample_count": int(border_total),
        "border_error_count": int(border_err),
        "error_in_border_pct": float(err_in_border_pct),
        "confident_error_count": int(np.sum(err_mask & ((oof_probs < 0.30) | (oof_probs > 0.70))))
    }

    # 종합 결과 JSON 저장
    full_analysis = {
        "total_samples": N,
        "confusion_matrix": {
            "TP": int(np.sum(tp_mask)),
            "TN": int(np.sum(tn_mask)),
            "FP": int(np.sum(fp_mask)),
            "FN": int(np.sum(fn_mask)),
            "total_errors": int(np.sum(err_mask)),
            "overall_error_rate_pct": float(np.sum(err_mask) / N * 100)
        },
        "view_breakdown": view_stats,
        "subject_breakdown": sub_stats,
        "defect_co_occurrence": defect_corr,
        "feature_comparison": feature_comparison,
        "borderline_stats": borderline_stats
    }

    with open(OUT_ANALYSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 오분류 심층 분석 결과 JSON 저장 완료: {OUT_ANALYSIS_JSON}")
    return full_analysis


if __name__ == "__main__":
    run_error_analysis()
