# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: diagnose_d1_v3_errors.py
Purpose: d1 v3.0 모델의 오분류(FP 50건, FN 39건) 정밀 진단 및 원인 규명
         1) 피험자별 FP / FN 분포
         2) 카메라 시점별(시점 A~E) FP / FN 분포
         3) 복합 결함과의 교차 간섭 (d2, d3, d4, d5 오류가 척추 판정에 미친 영향)
         4) FP / FN 샘플들의 생체역학 피처 이상치(Outlier) 패턴 분석
"""

import os
import sys
import glob
import json
import warnings
from typing import Dict, List, Any

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d1_model_v2 import extract_d1_features_28d, FEATURE_NAMES_28_D1
from src.models.second_models.babel_deadlift.d1_model_v3 import (
    extract_morphology_ratios, MorphologyResidualizer, FEATURE_NAMES_34_D1, BabelDeadliftD1ModelV3
)
from src.models.second_models.babel_squat.build_c1_temporal_24d_tensor import extract_temporal_24d
from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v3.0 오분류(FP/FN) 전수 정밀 진단 분석")
    print("=" * 100)

    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    records = []

    for npz_path in npz_files:
        base_name = os.path.basename(npz_path).replace(".npz", "")
        parts = base_name.split("_")
        vn = parts[-1]
        folder_name = "_".join(parts[:-1])

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if not os.path.exists(label_p):
            continue

        with open(label_p, "r", encoding="utf-8", errors="ignore") as f:
            d_json = json.load(f)

        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}

        # 5대 오류 전수 기록
        # 정상=True, 오류=False
        y_d1 = 0 if conds.get("척추의 중립", True) else 1
        y_d2 = 0 if conds.get("발과 무릎의 방향 일치", True) else 1
        y_d3 = 0 if conds.get("바벨 궤적과 몸 밀착", True) else 1
        y_d4 = 0 if conds.get("무릎과 골반이 동시에 펴짐", True) else 1
        y_d5 = 0 if conds.get("발바닥 지면 고정", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_64 = data["kps_33_3d_64f"]

        records.append({
            "seq_id": base_name,
            "folder": folder_name,
            "subject": f"피험자{sub}",
            "view": vn,
            "y_true": y_d1,
            "other_defects": {
                "d2_knee_foot": y_d2,
                "d3_bar_path": y_d3,
                "d4_sync_ext": y_d4,
                "d5_foot_floor": y_d5
            },
            "kps_64": kps_64
        })

    N = len(records)
    print(f"[*] 총 분석 대상 시퀀스 수: {N}개")

    # 모델 로드
    model = BabelDeadliftD1ModelV3(weights_dir=WEIGHTS_DIR)
    if not model.is_fitted:
        print("[-] 모델 가중치를 로드할 수 없습니다.")
        return

    # 전 샘플 예측 수행
    for r in records:
        f34 = model.extract_features_34d(r["kps_64"])
        p_defect = model.predict_proba_from_seq(r["kps_64"])
        pred_bin = int(p_defect >= 0.35)

        r["f34"] = f34
        r["p_defect"] = p_defect
        r["pred_bin"] = pred_bin

        # 오분류 유형 분류
        if r["y_true"] == 0 and pred_bin == 0:
            r["cat"] = "TN"
        elif r["y_true"] == 0 and pred_bin == 1:
            r["cat"] = "FP"  # 정상 오진
        elif r["y_true"] == 1 and pred_bin == 0:
            r["cat"] = "FN"  # 결함 미탐
        else:
            r["cat"] = "TP"

    all_fp = [r for r in records if r["cat"] == "FP"]
    all_fn = [r for r in records if r["cat"] == "FN"]
    all_tn = [r for r in records if r["cat"] == "TN"]
    all_tp = [r for r in records if r["cat"] == "TP"]

    print(f"\n[*] 전체 분류 결과 집계: TN={len(all_tn)}, TP={len(all_tp)} | ★ FP(정상 오진)={len(all_fp)}건, FN(결함 미탐)={len(all_fn)}건")

    # ==========================================================================
    # 1. 피험자별 오분류 분포
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  [1. 피험자별 FP / FN 오분류 분포]")
    print(f"{'피험자':<10} | {'샘플수':<8} | {'실제정상':<10} | {'실제결함':<10} | {'FP (오진)':<12} | {'FN (미탐)':<12} | {'정확도':<10}")
    print("-" * 80)
    subjs = sorted(list(set(r["subject"] for r in records)))
    for s in subjs:
        s_recs = [r for r in records if r["subject"] == s]
        n_s = len(s_recs)
        n_norm = sum(1 for r in s_recs if r["y_true"] == 0)
        n_def = sum(1 for r in s_recs if r["y_true"] == 1)
        fp_s = sum(1 for r in s_recs if r["cat"] == "FP")
        fn_s = sum(1 for r in s_recs if r["cat"] == "FN")
        acc_s = sum(1 for r in s_recs if r["cat"] in ["TN", "TP"]) / n_s

        print(f"  {s:<8} | {n_s:>6} | {n_norm:>8} | {n_def:>8} | {fp_s:>10}건 | {fn_s:>10}건 | {acc_s*100:>8.1f}%")

    # ==========================================================================
    # 2. 카메라 시점별 오분류 분포
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  [2. 카메라 시점별 FP / FN 오분류 분포]")
    print(f"{'카메라 시점':<26} | {'샘플수':<8} | {'FP (오진)':<12} | {'FN (미탐)':<12} | {'FP율(정상대비)':<15} | {'FN율(결함대비)':<15}")
    print("-" * 95)
    for v_code, v_name in VIEW_ALIAS_MAP.items():
        v_recs = [r for r in records if r["view"] == v_code]
        n_v = len(v_recs)
        n_norm = sum(1 for r in v_recs if r["y_true"] == 0)
        n_def = sum(1 for r in v_recs if r["y_true"] == 1)
        fp_v = sum(1 for r in v_recs if r["cat"] == "FP")
        fn_v = sum(1 for r in v_recs if r["cat"] == "FN")

        fp_rate = fp_v / n_norm if n_norm > 0 else 0
        fn_rate = fn_v / n_def if n_def > 0 else 0

        print(f"  {v_name:<24} | {n_v:>6} | {fp_v:>10}건 | {fn_v:>10}건 | {fp_rate*100:>13.1f}% | {fn_rate*100:>13.1f}%")

    # ==========================================================================
    # 3. 다른 오류 라벨(d2, d3, d4, d5)과의 교차 간섭 분석 (핵심!)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  [3. 다른 오류 라벨(d2~d5)과의 교차 간섭 (Cross-Defect Contamination)]")
    print("  -> FP(정상인데 d1 오류로 오진된 50건)는 실제로 다른 오류를 가지고 있었는가?")
    print("-" * 80)

    fp_has_other = 0
    fp_defect_counts = {"d2_knee_foot": 0, "d3_bar_path": 0, "d4_sync_ext": 0, "d5_foot_floor": 0, "pure_normal": 0}
    for r in all_fp:
        od = r["other_defects"]
        tot_other = sum(od.values())
        if tot_other > 0:
            fp_has_other += 1
            for k, v in od.items():
                if v == 1:
                    fp_defect_counts[k] += 1
        else:
            fp_defect_counts["pure_normal"] += 1

    print(f"  * FP 50건 중 다른 오류(d2~d5)를 가진 복합 오류 샘플: {fp_has_other}건 ({fp_has_other/len(all_fp)*100:.1f}%)")
    print(f"  * FP 50건 중 5대 결함이 전혀 없는 순수 정상 샘플: {fp_defect_counts['pure_normal']}건 ({fp_defect_counts['pure_normal']/len(all_fp)*100:.1f}%)")
    print(f"    - d4(무릎/골반 동시 신전 실패 / 엉덩이 선행): {fp_defect_counts['d4_sync_ext']}건 동시 발생")
    print(f"    - d3(바벨 궤적과 몸 밀착 실패): {fp_defect_counts['d3_bar_path']}건 동시 발생")
    print(f"    - d2(발과 무릎 방향 불일치): {fp_defect_counts['d2_knee_foot']}건 동시 발생")
    print(f"    - d5(발바닥 지면 고정 실패): {fp_defect_counts['d5_foot_floor']}건 동시 발생")

    print("\n  -> FN(실제 d1 오류인데 정상으로 놓친 39건)의 특성 분석:")
    fn_has_other = 0
    fn_pure_d1 = 0
    for r in all_fn:
        od = r["other_defects"]
        if sum(od.values()) > 0:
            fn_has_other += 1
        else:
            fn_pure_d1 += 1
    print(f"  * FN 39건 중 순수 d1 결함만 단독 발생한 샘플: {fn_pure_d1}건 ({fn_pure_d1/len(all_fn)*100:.1f}%)")
    print(f"  * FN 39건 중 다른 결함과 함께 발생한 복합 샘플: {fn_has_other}건 ({fn_has_other/len(all_fn)*100:.1f}%)")

    # ==========================================================================
    # 4. 생체역학 피처 수준에서의 에러 원인 비교 (TN vs FP, TP vs FN)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  [4. 생체역학 주요 피처 평균값 비교 분석]")
    print(f"{'피처 명칭':<32} | {'TN (진짜정상)':<14} | {'FP (정상오진)':<14} | {'TP (진짜결함)':<14} | {'FN (결함미탐)':<14}")
    print("-" * 95)

    feat_indices = [
        (0, "요추 2차 곡률 (poly_a)"),
        (9, "능동구간 Cobb 평균 (cobb_mean)"),
        (10, "최대 요추 굴곡각 (cobb_max)"),
        (17, "셋업 체간 전경각 (torso_pitch)"),
        (22, "요추 전단력 프록시 (shear_stress)"),
        (23, "셋업 요추 곡률 (setup_poly_a)"),
        (24, "Floor Break Cobb 변화량"),
        (26, "피니시 락아웃 과신전 (lockout)"),
        (27, "척추-대퇴 신전비 (sync_ratio)"),
        (28, "체형 대비 셋업각 잔차 (res_pitch)"),
        (29, "체형 대비 셋업곡률 잔차 (res_poly)")
    ]

    F_tn = np.array([r["f34"] for r in all_tn])
    F_fp = np.array([r["f34"] for r in all_fp])
    F_tp = np.array([r["f34"] for r in all_tp])
    F_fn = np.array([r["f34"] for r in all_fn])

    for idx, fname in feat_indices:
        m_tn = float(np.mean(F_tn[:, idx]))
        m_fp = float(np.mean(F_fp[:, idx]))
        m_tp = float(np.mean(F_tp[:, idx]))
        m_fn = float(np.mean(F_fn[:, idx]))
        print(f"  {fname:<30} | {m_tn:>12.3f} | {m_fp:>12.3f} | {m_tp:>12.3f} | {m_fn:>12.3f}")

    # 진단 리포트 JSON 저장
    diagnosis_data = {
        "total_samples": N,
        "counts": {"TN": len(all_tn), "TP": len(all_tp), "FP": len(all_fp), "FN": len(all_fn)},
        "fp_other_defect_contamination": {
            "has_other_defects": fp_has_other,
            "pure_normal": fp_defect_counts["pure_normal"],
            "breakdown": fp_defect_counts
        }
    }
    with open(os.path.join(CURRENT_DIR, "d1_v3_error_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(diagnosis_data, f, indent=2, ensure_ascii=False)
    print("\n[+] 진단 결과 JSON 저장 완료: d1_v3_error_diagnosis.json")


if __name__ == "__main__":
    main()
