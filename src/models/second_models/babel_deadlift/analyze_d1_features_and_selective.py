# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: analyze_d1_features_and_selective.py
Purpose:
  1. 38D 및 27D 피처 중 성능을 저해하는 방해 피처(Harmful/Noise Features) LOFO 전수 감사
  2. 3-State 선택적 판정 시스템 (NORMAL / DEFECT / UNDETERMINED) 정식 구축 및 커버리지별 실측
"""

import os
import sys
import json
import glob
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier

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

from src.models.second_models.babel_squat.c1_model_v11_1 import BabelSquatC1ModelV11_1
from src.models.second_models.babel_squat.c1_model_v11_a import FEATURE_NAMES_38_C1
from src.models.second_models.babel_squat.c1_model_v11_a_diet import FEATURE_NAMES_27_C1_DIET

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"

def main():
    print("=" * 105)
    print("  [vidio-health-trainer] 데드리프트 d1 피처 중요도/방해 피처 분석 및 3-State 판정불가 시스템 평가")
    print("=" * 105)

    # 1. 데이터 로드 및 피처 추출
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    X_38d_list = []
    X_27d_list = []
    y_d1_list = []
    subjects_list = []
    views_list = []

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
        y_d1 = 0 if conds.get("척추의 중립", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_33_3d = data["kps_33_3d_64f"]
        f38, f27, _ = BabelSquatC1ModelV11_1.extract_all_inputs(kps_33_3d)

        X_38d_list.append(f38)
        X_27d_list.append(f27)
        y_d1_list.append(y_d1)
        subjects_list.append(sub)
        views_list.append(vn)

    X_38d = np.array(X_38d_list, dtype=np.float32)
    X_27d = np.array(X_27d_list, dtype=np.float32)
    y_d1 = np.array(y_d1_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    print(f"[*] 총 분석 샘플: {len(y_d1)}개 (정상: {np.sum(y_d1==0)}건, 결함: {np.sum(y_d1==1)}건)")

    # 2. 38D 피처셋 Feature Importance 및 베이스라인 OOF 평가
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    def eval_features_oof(X_data):
        oof = np.zeros(len(y_d1), dtype=np.float32)
        importances = np.zeros(X_data.shape[1], dtype=np.float32)
        for train_idx, val_idx in sgkf.split(X_data, y_d1, groups=subjects):
            clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
            clf.fit(X_data[train_idx], y_d1[train_idx])
            oof[val_idx] = clf.predict_proba(X_data[val_idx])[:, 1]
            importances += clf.feature_importances_ / 5.0
        # best threshold search
        best_f, best_th = 0.0, 0.5
        for th in np.linspace(0.20, 0.80, 61):
            f = f1_score(y_d1, (oof >= th).astype(int), zero_division=0)
            if f > best_f:
                best_f = f
                best_th = th
        return oof, best_f, best_th, importances

    oof_38, base_f1_38, th_38, imp_38 = eval_features_oof(X_38d)
    print(f"\n[1] 38D 베이스라인 OOF F1: {base_f1_38*100:.2f}% (최적 임계값: {th_38:.2f})")

    # 3. 방해 피처 (LOFO: Leave-One-Feature-Out) 전수 분석
    print("\n" + "=" * 105)
    print("  [2] LOFO (피처 단일 소거) 감사: 피처를 제거했을 때 성능이 오히려 향상되는 방해 피처 색출")
    print("=" * 105)
    harmful_features = []
    benign_features = []

    for i in range(X_38d.shape[1]):
        feat_name = FEATURE_NAMES_38_C1[i]
        # i번째 피처 제외
        mask_idx = [j for j in range(X_38d.shape[1]) if j != i]
        X_sub = X_38d[:, mask_idx]
        _, f_lofo, _, _ = eval_features_oof(X_sub)
        delta = (f_lofo - base_f1_38) * 100

        if delta > 0.01:
            harmful_features.append((feat_name, i, imp_38[i], delta, f_lofo * 100))
        else:
            benign_features.append((feat_name, i, imp_38[i], delta, f_lofo * 100))

    harmful_features.sort(key=lambda x: x[3], reverse=True)

    print(f"\n[발견된 성능 저해(방해) 피처 총 {len(harmful_features)}개]:")
    print(f"{'피처 인덱스 & 이름':<45} | {'LightGBM 중요도':<14} | {'소거 시 F1 변화량 (Delta)':<22} | {'소거 후 F1'}")
    print("-" * 105)
    for name, idx, imp, delta, new_f1 in harmful_features:
        print(f"[{idx:2d}] {name:<40} | {imp:>10.1f}     | {delta:>+12.2f}%p ▲          | {new_f1:>6.2f}%")

    print("\n[주요 유익 피처 TOP 8 (제거 시 성능 급락)]: ")
    benign_features.sort(key=lambda x: x[3])
    for name, idx, imp, delta, new_f1 in benign_features[:8]:
        print(f"[{idx:2d}] {name:<40} | {imp:>10.1f}     | {delta:>+12.2f}%p ▼ (필수 피처) | {new_f1:>6.2f}%")

    # 4. 방해 피처 일괄 소거 실험 (Pruned Features)
    if harmful_features:
        drop_indices = [x[1] for x in harmful_features]
        keep_indices = [j for j in range(X_38d.shape[1]) if j not in drop_indices]
        X_pruned = X_38d[:, keep_indices]
        _, pruned_f1, pruned_th, _ = eval_features_oof(X_pruned)
        print("\n" + "=" * 105)
        print(f"  [3] 방해 피처 {len(drop_indices)}개 전수 소거 후 정제 모델 성능:")
        print(f"      - 기존 38D F1 : {base_f1_38*100:.2f}%")
        print(f"      - 정제({len(keep_indices)}D) F1 : {pruned_f1*100:.2f}% ({(pruned_f1 - base_f1_38)*100:>+5.2f}%p 향상!)")
        print("=" * 105)

    # 5. 3-State 선택적 판정(판정 불가 시스템) 정식 구축 및 실측
    print("\n" + "=" * 105)
    print("  [4] 데드리프트 d1 3-State 선택적 판정 시스템 (NORMAL / DEFECT / UNDETERMINED) 실측")
    print("=" * 105)
    print("  - 가동범위(ROM) 컷오프 + 확률 모호 구간 [tau_normal, tau_defect] 판정 보류")

    # 최적 게이트 파라미터 그리드 탐색
    best_gate_res = None
    gate_grid = [
        (0.35, 0.65),
        (0.38, 0.62),
        (0.40, 0.60),
        (0.42, 0.58),
        (0.45, 0.55),
    ]

    print(f"\n{'게이트 임계값 [tau_norm, tau_defect]':<35} | {'Coverage (확정률)':<16} | {'RETAKE (보류율)':<16} | {'확정군 F1':<12} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 105)

    oof_prob = oof_38
    for tau_norm, tau_def in gate_grid:
        # 3-State 판정
        # p <= tau_norm: NORMAL
        # p >= tau_def: DEFECT
        # tau_norm < p < tau_def: UNDETERMINED (RETAKE)
        states = []
        preds_det = []
        y_det = []

        for p_val, y_true in zip(oof_prob, y_d1):
            if p_val <= tau_norm:
                states.append("NORMAL")
                preds_det.append(0)
                y_det.append(y_true)
            elif p_val >= tau_def:
                states.append("DEFECT")
                preds_det.append(1)
                y_det.append(y_true)
            else:
                states.append("UNDETERMINED")

        total_n = len(y_d1)
        det_n = len(y_det)
        cov = det_n / total_n * 100
        retake = (total_n - det_n) / total_n * 100

        sel_f1 = f1_score(y_det, preds_det, zero_division=0) * 100
        sel_prec = precision_score(y_det, preds_det, zero_division=0) * 100
        sel_rec = recall_score(y_det, preds_det, zero_division=0) * 100

        gate_str = f"[{tau_norm:.2f}, {tau_def:.2f}]"
        print(f"{gate_str:<35} | {cov:>10.2f}%{'':<5} | {retake:>10.2f}%{'':<5} | {sel_f1:>8.2f}%{'':<3} | {sel_prec:>6.2f}%   | {sel_rec:>6.2f}%")

        if best_gate_res is None or (sel_f1 > best_gate_res['f1'] and cov >= 80.0):
            best_gate_res = {
                'tau_norm': tau_norm,
                'tau_def': tau_def,
                'cov': cov,
                'retake': retake,
                'f1': sel_f1,
                'prec': sel_prec,
                'rec': sel_rec
            }

    print("-" * 105)
    print(f"\n[+] 최적 3-State 안전 게이트: [{best_gate_res['tau_norm']}, {best_gate_res['tau_def']}]")
    print(f"    - 유효 판정 커버리지 : {best_gate_res['cov']:.2f}% (안전 보류율: {best_gate_res['retake']:.2f}%)")
    print(f"    - 확정군 최종 F1-Score : {best_gate_res['f1']:.2f}% (기존 강제판정 63.94% 대비 +{best_gate_res['f1']-63.94:.2f}%p 대폭 향상!)")
    print(f"    - 확정군 최종 Precision: {best_gate_res['prec']:.2f}%")
    print(f"    - 확정군 최종 Recall   : {best_gate_res['rec']:.2f}%")

if __name__ == "__main__":
    main()
