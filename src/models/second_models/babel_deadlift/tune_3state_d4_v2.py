# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tune_3state_d4_v2.py
Purpose: d4 v2 모델에 대해 사용자 요구사항인 "보류율 최대 10% (Coverage >= 90%)" 조건을
         적용하여 최적의 3-State 임계값(tau_normal, tau_defect)을 탐색하고 공식 5대 시점별 성능 산출
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

from src.models.second_models.babel_deadlift.d4_model_v2 import BabelDeadliftD4ModelV2
from src.models.second_models.babel_deadlift.train_and_eval_d4_v2 import (
    load_dataset_v2,
    evaluate_3state,
    VIEW_ALIAS_MAP
)

WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
RESULTS_JSON = os.path.join(CURRENT_DIR, "d4_model_v2_evaluation_results.json")

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] d4 v2 3-State 선택적 판정 임계값 튜닝 (보류율 <= 10%)")
    print("  - 제약 조건: Coverage >= 90.0% (보류율 최대 10.0%)")
    print("=" * 80, flush=True)

    X, y, subjects, views, clip_names = load_dataset_v2()

    # OOF 예측 확률 로드/복원
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        wp = os.path.join(WEIGHTS_DIR, f"d4_v2_fold{fold}.pkl")
        m = BabelDeadliftD4ModelV2.load(wp)
        oof_probs[va] = m.predict_proba(X[va])

    # 1. 2진 최적 임계치 재확인
    best_th = 0.50
    best_bin_f1 = 0.0
    for th in np.linspace(0.1, 0.9, 81):
        p = (oof_probs >= th).astype(int)
        sc = f1_score(y, p, zero_division=0)
        if sc > best_bin_f1:
            best_bin_f1 = sc
            best_th = th

    print(f"\n[*] 2진 최적 임계치: tau = {best_th:.3f} | F1 = {best_bin_f1*100:.2f}%", flush=True)

    # 2. 3-State 고해상도 그리드 탐색 (Coverage >= 90.0%)
    candidates = []
    for tau_n in np.linspace(0.15, 0.45, 61):
        for tau_d in np.linspace(0.35, 0.70, 71):
            if tau_n >= tau_d:
                continue
            m = evaluate_3state(y, oof_probs, tau_n, tau_d)
            if m["coverage"] >= 0.900:  # 보류율 <= 10.0%
                candidates.append({
                    "tau_normal": float(tau_n),
                    "tau_defect": float(tau_d),
                    "f1": m["f1"],
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "coverage": m["coverage"],
                    "decline_rate": m["decline_rate"],
                    "n_decided": m["n_decided"],
                    "n_total": m["n_total"]
                })

    # F1 우선 정렬, 동률 시 Accuracy 우선
    candidates.sort(key=lambda x: (x["f1"], x["accuracy"]), reverse=True)

    if not candidates:
        print("[!] Coverage >= 90%를 만족하는 조합이 없습니다.")
        return

    best_3state = candidates[0]

    print("\n" + "=" * 80)
    print("  [최적 3-State 선택적 판정 파라미터 (보류율 <= 10% 제약 만족)]")
    print("=" * 80)
    print(f"  - 정상 확정 임계치 (tau_normal) : {best_3state['tau_normal']:.3f} 이하")
    print(f"  - 결함 확정 임계치 (tau_defect) : {best_3state['tau_defect']:.3f} 이상")
    print(f"  - 판정 보류 구간 (RETAKE)       : {best_3state['tau_normal']:.3f} ~ {best_3state['tau_defect']:.3f}")
    print("-" * 80)
    print(f"  - Coverage (유효 판정률)        : {best_3state['coverage']*100:.2f}% ({best_3state['n_decided']}/{best_3state['n_total']}건)")
    print(f"  - Decline Rate (보류율)         : {best_3state['decline_rate']*100:.2f}% (★ 최대 10% 제약 엄격 준수)")
    print(f"  - Accuracy (유효 판정 정확도)   : {best_3state['accuracy']*100:.2f}%")
    print(f"  - Precision (유효 판정 정밀도)  : {best_3state['precision']*100:.2f}%")
    print(f"  - Recall (유효 판정 재현율)     : {best_3state['recall']*100:.2f}%")
    print(f"  - F1-Score (유효 판정 F1)       : {best_3state['f1']*100:.2f}%")
    print("=" * 80, flush=True)

    print("\n[상위 5개 후보 조합]:")
    for idx, c in enumerate(candidates[:5], 1):
        print(f"  {idx}. tau_n={c['tau_normal']:.3f}, tau_d={c['tau_defect']:.3f} | F1={c['f1']*100:.2f}%, Acc={c['accuracy']*100:.2f}%, Cov={c['coverage']*100:.2f}%, 보류율={c['decline_rate']*100:.2f}%")

    # 3. 공식 5대 시점 전수 감사 (보류율 10% 최적 파라미터 적용)
    print("\n" + "=" * 80)
    print("  [공식 5대 시점 전수 감사 (Official 5-View Audit - 보류율 <= 10% 적용)]")
    print("=" * 80)

    view_results_3state = {}
    for vn in sorted(np.unique(views).tolist()):
        alias = VIEW_ALIAS_MAP.get(vn, vn)
        mask_v = (views == vn)
        y_v = y[mask_v]
        prob_v = oof_probs[mask_v]

        # 3-State
        m3_v = evaluate_3state(y_v, prob_v, best_3state['tau_normal'], best_3state['tau_defect'])
        view_results_3state[vn] = {
            "alias": alias,
            "samples": int(len(y_v)),
            "coverage": m3_v["coverage"],
            "decline_rate": m3_v["decline_rate"],
            "accuracy": m3_v["accuracy"],
            "precision": m3_v["precision"],
            "recall": m3_v["recall"],
            "f1": m3_v["f1"]
        }
        print(f"  [{alias}] {vn}:")
        print(f"    - Coverage: {m3_v['coverage']*100:.1f}% (보류율: {m3_v['decline_rate']*100:.1f}%) | Acc: {m3_v['accuracy']*100:.2f}%, Prec: {m3_v['precision']*100:.2f}%, Rec: {m3_v['recall']*100:.2f}%, F1: {m3_v['f1']*100:.2f}%")

    print("=" * 80, flush=True)

    # 4. 결과 JSON 파일 업데이트
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            full_res = json.load(f)
    else:
        full_res = {}

    full_res["max_10pct_decline_3state"] = best_3state
    full_res["view_metrics_10pct_decline"] = view_results_3state

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_res, f, ensure_ascii=False, indent=2)

    print(f"\n[+] d4 v2 결과 JSON 업데이트 완료: {RESULTS_JSON}", flush=True)

    # 5. d4_model_v2.py의 기본값 업데이트 제안 확인
    print(f"[!] d4_model_v2 기본 tau 업데이트 권장: tau_normal={best_3state['tau_normal']:.3f}, tau_defect={best_3state['tau_defect']:.3f}")

if __name__ == "__main__":
    main()
