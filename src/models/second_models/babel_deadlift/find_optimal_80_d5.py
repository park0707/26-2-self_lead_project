# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: find_optimal_80_d5.py
Purpose: d5 모델에 대해 F1 >= 80.0% 및 Accuracy >= 80.0%를 만족하면서 
         최대 커버리지(Maximum Coverage)를 확보하는 최적 3-State 임계치 정밀 그리드 서치
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RESULTS_JSON = r"D:\src\models\second_models\babel_deadlift\d5_model_v1_evaluation_results.json"
CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] d5 모델 최소 성능 80%+ 달성을 위한 3-State 정밀 파레토 탐색")
    print("=" * 80)

    # 1. d5 v1.0 모델 가중치 및 OOF 예측 로드
    from src.models.second_models.babel_deadlift.train_and_eval_d5_model import load_dataset
    from src.models.second_models.babel_deadlift.d5_model_v1 import BabelDeadliftD5ModelV1
    from sklearn.model_selection import StratifiedGroupKFold

    X, y, groups, views, clip_names = load_dataset()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    weights_dir = r"D:\src\models\second_models\babel_deadlift\weights"
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        weight_path = os.path.join(weights_dir, f"d5_v1_fold{fold+1}.pkl")
        model = BabelDeadliftD5ModelV1.load(weight_path)
        oof_probs[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    print(f"[*] OOF 확률 수집 완료: {len(oof_probs)}건")

    # 정밀 그리드 서치 (0.005 단위)
    candidates = []
    for tau_norm in np.arange(0.20, 0.48, 0.005):
        for tau_def in np.arange(0.50, 0.75, 0.005):
            if tau_norm >= tau_def:
                continue
            is_retained = (oof_probs < tau_norm) | (oof_probs >= tau_def)
            cov = np.mean(is_retained)
            retained_y = y[is_retained]
            retained_preds = (oof_probs[is_retained] >= tau_def).astype(int)

            if len(retained_y) == 0:
                continue

            acc = accuracy_score(retained_y, retained_preds)
            f1 = f1_score(retained_y, retained_preds, zero_division=0)
            prec = precision_score(retained_y, retained_preds, zero_division=0)
            rec = recall_score(retained_y, retained_preds, zero_division=0)

            # 조건: F1 >= 80.0% AND Accuracy >= 80.0%
            if f1 >= 0.80 and acc >= 0.80:
                candidates.append({
                    "tau_norm": round(float(tau_norm), 3),
                    "tau_def": round(float(tau_def), 3),
                    "coverage": round(float(cov), 4),
                    "retained_count": int(np.sum(is_retained)),
                    "retake_count": int(np.sum(~is_retained)),
                    "accuracy": round(float(acc), 4),
                    "f1_score": round(float(f1), 4),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4)
                })

    print(f"[+] F1 >= 80.0% 및 Acc >= 80.0% 만족 후보 수: {len(candidates)}개")

    # 커버리지 기준 내림차순 정렬 (최대 커버리지 우선)
    candidates.sort(key=lambda x: (x["coverage"], x["f1_score"]), reverse=True)

    print("\n--- [최고 커버리지 상위 10개 최적 솔루션 (F1 >= 80% & Acc >= 80%)] ---")
    for i, c in enumerate(candidates[:10]):
        print(f"[{i+1}] 커버리지: {c['coverage']*100:.2f}% ({c['retained_count']}건 / 보류 {c['retake_count']}건) | "
              f"구간: [{c['tau_norm']}, {c['tau_def']}] | "
              f"F1: {c['f1_score']*100:.2f}% | Acc: {c['accuracy']*100:.2f}% | "
              f"Prec: {c['precision']*100:.2f}% | Rec: {c['recall']*100:.2f}%")

    out_file = r"D:\src\models\second_models\babel_deadlift\d5_optimal_80_solutions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(candidates[:20], f, ensure_ascii=False, indent=2)
    print(f"\n[+] 상위 20개 최적 솔루션 JSON 저장: {out_file}")

if __name__ == "__main__":
    main()
