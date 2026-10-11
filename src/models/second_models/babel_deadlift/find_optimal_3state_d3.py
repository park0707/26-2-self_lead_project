# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: find_optimal_3state_d3.py
Purpose: 바벨 데드리프트 d3 모델의 3-State 선택적 판정 파레토 프론티어 정밀 탐색
"""

import os
import sys
import glob
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d3_model_v1 import (
    BabelDeadliftD3ModelV1,
    extract_d3_v1_features_from_seq
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")

npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
X_list, y_list, subjects = [], [], []

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
    y_d3 = 0 if conds.get("발바닥 지면 고정", True) else 1
    sub = 1
    for d in range(1, 10):
        if f"피험자{d}" in folder_name:
            sub = d
            break

    data = np.load(npz_path)
    kps_64 = data["kps_33_3d_64f"]
    X_list.append(extract_d3_v1_features_from_seq(kps_64))
    y_list.append(y_d3)
    subjects.append(sub)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
subjects = np.array(subjects)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(y), dtype=np.float32)

for fold, (tr, va) in enumerate(sgkf.split(X, y, groups=subjects), 1):
    w_p = os.path.join(WEIGHTS_DIR, f"d3_v1_fold{fold}.pkl")
    m = BabelDeadliftD3ModelV1()
    m.load_weights(w_p)
    oof_probs[va] = m.predict_proba_features(X[va])

print("=" * 80)
print(f"  [vidio-health-trainer] 바벨 데드리프트 d3 모델 3-State 파레토 분석")
print(f"  - 총 샘플 수: {len(y)}개 (정상: {np.sum(y==0)}, 결함: {np.sum(y==1)})")
print("=" * 80)

for min_cov in [0.90, 0.85, 0.80, 0.75]:
    best_candidate = None
    best_f1 = 0.0

    for tn in np.linspace(0.10, 0.48, 39):
        for td in np.linspace(0.52, 0.90, 39):
            if tn >= td:
                continue
            decided = (oof_probs <= tn) | (oof_probs >= td)
            cov = float(np.mean(decided))
            if cov < min_cov:
                continue

            y_dec = y[decided]
            p_dec = (oof_probs[decided] >= td).astype(int)

            if len(np.unique(y_dec)) > 1:
                score = f1_score(y_dec, p_dec, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    acc = accuracy_score(y_dec, p_dec)
                    prec = precision_score(y_dec, p_dec, zero_division=0)
                    rec = recall_score(y_dec, p_dec, zero_division=0)
                    best_candidate = {
                        "min_cov": min_cov,
                        "tau_norm": tn,
                        "tau_def": td,
                        "coverage": cov,
                        "f1": score,
                        "acc": acc,
                        "prec": prec,
                        "rec": rec,
                        "total_decided": int(np.sum(decided)),
                        "total_retake": int(np.sum(~decided))
                    }

    if best_candidate is not None:
        c = best_candidate
        print(f"[*] 목표 커버리지 >={min_cov*100:.0f}% 최적 세팅:")
        print(f"    - tau = [{c['tau_norm']:.3f}, {c['tau_def']:.3f}] | 실제 커버리지: {c['coverage']*100:.2f}% (보류: {c['total_retake']}건)")
        print(f"    - F1: {c['f1']*100:.2f}% | Precision: {c['prec']*100:.2f}% | Recall: {c['rec']*100:.2f}% | Acc: {c['acc']*100:.2f}%\n")
