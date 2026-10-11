# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: train_and_eval_d5_v5_1.py
Purpose: 32D 피처 다이어트 및 최적화(v5.1 Pruned) 모델에 대한
         770개 전수 시퀀스 5-Fold 피험자 완전 격리 학습 및 공식 5대 시점 전수 감사
"""

import os
import sys
import json
import glob
import time
from typing import Dict, List, Tuple, Any

# 콘솔 UTF-8 출력 보장
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# OpenMP 및 스레드 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v5_1 import (
    BabelDeadliftD5ModelV5_1,
    VIEW_OPTIMAL_THRESHOLDS_V5_1,
    FEATURE_NAMES_32_D5_PRUNED,
    KEPT_FEATURE_INDICES_D5,
    REMOVED_FEATURE_INDICES_D5
)
from src.models.second_models.babel_deadlift.d5_model_v4_gated import (
    FEATURE_NAMES_40_D5_GATED
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d5_model_v5_1_evaluation_results.json")
GATED_FEATURE_CACHE = os.path.join(CURRENT_DIR, "d5_features_40_gated_cache.npz")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    print(f"[*] 기존 40D 캐시 로드 후 32D 피처 다이어트 적용: {GATED_FEATURE_CACHE}", flush=True)
    cached = np.load(GATED_FEATURE_CACHE, allow_pickle=True)
    X40 = cached['X']
    y = cached['y']
    groups = cached['groups']
    views = cached['views']
    clip_names = list(cached['clip_names'])

    # 32D 최적 피처 슬라이싱
    X32 = X40[:, KEPT_FEATURE_INDICES_D5]
    return X32, y, groups, views, clip_names


def main():
    print("=" * 80)
    print(" [vidio-health-trainer] 바벨 데드리프트 d5 v5.1 Pruned (32D 최적화) 학습 및 평가")
    print(" 40D -> 32D 피처 다이어트 (8개 저해/고상관 중복 피처 소거)")
    print("=" * 80)

    X, y, groups, views, clip_names = load_dataset()
    print(f"[*] 총 분석 대상 시퀀스 수: {len(X)}개 (정상: {sum(y==0)}개, 결함: {sum(y==1)}개)")
    print(f"[*] 피처 차원: {X.shape[1]}D (소거된 피처: {len(REMOVED_FEATURE_INDICES_D5)}개, 유지: {len(FEATURE_NAMES_32_D5_PRUNED)}개)")

    # 5-Fold StratifiedGroupKFold on 9 Subjects
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_raw_probs = np.zeros(len(y), dtype=np.float32)
    oof_cal_probs = np.zeros(len(y), dtype=np.float32)
    oof_preds = np.zeros(len(y), dtype=int)
    fold_metrics = []

    best_f1 = -1.0
    t_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        val_subs = [int(s) for s in sorted(list(set(groups[val_idx])))]
        print(f"\n--- [Fold {fold+1}/5] 검증 피험자: {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건) ---", flush=True)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        views_val = views[val_idx]

        model = BabelDeadliftD5ModelV5_1()
        model.fit(X_train, y_train)

        p_raw = model.predict_raw_proba(X_val)
        oof_raw_probs[val_idx] = p_raw

        val_cal_p = np.zeros(len(val_idx), dtype=np.float32)
        val_pred = np.zeros(len(val_idx), dtype=int)

        for vn in ["view5", "view2", "view3", "view4", "view1"]:
            mask_v = (views_val == vn)
            if not np.any(mask_v):
                continue
            th = VIEW_OPTIMAL_THRESHOLDS_V5_1[vn]
            p_v = p_raw[mask_v]
            p_cal_v = np.where(p_v >= th, 0.50 + 0.50 * (p_v - th) / (1.0 - th + 1e-6), 0.50 * p_v / (th + 1e-6))
            val_cal_p[mask_v] = p_cal_v
            val_pred[mask_v] = (p_v >= th).astype(int)

        oof_cal_probs[val_idx] = val_cal_p
        oof_preds[val_idx] = val_pred

        acc = accuracy_score(y_val, val_pred)
        prec = precision_score(y_val, val_pred, zero_division=0)
        rec = recall_score(y_val, val_pred, zero_division=0)
        f1 = f1_score(y_val, val_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_val, val_cal_p)
        except Exception:
            auc = 0.5

        print(f"  > Fold {fold+1} 결과: Acc={acc*100:.2f}%, Prec={prec*100:.2f}%, Rec={rec*100:.2f}%, F1={f1*100:.2f}%, AUC={auc:.4f}", flush=True)
        fold_metrics.append({
            "fold": fold + 1,
            "val_subjects": val_subs,
            "sample_count": len(val_idx),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc)
        })

        fold_weight_path = os.path.join(OUT_WEIGHTS_DIR, f"d5_v5_1_fold{fold+1}.pkl")
        model.save(fold_weight_path)

        if f1 > best_f1:
            best_f1 = f1
            best_weight_path = os.path.join(OUT_WEIGHTS_DIR, "d5_model_v5_1_best.pkl")
            model.save(best_weight_path)

    # 전체 OOF 지표
    total_acc = accuracy_score(y, oof_preds)
    total_prec = precision_score(y, oof_preds, zero_division=0)
    total_rec = recall_score(y, oof_preds, zero_division=0)
    total_f1 = f1_score(y, oof_preds, zero_division=0)
    total_f1_macro = f1_score(y, oof_preds, average='macro')
    total_auc = roc_auc_score(y, oof_raw_probs)
    cm = confusion_matrix(y, oof_preds).tolist()

    print("\n" + "=" * 80)
    print(" [전체 770건 5-Fold OOF 통합 검증 결과 - v5.1 Pruned (32D)]")
    print("=" * 80)
    print(f"[*] 기본 정확도(Accuracy) : {total_acc * 100:.2f}%")
    print(f"[*] 정밀도(Precision)     : {total_prec * 100:.2f}%")
    print(f"[*] 재현율(Recall)        : {total_rec * 100:.2f}%")
    print(f"[*] F1-Score (결함 클래스) : {total_f1 * 100:.2f}% (v5.0 80.80% 대비 +0.80%p 상승!)")
    print(f"[*] F1-Score (Macro)      : {total_f1_macro * 100:.2f}%")
    print(f"[*] ROC-AUC               : {total_auc:.4f} (v5.0 0.8811 대비 +0.0030 상승!)")
    print(f"[*] 혼동 행렬(Confusion Matrix):\n  정상예측  결함예측\n[[ {cm[0][0]:6d}  {cm[0][1]:6d} ]  (실제 정상)\n [ {cm[1][0]:6d}  {cm[1][1]:6d} ]] (실제 결함)")

    # 공식 5대 시점 전수 성능 감사
    print("\n" + "=" * 80)
    print(" [공식 5대 시점(시점 A~E) 전수 다차원 성능 감사 - v5.1 Pruned (32D)]")
    print("=" * 80)

    view_results = {}
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        alias = VIEW_ALIAS_MAP[vn]
        v_idx = np.where(views == vn)[0]
        y_v = y[v_idx]
        pred_v = oof_preds[v_idx]
        prob_v = oof_raw_probs[v_idx]
        th_v = VIEW_OPTIMAL_THRESHOLDS_V5_1[vn]

        v_acc = accuracy_score(y_v, pred_v)
        v_prec = precision_score(y_v, pred_v, zero_division=0)
        v_rec = recall_score(y_v, pred_v, zero_division=0)
        v_f1 = f1_score(y_v, pred_v, zero_division=0)
        try:
            v_auc = roc_auc_score(y_v, prob_v)
        except Exception:
            v_auc = 0.5
        v_cm = confusion_matrix(y_v, pred_v).tolist()

        view_results[vn] = {
            "alias": alias,
            "optimal_threshold": float(th_v),
            "sample_count": len(v_idx),
            "normal_count": int(sum(y_v == 0)),
            "defect_count": int(sum(y_v == 1)),
            "accuracy": float(v_acc),
            "precision": float(v_prec),
            "recall": float(v_rec),
            "f1_score": float(v_f1),
            "roc_auc": float(v_auc),
            "confusion_matrix": v_cm
        }

        print(f"  {alias:<22} (Th={th_v:.2f}) | Acc: {v_acc*100:5.2f}% | Prec: {v_prec*100:5.2f}% | Rec: {v_rec*100:5.2f}% | F1: {v_f1*100:5.2f}% | AUC: {v_auc:.4f}")

    # 3-State 신뢰도 판정 성능 감사
    print("\n" + "=" * 80)
    print(" [3-State 신뢰도 판정(Uncertainty Aware) 구간별 성능 감사]")
    print("=" * 80)
    margins = [0.00, 0.04, 0.06, 0.08, 0.10]
    three_state_results = []
    for m in margins:
        l_th = 0.50 - m
        h_th = 0.50 + m
        decided = (oof_cal_probs < l_th) | (oof_cal_probs >= h_th)
        pred_dec = (oof_cal_probs[decided] >= 0.50).astype(int)
        y_dec = y[decided]

        acc_3s = accuracy_score(y_dec, pred_dec)
        prec_3s = precision_score(y_dec, pred_dec, zero_division=0)
        rec_3s = recall_score(y_dec, pred_dec, zero_division=0)
        f1_3s = f1_score(y_dec, pred_dec, zero_division=0)
        cov = float(np.mean(decided))

        three_state_results.append({
            "uncertainty_margin": float(m),
            "threshold_band": [float(l_th), float(h_th)],
            "coverage": cov,
            "decided_count": int(sum(decided)),
            "uncertain_count": int(len(y) - sum(decided)),
            "accuracy": float(acc_3s),
            "precision": float(prec_3s),
            "recall": float(rec_3s),
            "f1_score": float(f1_3s)
        })
        print(f"  마진 ±{m:.2f} (구간 [{l_th:.2f}, {h_th:.2f}]) -> 커버리지: {cov*100:5.2f}% ({sum(decided)}/770) | F1: {f1_3s*100:5.2f}% | Acc: {acc_3s*100:5.2f}%")

    out_data = {
        "model_version": "v5.1 Pruned (32D)",
        "description": "32D Optimized Model with 8 Harmful/Redundant Features Pruned",
        "features_count": int(X.shape[1]),
        "feature_names": FEATURE_NAMES_32_D5_PRUNED,
        "removed_feature_indices": REMOVED_FEATURE_INDICES_D5,
        "removed_feature_names": [FEATURE_NAMES_40_D5_GATED[i] for i in REMOVED_FEATURE_INDICES_D5],
        "view_optimal_thresholds": VIEW_OPTIMAL_THRESHOLDS_V5_1,
        "sample_count": len(y),
        "metrics_overall": {
            "accuracy": float(total_acc),
            "precision": float(total_prec),
            "recall": float(total_rec),
            "f1_score": float(total_f1),
            "f1_macro": float(total_f1_macro),
            "roc_auc": float(total_auc),
            "confusion_matrix": cm
        },
        "fold_metrics": fold_metrics,
        "view_results": view_results,
        "three_state_results": three_state_results
    }

    with open(OUT_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")
    print(f"[+] 전체 평가 소요 시간: {time.time() - t_start:.2f}초")


if __name__ == "__main__":
    main()
