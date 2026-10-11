# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: train_and_eval_d5_v6.py
Purpose: v6 37D 영점화/체형보정 모델 5-Fold StratifiedGroupKFold 학습 및 전수 다차원 성능 감사
Output: D:\src\models\second_models\babel_deadlift\d5_model_v6_evaluation_results.json
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Any

# 콘솔 UTF-8 출력 보장
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# OpenMP 및 스레드 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v6 import (
    BabelDeadliftD5ModelV6,
    VIEW_OPTIMAL_THRESHOLDS_V6
)
from src.models.second_models.babel_deadlift.d5_model_v6_features import (
    FEATURE_NAMES_37_D5_V6
)

V6_CACHE_PATH = os.path.join(CURRENT_DIR, "d5_features_37_v6_cache.npz")
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d5_model_v6_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    cached = np.load(V6_CACHE_PATH, allow_pickle=True)
    X = cached['X']
    y = cached['y']
    groups = cached['groups']
    views = cached['views']
    clip_names = list(cached['clip_names'])
    return X, y, groups, views, clip_names


def find_optimal_thresholds(y_true: np.ndarray, raw_probs: np.ndarray, views: np.ndarray) -> Dict[str, float]:
    """시점별 F1 극대화 임계값 그리드 서치"""
    opt_th = {}
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        mask_v = (views == vn)
        y_v = y_true[mask_v]
        p_v = raw_probs[mask_v]

        best_f1 = -1.0
        best_t = 0.50
        for t in np.linspace(0.20, 0.70, 51):
            pred_t = (p_v >= t).astype(int)
            f1_t = f1_score(y_v, pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_t = round(float(t), 2)
        opt_th[vn] = best_t
    return opt_th


def main():
    print("=" * 80, flush=True)
    print(" [vidio-health-trainer] 바벨 데드리프트 d5 v6.0 (37D Zero-Calibrated) 학습 및 평가", flush=True)
    print(" 5-Fold StratifiedGroupKFold on 9 Subjects 전수 다차원 감사", flush=True)
    print("=" * 80, flush=True)

    X, y, groups, views, clip_names = load_dataset()
    print(f"[*] 데이터셋: {len(X)}개 (정상: {sum(y==0)}개, 결함: {sum(y==1)}개)", flush=True)
    print(f"[*] 피처 차원: {X.shape[1]}D ({len(FEATURE_NAMES_37_D5_V6)} features)", flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_raw_probs = np.zeros(len(y), dtype=np.float32)
    fold_metrics = []

    best_fold_f1 = -1.0
    t_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        val_subs = [int(s) for s in sorted(list(set(groups[val_idx])))]
        print(f"\n--- [Fold {fold+1}/5] 검증 피험자: {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건) ---", flush=True)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = BabelDeadliftD5ModelV6()
        model.fit(X_train, y_train)

        p_raw = model.predict_raw_proba(X_val)
        oof_raw_probs[val_idx] = p_raw

        fold_acc = accuracy_score(y_val, (p_raw >= 0.50).astype(int))
        fold_f1 = f1_score(y_val, (p_raw >= 0.50).astype(int), zero_division=0)
        fold_auc = roc_auc_score(y_val, p_raw)

        print(f"  > Fold {fold+1} 기본 (Th=0.50): Acc={fold_acc*100:.2f}%, F1={fold_f1*100:.2f}%, AUC={fold_auc:.4f}", flush=True)
        fold_metrics.append({
            "fold": fold + 1,
            "val_subjects": val_subs,
            "sample_count": len(val_idx),
            "accuracy": float(fold_acc),
            "f1_score": float(fold_f1),
            "roc_auc": float(fold_auc)
        })

        fold_weight_path = os.path.join(OUT_WEIGHTS_DIR, f"d5_v6_fold{fold+1}.pkl")
        model.save(fold_weight_path)

        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_weight_path = os.path.join(OUT_WEIGHTS_DIR, "d5_model_v6_best.pkl")
            model.save(best_weight_path)

    # 1. 원시 OOF 성능 지표 (단일 0.50 기준)
    raw_acc = accuracy_score(y, (oof_raw_probs >= 0.50).astype(int))
    raw_f1 = f1_score(y, (oof_raw_probs >= 0.50).astype(int), zero_division=0)
    raw_auc = roc_auc_score(y, oof_raw_probs)
    print("\n" + "=" * 80)
    print(f"[*] 5-Fold 전체 OOF 원시 성능 (단일 Th=0.50): F1={raw_f1*100:.2f}%, Acc={raw_acc*100:.2f}%, AUC={raw_auc:.4f}")

    # 2. 시점별 최적 임계값 자동 최적화
    opt_thresholds = find_optimal_thresholds(y, oof_raw_probs, views)
    print("\n" + "=" * 80)
    print(" [시점별 F1 극대화 최적 임계값 도출]")
    print("=" * 80)
    for vn, th in opt_thresholds.items():
        print(f"  - {VIEW_ALIAS_MAP[vn]:<22}: 최적 임계값 Th={th:.2f}")

    # 3. 최적 임계값 적용 캘리브레이션 확률 및 최종 판정 산출
    oof_cal_probs = np.zeros(len(y), dtype=np.float32)
    oof_preds = np.zeros(len(y), dtype=int)

    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        mask_v = (views == vn)
        th = opt_thresholds[vn]
        p_v = oof_raw_probs[mask_v]
        p_cal_v = np.where(p_v >= th, 0.50 + 0.50 * (p_v - th) / (1.0 - th + 1e-6), 0.50 * p_v / (th + 1e-6))
        oof_cal_probs[mask_v] = p_cal_v
        oof_preds[mask_v] = (p_v >= th).astype(int)

    total_acc = accuracy_score(y, oof_preds)
    total_prec = precision_score(y, oof_preds, zero_division=0)
    total_rec = recall_score(y, oof_preds, zero_division=0)
    total_f1 = f1_score(y, oof_preds, zero_division=0)
    total_f1_macro = f1_score(y, oof_preds, average='macro')
    total_auc = roc_auc_score(y, oof_raw_probs)
    cm = confusion_matrix(y, oof_preds).tolist()

    print("\n" + "=" * 80)
    print(" [전체 770건 5-Fold OOF 시점 보정 최종 검증 결과 - v6.0 Zero-Calibrated (37D)]")
    print("=" * 80)
    print(f"[*] 기본 정확도(Accuracy) : {total_acc * 100:.2f}%")
    print(f"[*] 정밀도(Precision)     : {total_prec * 100:.2f}%")
    print(f"[*] 재현율(Recall)        : {total_rec * 100:.2f}%")
    print(f"[*] F1-Score (결함 클래스) : ★ {total_f1 * 100:.2f}% (v5.1 81.60% 대비 대조)")
    print(f"[*] F1-Score (Macro)      : {total_f1_macro * 100:.2f}%")
    print(f"[*] ROC-AUC               : ★ {total_auc:.4f} (v5.1 0.8841 대비 대조)")
    print(f"[*] 혼동 행렬(Confusion Matrix):\n  정상예측  결함예측\n[[ {cm[0][0]:6d}  {cm[0][1]:6d} ]  (실제 정상)\n [ {cm[1][0]:6d}  {cm[1][1]:6d} ]] (실제 결함)")
    print(f"[*] 총 오류 수: {cm[0][1] + cm[1][0]}건 (FP: {cm[0][1]}건, FN: {cm[1][0]}건)")

    # 4. 공식 5대 시점 전수 성능 감사
    print("\n" + "=" * 80)
    print(" [공식 5대 시점(시점 A~E) 전수 다차원 성능 감사]")
    print("=" * 80)

    view_results = {}
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        alias = VIEW_ALIAS_MAP[vn]
        v_idx = np.where(views == vn)[0]
        y_v = y[v_idx]
        pred_v = oof_preds[v_idx]
        prob_v = oof_raw_probs[v_idx]
        th_v = opt_thresholds[vn]

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

    # 5. 피험자 1~9번 전수 성능 감사 (체형 편차 검증)
    print("\n" + "=" * 80)
    print(" [피험자 1~9번 전수 성능 감사 (체형 편차 개선도 대조)]")
    print("=" * 80)

    subject_results = {}
    for sub in sorted(list(set(groups))):
        s_idx = np.where(groups == sub)[0]
        y_s = y[s_idx]
        pred_s = oof_preds[s_idx]
        s_cm = confusion_matrix(y_s, pred_s).tolist()

        s_acc = accuracy_score(y_s, pred_s)
        s_f1 = f1_score(y_s, pred_s, zero_division=0)
        fp_s = s_cm[0][1] if len(s_cm) > 1 else 0
        fn_s = s_cm[1][0] if len(s_cm) > 1 else 0

        subject_results[str(sub)] = {
            "total_count": len(s_idx),
            "accuracy": float(s_acc),
            "f1_score": float(s_f1),
            "fp_count": int(fp_s),
            "fn_count": int(fn_s),
            "confusion_matrix": s_cm
        }
        print(f"  피험자 {sub}번 (샘플 {len(s_idx):2d}건) | Acc: {s_acc*100:5.2f}% | F1: {s_f1*100:5.2f}% | FP: {fp_s:2d}건 | FN: {fn_s:2d}건")

    # 6. 3-State 신뢰도 판정 성능 감사
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
        "model_version": "v6.0 Zero-Calibrated (37D)",
        "description": "37D Zero-Calibrated and Morphological Ratio Tri-Ensemble Model",
        "feature_count": 37,
        "feature_names": FEATURE_NAMES_37_D5_V6,
        "optimal_thresholds": opt_thresholds,
        "overall_metrics": {
            "accuracy": float(total_acc),
            "precision": float(total_prec),
            "recall": float(total_rec),
            "f1_score": float(total_f1),
            "f1_macro": float(total_f1_macro),
            "roc_auc": float(total_auc),
            "total_errors": int(cm[0][1] + cm[1][0]),
            "fp_count": int(cm[0][1]),
            "fn_count": int(cm[1][0]),
            "confusion_matrix": cm
        },
        "view_breakdown": view_results,
        "subject_breakdown": subject_results,
        "three_state_results": three_state_results,
        "fold_metrics": fold_metrics
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] v6 결과 JSON 저장 완료: {OUT_RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
