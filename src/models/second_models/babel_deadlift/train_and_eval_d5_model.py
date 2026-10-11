# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: train_and_eval_d5_model.py
Purpose: 28D 바벨 밀착 생체역학 피처셋 및 Tri-Ensemble 아키텍처를 기반으로,
         50:50 클래스 균형 770개 시퀀스(피험자 1~9번 전원, 공식 5대 시점 전수)에 대한
         5-Fold StratifiedGroupKFold 피험자 완전 격리 학습 및 다차원 성능 감사 수행
"""

import os
import sys
import json
import glob
import time
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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

from src.models.second_models.babel_deadlift.d5_model_v1 import (
    BabelDeadliftD5ModelV1,
    extract_d5_v1_features_from_seq,
    FEATURE_NAMES_28_D5
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d5_model_v1_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """d5 캐시 디렉터리에서 770개 시퀀스 로드 및 28D 피처 추출"""
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 분석 대상 데드리프트 시퀀스 수: {len(npz_files)}개")

    feats_list = []
    y_list = []
    subjects_list = []
    views_list = []
    clip_names = []

    t0 = time.time()
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

        # d5 바벨 궤적과 몸 밀착 위반 (오류 발생 시 1, 정상 시 0)
        y_d5 = 0 if conds.get("바벨 궤적과 몸 밀착", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_64f = data["kps_33_3d_64f"]

        feats = extract_d5_v1_features_from_seq(kps_64f, view_name=vn)

        feats_list.append(feats)
        y_list.append(y_d5)
        subjects_list.append(sub)
        views_list.append(vn)
        clip_names.append(folder_name)

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    groups = np.array(subjects_list, dtype=int)
    views = np.array(views_list)

    print(f"[+] 데이터 로드 및 28D 피처 추출 완료: {X.shape} (경과시간: {time.time()-t0:.1f}초)")
    print(f"    - 클래스 분포: 결함(1) {np.sum(y == 1)}건 ({np.mean(y == 1)*100:.1f}%) vs 정상(0) {np.sum(y == 0)}건 ({np.mean(y == 0)*100:.1f}%)")
    return X, y, groups, views, clip_names


def run_5fold_cross_validation(X: np.ndarray, y: np.ndarray, groups: np.ndarray, views: np.ndarray):
    print("\n" + "=" * 90)
    print("  [Step 1] 5-Fold StratifiedGroupKFold 피험자 완전 격리 교차검증 시작")
    print("=" * 90)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    oof_preds = np.zeros(len(y), dtype=int)
    fold_metrics = []

    best_fold_model = None
    best_f1 = -1.0

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        val_subs = [int(s) for s in sorted(list(set(groups[val_idx])))]
        print(f"\n--- [Fold {fold+1}/5] 검증 피험자: {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건) ---")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = BabelDeadliftD5ModelV1()
        model.fit(X_train, y_train)

        # 예측
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)

        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds

        acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        try:
            auc = roc_auc_score(y_val, val_probs)
        except Exception:
            auc = 0.5

        print(f"  > Fold {fold+1} 결과: Acc={acc*100:.2f}%, Prec={prec*100:.2f}%, Rec={rec*100:.2f}%, F1={f1*100:.2f}%, AUC={auc:.4f}")
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

        # 가중치 저장
        fold_weight_path = os.path.join(OUT_WEIGHTS_DIR, f"d5_v1_fold{fold+1}.pkl")
        model.save(fold_weight_path)

        if f1 > best_f1:
            best_f1 = f1
            best_fold_model = model
            best_weight_path = os.path.join(OUT_WEIGHTS_DIR, "d5_model_v1_best.pkl")
            model.save(best_weight_path)

    # 전체 OOF 지표
    total_acc = accuracy_score(y, oof_preds)
    total_prec = precision_score(y, oof_preds, zero_division=0)
    total_rec = recall_score(y, oof_preds, zero_division=0)
    total_f1 = f1_score(y, oof_preds, zero_division=0)
    total_auc = roc_auc_score(y, oof_probs)
    cm = confusion_matrix(y, oof_preds).tolist()

    print("\n" + "=" * 90)
    print(f"  [전체 OOF 종합 결과 (th=0.50)]")
    print(f"  - Accuracy : {total_acc*100:.2f}%")
    print(f"  - Precision: {total_prec*100:.2f}%")
    print(f"  - Recall   : {total_rec*100:.2f}%")
    print(f"  - F1-Score : {total_f1*100:.2f}%")
    print(f"  - ROC-AUC  : {total_auc:.4f}")
    print(f"  - 혼동 행렬: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print("=" * 90)

    # 최적 Threshold 탐색
    best_th = 0.50
    best_th_f1 = total_f1
    for th in np.arange(0.20, 0.80, 0.02):
        th_preds = (oof_probs >= th).astype(int)
        th_f1 = f1_score(y, th_preds, zero_division=0)
        if th_f1 > best_th_f1:
            best_th_f1 = th_f1
            best_th = float(th)

    th_preds = (oof_probs >= best_th).astype(int)
    print(f"\n[*] 최적 Threshold ({best_th:.2f}) 적용 시:")
    print(f"    - F1-Score : {best_th_f1*100:.2f}% (기본 0.50 대비 +{(best_th_f1 - total_f1)*100:.2f}%p)")
    print(f"    - Accuracy : {accuracy_score(y, th_preds)*100:.2f}%")
    print(f"    - Recall   : {recall_score(y, th_preds)*100:.2f}%")
    print(f"    - Precision: {precision_score(y, th_preds)*100:.2f}%")

    # 공식 5대 시점 전수 감사
    print("\n" + "=" * 90)
    print("  [Step 2] 공식 5대 카메라 시점 전수 성능 감사 (Multi-View Audit)")
    print("=" * 90)
    view_audit = {}
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        alias = VIEW_ALIAS_MAP[vn]
        v_mask = (views == vn)
        v_y = y[v_mask]
        v_probs = oof_probs[v_mask]
        v_preds = (v_probs >= 0.50).astype(int)

        v_acc = accuracy_score(v_y, v_preds)
        v_prec = precision_score(v_y, v_preds, zero_division=0)
        v_rec = recall_score(v_y, v_preds, zero_division=0)
        v_f1 = f1_score(v_y, v_preds, zero_division=0)
        try:
            v_auc = roc_auc_score(v_y, v_probs)
        except Exception:
            v_auc = 0.5

        print(f"  [{alias}] 샘플={len(v_y)}개 | Acc={v_acc*100:.2f}% | F1={v_f1*100:.2f}% | Prec={v_prec*100:.2f}% | Rec={v_rec*100:.2f}% | AUC={v_auc:.4f}")
        view_audit[vn] = {
            "alias": alias,
            "sample_count": int(np.sum(v_mask)),
            "accuracy": float(v_acc),
            "precision": float(v_prec),
            "recall": float(v_rec),
            "f1_score": float(v_f1),
            "roc_auc": float(v_auc)
        }

    # 3-State 선택적 판정 파레토 분석
    print("\n" + "=" * 90)
    print("  [Step 3] 3-State 선택적 판정 파레토 분석 (RETAKE 격리)")
    print("=" * 90)
    pareto_results = []
    target_coverages = [0.90, 0.85, 0.80, 0.75]

    for target_cov in target_coverages:
        best_cand = None
        best_cand_f1 = -1.0
        for tau_norm in np.arange(0.20, 0.46, 0.02):
            for tau_def in np.arange(0.52, 0.80, 0.02):
                if tau_norm >= tau_def:
                    continue
                # 판정
                is_retained = (oof_probs < tau_norm) | (oof_probs >= tau_def)
                cov = np.mean(is_retained)
                if cov < target_cov:
                    continue
                retained_y = y[is_retained]
                retained_pred = (oof_probs[is_retained] >= tau_def).astype(int)
                if len(retained_y) == 0:
                    continue
                c_f1 = f1_score(retained_y, retained_pred, zero_division=0)
                if c_f1 > best_cand_f1:
                    best_cand_f1 = c_f1
                    c_acc = accuracy_score(retained_y, retained_pred)
                    c_prec = precision_score(retained_y, retained_pred, zero_division=0)
                    c_rec = recall_score(retained_y, retained_pred, zero_division=0)
                    best_cand = {
                        "target_coverage": target_cov,
                        "actual_coverage": float(cov),
                        "tau_norm": float(tau_norm),
                        "tau_def": float(tau_def),
                        "retained_count": int(np.sum(is_retained)),
                        "retake_count": int(np.sum(~is_retained)),
                        "accuracy": float(c_acc),
                        "precision": float(c_prec),
                        "recall": float(c_rec),
                        "f1_score": float(c_f1)
                    }
        if best_cand:
            pareto_results.append(best_cand)
            print(f"  [커버리지 >={target_cov*100:.0f}%] 실제={best_cand['actual_coverage']*100:.2f}% | 보류={best_cand['retake_count']}건 | [tau_norm={best_cand['tau_norm']:.2f}, tau_def={best_cand['tau_def']:.2f}] -> Acc={best_cand['accuracy']*100:.2f}%, F1={best_cand['f1_score']*100:.2f}%, Prec={best_cand['precision']*100:.2f}%, Rec={best_cand['recall']*100:.2f}%")

    # 결과 JSON 저장
    full_results = {
        "model_name": "d5_model_v1.py",
        "defect_name": "바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)",
        "total_samples": len(y),
        "total_clips": len(y) // 5,
        "class_distribution": {
            "defect_count": int(np.sum(y == 1)),
            "clean_count": int(np.sum(y == 0))
        },
        "default_threshold": 0.50,
        "oof_metrics": {
            "accuracy": float(total_acc),
            "precision": float(total_prec),
            "recall": float(total_rec),
            "f1_score": float(total_f1),
            "roc_auc": float(total_auc),
            "confusion_matrix": cm
        },
        "best_threshold": {
            "threshold": float(best_th),
            "f1_score": float(best_th_f1),
            "accuracy": float(accuracy_score(y, th_preds)),
            "precision": float(precision_score(y, th_preds, zero_division=0)),
            "recall": float(recall_score(y, th_preds, zero_division=0))
        },
        "fold_metrics": fold_metrics,
        "view_audit": view_audit,
        "pareto_results": pareto_results
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 전체 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")

    return full_results


def main():
    print("=" * 90)
    print("  [vidio-health-trainer] 바벨 데드리프트 d5 모델 학습 및 실측 평가 파이프라인")
    print("=" * 90)
    X, y, groups, views, clip_names = load_dataset()
    results = run_5fold_cross_validation(X, y, groups, views)


if __name__ == "__main__":
    main()
