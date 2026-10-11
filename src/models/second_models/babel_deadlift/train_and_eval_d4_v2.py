# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d4. 무릎과 골반이 동시에 펴짐 위반 (Knee-Hip Simultaneous Extension Synchrony)
Script: train_and_eval_d4_v2.py
Purpose: v2.0 29D 생체역학 피처셋(각도 비율 5D 결합) 및 Tri-Ensemble 아키텍처를 기반으로,
         바벨 데드리프트 770개 멀티뷰 시퀀스(피험자 1~9번 전원, 공식 5대 시점 전수)로
         5-Fold 피험자 격리 재학습, 공식 5대 시점 전수 감사, v1 vs v2 정밀 비교 수행
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

from src.models.second_models.babel_deadlift.d4_model_v2 import (
    BabelDeadliftD4ModelV2,
    extract_d4_v2_features_from_seq,
    FEATURE_NAMES_29_D4
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d4_model_v2_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset_v2() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """데드리프트 770개 멀티뷰 시퀀스 로드 및 29D 피처 추출"""
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 분석 대상 데드리프트 시퀀스 수: {len(npz_files)}개", flush=True)

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

        # d4 무릎과 골반이 동시에 펴짐 위반 (정상 True -> 0, 결함 False -> 1)
        y_d4 = 0 if conds.get("무릎과 골반이 동시에 펴짐", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_33_3d = data["kps_33_3d_64f"]

        f29 = extract_d4_v2_features_from_seq(kps_33_3d)

        feats_list.append(f29)
        y_list.append(y_d4)
        subjects_list.append(sub)
        views_list.append(vn)
        clip_names.append(base_name)

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    print(f"[+] 29D 피처 추출 완료 (경과: {time.time()-t0:.1f}초)", flush=True)
    print(f"  - 피처 규격 : {X.shape}", flush=True)
    print(f"  - 라벨 분포 : 정상(0) {np.sum(y==0)}건 vs 결함(1) {np.sum(y==1)}건 (결함 비율: {np.mean(y)*100:.1f}%)", flush=True)

    return X, y, subjects, views, clip_names


def evaluate_3state(y_true: np.ndarray, y_prob: np.ndarray, tau_normal: float, tau_defect: float) -> Dict[str, Any]:
    """3-State 선택적 판정 지표 계산"""
    decided_mask = (y_prob <= tau_normal) | (y_prob >= tau_defect)
    coverage = float(np.mean(decided_mask))
    decline_rate = float(1.0 - coverage)

    y_decided_true = y_true[decided_mask]
    y_decided_pred = (y_prob[decided_mask] >= tau_defect).astype(int)

    if len(y_decided_true) > 0 and len(np.unique(y_decided_true)) > 1:
        f1 = float(f1_score(y_decided_true, y_decided_pred))
        acc = float(accuracy_score(y_decided_true, y_decided_pred))
        prec = float(precision_score(y_decided_true, y_decided_pred, zero_division=0))
        rec = float(recall_score(y_decided_true, y_decided_pred, zero_division=0))
    elif len(y_decided_true) > 0:
        acc = float(accuracy_score(y_decided_true, y_decided_pred))
        f1 = 0.0
        prec = 0.0
        rec = 0.0
    else:
        acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0

    return {
        "coverage": coverage,
        "decline_rate": decline_rate,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "n_decided": int(np.sum(decided_mask)),
        "n_total": int(len(y_true))
    }


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 d4 v2 (29D 무릎-골반 비율 결합) 5-Fold 학습 & 평가")
    print("  - 모델 구조: 29D 생체역학 피처 + 3-Way Tri-Ensemble (LightGBM + CatBoost + RF)")
    print("  - 데이터셋: 770개 시퀀스 (피험자 1~9번 전원, 공식 5대 시점 전수)")
    print("=" * 80, flush=True)

    # 1. 데이터 로드 및 29D 피처 추출
    X, y, subjects, views, clip_names = load_dataset_v2()

    # 2. 피험자 격리 5-Fold 교차 검증
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    fold_models = []
    print("\n[*] 5-Fold 피험자 격리 학습 시작 (CatBoost + LightGBM + Random Forest)...", flush=True)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        va_subs = np.unique(subjects[val_idx]).tolist()

        model = BabelDeadliftD4ModelV2()
        model.fit(X_tr, y_tr, random_state=42 + fold)
        fold_models.append(model)

        val_probs = model.predict_proba(X_va)
        oof_probs[val_idx] = val_probs

        val_pred_50 = (val_probs >= 0.50).astype(int)
        f1_fold = f1_score(y_va, val_pred_50, zero_division=0)
        acc_fold = accuracy_score(y_va, val_pred_50)
        auc_fold = roc_auc_score(y_va, val_probs) if len(np.unique(y_va)) > 1 else 0.5
        print(f"  - Fold {fold}/5: 검증 피험자={va_subs}, 샘플수={len(y_va)} | Acc={acc_fold*100:.2f}%, F1={f1_fold*100:.2f}%, AUC={auc_fold:.4f}", flush=True)

        # Fold 가중치 저장
        fold_weight_path = os.path.join(OUT_WEIGHTS_DIR, f"d4_v2_fold{fold}.pkl")
        model.save(fold_weight_path)

    # 3. OOF 종합 성능 평가 (기본 임계치 0.50)
    pred_50 = (oof_probs >= 0.50).astype(int)
    f1_50 = float(f1_score(y, pred_50, zero_division=0))
    acc_50 = float(accuracy_score(y, pred_50))
    prec_50 = float(precision_score(y, pred_50, zero_division=0))
    rec_50 = float(recall_score(y, pred_50, zero_division=0))
    auc_total = float(roc_auc_score(y, oof_probs))
    cm_50 = confusion_matrix(y, pred_50)

    print("\n" + "=" * 80)
    print("  [OOF 종합 성능 (기본 임계치 0.50)]")
    print(f"  - Accuracy  : {acc_50*100:.2f}%")
    print(f"  - Precision : {prec_50*100:.2f}%")
    print(f"  - Recall    : {rec_50*100:.2f}%")
    print(f"  - F1-Score  : {f1_50*100:.2f}%")
    print(f"  - ROC-AUC   : {auc_total:.4f}")
    print(f"  - Confusion Matrix:\n{cm_50}")
    print("=" * 80, flush=True)

    # 4. 최적 2진 임계치 탐색
    best_th = 0.50
    best_f1 = 0.0
    for th in np.linspace(0.1, 0.9, 81):
        p = (oof_probs >= th).astype(int)
        score = f1_score(y, p, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = th

    pred_opt = (oof_probs >= best_th).astype(int)
    print(f"\n[*] 최적 2진 임계치: tau = {best_th:.3f}")
    print(f"  - Accuracy  : {accuracy_score(y, pred_opt)*100:.2f}%")
    print(f"  - Precision : {precision_score(y, pred_opt, zero_division=0)*100:.2f}%")
    print(f"  - Recall    : {recall_score(y, pred_opt, zero_division=0)*100:.2f}%")
    print(f"  - F1-Score  : {best_f1*100:.2f}%", flush=True)

    # 5. 3-State 선택적 판정 파라미터 탐색 (Coverage >= 80%)
    print("\n[*] 3-State 선택적 판정 파라미터 탐색 (Coverage >= 80%)...", flush=True)
    best_3state = None
    best_3state_score = -1.0
    for tau_n in np.linspace(0.15, 0.45, 13):
        for tau_d in np.linspace(0.50, 0.80, 13):
            if tau_n >= tau_d:
                continue
            metrics = evaluate_3state(y, oof_probs, tau_n, tau_d)
            if metrics["coverage"] >= 0.80:
                if metrics["f1"] > best_3state_score:
                    best_3state_score = metrics["f1"]
                    best_3state = {
                        "tau_normal": float(tau_n),
                        "tau_defect": float(tau_d),
                        **metrics
                    }

    if best_3state is not None:
        print(f"  - 최적 3-State: tau_normal={best_3state['tau_normal']:.2f}, tau_defect={best_3state['tau_defect']:.2f}")
        print(f"  - Coverage    : {best_3state['coverage']*100:.2f}% (보류율: {best_3state['decline_rate']*100:.2f}%)")
        print(f"  - Accuracy    : {best_3state['accuracy']*100:.2f}%")
        print(f"  - Precision   : {best_3state['precision']*100:.2f}%")
        print(f"  - Recall      : {best_3state['recall']*100:.2f}%")
        print(f"  - F1-Score    : {best_3state['f1']*100:.2f}%", flush=True)

    # 6. 공식 5대 시점 전수 감사
    print("\n" + "=" * 80)
    print("  [공식 5대 시점 전수 감사 (Official 5-View Audit)]")
    view_metrics = {}
    for vn in sorted(np.unique(views).tolist()):
        alias = VIEW_ALIAS_MAP.get(vn, vn)
        mask_v = (views == vn)
        y_v = y[mask_v]
        prob_v = oof_probs[mask_v]
        pred_v_50 = (prob_v >= 0.50).astype(int)
        pred_v_opt = (prob_v >= best_th).astype(int)

        f1_v_50 = float(f1_score(y_v, pred_v_50, zero_division=0))
        acc_v_50 = float(accuracy_score(y_v, pred_v_50))
        auc_v = float(roc_auc_score(y_v, prob_v)) if len(np.unique(y_v)) > 1 else 0.5

        f1_v_opt = float(f1_score(y_v, pred_v_opt, zero_division=0))
        acc_v_opt = float(accuracy_score(y_v, pred_v_opt))

        m3_v = evaluate_3state(y_v, prob_v, best_3state['tau_normal'], best_3state['tau_defect']) if best_3state else {}

        view_metrics[vn] = {
            "alias": alias,
            "total_samples": int(len(y_v)),
            "defects": int(np.sum(y_v == 1)),
            "normals": int(np.sum(y_v == 0)),
            "f1_base": f1_v_50,
            "acc_base": acc_v_50,
            "auc": auc_v,
            "f1_opt": f1_v_opt,
            "acc_opt": acc_v_opt,
            "3state_f1": m3_v.get("f1", 0.0),
            "3state_acc": m3_v.get("accuracy", 0.0),
            "3state_coverage": m3_v.get("coverage", 0.0)
        }
        print(f"  [{alias}] {vn}:")
        print(f"    * Base(0.50)   : Acc={acc_v_50*100:.2f}%, F1={f1_v_50*100:.2f}%, AUC={auc_v:.4f}")
        print(f"    * Opt({best_th:.2f})    : Acc={acc_v_opt*100:.2f}%, F1={f1_v_opt*100:.2f}%")
        if best_3state:
            print(f"    * 3-State      : Cov={m3_v['coverage']*100:.1f}%, Acc={m3_v['accuracy']*100:.2f}%, F1={m3_v['f1']*100:.2f}%")

    print("=" * 80, flush=True)

    # 7. 피처 중요도 분석
    print("\n[*] 29D 생체역학 피처 중요도 분석 (LightGBM 기준)...", flush=True)
    importance_sums = np.zeros(len(FEATURE_NAMES_29_D4), dtype=np.float32)
    valid_count = 0
    for m in fold_models:
        if m.clf_lgb is not None:
            importance_sums += m.clf_lgb.feature_importances_
            valid_count += 1
    if valid_count > 0:
        importance_sums /= valid_count

    sorted_indices = np.argsort(importance_sums)[::-1]
    top_features = []
    print("  [Top 12 중요 피처]")
    for rank, idx in enumerate(sorted_indices[:12], 1):
        fname = FEATURE_NAMES_29_D4[idx]
        score = float(importance_sums[idx])
        is_new = "★ (신규 비율 피처)" if idx >= 24 else ""
        top_features.append({"rank": rank, "feature": fname, "importance": score, "is_new": bool(idx >= 24)})
        print(f"    {rank:2d}. {fname:<32} : {score:.2f} {is_new}")

    # 8. 결과 JSON 저장
    full_results = {
        "model_version": "d4_v2",
        "feature_dim": 29,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(y),
        "defects": int(np.sum(y == 1)),
        "normals": int(np.sum(y == 0)),
        "base_0_50": {
            "accuracy": acc_50,
            "precision": prec_50,
            "recall": rec_50,
            "f1": f1_50,
            "roc_auc": auc_total
        },
        "opt_binary": {
            "threshold": float(best_th),
            "accuracy": float(accuracy_score(y, pred_opt)),
            "precision": float(precision_score(y, pred_opt, zero_division=0)),
            "recall": float(recall_score(y, pred_opt, zero_division=0)),
            "f1": float(best_f1)
        },
        "best_3state": best_3state,
        "view_metrics": view_metrics,
        "top_features": top_features
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    print(f"\n[+] v2 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
