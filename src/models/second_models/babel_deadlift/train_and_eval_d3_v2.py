# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: train_and_eval_d3_v2.py
Purpose: 위상 불변 적응형 지면 기준선 및 28D 생체역학 지면 접지 피처셋 기반
         d3 v2.0 모델 5-Fold 피험자 격리 학습 및 v1 vs v2 성능 대조 평가
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

from src.models.second_models.babel_deadlift.d3_model_v2 import (
    BabelDeadliftD3ModelV2,
    extract_d3_v2_features_from_seq,
    FEATURE_NAMES_28_D3
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_model_v2_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """데드리프트 385개 멀티뷰 시퀀스 로드 및 28D 피처 추출"""
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
        y_d3 = 0 if conds.get("발바닥 지면 고정", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        data = np.load(npz_path)
        kps_33_3d = data["kps_33_3d_64f"]

        f28 = extract_d3_v2_features_from_seq(kps_33_3d)

        feats_list.append(f28)
        y_list.append(y_d3)
        subjects_list.append(sub)
        views_list.append(vn)
        clip_names.append(base_name)

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    print(f"[+] 28D 피처 추출 완료 (경과: {time.time()-t0:.1f}초)")
    print(f"  - 28D 피처 형태 : {X.shape}")
    print(f"  - 라벨 분포     : 정상(0) {np.sum(y==0)}건 vs 결함(1) {np.sum(y==1)}건 (결함 비율: {np.mean(y)*100:.1f}%)")
    print(f"  - 피험자 분포   : {np.unique(subjects).tolist()}")
    print(f"  - 시점 분포     : {np.unique(views).tolist()}")

    return X, y, subjects, views, clip_names


def evaluate_3state(y_true: np.ndarray, y_prob: np.ndarray, tau_normal: float, tau_defect: float) -> Dict[str, Any]:
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
        cm = confusion_matrix(y_decided_true, y_decided_pred).tolist()
    else:
        f1, acc, prec, rec = 0.0, 0.0, 0.0, 0.0
        cm = [[0, 0], [0, 0]]

    return {
        "tau_normal": tau_normal,
        "tau_defect": tau_defect,
        "coverage": coverage,
        "decline_rate": decline_rate,
        "total_decided": int(np.sum(decided_mask)),
        "total_undetermined": int(np.sum(~decided_mask)),
        "f1_score": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": {
            "TN": cm[0][0] if len(cm) > 1 else 0,
            "FP": cm[0][1] if len(cm) > 1 else 0,
            "FN": cm[1][0] if len(cm) > 1 else 0,
            "TP": cm[1][1] if len(cm) > 1 else 0
        }
    }


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d3 v2.0 모델 재학습 및 평가")
    print("  - 핵심 기술: 위상 불변 적응형 기립 기준선 + 28D 생체역학 지면 접지 피처셋")
    print("  - 데이터셋: AI-Hub 바벨 데드리프트 385개 멀티뷰 시퀀스 (피험자 1~9번 전원, 공식 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (피험자 완전 격리)")
    print("=" * 100)

    # 1. 데이터셋 로드
    X, y, subjects, views, clip_names = load_dataset()

    # 2. 5-Fold 피험자 격리 검증 설정
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    fold_models = []
    print("\n[*] 5-Fold 피험자 격리 학습 시작 (CatBoost + LightGBM + Random Forest)...")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        va_subs = np.unique(subjects[val_idx]).tolist()

        model = BabelDeadliftD3ModelV2()
        model.fit_ensemble(X_tr, y_tr, random_state=42 + fold)
        fold_models.append(model)

        val_probs = model.predict_proba_features(X_va)
        oof_probs[val_idx] = val_probs

        val_pred_50 = (val_probs >= 0.50).astype(int)
        f1_fold = f1_score(y_va, val_pred_50, zero_division=0)
        acc_fold = accuracy_score(y_va, val_pred_50)
        auc_fold = roc_auc_score(y_va, val_probs) if len(np.unique(y_va)) > 1 else 0.5
        print(f"  - Fold {fold}/5: 검증 피험자={va_subs}, 샘플수={len(y_va)} | Acc={acc_fold*100:.2f}%, F1={f1_fold*100:.2f}%, AUC={auc_fold:.4f}")

        fold_weight_path = os.path.join(OUT_WEIGHTS_DIR, f"d3_v2_fold{fold}.pkl")
        model.save_weights(fold_weight_path)

    # 3. OOF 종합 평가 (Threshold=0.50)
    pred_50 = (oof_probs >= 0.50).astype(int)
    f1_50 = float(f1_score(y, pred_50, zero_division=0))
    acc_50 = float(accuracy_score(y, pred_50))
    prec_50 = float(precision_score(y, pred_50, zero_division=0))
    rec_50 = float(recall_score(y, pred_50, zero_division=0))
    auc_total = float(roc_auc_score(y, oof_probs))
    cm_50 = confusion_matrix(y, pred_50)

    print("\n" + "=" * 80)
    print("  [d3 v2.0 OOF 종합 성능 (기본 임계치 0.50)]")
    print(f"  - Accuracy  : {acc_50*100:.2f}% (v1.0 대비: {acc_50*100 - 75.84:+.2f}%p)")
    print(f"  - Precision : {prec_50*100:.2f}% (v1.0 대비: {prec_50*100 - 48.54:+.2f}%p)")
    print(f"  - Recall    : {rec_50*100:.2f}% (v1.0 대비: {rec_50*100 - 55.56:+.2f}%p)")
    print(f"  - F1-Score  : {f1_50*100:.2f}% (v1.0 대비: {f1_50*100 - 51.81:+.2f}%p)")
    print(f"  - ROC-AUC   : {auc_total:.4f} (v1.0 대비: {auc_total - 0.7826:+.4f})")
    print(f"  - Confusion Matrix:\n{cm_50}")
    print("=" * 80)

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
    print(f"\n[*] d3 v2.0 최적 2진 임계치: tau = {best_th:.3f}")
    print(f"  - Accuracy  : {accuracy_score(y, pred_opt)*100:.2f}%")
    print(f"  - Precision : {precision_score(y, pred_opt, zero_division=0)*100:.2f}%")
    print(f"  - Recall    : {recall_score(y, pred_opt, zero_division=0)*100:.2f}%")
    print(f"  - Best F1   : {best_f1*100:.2f}%")

    # 5. 공식 5대 시점 전수 감사
    print("\n" + "=" * 90)
    print("  [d3 v2.0 공식 5대 카메라 시점 전수 성능 감사]")
    print("=" * 90)
    view_audit = {}
    for v_code in ["view5", "view2", "view3", "view4", "view1"]:
        v_name = VIEW_ALIAS_MAP[v_code]
        mask = (views == v_code)
        if np.sum(mask) == 0:
            continue
        v_y = y[mask]
        v_probs = oof_probs[mask]
        v_pred = (v_probs >= best_th).astype(int)

        v_f1 = float(f1_score(v_y, v_pred, zero_division=0))
        v_acc = float(accuracy_score(v_y, v_pred))
        v_prec = float(precision_score(v_y, v_pred, zero_division=0))
        v_rec = float(recall_score(v_y, v_pred, zero_division=0))
        v_auc = float(roc_auc_score(v_y, v_probs)) if len(np.unique(v_y)) > 1 else 0.5

        view_audit[v_name] = {
            "view_code": v_code,
            "sample_count": int(np.sum(mask)),
            "f1": v_f1,
            "accuracy": v_acc,
            "precision": v_prec,
            "recall": v_rec,
            "auc": v_auc
        }
        print(f"  - {v_name:<25}: Acc={v_acc*100:.2f}%, F1={v_f1*100:.2f}%, Prec={v_prec*100:.2f}%, Rec={v_rec*100:.2f}%, AUC={v_auc:.4f}")

    # 6. 3-State 선택적 판정 파레토 탐색
    print("\n[*] d3 v2.0 3-State 선택적 판정(Selective Classification) 그리드 탐색...")
    best_3s = None
    best_3s_score = 0.0

    for tn in np.linspace(0.10, 0.48, 39):
        for td in np.linspace(0.52, 0.90, 39):
            if tn >= td:
                continue
            res_3s = evaluate_3state(y, oof_probs, tn, td)
            if res_3s["coverage"] >= 0.90:
                if res_3s["f1_score"] > best_3s_score:
                    best_3s_score = res_3s["f1_score"]
                    best_3s = res_3s

    if best_3s is not None:
        print(f"  [★ d3 v2.0 3-State 최적 세팅 (커버리지 >= 90%)]")
        print(f"  - tau_normal = {best_3s['tau_normal']:.3f}, tau_defect = {best_3s['tau_defect']:.3f}")
        print(f"  - Coverage   : {best_3s['coverage']*100:.2f}% (보류: {best_3s['total_undetermined']}건)")
        print(f"  - Accuracy   : {best_3s['accuracy']*100:.2f}%")
        print(f"  - Precision  : {best_3s['precision']*100:.2f}%")
        print(f"  - Recall     : {best_3s['recall']*100:.2f}%")
        print(f"  - 3-State F1 : {best_3s['f1_score']*100:.2f}%")

    # 7. 결과 JSON 저장
    results_dict = {
        "model_name": "BabelDeadliftD3ModelV2",
        "exercise": "02_바벨_데드리프트",
        "defect": "d3_발바닥_지면_고정",
        "feature_dim": 28,
        "feature_names": FEATURE_NAMES_28_D3,
        "binary_metrics_th50": {
            "accuracy": acc_50,
            "precision": prec_50,
            "recall": rec_50,
            "f1_score": f1_50,
            "roc_auc": auc_total,
            "confusion_matrix": {
                "TN": int(cm_50[0, 0]),
                "FP": int(cm_50[0, 1]),
                "FN": int(cm_50[1, 0]),
                "TP": int(cm_50[1, 1])
            }
        },
        "binary_metrics_optimal": {
            "threshold": best_th,
            "accuracy": float(accuracy_score(y, pred_opt)),
            "precision": float(precision_score(y, pred_opt, zero_division=0)),
            "recall": float(recall_score(y, pred_opt, zero_division=0)),
            "f1_score": best_f1
        },
        "v1_vs_v2_comparison": {
            "d3_v1_f1_th50": 51.81,
            "d3_v2_f1_th50": f1_50 * 100,
            "delta_f1_th50": (f1_50 * 100) - 51.81,
            "d3_v1_3state_f1": 54.02,
            "d3_v2_3state_f1": best_3s["f1_score"] * 100 if best_3s else 0.0,
            "delta_3state_f1": (best_3s["f1_score"] * 100 - 54.02) if best_3s else 0.0
        },
        "official_3state_setting": best_3s,
        "view_audit": view_audit
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 공식 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")

    # 8. 전체 데이터 기반 최종 프로덕션 모델 저장
    print("\n[*] 전체 데이터 기반 최종 d3 v2.0 프로덕션 가중치 학습 및 저장...")
    full_model = BabelDeadliftD3ModelV2()
    if best_3s is not None:
        full_model.tau_normal = best_3s["tau_normal"]
        full_model.tau_defect = best_3s["tau_defect"]
    full_model.decision_threshold = best_th
    full_model.fit_ensemble(X, y, random_state=42)
    final_weight_path = os.path.join(OUT_WEIGHTS_DIR, "d3_model_v2_best.pkl")
    full_model.save_weights(final_weight_path)
    print(f"[+] d3 v2.0 최종 가중치 저장 완료: {final_weight_path}")


if __name__ == "__main__":
    main()
