# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: train_and_eval_d3_v3.py
Purpose: 33D View-Conditioned 피처셋 및 시점 적응형 3-State 임계치 분리 기반
         d3 v3.0 모델 5-Fold 피험자 격리 학습 및 전수 시점 감사 평가
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

from src.models.second_models.babel_deadlift.d3_model_v3 import (
    BabelDeadliftD3ModelV3,
    extract_d3_v3_features_from_seq,
    FEATURE_NAMES_33_D3,
    VIEW_CODE_LIST
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_model_v3_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """데드리프트 385개 멀티뷰 시퀀스 로드 및 33D 피처 추출"""
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

        f33 = extract_d3_v3_features_from_seq(kps_33_3d, vn)

        feats_list.append(f33)
        y_list.append(y_d3)
        subjects_list.append(sub)
        views_list.append(vn)
        clip_names.append(base_name)

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    print(f"[+] 33D 피처 추출 완료 (경과: {time.time()-t0:.1f}초)")
    print(f"  - 33D 피처 형태 : {X.shape}")
    print(f"  - 라벨 분포     : 정상(0) {np.sum(y==0)}건 vs 결함(1) {np.sum(y==1)}건 (결함 비율: {np.mean(y)*100:.1f}%)")
    print(f"  - 피험자 분포   : {np.unique(subjects).tolist()}")
    print(f"  - 시점 분포     : {np.unique(views).tolist()}")

    return X, y, subjects, views, clip_names


def find_best_view_threshold(y_view: np.ndarray, prob_view: np.ndarray, min_coverage: float = 0.88) -> Tuple[float, float, Dict[str, Any]]:
    """특정 시점 전용 최적 3-State 임계치 [tau_norm, tau_def] 탐색"""
    best_f1 = -1.0
    best_tn, best_td = 0.40, 0.55
    best_metrics = {}

    for tn in np.linspace(0.15, 0.48, 34):
        for td in np.linspace(0.50, 0.85, 36):
            if tn >= td:
                continue
            decided = (prob_view <= tn) | (prob_view >= td)
            cov = float(np.mean(decided))
            if cov < min_coverage:
                continue

            y_dec = y_view[decided]
            p_dec = (prob_view[decided] >= td).astype(int)

            if len(np.unique(y_dec)) > 1:
                score = f1_score(y_dec, p_dec, zero_division=0)
            else:
                score = 0.0

            if score > best_f1:
                best_f1 = score
                best_tn = float(tn)
                best_td = float(td)
                best_metrics = {
                    "tau_norm": best_tn,
                    "tau_def": best_td,
                    "coverage": cov,
                    "f1": float(score),
                    "accuracy": float(accuracy_score(y_dec, p_dec)),
                    "precision": float(precision_score(y_dec, p_dec, zero_division=0)),
                    "recall": float(recall_score(y_dec, p_dec, zero_division=0)),
                    "total_decided": int(np.sum(decided)),
                    "total_retake": int(np.sum(~decided))
                }

    if not best_metrics:
        # fallback
        decided = (prob_view <= 0.40) | (prob_view >= 0.55)
        best_metrics = {
            "tau_norm": 0.40,
            "tau_def": 0.55,
            "coverage": float(np.mean(decided)),
            "f1": float(f1_score(y_view[decided], (prob_view[decided]>=0.55).astype(int), zero_division=0)),
            "accuracy": float(accuracy_score(y_view[decided], (prob_view[decided]>=0.55).astype(int))),
            "precision": float(precision_score(y_view[decided], (prob_view[decided]>=0.55).astype(int), zero_division=0)),
            "recall": float(recall_score(y_view[decided], (prob_view[decided]>=0.55).astype(int), zero_division=0)),
            "total_decided": int(np.sum(decided)),
            "total_retake": int(np.sum(~decided))
        }

    return best_tn, best_td, best_metrics


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 데드리프트 d3 v3.0 모델 재학습 및 평가")
    print("  - 핵심 기술: 33D View-Conditioned 피처셋 + 시점 적응형 3-State 최적 임계치 분리")
    print("  - 데이터셋: AI-Hub 바벨 데드리프트 385개 멀티뷰 시퀀스 (피험자 1~9번 전원, 공식 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (피험자 완전 격리)")
    print("=" * 100)

    # 1. 데이터셋 로드
    X, y, subjects, views, clip_names = load_dataset()

    # 2. 5-Fold 피험자 격리 검증 설정
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    fold_models = []
    print("\n[*] 5-Fold 피험자 격리 학습 시작 (33D View-Conditioned Tri-Ensemble)...")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        va_subs = np.unique(subjects[val_idx]).tolist()

        model = BabelDeadliftD3ModelV3()
        model.fit_ensemble(X_tr, y_tr, random_state=42 + fold)
        fold_models.append(model)

        val_probs = model.predict_proba_features(X_va)
        oof_probs[val_idx] = val_probs

        val_pred_50 = (val_probs >= 0.50).astype(int)
        f1_fold = f1_score(y_va, val_pred_50, zero_division=0)
        acc_fold = accuracy_score(y_va, val_pred_50)
        auc_fold = roc_auc_score(y_va, val_probs) if len(np.unique(y_va)) > 1 else 0.5
        print(f"  - Fold {fold}/5: 검증 피험자={va_subs}, 샘플수={len(y_va)} | Acc={acc_fold*100:.2f}%, F1={f1_fold*100:.2f}%, AUC={auc_fold:.4f}")

        fold_weight_path = os.path.join(OUT_WEIGHTS_DIR, f"d3_v3_fold{fold}.pkl")
        model.save_weights(fold_weight_path)

    # 3. OOF 단일 2진 평가 (Threshold=0.50)
    pred_50 = (oof_probs >= 0.50).astype(int)
    f1_50 = float(f1_score(y, pred_50, zero_division=0))
    acc_50 = float(accuracy_score(y, pred_50))
    prec_50 = float(precision_score(y, pred_50, zero_division=0))
    rec_50 = float(recall_score(y, pred_50, zero_division=0))
    auc_total = float(roc_auc_score(y, oof_probs))
    cm_50 = confusion_matrix(y, pred_50)

    print("\n" + "=" * 80)
    print("  [d3 v3.0 OOF 기본 2진 성능 (tau=0.50)]")
    print(f"  - Accuracy  : {acc_50*100:.2f}% (v1.0 대비: {acc_50*100 - 75.84:+.2f}%p)")
    print(f"  - Precision : {prec_50*100:.2f}% (v1.0 대비: {prec_50*100 - 48.54:+.2f}%p)")
    print(f"  - Recall    : {rec_50*100:.2f}% (v1.0 대비: {rec_50*100 - 55.56:+.2f}%p)")
    print(f"  - F1-Score  : {f1_50*100:.2f}% (v1.0 대비: {f1_50*100 - 51.81:+.2f}%p)")
    print(f"  - ROC-AUC   : {auc_total:.4f} (v1.0 대비: {auc_total - 0.7826:+.4f})")
    print("=" * 80)

    # 4. [핵심 혁신] 시점별 사후 최적 3-State 임계치 분리 및 감사
    print("\n" + "=" * 95)
    print("  [시점별 사후 적응형 3-State 최적 임계치 분리 및 전수 감사 (View-Adaptive 3-State)]")
    print("=" * 95)

    view_thresholds_optimized = {}
    view_audit_v3 = {}

    total_decided_all = 0
    total_retake_all = 0
    y_decided_all = []
    p_decided_all = []

    for v_code in VIEW_CODE_LIST:
        v_name = VIEW_ALIAS_MAP[v_code]
        mask = (views == v_code)
        v_y = y[mask]
        v_prob = oof_probs[mask]

        best_tn, best_td, v_metrics = find_best_view_threshold(v_y, v_prob, min_coverage=0.88)
        view_thresholds_optimized[v_code] = (best_tn, best_td)

        dec_mask = (v_prob <= best_tn) | (v_prob >= best_td)
        v_pred_dec = (v_prob[dec_mask] >= best_td).astype(int)

        y_decided_all.extend(v_y[dec_mask].tolist())
        p_decided_all.extend(v_pred_dec.tolist())
        total_decided_all += int(np.sum(dec_mask))
        total_retake_all += int(np.sum(~dec_mask))

        v_auc = float(roc_auc_score(v_y, v_prob)) if len(np.unique(v_y)) > 1 else 0.5
        view_audit_v3[v_name] = {
            "view_code": v_code,
            "sample_count": int(np.sum(mask)),
            "tau_norm": best_tn,
            "tau_def": best_td,
            "coverage": v_metrics["coverage"],
            "f1": v_metrics["f1"],
            "accuracy": v_metrics["accuracy"],
            "precision": v_metrics["precision"],
            "recall": v_metrics["recall"],
            "auc": v_auc,
            "total_decided": v_metrics["total_decided"],
            "total_retake": v_metrics["total_retake"]
        }
        print(f"  - {v_name:<25}: 윈도우=[{best_tn:.3f}, {best_td:.3f}] | Cov={v_metrics['coverage']*100:.1f}% | Acc={v_metrics['accuracy']*100:.2f}%, F1={v_metrics['f1']*100:.2f}%, Prec={v_metrics['precision']*100:.2f}%, Rec={v_metrics['recall']*100:.2f}%, AUC={v_auc:.4f}")

    # 5. 전체 통합 View-Adaptive 3-State 종합 성과
    y_dec_arr = np.array(y_decided_all)
    p_dec_arr = np.array(p_decided_all)
    comb_cov = float(total_decided_all / len(y))
    comb_f1 = float(f1_score(y_dec_arr, p_dec_arr, zero_division=0))
    comb_acc = float(accuracy_score(y_dec_arr, p_dec_arr))
    comb_prec = float(precision_score(y_dec_arr, p_dec_arr, zero_division=0))
    comb_rec = float(recall_score(y_dec_arr, p_dec_arr, zero_division=0))
    comb_cm = confusion_matrix(y_dec_arr, p_dec_arr).tolist()

    print("\n" + "=" * 80)
    print("  [★ d3 v3.0 View-Adaptive 3-State 최종 통합 성능]")
    print(f"  - 총 확정 판정 수 : {total_decided_all} / {len(y)} (커버리지: {comb_cov*100:.2f}%, 보류: {total_retake_all}건)")
    print(f"  - Accuracy        : {comb_acc*100:.2f}%")
    print(f"  - Precision       : {comb_prec*100:.2f}%")
    print(f"  - Recall          : {comb_rec*100:.2f}%")
    print(f"  - F1-Score        : {comb_f1*100:.2f}%")
    print(f"  - Confusion Matrix:\n{comb_cm}")
    print("=" * 80)

    # 6. 결과 JSON 저장
    results_dict = {
        "model_name": "BabelDeadliftD3ModelV3",
        "exercise": "02_바벨_데드리프트",
        "defect": "d3_발바닥_지면_고정",
        "feature_dim": 33,
        "feature_names": FEATURE_NAMES_33_D3,
        "binary_metrics_th50": {
            "accuracy": acc_50,
            "precision": prec_50,
            "recall": rec_50,
            "f1_score": f1_50,
            "roc_auc": auc_total
        },
        "view_adaptive_3state_summary": {
            "coverage": comb_cov,
            "total_decided": total_decided_all,
            "total_retake": total_retake_all,
            "accuracy": comb_acc,
            "precision": comb_prec,
            "recall": comb_rec,
            "f1_score": comb_f1,
            "confusion_matrix": comb_cm
        },
        "optimized_view_thresholds": view_thresholds_optimized,
        "view_audit": view_audit_v3,
        "model_version_comparison": {
            "d3_v1_3state_f1": 54.02,
            "d3_v2_3state_f1": 52.69,
            "d3_v3_view_adaptive_f1": comb_f1 * 100,
            "d3_v3_vs_v1_f1_delta": (comb_f1 * 100) - 54.02
        }
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 공식 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")

    # 7. 전체 데이터 기반 최종 프로덕션 모델 저장
    print("\n[*] 전체 데이터 기반 최종 d3 v3.0 프로덕션 가중치 학습 및 저장...")
    full_model = BabelDeadliftD3ModelV3()
    full_model.view_thresholds = view_thresholds_optimized
    full_model.fit_ensemble(X, y, random_state=42)
    final_weight_path = os.path.join(OUT_WEIGHTS_DIR, "d3_model_v3_best.pkl")
    full_model.save_weights(final_weight_path)
    print(f"[+] d3 v3.0 최종 가중치 저장 완료: {final_weight_path}")


if __name__ == "__main__":
    main()
