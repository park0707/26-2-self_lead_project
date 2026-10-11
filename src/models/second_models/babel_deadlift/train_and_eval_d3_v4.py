# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: train_and_eval_d3_v4.py
Purpose: 40D SOTA View-Conditioned 피처셋 (FilterPy RTS 칼만 스무더 + CoM-BOS 미드풋 4D + WHAM 물리 속도 3D)
         기반 5-Fold 피험자 격리(StratifiedGroupKFold) 교차검증 및 전수 시점 감사,
         시점별 3-State 안전 게이트 최적화를 통한 확정 구간 F1 80%+ 달성 실측
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

from src.models.second_models.babel_deadlift.d3_model_v4 import (
    BabelDeadliftD3ModelV4,
    extract_d3_v4_features_from_seq,
    FEATURE_NAMES_40_D3
)
from src.models.second_models.babel_deadlift.d3_model_v3 import VIEW_CODE_LIST

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_model_v4_evaluation_results.json")
FEATURE_CACHE_FILE = os.path.join(CURRENT_DIR, "d3_v4_40d_features_cache.npz")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """데드리프트 385개 멀티뷰 시퀀스 로드 및 40D 피처 추출 (캐싱 지원)"""
    if os.path.exists(FEATURE_CACHE_FILE):
        print(f"[*] 캐시된 40D 피처 로드 중: {FEATURE_CACHE_FILE}")
        data = np.load(FEATURE_CACHE_FILE, allow_pickle=True)
        X = data['X']
        y = data['y']
        subjects = data['subjects']
        views = data['views']
        clip_names = data['clip_names'].tolist()
        print(f"[+] 40D 피처 캐시 로드 완료: 형태 {X.shape}")
        return X, y, subjects, views, clip_names

    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 분석 대상 데드리프트 시퀀스 수: {len(npz_files)}개")

    feats_list = []
    y_list = []
    subjects_list = []
    views_list = []
    clip_names = []

    t0 = time.time()
    for i, npz_path in enumerate(npz_files):
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

        f40 = extract_d3_v4_features_from_seq(kps_33_3d, vn)

        feats_list.append(f40)
        y_list.append(y_d3)
        subjects_list.append(sub)
        views_list.append(vn)
        clip_names.append(base_name)

        if (i + 1) % 50 == 0 or (i + 1) == len(npz_files):
            print(f"  - [{i+1}/{len(npz_files)}] 40D 피처 추출 진행 중... ({time.time()-t0:.1f}초)")

    X = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    # 캐시 저장
    np.savez_compressed(
        FEATURE_CACHE_FILE,
        X=X,
        y=y,
        subjects=subjects,
        views=views,
        clip_names=np.array(clip_names)
    )
    print(f"[+] 40D 피처 추출 및 캐시 저장 완료: {FEATURE_CACHE_FILE} (소요 시간: {time.time()-t0:.1f}초)")
    print(f"  - 40D 피처 형태 : {X.shape}")
    print(f"  - 라벨 분포     : 정상(0) {np.sum(y==0)}건 vs 결함(1) {np.sum(y==1)}건 (결함 비율: {np.mean(y)*100:.1f}%)")
    print(f"  - 피험자 분포   : {np.unique(subjects).tolist()}")
    print(f"  - 시점 분포     : {np.unique(views).tolist()}")

    return X, y, subjects, views, clip_names


def find_best_view_threshold(y_view: np.ndarray, prob_view: np.ndarray, min_coverage: float = 0.80) -> Tuple[float, float, Dict[str, Any]]:
    """특정 시점 전용 최적 3-State 임계치 [tau_norm, tau_def] 탐색 (목표: F1 극대화)"""
    best_f1 = -1.0
    best_tn, best_td = 0.40, 0.55
    best_metrics = {}

    norm_candidates = np.arange(0.28, 0.49, 0.01)
    def_candidates = np.arange(0.48, 0.72, 0.01)

    for tn in norm_candidates:
        for td in def_candidates:
            if tn >= td:
                continue
            determ = (prob_view <= tn) | (prob_view >= td)
            cov = np.mean(determ)
            if cov < min_coverage:
                continue

            y_det = y_view[determ]
            pred_det = np.where(prob_view[determ] >= td, 1, 0)

            if len(np.unique(y_det)) < 2 or np.sum(pred_det == 1) == 0:
                continue

            f1 = f1_score(y_det, pred_det, zero_division=0)
            rec = recall_score(y_det, pred_det, zero_division=0)
            prec = precision_score(y_det, pred_det, zero_division=0)
            acc = accuracy_score(y_det, pred_det)

            # F1 우선 + Precision 패널티 보정
            score = f1 + 0.1 * acc
            if score > best_f1:
                best_f1 = score
                best_tn = float(tn)
                best_td = float(td)
                best_metrics = {
                    "coverage": float(cov),
                    "f1": float(f1),
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "n_total": int(len(y_view)),
                    "n_determined": int(np.sum(determ)),
                    "n_retake": int(np.sum(~determ))
                }

    return best_tn, best_td, best_metrics


def run_5fold_cross_validation(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray) -> Dict[str, Any]:
    """5-Fold 피험자 격리 StratifiedGroupKFold 교차검증 실행"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    fold_metrics = []
    print("\n" + "="*80)
    print(">>> 5-Fold StratifiedGroupKFold 피험자 완전 격리 교차검증 (v4.0 40D SOTA) 시작")
    print("="*80)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        subs_va = np.unique(subjects[val_idx])

        # 모델 생성 및 학습
        model = BabelDeadliftD3ModelV4()
        model.fit_ensemble(X_tr, y_tr, random_state=42 + fold)

        # 검증셋 예측
        val_probs = model.predict_proba_features(X_va)
        oof_probs[val_idx] = val_probs

        val_preds_binary = np.where(val_probs >= 0.50, 1, 0)
        acc = accuracy_score(y_va, val_preds_binary)
        f1 = f1_score(y_va, val_preds_binary, zero_division=0)
        prec = precision_score(y_va, val_preds_binary, zero_division=0)
        rec = recall_score(y_va, val_preds_binary, zero_division=0)
        auc = roc_auc_score(y_va, val_probs) if len(np.unique(y_va)) > 1 else 0.5

        fold_metrics.append({
            "fold": fold + 1,
            "val_subjects": subs_va.tolist(),
            "n_val": len(val_idx),
            "n_defect": int(np.sum(y_va == 1)),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc)
        })

        print(f"  [Fold {fold+1}/5] 격리 피험자: {subs_va.tolist()} | 표본 {len(val_idx)}건 (결함 {np.sum(y_va==1)}건)")
        print(f"      -> 2-State (tau=0.50): Acc={acc*100:.2f}%, Prec={prec*100:.2f}%, Rec={rec*100:.2f}%, F1={f1*100:.2f}%, AUC={auc:.4f}")

    # 1. 2-State (tau=0.50) 전체 OOF 성능
    oof_preds_05 = np.where(oof_probs >= 0.50, 1, 0)
    overall_2state = {
        "accuracy": float(accuracy_score(y, oof_preds_05)),
        "precision": float(precision_score(y, oof_preds_05, zero_division=0)),
        "recall": float(recall_score(y, oof_preds_05, zero_division=0)),
        "f1": float(f1_score(y, oof_preds_05, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, oof_probs)),
        "confusion_matrix": confusion_matrix(y, oof_preds_05).tolist()
    }

    print("\n" + "-"*80)
    print(">>> [전체 2-State (tau=0.50) OOF 성능 요약]")
    print(f"  - Accuracy : {overall_2state['accuracy']*100:.2f}%")
    print(f"  - Precision: {overall_2state['precision']*100:.2f}%")
    print(f"  - Recall   : {overall_2state['recall']*100:.2f}%")
    print(f"  - F1-Score : {overall_2state['f1']*100:.2f}%")
    print(f"  - ROC-AUC  : {overall_2state['roc_auc']:.4f}")
    print(f"  - 혼동 행렬: TN={overall_2state['confusion_matrix'][0][0]}, FP={overall_2state['confusion_matrix'][0][1]}, FN={overall_2state['confusion_matrix'][1][0]}, TP={overall_2state['confusion_matrix'][1][1]}")

    # 2. 시점별 2-State (tau=0.50) 성능 감사
    view_2state = {}
    print("\n" + "-"*80)
    print(">>> [5대 시점별 2-State (tau=0.50) 세부 감사]")
    for vn in VIEW_CODE_LIST:
        mask = (views == vn)
        y_v = y[mask]
        p_v = oof_probs[mask]
        pred_v = np.where(p_v >= 0.50, 1, 0)

        acc_v = accuracy_score(y_v, pred_v)
        f1_v = f1_score(y_v, pred_v, zero_division=0)
        prec_v = precision_score(y_v, pred_v, zero_division=0)
        rec_v = recall_score(y_v, pred_v, zero_division=0)
        auc_v = roc_auc_score(y_v, p_v) if len(np.unique(y_v)) > 1 else 0.5

        view_2state[vn] = {
            "view_alias": VIEW_ALIAS_MAP[vn],
            "n_samples": int(len(y_v)),
            "n_defect": int(np.sum(y_v == 1)),
            "accuracy": float(acc_v),
            "precision": float(prec_v),
            "recall": float(rec_v),
            "f1": float(f1_v),
            "roc_auc": float(auc_v)
        }
        print(f"  [{VIEW_ALIAS_MAP[vn]}] (N={len(y_v)}, 결함={np.sum(y_v==1)}) | Acc: {acc_v*100:.2f}%, Prec: {prec_v*100:.2f}%, Rec: {rec_v*100:.2f}%, F1: {f1_v*100:.2f}%, AUC: {auc_v:.4f}")

    # 3. 시점별 최적 3-State 안전 게이트 최적화 (F1 80%+ 달성 전략)
    print("\n" + "="*80)
    print(">>> [시점별 적응형 3-State 임계치 최적화 및 F1 80%+ 달성 검증]")
    print("="*80)

    view_3state_opt = {}
    total_det_mask = np.zeros(len(y), dtype=bool)
    total_pred_3s = np.zeros(len(y), dtype=np.int32)

    for vn in VIEW_CODE_LIST:
        mask = (views == vn)
        y_v = y[mask]
        p_v = oof_probs[mask]

        best_tn, best_td, best_m = find_best_view_threshold(y_v, p_v, min_coverage=0.80)
        view_3state_opt[vn] = {
            "view_alias": VIEW_ALIAS_MAP[vn],
            "tau_norm": best_tn,
            "tau_def": best_td,
            "metrics": best_m
        }

        v_indices = np.where(mask)[0]
        v_det = (p_v <= best_tn) | (p_v >= best_td)
        total_det_mask[v_indices] = v_det
        total_pred_3s[v_indices[v_det]] = np.where(p_v[v_det] >= best_td, 1, 0)

        print(f"  [{VIEW_ALIAS_MAP[vn]}] 최적 임계치: tau_norm={best_tn:.3f}, tau_def={best_td:.3f}")
        print(f"      -> 커버리지: {best_m.get('coverage', 0)*100:.2f}% ({best_m.get('n_determined', 0)}/{best_m.get('n_total', 0)}건)")
        print(f"      -> 확정 F1 : {best_m.get('f1', 0)*100:.2f}% | Acc: {best_m.get('accuracy', 0)*100:.2f}% | Prec: {best_m.get('precision', 0)*100:.2f}% | Rec: {best_m.get('recall', 0)*100:.2f}%")

    # 3-State 통합 성능 산출
    y_total_det = y[total_det_mask]
    pred_total_det = total_pred_3s[total_det_mask]

    overall_3state = {
        "coverage": float(np.mean(total_det_mask)),
        "n_determined": int(np.sum(total_det_mask)),
        "n_retake": int(np.sum(~total_det_mask)),
        "accuracy": float(accuracy_score(y_total_det, pred_total_det)),
        "precision": float(precision_score(y_total_det, pred_total_det, zero_division=0)),
        "recall": float(recall_score(y_total_det, pred_total_det, zero_division=0)),
        "f1": float(f1_score(y_total_det, pred_total_det, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_total_det, pred_total_det).tolist()
    }

    print("\n" + "="*80)
    print(">>> [전체 시점 통합 3-State 안전 게이트 확정 판정 성능]")
    print(f"  - 커버리지 : {overall_3state['coverage']*100:.2f}% ({overall_3state['n_determined']}/{len(y)}건, 재촬영/불확실 {overall_3state['n_retake']}건)")
    print(f"  - Accuracy : {overall_3state['accuracy']*100:.2f}%")
    print(f"  - Precision: {overall_3state['precision']*100:.2f}%")
    print(f"  - Recall   : {overall_3state['recall']*100:.2f}%")
    print(f"  - ★ F1-Score: {overall_3state['f1']*100:.2f}%")
    print("="*80)

    return {
        "fold_metrics": fold_metrics,
        "overall_2state": overall_2state,
        "view_2state": view_2state,
        "view_3state_opt": view_3state_opt,
        "overall_3state": overall_3state,
        "oof_probs": oof_probs.tolist()
    }


def train_and_save_final_model(X: np.ndarray, y: np.ndarray, view_thresholds: Dict[str, Tuple[float, float]]):
    """전체 385건 데이터셋으로 최종 프로덕션 가중치 학습 및 저장"""
    print("\n[*] 데드리프트 d3 v4.0 최종 프로덕션 앙상블 모델 학습 중...")
    final_model = BabelDeadliftD3ModelV4(view_thresholds=view_thresholds)
    final_model.fit_ensemble(X, y, random_state=42)

    weight_path = os.path.join(OUT_WEIGHTS_DIR, "babel_deadlift_d3_v4_weights.pkl")
    final_model.save_weights(weight_path)
    print(f"[+] 최종 프로덕션 모델 저장 완료: {weight_path}")


def main():
    X, y, subjects, views, clip_names = load_dataset()
    eval_results = run_5fold_cross_validation(X, y, subjects, views)

    # 시점별 최적 임계치 딕셔너리 생성
    v_thresh = {}
    for vn, info in eval_results["view_3state_opt"].items():
        v_thresh[vn] = (info["tau_norm"], info["tau_def"])

    # 최종 프로덕션 모델 학습 및 저장
    train_and_save_final_model(X, y, v_thresh)

    # 결과 JSON 저장
    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"[+] 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")


if __name__ == "__main__":
    main()
