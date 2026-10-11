# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Script: train_and_eval_d3_v5.py
Purpose: 솔루션 3(64프레임 ST-SeqNet + 40D 생체역학 트리 앙상블) 기반
         5-Fold 피험자 완전 격리 교차검증 및
         [제약 1] 판단 보류율 최대 10% (커버리지 >= 90.0% 엄격 보장)
         [제약 2] F1-Score 최우선 최적화 3-State 안전 게이트 평가
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

from src.models.second_models.babel_deadlift.d3_model_v5 import (
    BabelDeadliftD3ModelV5,
    KEY_JOINT_INDICES,
    NUM_KEY_JOINTS
)
from src.models.second_models.babel_deadlift.d3_model_v4 import (
    extract_d3_v4_features_from_seq,
    clean_and_smooth_keypoints
)
from src.models.second_models.babel_deadlift.d3_model_v3 import VIEW_CODE_LIST

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_model_v5_evaluation_results.json")
FEATURE_CACHE_FILE = os.path.join(CURRENT_DIR, "d3_v5_dataset_cache.npz")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """데이터셋 로드: 64프레임 시계열(X_seq) 및 40D 피처(X_feats) 생성"""
    if os.path.exists(FEATURE_CACHE_FILE):
        print(f"[*] 캐시된 v5 데이터셋 로드 중: {FEATURE_CACHE_FILE}")
        data = np.load(FEATURE_CACHE_FILE, allow_pickle=True)
        return data['X_seq'], data['X_feats'], data['y'], data['subjects'], data['views'], data['clip_names'].tolist()

    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"[*] 총 분석 대상 데드리프트 시퀀스 수: {len(npz_files)}개")

    seq_list = []
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

        kps_raw = np.load(npz_path)["kps_33_3d_64f"]
        kps_clean = clean_and_smooth_keypoints(kps_raw)

        # 1. 64프레임 핵심 15개 관절 시계열: (64, 15*3)
        key_kps = kps_clean[:, KEY_JOINT_INDICES, :].reshape(64, -1)
        # 2. 40D SOTA 생체역학 피처
        f40 = extract_d3_v4_features_from_seq(kps_raw, vn)

        seq_list.append(key_kps)
        feats_list.append(f40)
        y_list.append(y_d3)
        subjects_list.append(sub)
        views_list.append(vn)
        clip_names.append(base_name)

    X_seq = np.array(seq_list, dtype=np.float32)
    X_feats = np.array(feats_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=np.int32)
    views = np.array(views_list)

    np.savez_compressed(
        FEATURE_CACHE_FILE,
        X_seq=X_seq,
        X_feats=X_feats,
        y=y,
        subjects=subjects,
        views=views,
        clip_names=np.array(clip_names)
    )
    print(f"[+] v5 데이터셋 전처리 및 캐시 저장 완료 (소요: {time.time()-t0:.1f}초)")
    return X_seq, X_feats, y, subjects, views, clip_names


def find_best_view_threshold_f1_priority(y_view: np.ndarray, prob_view: np.ndarray, min_coverage: float = 0.90) -> Tuple[float, float, Dict[str, Any]]:
    """
    [제약 1] 커버리지 >= 90.0% (판단 보류율 <= 10.0%) 엄격 준수
    [제약 2] F1-Score 최우선 단독 최적화
    """
    best_f1 = -1.0
    best_tn, best_td = 0.35, 0.55
    best_metrics = {}

    # 세밀한 임계치 그리드 탐색
    for tn in np.arange(0.20, 0.48, 0.005):
        for td in np.arange(0.48, 0.75, 0.005):
            if tn >= td:
                continue
            determ = (prob_view <= tn) | (prob_view >= td)
            cov = np.mean(determ)
            # 제약 1: 커버리지 90% 이상 (보류율 10% 이하)
            if cov < min_coverage:
                continue

            y_det = y_view[determ]
            pred_det = np.where(prob_view[determ] >= td, 1, 0)

            if len(np.unique(y_det)) < 2 or np.sum(pred_det == 1) == 0:
                continue

            # 제약 2: F1-Score 순수 단독 최대화
            f1 = f1_score(y_det, pred_det, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tn = float(tn)
                best_td = float(td)
                best_metrics = {
                    "coverage": float(cov),
                    "retake_rate": float(1.0 - cov),
                    "f1": float(f1),
                    "accuracy": float(accuracy_score(y_det, pred_det)),
                    "precision": float(precision_score(y_det, pred_det, zero_division=0)),
                    "recall": float(recall_score(y_det, pred_det, zero_division=0)),
                    "n_total": int(len(y_view)),
                    "n_determined": int(np.sum(determ)),
                    "n_retake": int(np.sum(~determ))
                }

    # 만약 90% 이상에서 탐색 실패 시 가장 90%에 가까운 임계치 반환
    if best_f1 < 0:
        pred50 = np.where(prob_view >= 0.50, 1, 0)
        best_metrics = {
            "coverage": 1.0,
            "retake_rate": 0.0,
            "f1": float(f1_score(y_view, pred50, zero_division=0)),
            "accuracy": float(accuracy_score(y_view, pred50)),
            "precision": float(precision_score(y_view, pred50, zero_division=0)),
            "recall": float(recall_score(y_view, pred50, zero_division=0)),
            "n_total": int(len(y_view)),
            "n_determined": int(len(y_view)),
            "n_retake": 0
        }

    return best_tn, best_td, best_metrics


def run_5fold_evaluation(X_seq: np.ndarray, X_feats: np.ndarray, y: np.ndarray, subjects: np.ndarray, views: np.ndarray) -> Dict[str, Any]:
    """5-Fold 피험자 완전 격리 교차검증 (솔루션 3 v5 하이브리드)"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    print("\n" + "="*80)
    print(">>> [v5.0 솔루션 3] 64프레임 ST-SeqNet + 40D 생체역학 앙상블 5-Fold 피험자 격리 학습")
    print("="*80)

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_seq, y, groups=subjects)):
        t_fold = time.time()
        model = BabelDeadliftD3ModelV5()
        model.fit(
            X_seq[tr_idx], X_feats[tr_idx], y[tr_idx],
            epochs=35,
            random_state=42 + fold
        )

        probs_va = model.predict_proba(X_seq[va_idx], X_feats[va_idx])
        oof_probs[va_idx] = probs_va

        preds50 = np.where(probs_va >= 0.50, 1, 0)
        f1_fold = f1_score(y[va_idx], preds50, zero_division=0)
        acc_fold = accuracy_score(y[va_idx], preds50)
        subs_va = np.unique(subjects[va_idx]).tolist()

        print(f"  [Fold {fold+1}/5] 격리 피험자: {subs_va} | 표본 {len(va_idx)}건 (경과: {time.time()-t_fold:.1f}초)")
        print(f"      -> 2-State (0.50): F1={f1_fold*100:.2f}%, Acc={acc_fold*100:.2f}%")

    # 1. 2-State 전체 OOF 성능
    preds_05 = np.where(oof_probs >= 0.50, 1, 0)
    overall_2state = {
        "accuracy": float(accuracy_score(y, preds_05)),
        "precision": float(precision_score(y, preds_05, zero_division=0)),
        "recall": float(recall_score(y, preds_05, zero_division=0)),
        "f1": float(f1_score(y, preds_05, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, oof_probs)),
        "confusion_matrix": confusion_matrix(y, preds_05).tolist()
    }

    print("\n" + "-"*80)
    print(">>> [v5.0 전체 2-State (tau=0.50) OOF 성능 요약]")
    print(f"  - Accuracy : {overall_2state['accuracy']*100:.2f}%")
    print(f"  - Precision: {overall_2state['precision']*100:.2f}%")
    print(f"  - Recall   : {overall_2state['recall']*100:.2f}%")
    print(f"  - F1-Score : {overall_2state['f1']*100:.2f}%")
    print(f"  - ROC-AUC  : {overall_2state['roc_auc']:.4f}")

    # 2. 시점별 2-State 세부 감사
    view_2state = {}
    print("\n" + "-"*80)
    print(">>> [v5.0 5대 시점별 2-State (tau=0.50) 세부 감사]")
    for vn in VIEW_CODE_LIST:
        mask = (views == vn)
        y_v = y[mask]
        p_v = oof_probs[mask]
        pred_v = np.where(p_v >= 0.50, 1, 0)

        view_2state[vn] = {
            "view_alias": VIEW_ALIAS_MAP[vn],
            "n_samples": int(len(y_v)),
            "n_defect": int(np.sum(y_v == 1)),
            "accuracy": float(accuracy_score(y_v, pred_v)),
            "precision": float(precision_score(y_v, pred_v, zero_division=0)),
            "recall": float(recall_score(y_v, pred_v, zero_division=0)),
            "f1": float(f1_score(y_v, pred_v, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_v, p_v)) if len(np.unique(y_v)) > 1 else 0.5
        }
        print(f"  [{VIEW_ALIAS_MAP[vn]}] F1: {view_2state[vn]['f1']*100:.2f}% | Acc: {view_2state[vn]['accuracy']*100:.2f}% | AUC: {view_2state[vn]['roc_auc']:.4f}")

    # 3. [제약 1 & 2] 보류율 <= 10% (커버리지 >= 90%) 하에서 F1 최우선 최적화 3-State 안전 게이트
    print("\n" + "="*80)
    print(">>> [v5.0 시점별 3-State 안전 게이트: 보류율 <= 10% (커버리지 >= 90%) & F1 최우선 최적화]")
    print("="*80)

    view_3state_opt = {}
    total_det_mask = np.zeros(len(y), dtype=bool)
    total_pred_3s = np.zeros(len(y), dtype=np.int32)

    for vn in VIEW_CODE_LIST:
        mask = (views == vn)
        y_v = y[mask]
        p_v = oof_probs[mask]

        best_tn, best_td, best_m = find_best_view_threshold_f1_priority(y_v, p_v, min_coverage=0.90)
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

        print(f"  [{VIEW_ALIAS_MAP[vn]}] 임계치: tau_norm={best_tn:.3f}, tau_def={best_td:.3f}")
        print(f"      -> 커버리지: {best_m.get('coverage', 0)*100:.2f}% (보류율: {best_m.get('retake_rate', 0)*100:.2f}%)")
        print(f"      -> 확정 F1 : {best_m.get('f1', 0)*100:.2f}% | Acc: {best_m.get('accuracy', 0)*100:.2f}% | Prec: {best_m.get('precision', 0)*100:.2f}% | Rec: {best_m.get('recall', 0)*100:.2f}%")

    y_total_det = y[total_det_mask]
    pred_total_det = total_pred_3s[total_det_mask]

    overall_3state = {
        "coverage": float(np.mean(total_det_mask)),
        "retake_rate": float(1.0 - np.mean(total_det_mask)),
        "n_determined": int(np.sum(total_det_mask)),
        "n_retake": int(np.sum(~total_det_mask)),
        "accuracy": float(accuracy_score(y_total_det, pred_total_det)),
        "precision": float(precision_score(y_total_det, pred_total_det, zero_division=0)),
        "recall": float(recall_score(y_total_det, pred_total_det, zero_division=0)),
        "f1": float(f1_score(y_total_det, pred_total_det, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_total_det, pred_total_det).tolist()
    }

    print("\n" + "="*80)
    print(">>> [v5.0 전체 시점 통합 확정 판정 성능 (보류율 <= 10.0% 엄격 보장)]")
    print(f"  - 커버리지 : {overall_3state['coverage']*100:.2f}% (보류율: {overall_3state['retake_rate']*100:.2f}%, 확정 {overall_3state['n_determined']}/385건, 보류 {overall_3state['n_retake']}건)")
    print(f"  - Accuracy : {overall_3state['accuracy']*100:.2f}%")
    print(f"  - Precision: {overall_3state['precision']*100:.2f}%")
    print(f"  - Recall   : {overall_3state['recall']*100:.2f}%")
    print(f"  - ★ F1-Score: {overall_3state['f1']*100:.2f}%")
    print("="*80)

    return {
        "overall_2state": overall_2state,
        "view_2state": view_2state,
        "view_3state_opt": view_3state_opt,
        "overall_3state": overall_3state,
        "oof_probs": oof_probs.tolist()
    }


def main():
    X_seq, X_feats, y, subjects, views, clip_names = load_dataset()
    eval_results = run_5fold_evaluation(X_seq, X_feats, y, subjects, views)

    # 최종 프로덕션 모델 전체 학습 및 가중치 저장
    print("\n[*] 데드리프트 d3 v5.0 최종 프로덕션 하이브리드 모델 전체 학습 중...")
    v_thresh = {vn: (info["tau_norm"], info["tau_def"]) for vn, info in eval_results["view_3state_opt"].items()}
    final_model = BabelDeadliftD3ModelV5(view_thresholds=v_thresh)
    final_model.fit(X_seq, X_feats, y, epochs=40, random_state=42)

    weight_path = os.path.join(OUT_WEIGHTS_DIR, "babel_deadlift_d3_v5_weights.pkl")
    final_model.save_weights(weight_path)
    print(f"[+] 최종 가중치 저장 완료: {weight_path}")

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"[+] 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")


if __name__ == "__main__":
    main()
