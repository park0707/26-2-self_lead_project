# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: train_and_eval_d5_v5_moe.py
Purpose: 45D 외부 AI 자문 반영 피처셋 + View-Aware Soft MoE 모델에 대한
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

from src.models.second_models.babel_deadlift.d5_model_v5_moe import (
    BabelDeadliftD5ModelV5MoE,
    extract_d5_v5_moe_features_from_seq,
    FEATURE_NAMES_45_D5_MOE
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d5_model_v5_moe_evaluation_results.json")
MOE_FEATURE_CACHE = os.path.join(CURRENT_DIR, "d5_features_45_moe_cache.npz")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if os.path.exists(MOE_FEATURE_CACHE):
        print(f"[*] 기존 45D MoE 피처 캐시 로드: {MOE_FEATURE_CACHE}", flush=True)
        cached = np.load(MOE_FEATURE_CACHE, allow_pickle=True)
        return cached['X'], cached['y'], cached['groups'], cached['views'], list(cached['clip_names'])

    print("[*] 45D Soft-MoE 피처 추출 시작 (770개 전수)...", flush=True)
    cache_v3_path = os.path.join(CURRENT_DIR, "d5_features_32_cache.npz")
    d32 = np.load(cache_v3_path, allow_pickle=True)
    y = d32['y'].astype(np.int32)
    groups = d32['groups'].astype(np.int32)
    views = d32['views']
    clip_names = list(d32['clip_names'])
    N = len(y)

    feats_list = []
    t0 = time.time()
    for idx in range(N):
        cn = clip_names[idx]
        vn = views[idx]
        npz_p = os.path.join(CACHE_DIR, f"{cn}_{vn}.npz")
        with np.load(npz_p) as d_npz:
            kps = d_npz["kps_33_3d_64f"]
        f45 = extract_d5_v5_moe_features_from_seq(kps, view_name=vn)
        feats_list.append(f45)

        if (idx + 1) % 200 == 0 or (idx + 1) == N:
            print(f"    -> 진행률: [{idx+1}/{N}] ({(idx+1)/N*100:.1f}%) | 경과: {time.time()-t0:.1f}초", flush=True)

    X = np.array(feats_list, dtype=np.float32)
    np.savez_compressed(MOE_FEATURE_CACHE, X=X, y=y, groups=groups, views=views, clip_names=clip_names)
    print(f"[+] 45D 피처 추출 및 캐시 저장 완료: {MOE_FEATURE_CACHE} ({X.shape})", flush=True)
    return X, y, groups, views, clip_names


def main():
    print("=" * 80)
    print(" [vidio-health-trainer] 바벨 데드리프트 d5 v5.0 Soft-MoE 학습 및 평가")
    print(" 45D 외부 AI 자문 킬러 피처 + View-Aware Soft MoE")
    print("=" * 80)

    X, y, groups, views, clip_names = load_dataset()
    print(f"[*] 총 분석 대상 시퀀스 수: {len(X)}개 (정상: {sum(y==0)}개, 결함: {sum(y==1)}개)")
    print(f"[*] 피처 차원: {X.shape[1]}D ({len(FEATURE_NAMES_45_D5_MOE)}개 피처 정합 확인)")

    # 5-Fold StratifiedGroupKFold on 9 Subjects
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    oof_preds = np.zeros(len(y), dtype=int)
    fold_metrics = []

    best_f1 = -1.0
    t_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        val_subs = [int(s) for s in sorted(list(set(groups[val_idx])))]
        print(f"\n--- [Fold {fold+1}/5] 검증 피험자: {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건) ---", flush=True)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = BabelDeadliftD5ModelV5MoE()
        model.fit(X_train, y_train)

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

        fold_weight_path = os.path.join(OUT_WEIGHTS_DIR, f"d5_v5_moe_fold{fold+1}.pkl")
        model.save(fold_weight_path)

        if f1 > best_f1:
            best_f1 = f1
            best_weight_path = os.path.join(OUT_WEIGHTS_DIR, "d5_model_v5_moe_best.pkl")
            model.save(best_weight_path)

    # 전체 OOF 지표
    total_acc = accuracy_score(y, oof_preds)
    total_prec = precision_score(y, oof_preds, zero_division=0)
    total_rec = recall_score(y, oof_preds, zero_division=0)
    total_f1 = f1_score(y, oof_preds, zero_division=0)
    total_f1_macro = f1_score(y, oof_preds, average='macro')
    total_auc = roc_auc_score(y, oof_probs)
    cm = confusion_matrix(y, oof_preds).tolist()

    print("\n" + "=" * 80)
    print(" [전체 770건 5-Fold OOF 통합 검증 결과 - v5.0 Soft-MoE]")
    print("=" * 80)
    print(f"[*] 기본 정확도(Accuracy) : {total_acc * 100:.2f}%")
    print(f"[*] 정밀도(Precision)     : {total_prec * 100:.2f}%")
    print(f"[*] 재현율(Recall)        : {total_rec * 100:.2f}%")
    print(f"[*] F1-Score (결함 클래스) : {total_f1 * 100:.2f}%")
    print(f"[*] F1-Score (Macro)      : {total_f1_macro * 100:.2f}%")
    print(f"[*] ROC-AUC               : {total_auc:.4f}")
    print(f"[*] 혼동 행렬(Confusion Matrix):\n  정상예측  결함예측\n[[ {cm[0][0]:6d}  {cm[0][1]:6d} ]  (실제 정상)\n [ {cm[1][0]:6d}  {cm[1][1]:6d} ]] (실제 결함)")

    # 공식 5대 시점 전수 성능 감사
    print("\n" + "=" * 80)
    print(" [공식 5대 시점(시점 A~E) 전수 다차원 성능 감사 - v5.0 Soft-MoE]")
    print("=" * 80)

    view_results = {}
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        alias = VIEW_ALIAS_MAP[vn]
        v_idx = np.where(views == vn)[0]
        y_v = y[v_idx]
        pred_v = oof_preds[v_idx]
        prob_v = oof_probs[v_idx]

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

        print(f"  {alias:<22} | Acc: {v_acc*100:5.2f}% | Prec: {v_prec*100:5.2f}% | Rec: {v_rec*100:5.2f}% | F1: {v_f1*100:5.2f}% | AUC: {v_auc:.4f}")

    # 3-State 신뢰도 기반 필터링 (불확실성 구간 배제) 성능 평가
    # low_th ~ high_th 사이를 재촬영/보류(Uncertain)로 판정할 때의 커버리지 및 F1
    print("\n" + "=" * 80)
    print(" [3-State 신뢰도 판정(Uncertainty Aware) 구간별 성능 감사]")
    print("=" * 80)
    threshold_grid = [
        (0.50, 0.50),
        (0.45, 0.55),
        (0.42, 0.54),
        (0.40, 0.53),
        (0.38, 0.55)
    ]
    three_state_results = []
    for l_th, h_th in threshold_grid:
        if l_th == h_th:
            preds_3s = (oof_probs >= l_th).astype(int)
            mask_decided = np.ones(len(y), dtype=bool)
        else:
            mask_decided = (oof_probs < l_th) | (oof_probs >= h_th)
            preds_3s = np.zeros(len(y), dtype=int)
            preds_3s[oof_probs >= h_th] = 1
            preds_3s[oof_probs < l_th] = 0

        cov = np.mean(mask_decided)
        y_dec = y[mask_decided]
        p_dec = preds_3s[mask_decided]

        acc_3s = accuracy_score(y_dec, p_dec)
        prec_3s = precision_score(y_dec, p_dec, zero_division=0)
        rec_3s = recall_score(y_dec, p_dec, zero_division=0)
        f1_3s = f1_score(y_dec, p_dec, zero_division=0)

        three_state_results.append({
            "threshold_band": [l_th, h_th],
            "coverage": float(cov),
            "decided_count": int(sum(mask_decided)),
            "uncertain_count": int(len(y) - sum(mask_decided)),
            "accuracy": float(acc_3s),
            "precision": float(prec_3s),
            "recall": float(rec_3s),
            "f1_score": float(f1_3s)
        })
        print(f"  밴드 [{l_th:.2f}, {h_th:.2f}] -> 커버리지: {cov*100:5.2f}% ({sum(mask_decided)}/{len(y)}) | F1: {f1_3s*100:5.2f}% | Acc: {acc_3s*100:5.2f}%")

    out_data = {
        "model_version": "v5.0 Soft-MoE",
        "description": "45D View-Aware Soft MoE (Gemini Y-axis rhythm, Ape Index dynamic margin, polygon area ratio, COM offset)",
        "features_count": int(X.shape[1]),
        "feature_names": FEATURE_NAMES_45_D5_MOE,
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
