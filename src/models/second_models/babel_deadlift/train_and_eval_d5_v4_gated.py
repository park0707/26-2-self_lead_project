# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: train_and_eval_d5_v4_gated.py
Purpose: 40D 시점 적응형 게이팅(View-Adaptive Gating, 방식 1) 모델에 대한
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

from src.models.second_models.babel_deadlift.d5_model_v4_gated import (
    BabelDeadliftD5ModelV4Gated,
    extract_d5_v4_gated_features_from_seq,
    FEATURE_NAMES_40_D5_GATED
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d5_model_v4_gated_evaluation_results.json")
GATED_FEATURE_CACHE = os.path.join(CURRENT_DIR, "d5_features_40_gated_cache.npz")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if os.path.exists(GATED_FEATURE_CACHE):
        print(f"[*] 기존 40D 게이팅 피처 캐시 로드: {GATED_FEATURE_CACHE}", flush=True)
        cached = np.load(GATED_FEATURE_CACHE, allow_pickle=True)
        return cached['X'], cached['y'], cached['groups'], cached['views'], list(cached['clip_names'])

    print("[*] 40D 시점 적응형 게이팅 피처 추출 시작...", flush=True)
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
        f40 = extract_d5_v4_gated_features_from_seq(kps, view_name=vn)
        feats_list.append(f40)

        if (idx + 1) % 200 == 0 or (idx + 1) == N:
            print(f"    -> 진행률: [{idx+1}/{N}] ({(idx+1)/N*100:.1f}%) | 경과: {time.time()-t0:.1f}초", flush=True)

    X = np.array(feats_list, dtype=np.float32)
    np.savez_compressed(GATED_FEATURE_CACHE, X=X, y=y, groups=groups, views=views, clip_names=clip_names)
    print(f"[+] 40D 피처 추출 및 캐시 저장 완료: {GATED_FEATURE_CACHE} ({X.shape})", flush=True)
    return X, y, groups, views, clip_names


def main():
    print("=" * 80)
    print(" [vidio-health-trainer] 바벨 데드리프트 d5 v4.0 Gated (시점 적응형 게이팅) 학습 및 평가")
    print("=" * 80)

    X, y, groups, views, clip_names = load_dataset()
    print(f"[*] 총 분석 대상 시퀀스 수: {len(X)}개 (정상: {sum(y==0)}개, 결함: {sum(y==1)}개)")
    print(f"[*] 피처 차원: {X.shape[1]}D ({len(FEATURE_NAMES_40_D5_GATED)}개 피처 정합 확인)")

    # 5-Fold StratifiedGroupKFold on 9 Subjects
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    oof_preds = np.zeros(len(y), dtype=int)
    fold_metrics = []
    feature_importances = np.zeros(X.shape[1], dtype=np.float32)

    best_f1 = -1.0
    t_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        val_subs = [int(s) for s in sorted(list(set(groups[val_idx])))]
        print(f"\n--- [Fold {fold+1}/5] 검증 피험자: {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건) ---", flush=True)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = BabelDeadliftD5ModelV4Gated()
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

        if model.lgb_model is not None:
            fi = model.lgb_model.feature_importances_
            feature_importances += (fi / (fi.sum() + 1e-6)) / 5.0

        fold_weight_path = os.path.join(OUT_WEIGHTS_DIR, f"d5_v4_gated_fold{fold+1}.pkl")
        model.save(fold_weight_path)

        if f1 > best_f1:
            best_f1 = f1
            best_weight_path = os.path.join(OUT_WEIGHTS_DIR, "d5_model_v4_gated_best.pkl")
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
    print(" [전체 770건 5-Fold OOF 통합 검증 결과 - v4.0 Gated]")
    print("=" * 80)
    print(f"[*] 기본 정확도(Accuracy) : {total_acc * 100:.2f}%")
    print(f"[*] 정밀도(Precision)     : {total_prec * 100:.2f}%")
    print(f"[*] 재현율(Recall)        : {total_rec * 100:.2f}%")
    print(f"[*] F1-Score (결함 클래스) : {total_f1 * 100:.2f}%")
    print(f"[*] F1-Score (Macro)      : {total_f1_macro * 100:.2f}%")
    print(f"[*] ROC-AUC               : {total_auc:.4f}")
    print(f"[*] 혼동 행렬(Confusion Matrix):\n  정상예측  결함예측\n[[ {cm[0][0]:6d}  {cm[0][1]:6d} ]  (실제 정상)\n [ {cm[1][0]:6d}  {cm[1][1]:6d} ]] (실제 결함)")

    # 3-State 파레토 최적화 탐색 [0.405, 0.525] 및 커버리지별 지표
    print("\n" + "=" * 80)
    print(" [3-State 불확실성 기권(Selective Abstention) 파레토 구간 분석]")
    print("=" * 80)
    pareto_results = []
    targets = [0.90, 0.88, 0.85, 0.80, 0.75]

    for tgt_cov in targets:
        best_p_f1 = -1.0
        best_cfg = None

        for margin in np.arange(0.01, 0.20, 0.005):
            t_norm = 0.50 - margin
            t_def = 0.50 + margin

            unc_mask = (oof_probs >= t_norm) & (oof_probs <= t_def)
            act_cov = 1.0 - np.mean(unc_mask)
            if abs(act_cov - tgt_cov) <= 0.015:
                cert_mask = ~unc_mask
                p_f1 = f1_score(y[cert_mask], oof_preds[cert_mask], average='macro')
                p_acc = accuracy_score(y[cert_mask], oof_preds[cert_mask])
                p_prec = precision_score(y[cert_mask], oof_preds[cert_mask], zero_division=0)
                p_rec = recall_score(y[cert_mask], oof_preds[cert_mask], zero_division=0)

                if p_f1 > best_p_f1:
                    best_p_f1 = p_f1
                    best_cfg = {
                        "target_coverage": tgt_cov,
                        "actual_coverage": float(act_cov),
                        "tau_norm": float(t_norm),
                        "tau_def": float(t_def),
                        "retained_count": int(np.sum(cert_mask)),
                        "retake_count": int(np.sum(unc_mask)),
                        "accuracy": float(p_acc),
                        "precision": float(p_prec),
                        "recall": float(p_rec),
                        "f1_score": float(p_f1)
                    }

        if best_cfg is not None:
            pareto_results.append(best_cfg)
            print(f"[+] 목표 커버리지 {best_cfg['target_coverage']*100:.0f}% (실제 {best_cfg['actual_coverage']*100:.1f}%, 유지 {best_cfg['retained_count']}건 / 재촬영 {best_cfg['retake_count']}건):")
            print(f"    - 불확실 구간: [{best_cfg['tau_norm']:.3f}, {best_cfg['tau_def']:.3f}]")
            print(f"    - 확신 구간 F1: {best_cfg['f1_score']*100:.2f}% | Acc: {best_cfg['accuracy']*100:.2f}% | Prec: {best_cfg['precision']*100:.2f}% | Rec: {best_cfg['recall']*100:.2f}%")

    # 5대 공식 시점별 전수 성능 감사
    print("\n" + "=" * 80)
    print(" [공식 5대 시점 전수 성능 감사 (View Audit)]")
    print("=" * 80)
    view_audit = {}
    for vk in ["view5", "view2", "view3", "view4", "view1"]:
        v_mask = (views == vk)
        y_v = y[v_mask]
        p_v = oof_preds[v_mask]
        prob_v = oof_probs[v_mask]

        v_acc = accuracy_score(y_v, p_v)
        v_prec = precision_score(y_v, p_v, zero_division=0)
        v_rec = recall_score(y_v, p_v, zero_division=0)
        v_f1 = f1_score(y_v, p_v, zero_division=0)
        v_f1_macro = f1_score(y_v, p_v, average='macro')
        try:
            v_auc = roc_auc_score(y_v, prob_v)
        except Exception:
            v_auc = 0.5

        alias = VIEW_ALIAS_MAP[vk]
        view_audit[vk] = {
            "alias": alias,
            "sample_count": int(np.sum(v_mask)),
            "accuracy": float(v_acc),
            "precision": float(v_prec),
            "recall": float(v_rec),
            "f1_score": float(v_f1),
            "f1_macro": float(v_f1_macro),
            "roc_auc": float(v_auc)
        }
        print(f"[*] {alias:<24} ({np.sum(v_mask):3d}건): F1(Macro)={v_f1_macro*100:5.2f}% | F1(Defect)={v_f1*100:5.2f}% | Acc={v_acc*100:5.2f}% | AUC={v_auc:.4f}")

    # 피처 중요도 Top 15
    fi_series = pd.Series(feature_importances, index=FEATURE_NAMES_40_D5_GATED).sort_values(ascending=False)
    print("\n" + "=" * 80)
    print(" [피처 중요도 Top 15 (게이팅 피처 기여도 확인)]")
    print("=" * 80)
    for rank, (feat, score) in enumerate(fi_series.head(15).items(), 1):
        is_gated = "★ [게이팅 피처]" if "gated" in feat or "gate" in feat else ""
        print(f"  {rank:2d}위: {feat:<36} | {score*100:5.2f}% {is_gated}")

    # 결과 JSON 저장
    result_data = {
        "model_name": "d5_model_v4_gated.py",
        "feature_dim": int(X.shape[1]),
        "total_samples": len(X),
        "oof_metrics": {
            "accuracy": float(total_acc),
            "precision": float(total_prec),
            "recall": float(total_rec),
            "f1_score": float(total_f1),
            "f1_macro": float(total_f1_macro),
            "roc_auc": float(total_auc),
            "confusion_matrix": cm
        },
        "fold_metrics": fold_metrics,
        "view_audit": view_audit,
        "pareto_results": pareto_results,
        "feature_importance_top20": fi_series.head(20).to_dict()
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] v4.0 Gated 평가 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")


if __name__ == "__main__":
    main()
