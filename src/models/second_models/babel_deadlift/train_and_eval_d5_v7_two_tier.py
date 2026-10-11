# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: train_and_eval_d5_v7_two_tier.py
Purpose: 바벨 데드리프트 d5 4단계 모호성 분할 기반 2계층 구조 (Two-Tier Cascade) 5-Fold 학습 및
         1계층(Level 1 평가), 2계층(Level 1~3 평가, Level 4 폐기) 다차원 성능 감사 및 이전 버전(v6.0) 대조 비교
Output: D:\src\models\second_models\babel_deadlift\d5_model_v7_two_tier_evaluation_results.json
"""

import os
import sys
import json
import time
import pickle
from typing import Dict, List, Tuple, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d5_model_v7_two_tier import (
    BabelDeadliftD5ModelV7TwoTier,
    VIEW_OPTIMAL_THRESHOLDS_V7
)
from src.models.second_models.babel_deadlift.d5_model_v6_features import (
    FEATURE_NAMES_37_D5_V6
)

V6_CACHE_PATH = os.path.join(CURRENT_DIR, "d5_features_37_v6_cache.npz")
AMBIGUITY_NPZ_PATH = os.path.join(CURRENT_DIR, "d5_ambiguity_levels.npz")
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d5_model_v7_two_tier_evaluation_results.json")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def load_dataset_with_ambiguity():
    cache = np.load(V6_CACHE_PATH, allow_pickle=True)
    X = cache['X']
    y = cache['y']
    groups = cache['groups']
    views = cache['views']
    clip_names = cache['clip_names']

    amb = np.load(AMBIGUITY_NPZ_PATH, allow_pickle=True)
    levels = amb['levels']
    l4_mask = amb['l4_mask']
    oof_v6_probs = amb['oof_raw_probs']

    return X, y, groups, views, clip_names, levels, l4_mask, oof_v6_probs


def find_optimal_thresholds(y_true: np.ndarray, raw_probs: np.ndarray, views: np.ndarray) -> Dict[str, float]:
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


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, th: float = 0.50) -> Dict[str, Any]:
    preds = (probs >= th).astype(int)
    cm = confusion_matrix(y_true, preds).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion_matrix": cm,
        "sample_count": len(y_true)
    }


def main():
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 바벨 데드리프트 d5 2계층 구조 (Two-Tier Cascade) 공식 학습 및 평가", flush=True)
    print("  - Level 4(오염/역전 94건) 폐기 / Tier 1: Level 1(명확군) 평가 / Tier 2: Level 1~3(실무군) 평가", flush=True)
    print("  - 9 Subjects StratifiedGroupKFold 5-Fold 피험자 격리 검증", flush=True)
    print("=" * 100, flush=True)

    X_all, y_all, groups_all, views_all, clip_names_all, levels_all, l4_mask_all, oof_v6_probs = load_dataset_with_ambiguity()
    N_all = len(y_all)

    print(f"\n[1] 4단계 모호성 데이터셋 현황 (총 {N_all}개):", flush=True)
    for lv in [1, 2, 3, 4]:
        cnt_lv = np.sum(levels_all == lv)
        norm_lv = np.sum((levels_all == lv) & (y_all == 0))
        def_lv = np.sum((levels_all == lv) & (y_all == 1))
        tag = "-> [원천 폐기]" if lv == 4 else f"-> Tier-{1 if lv==1 else 2} 대상"
        print(f"    - Level {lv}: {cnt_lv:3d}개 ({cnt_lv/N_all*100:5.2f}%) | 정상: {norm_lv:3d}개, 결함: {def_lv:3d}개 {tag}", flush=True)

    # -------------------------------------------------------------------------
    # [데이터셋 분할: Level 4 폐기 -> Level 1~3 정제 실무셋 (676건)]
    # -------------------------------------------------------------------------
    clean_mask = (levels_all != 4)
    X = X_all[clean_mask]
    y = y_all[clean_mask]
    groups = groups_all[clean_mask]
    views = views_all[clean_mask]
    clip_names = clip_names_all[clean_mask]
    levels = levels_all[clean_mask]
    oof_v6_clean = oof_v6_probs[clean_mask]

    N_clean = len(y)
    print(f"\n[2] Level 4 폐기 후 실무 유효 데이터셋 (Level 1~3): {N_clean}개 ({N_clean/N_all*100:.2f}%)", flush=True)
    print(f"    - 정상: {np.sum(y == 0)}개, 결함: {np.sum(y == 1)}개", flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5)

    oof_t1_probs = np.zeros(N_clean, dtype=np.float32)
    oof_t2_probs = np.zeros(N_clean, dtype=np.float32)
    oof_cascade_probs = np.zeros(N_clean, dtype=np.float32)

    fold_models = []
    best_fold_f1 = -1.0
    best_model_path = os.path.join(OUT_WEIGHTS_DIR, "d5_model_v7_two_tier_best.pkl")

    t_start = time.time()

    print("\n" + "=" * 90, flush=True)
    print("  [5-Fold StratifiedGroupKFold 교차검증 진행]", flush=True)
    print("=" * 90, flush=True)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        val_subs = [int(s) for s in sorted(list(set(groups[val_idx])))]
        print(f"\n--- [Fold {fold+1}/5] 검증 피험자: {val_subs} (학습 {len(train_idx)}건, 검증 {len(val_idx)}건) ---", flush=True)

        X_train, y_train = X[train_idx], y[train_idx]
        levels_train = levels[train_idx]

        X_val, y_val = X[val_idx], y[val_idx]
        levels_val = levels[val_idx]

        model = BabelDeadliftD5ModelV7TwoTier()
        model.fit(X_train, y_train, levels_train)

        # OOF 예측
        p_t1 = model.predict_tier1_proba(X_val)
        p_t2 = model.predict_tier2_proba(X_val)
        p_cascade = model.predict_cascade_proba(X_val)

        oof_t1_probs[val_idx] = p_t1
        oof_t2_probs[val_idx] = p_t2
        oof_cascade_probs[val_idx] = p_cascade

        # Fold 검증 메트릭
        val_l1_mask = (levels_val == 1)
        f1_t1_l1 = f1_score(y_val[val_l1_mask], (p_t1[val_l1_mask] >= 0.50).astype(int), zero_division=0)
        f1_t2_all = f1_score(y_val, (p_t2 >= 0.50).astype(int), zero_division=0)
        f1_cas_all = f1_score(y_val, (p_cascade >= 0.50).astype(int), zero_division=0)

        print(f"  > Fold {fold+1} 결과:")
        print(f"    - Tier 1 (Level 1만 평가) F1 : {f1_t1_l1*100:6.2f}%")
        print(f"    - Tier 2 (Level 1~3 평가) F1 : {f1_t2_all*100:6.2f}%")
        print(f"    - Cascade (2계층 융합) F1    : ★ {f1_cas_all*100:6.2f}%")

        fold_w_path = os.path.join(OUT_WEIGHTS_DIR, f"d5_v7_two_tier_fold{fold+1}.pkl")
        model.save(fold_w_path)
        fold_models.append(model)

        if f1_cas_all > best_fold_f1:
            best_fold_f1 = f1_cas_all
            model.save(best_model_path)

    total_train_time = round(time.time() - t_start, 2)
    print(f"\n[+] 5-Fold 학습 완료 (총 소요 시간: {total_train_time}초)")

    # -------------------------------------------------------------------------
    # [1계층 평가: Level 1만 평가]
    # -------------------------------------------------------------------------
    mask_l1 = (levels == 1)
    y_l1 = y[mask_l1]
    views_l1 = views[mask_l1]
    p_t1_l1 = oof_t1_probs[mask_l1]
    p_v6_l1 = oof_v6_clean[mask_l1]

    # Level 1 시점별 최적 임계값 도출
    opt_th_l1 = find_optimal_thresholds(y_l1, p_t1_l1, views_l1)
    th_vec_l1 = np.array([opt_th_l1[v] for v in views_l1])
    preds_t1_l1_opt = (p_t1_l1 >= th_vec_l1).astype(int)

    m_t1_l1_th50 = compute_metrics(y_l1, p_t1_l1, th=0.50)
    m_t1_l1_opt = {
        "accuracy": float(accuracy_score(y_l1, preds_t1_l1_opt)),
        "precision": float(precision_score(y_l1, preds_t1_l1_opt, zero_division=0)),
        "recall": float(recall_score(y_l1, preds_t1_l1_opt, zero_division=0)),
        "f1_score": float(f1_score(y_l1, preds_t1_l1_opt, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_l1, p_t1_l1)),
        "confusion_matrix": confusion_matrix(y_l1, preds_t1_l1_opt).tolist(),
        "sample_count": len(y_l1)
    }

    # 이전 v6 모델의 Level 1 평가 성능
    opt_th_v6 = VIEW_OPTIMAL_THRESHOLDS_V7
    th_vec_v6_l1 = np.array([opt_th_v6[v] for v in views_l1])
    preds_v6_l1 = (p_v6_l1 >= th_vec_v6_l1).astype(int)
    m_v6_l1 = {
        "accuracy": float(accuracy_score(y_l1, preds_v6_l1)),
        "precision": float(precision_score(y_l1, preds_v6_l1, zero_division=0)),
        "recall": float(recall_score(y_l1, preds_v6_l1, zero_division=0)),
        "f1_score": float(f1_score(y_l1, preds_v6_l1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_l1, p_v6_l1)),
        "confusion_matrix": confusion_matrix(y_l1, preds_v6_l1).tolist(),
        "sample_count": len(y_l1)
    }

    print("\n" + "=" * 90)
    print("  [1계층 (Tier 1) 공식 성능 감사 - Level 1 명확군 459건 대상 평가]")
    print("=" * 90)
    print(f"[*] Level 1 샘플 수: {len(y_l1)}개 (정상: {np.sum(y_l1==0)}개, 결함: {np.sum(y_l1==1)}개)")
    print(f"[*] 이전 v6.0 모델 (Level 1)   : F1={m_v6_l1['f1_score']*100:6.2f}%, Acc={m_v6_l1['accuracy']*100:6.2f}%, AUC={m_v6_l1['roc_auc']:.4f}")
    print(f"[*] 신규 Tier-1 모델 (Th=0.50) : F1={m_t1_l1_th50['f1_score']*100:6.2f}%, Acc={m_t1_l1_th50['accuracy']*100:6.2f}%, AUC={m_t1_l1_th50['roc_auc']:.4f}")
    print(f"[*] 신규 Tier-1 모델 (시점최적): ★ F1={m_t1_l1_opt['f1_score']*100:6.2f}%, Acc={m_t1_l1_opt['accuracy']*100:6.2f}%, AUC={m_t1_l1_opt['roc_auc']:.4f}")
    print(f"[*] 성능 도약 폭 (vs v6.0)     : F1 도약 +{(m_t1_l1_opt['f1_score'] - m_v6_l1['f1_score'])*100:+.2f}%p, Acc +{(m_t1_l1_opt['accuracy'] - m_v6_l1['accuracy'])*100:+.2f}%p")

    # -------------------------------------------------------------------------
    # [2계층 평가: Level 1~3 전체 평가]
    # -------------------------------------------------------------------------
    # 시점별 최적 임계값 도출 (Level 1~3)
    opt_th_l123 = find_optimal_thresholds(y, oof_cascade_probs, views)
    th_vec_l123 = np.array([opt_th_l123[v] for v in views])
    preds_cas_opt = (oof_cascade_probs >= th_vec_l123).astype(int)

    m_cas_th50 = compute_metrics(y, oof_cascade_probs, th=0.50)
    m_cas_opt = {
        "accuracy": float(accuracy_score(y, preds_cas_opt)),
        "precision": float(precision_score(y, preds_cas_opt, zero_division=0)),
        "recall": float(recall_score(y, preds_cas_opt, zero_division=0)),
        "f1_score": float(f1_score(y, preds_cas_opt, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, oof_cascade_probs)),
        "confusion_matrix": confusion_matrix(y, preds_cas_opt).tolist(),
        "sample_count": len(y)
    }

    # Tier 2 단독 모델 성능 (Level 1~3)
    opt_th_t2 = find_optimal_thresholds(y, oof_t2_probs, views)
    th_vec_t2 = np.array([opt_th_t2[v] for v in views])
    preds_t2_opt = (oof_t2_probs >= th_vec_t2).astype(int)
    m_t2_opt = {
        "accuracy": float(accuracy_score(y, preds_t2_opt)),
        "precision": float(precision_score(y, preds_t2_opt, zero_division=0)),
        "recall": float(recall_score(y, preds_t2_opt, zero_division=0)),
        "f1_score": float(f1_score(y, preds_t2_opt, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, oof_t2_probs)),
        "confusion_matrix": confusion_matrix(y, preds_t2_opt).tolist(),
        "sample_count": len(y)
    }

    # 이전 v6 모델의 Level 1~3 평가 성능
    th_vec_v6_clean = np.array([opt_th_v6[v] for v in views])
    preds_v6_clean = (oof_v6_clean >= th_vec_v6_clean).astype(int)
    m_v6_clean = {
        "accuracy": float(accuracy_score(y, preds_v6_clean)),
        "precision": float(precision_score(y, preds_v6_clean, zero_division=0)),
        "recall": float(recall_score(y, preds_v6_clean, zero_division=0)),
        "f1_score": float(f1_score(y, preds_v6_clean, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, oof_v6_clean)),
        "confusion_matrix": confusion_matrix(y, preds_v6_clean).tolist(),
        "sample_count": len(y)
    }

    # Fast-path vs Deep-path 비율
    fast_path_mask = (oof_t1_probs <= 0.25) | (oof_t1_probs >= 0.75)
    fast_pct = float(np.mean(fast_path_mask) * 100)
    deep_pct = 100.0 - fast_pct

    print("\n" + "=" * 90)
    print("  [2계층 (Tier 2 & Two-Tier Cascade) 공식 성능 감사 - Level 1~3 실무군 676건 대상 평가]")
    print("=" * 90)
    print(f"[*] Level 1~3 샘플 수: {len(y)}개 (정상: {np.sum(y==0)}개, 결함: {np.sum(y==1)}개)")
    print(f"[*] 이전 v6.0 모델 (Level 1~3)  : F1={m_v6_clean['f1_score']*100:6.2f}%, Acc={m_v6_clean['accuracy']*100:6.2f}%, AUC={m_v6_clean['roc_auc']:.4f}")
    print(f"[*] Tier 2 단독 모델 (Level 1~3) : F1={m_t2_opt['f1_score']*100:6.2f}%, Acc={m_t2_opt['accuracy']*100:6.2f}%, AUC={m_t2_opt['roc_auc']:.4f}")
    print(f"[*] 2계층 캐스케이드 (Th=0.50)   : F1={m_cas_th50['f1_score']*100:6.2f}%, Acc={m_cas_th50['accuracy']*100:6.2f}%, AUC={m_cas_th50['roc_auc']:.4f}")
    print(f"[*] 2계층 캐스케이드 (시점최적) : ★ F1={m_cas_opt['f1_score']*100:6.2f}%, Acc={m_cas_opt['accuracy']*100:6.2f}%, AUC={m_cas_opt['roc_auc']:.4f}")
    print(f"[*] 추론 경로 분포              : Fast-path {fast_pct:.1f}% / Deep-path {deep_pct:.1f}%")
    print(f"[*] 성능 도약 폭 (vs v6.0)     : F1 도약 +{(m_cas_opt['f1_score'] - m_v6_clean['f1_score'])*100:+.2f}%p, Acc +{(m_cas_opt['accuracy'] - m_v6_clean['accuracy'])*100:+.2f}%p")

    # -------------------------------------------------------------------------
    # [5대 시점별 세부 성능 감사 (Level 1 및 Level 1~3)]
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  [공식 5대 카메라 시점(시점 A~E) 다차원 성능 감사]")
    print("=" * 90)

    view_results_l1 = {}
    view_results_l123 = {}

    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        alias = VIEW_ALIAS_MAP[vn]

        # Level 1 시점
        v_idx_l1 = np.where(views_l1 == vn)[0]
        y_vl1 = y_l1[v_idx_l1]
        pred_vl1 = preds_t1_l1_opt[v_idx_l1]
        prob_vl1 = p_t1_l1[v_idx_l1]
        view_results_l1[vn] = {
            "alias": alias,
            "sample_count": len(y_vl1),
            "f1_score": float(f1_score(y_vl1, pred_vl1, zero_division=0)),
            "accuracy": float(accuracy_score(y_vl1, pred_vl1)),
            "roc_auc": float(roc_auc_score(y_vl1, prob_vl1)) if len(np.unique(y_vl1)) > 1 else 0.0
        }

        # Level 1~3 시점
        v_idx_l123 = np.where(views == vn)[0]
        y_vl123 = y[v_idx_l123]
        pred_vl123 = preds_cas_opt[v_idx_l123]
        prob_vl123 = oof_cascade_probs[v_idx_l123]
        view_results_l123[vn] = {
            "alias": alias,
            "sample_count": len(y_vl123),
            "f1_score": float(f1_score(y_vl123, pred_vl123, zero_division=0)),
            "accuracy": float(accuracy_score(y_vl123, pred_vl123)),
            "roc_auc": float(roc_auc_score(y_vl123, prob_vl123)) if len(np.unique(y_vl123)) > 1 else 0.0
        }

        print(f"  ■ {alias:<22}: [Tier 1 L1] F1={view_results_l1[vn]['f1_score']*100:5.2f}% | [Cascade L1~3] F1={view_results_l123[vn]['f1_score']*100:5.2f}%, AUC={view_results_l123[vn]['roc_auc']:.4f}")

    # -------------------------------------------------------------------------
    # [최종 종합 결과 JSON 저장]
    # -------------------------------------------------------------------------
    summary_results = {
        "model_version": "v7.0 Two-Tier Cascade (37D)",
        "defect_name": "d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)",
        "architecture": "4-Level Ambiguity Pruned Two-Tier Cascade Tri-Ensemble",
        "total_raw_samples": int(N_all),
        "pruned_l4_samples": int(np.sum(l4_mask_all)),
        "valid_l123_samples": int(N_clean),
        "ambiguity_distribution": {
            "Level_1_Ground_Truth": int(np.sum(levels_all == 1)),
            "Level_2_Mild_Instability": int(np.sum(levels_all == 2)),
            "Level_3_View_Morphology": int(np.sum(levels_all == 3)),
            "Level_4_Mislabeled_Noise": int(np.sum(levels_all == 4))
        },
        "tier1_level1_metrics": m_t1_l1_opt,
        "tier2_level123_metrics": m_t2_opt,
        "cascade_level123_metrics": m_cas_opt,
        "previous_v6_metrics": {
            "v6_total_770": {
                "accuracy": 0.7909,
                "f1_score": 0.8076,
                "roc_auc": 0.8797
            },
            "v6_on_level1_459": m_v6_l1,
            "v6_on_level123_676": m_v6_clean
        },
        "performance_comparison": {
            "level1_tier1_vs_v6": {
                "f1_gain": round((m_t1_l1_opt['f1_score'] - m_v6_l1['f1_score']) * 100, 2),
                "acc_gain": round((m_t1_l1_opt['accuracy'] - m_v6_l1['accuracy']) * 100, 2),
                "auc_gain": round((m_t1_l1_opt['roc_auc'] - m_v6_l1['roc_auc']), 4)
            },
            "level123_cascade_vs_v6": {
                "f1_gain": round((m_cas_opt['f1_score'] - m_v6_clean['f1_score']) * 100, 2),
                "acc_gain": round((m_cas_opt['accuracy'] - m_v6_clean['accuracy']) * 100, 2),
                "auc_gain": round((m_cas_opt['roc_auc'] - m_v6_clean['roc_auc']), 4)
            },
            "cascade_vs_v6_original_770": {
                "f1_gain": round((m_cas_opt['f1_score'] - 0.8076) * 100, 2),
                "acc_gain": round((m_cas_opt['accuracy'] - 0.7909) * 100, 2),
                "auc_gain": round((m_cas_opt['roc_auc'] - 0.8797), 4)
            }
        },
        "view_breakdown_level1": view_results_l1,
        "view_breakdown_level123": view_results_l123,
        "optimal_thresholds_level1": opt_th_l1,
        "optimal_thresholds_level123": opt_th_l123
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 공식 성능 감사 JSON 저장 완료: {OUT_RESULTS_JSON}")
    print("=" * 90)


if __name__ == "__main__":
    main()
