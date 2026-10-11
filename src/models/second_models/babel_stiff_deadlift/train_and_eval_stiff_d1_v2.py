# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 3: sd3 (바벨 궤적과 몸 밀착 위반 / Barbell Path Proximity Defect)
Version: v2.0 (sd3_v2)
Module: train_and_eval_stiff_d1_v2.py
Purpose: 6대 상호 간섭 피처 Pruning 후 32D 최적화 피처셋 기반
         5-Fold StratifiedGroupKFold 교차검증, 3-State 최적화 및 최종 가중치 번들 영속화
Rule: AGENTS.md 및 공식 5대 시점 규격, 순수 비전 MediaPipe 33관절(64F) 입력 준수
"""

import os
import sys
import time
import pickle
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier

# Windows OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import (
    FEATURE_NAMES_38_STIFF_D1,
    FEATURE_NAMES_32_STIFF_D1,
    FEATURE_NAMES_SD3_V2,
    KEPT_INDICES_32,
    PRUNED_FEATURE_NAMES_6,
    VIEW_ALIAS_MAP
)

CACHE_PKL_PATH = os.path.join(CURRENT_DIR, "stiff_d1_dataset_cache.pkl")
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
BUNDLE_PKL_PATH = os.path.join(WEIGHTS_DIR, "sd3_v2_bundle.pkl")
ALT_WEIGHTS_PKL = os.path.join(CURRENT_DIR, "sd3_model_v2_weights.pkl")
EVAL_JSON_PATH = os.path.join(CURRENT_DIR, "stiff_d1_v2_evaluation_results.json")


def load_dataset_v2() -> Tuple[pd.DataFrame, np.ndarray]:
    """캐시된 38D 데이터셋으로부터 32D 최적화 피처 행렬 X_v2 로드"""
    if not os.path.exists(CACHE_PKL_PATH):
        raise FileNotFoundError(f"Cache file not found: {CACHE_PKL_PATH}")
    with open(CACHE_PKL_PATH, 'rb') as fp:
        cached = pickle.load(fp)
    df = cached['df']
    X_38 = cached['X']
    X_32 = X_38[:, KEPT_INDICES_32].astype(np.float32)
    return df, X_32


def main():
    print("=" * 80, flush=True)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 d1 (sd3_v2) 32D 최적화 모델 학습 및 평가", flush=True)
    print("=" * 80, flush=True)

    df, X = load_dataset_v2()
    y = df['y'].values
    subjs = df['subj'].values
    views = df['view'].values
    keys = df['key'].values

    n_samples = len(y)
    n_defects = int(np.sum(y == 1))
    n_cleans = int(np.sum(y == 0))
    n_subjs = len(np.unique(subjs))

    print(f"\n[*] 32D 데이터셋 로드 완료:")
    print(f"    - 총 시퀀스: {n_samples}건 (결함: {n_defects}건, 정상: {n_cleans}건)")
    print(f"    - 피험자: AI-Hub 1~9번 전원 ({n_subjs}명)")
    print(f"    - 피처 차원: {X.shape[1]}D (제거된 간섭 피처: {len(PRUNED_FEATURE_NAMES_6)}개)")
    print(f"    - 피처 목록: {FEATURE_NAMES_32_STIFF_D1[:3]} ... {FEATURE_NAMES_32_STIFF_D1[-3:]}", flush=True)

    # 5-Fold StratifiedGroupKFold 교차 검증
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(n_samples, dtype=np.float32)

    fold_metrics = []
    print(f"\n[*] 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습 시작...", flush=True)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjs)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_subjs = np.unique(subjs[val_idx])

        # 1. LightGBM (Windows OpenMP 안정성: n_jobs=1)
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.035,
            max_depth=5,
            num_leaves=16,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)

        # 2. CatBoost (Windows 안정성: thread_count=1)
        clf_cb = CatBoostClassifier(
            iterations=180,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0,
            thread_count=1
        )
        clf_cb.fit(X_tr, y_tr)

        p_lgb = clf_lgb.predict_proba(X_val)[:, 1]
        p_cb = clf_cb.predict_proba(X_val)[:, 1]
        p_fold = 0.55 * p_lgb + 0.45 * p_cb
        oof_probs[val_idx] = p_fold

        f1_fold = f1_score(y_val, (p_fold >= 0.50).astype(int), zero_division=0) * 100.0
        auc_fold = roc_auc_score(y_val, p_fold) * 100.0
        fold_metrics.append({'fold': fold + 1, 'f1': f1_fold, 'auc': auc_fold, 'subjs': list(val_subjs)})
        print(f"  [Fold {fold+1}/5] 검증 피험자: {val_subjs} | F1: {f1_fold:5.2f}% | AUC: {auc_fold:5.2f}%", flush=True)

    # 2-State 성능 산출
    preds_50 = (oof_probs >= 0.50).astype(int)
    f1_2s = float(f1_score(y, preds_50, zero_division=0) * 100.0)
    prec_2s = float(precision_score(y, preds_50, zero_division=0) * 100.0)
    rec_2s = float(recall_score(y, preds_50, zero_division=0) * 100.0)
    acc_2s = float(accuracy_score(y, preds_50) * 100.0)
    auc_2s = float(roc_auc_score(y, oof_probs) * 100.0)
    cm_2s = confusion_matrix(y, preds_50, labels=[0, 1])
    tn_2s, fp_2s, fn_2s, tp_2s = cm_2s.ravel()

    print("\n" + "=" * 80, flush=True)
    print("  sd3_v2 (32D) 베이스라인 2-State (임계치 0.50) OOF 전체 성능", flush=True)
    print("=" * 80, flush=True)
    print(f"  - F1-Score : {f1_2s:.2f}%")
    print(f"  - Precision: {prec_2s:.2f}%")
    print(f"  - Recall   : {rec_2s:.2f}%")
    print(f"  - Accuracy : {acc_2s:.2f}%")
    print(f"  - ROC-AUC  : {auc_2s:.2f}%")
    print(f"  - 혼동행렬 : TP={tp_2s}, TN={tn_2s}, FP={fp_2s}, FN={fn_2s}", flush=True)

    # 3-State Selective Gating 최적화 그리드 서치 (보류율 <= 10.0% / 커버리지 >= 90.0%)
    print("\n" + "=" * 80, flush=True)
    print("  3-State Selective Gating 최적화 (보류율 <= 10.0% 제약 하 F1 극대화)", flush=True)
    print("=" * 80, flush=True)

    best_3state = None
    best_f1 = -1.0

    tc_range = np.linspace(0.15, 0.48, 34)
    td_range = np.linspace(0.50, 0.85, 36)

    for tc in tc_range:
        for td in td_range:
            if td <= tc:
                continue
            decided_mask = (oof_probs <= tc) | (oof_probs >= td)
            cov = float(np.mean(decided_mask))
            if cov < 0.90:
                continue

            y_dec = y[decided_mask]
            p_dec = (oof_probs[decided_mask] >= td).astype(int)

            if len(np.unique(y_dec)) < 2:
                continue

            cur_f1 = f1_score(y_dec, p_dec, zero_division=0)
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                cm_dec = confusion_matrix(y_dec, p_dec, labels=[0, 1])
                tn_d, fp_d, fn_d, tp_d = cm_dec.ravel()
                best_3state = {
                    'tau_clean': float(tc),
                    'tau_defect': float(td),
                    'coverage': float(cov * 100.0),
                    'abstain_rate': float((1.0 - cov) * 100.0),
                    'f1': float(cur_f1 * 100.0),
                    'precision': float(precision_score(y_dec, p_dec, zero_division=0) * 100.0),
                    'recall': float(recall_score(y_dec, p_dec, zero_division=0) * 100.0),
                    'accuracy': float(accuracy_score(y_dec, p_dec) * 100.0),
                    'tp': int(tp_d),
                    'tn': int(tn_d),
                    'fp': int(fp_d),
                    'fn': int(fn_d),
                    'decided_count': int(len(y_dec)),
                    'total_count': int(len(y))
                }

    print(f"[최적 3-State 판정 윈도우 확정]")
    print(f"  - Tau Clean (정상 한계치) : {best_3state['tau_clean']:.2f}")
    print(f"  - Tau Defect (결함 한계치): {best_3state['tau_defect']:.2f}")
    print(f"  - 커버리지 (Coverage)     : {best_3state['coverage']:.2f}% (보류율: {best_3state['abstain_rate']:.2f}%)")
    print(f"  ★ 3-State F1-Score        : {best_3state['f1']:.2f}% (2-State 대비 {best_3state['f1'] - f1_2s:+5.2f}%p)")
    print(f"  - 3-State 정밀도 / 재현율 : {best_3state['precision']:.2f}% / {best_3state['recall']:.2f}%")
    print(f"  - 3-State 정확도 (Accuracy): {best_3state['accuracy']:.2f}%")
    print(f"  ★ 오탐 (FP, False Positive): {fp_2s}건 -> {best_3state['fp']}건 ({best_3state['fp'] - fp_2s:+d}건 감소!)", flush=True)

    # 공식 5대 시점별 세부 성능 감사
    print("\n" + "=" * 80, flush=True)
    print("  공식 5대 카메라 시점별 최종 성능 감사 (AGENTS.md 표준)", flush=True)
    print("=" * 80, flush=True)

    tc = best_3state['tau_clean']
    td = best_3state['tau_defect']
    view_metrics = {}

    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        if np.sum(v_mask) == 0:
            continue
        y_v = y[v_mask]
        p_v = oof_probs[v_mask]

        mask_v_dec = (p_v <= tc) | (p_v >= td)
        y_v_dec = y_v[mask_v_dec]
        p_v_dec = (p_v[mask_v_dec] >= td).astype(int)

        cov_v = float(np.mean(mask_v_dec) * 100.0)
        f1_v = float(f1_score(y_v_dec, p_v_dec, zero_division=0) * 100.0) if len(np.unique(y_v_dec)) > 1 else 0.0
        prec_v = float(precision_score(y_v_dec, p_v_dec, zero_division=0) * 100.0) if len(np.unique(y_v_dec)) > 1 else 0.0
        rec_v = float(recall_score(y_v_dec, p_v_dec, zero_division=0) * 100.0) if len(np.unique(y_v_dec)) > 1 else 0.0
        acc_v = float(accuracy_score(y_v_dec, p_v_dec) * 100.0) if len(y_v_dec) > 0 else 0.0
        auc_v = float(roc_auc_score(y_v, p_v) * 100.0) if len(np.unique(y_v)) > 1 else 0.0

        cm_v = confusion_matrix(y_v_dec, p_v_dec, labels=[0, 1])
        tn_v, fp_v, fn_v, tp_v = cm_v.ravel()

        view_metrics[v_code] = {
            'alias': v_alias,
            'total_samples': int(len(y_v)),
            'decided_samples': int(len(y_v_dec)),
            'coverage': cov_v,
            'f1': f1_v,
            'precision': prec_v,
            'recall': rec_v,
            'accuracy': acc_v,
            'auc': auc_v,
            'tp': int(tp_v),
            'tn': int(tn_v),
            'fp': int(fp_v),
            'fn': int(fn_v)
        }
        print(f"[{v_alias}]")
        print(f"  3-State F1: {f1_v:5.2f}% | 정밀도: {prec_v:5.2f}% | 재현율: {rec_v:5.2f}% | 정확도: {acc_v:5.2f}% | AUC: {auc_v:5.2f}% | FP: {fp_v} | 커버리지: {cov_v:.1f}%", flush=True)

    # 전체 데이터 기반 최종 프로덕션 가중치 학습
    print("\n[*] 전체 1,035개 데이터셋 기반 최종 프로덕션 sd3_v2 가중치 번들 학습 중...", flush=True)
    final_lgb = lgb.LGBMClassifier(
        n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=16,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
        random_state=42, verbose=-1, n_jobs=1
    )
    final_lgb.fit(X, y)

    final_cb = CatBoostClassifier(
        iterations=200, learning_rate=0.035, depth=5, l2_leaf_reg=3.0,
        random_seed=42, verbose=0, thread_count=1
    )
    final_cb.fit(X, y)

    bundle = {
        'model_name': 'sd3_v2',
        'exercise': '바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)',
        'defect': 'sd3 (바벨 궤적과 몸 밀착 위반)',
        'version': 'v2.0 Pruned 32D',
        'feature_names': FEATURE_NAMES_32_STIFF_D1,
        'feature_dim': len(FEATURE_NAMES_32_STIFF_D1),
        'pruned_features': PRUNED_FEATURE_NAMES_6,
        'lgb_model': final_lgb,
        'cb_model': final_cb,
        'weights': (0.55, 0.45),
        'decision_policy': '3-State View-Calibrated Gating',
        'optimal_3state': best_3state,
        'metrics_2state': {
            'f1': f1_2s, 'precision': prec_2s, 'recall': rec_2s,
            'accuracy': acc_2s, 'auc': auc_2s
        },
        'view_metrics': view_metrics,
        'view_alias_map': VIEW_ALIAS_MAP
    }

    with open(BUNDLE_PKL_PATH, 'wb') as fp:
        pickle.dump(bundle, fp)
    with open(ALT_WEIGHTS_PKL, 'wb') as fp:
        pickle.dump(bundle, fp)
    print(f"[+] 최종 sd3_v2 가중치 번들 저장 완료: {BUNDLE_PKL_PATH}")
    print(f"[+] 대체 가중치 파일 저장 완료: {ALT_WEIGHTS_PKL}", flush=True)

    # 평가 결과 JSON 저장
    eval_summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': 'sd3_v2',
        'version': 'v2.0 Pruned 32D',
        'exercise': '바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)',
        'defect': 'sd3 (바벨 궤적과 몸 밀착 위반)',
        'feature_dim': 32,
        'pruned_features': PRUNED_FEATURE_NAMES_6,
        'metrics_2state': {
            'f1': f1_2s, 'precision': prec_2s, 'recall': rec_2s,
            'accuracy': acc_2s, 'auc': auc_2s,
            'tp': int(tp_2s), 'tn': int(tn_2s), 'fp': int(fp_2s), 'fn': int(fn_2s)
        },
        'optimal_3state': best_3state,
        'view_metrics': view_metrics,
        'fold_metrics': fold_metrics
    }
    with open(EVAL_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(eval_summary, fp, ensure_ascii=False, indent=2)
    print(f"[+] 최종 sd3_v2 평가 결과 JSON 저장 완료: {EVAL_JSON_PATH}", flush=True)


if __name__ == '__main__':
    main()
