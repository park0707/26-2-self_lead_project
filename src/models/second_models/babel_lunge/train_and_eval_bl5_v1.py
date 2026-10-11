# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: train_and_eval_bl5_v1.py
Purpose: 바벨 런지 5번째 오류 '상체의 과도한 숙임/젖힘 여부' bl5_v1 모델 학습,
         5-Fold 피험자 전수 격리 교차검증, 전체 3-State F1 최적화 및 5대 시점 감사
Requirements:
  - 모델 이름: bl5_v1.py
  - 계층별 분리 성능 대신 전체 3-State F1-Score로 성능 종합 보고
  - AGENTS.md 5대 공식 시점(시점 A~E) 및 9명 피험자 전수 감사 준수
"""

import os
import sys

# Windows / OpenMP 충돌 방지 및 단일 스레드 안전화
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import pickle
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\lunge_c5_32d_features.npz"
MODEL_DIR = r"D:\src\models\second_models\babel_lunge"
WEIGHT_DIR = os.path.join(MODEL_DIR, "weights")
BUNDLE_SAVE_PATH = os.path.join(WEIGHT_DIR, "bl5_v1_bundle.pkl")
BENCHMARK_SAVE_PATH = os.path.join(MODEL_DIR, "bl5_v1_benchmark_results.json")

os.makedirs(WEIGHT_DIR, exist_ok=True)

# AGENTS.md 공식 5대 시점 매핑
OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """캐시된 32D 피처 텐서 로드"""
    if not os.path.exists(FEATURE_CACHE_PATH):
        raise FileNotFoundError(f"피처 캐시 파일이 존재하지 않습니다: {FEATURE_CACHE_PATH}")

    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    views = data['views'].astype(str)
    feat_names = list(data['feature_names'])

    print(f"[*] 데이터셋 로드 성공:", flush=True)
    print(f"    - 총 샘플 수: {X.shape[0]}개 (피처 차원: {X.shape[1]}D)", flush=True)
    print(f"    - 정상(0): {np.sum(y == 0)}개 ({np.mean(y == 0)*100:.1f}%)", flush=True)
    print(f"    - 결함(1): {np.sum(y == 1)}개 ({np.mean(y == 1)*100:.1f}%)", flush=True)
    print(f"    - 피험자: {np.unique(subjects).tolist()} ({len(np.unique(subjects))}명 전수)", flush=True)
    print(f"    - 시점: {np.unique(views).tolist()} (5대 시점 전수)", flush=True)
    return X, y, subjects, views, feat_names


def train_and_eval_pipeline():
    print("=" * 80, flush=True)
    print("  [vidio-health-trainer] bl5_v1 전체 3-State F1 최적화 및 5대 시점 감사 파이프라인", flush=True)
    print("=" * 80, flush=True)

    X, y, subjects, views, feat_names = load_dataset()
    N = len(y)

    # 5-Fold StratifiedGroupKFold (피험자 전수 격리)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(N, dtype=np.float32)
    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_rf = np.zeros(N, dtype=np.float32)
    oof_ensemble = np.zeros(N, dtype=np.float32)

    trained_lgb_models = []
    trained_cb_models = []
    trained_rf_models = []

    print(f"\n[*] 5-Fold 피험자 전수 격리 교차검증 시작 (Tri-Ensemble: LGB 45% + CB 35% + RF 20%)...", flush=True)
    t0 = time.time()

    fold = 1
    for train_idx, val_idx in sgkf.split(X, y, groups=subjects):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        val_subjs = np.unique(subjects[val_idx])

        # 1. LightGBM
        scale_pos = float(np.sum(y_tr == 0) / max(1, np.sum(y_tr == 1)))
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            scale_pos_weight=scale_pos,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=1
        )
        lgb_model.fit(X_tr, y_tr)
        val_p_lgb = lgb_model.predict_proba(X_va)[:, 1]
        oof_preds_lgb[val_idx] = val_p_lgb
        trained_lgb_models.append(lgb_model)

        # 2. CatBoost
        cb_model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.05,
            depth=4,
            auto_class_weights='Balanced',
            random_seed=42 + fold,
            verbose=0,
            thread_count=1
        )
        cb_model.fit(X_tr, y_tr)
        val_p_cb = cb_model.predict_proba(X_va)[:, 1]
        oof_preds_cb[val_idx] = val_p_cb
        trained_cb_models.append(cb_model)

        # 3. Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=1
        )
        rf_model.fit(X_tr, y_tr)
        val_p_rf = rf_model.predict_proba(X_va)[:, 1]
        oof_preds_rf[val_idx] = val_p_rf
        trained_rf_models.append(rf_model)

        # Fold Tri-Ensemble
        val_p_ens = 0.45 * val_p_lgb + 0.35 * val_p_cb + 0.20 * val_p_rf
        oof_ensemble[val_idx] = val_p_ens

        fold_f1 = f1_score(y_va, (val_p_ens >= 0.5).astype(int), zero_division=0)
        fold_auc = roc_auc_score(y_va, val_p_ens) if len(np.unique(y_va)) > 1 else 0.5
        print(f"  [Fold {fold}] 격리 피험자: {val_subjs} | 검증 샘플: {len(val_idx)}개 | 기본 F1: {fold_f1*100:.2f}% | AUC: {fold_auc:.4f}", flush=True)
        fold += 1

    elapsed = time.time() - t0
    print(f"\n[+] 5-Fold 교차검증 완료 (소요 시간: {elapsed:.1f}초)", flush=True)

    # 100% Full-Coverage 베이스라인 성능 (임계값 0.50)
    base_preds = (oof_ensemble >= 0.50).astype(int)
    base_f1 = f1_score(y, base_preds, zero_division=0)
    base_acc = accuracy_score(y, base_preds)
    base_rec = recall_score(y, base_preds, zero_division=0)
    base_prec = precision_score(y, base_preds, zero_division=0)
    base_auc = roc_auc_score(y, oof_ensemble)

    print("\n" + "=" * 60, flush=True)
    print("  [100% Full-Coverage 기본 성능 (전수 채점, tau=0.50)]", flush=True)
    print("=" * 60, flush=True)
    print(f"  - 결함 F1-Score: {base_f1*100:.2f}%", flush=True)
    print(f"  - 정확도 (Accuracy): {base_acc*100:.2f}%", flush=True)
    print(f"  - 재현율 (Recall): {base_rec*100:.2f}%", flush=True)
    print(f"  - 정밀도 (Precision): {base_prec*100:.2f}%", flush=True)
    print(f"  - ROC-AUC: {base_auc:.4f}", flush=True)

    # 전체 3-State F1-Score 최적화 그리드 서치
    # 목표: 커버리지 88% 이상 유지하면서 3-State F1 극대화
    print("\n" + "=" * 60, flush=True)
    print("  [전체 3-State 선택적 판정 F1 최적화 탐색 (Coverage >= 88%)]", flush=True)
    print("=" * 60, flush=True)

    best_3state = {
        'tau': 0.50,
        'margin': 0.05,
        'f1': 0.0,
        'acc': 0.0,
        'prec': 0.0,
        'rec': 0.0,
        'coverage': 0.0,
        'retained_n': 0,
        'rejected_n': 0
    }

    for tau in np.arange(0.35, 0.65, 0.02):
        for margin in np.arange(0.02, 0.12, 0.01):
            low_b = tau - margin
            high_b = tau + margin

            accepted_mask = (oof_ensemble <= low_b) | (oof_ensemble >= high_b)
            coverage = np.mean(accepted_mask)

            if coverage < 0.88:
                continue

            y_acc = y[accepted_mask]
            prob_acc = oof_ensemble[accepted_mask]
            pred_acc = (prob_acc >= tau).astype(int)

            f1 = f1_score(y_acc, pred_acc, zero_division=0)
            if f1 > best_3state['f1']:
                best_3state = {
                    'tau': float(round(tau, 3)),
                    'margin': float(round(margin, 3)),
                    'f1': float(f1),
                    'acc': float(accuracy_score(y_acc, pred_acc)),
                    'prec': float(precision_score(y_acc, pred_acc, zero_division=0)),
                    'rec': float(recall_score(y_acc, pred_acc, zero_division=0)),
                    'coverage': float(coverage),
                    'retained_n': int(np.sum(accepted_mask)),
                    'rejected_n': int(np.sum(~accepted_mask))
                }

    print(f"  [★ 최적 3-State 확정 판정 성능 지표]", flush=True)
    print(f"    - 최적 결정 임계값 (tau): {best_3state['tau']:.2f} (마진: +/-{best_3state['margin']:.2f})", flush=True)
    print(f"    - 신뢰 보류(Retake) 구간: [{best_3state['tau']-best_3state['margin']:.2f}, {best_3state['tau']+best_3state['margin']:.2f}]", flush=True)
    print(f"    - 판정 커버리지 (Coverage): {best_3state['coverage']*100:.2f}% (확정 {best_3state['retained_n']}건 / 보류 {best_3state['rejected_n']}건)", flush=True)
    print(f"    - 전체 3-State 결함 F1-Score: {best_3state['f1']*100:.2f}% ★", flush=True)
    print(f"    - 3-State 정확도 (Accuracy): {best_3state['acc']*100:.2f}%", flush=True)
    print(f"    - 3-State 정밀도 (Precision): {best_3state['prec']*100:.2f}%", flush=True)
    print(f"    - 3-State 재현율 (Recall): {best_3state['rec']*100:.2f}%", flush=True)

    # AGENTS.md 공식 5대 시점별 전수 성능 감사
    print("\n" + "=" * 60, flush=True)
    print("  [AGENTS.md 규격: 공식 5대 시점별 세부 성능 전수 감사]", flush=True)
    print("=" * 60, flush=True)

    view_audit_results = {}
    tau_opt = best_3state['tau']
    margin_opt = best_3state['margin']

    for v_code in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_name = OFFICIAL_VIEW_MAP[v_code]
        v_mask = (views == v_code)
        n_v = np.sum(v_mask)

        if n_v == 0:
            continue

        y_v = y[v_mask]
        p_v = oof_ensemble[v_mask]

        pred_full_v = (p_v >= 0.50).astype(int)
        full_f1_v = f1_score(y_v, pred_full_v, zero_division=0)
        full_acc_v = accuracy_score(y_v, pred_full_v)
        full_rec_v = recall_score(y_v, pred_full_v, zero_division=0)
        full_prec_v = precision_score(y_v, pred_full_v, zero_division=0)
        auc_v = roc_auc_score(y_v, p_v) if len(np.unique(y_v)) > 1 else 0.5

        acc_mask_v = (p_v <= tau_opt - margin_opt) | (p_v >= tau_opt + margin_opt)
        cov_v = np.mean(acc_mask_v)
        y_v_acc = y_v[acc_mask_v]
        pred_v_acc = (p_v[acc_mask_v] >= tau_opt).astype(int)

        f1_3s_v = f1_score(y_v_acc, pred_v_acc, zero_division=0) if len(y_v_acc) > 0 else 0.0
        acc_3s_v = accuracy_score(y_v_acc, pred_v_acc) if len(y_v_acc) > 0 else 0.0
        prec_3s_v = precision_score(y_v_acc, pred_v_acc, zero_division=0) if len(y_v_acc) > 0 else 0.0
        rec_3s_v = recall_score(y_v_acc, pred_v_acc, zero_division=0) if len(y_v_acc) > 0 else 0.0

        view_audit_results[v_code] = {
            'official_name': v_name,
            'sample_count': int(n_v),
            'full_coverage': {
                'f1': float(full_f1_v),
                'acc': float(full_acc_v),
                'prec': float(full_prec_v),
                'rec': float(full_rec_v),
                'auc': float(auc_v)
            },
            'three_state': {
                'coverage': float(cov_v),
                'f1': float(f1_3s_v),
                'acc': float(acc_3s_v),
                'prec': float(prec_3s_v),
                'rec': float(rec_3s_v)
            }
        }

        print(f"  * {v_name} [{v_code}]:", flush=True)
        print(f"      - 샘플 수: {n_v}개 | Full F1: {full_f1_v*100:.2f}% | Full 정확도: {full_acc_v*100:.2f}% | AUC: {auc_v:.4f}", flush=True)
        print(f"      - 3-State 커버리지: {cov_v*100:.1f}% | 3-State F1: {f1_3s_v*100:.2f}% | 3-State 정확도: {acc_3s_v*100:.2f}%", flush=True)

    # 전체 결과 패키징 및 저장
    benchmark_data = {
        'model_name': 'bl5_v1',
        'exercise': '04_바벨_런지',
        'defect': 'c5. 상체의 과도한 숙임/젖힘 여부',
        'total_samples': N,
        'feature_dim': X.shape[1],
        'full_coverage_baseline': {
            'f1': float(base_f1),
            'accuracy': float(base_acc),
            'precision': float(base_prec),
            'recall': float(base_rec),
            'roc_auc': float(base_auc)
        },
        'overall_three_state': best_3state,
        'views_audit': view_audit_results
    }

    with open(BENCHMARK_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 공식 벤치마크 JSON 저장: {BENCHMARK_SAVE_PATH}", flush=True)

    # 가중치 번들 저장 (추론 서빙용)
    bundle = {
        'model_name': 'bl5_v1',
        'lgb_models': trained_lgb_models,
        'cb_models': trained_cb_models,
        'rf_models': trained_rf_models,
        'feature_names': feat_names,
        'best_3state_params': best_3state,
        'view_audit': view_audit_results
    }
    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] 운영 서빙 가중치 번들 저장: {BUNDLE_SAVE_PATH} ({os.path.getsize(BUNDLE_SAVE_PATH)/1024/1024:.2f} MB)", flush=True)

    return benchmark_data


if __name__ == "__main__":
    train_and_eval_pipeline()
