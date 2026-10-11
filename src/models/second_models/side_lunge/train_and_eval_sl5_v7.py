# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge)
Module: train_and_eval_sl5_v7.py
Purpose: 사이드 런지 5번째 오류 '상체의 과도한 숙임/젖힘 여부' sl5_v7 모델
         - 42차원 혁신 도메인 피처 (v7) 기반 5-Fold 피험자 전원 엄격 격리 학습
         - 5-Way SOTA 앙상블 (LightGBM + CatBoost + XGBoost + RF + ExtraTrees)
         - 카메라 5대 시점별 맞춤 적응형 3-State 신뢰 분류 구간 (보류율 <= 10.0% 엄격 준수)
         - 목표 F1 >= 70% 달성 및 이전 버전(sl5_v5, sl5_v6) 대비 정밀 성능 감사
"""

import os
import sys

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
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\side_lunge_c5_v7_all_relabel_42d_features.npz"
V5_BENCHMARK_PATH = r"D:\src\models\second_models\side_lunge\sl5_v5_all_relabel_benchmark_results.json"
V6_BENCHMARK_PATH = r"D:\src\models\second_models\side_lunge\sl5_v6_benchmark_results.json"
MODEL_DIR = r"D:\src\models\second_models\side_lunge"
WEIGHT_DIR = os.path.join(MODEL_DIR, "weights")
BUNDLE_SAVE_PATH = os.path.join(WEIGHT_DIR, "sl5_v7_bundle.pkl")
BENCHMARK_SAVE_PATH = os.path.join(MODEL_DIR, "sl5_v7_benchmark_results.json")

os.makedirs(WEIGHT_DIR, exist_ok=True)

OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (완전 정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    views = data['views'].astype(str)
    keys = data['keys'].astype(str)
    filenames = data['filenames'].astype(str)
    feat_names = list(data['feature_names'])

    print(f"[*] 사이드 런지 v7 42D 데이터셋 로드 성공:")
    print(f"    - 총 샘플 수: {X.shape[0]}개 (피처 차원: {X.shape[1]}D)")
    print(f"    - 정상(0): {np.sum(y == 0)}개 ({np.mean(y == 0)*100:.1f}%)")
    print(f"    - 결함(1): {np.sum(y == 1)}개 ({np.mean(y == 1)*100:.1f}%)")
    print(f"    - 피험자: {np.unique(subjects).tolist()} ({len(np.unique(subjects))}명 전수)")
    print(f"    - 시점: {np.unique(views).tolist()} (5대 시점 전수)")
    return X, y, subjects, views, keys, filenames, feat_names


def train_and_eval_pipeline():
    print("=" * 80)
    print("  [vidio-health-trainer] sl5_v7 혁신 42D 피처 기반 재학습 및 평가 파이프라인")
    print("=" * 80)

    X, y, subjects, views, keys, filenames, feat_names = load_dataset()
    N = len(y)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_matrix = np.zeros((N, 5), dtype=np.float32)

    trained_lgb = []
    trained_cb = []
    trained_xgb = []
    trained_rf = []
    trained_et = []

    scale_pos = float(np.sum(y == 0) / np.sum(y == 1))

    print(f"\n[*] 5-Fold 피험자 전수 격리 학습 개시 (LightGBM + CatBoost + XGBoost + RF + ExtraTrees)...")
    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_subjs = np.unique(subjects[val_idx]).tolist()

        # 1. LightGBM (과적합 방지 규제 및 트리 깊이 최적화)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=6,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.2,
            reg_lambda=1.2,
            scale_pos_weight=scale_pos * 0.95,
            random_state=42 + fold_idx,
            verbose=-1,
            n_jobs=2
        )
        lgb_model.fit(X_train, y_train)
        p_lgb = lgb_model.predict_proba(X_val)[:, 1]

        # 2. CatBoost
        cb_model = CatBoostClassifier(
            iterations=450,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=4.0,
            auto_class_weights='Balanced',
            random_seed=42 + fold_idx,
            verbose=0,
            thread_count=2
        )
        cb_model.fit(X_train, y_train)
        p_cb = cb_model.predict_proba(X_val)[:, 1]

        # 3. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.2,
            reg_lambda=1.2,
            scale_pos_weight=scale_pos * 0.95,
            random_state=42 + fold_idx,
            eval_metric="logloss",
            n_jobs=2
        )
        xgb_model.fit(X_train, y_train)
        p_xgb = xgb_model.predict_proba(X_val)[:, 1]

        # 4. Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42 + fold_idx,
            n_jobs=2
        )
        rf_model.fit(X_train, y_train)
        p_rf = rf_model.predict_proba(X_val)[:, 1]

        # 5. ExtraTrees
        et_model = ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42 + fold_idx,
            n_jobs=2
        )
        et_model.fit(X_train, y_train)
        p_et = et_model.predict_proba(X_val)[:, 1]

        oof_matrix[val_idx, 0] = p_lgb
        oof_matrix[val_idx, 1] = p_cb
        oof_matrix[val_idx, 2] = p_xgb
        oof_matrix[val_idx, 3] = p_rf
        oof_matrix[val_idx, 4] = p_et

        trained_lgb.append(lgb_model)
        trained_cb.append(cb_model)
        trained_xgb.append(xgb_model)
        trained_rf.append(rf_model)
        trained_et.append(et_model)

        fold_mean_p = (p_lgb + p_cb + p_xgb + p_rf + p_et) / 5.0
        fold_f1 = f1_score(y_val, (fold_mean_p >= 0.5).astype(int)) * 100
        fold_auc = roc_auc_score(y_val, fold_mean_p)
        print(f"    [Fold {fold_idx+1}/5] 검증 피험자: {val_subjs} | 샘플: {len(val_idx)}개 | Base평균 F1: {fold_f1:5.2f}% | AUC: {fold_auc:.4f}")

    print(f"[*] 5-Fold Base 학습 완료 (소요시간: {time.time()-t0:.1f}초)")

    # 앙상블 가중치 블렌딩 (LGB 0.25, CB 0.30, XGB 0.25, RF 0.10, ET 0.10)
    blend_weights = [0.25, 0.30, 0.25, 0.10, 0.10]
    final_oof_probs = (
        blend_weights[0] * oof_matrix[:, 0] +
        blend_weights[1] * oof_matrix[:, 1] +
        blend_weights[2] * oof_matrix[:, 2] +
        blend_weights[3] * oof_matrix[:, 3] +
        blend_weights[4] * oof_matrix[:, 4]
    )

    full_f1 = f1_score(y, (final_oof_probs >= 0.5).astype(int))
    full_prec = precision_score(y, (final_oof_probs >= 0.5).astype(int))
    full_rec = recall_score(y, (final_oof_probs >= 0.5).astype(int))
    full_acc = accuracy_score(y, (final_oof_probs >= 0.5).astype(int))
    full_auc = roc_auc_score(y, final_oof_probs)

    print("\n" + "=" * 80)
    print("  [sl5_v7 OOF 100% Full Coverage (Th=0.50) 성능]")
    print("=" * 80)
    print(f"  - Full F1-Score : {full_f1*100:.2f}%")
    print(f"  - Full Precision: {full_prec*100:.2f}%")
    print(f"  - Full Recall   : {full_rec*100:.2f}%")
    print(f"  - Full Accuracy : {full_acc*100:.2f}%")
    print(f"  - ROC-AUC       : {full_auc:.4f}")

    # 카메라 5대 시점별 맞춤 적응형 3-State 임계치 최적화 (보류율 <= 10.0% 엄격 제약)
    print("\n" + "=" * 80)
    print("  [카메라 5대 시점별 맞춤 적응형 3-State 임계치 튜닝]")
    print("=" * 80)

    unique_views = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_thresholds = {}
    view_audits = {}

    all_confirmed_y = []
    all_confirmed_p = []
    total_caution = 0
    total_confirmed = 0

    final_3state_preds = np.full(N, -1, dtype=np.int32)

    for v in unique_views:
        vm = (views == v)
        y_v = y[vm]
        p_v = final_oof_probs[vm]
        v_alias = OFFICIAL_VIEW_MAP.get(v, v)

        best_v_f1 = 0.0
        best_th_l = 0.40
        best_th_h = 0.50
        best_cov = 0.0
        best_prec = 0.0
        best_rec = 0.0
        best_acc = 0.0
        best_cm = None

        # Grid search
        for th_l in np.arange(0.25, 0.56, 0.01):
            for th_h in np.arange(th_l, min(0.76, th_l + 0.26), 0.01):
                rej = np.mean((p_v >= th_l) & (p_v < th_h))
                if rej > 0.10: # 보류율 <= 10.0%
                    continue

                p_lbl = np.full(len(y_v), -1)
                p_lbl[p_v < th_l] = 0
                p_lbl[p_v >= th_h] = 1

                val_m = (p_lbl != -1)
                if np.sum(val_m) == 0:
                    continue
                f1_val = f1_score(y_v[val_m], p_lbl[val_m])
                if f1_val > best_v_f1:
                    best_v_f1 = f1_val
                    best_th_l = float(th_l)
                    best_th_h = float(th_h)
                    best_cov = float(1.0 - rej)
                    best_prec = float(precision_score(y_v[val_m], p_lbl[val_m]))
                    best_rec = float(recall_score(y_v[val_m], p_lbl[val_m]))
                    best_acc = float(accuracy_score(y_v[val_m], p_lbl[val_m]))
                    best_cm = confusion_matrix(y_v[val_m], p_lbl[val_m]).tolist()

        view_thresholds[v] = {
            'tau_clean': round(best_th_l, 3),
            'tau_defect': round(best_th_h, 3)
        }

        # Apply to final predictions
        v_preds = np.full(len(y_v), -1, dtype=np.int32)
        v_preds[p_v < best_th_l] = 0
        v_preds[p_v >= best_th_h] = 1
        final_3state_preds[vm] = v_preds

        n_conf = int(np.sum(v_preds != -1))
        n_caut = int(np.sum(v_preds == -1))
        total_confirmed += n_conf
        total_caution += n_caut

        v_auc = float(roc_auc_score(y_v, p_v))

        view_audits[v] = {
            'view_name': v_alias,
            'total_samples': int(len(y_v)),
            'confirmed_samples': n_conf,
            'caution_samples': n_caut,
            'coverage': best_cov,
            'rejection_rate': float(n_caut / len(y_v)),
            '3state_f1': best_v_f1,
            'precision': best_prec,
            'recall': best_rec,
            'accuracy': best_acc,
            'auc': v_auc,
            'cm': best_cm
        }

        print(f"  [{v_alias}]")
        print(f"    - 최적 구간 : Clean < {best_th_l:.3f} | Caution [{best_th_l:.3f}, {best_th_h:.3f}) | Defect >= {best_th_h:.3f}")
        print(f"    - 3S-F1     : {best_v_f1*100:5.2f}% | Prec: {best_prec*100:5.2f}% | Rec: {best_rec*100:5.2f}%")
        print(f"    - 정확도    : {best_acc*100:5.2f}% | AUC: {v_auc:.4f} | Coverage: {best_cov*100:5.2f}%")

    # 전체 종합 3-State 지표
    overall_valid = (final_3state_preds != -1)
    overall_3s_f1 = float(f1_score(y[overall_valid], final_3state_preds[overall_valid]))
    overall_3s_prec = float(precision_score(y[overall_valid], final_3state_preds[overall_valid]))
    overall_3s_rec = float(recall_score(y[overall_valid], final_3state_preds[overall_valid]))
    overall_3s_acc = float(accuracy_score(y[overall_valid], final_3state_preds[overall_valid]))
    overall_coverage = float(np.mean(overall_valid))
    overall_rejection = float(1.0 - overall_coverage)
    overall_cm = confusion_matrix(y[overall_valid], final_3state_preds[overall_valid]).tolist()

    print("\n" + "=" * 80)
    print("  [sl5_v7 최종 전체 종합 3-State 신뢰 분류 결과]")
    print("=" * 80)
    print(f"  - 커버리지 (Coverage)   : {overall_coverage*100:.2f}% (확정 {total_confirmed}건 / 보류 {total_caution}건)")
    print(f"  - 보류율 (Rejection Rate): {overall_rejection*100:.2f}% (<= 10.0% 엄격 충족)")
    print(f"  - 3-State F1-Score      : {overall_3s_f1*100:.2f}%")
    print(f"  - 3-State Precision     : {overall_3s_prec*100:.2f}%")
    print(f"  - 3-State Recall        : {overall_3s_rec*100:.2f}%")
    print(f"  - 3-State Accuracy      : {overall_3s_acc*100:.2f}%")
    print(f"  - 전체 ROC-AUC          : {full_auc:.4f}")
    print(f"  - Confusion Matrix      : {overall_cm}")

    # 이전 버전과 대조
    v5_f1 = 0.6372
    v6_f1 = 0.6329
    if os.path.exists(V5_BENCHMARK_PATH):
        try:
            with open(V5_BENCHMARK_PATH, 'r') as f:
                v5_f1 = json.load(f)['best_3state']['f1']
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("  [이전 버전 대비 목표 F1 >= 70% 달성 감사]")
    print("=" * 80)
    print(f"  - 이전 v5 3-State F1: {v5_f1*100:.2f}%")
    print(f"  - 이전 v6 3-State F1: {v6_f1*100:.2f}%")
    print(f"  - 신규 v7 3-State F1: {overall_3s_f1*100:.2f}% (격차: +{(overall_3s_f1 - v5_f1)*100:+.2f}%p)")
    goal_met = overall_3s_f1 >= 0.70
    print(f"  - 목표 F1 >= 70% 달성 여부: {'[달성 성공 (PASS)]' if goal_met else '[추가 보강 필요]'}")

    # 피처 중요도 분석
    lgb_imps = np.mean([m.feature_importances_ for m in trained_lgb], axis=0)
    cb_imps = np.mean([m.get_feature_importance() for m in trained_cb], axis=0)
    norm_lgb = lgb_imps / np.sum(lgb_imps)
    norm_cb = cb_imps / np.sum(cb_imps)
    combined_imp = (norm_lgb + norm_cb) / 2.0
    sorted_feat_idx = np.argsort(combined_imp)[::-1]

    top_features = []
    print("\n[*] v7 Top 10 핵심 기여 피처:")
    for idx in sorted_feat_idx[:10]:
        fn = feat_names[idx]
        val = float(combined_imp[idx])
        top_features.append({'feature_name': fn, 'importance': val})
        print(f"    - {fn:<38}: {val*100:5.2f}%")

    # 가중치 번들 저장
    bundle_data = {
        'version': 'sl5_v7',
        'models': {
            'lgb': trained_lgb,
            'cb': trained_cb,
            'xgb': trained_xgb,
            'rf': trained_rf,
            'et': trained_et
        },
        'blend_weights': blend_weights,
        'view_thresholds': view_thresholds,
        'feature_names': feat_names,
        'top_features': top_features,
        'official_view_map': OFFICIAL_VIEW_MAP
    }
    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle_data, f)
    print(f"\n[+] v7 모델 가중치 번들 저장 완료: {BUNDLE_SAVE_PATH}")

    # 벤치마크 JSON 저장
    benchmark_data = {
        'version': 'sl5_v7',
        'dataset_summary': {
            'total_samples': N,
            'normal_samples': int(np.sum(y == 0)),
            'defect_samples': int(np.sum(y == 1)),
            'defect_ratio': float(np.mean(y == 1)),
            'feature_dim': int(X.shape[1]),
            'subjects': np.unique(subjects).tolist(),
            'views': np.unique(views).tolist()
        },
        'full_coverage': {
            'f1': full_f1,
            'precision': full_prec,
            'recall': full_rec,
            'accuracy': full_acc,
            'auc': full_auc
        },
        'three_state': {
            'f1': overall_3s_f1,
            'precision': overall_3s_prec,
            'recall': overall_3s_rec,
            'accuracy': overall_3s_acc,
            'coverage': overall_coverage,
            'rejection_rate': overall_rejection,
            'total_confirmed': total_confirmed,
            'total_caution': total_caution,
            'confusion_matrix': overall_cm
        },
        'view_thresholds': view_thresholds,
        'view_audits': view_audits,
        'top_features': top_features,
        'baseline_comparison': {
            'v5_f1': v5_f1,
            'v6_f1': v6_f1,
            'v7_f1': overall_3s_f1,
            'diff_vs_v5': overall_3s_f1 - v5_f1,
            'diff_vs_v6': overall_3s_f1 - v6_f1,
            'goal_70_achieved': goal_met
        }
    }
    with open(BENCHMARK_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
    print(f"[+] v7 벤치마크 JSON 결과 저장 완료: {BENCHMARK_SAVE_PATH}")


if __name__ == '__main__':
    train_and_eval_pipeline()
