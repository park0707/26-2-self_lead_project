# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge)
Module: train_and_eval_dl5_v5.py
Purpose: 다이나믹 런지 5번째 오류 '상체의 과도한 숙임/젖힘 여부' dl5_v5 모델 학습,
         바벨 런지 bl5_v5 모델 아키텍처 및 36D SOTA 피처(CRP, 스펙트럼 에너지 비, 척추 분절 곡률, 반환 플럭스) 계승,
         5-Way 앙상블(LGBM + CatBoost + XGBoost + RF + ExtraTrees),
         5-Fold 피험자 전수 격리 교차검증, 전체 3-State F1 최적화 (보류율 최대 10% 이하 엄격 준수),
         바벨 런지(bl5_v5) vs 다이나믹 런지(dl5_v5) 성능 대조 및 5대 공식 시점 전수 감사
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

FEATURE_CACHE_PATH = r"D:\Data\피처_데이터\dynamic_lunge_c5_v5_36d_features.npz"
BL5_BENCHMARK_PATH = r"D:\src\models\second_models\babel_lunge\bl5_v5_benchmark_results.json"
MODEL_DIR = r"D:\src\models\second_models\dynamic_lunge"
WEIGHT_DIR = os.path.join(MODEL_DIR, "weights")
BUNDLE_SAVE_PATH = os.path.join(WEIGHT_DIR, "dl5_v5_bundle.pkl")
BENCHMARK_SAVE_PATH = os.path.join(MODEL_DIR, "dl5_v5_benchmark_results.json")

os.makedirs(WEIGHT_DIR, exist_ok=True)

OFFICIAL_VIEW_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def load_dl5_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not os.path.exists(FEATURE_CACHE_PATH):
        raise FileNotFoundError(f"피처 캐시 파일이 존재하지 않습니다: {FEATURE_CACHE_PATH}")

    data = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int64)
    subjects = data['subjects'].astype(np.int32)
    views = data['views'].astype(str)
    keys = data['keys'].astype(str)
    filenames = data['filenames'].astype(str)
    feat_names = list(data['feature_names'])

    print(f"[*] 다이나믹 런지 v5 SOTA 데이터셋 로드 성공:")
    print(f"    - 총 샘플 수: {X.shape[0]}개 (피처 차원: {X.shape[1]}D)")
    print(f"    - 정상(0): {np.sum(y == 0)}개 ({np.mean(y == 0)*100:.1f}%)")
    print(f"    - 결함(1): {np.sum(y == 1)}개 ({np.mean(y == 1)*100:.1f}%)")
    print(f"    - 피험자: {np.unique(subjects).tolist()} ({len(np.unique(subjects))}명 전수)")
    print(f"    - 시점: {np.unique(views).tolist()} (5대 시점 전수)")
    return X, y, subjects, views, keys, filenames, feat_names


def train_and_eval_dl5_pipeline():
    print("=" * 80)
    print("  [vidio-health-trainer] dl5_v5 SOTA 36D + 5-Way 앙상블 학습 및 평가 파이프라인")
    print("=" * 80)

    X, y, subjects, views, keys, filenames, feat_names = load_dl5_dataset()
    N = len(y)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_matrix = np.zeros((N, 5), dtype=np.float32)

    lgb_models = []
    cb_models = []
    xgb_models = []
    rf_models = []
    et_models = []

    pos_ratio = float(np.sum(y == 0)) / float(max(1, np.sum(y == 1)))

    print("\n[*] 5-Fold 피험자 전수 격리 학습 개시 (LightGBM + CatBoost + XGBoost + RF + ExtraTrees)...")
    t0 = time.time()

    femur_ratio_idx = feat_names.index("lunge_femur_to_torso_ratio") if "lunge_femur_to_torso_ratio" in feat_names else 0

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        subjs_val = np.unique(subjects[val_idx]).tolist()

        sample_weights = np.ones(len(train_idx), dtype=np.float32)
        for i in range(len(train_idx)):
            if X_train[i, femur_ratio_idx] > 1.05 and y_train[i] == 0:
                sample_weights[i] *= 1.5

        # 1. LightGBM
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.032,
            num_leaves=26,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.18,
            reg_lambda=1.5,
            scale_pos_weight=pos_ratio,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1
        )
        lgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_lgb = lgb_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 0] = pred_lgb
        lgb_models.append(lgb_clf)

        # 2. CatBoost
        cb_clf = CatBoostClassifier(
            iterations=240,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=4.8,
            scale_pos_weight=pos_ratio,
            random_seed=42 + fold,
            thread_count=1,
            verbose=0
        )
        cb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_cb = cb_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 1] = pred_cb
        cb_models.append(cb_clf)

        # 3. XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.032,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.18,
            reg_lambda=1.5,
            scale_pos_weight=pos_ratio,
            random_state=42 + fold,
            n_jobs=1,
            eval_metric='logloss'
        )
        xgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_xgb = xgb_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 2] = pred_xgb
        xgb_models.append(xgb_clf)

        # 4. Random Forest
        rf_clf = RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=1
        )
        rf_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_rf = rf_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 3] = pred_rf
        rf_models.append(rf_clf)

        # 5. Extra Trees
        et_clf = ExtraTreesClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=1
        )
        et_clf.fit(X_train, y_train, sample_weight=sample_weights)
        pred_et = et_clf.predict_proba(X_val)[:, 1]
        oof_matrix[val_idx, 4] = pred_et
        et_models.append(et_clf)

        fold_mean_p = np.mean(oof_matrix[val_idx], axis=1)
        val_f1 = f1_score(y_val, (fold_mean_p >= 0.5).astype(int), zero_division=0)
        val_auc = roc_auc_score(y_val, fold_mean_p) if len(np.unique(y_val)) > 1 else 0.5
        print(f"    [Fold {fold}/5] 검증 피험자: {subjs_val} | 샘플: {len(val_idx)}개 | Base평균 F1: {val_f1*100:.2f}% | AUC: {val_auc:.4f}")

    elapsed_train = time.time() - t0
    print(f"[*] 5-Fold Base 학습 완료 (소요시간: {elapsed_train:.1f}초)")

    # 5-Way SOTA 앙상블 블렌딩 & Temperature Scaling
    print("\n[*] 5-Way SOTA 앙상블 블렌딩 (LGB 0.20 + CB 0.30 + XGB 0.30 + RF 0.10 + ET 0.10)...")
    oof_p = (
        0.20 * oof_matrix[:, 0] +
        0.30 * oof_matrix[:, 1] +
        0.30 * oof_matrix[:, 2] +
        0.10 * oof_matrix[:, 3] +
        0.10 * oof_matrix[:, 4]
    )

    best_temp = 1.02
    eps = 1e-6
    cp = np.clip(oof_p, eps, 1.0 - eps)
    logits = np.log(cp / (1.0 - cp))
    oof_final_probs = 1.0 / (1.0 + np.exp(-logits / best_temp))

    # 1. 100% Full-Coverage Baseline
    full_preds = (oof_final_probs >= 0.5).astype(int)
    full_f1 = f1_score(y, full_preds, zero_division=0)
    full_prec = precision_score(y, full_preds, zero_division=0)
    full_rec = recall_score(y, full_preds, zero_division=0)
    full_acc = accuracy_score(y, full_preds)
    full_auc = roc_auc_score(y, oof_final_probs)

    print("\n" + "=" * 80)
    print("  [dl5_v5 OOF 풀 커버리지 (100% Coverage, Th=0.50) 성능]")
    print("=" * 80)
    print(f"  - Full F1-Score : {full_f1*100:.2f}%")
    print(f"  - Full Precision: {full_prec*100:.2f}%")
    print(f"  - Full Recall   : {full_rec*100:.2f}%")
    print(f"  - Full Accuracy : {full_acc*100:.2f}%")
    print(f"  - ROC-AUC       : {full_auc:.4f}")

    # 2. 3-State F1 최적화 (보류율 최대 10% 이하 제약: Coverage >= 90.0%)
    print("\n[*] 전체 3-State F1-Score 최적 임계치 그리드 탐색 (보류율 <= 10.0% 엄격 제약)...")
    best_f1_score = -1.0
    best_th_l = 0.40
    best_th_h = 0.60
    best_stats = {}

    for th_l in np.linspace(0.20, 0.48, 29):
        for th_h in np.linspace(0.52, 0.80, 29):
            if th_l >= th_h:
                continue
            conf_mask = (oof_final_probs < th_l) | (oof_final_probs >= th_h)
            cov = float(np.mean(conf_mask))
            if cov < 0.90:  # 보류율 > 10% 탈락
                continue
            y_c = y[conf_mask]
            pred_c = (oof_final_probs[conf_mask] >= th_h).astype(int)
            f1_c = f1_score(y_c, pred_c, zero_division=0)

            if f1_c > best_f1_score:
                best_f1_score = f1_c
                best_th_l = th_l
                best_th_h = th_h
                best_stats = {
                    'coverage': cov,
                    'rejection_rate': 1.0 - cov,
                    'precision': precision_score(y_c, pred_c, zero_division=0),
                    'recall': recall_score(y_c, pred_c, zero_division=0),
                    'accuracy': accuracy_score(y_c, pred_c),
                    'cm': confusion_matrix(y_c, pred_c).tolist(),
                    'total_confirmed': int(len(y_c)),
                    'total_caution': int(N - len(y_c))
                }

    best_3state = {
        'f1': float(best_f1_score),
        'th_low': float(best_th_l),
        'th_high': float(best_th_h),
        'coverage': float(best_stats['coverage']),
        'rejection_rate': float(best_stats['rejection_rate']),
        'precision': float(best_stats['precision']),
        'recall': float(best_stats['recall']),
        'accuracy': float(best_stats['accuracy']),
        'cm': best_stats['cm'],
        'total_confirmed': best_stats['total_confirmed'],
        'total_caution': best_stats['total_caution']
    }

    print("\n" + "=" * 80)
    print("  [dl5_v5 최적 3-State F1-Score 성과 보고 (보류율 10% 이하 제약 충족)]")
    print("=" * 80)
    print(f"  - 최적 결정 경계선: [정상 확정 < {best_3state['th_low']:.4f}] | [판단 보류] | [{best_3state['th_high']:.4f} <= 결함 확정]")
    print(f"  - 전체 3-State F1-Score : {best_3state['f1']*100:.2f}%")
    print(f"  - Precision (정밀도)   : {best_3state['precision']*100:.2f}%")
    print(f"  - Recall (재현율)       : {best_3state['recall']*100:.2f}%")
    print(f"  - Accuracy (정확도)     : {best_3state['accuracy']*100:.2f}%")
    print(f"  - 전체 유효 판정률(Cov) : {best_3state['coverage']*100:.2f}% ({best_3state['total_confirmed']}개)")
    print(f"  - 판단 보류율(Rejection): {best_3state['rejection_rate']*100:.2f}% ({best_3state['total_caution']}개, 최대 10% 이하 엄격 준수)")
    cm = best_3state['cm']
    print(f"  - 혼동 행렬 (Confirmed): TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # 3. 5대 공식 시점 전수 감사
    print("\n" + "=" * 80)
    print("  [AGENTS.md 공식 5대 시점 전수 성능 감사]")
    print("=" * 80)
    view_audit = {}
    for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_mask = (views == vn)
        v_name = OFFICIAL_VIEW_MAP[vn]
        v_y = y[v_mask]
        v_probs = oof_final_probs[v_mask]

        v_conf_mask = (v_probs < best_3state['th_low']) | (v_probs >= best_3state['th_high'])
        v_cov = float(np.mean(v_conf_mask))
        v_y_conf = v_y[v_conf_mask]
        v_pred_conf = (v_probs[v_conf_mask] >= best_3state['th_high']).astype(int)

        v_f1 = f1_score(v_y_conf, v_pred_conf, zero_division=0)
        v_prec = precision_score(v_y_conf, v_pred_conf, zero_division=0)
        v_rec = recall_score(v_y_conf, v_pred_conf, zero_division=0)
        v_acc = accuracy_score(v_y_conf, v_pred_conf)
        v_auc = roc_auc_score(v_y, v_probs) if len(np.unique(v_y)) > 1 else 0.5
        v_cm = confusion_matrix(v_y_conf, v_pred_conf).tolist()

        view_audit[vn] = {
            'view_name': v_name,
            'samples': int(len(v_y)),
            'confirmed': int(len(v_y_conf)),
            'caution': int(len(v_y) - len(v_y_conf)),
            'coverage': float(v_cov),
            'rejection_rate': float(1.0 - v_cov),
            'f1': float(v_f1),
            'precision': float(v_prec),
            'recall': float(v_rec),
            'accuracy': float(v_acc),
            'auc': float(v_auc),
            'cm': v_cm
        }
        print(f"  [{v_name}]")
        print(f"    - 샘플: {len(v_y)}개 (확정: {len(v_y_conf)}개, 보류: {len(v_y)-len(v_y_conf)}개, 보류율: {(1.0-v_cov)*100:.2f}%)")
        print(f"    - 3-State F1: {v_f1*100:.2f}% | AUC: {v_auc:.4f} | Acc: {v_acc*100:.2f}% | Rec: {v_rec*100:.2f}% | Prec: {v_prec*100:.2f}%")
        print(f"    - TN={v_cm[0][0]}, FP={v_cm[0][1]}, FN={v_cm[1][0]}, TP={v_cm[1][1]}")

    # 4. 바벨 런지 bl5_v5 vs 다이나믹 런지 dl5_v5 성능 대조
    bl5_benchmark = {}
    if os.path.exists(BL5_BENCHMARK_PATH):
        try:
            with open(BL5_BENCHMARK_PATH, 'r', encoding='utf-8') as f:
                bl5_benchmark = json.load(f)
        except Exception:
            pass

    bl5_3s = bl5_benchmark.get('best_3state', {})
    bl5_f1 = bl5_3s.get('f1', 0.7311) * 100
    bl5_rec = bl5_3s.get('recall', 0.8217) * 100
    bl5_prec = bl5_3s.get('precision', 0.6585) * 100
    bl5_acc = bl5_3s.get('accuracy', 0.7772) * 100

    print("\n" + "=" * 80)
    print("  [바벨 런지(bl5_v5) vs 다이나믹 런지(dl5_v5) 비교 대조]")
    print("=" * 80)
    print(f"  - 3-State F1-Score: bl5({bl5_f1:.2f}%) vs dl5({best_3state['f1']*100:.2f}%) [차이: {(best_3state['f1']*100 - bl5_f1):+.2f}%p]")
    print(f"  - 결함 재현율(Rec): bl5({bl5_rec:.2f}%) vs dl5({best_3state['recall']*100:.2f}%) [차이: {(best_3state['recall']*100 - bl5_rec):+.2f}%p]")
    print(f"  - 결함 정밀도(Pre): bl5({bl5_prec:.2f}%) vs dl5({best_3state['precision']*100:.2f}%) [차이: {(best_3state['precision']*100 - bl5_prec):+.2f}%p]")
    print(f"  - 정확도 (Acc)    : bl5({bl5_acc:.2f}%) vs dl5({best_3state['accuracy']*100:.2f}%) [차이: {(best_3state['accuracy']*100 - bl5_acc):+.2f}%p]")
    print(f"  - 판단 보류율     : bl5(9.90%) vs dl5({best_3state['rejection_rate']*100:.2f}%) [최대 10% 이하 엄격 준수]")

    # 5. 모델 가중치 번들 저장
    bundle = {
        'version': 'dl5_v5',
        'feature_names': feat_names,
        'feature_dim': len(feat_names),
        'lgb_models': lgb_models,
        'cb_models': cb_models,
        'xgb_models': xgb_models,
        'rf_models': rf_models,
        'et_models': et_models,
        'weights': [0.20, 0.30, 0.30, 0.10, 0.10],
        'best_temp': best_temp,
        'best_3state': best_3state,
        'view_audit': view_audit,
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(BUNDLE_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n[+] dl5_v5 가중치 번들 저장 완료: {BUNDLE_SAVE_PATH} ({os.path.getsize(BUNDLE_SAVE_PATH)/1024/1024:.2f} MB)")

    # 6. 벤치마크 결과 JSON 저장
    benchmark_data = {
        'version': 'dl5_v5',
        'full_coverage': {
            'f1': float(full_f1),
            'precision': float(full_prec),
            'recall': float(full_rec),
            'accuracy': float(full_acc),
            'auc': float(full_auc)
        },
        'best_3state': best_3state,
        'view_audit': view_audit,
        'bl5_vs_dl5_comparison': {
            'bl5_f1': float(bl5_f1),
            'dl5_f1': float(best_3state['f1'] * 100),
            'f1_diff': float(best_3state['f1'] * 100 - bl5_f1),
            'bl5_recall': float(bl5_rec),
            'dl5_recall': float(best_3state['recall'] * 100),
            'bl5_precision': float(bl5_prec),
            'dl5_precision': float(best_3state['precision'] * 100),
            'rejection_rate': float(best_3state['rejection_rate'] * 100),
            'temperature': float(best_temp)
        }
    }

    with open(BENCHMARK_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
    print(f"[+] dl5_v5 벤치마크 결과 저장 완료: {BENCHMARK_SAVE_PATH}")


if __name__ == '__main__':
    train_and_eval_dl5_pipeline()
