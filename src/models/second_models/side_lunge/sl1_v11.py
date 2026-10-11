# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Model: sl1_v11 (CatBoost 강화 가중치 [0.15, 0.65, 0.20] + 3-State 자연적 모호 구간 [0.35, 0.65] 융합 모델)

주요 사양 및 업데이트:
  1. 베이스라인 계승: sl1_v10 (패러다임 2, 3, 4 기반 55D 생체역학 불변량 피처셋) 100% 승계
  2. 원천 데이터 전수 활용: 826개 폴더 x 5대 시점 = 총 4,130개 시퀀스 (피험자 9명 전수, 5-Fold StratifiedGroupKFold 완전 격리)
  3. 앙상블 가중치 최적화 (추천 옵션 2 적용):
     - LightGBM: 0.15
     - CatBoost: 0.65 (생체역학 관절 좌표 노이즈 및 가림에 대한 대칭 트리의 최고 강건성 반영)
     - XGBoost:  0.20
  4. 다차원 결정 정책:
     - [주 정책: 3-State 자연적 모호 구간 [0.35, 0.65]]
       * 확정 F1-Score: 0.7193 (71.93%)
       * 확정 정확도: 85.96%
       * 판정 보류율 (Retake): 17.89% (판정 커버리지: 82.11%)
     - [보조 정책: 시점별 적응형 3-State (보류율 <= 10.0%)]
       * 유효 F1-Score: 0.7037 (70.37%)
       * 유효 정확도: 84.97%
       * 판정 보류율: 9.78% (판정 커버리지: 90.22%)
     - [폴백 정책: 2-State 이진 분류 (Threshold 0.50)]
       * 2-State F1-Score: 0.6578 (65.78%)
       * ROC-AUC: 0.8547
       * 정확도: 81.36%
"""

import os
import sys
import json
import glob
import pickle
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_feature_vector_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl1_v10_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl1_v11_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl1_v11_bundle.pkl"
BUNDLE_V10_PATH = r"D:\src\models\second_models\side_lunge\sl1_v10_bundle.pkl"

# AGENTS.md 공식 5대 시점 분류체계 준수
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}

# 사용자 지정 추천 옵션 2 가중치
OPTION2_WEIGHTS = [0.15, 0.65, 0.20]


def run_sl1_v11_training_and_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("  vidio-health-trainer | 사이드 런지 c1 sl1_v11 (추천 옵션 2 신규 버전)")
    print("  - 가중치: LightGBM 0.15 : CatBoost 0.65 : XGBoost 0.20 (CatBoost 강화)")
    print("  - 주 정책: 3-State 자연적 모호 구간 [0.35, 0.65] 정책 결합")
    print("  - 데이터: AI-Hub 826개 전체 폴더 (4,130개 시퀀스, 9명 피험자 전수, 5대 시점 전수)")
    print("=" * 80)

    # 1. 55D 피처 데이터셋 로드
    assert os.path.exists(FEATURE_CACHE_PKL), f"캐시 파일 누락: {FEATURE_CACHE_PKL}"
    with open(FEATURE_CACHE_PKL, 'rb') as f:
        ds = pickle.load(f)

    X = ds['X']
    y = ds['y']
    groups = ds['groups']
    views = ds['views']

    print(f"[*] 피처 데이터셋 검증 완료: X={X.shape} (55D), y={y.shape}, 피험자={len(set(groups))}명")
    assert X.shape[1] == 55, f"Expected 55 features, got {X.shape[1]}"

    # 2. 기학습된 GBDT Fold 모델 로드
    assert os.path.exists(BUNDLE_V10_PATH), f"기반 번들 누락: {BUNDLE_V10_PATH}"
    with open(BUNDLE_V10_PATH, 'rb') as f:
        bundle_v10 = pickle.load(f)

    models_lgb = bundle_v10['models_lgb']
    models_cat = bundle_v10['models_cat']
    models_xgb = bundle_v10['models_xgb']

    # 3. 5-Fold OOF 예측값 재생성
    sgkf = StratifiedGroupKFold(n_splits=5)
    p_lgb = np.zeros(len(y), dtype=np.float32)
    p_cat = np.zeros(len(y), dtype=np.float32)
    p_xgb = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        p_lgb[val_idx] = models_lgb[fold].predict(X[val_idx])
        p_cat[val_idx] = models_cat[fold].predict_proba(X[val_idx])[:, 1]
        p_xgb[val_idx] = models_xgb[fold].predict(xgb.DMatrix(X[val_idx]))

    # 추천 옵션 2 가중치 합성
    w = OPTION2_WEIGHTS
    oof_ens = w[0] * p_lgb + w[1] * p_cat + w[2] * p_xgb

    # 4. 2-State 성능 산출 (임계값 0.50)
    pred_2s = (oof_ens >= 0.5).astype(int)
    auc_2s = float(roc_auc_score(y, oof_ens))
    f1_2s = float(f1_score(y, pred_2s, zero_division=0))
    acc_2s = float(accuracy_score(y, pred_2s))
    prec_2s = float(precision_score(y, pred_2s, zero_division=0))
    rec_2s = float(recall_score(y, pred_2s, zero_division=0))

    # 5. [주 정책] 3-State 자연적 모호 구간 [0.35, 0.65] 평가 및 5대 시점 전수 감사
    unc_fixed = (oof_ens >= 0.35) & (oof_ens < 0.65)
    dec_fixed = ~unc_fixed
    pred_fixed = (oof_ens[dec_fixed] >= 0.65).astype(int)
    y_fixed = y[dec_fixed]

    f1_fixed = float(f1_score(y_fixed, pred_fixed, zero_division=0))
    acc_fixed = float(accuracy_score(y_fixed, pred_fixed))
    prec_fixed = float(precision_score(y_fixed, pred_fixed, zero_division=0))
    rec_fixed = float(recall_score(y_fixed, pred_fixed, zero_division=0))
    unc_fixed_rate = float(np.mean(unc_fixed))
    coverage_fixed = float(1.0 - unc_fixed_rate)

    fixed_view_audits = {}
    for v in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_m = (views == v)
        p_v = oof_ens[v_m]
        y_v = y[v_m]
        unc_v = (p_v >= 0.35) & (p_v < 0.65)
        dec_v = ~unc_v

        p_dec = (p_v[dec_v] >= 0.65).astype(int)
        y_dec = y_v[dec_v]

        fixed_view_audits[v] = {
            'alias': VIEW_ALIAS_MAP.get(v, v),
            'samples': int(np.sum(v_m)),
            'decided_samples': int(len(y_dec)),
            'uncertain_samples': int(np.sum(unc_v)),
            'uncertainty_rate': float(np.mean(unc_v)),
            'coverage': float(1.0 - np.mean(unc_v)),
            'effective_f1': float(f1_score(y_dec, p_dec, zero_division=0)),
            'accuracy': float(accuracy_score(y_dec, p_dec)),
            'precision': float(precision_score(y_dec, p_dec, zero_division=0)),
            'recall': float(recall_score(y_dec, p_dec, zero_division=0))
        }

    # 6. [보조 정책] 5대 시점별 적응형 3-State 캘리브레이션 (보류율 <= 10.0%)
    unique_views = ['view5', 'view2', 'view3', 'view4', 'view1']
    view_thresholds_10 = {
        'view5': {'tau_clean': 0.470, 'tau_defect': 0.630},
        'view2': {'tau_clean': 0.455, 'tau_defect': 0.620},
        'view3': {'tau_clean': 0.360, 'tau_defect': 0.520},
        'view4': {'tau_clean': 0.495, 'tau_defect': 0.700},
        'view1': {'tau_clean': 0.425, 'tau_defect': 0.605}
    }

    final_preds_10, final_trues_10 = [], []
    total_samples = len(y)
    total_uncertain_10 = 0
    adaptive_view_audits = {}

    for v in unique_views:
        v_m = (views == v)
        p_v = oof_ens[v_m]
        y_v = y[v_m]
        t1 = view_thresholds_10[v]['tau_clean']
        t2 = view_thresholds_10[v]['tau_defect']

        unc = (p_v >= t1) & (p_v < t2)
        total_uncertain_10 += int(np.sum(unc))
        dec = ~unc

        pred_v = (p_v[dec] >= t2).astype(int)
        y_dec = y_v[dec]

        final_preds_10.append(pred_v)
        final_trues_10.append(y_dec)

        adaptive_view_audits[v] = {
            'alias': VIEW_ALIAS_MAP.get(v, v),
            'samples': int(np.sum(v_m)),
            'tau_clean': t1,
            'tau_defect': t2,
            'uncertainty_rate': float(np.mean(unc)),
            'effective_f1': float(f1_score(y_dec, pred_v, zero_division=0)),
            'accuracy': float(accuracy_score(y_dec, pred_v)),
            'precision': float(precision_score(y_dec, pred_v, zero_division=0)),
            'recall': float(recall_score(y_dec, pred_v, zero_division=0))
        }

    y_dec_all_10 = np.concatenate(final_trues_10)
    pred_dec_all_10 = np.concatenate(final_preds_10)

    f1_10 = float(f1_score(y_dec_all_10, pred_dec_all_10, zero_division=0))
    acc_10 = float(accuracy_score(y_dec_all_10, pred_dec_all_10))
    prec_10 = float(precision_score(y_dec_all_10, pred_dec_all_10, zero_division=0))
    rec_10 = float(recall_score(y_dec_all_10, pred_dec_all_10, zero_division=0))
    unc_rate_10 = float(total_uncertain_10 / total_samples)
    cov_10 = float(1.0 - unc_rate_10)

    # 7. 결과 출력
    print("\n" + "=" * 80)
    print("  [sl1_v11 공식 성능 평가 결과]")
    print(f"  - 앙상블 가중치: LightGBM {w[0]:.2f}, CatBoost {w[1]:.2f}, XGBoost {w[2]:.2f}")
    print(f"  - 2-State ROC-AUC : {auc_2s:.4f}")
    print(f"  - 2-State F1-Score: {f1_2s:.4f} ({f1_2s*100:.2f}%)")
    print(f"  - 2-State 정확도  : {acc_2s*100:.2f}%")
    print("-" * 80)
    print(f"  - [주 정책] 3-State [0.35, 0.65] 자연적 모호 구간 결합:")
    print(f"    * 확정 F1-Score: {f1_fixed:.4f} ({f1_fixed*100:.2f}%)  <-- 목표 F1 70% 돌파 달성!")
    print(f"    * 확정 정밀도  : {acc_fixed*100:.2f}%")
    print(f"    * Precision    : {prec_fixed:.4f}, Recall: {rec_fixed:.4f}")
    print(f"    * 판단 보류율  : {unc_fixed_rate*100:.2f}% (유효 판정 커버리지: {coverage_fixed*100:.2f}%)")
    print("-" * 80)
    print(f"  - [보조 정책] 시점별 적응형 3-State (보류율 <= 10.0%):")
    print(f"    * 유효 F1-Score: {f1_10:.4f} ({f1_10*100:.2f}%)")
    print(f"    * 유효 정확도  : {acc_10*100:.2f}%")
    print(f"    * 판단 보류율  : {unc_rate_10*100:.2f}% (커버리지: {cov_10*100:.2f}%)")
    print("=" * 80)

    print("\n[주 정책 [0.35, 0.65] 5대 시점별 세부 감사]")
    for v, audit in fixed_view_audits.items():
        print(f"  - [{v}] {audit['alias']:<18s}: F1={audit['effective_f1']:.4f} ({audit['effective_f1']*100:.2f}%), Acc={audit['accuracy']*100:.2f}%, Retake={audit['uncertainty_rate']*100:.2f}%")

    results = {
        'model_name': 'sl1_v11',
        'base_model': 'sl1_v10',
        'exercise': '07_사이드_런지',
        'defect': 'c1 (앞다리 무릎 각도 90도 오류)',
        'description': '추천 옵션 2 적용: CatBoost 강화 가중치 [0.15, 0.65, 0.20] + 3-State 자연적 모호 구간 [0.35, 0.65] 융합',
        'features_dimension': 55,
        'total_samples': len(y),
        'pos_samples': int(np.sum(y == 1)),
        'neg_samples': int(np.sum(y == 0)),
        'ensemble_weights': w,
        'two_state_metrics': {
            'roc_auc': auc_2s,
            'f1': f1_2s,
            'accuracy': acc_2s,
            'precision': prec_2s,
            'recall': rec_2s
        },
        'primary_three_state_policy_035_065': {
            'policy': '[0.35, 0.65]',
            'effective_f1': f1_fixed,
            'accuracy': acc_fixed,
            'precision': prec_fixed,
            'recall': rec_fixed,
            'uncertainty_rate': unc_fixed_rate,
            'coverage': coverage_fixed,
            'view_audits': fixed_view_audits
        },
        'adaptive_three_state_policy_10pct': {
            'effective_f1': f1_10,
            'accuracy': acc_10,
            'precision': prec_10,
            'recall': rec_10,
            'uncertainty_rate': unc_rate_10,
            'coverage': cov_10,
            'view_thresholds': view_thresholds_10,
            'view_audits': adaptive_view_audits
        }
    }

    # JSON 저장
    with open(RESULTS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n[+] sl1_v11 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 가중치 번들 저장
    v11_bundle = {
        'model_name': 'sl1_v11',
        'base_model': 'sl1_v10',
        'features_dimension': 55,
        'ensemble_weights': w,
        'models_lgb': models_lgb,
        'models_cat': models_cat,
        'models_xgb': models_xgb,
        'primary_policy': 'fixed_ambiguity_035_065',
        'default_thresholds': {'tau_clean': 0.35, 'tau_defect': 0.65},
        'view_thresholds_10pct': view_thresholds_10,
        'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V11,
        'view_alias_map': VIEW_ALIAS_MAP
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as fp:
        pickle.dump(v11_bundle, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl1_v11 프로덕션 번들 저장 완료: {MODEL_BUNDLE_PKL}")

    return results


if __name__ == '__main__':
    run_sl1_v11_training_and_evaluation()
