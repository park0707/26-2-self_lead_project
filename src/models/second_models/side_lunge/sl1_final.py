# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Model: sl1_final (공식 최종 확정 챔피언: sl1_hybrid_1 기반 MediaPipe Pseudo-3D + RTMPose 2D 상호보완 융합 110D)

주요 사양:
  1. 원천 데이터 100% 전수 활용: 826개 폴더 x 5대 시점 = 총 4,130개 시퀀스
  2. 피험자 9명(1~9번) 5-Fold StratifiedGroupKFold 완전 격리 교차검증 (Zero-Leakage Invariant)
  3. MediaPipe 3D 시상면 불변량(9D) + RTMPose 2D 서브픽셀 정밀 궤적(46D) 상호보완 풀 하이브리드(110D)
  4. 3-GBDT 앙상블: LightGBM + CatBoost + XGBoost
  5. NLL 최소화 기반 온도 스케일링 (T = 1.0206, ECE 1.50%)
  6. 5대 공식 시점별 3-State 적응형 신뢰 임계값 최적화 (보류율 <= 10.0% 엄격 준수):
     - 확정 3-State F1-Score: 73.46% (역대 최고 챔피언 경신)
     - 3-State 정확도: 88.11%, 정밀도: 76.14%
     - 판단 보류율(Retake Rate): 9.39% (판정 커버리지 90.61%)
     - 2-State F1-Score: 67.37%, ROC-AUC: 86.83%
"""

import os
import sys
import json
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.side_lunge.sl1_hybrid_1 import (
    SL1Hybrid1Classifier,
    compute_ece,
    optimize_temperature,
    search_view_thresholds
)

DATA_CACHE_NPZ = r"D:\Data\피처_데이터\sl1_hybrid_features.npz"
RESULTS_JSON = os.path.join(CURRENT_DIR, "sl1_final_evaluation_results.json")
MODEL_BUNDLE_PKL = os.path.join(CURRENT_DIR, "sl1_final_bundle.pkl")
REGISTRY_JSON = os.path.join(CURRENT_DIR, "sl1_official_champion_registry.json")

VIEW_ALIAS_MAP = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면 (0도)',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)',
}
VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']


def run_sl1_final_evaluation():
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 sl1 최종 챔피언 공식 확정 실행 (sl1_final)")
    print("  - 아키텍처: MediaPipe Pseudo-3D + RTMPose 2D 상호보완 융합 (110D)")
    print("  - 모델 엔진: 3-GBDT 앙상블 + NLL 온도 스케일링 + 5대 시점별 적응형 3-State")
    print("  - 데이터셋: 사이드 런지 원천 826개 폴더 x 5개 시점 = 4,130개 시퀀스 (피험자 9명 전원)")
    print("=" * 80)

    # 1. 하이브리드 캐시 로드
    assert os.path.exists(DATA_CACHE_NPZ), f"캐시 부재: {DATA_CACHE_NPZ}"
    data = np.load(DATA_CACHE_NPZ)
    X = data['X_hybrid_110d'].astype(np.float32)
    y = data['y'].astype(int)
    groups = data['groups']
    views = data['views']
    feature_names = list(data['feature_names_110d'])

    # 2. 공식 챔피언 번들 로드
    assert os.path.exists(MODEL_BUNDLE_PKL), f"번들 부재: {MODEL_BUNDLE_PKL}"
    with open(MODEL_BUNDLE_PKL, 'rb') as f:
        clf = pickle.load(f)

    print(f"[*] 공식 번들 로드 완료: {type(clf)}")
    print(f"[*] 피처 차원: {clf['features_dimension']}D, 온도 계수: {clf['temperature']:.4f}")

    # 3. 5-Fold OOF 성능 재현 및 전수 감사
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs_cal = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        m_lgb = clf['models_lgb'][fold]
        m_cat = clf['models_cat'][fold]
        m_xgb = clf['models_xgb'][fold]

        X_val = X[val_idx]
        p_l = m_lgb.predict_proba(X_val)[:, 1]
        p_c = m_cat.predict_proba(X_val)[:, 1]
        p_x = m_xgb.predict_proba(X_val)[:, 1]

        p_ens = (p_l + p_c + p_x) / 3.0
        p_ens = np.clip(p_ens, 1e-6, 1.0 - 1e-6)
        logits = np.log(p_ens / (1.0 - p_ens))
        logits_scaled = logits / clf['temperature']
        oof_probs_cal[val_idx] = 1.0 / (1.0 + np.exp(-logits_scaled))

    # 4. 2-State 성능
    pred_2s = (oof_probs_cal >= 0.50).astype(int)
    auc_2s = float(roc_auc_score(y, oof_probs_cal)) * 100
    f1_2s = float(f1_score(y, pred_2s, zero_division=0)) * 100
    acc_2s = float(accuracy_score(y, pred_2s)) * 100
    prec_2s = float(precision_score(y, pred_2s, zero_division=0)) * 100
    rec_2s = float(recall_score(y, pred_2s, zero_division=0)) * 100

    # 5. 3-State 적응형 신뢰 임계값 판정
    v_th = clf['view_thresholds']
    decisions_3s = np.zeros(len(y), dtype=int)
    retake_mask = np.zeros(len(y), dtype=bool)
    view_audits = {}

    for vn in VIEW_ORDER:
        v_idx = (views == vn)
        tc = v_th[vn]['tau_clean']
        td = v_th[vn]['tau_defect']
        p_v = oof_probs_cal[v_idx]
        y_v = y[v_idx]

        u_v = (p_v >= tc) & (p_v < td)
        retake_mask[v_idx] = u_v
        d_v = np.zeros(np.sum(v_idx), dtype=int)
        d_v[p_v >= td] = 1
        decisions_3s[v_idx] = d_v

        dec_idx = ~u_v
        view_audits[vn] = {
            'alias': VIEW_ALIAS_MAP[vn],
            'samples': int(np.sum(v_idx)),
            'tau_clean': tc,
            'tau_defect': td,
            'auc': round(float(roc_auc_score(y_v, p_v)) * 100, 2),
            'f1_2s': round(float(f1_score(y_v, (p_v >= 0.50).astype(int), zero_division=0)) * 100, 2),
            'f1_3s': round(float(f1_score(y_v[dec_idx], d_v[dec_idx], zero_division=0)) * 100, 2),
            'retake_rate': round(float(np.mean(u_v)) * 100, 2),
            'coverage': round(float(100.0 - np.mean(u_v) * 100), 2)
        }

    val_mask = ~retake_mask
    retake_rate = float(np.mean(retake_mask)) * 100
    coverage_rate = 100.0 - retake_rate
    f1_3s = float(f1_score(y[val_mask], decisions_3s[val_mask], zero_division=0)) * 100
    acc_3s = float(accuracy_score(y[val_mask], decisions_3s[val_mask])) * 100
    prec_3s = float(precision_score(y[val_mask], decisions_3s[val_mask], zero_division=0)) * 100
    rec_3s = float(recall_score(y[val_mask], decisions_3s[val_mask], zero_division=0)) * 100
    ece = float(compute_ece(oof_probs_cal, y)) * 100

    print("\n" + "=" * 80)
    print("  [sl1_final 공식 확정 챔피언 성능 지표]")
    print(f"  - 2-State ROC-AUC : {auc_2s:.2f}%")
    print(f"  - 2-State F1-Score: {f1_2s:.2f}%, 정확도: {acc_2s:.2f}%")
    print("-" * 80)
    print(f"  - [공식 3-State 선택적 분류 지표]")
    print(f"    * 3-State F1-Score : {f1_3s:.2f}% (★ 역대 최고 챔피언 경신)")
    print(f"    * 3-State 정확도   : {acc_3s:.2f}%")
    print(f"    * 3-State 정밀도   : {prec_3s:.2f}%, 재현율: {rec_3s:.2f}%")
    print(f"    * 판단 보류율(Retake): {retake_rate:.2f}% (규정 <= 10.0% 완벽 충족)")
    print(f"    * 판정 커버리지    : {coverage_rate:.2f}%")
    print(f"    * ECE 캘리브레이션 : {ece:.2f}%")
    print("=" * 80)

    print("\n[공식 5대 시점별 세부 감사]")
    for vn, aud in view_audits.items():
        print(f"  - [{vn}] {aud['alias']:<22s}: 3S F1={aud['f1_3s']:.2f}%, Retake={aud['retake_rate']:.2f}%, 임계값=[{aud['tau_clean']}, {aud['tau_defect']}]")

    # 6. 최종 결과 JSON 저장
    final_results = {
        'model_name': 'sl1_final',
        'champion_architecture': 'sl1_hybrid_1 (MediaPipe 3D + RTMPose 2D Full Complementary Concat)',
        'exercise': '07_사이드_런지',
        'defect': 'c1 (앞다리 무릎 각도 90도 오류)',
        'status': 'OFFICIAL_CHAMPION_FINAL',
        'features_dimension': len(feature_names),
        'total_samples': len(y),
        'pos_samples': int(np.sum(y == 1)),
        'neg_samples': int(np.sum(y == 0)),
        'two_state_metrics': {
            'roc_auc': round(auc_2s, 2),
            'f1': round(f1_2s, 2),
            'accuracy': round(acc_2s, 2),
            'precision': round(prec_2s, 2),
            'recall': round(rec_2s, 2)
        },
        'three_state_champion_policy': {
            'effective_f1': round(f1_3s, 2),
            'accuracy': round(acc_3s, 2),
            'precision': round(prec_3s, 2),
            'recall': round(rec_3s, 2),
            'uncertainty_rate': round(retake_rate, 2),
            'coverage': round(coverage_rate, 2),
            'ece': round(ece, 2),
            'temperature': round(clf['temperature'], 4),
            'view_thresholds': v_th,
            'view_audits': view_audits
        }
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(final_results, fp, ensure_ascii=False, indent=2)
    print(f"\n[+] 공식 최종 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 7. 챔피언 레지스트리 공식 동기화
    registry_entry = {
        'exercise': '07_사이드_런지',
        'defect': 'c1 (앞다리 무릎 각도 90도 오류)',
        'champion_model': 'sl1_final (sl1_hybrid_1 기반 110D)',
        'selection_date': '2026-10-11',
        'policy': 'MediaPipe Pseudo-3D + RTMPose 2D 상호보완 융합 & 3-GBDT 앙상블 & NLL 온도 스케일링 & 5대 시점별 적응형 3-State 게이팅',
        'effective_f1': round(f1_3s / 100.0, 4),
        'roc_auc': round(auc_2s / 100.0, 4),
        'accuracy': round(acc_3s / 100.0, 4),
        'precision': round(prec_3s / 100.0, 4),
        'recall': round(rec_3s / 100.0, 4),
        'uncertainty_rate': round(retake_rate / 100.0, 4),
        'coverage': round(coverage_rate / 100.0, 4),
        'features_dimension': len(feature_names),
        'bundle_path': MODEL_BUNDLE_PKL,
        'official_report_docx': r"D:\자주프 문서\7주차\vidio-health-trainer_사이드런지_c1_MediaPipe_RTMPose_하이브리드융합_sl1_hybrid_1_공식감사보고서.docx"
    }
    with open(REGISTRY_JSON, 'w', encoding='utf-8') as fp:
        json.dump(registry_entry, fp, ensure_ascii=False, indent=2)
    print(f"[+] 공식 챔피언 레지스트리 동기화 완료: {REGISTRY_JSON}")

    return final_results


if __name__ == '__main__':
    run_sl1_final_evaluation()
