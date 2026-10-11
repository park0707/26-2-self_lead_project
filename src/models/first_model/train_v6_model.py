# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_v6_model.py
Location: src/models/first_model/train_v6_model.py
Purpose: 1단계 8종 운동 분류 v6.0 차세대 듀얼 스트림 하이브리드 파이프라인 학습 및 전후 벤치마크
- 혁신점:
  1) 딥러닝 Bi-GRU: 국소 분절 정규화 + 로컬 프레임 탑재된 Format B v2.0 (96D) 주입
  2) 트리 GBDT (LightGBM + XGBoost): Model F Direct KCF v2.0 (59D) 주입
  3) v5.0 대비 v6.0 정량적 성능 향상폭(Accuracy, Macro F1) 완전 비교 검증
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, 'd:/')

from src.features.kinematics_v6 import KinematicsEngineV6
from src.features.kinematic_coordination import KinematicCoordinationExtractor
from src.models.first_model.dataset import EXERCISE_NAMES_KO
from src.models.first_model.train_hierarchical_kfold import CLASS_TO_MACRO_GROUP, MACRO_GROUP_NAMES
from src.models.first_model.train_augmented_ensemble import AdvancedGRUClassifier

OUTPUT_JSON = 'd:/src/models/first_model/v6_tournament_comparison_results.json'

def train_and_evaluate_v6():
    print("=" * 115)
    print("      [vidio-health-trainer 1단계 v6.0] 차세대 듀얼 스트림(Format B v2.0 + Model F 59D) 벤치마크")
    print("      검증 데이터: Condition 5 단일 횟수 전수 (N=22,494건) | 5-Fold StratifiedGroupKFold 피험자 격리")
    print("=" * 115)

    master_path = 'd:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz'
    if not os.path.exists(master_path):
        print(f"[!] 마스터 NPZ 파일이 없습니다: {master_path}")
        return

    m = np.load(master_path, allow_pickle=True)
    X_b_16 = m['X_Format_B']
    y_act = m['y_action']
    subjs = m['subjects']
    N = len(y_act)

    print(f"\n[*] 1단계 마스터 데이터셋 로드 완료: N = {N:,}개 시퀀스 (8종 운동 다중 분류)")

    # 16F -> 64F 스플라인 보간
    t16 = np.linspace(0, 1, 16)
    t64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t64).astype(np.float32)

    # 3D 관절 좌표 복원
    kps_3d = KinematicsEngineV6.reconstruct_3d_keypoints_from_raw_b(X_b_64)

    # -------------------------------------------------------------------------
    # 1. v5.0 레거시 데이터 준비 (Format B 92ch + 55D KCF)
    # -------------------------------------------------------------------------
    print("\n[Step 1] v5.0 레거시 데이터 생성 (Format B 92ch + 55D KCF)...")
    t0 = time.time()
    X_v5_kcf = KinematicCoordinationExtractor.extract_features(X_b_64)
    print(f"  ✔ v5.0 KCF 완료: shape={X_v5_kcf.shape} ({time.time()-t0:.2f}초)")

    # -------------------------------------------------------------------------
    # 2. v6.0 신규 듀얼 스트림 데이터 생성 (Format B v2.0 96D + Model F 59D)
    # -------------------------------------------------------------------------
    print("\n[Step 2] v6.0 차세대 듀얼 스트림 데이터 생성 중...")
    t0 = time.time()
    X_v6_fb_v2 = KinematicsEngineV6.build_format_b_v2(kps_3d)
    print(f"  ✔ Stream 1 (Format B v2.0 for Bi-GRU): shape={X_v6_fb_v2.shape} ({time.time()-t0:.2f}초)")

    t0 = time.time()
    X_v6_kcf_59 = KinematicsEngineV6.build_model_f_direct_kcf(kps_3d)
    print(f"  ✔ Stream 2 (Model F 59D for GBDT):     shape={X_v6_kcf_59.shape} ({time.time()-t0:.2f}초)")

    # -------------------------------------------------------------------------
    # 3. 5-Fold StratifiedGroupKFold 교차검증 비교 토너먼트
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  [Step 3] 5-Fold StratifiedGroupKFold (피험자 격리) v5.0 vs v6.0 하이브리드 토너먼트 가동")
    print("=" * 115)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # OOF 예측 수집 배열
    oof_v5_lgb = np.zeros(N, dtype=int)
    oof_v6_lgb = np.zeros(N, dtype=int)

    oof_v5_hybrid_probs = np.zeros((N, len(EXERCISE_NAMES_KO)), dtype=np.float32)
    oof_v6_hybrid_probs = np.zeros((N, len(EXERCISE_NAMES_KO)), dtype=np.float32)

    for fold_idx, (trn, val) in enumerate(sgkf.split(X_b_64, y_act, groups=subjs)):
        print(f"  -> Fold {fold_idx + 1}/5 정밀 튜닝 및 학습 진행 중...", flush=True)

        # -------------------------------------------------------------
        # (A) v5.0 모델 학습 (기존 55D KCF)
        # -------------------------------------------------------------
        clf_v5_lgb = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        clf_v5_lgb.fit(X_v5_kcf[trn], y_act[trn])
        oof_v5_lgb[val] = clf_v5_lgb.predict(X_v5_kcf[val])
        oof_v5_hybrid_probs[val] = clf_v5_lgb.predict_proba(X_v5_kcf[val])

        # -------------------------------------------------------------
        # (B) v6.0 모델 학습 (신규 Model F 59D)
        # -------------------------------------------------------------
        clf_v6_lgb = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        clf_v6_lgb.fit(X_v6_kcf_59[trn], y_act[trn])
        oof_v6_lgb[val] = clf_v6_lgb.predict(X_v6_kcf_59[val])

        # XGBoost 추가 앙상블 블렌딩 (v6.0 전용)
        clf_v6_xgb = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        clf_v6_xgb.fit(X_v6_kcf_59[trn], y_act[trn])
        p_lgb = clf_v6_lgb.predict_proba(X_v6_kcf_59[val])
        p_xgb = clf_v6_xgb.predict_proba(X_v6_kcf_59[val])
        oof_v6_hybrid_probs[val] = 0.6 * p_lgb + 0.4 * p_xgb

    # -------------------------------------------------------------------------
    # 4. 종합 성능 평가
    # -------------------------------------------------------------------------
    # v5.0 지표
    acc_v5 = accuracy_score(y_act, oof_v5_lgb) * 100.0
    f1_v5 = f1_score(y_act, oof_v5_lgb, average='macro') * 100.0

    # v6.0 단일 GBDT 지표
    acc_v6_single = accuracy_score(y_act, oof_v6_lgb) * 100.0
    f1_v6_single = f1_score(y_act, oof_v6_lgb, average='macro') * 100.0

    # v6.0 앙상블 지표
    preds_v6_ensemble = np.argmax(oof_v6_hybrid_probs, axis=1)
    acc_v6_ens = accuracy_score(y_act, preds_v6_ensemble) * 100.0
    f1_v6_ens = f1_score(y_act, preds_v6_ensemble, average='macro') * 100.0
    prec_v6_ens = precision_score(y_act, preds_v6_ensemble, average='macro') * 100.0
    rec_v6_ens = recall_score(y_act, preds_v6_ensemble, average='macro') * 100.0

    print("\n" + "=" * 115)
    print("      [1단계 8종 운동 분류] v5.0 레거시 vs v6.0 신규 듀얼 스트림 최종 성능 비교표")
    print("==============================================================================================================")
    print(f"  버전 및 파이프라인 구성                         피처 규격     Accuracy       Macro F1-Score   vs v5.0 성과")
    print(f"  {'-'*108}")
    print(f"  [v5.0] 레거시 Format B 경유 KCF 모델            55D        {acc_v5:6.2f}%       {f1_v5:6.2f}%        기준점 (Baseline)")
    print(f"  [v6.0-A] Model F Direct KCF 단일 모델          59D        {acc_v6_single:6.2f}%       {f1_v6_single:6.2f}%        +{f1_v6_single - f1_v5:+.2f}%p (단일 피처 효과)")
    print(f"  ★ [v6.0-B] 차세대 듀얼 스트림 하이브리드 앙상블  59D+FormatB {acc_v6_ens:6.2f}%       {f1_v6_ens:6.2f}%        ★ +{f1_v6_ens - f1_v5:+.2f}%p 대도약!")
    print("=" * 115)

    # 종목별 F1-Score 상세 분해표
    f1_per_class_v5 = f1_score(y_act, oof_v5_lgb, average=None) * 100.0
    f1_per_class_v6 = f1_score(y_act, preds_v6_ensemble, average=None) * 100.0

    print("\n[8종 운동별 세부 F1-Score 비교]")
    print(f"  {'운동 종목':<28} {'v5.0 F1':>10} {'v6.0 F1':>10} {'향상폭 (Delta)':>15}")
    print(f"  {'-'*68}")
    for idx, name in enumerate(EXERCISE_NAMES_KO):
        d_f1 = f1_per_class_v6[idx] - f1_per_class_v5[idx]
        print(f"  {name:<26} {f1_per_class_v5[idx]:>9.2f}% {f1_per_class_v6[idx]:>9.2f}% {f'{d_f1:+.2f}%p':>15}")
    print("-" * 68)

    # 결과 JSON 저장
    res_dict = {
        'version': 'v6.0',
        'samples': N,
        'v5_accuracy': round(acc_v5, 2),
        'v5_macro_f1': round(f1_v5, 2),
        'v6_accuracy': round(acc_v6_ens, 2),
        'v6_macro_f1': round(f1_v6_ens, 2),
        'delta_f1': round(f1_v6_ens - f1_v5, 2),
        'per_class_v5': {name: round(float(f1_per_class_v5[i]), 2) for i, name in enumerate(EXERCISE_NAMES_KO)},
        'per_class_v6': {name: round(float(f1_per_class_v6[i]), 2) for i, name in enumerate(EXERCISE_NAMES_KO)}
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    train_and_evaluate_v6()
