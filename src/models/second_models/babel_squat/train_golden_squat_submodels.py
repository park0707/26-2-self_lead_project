# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: train_golden_squat_submodels.py
Purpose: 사용자 57선 검증 기반 골든 라벨(Golden Ground Truth)로 4대 결함 서브모델 5-Fold 피험자 격리 재학습
         및 원천 라벨 학습 모델 대비 EMR/F1 성능 향상 실증
"""

import os
import sys
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

GOLDEN_PATH = r"d:\src\models\second_models\babel_squat\squat_golden_ground_truth.npz"


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 골든 라벨 기반 4대 결함 서브모델 5-Fold 피험자 격리 재학습")
    print("=" * 115)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    X = ds['X'].astype(np.float32)
    subjects = ds['subjects'].astype(np.int32)
    N = len(X)

    gold_data = np.load(GOLDEN_PATH, allow_pickle=True)
    Y_raw = gold_data['Y_defects_raw'].astype(np.int32)
    Y_golden = gold_data['Y_defects_golden'].astype(np.int32)
    defect_names = gold_data['defect_names']

    # 2. 피험자 격리 5-Fold 분할 (StratifiedGroupKFold on subjects)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, Y_golden[:, 0], groups=subjects))

    # OOF 예측 확률 저장 배열
    oof_raw_model = np.zeros((N, 4), dtype=np.float32)
    oof_golden_model = np.zeros((N, 4), dtype=np.float32)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    print("\n[*] 4대 결함별 5-Fold 피험자 격리 재학습 진행 중...")
    for k, d in enumerate(defect_names):
        print(f"\n  >> [{d}] 재학습 시작...")
        for fold_idx, (trn_idx, val_idx) in enumerate(folds):
            X_trn, X_val = X[trn_idx], X[val_idx]

            # [A] 원천 라벨 모델 학습
            y_trn_raw = Y_raw[trn_idx, k]
            clf_raw = lgb.LGBMClassifier(**lgb_params)
            clf_raw.fit(X_trn, y_trn_raw)
            oof_raw_model[val_idx, k] = clf_raw.predict_proba(X_val)[:, 1]

            # [B] 골든 정답 라벨 모델 학습
            y_trn_gold = Y_golden[trn_idx, k]
            clf_gold = lgb.LGBMClassifier(**lgb_params)
            clf_gold.fit(X_trn, y_trn_gold)
            oof_golden_model[val_idx, k] = clf_gold.predict_proba(X_val)[:, 1]

    # 3. 성능 비교 평가 (골든 참값 라벨 기준 평가)
    thresholds = [0.50, 0.50, 0.50, 0.50]
    Y_pred_raw = (oof_raw_model >= 0.50).astype(int)
    Y_pred_gold = (oof_golden_model >= 0.50).astype(int)

    print("\n" + "=" * 115)
    print("  [결과 비교] 원천 라벨 학습 모델 vs 골든 라벨 재학습 모델 (Ground Truth = 골든 라벨)")
    print("=" * 115)
    print(f"{'결함 항목':<12} | {'원천학습 Precision':>16} {'골든학습 Precision':>16} | {'원천학습 Recall':>14} {'골든학습 Recall':>14} | {'원천학습 F1':>12} {'골든학습 F1':>12} {'F1 변화폭':>10}")
    print("-" * 115)

    for k, d in enumerate(defect_names):
        p_r, r_r, f_r, _ = precision_recall_fscore_support(Y_golden[:, k], Y_pred_raw[:, k], average='binary', zero_division=0)
        p_g, r_g, f_g, _ = precision_recall_fscore_support(Y_golden[:, k], Y_pred_gold[:, k], average='binary', zero_division=0)
        diff = f_g - f_r
        print(f"{d:<14} | {p_r*100:>15.2f}% {p_g*100:>15.2f}% | {r_r*100:>13.2f}% {r_g*100:>13.2f}% | {f_r*100:>11.2f}% {f_g*100:>11.2f}% {diff*100:>+9.2f}%p")

    print("-" * 115)
    # 완전 정상 (Clean Rep) F1 비교
    y_clean_true = (np.sum(Y_golden, axis=1) == 0).astype(int)
    y_clean_pred_raw = (np.sum(Y_pred_raw, axis=1) == 0).astype(int)
    y_clean_pred_gold = (np.sum(Y_pred_gold, axis=1) == 0).astype(int)

    p_cr, r_cr, f_cr, _ = precision_recall_fscore_support(y_clean_true, y_clean_pred_raw, average='binary', zero_division=0)
    p_cg, r_cg, f_cg, _ = precision_recall_fscore_support(y_clean_true, y_clean_pred_gold, average='binary', zero_division=0)
    print(f"{'정상(Clean)':<14} | {p_cr*100:>15.2f}% {p_cg*100:>15.2f}% | {r_cr*100:>13.2f}% {r_cg*100:>13.2f}% | {f_cr*100:>11.2f}% {f_cg*100:>11.2f}% {(f_cg-f_cr)*100:>+9.2f}%p")

    print("-" * 115)
    emr_raw = np.mean(np.all(Y_golden == Y_pred_raw, axis=1)) * 100
    emr_gold = np.mean(np.all(Y_golden == Y_pred_gold, axis=1)) * 100
    print(f"★ 4대 라벨 100% 완전 일치율 (EMR): {emr_raw:.2f}%  ->  {emr_gold:.2f}%  ({emr_gold - emr_raw:+5.2f}%p 상승!)")

    diff_raw = np.sum(Y_golden != Y_pred_raw, axis=1)
    diff_gold = np.sum(Y_golden != Y_pred_gold, axis=1)
    print(f"  * 1개 이하 오류율 (3개 이상 일치): {np.mean(diff_raw <= 1)*100:.2f}%  ->  {np.mean(diff_gold <= 1)*100:.2f}%")
    print("=" * 115)


if __name__ == '__main__':
    main()
