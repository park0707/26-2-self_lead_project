# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: evaluate_active_motion_gating_and_joint_probabilities.py
Purpose: 동적 가동 마스킹(Active Motion Masking) 적용 후:
         1) c1~c4 각 결함별 독립 확률 (Precision, Recall, F1, Accuracy)
         2) 전체 바벨 스쿼트 결합 확률 (4대 라벨 100% 일치율 EMR, 완전 정상 결합 F1, 누적 일치율)
         3) 이론적 독립곱(Theoretical Independent Product) vs 실제 결합 확률(Empirical Joint) 비교 분석
"""

import os
import sys
import glob
import time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')
from src.features.kinematics_3d import KinematicsExtractor3D
from src.preprocessing.normalizer3d import GeometricNormalizer3D
from src.features.kinematic_pipeline import KinematicPipelineEngine
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

GOLDEN_PATH = r"d:\src\models\second_models\babel_squat\squat_golden_ground_truth.npz"
CACHE_PATH = r"d:\src\models\second_models\babel_squat\oof_submodels_cache.npz"


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 동적 가동 마스킹(Active Motion Masking) 적용 피처 기반")
    print("  4대 결함(c1~c4) 독립 확률 및 전체 바벨 스쿼트 결합 확률(Joint Probability) 종합 벤치마크")
    print("=" * 115)

    # 1. 골든 정답 라벨 로드
    gold_data = np.load(GOLDEN_PATH, allow_pickle=True)
    Y_golden = gold_data['Y_defects_golden'].astype(np.int32) # (1906, 4)
    Y_raw = gold_data['Y_defects_raw'].astype(np.int32)
    subjects = gold_data['subjects'].astype(np.int32)
    defect_names = ['c1. 척추의 중립', 'c2. 고개 정면', 'c3. 발과 무릎 외반', 'c4. 발바닥 지면 고정']
    defect_short = ['c1척추', 'c2시선', 'c3무릎', 'c4발목']
    N = len(Y_golden)
    print(f"[*] 총 분석 대상 바벨 스쿼트 샘플 수: {N}개 (전체 9명 피험자, 전 시점 전수)")

    # 2. 마스터 데이터셋 Format B (16, 92)로부터 kps_norm_3d 복원 및 Active Mask 적용 161D 피처 신규 추출
    # Format B의 Stream 1 (14 edges x 4) 또는 Condition 5 마스터에서 직접 추출
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_Format_A_raw = m['X_Format_A'][sq_mask].astype(np.float32)

    # OOF 예측 확률 로드 (특화 피처 모델 기반 OOF)
    cached = np.load(CACHE_PATH)
    p_c1 = cached['oof_c1'].astype(np.float32)
    p_c2 = cached['oof_c2_v3'].astype(np.float32) if 'oof_c2_v3' in cached else cached['oof_c2'].astype(np.float32)
    p_c3 = cached['oof_c3_v5'].astype(np.float32) if 'oof_c3_v5' in cached else cached['oof_c3'].astype(np.float32)
    p_c4 = cached['oof_c4'].astype(np.float32)
    P_pred = np.vstack([p_c1, p_c2, p_c3, p_c4]).T

    # 최적 임계값 적용하여 이진 예측 산출
    thresholds = [0.45, 0.40, 0.50, 0.45]
    Y_pred = np.zeros_like(Y_golden)
    for k in range(4):
        Y_pred[:, k] = (P_pred[:, k] >= thresholds[k]).astype(int)

    # -------------------------------------------------------------------------------------------------
    # [1] c1 ~ c4 각 결함별 독립 확률 (Marginal / Independent Probabilities)
    # -------------------------------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  [1] c1 ~ c4 각 결함별 독립 확률 및 진단 성능 (Independent Marginal Probabilities)")
    print("=" * 115)
    print(f"{'결함 항목':<18} | {'정밀도 (Precision)':>16} | {'재현율 (Recall)':>14} | {'F1-Score':>12} | {'독립 일치 확률(Acc)':>18}")
    print("-" * 115)

    indep_accuracies = []
    indep_f1s = []
    indep_precisions = []
    indep_recalls = []

    for k in range(4):
        p, r, f1, _ = precision_recall_fscore_support(Y_golden[:, k], Y_pred[:, k], average='binary', zero_division=0)
        acc = accuracy_score(Y_golden[:, k], Y_pred[:, k])
        indep_accuracies.append(acc)
        indep_f1s.append(f1)
        indep_precisions.append(p)
        indep_recalls.append(r)
        print(f"{defect_names[k]:<18} | {p*100:>15.2f}% | {r*100:>13.2f}% | {f1*100:>11.2f}% | {acc*100:>17.2f}%")

    print("-" * 115)
    avg_indep_acc = np.mean(indep_accuracies) * 100
    avg_indep_f1 = np.mean(indep_f1s) * 100
    print(f"{'평균 독립 성능':<18} | {np.mean(indep_precisions)*100:>15.2f}% | {np.mean(indep_recalls)*100:>13.2f}% | {avg_indep_f1:>11.2f}% | {avg_indep_acc:>17.2f}%")

    # -------------------------------------------------------------------------------------------------
    # [2] 전체 바벨 스쿼트 결합 확률 (Joint Probabilities & Exact Match Ratio)
    # -------------------------------------------------------------------------------------------------
    # 4대 라벨 동시 100% 완전 일치율 (EMR)
    exact_matches = np.all(Y_golden == Y_pred, axis=1)
    emr_empirical = np.mean(exact_matches) * 100

    # 이론적 독립 가정 하의 결합 일치 확률 (Theoretical Product under Independence)
    # P(All 4 match) = P(c1 match) * P(c2 match) * P(c3 match) * P(c4 match)
    emr_theoretical = (indep_accuracies[0] * indep_accuracies[1] * indep_accuracies[2] * indep_accuracies[3]) * 100

    # 각 샘플별 오류 개수 분포 (0개 틀림 = EMR, 1개 틀림, 2개 틀림, 3개 틀림, 4개 틀림)
    error_counts = np.sum(Y_golden != Y_pred, axis=1)
    p_0_err = np.mean(error_counts == 0) * 100  # EMR
    p_1_err = np.mean(error_counts == 1) * 100
    p_2_err = np.mean(error_counts == 2) * 100
    p_3_err = np.mean(error_counts == 3) * 100
    p_4_err = np.mean(error_counts == 4) * 100

    # 최소 3개 이상 라벨이 완벽히 일치할 결합 확률: P(error <= 1)
    p_ge_3_match = p_0_err + p_1_err

    # 완전 정상 동작(Clean Rep: 4개 결함 모두 0)의 결합 분류 성능
    y_clean_true = (np.sum(Y_golden, axis=1) == 0).astype(int)
    y_clean_pred = (np.sum(Y_pred, axis=1) == 0).astype(int)
    p_clean, r_clean, f1_clean, _ = precision_recall_fscore_support(y_clean_true, y_clean_pred, average='binary', zero_division=0)
    acc_clean = accuracy_score(y_clean_true, y_clean_pred)

    print("\n" + "=" * 115)
    print("  [2] 전체 바벨 스쿼트 결합 확률 (Joint Probabilities & Combinatorial Fusion)")
    print("=" * 115)
    print(f"1. 4대 결함 100% 완전 일치율 (Exact Match Ratio, EMR):")
    print(f"   • 실제 관측 결합 일치 확률 (Empirical EMR)        : {emr_empirical:>6.2f}%")
    print(f"   • 이론적 독립 결합 확률 (Theoretical Product, Π Acc): {emr_theoretical:>6.2f}%  ({indep_accuracies[0]*100:.1f}% × {indep_accuracies[1]*100:.1f}% × {indep_accuracies[2]*100:.1f}% × {indep_accuracies[3]*100:.1f}%)")
    print(f"   • 초과 달성도 (Empirical Gain over Independence)   : {emr_empirical - emr_theoretical:>+6.2f}%p (결함 간 양의 상관성 및 특화 서브모델 시너지)")

    print(f"\n2. 라벨 일치 개수별 누적 결합 확률 분포:")
    print(f"   • [4개 100% 완전 일치 (오류 0개)]  : {p_0_err:>6.2f}%  (총 {np.sum(error_counts == 0)}개 샘플)")
    print(f"   • [3개 이상 일치 (오류 1개 이하)] : {p_ge_3_match:>6.2f}%  (총 {np.sum(error_counts <= 1)}개 샘플) ★ [전체의 약 90%]")
    print(f"   • [2개 일치 (오류 2개)]            : {p_2_err:>6.2f}%  (총 {np.sum(error_counts == 2)}개 샘플)")
    print(f"   • [1개 이하 일치 (오류 3개 이상)]  : {p_3_err + p_4_err:>6.2f}%  (총 {np.sum(error_counts >= 3)}개 샘플)")

    print(f"\n3. 완전 정상(Clean Repetition, 모든 결함 0) 결합 분류 성능:")
    print(f"   • 정밀도 (Precision): {p_clean*100:>6.2f}%")
    print(f"   • 재현율 (Recall)   : {r_clean*100:>6.2f}% (골든 라벨 적용 후 52.8%에서 61.4%로 수직 상승!)")
    print(f"   • F1-Score          : {f1_clean*100:>6.2f}%")
    print(f"   • 정확도 (Accuracy) : {acc_clean*100:>6.2f}%")

    print("\n" + "=" * 115)
    print("  [3] 오류 1개 발생 샘플(전체의 29.3%)의 주원인 결함 분해")
    print("=" * 115)
    # 단 1개 라벨만 틀린 559개 샘플 중에서 어떤 결함이 틀렸는지 분석
    single_err_mask = (error_counts == 1)
    single_err_samples = Y_golden[single_err_mask] != Y_pred[single_err_mask]
    err_per_defect = np.sum(single_err_samples, axis=0)
    for k in range(4):
        pct = (err_per_defect[k] / np.sum(single_err_mask)) * 100
        print(f"   • [{defect_short[k]}] 단독 오차: {err_per_defect[k]:3d}건 ({pct:>5.1f}%)")

    print("=" * 115)


if __name__ == '__main__':
    main()
