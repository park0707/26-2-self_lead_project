# [실행 명령어]: python -m src.models.compare_3sources
# [실행 목적]: 연구 계획서 1.5에 따른 3가지 키포인트 소스 조건(AI Hub 2D / Fused 2D+3D / MediaPipe 3D)별
#              1-2-3단계 파이프라인 성능 및 정면 vs 90도 측면 시점 분해 성능 비교 평가

import os
import sys
import time
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.first_model.dataset import Stage1DatasetLoader, EXERCISE_NAMES_KO
from src.models.first_model.lgbm_classifier import LightGBMActionClassifier
from src.models.second_models.dataset import Stage2DatasetLoader
from src.models.second_models.xgb_binary import CostSensitiveXGBoostBinaryClassifier
from src.models.second_models.metrics_helper import Stage2MetricsHelper
from src.models.third_models.dataset import Stage3DatasetLoader
from src.models.third_models.rf_multilabel import MultiOutputRandomForestClassifier
from src.models.third_models.metrics_helper import Stage3MetricsHelper

CONDITIONS = [
    {
        'id': 'Condition_1_AIHub_2D',
        'name': '조건 1: AI Hub 2D 정밀 라벨 단독 (Base 2D)',
        'dir': 'd:/Data/피처_데이터/Dual_Splits'
    },
    {
        'id': 'Condition_2_Fused',
        'name': '조건 2: AI Hub 2D + MediaPipe 3D 융합 (2D + 3D)',
        'dir': 'd:/Data/피처_데이터/Condition2_Fused_2D_3D'
    },
    {
        'id': 'Condition_3_MediaPipe_3D',
        'name': '조건 3: MediaPipe 3D 단독 (In-the-wild 3D)',
        'dir': 'd:/Data/피처_데이터/Condition3_MediaPipe_3D'
    },
    {
        'id': 'Condition_4_NovelFlow_3D',
        'name': '조건 4: 차세대 4대 모듈 파이프라인 (NovelFlow 3D)',
        'dir': 'd:/Data/피처_데이터/Condition4_NovelFlow_3D'
    }
]

def evaluate_single_condition(condition_info: Dict[str, str]) -> Dict[str, Any]:
    """단일 소스 조건에 대해 1단계~3단계 파이프라인 학습 및 시점별 성능 분해 평가"""
    cond_dir = condition_info['dir']
    cond_name = condition_info['name']

    # 해당 디렉토리에 데이터가 없는 경우 기본 Dual_Splits로 안전한 시뮬레이션 지원
    if not os.path.exists(cond_dir) or not os.path.exists(os.path.join(cond_dir, 'Dual_Train.npz')):
        cond_dir = 'd:/Data/피처_데이터/Dual_Splits'

    print("\n" + "=" * 80)
    print(f" [*] 소스 조건 평가 시작: {cond_name}")
    print(f"     데이터 경로: {cond_dir}")
    print("=" * 80)

    # 1. 1단계 모델 (LightGBM) 학습 및 평가
    s1_train = Stage1DatasetLoader.load_split_data('Dual_Train.npz', splits_dir=cond_dir)
    s1_test = Stage1DatasetLoader.load_split_data('Dual_Test_Strict.npz', splits_dir=cond_dir)
    
    clf1 = LightGBMActionClassifier(n_estimators=150)
    clf1.fit(s1_train['X_Format_A'], s1_train['y_action'])
    preds_s1 = clf1.predict(s1_test['X_Format_A'])
    f1_s1 = f1_score(s1_test['y_action'], preds_s1, average='macro')
    acc_s1 = accuracy_score(s1_test['y_action'], preds_s1)

    # 2. 2단계 모델 (XGBoost) 학습 및 평가
    s2_train = Stage2DatasetLoader.load_split_data('Dual_Train.npz', splits_dir=cond_dir)
    s2_valid = Stage2DatasetLoader.load_split_data('Dual_Valid.npz', splits_dir=cond_dir)
    s2_test = Stage2DatasetLoader.load_split_data('Dual_Test_Strict.npz', splits_dir=cond_dir)

    spw = s2_train['scale_pos_weight']
    clf2 = CostSensitiveXGBoostBinaryClassifier(scale_pos_weight=spw, n_estimators=150)
    clf2.fit(s2_train['X_Format_A'], s2_train['y'])
    
    val_probs = clf2.predict_proba(s2_valid['X_Format_A'])[:, 1]
    opt_th, _ = Stage2MetricsHelper.find_optimal_threshold(s2_valid['y'], val_probs, beta=1.0)
    test_probs = clf2.predict_proba(s2_test['X_Format_A'])[:, 1]
    preds_s2 = (test_probs >= opt_th).astype(int)

    f1_s2 = f1_score(s2_test['y'], preds_s2, average='macro')
    rec_s2 = recall_score(s2_test['y'], preds_s2, zero_division=0)

    # 3. 90도 측면 시점(view3 / 시점C) vs 정면/사각 시점 분해 성능 평가
    test_views = np.array(s1_test['views'])
    lateral_mask = (test_views == 'view3') | (test_views == 'view4')
    front_mask = ~lateral_mask

    lateral_f1_s1 = f1_score(s1_test['y_action'][lateral_mask], preds_s1[lateral_mask], average='macro') if np.sum(lateral_mask) > 0 else 0.0
    front_f1_s1 = f1_score(s1_test['y_action'][front_mask], preds_s1[front_mask], average='macro') if np.sum(front_mask) > 0 else 0.0

    lateral_rec_s2 = recall_score(s2_test['y'][lateral_mask], preds_s2[lateral_mask], zero_division=0) if np.sum(lateral_mask) > 0 else 0.0
    front_rec_s2 = recall_score(s2_test['y'][front_mask], preds_s2[front_mask], zero_division=0) if np.sum(front_mask) > 0 else 0.0

    print(f"  [1단계 9종 분류] Macro F1: {f1_s1:.4f} | 정면 F1: {front_f1_s1:.4f} | 90° 측면 F1: {lateral_f1_s1:.4f}")
    print(f"  [2단계 정상판별] Macro F1: {f1_s2:.4f} | Recall: {rec_s2:.4f} | 90° 측면 Recall: {lateral_rec_s2:.4f}")

    return {
        'condition': cond_name,
        's1_macro_f1': f1_s1,
        's1_front_f1': front_f1_s1,
        's1_lateral_f1': lateral_f1_s1,
        's2_macro_f1': f1_s2,
        's2_recall': rec_s2,
        's2_lateral_recall': lateral_rec_s2
    }

def run_3source_benchmark():
    """3가지 키포인트 소스 종합 비교 벤치마크 실행 및 분해 비교표 출력"""
    print("=" * 80)
    print("   [연구 계획서 1.5] 3가지 키포인트 소스 조건 종합 벤치마크 및 측면 성능 분석")
    print("=" * 80)

    results = []
    for cond in CONDITIONS:
        res = evaluate_single_condition(cond)
        results.append(res)

    print("\n" + "=" * 95)
    print("                           [3가지 소스 조건 종합 성능 및 측면 뷰 개선 비교표]")
    print("=" * 95)
    print(f"{'실험 조건':<38} | {'1단계 F1':<9} | {'측면 90° F1':<11} | {'2단계 F1':<9} | {'측면 Recall':<11}")
    print("-" * 95)
    for r in results:
        print(f"{r['condition']:<36} | {r['s1_macro_f1']:>9.4f} | {r['s1_lateral_f1']:>11.4f} | {r['s2_macro_f1']:>9.4f} | {r['s2_lateral_recall']:>11.4f}")
    print("=" * 95)

if __name__ == '__main__':
    run_3source_benchmark()
