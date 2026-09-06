# =========================================================================================================
# [모듈 설명]: 3단계 세부 결함(Conditions) 다중 라벨 진단 - Multi-Output LightGBM 분류기
# [입력 데이터]: 97차원 기하 통계 요약 피처 (Format A)
# [출력 목표]: 운동별 세부 결함 항목 (Conditions) 동시 다중 라벨 (Multi-label) 예측
#
# [실행 명령어 (Scheme A: CLI 모드 분리)]:
#   1. 최적 모델 단독 초고속 평가 (기본 모드, 소요시간 ~1초):
#      python -m src.models.third_models.lgbm_multilabel
#      또는 python -m src.models.third_models.lgbm_multilabel --mode best
#
#   2. 최적 모델 vs 기본값(Default) vs 랜덤 후보 5개 3중 비교 분석 (소요시간 ~10초):
#      python -m src.models.third_models.lgbm_multilabel --mode compare
# =========================================================================================================

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

BEST_PARAMS = {
    'learning_rate': 0.03,
    'max_depth': 10,
    'n_estimators': 100,
    'num_leaves': 63
}

class MultiOutputLightGBMClassifier:
    """3단계: Multi-Output LightGBM 세부 결함 다중 라벨 분류기"""

    def __init__(self, n_estimators: int = 150, num_leaves: int = 31, max_depth: int = -1, learning_rate: float = 0.08, random_state: int = 42):
        base_lgb = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
        self.model = MultiOutputClassifier(base_lgb, n_jobs=-1)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "MultiOutputLightGBMClassifier":
        self.model.fit(X_train, Y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        preds = self.predict(X_test)
        inference_time_ms = ((time.time() - t0) / max(1, len(X_test))) * 1000.0

        subset_acc = accuracy_score(Y_test, preds)
        h_loss = hamming_loss(Y_test, preds)
        micro_f1 = f1_score(Y_test, preds, average='micro', zero_division=0)
        macro_f1 = f1_score(Y_test, preds, average='macro', zero_division=0)

        return {
            'model_name': 'Multi-Output LightGBM',
            'split_name': split_name,
            'num_samples': len(Y_test),
            'subset_accuracy': subset_acc,
            'hamming_loss': h_loss,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'inference_time_ms': inference_time_ms,
            'fps': 1000.0 / max(1e-4, inference_time_ms),
            'predictions': preds
        }

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    from src.models.third_models.dataset import Stage3DatasetLoader
    from src.models.third_models.metrics_helper import Stage3MetricsHelper

    parser = argparse.ArgumentParser(description="Stage 3: Multi-Output LightGBM Multi-label Defect Classifier")
    parser.add_argument('--mode', type=str, choices=['best', 'compare'], default='best',
                        help="Execution mode: 'best' (Fast eval with BEST_PARAMS) or 'compare' (Default vs Random 5 vs Best)")
    parser.add_argument('--exercise', type=str, default='01_바벨_스쿼트',
                        help="Target exercise folder (default: '01_바벨_스쿼트')")
    args = parser.parse_args()

    splits = Stage3DatasetLoader.load_all_splits_for_exercise(args.exercise)
    cond_names = splits['train']['cond_names']
    X_train, Y_train = splits['train']['X_Format_A'], splits['train']['Y_multi']

    if args.mode == 'best':
        print("\n" + "=" * 105)
        print(f"    [3단계: Multi-Output LightGBM {args.exercise} 결함 진단 - 최적 모델(BEST_PARAMS) 단독 초고속 성능 평가]")
        print("=" * 105)
        clf = MultiOutputLightGBMClassifier(**BEST_PARAMS)
        clf.fit(X_train, Y_train)
        best_clf = clf
    else:
        print("\n" + "=" * 105)
        print(f"    [3단계: Multi-Output LightGBM {args.exercise} 결함 진단 - Default vs Random 5 vs Best 3중 비교 분석]")
        print("=" * 105)
        from src.models.third_models.tune_all import run_stage3_grid_search_lgbm
        def_trial, other_5, best_trial, best_clf = run_stage3_grid_search_lgbm(splits)

    split_map = [
        ('Valid Set (검증: 피험자 7~8, 시점 D)', 'valid'),
        ('★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)', 'test_strict'),
        ('Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)', 'test_unseen_subj'),
        ('View Test (시점 일반화: 피험자 1~6, 측면 C)', 'test_unseen_view')
    ]

    print("\n" + "=" * 115)
    print(f"      [3단계 Multi-Output LightGBM ({args.exercise}) 세부 결함 진단 다중 분할 평가 결과]")
    print("=" * 115)
    header = f"{'평가 데이터셋 분할':<38} | {'Exact Match':^12} | {'Micro F1':^10} | {'Macro F1':^10} | {'Hamming Loss':^13} | {'Jaccard IoU':^12}"
    print(header)
    print("-" * 115)

    for disp_name, key in split_map:
        X_ev = splits[key]['X_Format_A']
        Y_ev = splits[key]['Y_multi']
        preds = best_clf.predict(X_ev)
        res = Stage3MetricsHelper.evaluate_predictions(Y_ev, preds, cond_names=cond_names)
        print(f"{disp_name:<38} | {res.get('subset_accuracy', 0)*100:>10.2f}% | {res.get('micro_f1', 0)*100:>8.2f}% | {res.get('macro_f1', 0)*100:>8.2f}% | {res.get('hamming_loss', 0):>12.4f} | {res.get('jaccard_samples_iou', 0)*100:>10.2f}%")
    print("=" * 115)

    if args.mode == 'compare':
        print("\n" + "=" * 115)
        print("                  [하이퍼파라미터 튜닝 전후 3중 비교표 (Default vs Random 5 vs Best)]")
        print("=" * 115)
        cmp_header = f"{'구분 (Trial Type)':<20} | {'Strict Micro F1':^16} | {'Strict Exact Match':^18} | {'파라미터 구성 요약':<45}"
        print(cmp_header)
        print("-" * 115)
        print(f"{'기본 모델 (Default)':<20} | {def_trial['strict_f1']*100:>14.2f}% | {def_trial['strict_exact']*100:>16.2f}% | {str(def_trial['params'])[:45]}")
        for i, t in enumerate(other_5, 1):
            p_str = f"lr={t['params'].get('learning_rate')}, depth={t['params'].get('max_depth')}, leaves={t['params'].get('num_leaves')}, n_est={t['params'].get('n_estimators')}"
            print(f"{f'랜덤 후보 #{i}':<20} | {t['strict_f1']*100:>14.2f}% | {t['strict_exact']*100:>16.2f}% | {p_str:<45}")
        p_best = f"lr={best_trial['params'].get('learning_rate')}, depth={best_trial['params'].get('max_depth')}, leaves={best_trial['params'].get('num_leaves')}, n_est={best_trial['params'].get('n_estimators')}"
        print("-" * 115)
        imp = (best_trial['strict_f1'] - def_trial['strict_f1']) * 100
        print(f"{'★ 최적 모델 (Best)':<20} | {best_trial['strict_f1']*100:>14.2f}% | {best_trial['strict_exact']*100:>16.2f}% | {p_best:<45} (성능 향상: +{imp:.2f}%p)")
        print("=" * 115)

