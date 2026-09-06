# =========================================================================================================
# [모듈 설명]: 1단계 9종 운동 대분류 - ExtraTrees (Extremely Randomized Trees) 분류기
# [입력 데이터]: 97차원 기하 통계 요약 피처 (Format A: 관절 각도/거리/속도 11종 통계량)
# [출력 목표]: 9종 운동 대분류 (01_바벨_스쿼트 ~ 09_랫_풀_다운) Multi-class 예측
#
# [실행 명령어 (Scheme A: CLI 모드 분리)]:
#   1. 최적 모델 단독 초고속 평가 (기본 모드, 소요시간 ~2초):
#      python -m src.models.first_model.extratrees_classifier
#      또는 python -m src.models.first_model.extratrees_classifier --mode best
#
#   2. 최적 모델 vs 기본값(Default) vs 랜덤 후보 5개 3중 비교 분석 (소요시간 ~15초):
#      python -m src.models.first_model.extratrees_classifier --mode compare
# =========================================================================================================

import os
import sys
import time
import argparse
import itertools
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.first_model.metrics_helper import Stage1MetricsHelper

BEST_PARAMS = {'max_depth': 15, 'min_samples_split': 2, 'n_estimators': 300}

class ExtraTreesActionClassifier:
    """1단계: ExtraTrees (Extremely Randomized Trees) 9종 운동 분류기 (Grid Search & 비교 분석 내장)"""

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = 20, min_samples_split: int = 2, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced'
        )
        self.hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'class_weight': 'balanced',
            'random_state': random_state
        }
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ExtraTreesActionClassifier":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        preds = self.predict(X_test)
        inference_time_ms = ((time.time() - t0) / max(1, len(X_test))) * 1000.0
        metrics = Stage1MetricsHelper.evaluate_predictions(y_test, preds, inference_time_ms=inference_time_ms)
        metrics['model_name'] = 'ExtraTrees'
        metrics['split_name'] = split_name
        metrics['num_samples'] = len(y_test)
        metrics['predictions'] = preds
        return metrics

def run_extratrees_grid_search(data: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], ExtraTreesActionClassifier]:
    """ExtraTrees 하이퍼파라미터 그리드 탐색 및 최고 vs 후보 5종 비교"""
    X_train, y_train = data['train']['X_Format_A'], data['train']['y_action']
    X_valid, y_valid = data['valid']['X_Format_A'], data['valid']['y_action']
    X_strict, y_strict = data['test_strict']['X_Format_A'], data['test_strict']['y_action']
    X_subj, y_subj = data['test_unseen_subj']['X_Format_A'], data['test_unseen_subj']['y_action']

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5]
    }

    keys, values = zip(*param_grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"[*] ExtraTrees 하이퍼파라미터 그리드 탐색 시작 (총 {len(all_combos)}개 조합 검증)...")

    results = []
    best_valid_f1 = -1.0
    best_model = None

    for idx, combo in enumerate(all_combos, 1):
        clf = ExtraTreesActionClassifier(
            n_estimators=combo['n_estimators'],
            max_depth=combo['max_depth'],
            min_samples_split=combo['min_samples_split']
        )
        clf.fit(X_train, y_train)
        
        res_v = clf.evaluate(X_valid, y_valid, split_name='valid')
        res_st = clf.evaluate(X_strict, y_strict, split_name='strict')
        res_sb = clf.evaluate(X_subj, y_subj, split_name='subj')

        trial_info = {
            'params': combo,
            'valid_f1': res_v['macro_f1'],
            'strict_f1': res_st['macro_f1'],
            'subj_f1': res_sb['macro_f1'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['macro_f1'] > best_valid_f1:
            best_valid_f1 = res_v['macro_f1']
            best_model = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    
    indices = [1, min(4, len(results)-1), min(10, len(results)-1), min(18, len(results)-1), len(results)-1]
    unique_indices = sorted(list(set(indices)))
    other_5 = [results[i] for i in unique_indices if i < len(results) and i != 0][:5]

    return best_trial, other_5, best_model

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Stage 1: ExtraTrees 9-Class Action Classifier")
    parser.add_argument('--mode', type=str, choices=['best', 'compare'], default='best',
                        help="Execution mode: 'best' (Fast eval with BEST_PARAMS) or 'compare' (Default vs Random 5 vs Best)")
    args = parser.parse_args()

    data = Stage1DatasetLoader.load_all_dual_splits()

    if args.mode == 'best':
        print("\n" + "=" * 105)
        print("    [1단계: ExtraTrees 9종 운동 대분류 - 최적 모델(BEST_PARAMS) 단독 초고속 성능 평가]")
        print("=" * 105)
        clf = ExtraTreesActionClassifier(**BEST_PARAMS)
        clf.fit(data['train']['X_Format_A'], data['train']['y_action'])
        best_clf = clf
        best_params = BEST_PARAMS
    else:
        print("\n" + "=" * 105)
        print("    [1단계: ExtraTrees 9종 운동 대분류 - Default vs Random 5 vs Best 3중 비교 분석]")
        print("=" * 105)
        best_trial, other_5, best_clf = run_extratrees_grid_search(data)
        best_params = best_trial['params']

    Stage1MetricsHelper.print_model_header(
        model_name=f"ExtraTrees 9-Class Classifier ({'Best Tuned' if args.mode=='best' else 'Grid Search Compare'})",
        hyperparams=best_params,
        train_samples=data['train']['X_Format_A'].shape[0],
        feat_dim="Format A (97차원 기하 통계 요약 피처)"
    )

    split_results = {}
    split_map = [
        ('Valid Set (검증: 피험자 7~8, 시점 D)', 'valid'),
        ('★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)', 'test_strict'),
        ('Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)', 'test_unseen_subj'),
        ('View Test (시점 일반화: 피험자 1~6, 측면 C)', 'test_unseen_view')
    ]

    strict_res = None
    for display_name, key in split_map:
        res = best_clf.evaluate(data[key]['X_Format_A'], data[key]['y_action'], split_name=key)
        split_results[display_name] = res
        if key == 'test_strict':
            strict_res = res

    res_train = best_clf.evaluate(data['train']['X_Format_A'], data['train']['y_action'], split_name='train')
    adv_metrics = Stage1MetricsHelper.calculate_advanced_metrics(
        res_train=res_train,
        res_valid=split_results['Valid Set (검증: 피험자 7~8, 시점 D)'],
        res_strict=strict_res,
        res_subj=split_results['Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)'],
        res_view=split_results['View Test (시점 일반화: 피험자 1~6, 측면 C)']
    )

    Stage1MetricsHelper.print_multi_split_summary(split_results, adv_metrics=adv_metrics)
    if strict_res is not None:
        Stage1MetricsHelper.print_class_breakdown_report(strict_res, data['test_strict']['y_action'])
        Stage1MetricsHelper.print_top_confusion_analysis(strict_res, top_k=3)

    if args.mode == 'compare':
        Stage1MetricsHelper.print_top_vs_random_comparison_table(
            best_trial=best_trial,
            other_trials=other_5,
            param_keys=['n_estimators', 'max_depth', 'min_samples_split']
        )
