# [실행 명령어]: python -m src.models.third_models.tune_all
# [실행 목적]: 3단계 세부 결함(Conditions) 4대 모델 하이퍼파라미터 전수 그리드 탐색 및 3중 성능 비교 (Default vs Random 5 vs Best)

import os
import sys
import time
import json
import itertools
from typing import Dict, List, Any, Tuple
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.third_models.dataset import Stage3DatasetLoader
from src.models.third_models.metrics_helper import Stage3MetricsHelper
from src.models.third_models.rf_multilabel import MultiOutputRandomForestClassifier
from src.models.third_models.xgb_multilabel import MultiOutputXGBoostClassifier
from src.models.third_models.lgbm_multilabel import MultiOutputLightGBMClassifier
from src.models.third_models.gru_multilabel import TemporalGRUMultiLabelClassifier

def run_stage3_grid_search_rf(splits: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Any]:
    X_train, Y_train = splits['train']['X_Format_A'], splits['train']['Y_multi']
    X_val, Y_val = splits['valid']['X_Format_A'], splits['valid']['Y_multi']
    X_strict, Y_strict = splits['test_strict']['X_Format_A'], splits['test_strict']['Y_multi']
    cond_names = splits['train']['cond_names']

    param_grid = {
        'n_estimators': [80, 150, 250],
        'max_depth': [8, 12, 16, None],
        'min_samples_split': [2, 5]
    }
    keys, values = zip(*param_grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def_clf = MultiOutputRandomForestClassifier(n_estimators=150, max_depth=12)
    def_clf.fit(X_train, Y_train)
    res_v_def = Stage3MetricsHelper.evaluate_predictions(Y_val, def_clf.predict(X_val), cond_names=cond_names)
    res_st_def = Stage3MetricsHelper.evaluate_predictions(Y_strict, def_clf.predict(X_strict), cond_names=cond_names)
    default_trial = {'params': {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 2}, 'valid_f1': res_v_def['micro_f1'], 'strict_f1': res_st_def['micro_f1'], 'strict_iou': res_st_def['jaccard_samples_iou'], 'strict_exact': res_st_def['subset_accuracy']}

    results = []
    best_val_f1 = -1.0
    best_clf = None

    for combo in all_combos:
        clf = MultiOutputRandomForestClassifier(n_estimators=combo['n_estimators'], max_depth=combo['max_depth'])
        if combo['min_samples_split'] != 2:
            clf.model.estimator.min_samples_split = combo['min_samples_split']
        clf.fit(X_train, Y_train)

        res_v = Stage3MetricsHelper.evaluate_predictions(Y_val, clf.predict(X_val), cond_names=cond_names)
        res_st = Stage3MetricsHelper.evaluate_predictions(Y_strict, clf.predict(X_strict), cond_names=cond_names)

        trial_info = {
            'params': combo,
            'valid_f1': res_v['micro_f1'],
            'strict_f1': res_st['micro_f1'],
            'strict_iou': res_st['jaccard_samples_iou'],
            'strict_exact': res_st['subset_accuracy'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['micro_f1'] > best_val_f1:
            best_val_f1 = res_v['micro_f1']
            best_clf = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    indices = [1, min(4, len(results)-1), min(10, len(results)-1), min(18, len(results)-1), len(results)-1]
    unique_indices = sorted(list(set(indices)))
    other_5 = [results[i] for i in unique_indices if i < len(results) and i != 0][:5]

    return default_trial, other_5, best_trial, best_clf

def run_stage3_grid_search_xgb(splits: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Any]:
    X_train, Y_train = splits['train']['X_Format_A'], splits['train']['Y_multi']
    X_val, Y_val = splits['valid']['X_Format_A'], splits['valid']['Y_multi']
    X_strict, Y_strict = splits['test_strict']['X_Format_A'], splits['test_strict']['Y_multi']
    cond_names = splits['train']['cond_names']

    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.03, 0.08, 0.15],
        'n_estimators': [100, 200]
    }
    keys, values = zip(*param_grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def_clf = MultiOutputXGBoostClassifier(n_estimators=150, max_depth=6, learning_rate=0.08)
    def_clf.fit(X_train, Y_train)
    res_v_def = Stage3MetricsHelper.evaluate_predictions(Y_val, def_clf.predict(X_val), cond_names=cond_names)
    res_st_def = Stage3MetricsHelper.evaluate_predictions(Y_strict, def_clf.predict(X_strict), cond_names=cond_names)
    default_trial = {'params': {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.08}, 'valid_f1': res_v_def['micro_f1'], 'strict_f1': res_st_def['micro_f1'], 'strict_iou': res_st_def['jaccard_samples_iou'], 'strict_exact': res_st_def['subset_accuracy']}

    results = []
    best_val_f1 = -1.0
    best_clf = None

    for combo in all_combos:
        clf = MultiOutputXGBoostClassifier(**combo)
        clf.fit(X_train, Y_train)

        res_v = Stage3MetricsHelper.evaluate_predictions(Y_val, clf.predict(X_val), cond_names=cond_names)
        res_st = Stage3MetricsHelper.evaluate_predictions(Y_strict, clf.predict(X_strict), cond_names=cond_names)

        trial_info = {
            'params': combo,
            'valid_f1': res_v['micro_f1'],
            'strict_f1': res_st['micro_f1'],
            'strict_iou': res_st['jaccard_samples_iou'],
            'strict_exact': res_st['subset_accuracy'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['micro_f1'] > best_val_f1:
            best_val_f1 = res_v['micro_f1']
            best_clf = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    indices = [1, min(4, len(results)-1), min(8, len(results)-1), min(12, len(results)-1), len(results)-1]
    unique_indices = sorted(list(set(indices)))
    other_5 = [results[i] for i in unique_indices if i < len(results) and i != 0][:5]

    return default_trial, other_5, best_trial, best_clf

def run_stage3_grid_search_lgbm(splits: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Any]:
    X_train, Y_train = splits['train']['X_Format_A'], splits['train']['Y_multi']
    X_val, Y_val = splits['valid']['X_Format_A'], splits['valid']['Y_multi']
    X_strict, Y_strict = splits['test_strict']['X_Format_A'], splits['test_strict']['Y_multi']
    cond_names = splits['train']['cond_names']

    param_grid = {
        'num_leaves': [15, 31, 63],
        'max_depth': [-1, 6, 10],
        'learning_rate': [0.03, 0.08, 0.15],
        'n_estimators': [100, 200]
    }
    keys, values = zip(*param_grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def_clf = MultiOutputLightGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.08)
    def_clf.fit(X_train, Y_train)
    res_v_def = Stage3MetricsHelper.evaluate_predictions(Y_val, def_clf.predict(X_val), cond_names=cond_names)
    res_st_def = Stage3MetricsHelper.evaluate_predictions(Y_strict, def_clf.predict(X_strict), cond_names=cond_names)
    default_trial = {'params': {'n_estimators': 150, 'num_leaves': 31, 'learning_rate': 0.08}, 'valid_f1': res_v_def['micro_f1'], 'strict_f1': res_st_def['micro_f1'], 'strict_iou': res_st_def['jaccard_samples_iou'], 'strict_exact': res_st_def['subset_accuracy']}

    results = []
    best_val_f1 = -1.0
    best_clf = None

    for combo in all_combos:
        clf = MultiOutputLightGBMClassifier(**combo)
        clf.fit(X_train, Y_train)

        res_v = Stage3MetricsHelper.evaluate_predictions(Y_val, clf.predict(X_val), cond_names=cond_names)
        res_st = Stage3MetricsHelper.evaluate_predictions(Y_strict, clf.predict(X_strict), cond_names=cond_names)

        trial_info = {
            'params': combo,
            'valid_f1': res_v['micro_f1'],
            'strict_f1': res_st['micro_f1'],
            'strict_iou': res_st['jaccard_samples_iou'],
            'strict_exact': res_st['subset_accuracy'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['micro_f1'] > best_val_f1:
            best_val_f1 = res_v['micro_f1']
            best_clf = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    indices = [1, min(10, len(results)-1), min(25, len(results)-1), min(40, len(results)-1), len(results)-1]
    unique_indices = sorted(list(set(indices)))
    other_5 = [results[i] for i in unique_indices if i < len(results) and i != 0][:5]

    return default_trial, other_5, best_trial, best_clf

def run_stage3_grid_search_gru(splits: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Any]:
    X_train, Y_train = splits['train']['X_Format_B'], splits['train']['Y_multi']
    X_val, Y_val = splits['valid']['X_Format_B'], splits['valid']['Y_multi']
    X_strict, Y_strict = splits['test_strict']['X_Format_B'], splits['test_strict']['Y_multi']
    cond_names = splits['train']['cond_names']
    num_labels = splits['train']['num_conditions']

    param_grid = [
        {'conv_filters': 64, 'hidden_dim': 64, 'num_layers': 2, 'lr': 0.001, 'epochs': 5},
        {'conv_filters': 128, 'hidden_dim': 64, 'num_layers': 2, 'lr': 0.001, 'epochs': 5},
        {'conv_filters': 64, 'hidden_dim': 128, 'num_layers': 2, 'lr': 0.0005, 'epochs': 4},
        {'conv_filters': 128, 'hidden_dim': 128, 'num_layers': 2, 'lr': 0.0005, 'epochs': 4},
        {'conv_filters': 64, 'hidden_dim': 64, 'num_layers': 1, 'lr': 0.002, 'epochs': 4},
        {'conv_filters': 128, 'hidden_dim': 64, 'num_layers': 1, 'lr': 0.001, 'epochs': 4}
    ]

    def_combo = param_grid[0]
    def_clf = TemporalGRUMultiLabelClassifier(num_labels=num_labels, **def_combo)
    def_clf.fit(X_train, Y_train)
    res_v_def = Stage3MetricsHelper.evaluate_predictions(Y_val, def_clf.predict(X_val), cond_names=cond_names)
    res_st_def = Stage3MetricsHelper.evaluate_predictions(Y_strict, def_clf.predict(X_strict), cond_names=cond_names)
    default_trial = {'params': def_combo, 'valid_f1': res_v_def['micro_f1'], 'strict_f1': res_st_def['micro_f1'], 'strict_iou': res_st_def['jaccard_samples_iou'], 'strict_exact': res_st_def['subset_accuracy']}

    results = []
    best_val_f1 = -1.0
    best_clf = None

    for combo in param_grid:
        clf = TemporalGRUMultiLabelClassifier(num_labels=num_labels, **combo)
        clf.fit(X_train, Y_train)

        res_v = Stage3MetricsHelper.evaluate_predictions(Y_val, clf.predict(X_val), cond_names=cond_names)
        res_st = Stage3MetricsHelper.evaluate_predictions(Y_strict, clf.predict(X_strict), cond_names=cond_names)

        trial_info = {
            'params': combo,
            'valid_f1': res_v['micro_f1'],
            'strict_f1': res_st['micro_f1'],
            'strict_iou': res_st['jaccard_samples_iou'],
            'strict_exact': res_st['subset_accuracy'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['micro_f1'] > best_val_f1:
            best_val_f1 = res_v['micro_f1']
            best_clf = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    other_5 = results[1:6]

    return default_trial, other_5, best_trial, best_clf

def run_all_stage3_tuning(target_exercise: str = '01_바벨_스쿼트') -> Dict[str, Any]:
    print("=" * 105)
    print(f"      [3단계: {target_exercise} 세부 결함 4대 모델 전수 하이퍼파라미터 그리드 탐색 및 3중 비교]")
    print("=" * 105)

    splits = Stage3DatasetLoader.load_all_splits_for_exercise(target_exercise)
    runners = [
        ("Multi-Output LightGBM", run_stage3_grid_search_lgbm, 'Format_A'),
        ("Multi-Output XGBoost", run_stage3_grid_search_xgb, 'Format_A'),
        ("Multi-Output Random Forest", run_stage3_grid_search_rf, 'Format_A'),
        ("1D-CNN + Bi-GRU (Multi-Head)", run_stage3_grid_search_gru, 'Format_B')
    ]

    stage3_results = {}

    for m_name, fn, fmt in runners:
        print(f"\n[*] >>> [3단계: {m_name}] 하이퍼파라미터 탐색 중...")
        t0 = time.time()
        def_t, oth_5, bst_t, bst_clf = fn(splits)
        elapsed = time.time() - t0

        imp_f1 = (bst_t['strict_f1'] - def_t['strict_f1']) * 100.0
        imp_iou = (bst_t['strict_iou'] - def_t['strict_iou']) * 100.0

        print(f"    - 기본 Micro-F1: {def_t['strict_f1']*100:.2f}% (Jaccard IoU: {def_t['strict_iou']*100:.2f}%)")
        print(f"    - 최적 Micro-F1: {bst_t['strict_f1']*100:.2f}% (Jaccard IoU: {bst_t['strict_iou']*100:.2f}%) [F1 향상: +{imp_f1:.2f}%p]")
        print(f"    - 최적 파라미터: {bst_t['params']}")

        stage3_results[m_name] = {
            'input_format': fmt,
            'default_trial': def_t,
            'other_5_trials': oth_5,
            'best_trial': bst_t,
            'best_clf': bst_clf,
            'improvement_f1_pts': imp_f1,
            'improvement_iou_pts': imp_iou,
            'elapsed_sec': elapsed
        }

    return stage3_results

if __name__ == '__main__':
    run_all_stage3_tuning()
