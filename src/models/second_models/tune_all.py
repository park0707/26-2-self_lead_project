# [실행 명령어]: python -m src.models.second_models.tune_all
# [실행 목적]: 2단계 정상(1) vs 오류(0) 4대 모델 하이퍼파라미터 전수 그리드 탐색 및 3중 성능 비교 (Default vs Random 5 vs Best)

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

from src.models.second_models.dataset import Stage2DatasetLoader
from src.models.second_models.metrics_helper import Stage2MetricsHelper
from src.models.second_models.rf_binary import BalancedRandomForestBinaryClassifier
from src.models.second_models.xgb_binary import CostSensitiveXGBoostBinaryClassifier
from src.models.second_models.lgbm_binary import CostSensitiveLightGBMBinaryClassifier
from src.models.second_models.gru_binary import TemporalGRUBinaryClassifier

def run_stage2_grid_search_rf(splits: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Any]:
    """Balanced Random Forest 그리드 탐색"""
    X_train, y_train = splits['train']['X_Format_A'], splits['train']['y']
    X_val, y_val = splits['valid']['X_Format_A'], splits['valid']['y']
    X_strict, y_strict = splits['test_strict']['X_Format_A'], splits['test_strict']['y']

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 12, 16, None],
        'min_samples_split': [2, 5]
    }
    keys, values = zip(*param_grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 1. Default Trial (기본값)
    def_clf = BalancedRandomForestBinaryClassifier(n_estimators=200, max_depth=15)
    def_clf.fit(X_train, y_train)
    val_p = def_clf.predict_proba(X_val)[:, 1]
    th_def, _ = Stage2MetricsHelper.find_optimal_threshold(y_val, val_p)
    res_v_def = Stage2MetricsHelper.evaluate_predictions(y_val, val_p, threshold=th_def)
    res_st_def = Stage2MetricsHelper.evaluate_predictions(y_strict, def_clf.predict_proba(X_strict)[:, 1], threshold=th_def)
    default_trial = {'params': {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2}, 'valid_f1': res_v_def['macro_f1'], 'strict_f1': res_st_def['macro_f1'], 'strict_pr_auc': res_st_def['pr_auc']}

    results = []
    best_val_f1 = -1.0
    best_clf = None

    for combo in all_combos:
        clf = BalancedRandomForestBinaryClassifier(
            n_estimators=combo['n_estimators'],
            max_depth=combo['max_depth']
        )
        if combo['min_samples_split'] != 2:
            clf.model.min_samples_split = combo['min_samples_split']
        clf.fit(X_train, y_train)
        
        val_probs = clf.predict_proba(X_val)[:, 1]
        opt_th, _ = Stage2MetricsHelper.find_optimal_threshold(y_val, val_probs)
        res_v = Stage2MetricsHelper.evaluate_predictions(y_val, val_probs, threshold=opt_th)
        res_st = Stage2MetricsHelper.evaluate_predictions(y_strict, clf.predict_proba(X_strict)[:, 1], threshold=opt_th)

        trial_info = {
            'params': {**combo, 'opt_threshold': opt_th},
            'valid_f1': res_v['macro_f1'],
            'strict_f1': res_st['macro_f1'],
            'strict_pr_auc': res_st['pr_auc'],
            'strict_mcc': res_st['mcc'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['macro_f1'] > best_val_f1:
            best_val_f1 = res_v['macro_f1']
            best_clf = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    
    indices = [1, min(4, len(results)-1), min(10, len(results)-1), min(18, len(results)-1), len(results)-1]
    unique_indices = sorted(list(set(indices)))
    other_5 = [results[i] for i in unique_indices if i < len(results) and i != 0][:5]

    return default_trial, other_5, best_trial, best_clf

def run_stage2_grid_search_xgb(splits: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Any]:
    """Cost-Sensitive XGBoost 그리드 탐색"""
    X_train, y_train = splits['train']['X_Format_A'], splits['train']['y']
    X_val, y_val = splits['valid']['X_Format_A'], splits['valid']['y']
    X_strict, y_strict = splits['test_strict']['X_Format_A'], splits['test_strict']['y']
    spw = splits['train']['scale_pos_weight']

    param_grid = {
        'scale_pos_weight': [spw * 0.5, spw, spw * 1.5],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.03, 0.08, 0.15],
        'n_estimators': [150, 250]
    }
    keys, values = zip(*param_grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def_clf = CostSensitiveXGBoostBinaryClassifier(scale_pos_weight=spw, n_estimators=200, max_depth=5, learning_rate=0.08)
    def_clf.fit(X_train, y_train)
    val_p = def_clf.predict_proba(X_val)[:, 1]
    th_def, _ = Stage2MetricsHelper.find_optimal_threshold(y_val, val_p)
    res_v_def = Stage2MetricsHelper.evaluate_predictions(y_val, val_p, threshold=th_def)
    res_st_def = Stage2MetricsHelper.evaluate_predictions(y_strict, def_clf.predict_proba(X_strict)[:, 1], threshold=th_def)
    default_trial = {'params': {'scale_pos_weight': spw, 'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.08}, 'valid_f1': res_v_def['macro_f1'], 'strict_f1': res_st_def['macro_f1'], 'strict_pr_auc': res_st_def['pr_auc']}

    results = []
    best_val_f1 = -1.0
    best_clf = None

    for combo in all_combos:
        clf = CostSensitiveXGBoostBinaryClassifier(
            scale_pos_weight=combo['scale_pos_weight'],
            n_estimators=combo['n_estimators'],
            max_depth=combo['max_depth'],
            learning_rate=combo['learning_rate']
        )
        clf.fit(X_train, y_train)
        
        val_probs = clf.predict_proba(X_val)[:, 1]
        opt_th, _ = Stage2MetricsHelper.find_optimal_threshold(y_val, val_probs)
        res_v = Stage2MetricsHelper.evaluate_predictions(y_val, val_probs, threshold=opt_th)
        res_st = Stage2MetricsHelper.evaluate_predictions(y_strict, clf.predict_proba(X_strict)[:, 1], threshold=opt_th)

        trial_info = {
            'params': {**combo, 'opt_threshold': opt_th},
            'valid_f1': res_v['macro_f1'],
            'strict_f1': res_st['macro_f1'],
            'strict_pr_auc': res_st['pr_auc'],
            'strict_mcc': res_st['mcc'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['macro_f1'] > best_val_f1:
            best_val_f1 = res_v['macro_f1']
            best_clf = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    
    indices = [1, min(5, len(results)-1), min(15, len(results)-1), min(35, len(results)-1), len(results)-1]
    unique_indices = sorted(list(set(indices)))
    other_5 = [results[i] for i in unique_indices if i < len(results) and i != 0][:5]

    return default_trial, other_5, best_trial, best_clf

def run_stage2_grid_search_lgbm(splits: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Any]:
    """Cost-Sensitive LightGBM 그리드 탐색"""
    X_train, y_train = splits['train']['X_Format_A'], splits['train']['y']
    X_val, y_val = splits['valid']['X_Format_A'], splits['valid']['y']
    X_strict, y_strict = splits['test_strict']['X_Format_A'], splits['test_strict']['y']
    spw = splits['train']['scale_pos_weight']

    param_grid = {
        'scale_pos_weight': [spw * 0.5, spw, spw * 1.5],
        'num_leaves': [15, 31, 63],
        'max_depth': [-1, 6, 10],
        'learning_rate': [0.03, 0.08, 0.15],
        'n_estimators': [150, 250]
    }
    keys, values = zip(*param_grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def_clf = CostSensitiveLightGBMBinaryClassifier(scale_pos_weight=spw, n_estimators=200, num_leaves=31, learning_rate=0.08)
    def_clf.fit(X_train, y_train)
    val_p = def_clf.predict_proba(X_val)[:, 1]
    th_def, _ = Stage2MetricsHelper.find_optimal_threshold(y_val, val_p)
    res_v_def = Stage2MetricsHelper.evaluate_predictions(y_val, val_p, threshold=th_def)
    res_st_def = Stage2MetricsHelper.evaluate_predictions(y_strict, def_clf.predict_proba(X_strict)[:, 1], threshold=th_def)
    default_trial = {'params': {'scale_pos_weight': spw, 'n_estimators': 200, 'num_leaves': 31, 'learning_rate': 0.08}, 'valid_f1': res_v_def['macro_f1'], 'strict_f1': res_st_def['macro_f1'], 'strict_pr_auc': res_st_def['pr_auc']}

    results = []
    best_val_f1 = -1.0
    best_clf = None

    for combo in all_combos:
        clf = CostSensitiveLightGBMBinaryClassifier(
            scale_pos_weight=combo['scale_pos_weight'],
            n_estimators=combo['n_estimators'],
            num_leaves=combo['num_leaves'],
            max_depth=combo['max_depth'],
            learning_rate=combo['learning_rate']
        )
        clf.fit(X_train, y_train)
        
        val_probs = clf.predict_proba(X_val)[:, 1]
        opt_th, _ = Stage2MetricsHelper.find_optimal_threshold(y_val, val_probs)
        res_v = Stage2MetricsHelper.evaluate_predictions(y_val, val_probs, threshold=opt_th)
        res_st = Stage2MetricsHelper.evaluate_predictions(y_strict, clf.predict_proba(X_strict)[:, 1], threshold=opt_th)

        trial_info = {
            'params': {**combo, 'opt_threshold': opt_th},
            'valid_f1': res_v['macro_f1'],
            'strict_f1': res_st['macro_f1'],
            'strict_pr_auc': res_st['pr_auc'],
            'strict_mcc': res_st['mcc'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['macro_f1'] > best_val_f1:
            best_val_f1 = res_v['macro_f1']
            best_clf = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    
    indices = [1, min(10, len(results)-1), min(25, len(results)-1), min(60, len(results)-1), len(results)-1]
    unique_indices = sorted(list(set(indices)))
    other_5 = [results[i] for i in unique_indices if i < len(results) and i != 0][:5]

    return default_trial, other_5, best_trial, best_clf

def run_stage2_grid_search_gru(splits: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Any]:
    """1D-CNN + Bi-GRU (Focal Loss) 그리드 탐색"""
    X_train, y_train = splits['train']['X_Format_B'], splits['train']['y']
    X_val, y_val = splits['valid']['X_Format_B'], splits['valid']['y']
    X_strict, y_strict = splits['test_strict']['X_Format_B'], splits['test_strict']['y']

    param_grid = [
        {'conv_filters': 64, 'hidden_dim': 64, 'num_layers': 2, 'lr': 0.001, 'epochs': 2},
        {'conv_filters': 128, 'hidden_dim': 64, 'num_layers': 2, 'lr': 0.001, 'epochs': 2},
        {'conv_filters': 64, 'hidden_dim': 128, 'num_layers': 2, 'lr': 0.0005, 'epochs': 2},
        {'conv_filters': 128, 'hidden_dim': 128, 'num_layers': 2, 'lr': 0.0005, 'epochs': 2},
        {'conv_filters': 64, 'hidden_dim': 64, 'num_layers': 1, 'lr': 0.002, 'epochs': 2},
        {'conv_filters': 128, 'hidden_dim': 64, 'num_layers': 1, 'lr': 0.001, 'epochs': 2}
    ]

    def_combo = param_grid[0]
    def_clf = TemporalGRUBinaryClassifier(**def_combo)
    def_clf.fit(X_train, y_train)
    val_p = def_clf.predict_proba(X_val)
    val_p_pos = val_p[:, 1] if val_p.ndim > 1 else val_p
    th_def, _ = Stage2MetricsHelper.find_optimal_threshold(y_val, val_p_pos)
    res_v_def = Stage2MetricsHelper.evaluate_predictions(y_val, val_p_pos, threshold=th_def)
    st_p = def_clf.predict_proba(X_strict)
    st_p_pos = st_p[:, 1] if st_p.ndim > 1 else st_p
    res_st_def = Stage2MetricsHelper.evaluate_predictions(y_strict, st_p_pos, threshold=th_def)
    default_trial = {'params': def_combo, 'valid_f1': res_v_def['macro_f1'], 'strict_f1': res_st_def['macro_f1'], 'strict_pr_auc': res_st_def['pr_auc']}

    results = []
    best_val_f1 = -1.0
    best_clf = None

    for combo in param_grid:
        clf = TemporalGRUBinaryClassifier(**combo)
        clf.fit(X_train, y_train)
        
        val_probs = clf.predict_proba(X_val)
        val_probs_pos = val_probs[:, 1] if val_probs.ndim > 1 else val_probs
        opt_th, _ = Stage2MetricsHelper.find_optimal_threshold(y_val, val_probs_pos)
        res_v = Stage2MetricsHelper.evaluate_predictions(y_val, val_probs_pos, threshold=opt_th)
        
        st_probs = clf.predict_proba(X_strict)
        st_probs_pos = st_probs[:, 1] if st_probs.ndim > 1 else st_probs
        res_st = Stage2MetricsHelper.evaluate_predictions(y_strict, st_probs_pos, threshold=opt_th)

        trial_info = {
            'params': {**combo, 'opt_threshold': opt_th},
            'valid_f1': res_v['macro_f1'],
            'strict_f1': res_st['macro_f1'],
            'strict_pr_auc': res_st['pr_auc'],
            'strict_mcc': res_st['mcc'],
            'model': clf
        }
        results.append(trial_info)

        if res_v['macro_f1'] > best_val_f1:
            best_val_f1 = res_v['macro_f1']
            best_clf = clf

    results.sort(key=lambda x: x['valid_f1'], reverse=True)
    best_trial = results[0]
    other_5 = results[1:6]

    return default_trial, other_5, best_trial, best_clf

def run_all_stage2_tuning() -> Dict[str, Any]:
    print("=" * 105)
    print("      [2단계: 동작 정상/오류 판별 4대 모델 전수 하이퍼파라미터 그리드 탐색 및 3중 비교]")
    print("=" * 105)

    splits = Stage2DatasetLoader.load_all_dual_splits()
    runners = [
        ("Cost-Sensitive LightGBM", run_stage2_grid_search_lgbm, 'Format_A'),
        ("Cost-Sensitive XGBoost", run_stage2_grid_search_xgb, 'Format_A'),
        ("Balanced Random Forest", run_stage2_grid_search_rf, 'Format_A'),
        ("1D-CNN + Bi-GRU (Focal)", run_stage2_grid_search_gru, 'Format_B')
    ]

    stage2_results = {}

    for m_name, fn, fmt in runners:
        print(f"\n[*] >>> [2단계: {m_name}] 하이퍼파라미터 탐색 중...")
        t0 = time.time()
        def_t, oth_5, bst_t, bst_clf = fn(splits)
        elapsed = time.time() - t0

        imp_f1 = (bst_t['strict_f1'] - def_t['strict_f1']) * 100.0
        imp_prauc = (bst_t['strict_pr_auc'] - def_t['strict_pr_auc'])

        print(f"    - 기본 F1: {def_t['strict_f1']*100:.2f}% (PR-AUC: {def_t['strict_pr_auc']:.3f})")
        print(f"    - 최적 F1: {bst_t['strict_f1']*100:.2f}% (PR-AUC: {bst_t['strict_pr_auc']:.3f}) [향상: +{imp_f1:.2f}%p]")
        print(f"    - 최적 파라미터: {bst_t['params']}")

        stage2_results[m_name] = {
            'input_format': fmt,
            'default_trial': def_t,
            'other_5_trials': oth_5,
            'best_trial': bst_t,
            'best_clf': bst_clf,
            'improvement_f1_pts': imp_f1,
            'improvement_prauc': imp_prauc,
            'elapsed_sec': elapsed
        }

    return stage2_results

if __name__ == '__main__':
    run_all_stage2_tuning()
