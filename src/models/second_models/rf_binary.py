# =========================================================================================================
# [모듈 설명]: 2단계 동작 정상/오류 판별 - Balanced Random Forest 이진 분류기 (1:17.0 극단 불균형 대응)
# [입력 데이터]: 97차원 기하 통계 요약 피처 (Format A)
# [출력 목표]: 0=오류(비정상 동작), 1=정상 동작 이진 분류 및 최적 임계값(θ*) 적용
#
# [실행 명령어 (Scheme A: CLI 모드 분리)]:
#   1. 최적 모델 단독 초고속 평가 (기본 모드, 소요시간 ~1초):
#      python -m src.models.second_models.rf_binary
#      또는 python -m src.models.second_models.rf_binary --mode best
#
#   2. 최적 모델 vs 기본값(Default) vs 랜덤 후보 5개 3중 비교 분석 (소요시간 ~10초):
#      python -m src.models.second_models.rf_binary --mode compare
# =========================================================================================================

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

BEST_PARAMS = {
    'max_depth': 16,
    'min_samples_split': 5,
    'n_estimators': 200,
    'opt_threshold': 0.20503480159795717
}

class BalancedRandomForestBinaryClassifier:
    """2단계: Balanced Random Forest 정상(1)/오류(0) 이진 분류기"""

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = 15, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced_subsample'
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BalancedRandomForestBinaryClassifier":
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        inference_time_ms = ((time.time() - t0) / len(X_test)) * 1000.0

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        pos_f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        pos_prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
        pos_rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
        except Exception:
            roc_auc, pr_auc = 0.5, 0.0

        return {
            'model_name': 'Balanced Random Forest',
            'split_name': split_name,
            'num_samples': len(y_test),
            'accuracy': acc,
            'macro_f1': macro_f1,
            'pos_f1': pos_f1,
            'pos_precision': pos_prec,
            'pos_recall': pos_rec,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'inference_time_ms': inference_time_ms,
            'fps': 1000.0 / max(1e-4, inference_time_ms),
            'predictions': preds,
            'probabilities': probs
        }

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    from src.models.second_models.dataset import Stage2DatasetLoader
    from src.models.second_models.metrics_helper import Stage2MetricsHelper

    parser = argparse.ArgumentParser(description="Stage 2: Balanced Random Forest Binary Classifier")
    parser.add_argument('--mode', type=str, choices=['best', 'compare'], default='best',
                        help="Execution mode: 'best' (Fast eval with BEST_PARAMS) or 'compare' (Default vs Random 5 vs Best)")
    args = parser.parse_args()

    splits = Stage2DatasetLoader.load_all_dual_splits()
    X_train, y_train = splits['train']['X_Format_A'], splits['train']['y']

    if args.mode == 'best':
        print("\n" + "=" * 105)
        print("    [2단계: Balanced Random Forest 정상/오류 판별 - 최적 모델(BEST_PARAMS) 단독 초고속 성능 평가]")
        print("=" * 105)
        fit_params = {k: v for k, v in BEST_PARAMS.items() if k not in ['opt_threshold', 'min_samples_split']}
        opt_thresh = BEST_PARAMS.get('opt_threshold', 0.5)
        clf = BalancedRandomForestBinaryClassifier(**fit_params)
        if 'min_samples_split' in BEST_PARAMS:
            clf.model.min_samples_split = BEST_PARAMS['min_samples_split']
        clf.fit(X_train, y_train)
        best_clf = clf
    else:
        print("\n" + "=" * 105)
        print("    [2단계: Balanced Random Forest 정상/오류 판별 - Default vs Random 5 vs Best 3중 비교 분석]")
        print("=" * 105)
        from src.models.second_models.tune_all import run_stage2_grid_search_rf
        def_trial, other_5, best_trial, best_clf = run_stage2_grid_search_rf(splits)
        opt_thresh = best_trial['params'].get('opt_threshold', 0.5)

    split_map = [
        ('Valid Set (검증: 피험자 7~8, 시점 D)', 'valid'),
        ('★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)', 'test_strict'),
        ('Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)', 'test_unseen_subj'),
        ('View Test (시점 일반화: 피험자 1~6, 측면 C)', 'test_unseen_view')
    ]

    print("\n" + "=" * 115)
    print(f"      [2단계 Balanced Random Forest 다중 분할 평가 결과 (최적 임계값 θ* = {opt_thresh:.3f})]")
    print("=" * 115)
    header = f"{'평가 데이터셋 분할':<38} | {'Macro F1':^10} | {'Pos-F1(정상)':^12} | {'Safety F2':^11} | {'PR-AUC':^9} | {'MCC':^8} | {'G-Mean':^8}"
    print(header)
    print("-" * 115)

    for disp_name, key in split_map:
        X_ev = splits[key]['X_Format_A']
        y_ev = splits[key]['y']
        probs = best_clf.predict_proba(X_ev)[:, 1]
        res = Stage2MetricsHelper.evaluate_predictions(y_ev, probs, threshold=opt_thresh)
        print(f"{disp_name:<38} | {res['macro_f1']*100:>8.2f}% | {res['pos_f1']*100:>10.2f}% | {res['pos_f2_safety']*100:>9.2f}% | {res['pr_auc']:>8.3f} | {res['mcc']:>7.3f} | {res['g_mean']:>7.3f}")
    print("=" * 115)

    if args.mode == 'compare':
        print("\n" + "=" * 115)
        print("                  [하이퍼파라미터 튜닝 전후 3중 비교표 (Default vs Random 5 vs Best)]")
        print("=" * 115)
        cmp_header = f"{'구분 (Trial Type)':<20} | {'Strict Macro F1':^16} | {'Strict PR-AUC':^14} | {'파라미터 구성 요약':<45}"
        print(cmp_header)
        print("-" * 115)
        print(f"{'기본 모델 (Default)':<20} | {def_trial['strict_f1']*100:>14.2f}% | {def_trial['strict_pr_auc']:>13.3f} | {str(def_trial['params'])[:45]}")
        for i, t in enumerate(other_5, 1):
            p_str = f"n_est={t['params'].get('n_estimators')}, depth={t['params'].get('max_depth')}, min_split={t['params'].get('min_samples_split')}"
            print(f"{f'랜덤 후보 #{i}':<20} | {t['strict_f1']*100:>14.2f}% | {t['strict_pr_auc']:>13.3f} | {p_str:<45}")
        p_best = f"n_est={best_trial['params'].get('n_estimators')}, depth={best_trial['params'].get('max_depth')}, min_split={best_trial['params'].get('min_samples_split')}"
        print("-" * 115)
        imp = (best_trial['strict_f1'] - def_trial['strict_f1']) * 100
        print(f"{'★ 최적 모델 (Best)':<20} | {best_trial['strict_f1']*100:>14.2f}% | {best_trial['strict_pr_auc']:>13.3f} | {p_best:<45} (성능 향상: +{imp:.2f}%p)")
        print("=" * 115)

