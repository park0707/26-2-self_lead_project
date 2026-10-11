# =========================================================================================================
# [모듈 설명]: 1단계 9종 운동 대분류 - 시계열 Bi-GRU (0.7) + 정형 XGBoost (0.3) 하이브리드 소프트 앙상블 최고 성능 분류기
# [입력 데이터]: Format B (64, 91) 시계열 텐서 + Format A (151) 요약 통계 피처 (Condition 3 MediaPipe 3D)
# [출력 목표]: 9종 운동 대분류 Multi-class 예측 (Strict Macro F1: 87.59%+, Top-2 Acc: 98.99%)
#
# [실행 명령어 (CLI 테스트)]:
#   1. 1단계 복합 하이브리드 앙상블 모델 전수 성능 평가:
#      python -m src.models.first_model.ensemble_classifier
# =========================================================================================================

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import xgboost as xgb
import torch

from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.first_model.metrics_helper import Stage1MetricsHelper
from src.models.first_model.gru_classifier import TemporalGRUActionClassifier, BEST_PARAMS as GRU_BEST_PARAMS

OUTPUT_METRICS_JSON = 'd:/Data/피처_데이터/tuning_results/best_stage1_hybrid_ensemble_bi_gru__xgboost.json'

class HybridStage1EnsembleClassifier:
    """1단계: 시계열 Bi-GRU + 정형 XGBoost 하이브리드 소프트 앙상블 분류기"""

    def __init__(self, gru_weight: float = 0.7, xgb_weight: float = 0.3, temperature: float = 1.2):
        self.gru_weight = gru_weight
        self.xgb_weight = xgb_weight
        self.temperature = temperature

        self.gru_model = TemporalGRUActionClassifier(**GRU_BEST_PARAMS)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X_train_b: np.ndarray, X_train_a: np.ndarray, y_train: np.ndarray,
            X_val_b: Optional[np.ndarray] = None, X_val_a: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            verbose: bool = False):
        if verbose:
            print(f"[*] Training Bi-GRU on Format B {X_train_b.shape}...")
        self.gru_model.fit(X_train_b, y_train)

        if verbose:
            print(f"[*] Training XGBoost on Format A {X_train_a.shape}...")
        eval_set = [(X_val_a, y_val)] if (X_val_a is not None and y_val is not None) else None
        self.xgb_model.fit(X_train_a, y_train, eval_set=eval_set, verbose=False)

        self.is_fitted = True
        return self

    def predict_proba(self, X_b: np.ndarray, X_a: np.ndarray) -> np.ndarray:
        gru_probs = self.gru_model.predict_proba(X_b, temperature=self.temperature)
        xgb_probs = self.xgb_model.predict_proba(X_a)
        ens_probs = (self.gru_weight * gru_probs) + (self.xgb_weight * xgb_probs)
        return ens_probs

    def predict(self, X_b: np.ndarray, X_a: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_b, X_a)
        return np.argmax(probs, axis=-1)

    def evaluate(self, X_b: np.ndarray, X_a: np.ndarray, y: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        probs = self.predict_proba(X_b, X_a)
        preds = np.argmax(probs, axis=-1)
        inference_time_ms = ((time.time() - t0) / max(1, len(y))) * 1000.0

        # Calculate Top-2 accuracy
        top2_indices = np.argsort(probs, axis=-1)[:, -2:]
        top2_correct = sum(true_label in top2 for true_label, top2 in zip(y, top2_indices))
        top2_accuracy = (top2_correct / len(y)) * 100.0

        metrics = Stage1MetricsHelper.evaluate_predictions(y, preds, inference_time_ms=inference_time_ms)
        metrics['top2_accuracy'] = round(top2_accuracy, 2)
        metrics['model_name'] = f'Hybrid Ensemble (Bi-GRU {self.gru_weight:.1f} + XGBoost {self.xgb_weight:.1f})'
        metrics['split_name'] = split_name
        metrics['num_samples'] = len(y)
        metrics['predictions'] = preds
        metrics['probabilities'] = probs
        return metrics

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("\n" + "=" * 105)
    print("    [1단계: 시계열 Bi-GRU + 정형 XGBoost 9종 운동 대분류 - 하이브리드 앙상블 복합 성능 평가]")
    print("=" * 105)

    data = Stage1DatasetLoader.load_all_dual_splits()

    ensemble_params = {
        'gru_weight': 0.7,
        'xgb_weight': 0.3,
        'temperature': 1.2,
        'xgb_n_estimators': 300,
        'xgb_max_depth': 6,
        'gru_conv_filters': GRU_BEST_PARAMS['conv_filters'],
        'gru_hidden_dim': GRU_BEST_PARAMS['hidden_dim'],
        'gru_num_layers': GRU_BEST_PARAMS['num_layers'],
        'gru_dropout': GRU_BEST_PARAMS['dropout'],
        'gru_lr': GRU_BEST_PARAMS['lr'],
        'gru_epochs': GRU_BEST_PARAMS['epochs'],
        'gru_batch_size': GRU_BEST_PARAMS['batch_size']
    }

    Stage1MetricsHelper.print_model_header(
        model_name="Hybrid Ensemble (Bi-GRU 0.7 + XGBoost 0.3 with T=1.2)",
        hyperparams=ensemble_params,
        train_samples=data['train']['X_Format_B'].shape[0],
        feat_dim="Format B (64x91) + Format A (151D) [Condition 3 MediaPipe 3D 확장]"
    )

    ensemble = HybridStage1EnsembleClassifier(gru_weight=0.7, xgb_weight=0.3, temperature=1.2)
    ensemble.fit(
        X_train_b=data['train']['X_Format_B'],
        X_train_a=data['train']['X_Format_A'],
        y_train=data['train']['y_action'],
        X_val_b=data['valid']['X_Format_B'],
        X_val_a=data['valid']['X_Format_A'],
        y_val=data['valid']['y_action']
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
        res = ensemble.evaluate(data[key]['X_Format_B'], data[key]['X_Format_A'], data[key]['y_action'], split_name=key)
        split_results[display_name] = res
        if key == 'test_strict':
            strict_res = res

    res_train = ensemble.evaluate(data['train']['X_Format_B'], data['train']['X_Format_A'], data['train']['y_action'], split_name='train')
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

    Stage1MetricsHelper.save_best_params_json(
        model_name="Hybrid Ensemble (Bi-GRU + XGBoost)",
        best_params=ensemble_params,
        summary_metrics={
            'valid_macro_f1': split_results['Valid Set (검증: 피험자 7~8, 시점 D)']['macro_f1'],
            'strict_test_macro_f1': strict_res['macro_f1'] if strict_res else 0.0,
            'top2_accuracy': strict_res['top2_accuracy'] if strict_res else 0.0,
            'advanced_metrics': adv_metrics
        }
    )
