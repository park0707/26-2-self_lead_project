# [실행 명령어]: python -m src.models.first_model.baseline_classifier

import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from src.models.first_model.dataset import Stage1DatasetLoader
from src.models.first_model.metrics_helper import Stage1MetricsHelper

class BaselineActionClassifier:
    """1단계 하한 기준선: Decision Tree / Logistic Regression 9종 분류기"""

    def __init__(self, model_type: str = 'decision_tree', random_state: int = 42):
        self.model_type = model_type
        self.scaler = StandardScaler() if model_type == 'logistic_regression' else None
        
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=500,
                class_weight='balanced',
                solver='lbfgs',
                random_state=random_state
            )
            self.hyperparams = {'model_type': 'Multinomial Logistic Regression', 'max_iter': 500, 'solver': 'lbfgs', 'class_weight': 'balanced'}
        else:
            self.model = DecisionTreeClassifier(
                max_depth=12,
                class_weight='balanced',
                random_state=random_state
            )
            self.hyperparams = {'model_type': 'Decision Tree', 'max_depth': 12, 'class_weight': 'balanced', 'criterion': 'gini'}
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaselineActionClassifier":
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, split_name: str = "Test") -> Dict[str, Any]:
        t0 = time.time()
        preds = self.predict(X_test)
        inference_time_ms = ((time.time() - t0) / max(1, len(X_test))) * 1000.0
        metrics = Stage1MetricsHelper.evaluate_predictions(y_test, preds, inference_time_ms=inference_time_ms)
        metrics['model_name'] = f'Baseline_{self.model_type}'
        metrics['split_name'] = split_name
        metrics['num_samples'] = len(y_test)
        metrics['predictions'] = preds
        return metrics

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    data = Stage1DatasetLoader.load_all_dual_splits()
    X_train, y_train = data['train']['X_Format_A'], data['train']['y_action']

    for m_type in ['decision_tree', 'logistic_regression']:
        clf = BaselineActionClassifier(model_type=m_type)
        Stage1MetricsHelper.print_model_header(
            model_name=f"Baseline ({m_type.replace('_', ' ').title()})",
            hyperparams=clf.hyperparams,
            train_samples=X_train.shape[0],
            feat_dim=f"Format A (97차원 기하 통계 요약 피처)"
        )
        
        t0 = time.time()
        clf.fit(X_train, y_train)
        print(f"[*] 모델 학습 완료 (소요 시간: {time.time()-t0:.2f}초)")

        split_results = {}
        split_map = [
            ('Valid Set (검증: 피험자 7~8, 시점 D)', 'valid'),
            ('★ Strict Test (복합 홀드아웃: 피험자 9, 측면 C)', 'test_strict'),
            ('Subj Test (인물 일반화: 피험자 9, 시점 A,B,E)', 'test_unseen_subj'),
            ('View Test (시점 일반화: 피험자 1~6, 측면 C)', 'test_unseen_view')
        ]

        strict_res = None
        for display_name, key in split_map:
            res = clf.evaluate(data[key]['X_Format_A'], data[key]['y_action'], split_name=key)
            split_results[display_name] = res
            if key == 'test_strict':
                strict_res = res

        Stage1MetricsHelper.print_multi_split_summary(split_results)
        if strict_res is not None:
            Stage1MetricsHelper.print_class_breakdown_report(strict_res, data['test_strict']['y_action'])
            Stage1MetricsHelper.print_top_confusion_analysis(strict_res, top_k=3)
