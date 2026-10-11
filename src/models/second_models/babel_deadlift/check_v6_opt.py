# -*- coding: utf-8 -*-
import os
import sys
PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from src.models.second_models.babel_deadlift.train_and_eval_d3_v6_0 import (
    prepare_v6_0_dataset,
    VIEW_CODE_LIST,
    VIEW_ALIAS_MAP
)

def search_v6_optimal():
    data = np.load(r'd:\src\models\second_models\babel_deadlift\d3_v5_dataset_cache.npz', allow_pickle=True)
    y = data['y']
    views = data['views']

    with open(r'd:\src\models\second_models\babel_deadlift\d3_model_v5_3_evaluation_results.json', 'r', encoding='utf-8') as f:
        d53 = json.load(f)

    print("v5.3 vs v6.0 comparison ready")

if __name__ == '__main__':
    search_v6_optimal()
