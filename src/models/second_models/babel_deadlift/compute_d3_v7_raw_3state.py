import os
import sys
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d3_model_v7_0 import BabelDeadliftD3ModelV7
from src.utils.d3_ambiguity_levels import extract_d3_ambiguity_levels # 임시 경로거나 없으면 우회

def main():
    # 데이터 로드
    X_path = os.path.join(PROJECT_ROOT, "data", "extracted_features", "babel_deadlift", "X_d3.npy")
    y_path = os.path.join(PROJECT_ROOT, "data", "extracted_features", "babel_deadlift", "y_d3.npy")
    subj_path = os.path.join(PROJECT_ROOT, "data", "extracted_features", "babel_deadlift", "subj_d3.npy")
    
    if not os.path.exists(X_path):
        # 만약 원천 배열이 없으면? 
        pass

    # 모델 가중치 로드
    model = BabelDeadliftD3ModelV7()
    weights_path = os.path.join(CURRENT_DIR, "weights", "d3_model_v7_0_weights.pkl")
    try:
        model.load_weights(weights_path)
    except:
        print("Failed to load weights")
        return

    # test_d3_model_v7_0_inference.py 에서 데이터를 어떻게 로드하는지 확인 필요
if __name__ == '__main__':
    main()
