# -*- coding: utf-8 -*-
import os, sys, pickle, json, numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

CURRENT_DIR = r"D:\src\models\second_models\babel_stiff_deadlift"
PROJECT_ROOT = "d:/"
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 1. NPZ 데이터 로드
npz_path = os.path.join(CURRENT_DIR, "stiff_spine_ambiguity_levels.npz")
data = np.load(npz_path, allow_pickle=True)
y_all = data['labels']
amb = data['ambiguity_levels']
N = len(y_all)

# 2. sd2_model_v3_weights.pkl 로드
bundle_path = os.path.join(CURRENT_DIR, "sd2_model_v3_weights.pkl")
with open(bundle_path, 'rb') as f:
    bundle = pickle.load(f)

t2 = bundle['tier2_metrics']
print("=== [기존 Tier 2 실서비스 3-State 수치 (912개 기준)] ===")
print(f"  - 3-State F1-Score : {t2['three_state_f1']*100:.2f}%")
print(f"  - 보류율 (Abstain) : {t2['abstain_rate']*100:.2f}%")
print(f"  - 커버리지 (Cover) : {t2['coverage']*100:.2f}%")
print(f"  - 확정 정확도      : {t2['accuracy']*100:.2f}%")

# 3. 만약 Level 4(123개)를 '무조건 보류'로 처리하고 Level 1~3은 3-State로 처리할 때:
# 전체 1035개 중 확정된 개수 = 912 * 0.8794 = 802개
total_det_count = int(912 * t2['coverage'])
total_coverage = total_det_count / 1035.0
total_abstain = 1.0 - total_coverage
print("\n=== [전체 1,035개 전수(Level 4 포함) 기준 단일 통합 3-State 수치] ===")
print(f"  - 3-State F1-Score : {t2['three_state_f1']*100:.2f}%")
print(f"  - 보류율 (Abstain) : {total_abstain*100:.2f}% (총 {1035 - total_det_count}개)")
print(f"  - 커버리지 (Cover) : {total_coverage*100:.2f}% (총 {total_det_count}개)")
print(f"  - 확정 정확도      : {t2['accuracy']*100:.2f}%")
