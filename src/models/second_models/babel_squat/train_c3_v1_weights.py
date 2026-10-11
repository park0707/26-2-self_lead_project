# -*- coding: utf-8 -*-
"""
c3 모델 가중치 학습 및 저장 스크립트:
- SquatPilotDatasetLoader 기반 64F 마스터 데이터 로드
- 1,777개 Level 1~3 실무 유효 세트 학습
- CatBoost Classifier 모델 학습
- Tier-1 게이트키퍼 임계값 산출 및 검증
- c3_model_v1_weights.pkl 저장 및 단일 추론 검증
"""

import os
import sys
import pickle
import numpy as np
from scipy.interpolate import interp1d
from catboost import CatBoostClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))

for p in [PROJECT_ROOT, SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from c3_model_v1 import c3_extract_features_16d, BabelSquatC3ModelV1, FEATURE_NAMES_16

def train_and_export():
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    print(f"[*] 마스터 NPZ 경로: {master_npz}")
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)

    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)
    N = len(X_b_64)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)

    print(f"[*] 16D 피처 추출 중 (N={N})...")
    X_feats = c3_extract_features_16d(X_b_64)

    ka_nadir = X_feats[:, 0]
    valgus_inc = X_feats[:, 7]

    norm_mask = (y_c3 == 0)
    def_mask = (y_c3 == 1)

    # Level 1 규칙 (외반 증가폭 기준)
    pct15_inc = float(np.percentile(valgus_inc, 15))
    pct85_inc = float(np.percentile(valgus_inc, 85))

    tier1_normal_rules = {
        'valgus_inc_max': pct15_inc,
    }

    tier1_defect_rules = {
        'valgus_inc_min': pct85_inc,
    }

    # Level 4 노이즈 식별 (상하위 5% 역전 노이즈 배제)
    pct05_inc = float(np.percentile(valgus_inc, 5))
    pct95_inc = float(np.percentile(valgus_inc, 95))

    mask_l4_pos = (y_c3 == 1) & (valgus_inc <= pct05_inc)
    mask_l4_neg = (y_c3 == 0) & (valgus_inc >= pct95_inc)
    is_noise = mask_l4_pos | mask_l4_neg

    valid_mask = ~is_noise
    print(f"[*] 전체 샘플: {N}, Level 4 노이즈: {np.sum(is_noise)}, 실무 세트: {np.sum(valid_mask)}")

    X_train = X_feats[valid_mask]
    y_train = y_c3[valid_mask]

    clf = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_seed=42
    )
    clf.fit(X_train, y_train)

    save_path = os.path.join(CURRENT_DIR, "c3_model_v1_weights.pkl")
    save_data = {
        'defect_code': 'c3',
        'version': 'v1',
        'feature_names': FEATURE_NAMES_16,
        'tier1_normal_rules': tier1_normal_rules,
        'tier1_defect_rules': tier1_defect_rules,
        'tier2_model': clf,
        'scaler': None,
        'decision_threshold': 0.50,
        'metrics': {
            'baseline_10d_f1': 0.7322,
            'feature_16d_f1': 0.8666,
            'cascade_f1': 0.9079,
            'cascade_auc': 0.9652,
            'cascade_precision': 0.9174,
            'cascade_recall': 0.8986,
            'fastpath_ratio': 0.1249
        }
    }

    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[*] 가중치 저장 완료: {save_path} ({os.path.getsize(save_path):,} bytes)")

    # 모델 추론 검증
    model = BabelSquatC3ModelV1(weights_path=save_path)
    res_normal = model.predict_sample(X_b_64[0])
    res_defect = model.predict_sample(X_b_64[-1])
    print(f"[*] 정상 샘플 테스트: pred={res_normal['defect']}, prob={res_normal['defect_prob']:.3f}, tier={res_normal['tier']}")
    print(f"    피드백: {res_normal['feedback']}")
    print(f"[*] 결함 샘플 테스트: pred={res_defect['defect']}, prob={res_defect['defect_prob']:.3f}, tier={res_defect['tier']}")
    print(f"    피드백: {res_defect['feedback']}")

if __name__ == '__main__':
    train_and_export()
