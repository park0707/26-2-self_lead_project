# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 03_바벨_스티프_데드리프트
Script: package_stiff_integrated_bundle.py
Purpose: 통합 모델 전수 가중치 및 상위 메타신경망 번들 패키징
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.train_and_package_sd3_v4 import DilatedTCN
from src.models.second_models.babel_stiff_deadlift.integrated.stiff_integrated_predictor import StiffMetaNN

OOF_CACHE_PATH = os.path.join(CURRENT_DIR, "integrated_stiff_oof_cache.npz")
RESULTS_JSON_PATH = os.path.join(CURRENT_DIR, "integrated_tournament_results.json")
BUNDLE_SAVE_PATH = os.path.join(CURRENT_DIR, "stiff_integrated_v1_bundle.pkl")

SD2_CACHE_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_spine_ambiguity_levels.npz"
SD3_CACHE_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl"
DATA_FULL33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"

from src.models.second_models.babel_stiff_deadlift.evaluate_sd1_full33 import extract_sd1_features_from_full33

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 통합 진단 모델 전수 번들 패키징")
    print("=" * 80)

    # 1. 캐시 및 결과 로드
    cache = np.load(OOF_CACHE_PATH, allow_pickle=True)
    seq_ids = cache['seq_ids']
    N = len(seq_ids)
    y_clean = cache['y_clean']
    oof_p1 = cache['oof_sd1']
    oof_p2 = cache['oof_sd2']
    oof_p3 = cache['oof_sd3']
    y_sd1 = cache['y_sd1']
    y_sd2 = cache['y_sd2']
    y_sd3 = cache['y_sd3']
    views = cache['views']

    # 2. 서브모델 전수 학습 가중치 확보
    print("[*] 1. sd1 전수 모델 학습 중...")
    X_sd1 = np.zeros((N, 16), dtype=np.float32)
    for i, sid in enumerate(seq_ids):
        fpath = os.path.join(DATA_FULL33_DIR, f"{sid}.npz")
        kps = np.load(fpath, allow_pickle=True)['kps_33_3d_64f']
        X_sd1[i] = extract_sd1_features_from_full33(kps)

    sd1_cb = CatBoostClassifier(iterations=150, learning_rate=0.04, depth=4, l2_leaf_reg=5.0,
                                auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1)
    sd1_cb.fit(X_sd1, y_sd1)

    sd1_lgb = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=12,
                                 class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
    sd1_lgb.fit(X_sd1, y_sd1)

    print("[*] 2. sd2 전수 모델 학습 중...")
    sd2_data = np.load(SD2_CACHE_PATH, allow_pickle=True)
    sd2_sids = list(sd2_data['sequence_ids'])
    sd2_map = {s: i for i, s in enumerate(sd2_sids)}
    reorder_sd2 = [sd2_map[s] for s in seq_ids]
    X_sd2 = sd2_data['features_30d'][reorder_sd2]

    sd2_cb = CatBoostClassifier(iterations=200, learning_rate=0.04, depth=5, l2_leaf_reg=4.0,
                                auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1)
    sd2_cb.fit(X_sd2, y_sd2)

    sd2_lgb = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=16,
                                 class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
    sd2_lgb.fit(X_sd2, y_sd2)

    print("[*] 3. sd3 전수 모델 로드 중...")
    with open(SD3_CACHE_PATH, 'rb') as fp:
        c3 = pickle.load(fp)
    c3_df = c3['df']
    c3_sids = list(c3_df['filename'].values)
    c3_map = {s: i for i, s in enumerate(c3_sids)}
    reorder_sd3 = [c3_map[s] for s in seq_ids]
    X_sd3_36 = c3['X_36'][reorder_sd3]
    X_sd3_seq = c3['X_seq'][reorder_sd3]

    sd3_bundle_path = os.path.join(CURRENT_DIR, "..", "weights", "sd3_v4_bundle.pkl")
    with open(sd3_bundle_path, 'rb') as fp:
        sd3_b = pickle.load(fp)
    sd3_lgb = sd3_b['lgb_model']
    sd3_cb = sd3_b['cb_model']
    sd3_tcn_sd = sd3_b['tcn_state_dict']
    sd3_fw = sd3_b['view_fusion_weights']

    # 3. 메타 신경망 전수 학습
    print("[*] 4. 메타 신경망 (StiffMetaNN) 전수 학습 중...")
    cert1 = np.abs(oof_p1 - 0.5) * 2.0
    cert2 = np.abs(oof_p2 - 0.5) * 2.0
    cert3 = np.abs(oof_p3 - 0.5) * 2.0
    margin_min = np.minimum(np.minimum(cert1, cert2), cert3)
    p2_rule = np.maximum(oof_p2, oof_p1)
    p_prod_clean = (1.0 - oof_p1) * (1.0 - p2_rule) * (1.0 - oof_p3)

    v_dummies = pd.get_dummies(views).values.astype(np.float32)
    meta_X = np.column_stack([
        oof_p1, p2_rule, oof_p3, cert1, cert2, cert3, margin_min, p_prod_clean, v_dummies
    ]).astype(np.float32)

    scaler = StandardScaler()
    meta_X_s = scaler.fit_transform(meta_X)

    torch.manual_seed(42)
    meta_nn = StiffMetaNN(in_features=13, hidden_dim=16)
    opt = torch.optim.AdamW(meta_nn.parameters(), lr=0.01, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]))

    xt = torch.tensor(meta_X_s, dtype=torch.float32)
    yt = torch.tensor(y_clean, dtype=torch.float32)

    meta_nn.train()
    for _ in range(35):
        for bx, by in DataLoader(TensorDataset(xt, yt), batch_size=32, shuffle=True):
            opt.zero_grad()
            crit(meta_nn(bx), by).backward()
            opt.step()
    meta_nn.eval()

    bundle_data = {
        'model_name': 'stiff_integrated_v1',
        'sd1_cb': sd1_cb,
        'sd1_lgb': sd1_lgb,
        'sd2_cb': sd2_cb,
        'sd2_lgb': sd2_lgb,
        'sd3_lgb': sd3_lgb,
        'sd3_cb': sd3_cb,
        'sd3_tcn_state_dict': sd3_tcn_sd,
        'sd3_fusion_weights': sd3_fw,
        'meta_scaler': scaler,
        'meta_nn_state_dict': meta_nn.state_dict(),
        'tau_clean': 0.5141,
        'tau_defect': 0.4859
    }

    with open(BUNDLE_SAVE_PATH, 'wb') as fp:
        pickle.dump(bundle_data, fp)

    print(f"\n[+] 통합 번들 가중치 파일 패키징 완료: {BUNDLE_SAVE_PATH}")

if __name__ == '__main__':
    main()
