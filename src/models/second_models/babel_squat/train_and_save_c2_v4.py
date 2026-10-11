# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Alignment)
Module: train_and_save_c2_v4.py
Purpose: c2 v4 9D 정예 모델 5-Fold 학습, 가중치 저장 및 oof_submodels_cache.npz 갱신
"""

import os
import sys
import pickle
import time
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c2_model_v3 import V2_TO_V3_INDICES
from src.models.second_models.babel_squat.c2_model_v4 import V3_TO_V4_INDICES, FEATURE_NAMES_9, DROPPED_FEATURES_FROM_V3

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(CURRENT_DIR, 'c2_model_v4_weights.pkl')
OOF_CACHE = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')


def main():
    print("=" * 110)
    print("  [Step 1] c2 v4.0 9D 정예 모델 5-Fold 학습 및 공식 가중치 저장")
    print("=" * 110)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c2 = ds['Y_defects'][:, 1].astype(int)
    subjects = ds['subjects']
    N_total = len(y_c2)

    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names = list(h_data['feature_names'])
    elite_idx = [i for i, n in enumerate(feat_names) if n != 'c2_gaze_hysteresis_rmse']
    X_18d = X_19d[:, elite_idx]

    g_data = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats = g_data['gated_feats'].astype(np.float32)[:, :2]
    X_20d = np.concatenate([X_18d, gated_feats], axis=1)

    X_13d = X_20d[:, V2_TO_V3_INDICES]
    X_9d = X_13d[:, V3_TO_V4_INDICES]

    print(f"[*] 피처 압축 완료: 13D -> 9D (4대 간섭/노이즈 피처 소거: {DROPPED_FEATURES_FROM_V3})")
    print(f"[*] 데이터 규격: N={N_total}, 피처 차원={X_9d.shape[1]}D")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_c2_v4 = np.zeros(N_total, dtype=np.float32)
    trained_cb = []
    trained_lgb = []

    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_9d, y_c2, groups=subjects)):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X_9d[trn_idx], y_c2[trn_idx])

        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X_9d[trn_idx], y_c2[trn_idx])

        p_cb = cb.predict_proba(X_9d[val_idx])[:, 1]
        p_lgb = clf.predict_proba(X_9d[val_idx])[:, 1]
        p_fold = 0.60 * p_cb + 0.40 * p_lgb
        oof_c2_v4[val_idx] = p_fold

        f_val = f1_score(y_c2[val_idx], (p_fold >= 0.50).astype(int)) * 100
        print(f"  - Fold {fold+1}/5 완료 | Val F1: {f_val:.2f}% | 피험자: {set(subjects[val_idx])}")
        trained_cb.append(cb)
        trained_lgb.append(clf)

    pred_v4 = (oof_c2_v4 >= 0.50).astype(int)
    p_tot = precision_score(y_c2, pred_v4) * 100
    r_tot = recall_score(y_c2, pred_v4) * 100
    f_tot = f1_score(y_c2, pred_v4) * 100
    auc_tot = roc_auc_score(y_c2, oof_c2_v4) * 100
    missed_tot = int(np.sum((y_c2 == 1) & (pred_v4 == 0)))

    print("\n" + "=" * 105)
    print(f"[*] c2 v4.0 9D 최종 확정 실측 성적표 (N=1,906):")
    print(f"    - Precision: {p_tot:.2f}% | Recall: {r_tot:.2f}% | F1-Score: {f_tot:.2f}% 🏆 | ROC-AUC: {auc_tot:.2f}% | 미인식: {missed_tot}건")
    print(f"    - 학습 소요 시간: {time.time()-t0:.1f}초")
    print("=" * 105)

    # 가중치 저장
    payload = {
        'cb_models': trained_cb,
        'lgb_models': trained_lgb,
        'decision_threshold': 0.50,
        'feature_names': FEATURE_NAMES_9,
        'dropped_features': DROPPED_FEATURES_FROM_V3
    }
    with open(WEIGHTS_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[*] c2 v4.0 가중치 정식 저장 완료: {WEIGHTS_PATH}")

    # OOF 캐시 갱신
    if os.path.exists(OOF_CACHE):
        old_cache = np.load(OOF_CACHE)
        cache_dict = {k: old_cache[k] for k in old_cache.files}
        cache_dict['oof_c2_v3'] = old_cache['oof_c2']  # 기존 v3 보존
        cache_dict['oof_c2'] = oof_c2_v4               # v4로 갱신!
        np.savez_compressed(OOF_CACHE, **cache_dict)
        print(f"[*] OOF 캐시 갱신 완료: {OOF_CACHE} (c2 v4.0 정예 확률 장착, v3 백업 완료)")


if __name__ == '__main__':
    main()
