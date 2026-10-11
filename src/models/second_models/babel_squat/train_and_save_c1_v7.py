# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: train_and_save_c1_v7.py
Purpose: c1 v7.0 5-Fold 모델 가중치 패키징(c1_model_v7_weights.pkl) 및 oof_submodels_cache.npz 갱신
"""

import os
import sys
import pickle
import time
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v6 import V6A_KEEP_INDICES
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v4 import V4_LEGACY_KEEP_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PKL = os.path.join(CURRENT_DIR, 'c1_model_v7_weights.pkl')
OOF_CACHE = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')


def main():
    print("=" * 100)
    print("  [Step 1] c1 v7.0 46D Deep-Hybrid 5-Fold 앙상블 학습 및 가중치 패키징")
    print("=" * 100)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0].astype(int)
    subjects = ds['subjects']
    N_total = len(y_c1)

    # 1. 29D 정예 생체역학 기하학 피처
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_legacy_18d = X_v2_30d[:, :22][:, V4_LEGACY_KEEP_INDICES]
    v4_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v4_spine_feats.npz'), allow_pickle=True)
    X_v4_spine_10d = v4_npz['v4_spine_feats'].astype(np.float32)
    v5_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v5_spinepose_feats.npz'), allow_pickle=True)
    X_v5_spinepose_6d = v5_npz['v5_spinepose_feats'].astype(np.float32)
    X_v6_29d = np.hstack([X_legacy_18d, X_v4_spine_10d, X_v5_spinepose_6d])[:, V6A_KEEP_INDICES]

    # 2. 17D 딥러닝 시계열 임베딩 피처
    deep_npz = os.path.join(CURRENT_DIR, 'squat_c1_deep_temporal_feats.npz')
    assert os.path.exists(deep_npz), f"딥러닝 피처 파일 누락: {deep_npz}"
    deep_data = np.load(deep_npz, allow_pickle=True)
    deep_embeddings = deep_data['deep_embeddings'].astype(np.float32)
    deep_probs = deep_data['deep_probs'].astype(np.float32)
    X_deep_17d = np.hstack([deep_embeddings, deep_probs[:, None]])

    # 3. 46D 융합 피처셋
    X_v7_46d = np.hstack([X_v6_29d, X_deep_17d]).astype(np.float32)

    # 4. 5-Fold 학습 및 OOF 산출
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_c1_v7 = np.zeros(N_total, dtype=np.float32)
    trained_cb = []
    trained_lgb = []

    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_v7_46d, y_c1, groups=subjects)):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X_v7_46d[trn_idx], y_c1[trn_idx])
        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X_v7_46d[trn_idx], y_c1[trn_idx])

        p_cb = cb.predict_proba(X_v7_46d[val_idx])[:, 1]
        p_lgb = clf.predict_proba(X_v7_46d[val_idx])[:, 1]
        p_fold = 0.60 * p_cb + 0.40 * p_lgb
        oof_c1_v7[val_idx] = p_fold

        f_val = f1_score(y_c1[val_idx], (p_fold >= 0.50).astype(int)) * 100
        print(f"  - Fold {fold+1}/5 완료 | Val F1: {f_val:.2f}% | 피험자: {set(subjects[val_idx])}")
        trained_cb.append(cb)
        trained_lgb.append(clf)

    pred_v7 = (oof_c1_v7 >= 0.50).astype(int)
    p_tot = precision_score(y_c1, pred_v7) * 100
    r_tot = recall_score(y_c1, pred_v7) * 100
    f_tot = f1_score(y_c1, pred_v7) * 100
    auc_tot = roc_auc_score(y_c1, oof_c1_v7) * 100
    missed_tot = int(np.sum((y_c1 == 1) & (pred_v7 == 0)))

    print("\n" + "=" * 100)
    print(f"[*] c1 v7.0 5-Fold 전체 실측 성능:")
    print(f"    - Precision: {p_tot:.2f}% | Recall: {r_tot:.2f}% | F1-Score: {f_tot:.2f}% | ROC-AUC: {auc_tot:.2f}% | 미인식: {missed_tot}건")
    print(f"    - 소요 시간: {time.time()-t0:.1f}초")
    print("=" * 100)

    # 5. 가중치 저장
    os.makedirs(os.path.dirname(WEIGHTS_PKL), exist_ok=True)
    payload = {
        'cb_models': trained_cb,
        'lgb_models': trained_lgb,
        'decision_threshold': 0.50
    }
    with open(WEIGHTS_PKL, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[*] c1 v7.0 가중치 저장 완료: {WEIGHTS_PKL}")

    # 6. oof_submodels_cache.npz 갱신
    if os.path.exists(OOF_CACHE):
        old_cache = np.load(OOF_CACHE)
        cache_dict = {k: old_cache[k] for k in old_cache.files}
        cache_dict['oof_c1_v6'] = cache_dict['oof_c1'] # 기존 v6 백업
        cache_dict['oof_c1'] = oof_c1_v7               # v7로 갱신!
        np.savez_compressed(OOF_CACHE, **cache_dict)
        print(f"[*] OOF 캐시 갱신 완료: {OOF_CACHE} (기존 c1 v6 백업 및 v7 장착)")


if __name__ == '__main__':
    main()
