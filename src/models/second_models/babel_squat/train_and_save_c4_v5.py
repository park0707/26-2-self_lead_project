# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_save_c4_v5.py
Purpose: c4 v5.0 공식 25D 피처 기반 5-Fold StratifiedGroupKFold 학습, 공식 가중치 저장 및 전수 평가
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c4_model_v5 import (
    extract_c4_v5_features_from_seq,
    FEATURE_NAMES_25_C4,
    BabelSquatC4ModelV5
)

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_CACHE_PATH = r"D:\Data\피처_데이터\squat_c4_v5_25d_feats.npz"
WEIGHTS_PATH = r"D:\src\models\second_models\babel_squat\c4_model_v5_weights.pkl"
OOF_PATH = r"D:\Data\피처_데이터\oof_c4_v5_predictions.npz"


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c4(발바닥 지면 고정) v5.0 공식 가중치 학습 및 배포 파이프라인", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    y_c4 = master['Y_defects_golden'][:, 3].astype(int)
    subjects = master['subjects']
    views = master['views']
    N = len(seq_ids)

    d_cache = np.load(FEATS_CACHE_PATH)
    X_v5 = d_cache['X_c4_v5_25d']
    print(f"[*] v5 25D 피처 캐시 로드 완료: {X_v5.shape}", flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    trained_cb = []
    trained_lgb = []
    trained_rf = []

    oof_cb = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_rf = np.zeros(N, dtype=np.float32)

    print(f"[*] 9명 피험자 완전 격리 5-Fold 학습 시작...", flush=True)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_v5, y_c4, groups=subjects), 1):
        cb = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
                                random_seed=42 + fold, verbose=0)
        cb.fit(X_v5[trn_idx], y_c4[trn_idx])

        clf = lgb.LGBMClassifier(n_estimators=180, max_depth=6, learning_rate=0.03, num_leaves=31,
                                 random_state=42 + fold, verbose=-1, n_jobs=-1)
        clf.fit(X_v5[trn_idx], y_c4[trn_idx])

        rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42 + fold, n_jobs=-1)
        rf.fit(X_v5[trn_idx], y_c4[trn_idx])

        oof_cb[val_idx] = cb.predict_proba(X_v5[val_idx])[:, 1]
        oof_lgb[val_idx] = clf.predict_proba(X_v5[val_idx])[:, 1]
        oof_rf[val_idx] = rf.predict_proba(X_v5[val_idx])[:, 1]

        p_fold = 0.60 * oof_cb[val_idx] + 0.30 * oof_lgb[val_idx] + 0.10 * oof_rf[val_idx]
        f1_fold = f1_score(y_c4[val_idx], (p_fold >= 0.50).astype(int)) * 100
        auc_fold = roc_auc_score(y_c4[val_idx], p_fold) * 100
        print(f"    - Fold {fold}/5: Val F1: {f1_fold:5.2f}%, AUC: {auc_fold:5.2f}%", flush=True)

        trained_cb.append(cb)
        trained_lgb.append(clf)
        trained_rf.append(rf)

    # 공식 가중치 저장
    v5_model = BabelSquatC4ModelV5()
    v5_model.cb_models = trained_cb
    v5_model.lgb_models = trained_lgb
    v5_model.rf_models = trained_rf
    v5_model.weights = (0.60, 0.30, 0.10)
    v5_model.is_fitted = True
    v5_model.save_weights(WEIGHTS_PATH)

    oof_ens = 0.60 * oof_cb + 0.30 * oof_lgb + 0.10 * oof_rf
    np.savez_compressed(
        OOF_PATH,
        oof_preds=oof_ens,
        oof_cb=oof_cb,
        oof_lgb=oof_lgb,
        oof_rf=oof_rf,
        y_true=y_c4,
        views=views,
        subjects=subjects
    )
    print(f"[*] OOF 예측값 저장 완료: {OOF_PATH}", flush=True)

    f1_all = f1_score(y_c4, (oof_ens >= 0.50).astype(int)) * 100.0
    acc_all = accuracy_score(y_c4, (oof_ens >= 0.50).astype(int)) * 100.0
    auc_all = roc_auc_score(y_c4, oof_ens) * 100.0
    print("\n" + "=" * 115, flush=True)
    print(f"  [vidio-health-trainer] c4 v5.0 공식 모델 5-Fold 이진 성적: F1: {f1_all:.2f}%, Acc: {acc_all:.2f}%, AUC: {auc_all:.2f}%", flush=True)
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()
