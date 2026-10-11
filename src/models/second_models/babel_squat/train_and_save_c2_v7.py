# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_save_c2_v7.py
Purpose: c2 v7.0 공식 19D 적응형 피처 추출, 5-Fold StratifiedGroupKFold 학습 및 가중치 영구 배포
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

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c2_model_v7 import (
    extract_c2_v7_19d_features_from_seq,
    FEATURE_NAMES_19_C2,
    BabelSquatC2ModelV7
)

RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
FEATS_CACHE_PATH = r"D:\Data\피처_데이터\squat_c2_v7_19d_feats.npz"
WEIGHTS_PATH = r"D:\src\models\second_models\babel_squat\c2_model_v7_weights.pkl"
OOF_PATH = r"D:\Data\피처_데이터\oof_c2_v7_predictions.npz"


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c2(고개 정면) v7.0 공식 가중치 학습 및 배포 파이프라인 가동", flush=True)
    print("=" * 115, flush=True)

    master = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master['sequence_ids']
    y_c2 = master['Y_defects_golden'][:, 1].astype(int)
    subjects = master['subjects']
    views = master['views']
    N = len(seq_ids)

    # 1. 19D 피처 로드 또는 추출
    if os.path.exists(FEATS_CACHE_PATH):
        print(f"[*] [1/4] 기 저장된 v7.0 19D 피처 캐시 로드: {FEATS_CACHE_PATH}", flush=True)
        d_c = np.load(FEATS_CACHE_PATH)
        X_v7 = d_c['X_c2_v7_19d']
    else:
        print(f"[*] [1/4] 2,605개 시퀀스로부터 v7.0 19D 피처 추출...", flush=True)
        t0 = time.time()
        feats = []
        for i, sid in enumerate(seq_ids, 1):
            fp = os.path.join(RAW_DIR, f"{sid}.npz")
            kps = np.load(fp)['kps_33_3d_64f']
            f19 = extract_c2_v7_19d_features_from_seq(kps)
            feats.append(f19)
            if i % 500 == 0 or i == N:
                print(f"    - 진행률: {i:4d}/{N} ({i/N*100:5.1f}%) | 경과: {time.time()-t0:.1f}초", flush=True)
        X_v7 = np.array(feats, dtype=np.float32)

        print(f"[*] [2/4] v7 19D 피처 캐시 영구 저장: {FEATS_CACHE_PATH}", flush=True)
        np.savez_compressed(
            FEATS_CACHE_PATH,
            X_c2_v7_19d=X_v7,
            feature_names=np.array(FEATURE_NAMES_19_C2),
            sequence_ids=seq_ids
        )

    # 2. 5-Fold StratifiedGroupKFold 학습
    print(f"[*] [3/4] 9명 피험자 완전 격리 5-Fold 학습 시작...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    trained_cb = []
    trained_lgb = []
    oof_preds = np.zeros(N, dtype=np.float32)
    oof_cb_preds = np.zeros(N, dtype=np.float32)
    oof_lgb_preds = np.zeros(N, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_v7, y_c2, groups=subjects), 1):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
                                nan_mode='Min', random_seed=42 + fold, verbose=0)
        cb.fit(X_v7[trn_idx], y_c2[trn_idx])

        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31,
                                 random_state=42 + fold, verbose=-1, n_jobs=-1)
        clf.fit(X_v7[trn_idx], y_c2[trn_idx])

        p_cb = cb.predict_proba(X_v7[val_idx])[:, 1]
        p_lgb = clf.predict_proba(X_v7[val_idx])[:, 1]
        p_fold = 0.60 * p_cb + 0.40 * p_lgb

        oof_cb_preds[val_idx] = p_cb
        oof_lgb_preds[val_idx] = p_lgb
        oof_preds[val_idx] = p_fold

        trained_cb.append(cb)
        trained_lgb.append(clf)

        f1_fold = f1_score(y_c2[val_idx], (p_fold >= 0.50).astype(int)) * 100
        auc_fold = roc_auc_score(y_c2[val_idx], p_fold) * 100
        print(f"    - Fold {fold}/5: Val F1: {f1_fold:5.2f}%, AUC: {auc_fold:5.2f}% (검증 피험자: {np.unique(subjects[val_idx])})", flush=True)

    # 공식 가중치 저장
    print(f"[*] [4/4] 공식 가중치 저장: {WEIGHTS_PATH}", flush=True)
    v7_model = BabelSquatC2ModelV7()
    v7_model.cb_models = trained_cb
    v7_model.lgb_models = trained_lgb
    v7_model.is_fitted = True
    v7_model.save_weights(WEIGHTS_PATH)

    # OOF 예측값 저장
    np.savez_compressed(
        OOF_PATH,
        oof_preds=oof_preds,
        oof_cb_preds=oof_cb_preds,
        oof_lgb_preds=oof_lgb_preds,
        y_true=y_c2,
        views=views,
        subjects=subjects
    )
    print(f"[*] OOF 예측값 영구 저장 완료: {OOF_PATH}", flush=True)

    # 종합 평가 성적 출력
    f1_all = f1_score(y_c2, (oof_preds >= 0.50).astype(int)) * 100.0
    acc_all = accuracy_score(y_c2, (oof_preds >= 0.50).astype(int)) * 100.0
    auc_all = roc_auc_score(y_c2, oof_preds) * 100.0
    prec_all = precision_score(y_c2, (oof_preds >= 0.50).astype(int)) * 100.0
    rec_all = recall_score(y_c2, (oof_preds >= 0.50).astype(int)) * 100.0

    print("\n" + "=" * 115, flush=True)
    print("  [vidio-health-trainer] c2 v7.0 5-Fold 피험자 완전 격리 이진 성능 요약", flush=True)
    print("=" * 115, flush=True)
    print(f"  - 전체 F1 Score : {f1_all:6.2f}%", flush=True)
    print(f"  - 전체 Accuracy : {acc_all:6.2f}%", flush=True)
    print(f"  - 전체 ROC-AUC  : {auc_all:6.2f}%", flush=True)
    print(f"  - Precision     : {prec_all:6.2f}%", flush=True)
    print(f"  - Recall        : {rec_all:6.2f}%", flush=True)
    print("-" * 115, flush=True)
    print("  5대 시점별 F1 / AUC:")
    for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
        m_v = (views == v)
        f1_v = f1_score(y_c2[m_v], (oof_preds[m_v] >= 0.50).astype(int)) * 100.0
        auc_v = roc_auc_score(y_c2[m_v], oof_preds[m_v]) * 100.0
        acc_v = accuracy_score(y_c2[m_v], (oof_preds[m_v] >= 0.50).astype(int)) * 100.0
        print(f"    - {v}: F1: {f1_v:5.2f}% | Acc: {acc_v:5.2f}% | AUC: {auc_v:5.2f}%", flush=True)
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()
