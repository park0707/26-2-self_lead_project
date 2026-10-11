# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: test_hybrid_v4.py
Purpose: TCN 14D 원본 유지 + GBDT에 상위 3D delta 피처 결합 (27D) 하이브리드 검증
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

CACHE_V3_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v3.npz"
CACHE_V4_PATH = r"D:\Data\피처_데이터\lunge_c4_dataset_cache_v4.npz"

def test_hybrid():
    d3 = np.load(CACHE_V3_PATH, allow_pickle=True)
    d4 = np.load(CACHE_V4_PATH, allow_pickle=True)

    X_24d = d3['X_24d']
    X_30d = d4['X_30d']
    y = d3['y']
    subjects = d3['subjects']

    # 30D 중에서 상위 delta 피처 3개 (24: delta_trunk_roll_dyn, 25: delta_shoulder_roll_dyn, 26: delta_cervical_yaw_dyn)
    delta_3d = X_30d[:, [24, 25, 26]]
    X_27d = np.hstack([X_24d, delta_3d])

    print(f"[*] 테스트 피처 규격: {X_27d.shape}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))

    for tr, va in sgkf.split(X_27d, y, groups=subjects):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_27d[tr])
        X_va = scaler.transform(X_27d[va])
        y_tr, y_va = y[tr], y[va]

        # LGB
        m_lgb = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=15, subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=0.3, class_weight='balanced', random_state=42, verbose=-1)
        m_lgb.fit(X_tr, y_tr)
        oof_lgb[va] = m_lgb.predict_proba(X_va)[:, 1]

        # XGB
        pos_w = float(np.sum(y_tr == 0)) / float(max(1, np.sum(y_tr == 1)))
        m_xgb = xgb.XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, subsample=0.85, colsample_bytree=0.80, gamma=0.15, scale_pos_weight=pos_w, eval_metric='logloss', random_state=42)
        m_xgb.fit(X_tr, y_tr)
        oof_xgb[va] = m_xgb.predict_proba(X_va)[:, 1]

        # CAT
        m_cat = CatBoostClassifier(iterations=120, learning_rate=0.04, depth=4, l2_leaf_reg=3.0, auto_class_weights='Balanced', verbose=0, random_seed=42)
        m_cat.fit(X_tr, y_tr)
        oof_cat[va] = m_cat.predict_proba(X_va)[:, 1]

    f1_l = f1_score(y, (oof_lgb >= 0.5).astype(int), pos_label=1)
    f1_x = f1_score(y, (oof_xgb >= 0.5).astype(int), pos_label=1)
    f1_c = f1_score(y, (oof_cat >= 0.5).astype(int), pos_label=1)

    p_comb = 0.35 * oof_lgb + 0.35 * oof_xgb + 0.30 * oof_cat
    f1_gbdt = f1_score(y, (p_comb >= 0.5).astype(int), pos_label=1)

    print(f"[+] GBDT 2-State F1: LGB={f1_l*100:.2f}%, XGB={f1_x*100:.2f}%, CAT={f1_c*100:.2f}% | 앙상블={f1_gbdt*100:.2f}%")

if __name__ == "__main__":
    test_hybrid()
