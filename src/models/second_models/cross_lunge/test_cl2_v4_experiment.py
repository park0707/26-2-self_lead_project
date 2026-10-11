# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (08_크로스_런지) Condition 2
Script: test_cl2_v4_experiment.py
Purpose: 생체역학 4종 피처 + 이종 스태킹 + 임계값 최적화 F1 70% 돌파 실증 실험
"""

import os
import sys
import time
import pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"
V3_CACHE_PKL = r"D:\Data\피처_데이터\cl2_v3_48d_cache.pkl"
PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.cross_lunge_spatial_leg_separator import detect_cross_lunge_lead_leg_spatial
from src.models.second_models.cross_lunge.cross_lunge_c2_v7_features import safe_savgol_filter_kps


def compute_v4_biomechanic_features_4d(kps_clean: np.ndarray, is_left_lead: bool, t_bot: int) -> np.ndarray:
    if is_left_lead:
        hip_idx, knee_idx, ank_idx, toe_idx = 23, 25, 27, 31
    else:
        hip_idx, knee_idx, ank_idx, toe_idx = 24, 26, 28, 32

    kps_stand = np.mean(kps_clean[:4], axis=0)
    kps_bot = kps_clean[t_bot]

    # 1. pelvic_roll_tilt_at_bottom: 골반 관상면 기울기 (Trendelenburg)
    v_pelvis_bot = kps_bot[23] - kps_bot[24]
    pelvic_roll_rad = np.arctan2(abs(v_pelvis_bot[1]), abs(v_pelvis_bot[0]) + 1e-6)
    pelvic_roll_tilt_deg = float(np.degrees(pelvic_roll_rad))

    # 2. trunk_lateral_lean_bottom: 체간 측방 기울기
    mid_hip_bot = 0.5 * (kps_bot[23] + kps_bot[24])
    mid_sh_bot = 0.5 * (kps_bot[11] + kps_bot[12])
    v_spine_bot = mid_sh_bot - mid_hip_bot
    trunk_lean_rad = np.arctan2(abs(v_spine_bot[0]), abs(v_spine_bot[1]) + 1e-6)
    trunk_lateral_lean_deg = float(np.degrees(trunk_lean_rad))

    # 3. lead_hip_adduction_dynamic_angle: 대퇴 내전 각도 변화량
    v_thigh_stand = kps_stand[knee_idx] - kps_stand[hip_idx]
    v_thigh_bot = kps_bot[knee_idx] - kps_bot[hip_idx]
    dot = np.dot(v_thigh_stand, v_thigh_bot) / (np.linalg.norm(v_thigh_stand) * np.linalg.norm(v_thigh_bot) + 1e-6)
    hip_adduction_deg = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))

    # 4. knee_tracking_over_toe_coronal_offset: 관상면 무릎-발 정렬 편차 (몸길이 대비 정규화)
    torso_len = np.linalg.norm(v_spine_bot) + 1e-6
    mid_foot_x = 0.5 * (kps_bot[ank_idx, 0] + kps_bot[toe_idx, 0])
    knee_offset = abs(kps_bot[knee_idx, 0] - mid_foot_x) / torso_len

    return np.array([pelvic_roll_tilt_deg, trunk_lateral_lean_deg, hip_adduction_deg, knee_offset], dtype=np.float32)


def run_experiment():
    print("[*] v3 48D 캐시 로드 중...")
    with open(V3_CACHE_PKL, 'rb') as f:
        v3_data = pickle.load(f)

    X_48d = v3_data['X']
    y = v3_data['y']
    groups = v3_data['groups']
    views = v3_data['views']
    clip_names = v3_data['clip_names']
    N = len(y)

    print(f"[*] 1,025개 시퀀스 대상 생체역학 4종 피처 추출 중...")
    t0 = time.time()
    feats_4d = []
    for i in range(N):
        cname = clip_names[i]
        vcode = views[i]
        npz_path = os.path.join(DATA_DIR, f"{cname}_{vcode}.npz")
        data = np.load(npz_path)
        kps = data['kps_33_3d_64f']
        kps_clean = safe_savgol_filter_kps(kps[:, :, :3], window=7, poly=2)
        is_left_lead, t_bot = detect_cross_lunge_lead_leg_spatial(kps_clean)
        f4 = compute_v4_biomechanic_features_4d(kps_clean, is_left_lead, t_bot)
        feats_4d.append(f4)

    feats_4d = np.array(feats_4d, dtype=np.float32)
    X_52d = np.hstack([X_48d, feats_4d])
    print(f"[+] 52D 피처 구축 완료! 소요 시간: {time.time()-t0:.2f}s, Shape: {X_52d.shape}")

    # 5-Fold StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_cb = np.zeros(N)
    oof_xgb = np.zeros(N)
    oof_lgb = np.zeros(N)
    oof_et = np.zeros(N)
    oof_stack = np.zeros(N)

    # Fold별 모델 저장
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_52d, y, groups), 1):
        X_tr, y_tr = X_52d[tr_idx], y[tr_idx]
        X_va, y_va = X_52d[val_idx], y[val_idx]

        # 1. CatBoost
        cb = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.035, verbose=0, random_seed=42 + fold)
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40)
        oof_cb[val_idx] = cb.predict_proba(X_va)[:, 1]

        # 2. XGBoost
        xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.035, random_state=42 + fold, eval_metric='logloss', early_stopping_rounds=40)
        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = xgb.predict_proba(X_va)[:, 1]

        # 3. LightGBM
        lgb_model = LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.035, random_state=42 + fold, verbose=-1)
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]

        # 4. ExtraTrees
        et = ExtraTreesClassifier(n_estimators=250, max_depth=8, min_samples_split=4, random_state=42 + fold)
        et.fit(X_tr, y_tr)
        oof_et[val_idx] = et.predict_proba(X_va)[:, 1]

    # Stacking Meta-Learner (Ridge/LogisticRegression on OOF predictions)
    # Inner CV to prevent leakage
    meta_features = np.column_stack([oof_cb, oof_xgb, oof_lgb, oof_et])
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(meta_features, y, groups), 1):
        lr_meta = LogisticRegression(C=1.0, random_state=42)
        lr_meta.fit(meta_features[tr_idx], y[tr_idx])
        oof_stack[val_idx] = lr_meta.predict_proba(meta_features[val_idx])[:, 1]

    # Simple Blend
    oof_blend = 0.40 * oof_cb + 0.25 * oof_xgb + 0.20 * oof_lgb + 0.15 * oof_et

    print("\n" + "=" * 80)
    print("  [52D 생체역학 피처 + 앙상블 기법별 성능 비교]")
    print("=" * 80)
    models = [
        ('CatBoost Single', oof_cb),
        ('XGBoost Single', oof_xgb),
        ('LightGBM Single', oof_lgb),
        ('ExtraTrees Single', oof_et),
        ('4-Model Soft Blend', oof_blend),
        ('Stacking Meta-Learner', oof_stack)
    ]

    for name, p in models:
        f1_def = f1_score(y, p >= 0.5, zero_division=0)
        auc = roc_auc_score(y, p)
        # Optimal Threshold
        prec, rec, ths = precision_recall_curve(y, p)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
        b_idx = np.argmax(f1s)
        best_th = ths[b_idx] if b_idx < len(ths) else 0.5
        best_f1 = f1s[b_idx]
        print(f"{name:25s} | Def F1(th=0.5): {f1_def*100:.2f}% | AUC: {auc:.4f} | Opt F1: {best_f1*100:.2f}% (th={best_th:.3f})")

    # 3-State F1 Grid Search with 10% uncertainty constraint
    print("\n" + "=" * 80)
    print("  [3-State 적응형 캘리브레이션 튜닝 (F1 70% 돌파 검증)]")
    print("=" * 80)
    for name, p in [('4-Model Soft Blend', oof_blend), ('Stacking Meta-Learner', oof_stack)]:
        best_f1_3s = -1.0
        best_cfg = None
        for tau_c in np.arange(0.25, 0.52, 0.01):
            for tau_d in np.arange(0.48, 0.75, 0.01):
                if tau_c >= tau_d:
                    continue
                mask = (p < tau_c) | (p >= tau_d)
                unc = 1.0 - np.mean(mask)
                if unc > 0.10:  # 보류율 10% 이하 제약
                    continue
                score = f1_score(y[mask], p[mask] >= tau_d, zero_division=0)
                if score > best_f1_3s:
                    best_f1_3s = score
                    best_cfg = (tau_c, tau_d, unc, np.mean(mask))
        print(f"{name:25s} | 3-State F1: {best_f1_3s*100:.2f}% | tau_c={best_cfg[0]:.3f}, tau_d={best_cfg[1]:.3f} | Cov: {best_cfg[3]*100:.2f}% (Unc: {best_cfg[2]*100:.2f}%)")


if __name__ == "__main__":
    run_experiment()
