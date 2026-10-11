# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Defect: c2. 고개 정면 (Head/Gaze Alignment)
Module: run_c2_v4_optimization_tournament.py
Purpose: c2 v3 13D vs v4 방해 피처 소거 최적화 후보군 공식 5-Fold 실측 토너먼트 (N=1,906)
"""

import os
import sys
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
from src.models.second_models.babel_squat.c2_model_v3 import FEATURE_NAMES_13, V2_TO_V3_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(CURRENT_DIR, 'c2_v4_tournament_results.json')


def evaluate_feature_subset(X_subset, y_arr, subjs, sgkf):
    N = len(y_arr)
    oof_prob = np.zeros(N, dtype=np.float32)

    for trn_idx, val_idx in sgkf.split(X_subset, y_arr, groups=subjs):
        X_tr, y_tr = X_subset[trn_idx], y_arr[trn_idx]
        X_va, y_va = X_subset[val_idx], y_arr[val_idx]

        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X_tr, y_tr)

        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        p_cb = cb.predict_proba(X_va)[:, 1]
        p_lgb = clf.predict_proba(X_va)[:, 1]
        oof_prob[val_idx] = 0.60 * p_cb + 0.40 * p_lgb

    pred = (oof_prob >= 0.50).astype(int)
    f1 = f1_score(y_arr, pred, zero_division=0) * 100.0
    prec = precision_score(y_arr, pred, zero_division=0) * 100.0
    rec = recall_score(y_arr, pred, zero_division=0) * 100.0
    auc = roc_auc_score(y_arr, oof_prob) * 100.0
    missed = int(np.sum((y_arr == 1) & (pred == 0)))

    return {
        'f1': round(float(f1), 2),
        'precision': round(float(prec), 2),
        'recall': round(float(rec), 2),
        'roc_auc': round(float(auc), 2),
        'missed': missed
    }, oof_prob


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] c2 v3 13D vs v4 방해 피처 소거 최적화 후보군 공식 5-Fold 실측 토너먼트 (N=1,906)")
    print("=" * 115)

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
    names_13 = list(FEATURE_NAMES_13)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 방해 피처 후보:
    # #07 c2_gaze_relative_to_trunk_mean (+0.78%p)
    # #05 c2_head_yaw_asymmetry_max (+0.29%p)
    # #00 c2_gaze_pitch_min (+0.18%p)
    # #06 c2_head_forward_protrusion_max (+0.15%p)
    # #08 c2_cervical_velocity_peak_at_nadir (+0.07%p)
    # #09 c2_gaze_coupling_lag (+0.07%p)

    candidates = [
        ("v3.0 기준선 (13D 전체)", []),
        ("v4-A (12D: trunk_mean 소거)", ["c2_gaze_relative_to_trunk_mean"]),
        ("v4-B (11D: trunk_mean + yaw_max 소거)", ["c2_gaze_relative_to_trunk_mean", "c2_head_yaw_asymmetry_max"]),
        ("v4-C (10D: trunk_mean + yaw_max + pitch_min 소거)", ["c2_gaze_relative_to_trunk_mean", "c2_head_yaw_asymmetry_max", "c2_gaze_pitch_min"]),
        ("v4-D (9D: 4대 방해 피처 소거: trunk_mean + yaw_max + pitch_min + fwd_max)", [
            "c2_gaze_relative_to_trunk_mean", "c2_head_yaw_asymmetry_max", "c2_gaze_pitch_min", "c2_head_forward_protrusion_max"
        ]),
        ("v4-E (8D: 5대 방해 피처 소거: 4대 + vel_peak)", [
            "c2_gaze_relative_to_trunk_mean", "c2_head_yaw_asymmetry_max", "c2_gaze_pitch_min", "c2_head_forward_protrusion_max", "c2_cervical_velocity_peak_at_nadir"
        ]),
        ("v4-F (7D 정예 다이어트: 6대 전체 방해 피처 소거)", [
            "c2_gaze_relative_to_trunk_mean", "c2_head_yaw_asymmetry_max", "c2_gaze_pitch_min", "c2_head_forward_protrusion_max", "c2_cervical_velocity_peak_at_nadir", "c2_gaze_coupling_lag"
        ])
    ]

    results = []
    best_f1 = -1.0
    best_name = ""
    best_oof = None
    best_keep_names = None

    print(f"[*] 총 {len(candidates)}개 후보 세트 실측 토너먼트 진행...")

    for label, drop_cols in candidates:
        keep_indices = [i for i, n in enumerate(names_13) if n not in drop_cols]
        keep_names = [names_13[i] for i in keep_indices]
        X_sub = X_13d[:, keep_indices]

        t0 = time.time()
        m_res, oof_prob = evaluate_feature_subset(X_sub, y_c2, subjects, sgkf)
        elapsed = time.time() - t0

        results.append({
            'label': label,
            'dim': len(keep_indices),
            'dropped': drop_cols,
            'keep_features': keep_names,
            'metrics': m_res,
            'elapsed_sec': round(elapsed, 1),
            'oof_prob': oof_prob
        })

        if m_res['f1'] > best_f1:
            best_f1 = m_res['f1']
            best_name = label
            best_oof = oof_prob
            best_keep_names = keep_names

        print(f"  >>> {label:<45} | F1: {m_res['f1']:5.2f}% | Prec: {m_res['precision']:5.2f}% | Rec: {m_res['recall']:5.2f}% | AUC: {m_res['roc_auc']:5.2f}% | 미인식: {m_res['missed']}건 ({elapsed:.1f}초)")

    base_f1 = results[0]['metrics']['f1']

    print("\n" + "=" * 120)
    print("  [c2 고개 정면 모델: v3 vs v4 최적화 토너먼트 공식 실측 성적표 (N=1,906)]")
    print("=" * 120)
    print(f"{'후보 모델 및 피처 구성':<46} | {'차원':^6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<14} | {'ROC-AUC':<10} | {'미인식':<8} | {'v3 대비 (Δ)'}")
    print("-" * 120)

    for r in results:
        m = r['metrics']
        df = m['f1'] - base_f1
        sign = "+" if df > 0 else ""
        df_str = "기준선" if r == results[0] else f"{sign}{df:+.2f}%p"
        is_best = (r['label'] == best_name)
        star = "🏆 " if is_best else "   "
        print(f"{star + r['label']:<46} | {r['dim']:^6} | {m['precision']:6.2f}%    | {m['recall']:6.2f}%    | {m['f1']:6.2f}%        | {m['roc_auc']:6.2f}%    | {m['missed']}건    | {df_str:>10}")

    print("=" * 120)
    print(f"\n[*] 최종 챔피언 선정: {best_name} (F1 = {best_f1:.2f}%)")
    print(f"    - 잔존 정예 피처 수: {len(best_keep_names)}개")
    print(f"    - 확정 피처: {best_keep_names}")

    # JSON 저장 (oof_prob 제외)
    save_results = []
    for r in results:
        c = dict(r)
        del c['oof_prob']
        save_results.append(c)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({
            'champion': best_name,
            'best_f1': best_f1,
            'best_features': best_keep_names,
            'candidates': save_results
        }, f, indent=2, ensure_ascii=False)

    print(f"[*] 토너먼트 결과 JSON 저장 완료: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
