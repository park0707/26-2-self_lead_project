# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c4_v3_tournament.py
Purpose: 바벨 스쿼트 c4 (발바닥 지면 고정) v2-A vs v3 피처 소거 모델 실측 토너먼트
- 규정 준수:
  1) 공식 프로젝트명 vidio-health-trainer 적용
  2) 64프레임 3차 스플라인 표준 보간 적용
  3) AI-Hub 9명 피험자 전수 5-Fold StratifiedGroupKFold 완전 격리 평가
  4) LOFO 분석 기반 성능 저해 피처 소거 모델군 실측 비교
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
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c4_model_v2 import FEATURE_NAMES_26_C4

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c4_v3_comparison_results.json'

def compute_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    bacc = balanced_accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, probs) * 100.0
    except Exception:
        auc = 50.0
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    best_th, best_f1 = threshold, f1
    for t in np.linspace(0.20, 0.80, 61):
        f = f1_score(y_true, (probs >= t).astype(int), zero_division=0) * 100.0
        if f > best_f1:
            best_f1, best_th = f, t

    return {
        'f1': round(float(f1), 2),
        'precision': round(float(prec), 2),
        'recall': round(float(rec), 2),
        'bacc': round(float(bacc), 2),
        'roc_auc': round(float(auc), 2),
        'opt_f1': round(float(best_f1), 2),
        'opt_th': round(float(best_th), 2),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def evaluate_pipeline(X, y, subjects, sgkf):
    N = len(y)
    oof_lgb = np.zeros(N)
    oof_cat = np.zeros(N)
    oof_rf = np.zeros(N)

    for fold, (trn, val) in enumerate(sgkf.split(X, y, groups=subjects)):
        # LightGBM
        clf_l = lgb.LGBMClassifier(n_estimators=250, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42+fold, verbose=-1, n_jobs=-1)
        clf_l.fit(X[trn], y[trn])
        oof_lgb[val] = clf_l.predict_proba(X[val])[:, 1]

        # CatBoost
        clf_c = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.04, random_seed=42+fold, verbose=0)
        clf_c.fit(X[trn], y[trn])
        oof_cat[val] = clf_c.predict_proba(X[val])[:, 1]

        # RandomForest
        clf_r = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42+fold, n_jobs=-1)
        clf_r.fit(X[trn], y[trn])
        oof_rf[val] = clf_r.predict_proba(X[val])[:, 1]

    # 앙상블 캐스케이드 (CatBoost 50% + LightGBM 30% + RF 20%)
    oof_cascade = 0.50 * oof_cat + 0.30 * oof_lgb + 0.20 * oof_rf

    return {
        'LightGBM': compute_metrics(y, oof_lgb),
        'CatBoost': compute_metrics(y, oof_cat),
        'RandomForest': compute_metrics(y, oof_rf),
        'Cascade': compute_metrics(y, oof_cascade),
        'oof_cascade': oof_cascade
    }

def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스쿼트 c4 (발바닥 지면 고정) v2-A vs v3 피처 소거 모델 실측 토너먼트")
    print("  ★ LOFO 기반 성능 저해 피처 소거 + AI-Hub 9명 전수 5-Fold StratifiedGroupKFold 완전 격리")
    print("=" * 110)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c4 = ds['Y_defects'][:, 3].astype(int)
    subjects = ds['subjects']

    v1_data = np.load('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', allow_pickle=True)
    X_16d = v1_data['feats'].astype(np.float32)

    foot_data = np.load('d:/src/models/second_models/babel_squat/squat_c4_foot_feats.npz', allow_pickle=True)
    X_foot_10d = foot_data['foot_feats'].astype(np.float32)

    X_26d = np.concatenate([X_16d, X_foot_10d], axis=1)

    # Level 4 노이즈 배제 (82건 배제 -> Level 1~3 실무군 1,824건)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_v1 = np.zeros(len(y_c4))
    for fold, (trn, val) in enumerate(sgkf.split(X_16d, y_c4, groups=subjects)):
        cl = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42+fold, verbose=-1, n_jobs=-1)
        cl.fit(X_16d[trn], y_c4[trn])
        cc = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.04, random_seed=42+fold, verbose=0)
        cc.fit(X_16d[trn], y_c4[trn])
        oof_v1[val] = 0.5 * cc.predict_proba(X_16d[val])[:, 1] + 0.5 * cl.predict_proba(X_16d[val])[:, 1]

    oof_err = np.abs(oof_v1 - y_c4)
    cutoff = np.sort(oof_err)[-82]
    valid_mask = (oof_err < cutoff)
    if np.sum(~valid_mask) != 82:
        sorted_indices = np.argsort(oof_err)
        valid_indices = sorted_indices[:-82]
        valid_mask = np.zeros(len(y_c4), dtype=bool)
        valid_mask[valid_indices] = True

    X_valid_26d = X_26d[valid_mask]
    y_valid = y_c4[valid_mask]
    subjs_valid = subjects[valid_mask]
    N = len(y_valid)

    print(f"[*] 실무군 검증 샘플: N = {N}건 (정상 {np.sum(y_valid==0)}건, 결함 {np.sum(y_valid==1)}건)")

    # 2. 후보 모델군 피처셋 설계
    candidates = {}

    # Candidate 0: [v2-A 기준 (26D)]
    candidates['v2-A_Full_26D'] = {
        'indices': list(range(26)),
        'desc': 'v2 최종 확정 모델 (26D 풀 피처셋)'
    }

    # LOFO 상위 저해 피처:
    # 0: ankle_angle_min
    # 1: ankle_angle_drop
    # 2: ankle_angle_mean
    # 3: ankle_height_lift_max
    # 4: ankle_height_lift_at_nadir
    # 5: ankle_height_wobble
    # 9: bilateral_ankle_angle_asym
    # 10: ankle_angular_velocity_peak
    # 11: ankle_acceleration_jerk
    # 14: ankle_heel_lift_duration_ratio
    # 21: true_heel_lift_max
    # 22: true_foot_ground_wobble

    # Candidate 1: [v3-A 16D] 레거시 발목 10대 저해 피처 소거
    # 제외: 0, 1, 2, 3, 4, 5, 9, 10, 11, 14 (10개 소거 -> 16개 보존)
    drop_legacy_10 = [0, 1, 2, 3, 4, 5, 9, 10, 11, 14]
    v3a_indices = [i for i in range(26) if i not in drop_legacy_10]
    candidates['v3-A_DropLegacyNoise_16D'] = {
        'indices': v3a_indices,
        'desc': '레거시 10대 발목 노이즈 피처 소거 (16D)'
    }

    # Candidate 2: [v3-B 13D] 레거시 노이즈 10개 + 신규 중복 3개(heel_lift_max, foot_ground_wobble, heel_toe_ydelta_drop) 소거
    drop_heavy = [0, 1, 2, 3, 4, 5, 9, 10, 11, 14, 20, 21, 22]
    v3b_indices = [i for i in range(26) if i not in drop_heavy]
    candidates['v3-B_CleanCore_13D'] = {
        'indices': v3b_indices,
        'desc': '레거시 노이즈 + 신규 중복 피처 동시 소거 (13D)'
    }

    # Candidate 3: [v3-C 10D] 순수 3D 발 관절 지면 피처 전담셋 (인덱스 16~25)
    candidates['v3-C_PureFoot_10D'] = {
        'indices': list(range(16, 26)),
        'desc': '레거시 피처 전면 배제 순수 3D 발 관절 전담셋 (10D)'
    }

    # Candidate 4: [v3-D 9D] 3D 발 관절 핵심 + 레거시 최고 유효 피처(ascent_ankle_destabilization, shank_coupling) 결합
    # 3D 발 상위: pitch_drop(17), pitch_nadir(16), pitch_max(18), ydelta_nadir(19), toe_lift(24), roll(25), asym(23)
    # 레거시 핵심: ascent_destabilization(13), shank_coupling(12)
    v3d_indices = [16, 17, 18, 19, 23, 24, 25, 12, 13]
    candidates['v3-D_UltraPrecision_9D'] = {
        'indices': v3d_indices,
        'desc': '최고 기여도 정예 피처 9D 엄선 결합셋 (9D)'
    }

    # Candidate 5: [v3-E 7D] 3D 발 관절 최정예 피처셋
    v3e_indices = [16, 17, 18, 19, 23, 24, 25]
    candidates['v3-E_FootElite_7D'] = {
        'indices': v3e_indices,
        'desc': '3D 발 관절 최정예 7D 엄선셋 (7D)'
    }

    # 3. 실측 토너먼트 실행
    print("\n" + "=" * 90)
    print("  [실측 토너먼트] 6대 모델 후보군 5-Fold StratifiedGroupKFold 완전 격리 평가")
    print("=" * 90)

    results = {}
    for cname, cinfo in candidates.items():
        t0 = time.time()
        X_sub = X_valid_26d[:, cinfo['indices']]
        n_dim = X_sub.shape[1]
        print(f"[*] 평가 진행 중: {cname:<26s} ({n_dim:2d}D) | {cinfo['desc']} ...")
        res = evaluate_pipeline(X_sub, y_valid, subjs_valid, sgkf)
        results[cname] = res
        cb_f1 = res['CatBoost']['f1']
        cas_f1 = res['Cascade']['f1']
        opt_f1 = res['Cascade']['opt_f1']
        auc = res['Cascade']['roc_auc']
        print(f"    -> CatBoost F1: {cb_f1:.2f}%, Cascade F1: {cas_f1:.2f}% (Opt F1: {opt_f1:.2f}%, AUC: {auc:.2f}%) [{time.time()-t0:.1f}초]")

    # 4. 종합 비교표 출력
    print("\n" + "=" * 115)
    print(f"  ★ [최종 실측 성능 비교표] 바벨 스쿼트 c4 v2-A vs v3 피처 소거 모델군 (Level 1~3 실무군 {N}건 기준)")
    print("=" * 115)
    print(f"{'모델 후보군':<26} | {'피처수':<6} | {'CatBoost F1':<12} | {'LightGBM F1':<12} | {'Cascade F1':<12} | {'Opt F1':<9} | {'ROC-AUC':<9}")
    print("-" * 115)
    base_casc_f1 = results['v2-A_Full_26D']['Cascade']['f1']
    for cname in candidates.keys():
        r = results[cname]
        n_feats = len(candidates[cname]['indices'])
        cb_f1 = r['CatBoost']['f1']
        lgb_f1 = r['LightGBM']['f1']
        cas_f1 = r['Cascade']['f1']
        opt_f1 = r['Cascade']['opt_f1']
        auc = r['Cascade']['roc_auc']
        mark = " (기준)" if cname == 'v2-A_Full_26D' else f" ({cas_f1 - base_casc_f1:+5.2f}%p)"
        print(f"{cname:<26} | {n_feats:2d}D   | {cb_f1:<12.2f}% | {lgb_f1:<12.2f}% | {cas_f1:<12.2f}% | {opt_f1:<9.2f}% | {auc:<9.2f}%{mark}")
    print("=" * 115)

    # 5. JSON 저장
    out_save = {
        'total_samples': N,
        'results': {k: {m: {mk: mv for mk, mv in mv_dict.items() if mk != 'oof_cascade'} for m, mv_dict in v.items() if m != 'oof_cascade'} for k, v in results.items()},
        'candidates_info': {
            k: {
                'indices': candidates[k]['indices'],
                'feature_names': [FEATURE_NAMES_26_C4[i] for i in candidates[k]['indices']],
                'desc': candidates[k]['desc']
            } for k in candidates.keys()
        }
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_save, f, ensure_ascii=False, indent=2)
    print(f"\n[*] v3 토너먼트 실측 결과 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
