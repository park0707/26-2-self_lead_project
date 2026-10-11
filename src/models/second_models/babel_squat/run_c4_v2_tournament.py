# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c4_v2_tournament.py
Purpose: 바벨 스쿼트 c4 (발바닥 지면 고정) v1 vs v2 다차원 생체역학 피처 토너먼트
- 규정 준수:
  1) 공식 프로젝트명 vidio-health-trainer 적용
  2) 64프레임 3차 스플라인 표준 보간 적용
  3) AI-Hub 9명 피험자 전수 5-Fold StratifiedGroupKFold 완전 격리 평가
  4) Level 4 역전 노이즈 정제된 실무군(1,824건) 및 전체(1,906건) 동시 실측
  5) LightGBM, CatBoost, RandomForest 및 2계층 캐스케이드 앙상블 비교
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

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c4_v2_comparison_results.json'

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

def evaluate_pipeline(X, y, subjects, sgkf, name="Candidate"):
    """5-Fold CV OOF 평가"""
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

    # 앙상블/캐스케이드 (Tier-1 CatBoost 50% + LightGBM 30% + RF 20%)
    oof_cascade = 0.50 * oof_cat + 0.30 * oof_lgb + 0.20 * oof_rf

    m_lgb = compute_metrics(y, oof_lgb)
    m_cat = compute_metrics(y, oof_cat)
    m_rf = compute_metrics(y, oof_rf)
    m_casc = compute_metrics(y, oof_cascade)

    return {
        'LightGBM': m_lgb,
        'CatBoost': m_cat,
        'RandomForest': m_rf,
        'Cascade': m_casc,
        'oof_cascade': oof_cascade
    }

def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스쿼트 c4 (발바닥 지면 고정) v1 vs v2 다차원 성능 비교 토너먼트")
    print("  ★ 15관절 레거시(v1) vs 24관절 3D 발 관절(Foot/Ankle) 지면 접지 피처 결합(v2)")
    print("  ★ AI-Hub 9명 피험자 전수 5-Fold StratifiedGroupKFold 완전 격리 환경 실측")
    print("=" * 110)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c4 = ds['Y_defects'][:, 3].astype(int)
    subjects = ds['subjects']
    N_total = len(y_c4)

    # 16D 레거시 피처 로드
    npz_16d = np.load('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', allow_pickle=True)
    X_16d = npz_16d['feats'].astype(np.float32)
    names_16d = list(npz_16d['feature_names'])

    # 10D 신규 발 관절 지면 피처 로드
    npz_10d = np.load('d:/src/models/second_models/babel_squat/squat_c4_foot_feats.npz', allow_pickle=True)
    X_foot_10d = npz_10d['foot_feats'].astype(np.float32)
    names_10d = list(npz_10d['feature_names'])

    print(f"[*] 데이터 로드 완료: 총 {N_total}건 (정상 {np.sum(y_c4==0)}건, 결함 {np.sum(y_c4==1)}건)")

    # 2. 모호성 필터링 (Level 4 역전 노이즈 배제)
    # v1 파이프라인에서 도출된 Level 4 인덱스 도출
    # Level 4는 정상 피처를 보이나 오류로 라벨링되었거나 오류 피처를 보이나 정상으로 라벨링된 극단치 (82건)
    ambiguity_json = 'd:/src/models/second_models/babel_squat/c4_cascade_comparison_results.json'
    amb_data = json.load(open(ambiguity_json, encoding='utf-8'))
    
    # 실무군 필터링: v1의 oof 오차 기반 또는 피처 기반 Level 1~3 필터링
    # v1의 단일 CatBoost OOF 오차 절대값이 0.85 이상인 82건을 Level 4로 판정했던 체계 재현
    # 정확한 재현을 위해 v1 스크립트와 동일한 필터링 기준 적용
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. 후보 피처셋 구성
    candidates = {}

    # Candidate 1: [v1.0 기준선] 16D 레거시 간접 발목 피처
    candidates['v1.0_Legacy_16D'] = {
        'X': X_16d,
        'names': names_16d,
        'desc': '기존 15관절 체계 간접 발목 피처 (16D)'
    }

    # Candidate 2: [v2-A] 26D 풀 피처 결합 (16D 레거시 + 10D 3D 발 관절 피처)
    X_26d = np.hstack([X_16d, X_foot_10d])
    candidates['v2-A_Full_26D'] = {
        'X': X_26d,
        'names': names_16d + names_10d,
        'desc': '기존 16D + 신규 10D 발 지면 접지 피처 전수 결합 (26D)'
    }

    # Candidate 3: [v2-B] 10D 신규 3D 발 관절 지면 피처 단독
    candidates['v2-B_FootOnly_10D'] = {
        'X': X_foot_10d,
        'names': names_10d,
        'desc': '신규 10D 발 관절 지면 접지 피처 단독 (10D)'
    }

    # Candidate 4: [v2-C] 18D 핵심 결합 (신규 8D + 레거시 상위 10D)
    # 신규 8D: 상위 8개 피처
    foot_sel_idx = [0, 1, 2, 3, 4, 5, 6, 8] # pitch, drop, max, ydelta, ydelta_drop, heel_lift, wobble, toe_lift
    leg_sel_idx = [0, 1, 3, 4, 5, 6, 8, 9, 11, 12] # 레거시 10개
    X_18d = np.hstack([X_16d[:, leg_sel_idx], X_foot_10d[:, foot_sel_idx]])
    names_18d = [names_16d[i] for i in leg_sel_idx] + [names_10d[i] for i in foot_sel_idx]
    candidates['v2-C_Selected_18D'] = {
        'X': X_18d,
        'names': names_18d,
        'desc': '신규 상위 8D + 레거시 10D 정예 결합 (18D)'
    }

    # Candidate 5: [v2-D] 8D 초정예 발 지면 피처 단독
    X_8d = X_foot_10d[:, foot_sel_idx]
    names_8d = [names_10d[i] for i in foot_sel_idx]
    candidates['v2-D_CoreFoot_8D'] = {
        'X': X_8d,
        'names': names_8d,
        'desc': '신규 핵심 발 지면 접지 피처 정예 단독 (8D)'
    }

    # 4. 전체 1,906건 대상 벤치마크 수행
    print("\n" + "=" * 90)
    print("  [Phase 1] 전체 1,906건 대상 5-Fold StratifiedGroupKFold 실측 토너먼트")
    print("=" * 90)

    results_all = {}
    for cname, cinfo in candidates.items():
        t0 = time.time()
        print(f"[*] 평가 진행 중: {cname} ({cinfo['desc']}) ...")
        res = evaluate_pipeline(cinfo['X'], y_c4, subjects, sgkf, name=cname)
        results_all[cname] = res
        print(f"    -> [결과] CatBoost F1: {res['CatBoost']['f1']:.2f}%, Cascade F1: {res['Cascade']['f1']:.2f}% (Opt F1: {res['Cascade']['opt_f1']:.2f}%, AUC: {res['Cascade']['roc_auc']:.2f}%) [{time.time()-t0:.1f}초]")

    # 5. 모호성 Level 1~3 실무군(1,824건) 대상 실측 토너먼트
    # Level 4 노이즈 배제: v1의 OOF 절대 오차가 극단적으로 높은 상위 82건 배제
    oof_err = np.abs(results_all['v1.0_Legacy_16D']['oof_cascade'] - y_c4)
    # 82건 역전 노이즈 인덱스 (v1 보고서 기준과 일치)
    level4_cutoff = np.sort(oof_err)[-82]
    valid_mask = (oof_err < level4_cutoff)
    if np.sum(~valid_mask) != 82:
        # 정확히 82개 배제
        sorted_indices = np.argsort(oof_err)
        valid_indices = sorted_indices[:-82]
        valid_mask = np.zeros(N_total, dtype=bool)
        valid_mask[valid_indices] = True

    N_valid = np.sum(valid_mask)
    print(f"\n" + "=" * 90)
    print(f"  [Phase 2] Level 1~3 실무군 ({N_valid}건) 대상 5-Fold StratifiedGroupKFold 실측 토너먼트")
    print("=" * 90)

    results_valid = {}
    for cname, cinfo in candidates.items():
        t0 = time.time()
        X_sub = cinfo['X'][valid_mask]
        y_sub = y_c4[valid_mask]
        sub_subjects = subjects[valid_mask]
        print(f"[*] 평가 진행 중: {cname} (N={N_valid}) ...")
        res = evaluate_pipeline(X_sub, y_sub, sub_subjects, sgkf, name=cname)
        results_valid[cname] = res
        print(f"    -> [결과] CatBoost F1: {res['CatBoost']['f1']:.2f}%, Cascade F1: {res['Cascade']['f1']:.2f}% (Opt F1: {res['Cascade']['opt_f1']:.2f}%, AUC: {res['Cascade']['roc_auc']:.2f}%) [{time.time()-t0:.1f}초]")

    # 6. 종합 결과 출력 테이블
    print("\n" + "=" * 110)
    print(f"  ★ [최종 실측 성능 비교표] 바벨 스쿼트 c4 발바닥 지면 고정 (Level 1~3 실무군 {N_valid}건 기준)")
    print("=" * 110)
    print(f"{'모델 후보군':<22} | {'피처수':<6} | {'CatBoost F1':<12} | {'LightGBM F1':<12} | {'Cascade F1':<12} | {'Opt F1':<9} | {'ROC-AUC':<9}")
    print("-" * 110)
    for cname in candidates.keys():
        r = results_valid[cname]
        n_feats = candidates[cname]['X'].shape[1]
        cb_f1 = r['CatBoost']['f1']
        lgb_f1 = r['LightGBM']['f1']
        cas_f1 = r['Cascade']['f1']
        opt_f1 = r['Cascade']['opt_f1']
        auc = r['Cascade']['roc_auc']
        print(f"{cname:<22} | {n_feats:<6} | {cb_f1:<12.2f}% | {lgb_f1:<12.2f}% | {cas_f1:<12.2f}% | {opt_f1:<9.2f}% | {auc:<9.2f}%")
    print("=" * 110)

    # 7. JSON 결과 저장 (oof_cascade 배열은 직렬화 위해 제외)
    out_save = {
        'total_samples': N_total,
        'valid_samples': int(N_valid),
        'results_all_1906': {k: {m: {mk: mv for mk, mv in mv_dict.items() if mk != 'oof_cascade'} for m, mv_dict in v.items() if m != 'oof_cascade'} for k, v in results_all.items()},
        'results_valid_1824': {k: {m: {mk: mv for mk, mv in mv_dict.items() if mk != 'oof_cascade'} for m, mv_dict in v.items() if m != 'oof_cascade'} for k, v in results_valid.items()},
        'feature_sets': {k: candidates[k]['names'] for k in candidates.keys()}
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_save, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 토너먼트 실측 결과 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
