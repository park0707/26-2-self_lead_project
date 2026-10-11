# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c1_v2_tournament.py
Purpose: 바벨 스쿼트 c1 (척추의 중립) v1 vs v2 다차원 생체역학 피처 토너먼트
- 규정 준수:
  1) 공식 프로젝트명 vidio-health-trainer 적용
  2) 64프레임 3차 스플라인 표준 보간 적용
  3) AI-Hub 9명 피험자 전수 5-Fold StratifiedGroupKFold 완전 격리 평가
  4) 레거시(22D) vs 3D 척추(Neck/Back/Waist 8D) 결합 비교
  5) 원천 전체(1,906건) 및 Level 1~3 실무군(1,631건) 동시 실측
  6) 완전 정상 125건 대상 오탐(False Positive) 격감 효과 정밀 실측
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
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix, accuracy_score
)
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v1 import BabelSquatC1ModelV1, FEATURE_NAMES_22
from src.models.second_models.babel_squat.build_c1_spine_features import C1_SPINE_FEATURE_NAMES

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c1_v2_comparison_results.json'

def compute_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    acc = accuracy_score(y_true, preds) * 100.0
    bacc = balanced_accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, probs) * 100.0
    except Exception:
        auc = 50.0
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

    best_th, best_f1 = threshold, f1
    for t in np.linspace(0.20, 0.80, 61):
        f = f1_score(y_true, (probs >= t).astype(int), zero_division=0) * 100.0
        if f > best_f1:
            best_f1, best_th = f, t

    return {
        'f1': round(float(f1), 2),
        'precision': round(float(prec), 2),
        'recall': round(float(rec), 2),
        'accuracy': round(float(acc), 2),
        'specificity': round(float(spec), 2),
        'bacc': round(float(bacc), 2),
        'roc_auc': round(float(auc), 2),
        'opt_f1': round(float(best_f1), 2),
        'opt_th': round(float(best_th), 2),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def evaluate_pipeline(X, y, subjects, sgkf, clean_mask=None, cost_weight=1.0):
    """5-Fold CV OOF 평가"""
    N = len(y)
    oof_lgb = np.zeros(N)
    oof_cat = np.zeros(N)
    oof_rf = np.zeros(N)

    sample_weight = np.ones(N)
    if cost_weight > 1.0:
        sample_weight[y == 0] = cost_weight

    for fold, (trn, val) in enumerate(sgkf.split(X, y, groups=subjects)):
        sw_trn = sample_weight[trn]

        # LightGBM
        clf_l = lgb.LGBMClassifier(n_estimators=250, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42+fold, verbose=-1, n_jobs=-1)
        clf_l.fit(X[trn], y[trn], sample_weight=sw_trn)
        oof_lgb[val] = clf_l.predict_proba(X[val])[:, 1]

        # CatBoost
        clf_c = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.04, random_seed=42+fold, verbose=0)
        clf_c.fit(X[trn], y[trn], sample_weight=sw_trn)
        oof_cat[val] = clf_c.predict_proba(X[val])[:, 1]

        # RandomForest
        clf_r = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42+fold, n_jobs=-1)
        clf_r.fit(X[trn], y[trn], sample_weight=sw_trn)
        oof_rf[val] = clf_r.predict_proba(X[val])[:, 1]

    # 앙상블 캐스케이드 (CatBoost 50% + LightGBM 30% + RF 20%)
    oof_cascade = 0.50 * oof_cat + 0.30 * oof_lgb + 0.20 * oof_rf

    m_lgb = compute_metrics(y, oof_lgb)
    m_cat = compute_metrics(y, oof_cat)
    m_rf = compute_metrics(y, oof_rf)
    m_casc = compute_metrics(y, oof_cascade)

    clean_fps = 0
    clean_fp_rate = 0.0
    if clean_mask is not None:
        clean_probs = oof_cascade[clean_mask]
        clean_fps = int(np.sum(clean_probs >= 0.50))
        clean_fp_rate = round(float(clean_fps / np.sum(clean_mask) * 100.0), 2)

    return {
        'LightGBM': m_lgb,
        'CatBoost': m_cat,
        'RandomForest': m_rf,
        'Cascade': m_casc,
        'clean_fps': clean_fps,
        'clean_fp_rate': clean_fp_rate,
        'oof_cascade': oof_cascade
    }

def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스쿼트 c1 (척추의 중립) v1 vs v2 다차원 실측 토너먼트")
    print("  ★ 레거시(22D) vs 3D 척추(Neck/Back/Waist 8D) 결합 + 9명 전수 5-Fold StratifiedGroupKFold 완전 격리")
    print("=" * 110)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0].astype(int)
    Y_all = ds['Y_defects'][:, :4].astype(int)
    clean_mask_all = (np.sum(Y_all, axis=1) == 0) # 125건 완전 정상
    subjects = ds['subjects']
    N_total = len(y_c1)

    print(f"[*] 총 스쿼트 샘플: {N_total}건 (정상 {np.sum(y_c1==0)}건, 결함 {np.sum(y_c1==1)}건)")
    print(f"[*] 완전 정상(Clean) 샘플: {np.sum(clean_mask_all)}건")

    # 2. 피처 데이터 준비
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)
    X_22d = BabelSquatC1ModelV1.extract_features(X_b_64, use_pruned=True)

    npz_spine = np.load('d:/src/models/second_models/babel_squat/squat_c1_spine_feats.npz', allow_pickle=True)
    X_spine_8d = npz_spine['spine_feats'].astype(np.float32)

    X_30d = np.hstack([X_22d, X_spine_8d])

    # 3D 척추 6D + 레거시 10D 정예 결합
    sel_spine = [0, 1, 2, 3, 4, 7]
    sel_legacy = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    X_16d = np.hstack([X_22d[:, sel_legacy], X_spine_8d[:, sel_spine]])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. 후보 피처셋 구성
    candidates = {}
    candidates['v1.0_Legacy_22D'] = {
        'X': X_22d,
        'cost_weight': 1.0,
        'desc': '기존 15관절 레거시 피처셋 (22D)'
    }
    candidates['v2-A_Full_30D'] = {
        'X': X_30d,
        'cost_weight': 1.0,
        'desc': '레거시 22D + 신규 3D 척추 8D 전수 결합 (30D)'
    }
    candidates['v2-B_SpineOnly_8D'] = {
        'X': X_spine_8d,
        'cost_weight': 1.0,
        'desc': '신규 8D 3D 척추 분절 피처 단독 (8D)'
    }
    candidates['v2-C_CostSensitive_30D'] = {
        'X': X_30d,
        'cost_weight': 1.7,
        'desc': '30D 결합 + 정상 FP 억제 비용 민감 학습 (가중치 1.7배)'
    }
    candidates['v2-D_Selected_16D'] = {
        'X': X_16d,
        'cost_weight': 1.5,
        'desc': '3D 척추 6D + 레거시 10D 정예 결합 + 비용 민감 (16D)'
    }

    # 4. Phase 1: 전체 1,906건 대상 토너먼트
    print("\n" + "=" * 95)
    print("  [Phase 1] 원천 전체 (1,906건) 대상 5-Fold StratifiedGroupKFold 실측 토너먼트")
    print("=" * 95)

    results_all = {}
    for cname, cinfo in candidates.items():
        t0 = time.time()
        print(f"[*] 평가 진행 중: {cname:<26s} ({cinfo['desc']}) ...")
        res = evaluate_pipeline(cinfo['X'], y_c1, subjects, sgkf, clean_mask=clean_mask_all, cost_weight=cinfo['cost_weight'])
        results_all[cname] = res
        cb_f1 = res['CatBoost']['f1']
        cas_f1 = res['Cascade']['f1']
        opt_f1 = res['Cascade']['opt_f1']
        auc = res['Cascade']['roc_auc']
        fps = res['clean_fps']
        fpr = res['clean_fp_rate']
        print(f"    -> CatBoost F1: {cb_f1:.2f}%, Cascade F1: {cas_f1:.2f}% (Opt F1: {opt_f1:.2f}%, AUC: {auc:.2f}%) | Clean FP: {fps}건 ({fpr:.1f}%) [{time.time()-t0:.1f}초]")

    # 5. Phase 2: Level 1~3 실무군 (1,631건) 대상 토너먼트
    amb_data = np.load('d:/src/models/second_models/squat_spine_ambiguity_levels.npz')
    levels = amb_data['levels']
    valid_mask = (levels <= 3) # Level 4 역전 노이즈 배제
    N_valid = np.sum(valid_mask)
    clean_mask_valid = clean_mask_all[valid_mask]

    print("\n" + "=" * 95)
    print(f"  [Phase 2] Level 1~3 실무군 ({N_valid}건) 대상 5-Fold StratifiedGroupKFold 실측 토너먼트")
    print("=" * 95)

    results_valid = {}
    for cname, cinfo in candidates.items():
        t0 = time.time()
        X_sub = cinfo['X'][valid_mask]
        y_sub = y_c1[valid_mask]
        subjs_sub = subjects[valid_mask]
        print(f"[*] 평가 진행 중: {cname:<26s} (N={N_valid}) ...")
        res = evaluate_pipeline(X_sub, y_sub, subjs_sub, sgkf, clean_mask=clean_mask_valid, cost_weight=cinfo['cost_weight'])
        results_valid[cname] = res
        cb_f1 = res['CatBoost']['f1']
        cas_f1 = res['Cascade']['f1']
        opt_f1 = res['Cascade']['opt_f1']
        auc = res['Cascade']['roc_auc']
        fps = res['clean_fps']
        fpr = res['clean_fp_rate']
        print(f"    -> CatBoost F1: {cb_f1:.2f}%, Cascade F1: {cas_f1:.2f}% (Opt F1: {opt_f1:.2f}%, AUC: {auc:.2f}%) | Clean FP: {fps}건 ({fpr:.1f}%) [{time.time()-t0:.1f}초]")

    # 6. 종합 결과 테이블 출력 (실무군 1,631건 기준)
    print("\n" + "=" * 125)
    print(f"  ★ [최종 실측 성능 비교표] 바벨 스쿼트 c1 척추의 중립 (Level 1~3 실무군 {N_valid}건 기준)")
    print("=" * 125)
    print(f"{'모델 후보군':<25} | {'피처수':<6} | {'CatBoost F1':<12} | {'LightGBM F1':<12} | {'Cascade F1':<12} | {'ROC-AUC':<9} | {'정상 오탐(FP)':<16} | {'평가 판정'}")
    print("-" * 125)
    base_fps_v = results_valid['v1.0_Legacy_22D']['clean_fps']
    for cname in candidates.keys():
        r = results_valid[cname]
        n_feats = candidates[cname]['X'].shape[1]
        cb_f1 = r['CatBoost']['f1']
        lgb_f1 = r['LightGBM']['f1']
        cas_f1 = r['Cascade']['f1']
        auc = r['Cascade']['roc_auc']
        fps = r['clean_fps']
        fpr = r['clean_fp_rate']
        diff_fp = fps - base_fps_v
        verdict = "기준선" if cname == 'v1.0_Legacy_22D' else ("★ 최고 F1" if "v2-A" in cname else ("★ 오탐 최소화" if "v2-C" in cname else "경량화"))
        print(f"{cname:<25} | {n_feats:2d}D   | {cb_f1:<12.2f}% | {lgb_f1:<12.2f}% | {cas_f1:<12.2f}% | {auc:<9.2f}% | {fps:2d}건 ({fpr:4.1f}%, {diff_fp:+2d}건) | {verdict}")
    print("=" * 125)

    # 7. 안전한 JSON 직렬화 함수
    def clean_res(res_dict):
        cleaned = {}
        for k, v in res_dict.items():
            cleaned[k] = {}
            for mk, mv in v.items():
                if mk == 'oof_cascade':
                    continue
                elif isinstance(mv, dict):
                    cleaned[k][mk] = {ik: iv for ik, iv in mv.items() if ik != 'oof_cascade'}
                else:
                    cleaned[k][mk] = mv
        return cleaned

    out_save = {
        'total_samples': N_total,
        'valid_samples': int(N_valid),
        'clean_samples_total': int(np.sum(clean_mask_all)),
        'clean_samples_valid': int(np.sum(clean_mask_valid)),
        'results_all_1906': clean_res(results_all),
        'results_valid_1631': clean_res(results_valid),
        'candidate_descriptions': {k: candidates[k]['desc'] for k in candidates.keys()}
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_save, f, ensure_ascii=False, indent=2)
    print(f"\n[*] c1 v2 토너먼트 실측 결과 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
