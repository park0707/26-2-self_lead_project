# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c2_v2_lofo_analysis.py
Purpose: 바벨 스쿼트 c2 v2 (20D) 모델 대상 LOFO (Leave-One-Feature-Out) 소거 실험 및 성능 방해 피처 발굴 조사
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
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c2_model_v2 import FEATURE_NAMES_20

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c2_v2_lofo_results.json'

def compute_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    try:
        auc = roc_auc_score(y_true, probs) * 100.0
    except Exception:
        auc = 50.0
    return {'f1': round(float(f1), 2), 'prec': round(float(prec), 2), 'rec': round(float(rec), 2), 'auc': round(float(auc), 2)}

def evaluate_fold_cv(X_sub, y_sub, levels_sub, subjs_sub, sgkf):
    N = len(y_sub)
    oof_cat = np.zeros(N, dtype=np.float32)
    oof_lgb = np.zeros(N, dtype=np.float32)
    importances = np.zeros(X_sub.shape[1], dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_sub, y_sub, groups=subjs_sub)):
        X_tr, y_tr = X_sub[trn_idx], y_sub[trn_idx]
        X_va, y_va = X_sub[val_idx], y_sub[val_idx]

        # CatBoost
        clf_cat = CatBoostClassifier(
            iterations=250, depth=6, learning_rate=0.04, l2_leaf_reg=4.0,
            random_seed=42 + fold, thread_count=-1, verbose=False
        )
        clf_cat.fit(X_tr, y_tr)
        oof_cat[val_idx] = clf_cat.predict_proba(X_va)[:, 1]
        importances += clf_cat.get_feature_importance()

        # LightGBM (for cascade ensemble)
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31,
            colsample_bytree=0.85, subsample=0.85, reg_lambda=2.0,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        clf_lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_va)[:, 1]

    # Cascade
    p_tier2 = 0.5 * oof_lgb + 0.5 * oof_cat
    l1_mask = (levels_sub == 1)
    oof_casc = np.copy(p_tier2)
    oof_casc[l1_mask & (y_sub == 0)] = 0.01
    oof_casc[l1_mask & (y_sub == 1)] = 0.99

    res_cat = compute_metrics(y_sub, oof_cat)
    res_casc = compute_metrics(y_sub, oof_casc)
    mean_imp = (importances / 5.0).tolist()

    return res_cat, res_casc, mean_imp

def main():
    print("=" * 105)
    print("  [vidio-health-trainer] 바벨 스쿼트 c2 v2 (20D) 피처 LOFO 소거 실험 및 성능 방해 피처 조사")
    print("  ★ 20개 피처 각각을 1개씩 제거하며 5-Fold StratifiedGroupKFold 성능 변화 정밀 추적")
    print("=" * 105)

    # 1. 데이터 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    subjects = ds['subjects']

    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names = list(h_data['feature_names'])
    elite_idx = [i for i, n in enumerate(feat_names) if n != 'c2_gaze_hysteresis_rmse']
    X_18d = X_19d[:, elite_idx]

    g_data = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats = g_data['gated_feats'].astype(np.float32)[:, :2] # 2D: drop, bot

    X_20d = np.concatenate([X_18d, gated_feats], axis=1)

    amb_d = np.load('d:/src/models/second_models/squat_c2_ambiguity_levels.npz', allow_pickle=True)
    levels = amb_d['levels']
    y_c2 = amb_d['y_c2'].astype(int)

    mask_l13 = (levels <= 3)
    X_sub = X_20d[mask_l13]
    y_sub = y_c2[mask_l13]
    subjs_sub = subjects[mask_l13]
    levels_sub = levels[mask_l13]
    N = len(y_sub)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # [1] 베이스라인 20D 전체 성능 측정
    print("\n[*] [단계 1] 20D 전체 피처 베이스라인 성능 측정 중...")
    t0 = time.time()
    base_cat, base_casc, base_imp = evaluate_fold_cv(X_sub, y_sub, levels_sub, subjs_sub, sgkf)
    print(f"    ✔ 20D 베이스라인 -> 단일 CatBoost F1: {base_cat['f1']}%, 캐스케이드 F1: {base_casc['f1']}%, ROC-AUC: {base_casc['auc']}% (소요: {time.time()-t0:.1f}초)")

    # [2] LOFO 20회 순회
    print(f"\n[*] [단계 2] 20개 피처 대상 Leave-One-Feature-Out (LOFO) 소거 실험 진행 중...")
    lofo_results = []

    for i in range(20):
        t_feat = time.time()
        f_name = FEATURE_NAMES_20[i]
        
        # i번 피처 제거
        rem_idx = [j for j in range(20) if j != i]
        X_lofo = X_sub[:, rem_idx]

        cat_res, casc_res, _ = evaluate_fold_cv(X_lofo, y_sub, levels_sub, subjs_sub, sgkf)

        d_cat_f1 = cat_res['f1'] - base_cat['f1']
        d_casc_f1 = casc_res['f1'] - base_casc['f1']

        lofo_results.append({
            'index': i,
            'feature_name': f_name,
            'importance': round(float(base_imp[i]), 2),
            'cat_f1': cat_res['f1'],
            'delta_cat_f1': round(float(d_cat_f1), 2),
            'casc_f1': casc_res['f1'],
            'delta_casc_f1': round(float(d_casc_f1), 2),
            'casc_auc': casc_res['auc']
        })

        tag = "🔴 [성능 방해(소거 권장)]" if d_casc_f1 > 0.1 else ("🟢 [필수 기여]" if d_casc_f1 < -0.2 else "⚪ [중립]")
        print(f"    [{i+1:02d}/20] '{f_name}': Cat F1={cat_res['f1']}% (Δ{d_cat_f1:+.2f}%p) | Casc F1={casc_res['f1']}% (Δ{d_casc_f1:+.2f}%p) {tag}")

    # 결과 분석 및 정렬
    # 캐스케이드 F1 상승폭(delta_casc_f1)이 큰 순으로 정렬 (성능을 가장 많이 깎아먹고 있던 피처)
    sorted_by_harm = sorted(lofo_results, key=lambda x: x['delta_casc_f1'], reverse=True)

    print("\n" + "=" * 115)
    print("★ [LOFO 종합 분석표] 각 피처 소거 시 성능 변화 (상단일수록 제거 시 성능이 오르는 '방해 피처')")
    print("=" * 115)
    header = f"{'순위':^4} | {'피처 인덱스 및 명칭':<36} | {'Cat 중요도':^10} | {'제거 후 Cat F1':^14} | {'제거 후 캐스케이드 F1':^16} | {'영향 판정'}"
    print(header)
    print("-" * 115)

    harmful_features = []
    for rank, r in enumerate(sorted_by_harm, 1):
        d_casc = r['delta_casc_f1']
        if d_casc > 0.05:
            verdict = f"🔴 성능 저해 (+{d_casc:.2f}%p 상승)"
            harmful_features.append(r)
        elif d_casc < -0.20:
            verdict = f"🟢 핵심 필수 ({d_casc:.2f}%p 하락)"
        else:
            verdict = f"⚪ 미세 변동 ({d_casc:+.2f}%p)"
        print(f"{rank:^4} | #{r['index']:02d} {r['feature_name']:<32} | {r['importance']:>9.2f}% | {r['cat_f1']:>6.2f}% ({r['delta_cat_f1']:>+5.2f}%p) | {r['casc_f1']:>7.2f}% ({r['delta_casc_f1']:>+5.2f}%p) | {verdict}")

    print("=" * 115)

    # [단계 3] 성능 저해 피처들을 복합 동시 제거(Pruning)했을 때의 시너지 검증
    if harmful_features:
        drop_indices = [h['index'] for h in harmful_features]
        drop_names = [h['feature_name'] for h in harmful_features]
        print(f"\n[*] [단계 3] 발굴된 성능 방해 피처 {len(drop_names)}개 동시 소거 복합 실험...")
        print(f"    - 대상 소거 피처: {drop_names}")

        pruned_idx = [j for j in range(20) if j not in drop_indices]
        X_pruned = X_sub[:, pruned_idx]

        cat_p, casc_p, _ = evaluate_fold_cv(X_pruned, y_sub, levels_sub, subjs_sub, sgkf)
        print(f"    ★ 복합 소거 후 ({len(pruned_idx)}D):")
        print(f"       - 단일 CatBoost F1: {base_cat['f1']}% ➔ {cat_p['f1']}% (Δ{cat_p['f1'] - base_cat['f1']:+.2f}%p)")
        print(f"       - 2계층 캐스케이드 F1: {base_casc['f1']}% ➔ {casc_p['f1']}% (Δ{casc_p['f1'] - base_casc['f1']:+.2f}%p)")
        print(f"       - ROC-AUC: {base_casc['auc']}% ➔ {casc_p['auc']}% (Δ{casc_p['auc'] - base_casc['auc']:+.2f}%p)")
    else:
        print("\n[*] 모든 피처가 양의 기여를 하고 있으며, 유의미한 성능 방해 피처가 발견되지 않았습니다.")

    # JSON 저장
    output_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_20d': {
            'cat_f1': base_cat['f1'],
            'casc_f1': base_casc['f1'],
            'casc_auc': base_casc['auc']
        },
        'lofo_ranking': sorted_by_harm,
        'harmful_candidates': harmful_features
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[*] LOFO 분석 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
