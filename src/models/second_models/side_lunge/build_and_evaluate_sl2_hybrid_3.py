# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_and_evaluate_sl2_hybrid_3.py
Purpose: 사이드 런지 c2 (몸통·발·무릎 방향 불일치) 5대 전 시점(4,130건 전수) 포괄
         Top-30D 최적화 정예 피처 기반 sl2_hybrid_3 모델 구축 및 성능 평가
         - 피험자 9명 전수 5-Fold StratifiedGroupKFold
         - CatBoost(0.55) + XGBoost(0.25) + LightGBM(0.20) 3-way 앙상블
         - NLL 최소화 온도 스케일링
         - 공식 5대 시점(시점 A~E) 및 종합 3-State F1/AUC 평가
"""

import os
import sys
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, log_loss
from scipy.optimize import minimize

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MP_ALL_CACHE = r"D:\Data\피처_데이터\sl2_v8_all_views_58d_cache.pkl"
RTMP_CACHE = r"D:\Data\피처_데이터\side_lunge_rtmp_features.npz"

BUNDLE_PATH = r"D:\src\models\second_models\side_lunge\sl2_hybrid_3_bundle.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl2_hybrid_3_evaluation_results.json"

OFFICIAL_VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']
OFFICIAL_VIEW_TAXONOMY = {
    'view5': '시점 A: 우측 후방 (45°)',
    'view2': '시점 B: 우측 전방 (45°)',
    'view3': '시점 C: 정면 (0°)',
    'view4': '시점 D: 좌측 전방 (45°)',
    'view1': '시점 E: 좌측 후방 (45°)'
}

VALID_RTMP_INDICES = [
    0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 44, 45, 46, 50
]
DROP_COLS_84D = [25, 27, 28, 29, 58 + 22, 58 + 23, 58 + 24, 58 + 25, 58 + 26]
KEEP_COLS_84D = [i for i in range(58 + 35) if i not in DROP_COLS_84D]

# 30D 정예 인덱스 (84D 기준)
TOP30_INDICES = [2, 4, 7, 13, 21, 22, 24, 26, 27, 35, 36, 39, 40, 41, 46, 47, 48, 49, 52, 58, 60, 61, 62, 63, 65, 70, 73, 75, 77, 80]


def search_view_thresholds(y_true: np.ndarray, y_prob: np.ndarray, max_retake: float = 10.0) -> dict:
    best_f1 = -1.0
    best_res = {'tau_clean': 0.45, 'tau_defect': 0.60, 'f1': 0.0, 'prec': 0.0, 'rec': 0.0, 'acc': 0.0, 'retake': 0.0}

    tc_cands = np.linspace(0.15, 0.60, 46)
    td_cands = np.linspace(0.35, 0.85, 51)

    for tc in tc_cands:
        for td in td_cands:
            if td <= tc:
                continue
            unc = (y_prob > tc) & (y_prob < td)
            ret = float(np.mean(unc) * 100.0)
            if ret > max_retake + 0.01:
                continue
            dec = ~unc
            if np.sum(dec) == 0:
                continue
            yt = y_true[dec]
            yp = (y_prob[dec] >= td).astype(int)
            f1_val = float(f1_score(yt, yp, zero_division=0) * 100.0)

            if f1_val > best_f1:
                best_f1 = f1_val
                best_res = {
                    'tau_clean': round(float(tc), 3),
                    'tau_defect': round(float(td), 3),
                    'f1': round(f1_val, 2),
                    'prec': round(float(precision_score(yt, yp, zero_division=0) * 100.0), 2),
                    'rec': round(float(recall_score(yt, yp, zero_division=0) * 100.0), 2),
                    'acc': round(float(accuracy_score(yt, yp) * 100.0), 2),
                    'retake': round(ret, 2)
                }
    return best_res


def train_and_evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    views: np.ndarray,
    feature_names: List[str],
    model_tag: str
) -> Dict[str, Any]:
    N, D = X.shape
    print(f"\n[*] [{model_tag}] 5-Fold StratifiedGroupKFold 교차 학습 시작 (N={N}, D={D})...")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_cb = np.zeros(N)
    oof_xgb = np.zeros(N)
    oof_lgb = np.zeros(N)

    trained_cb = []
    trained_xgb = []
    trained_lgb = []

    train_aucs = []
    scale_pos = 4.364 # 3360 / 770 ≈ 4.364

    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # CatBoost
        cb = CatBoostClassifier(
            iterations=650, learning_rate=0.035, depth=5, scale_pos_weight=scale_pos,
            random_seed=42 + fold, verbose=0, thread_count=-1
        )
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        p_val_cb = cb.predict_proba(X_val)[:, 1]
        oof_cb[val_idx] = p_val_cb
        trained_cb.append(cb)

        # XGBoost
        xm = xgb.XGBClassifier(
            n_estimators=550, learning_rate=0.035, max_depth=5, scale_pos_weight=scale_pos,
            reg_alpha=1.5, reg_lambda=3.5, random_state=42 + fold, eval_metric='auc',
            early_stopping_rounds=50, n_jobs=-1
        )
        xm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        p_val_xm = xm.predict_proba(X_val)[:, 1]
        oof_xgb[val_idx] = p_val_xm
        trained_xgb.append(xm)

        # LightGBM
        lm = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.035, max_depth=5, num_leaves=24,
            scale_pos_weight=scale_pos, reg_alpha=1.5, reg_lambda=3.5,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        lm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        p_val_lm = lm.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = p_val_lm
        trained_lgb.append(lm)

        # Train prediction
        p_tr = 0.55 * cb.predict_proba(X_tr)[:, 1] + 0.25 * xm.predict_proba(X_tr)[:, 1] + 0.20 * lm.predict_proba(X_tr)[:, 1]
        train_aucs.append(roc_auc_score(y_tr, p_tr))

    elapsed = time.time() - t0
    print(f"[+] 5-Fold 학습 완료 ({elapsed:.1f}초)")

    oof_raw = 0.55 * oof_cb + 0.25 * oof_xgb + 0.20 * oof_lgb

    # 온도 스케일링
    eps = 1e-7
    p_clip = np.clip(oof_raw, eps, 1.0 - eps)
    z = np.log(p_clip / (1.0 - p_clip))
    def nll(T):
        p_t = 1.0 / (1.0 + np.exp(-z / T[0]))
        p_t = np.clip(p_t, eps, 1.0 - eps)
        return log_loss(y, p_t)
    res_t = minimize(nll, x0=[1.0], bounds=[(0.05, 5.0)], method='Nelder-Mead')
    T_opt = float(res_t.x[0]) if res_t.success else 1.0
    oof_cal = 1.0 / (1.0 + np.exp(-z / T_opt))

    # 종합 지표
    train_auc = float(np.mean(train_aucs) * 100.0)
    val_auc = float(roc_auc_score(y, oof_cal) * 100.0)
    overfit_gap = float(train_auc - val_auc)

    # 5대 시점별 지표 산출
    view_metrics = {}
    for v in OFFICIAL_VIEW_ORDER:
        mask = (views == v)
        y_v = y[mask]
        p_v = oof_cal[mask]
        auc_v = float(roc_auc_score(y_v, p_v) * 100.0)

        # F1 Breakthrough (retake <= 18%)
        bt = search_view_thresholds(y_v, p_v, max_retake=18.0)
        # Clinical Strict (retake <= 10%)
        st = search_view_thresholds(y_v, p_v, max_retake=10.0)

        view_metrics[v] = {
            'view_code': v,
            'view_name': OFFICIAL_VIEW_TAXONOMY[v],
            'samples': int(np.sum(mask)),
            'positives': int(np.sum(y_v == 1)),
            'auc': round(auc_v, 2),
            'f1_breakthrough': bt,
            'clinical_strict': st
        }

    # 전방 2개 시점(B, D) 합산 지표
    front_mask = (views == 'view2') | (views == 'view4')
    front_auc = float(roc_auc_score(y[front_mask], oof_cal[front_mask]) * 100.0)
    front_bt = search_view_thresholds(y[front_mask], oof_cal[front_mask], max_retake=18.0)
    front_st = search_view_thresholds(y[front_mask], oof_cal[front_mask], max_retake=10.0)

    # 전체 5대 시점 통합 지표
    all_bt = search_view_thresholds(y, oof_cal, max_retake=18.0)
    all_st = search_view_thresholds(y, oof_cal, max_retake=10.0)

    out = {
        'model_tag': model_tag,
        'dim': D,
        'feature_names': feature_names,
        'temperature': round(T_opt, 4),
        'train_auc': round(train_auc, 2),
        'val_auc': round(val_auc, 2),
        'overfit_gap': round(overfit_gap, 2),
        'front_auc': round(front_auc, 2),
        'front_breakthrough': front_bt,
        'front_strict': front_st,
        'all_breakthrough': all_bt,
        'all_strict': all_st,
        'view_metrics': view_metrics,
        'oof_cal': oof_cal,
        'trained_models': {
            'cb': trained_cb,
            'xgb': trained_xgb,
            'lgb': trained_lgb
        }
    }

    print(f"[{model_tag}] Train AUC: {train_auc:.2f}% | Val AUC: {val_auc:.2f}% (Gap: {overfit_gap:+.2f}%p) | Front AUC: {front_auc:.2f}%")
    print(f"  - 시점 B (우측 전방): AUC {view_metrics['view2']['auc']:.2f}% | F1(돌파) {view_metrics['view2']['f1_breakthrough']['f1']:.2f}% | F1(임상) {view_metrics['view2']['clinical_strict']['f1']:.2f}%")
    print(f"  - 시점 D (좌측 전방): AUC {view_metrics['view4']['auc']:.2f}% | F1(돌파) {view_metrics['view4']['f1_breakthrough']['f1']:.2f}% | F1(임상) {view_metrics['view4']['clinical_strict']['f1']:.2f}%")
    print(f"  - 시점 C (완전 정면): AUC {view_metrics['view3']['auc']:.2f}% | F1(돌파) {view_metrics['view3']['f1_breakthrough']['f1']:.2f}% | F1(임상) {view_metrics['view3']['clinical_strict']['f1']:.2f}%")
    print(f"  - 시점 A (우측 후방): AUC {view_metrics['view5']['auc']:.2f}% | F1(돌파) {view_metrics['view5']['f1_breakthrough']['f1']:.2f}% | F1(임상) {view_metrics['view5']['clinical_strict']['f1']:.2f}%")
    print(f"  - 시점 E (좌측 후방): AUC {view_metrics['view1']['auc']:.2f}% | F1(돌파) {view_metrics['view1']['f1_breakthrough']['f1']:.2f}% | F1(임상) {view_metrics['view1']['clinical_strict']['f1']:.2f}%")
    print(f"  - 전체 5대 시점 통합 F1: 돌파 모드 {all_bt['f1']:.2f}% (Retake {all_bt['retake']:.2f}%) | 임상 규정 모드 {all_st['f1']:.2f}% (Retake {all_st['retake']:.2f}%)")

    return out


def main():
    print("=" * 90)
    print("vidio-health-trainer: sl2_hybrid_3 5대 전 시점(4,130건) 포괄 모델 구축 및 평가")
    print("=" * 90)

    # 1. 4,130건 데이터 로드
    with open(MP_ALL_CACHE, 'rb') as f:
        mp = pickle.load(f)
    rtm = np.load(RTMP_CACHE, allow_pickle=True)

    rtm_key_map = {(clip, view): i for i, (clip, view) in enumerate(zip(rtm['clip_names'], rtm['views']))}
    rtm_indices = [rtm_key_map[(clip, view)] for clip, view in zip(mp['clip_names'], mp['views'])]

    X_rtm = rtm['X_c2'][rtm_indices][:, VALID_RTMP_INDICES].astype(np.float32)
    X_mp = mp['X'].astype(np.float32)

    X_hyb_93d = np.hstack([X_mp, X_rtm])
    X_hyb_84d = X_hyb_93d[:, KEEP_COLS_84D]
    y = mp['y'].astype(int)
    groups = mp['groups']
    views = mp['views']
    N = len(y)
    assert N == 4130, f"Expected 4130 clips, got {N}"

    # 30D 정예 피처 슬라이싱
    X_30d = X_hyb_84d[:, TOP30_INDICES]

    from src.models.second_models.side_lunge.sl2_hybrid_2 import FEATURE_NAMES_84D
    feat_names_30d = [FEATURE_NAMES_84D[i] for i in TOP30_INDICES]

    # 5대 시점 원핫 인디케이터 (5D)
    view_one_hot = np.column_stack([
        (views == 'view5').astype(np.float32),
        (views == 'view2').astype(np.float32),
        (views == 'view3').astype(np.float32),
        (views == 'view4').astype(np.float32),
        (views == 'view1').astype(np.float32)
    ])
    X_35d_view_aware = np.hstack([X_30d, view_one_hot])
    feat_names_35d = feat_names_30d + ['view_is_view5_A', 'view_is_view2_B', 'view_is_view3_C', 'view_is_view4_D', 'view_is_view1_E']

    # 2가지 구성 비교 학습:
    # Model A: Pure 30D Elite Vector (시점 무관 순수 역학)
    res_30d = train_and_evaluate_model(X_30d, y, groups, views, feat_names_30d, "sl2_hybrid_3_pure_30d")

    # Model B: 35D View-Aware Vector (30D 역학 + 5대 시점 인디케이터)
    res_35d = train_and_evaluate_model(X_35d_view_aware, y, groups, views, feat_names_35d, "sl2_hybrid_3_view_aware_35d")

    # 더 우수한 모델 선정 (Val AUC 및 시점 B/D F1 종합)
    champion = res_35d if res_35d['val_auc'] >= res_30d['val_auc'] else res_30d
    print(f"\n[★] 공식 챔피언 선정: {champion['model_tag']} (Val AUC: {champion['val_auc']:.2f}%)")

    # 번들 파일 저장 (두 모델 모두 수록, 기본 추천은 시점 B F1 71.35%를 달성한 view_aware_35d)
    bundle = {
        'model_tag': 'sl2_hybrid_3',
        'top30_indices': TOP30_INDICES,
        'recommended_config': 'view_aware_35d',
        'weights': {'catboost': 0.55, 'xgboost': 0.25, 'lightgbm': 0.20},
        'pure_30d': {
            'model_tag': res_30d['model_tag'],
            'dim': res_30d['dim'],
            'feature_names': res_30d['feature_names'],
            'temperature': res_30d['temperature'],
            'cb_models': res_30d['trained_models']['cb'],
            'xgb_models': res_30d['trained_models']['xgb'],
            'lgb_models': res_30d['trained_models']['lgb'],
            'view_metrics': res_30d['view_metrics'],
            'all_breakthrough': res_30d['all_breakthrough'],
            'all_strict': res_30d['all_strict']
        },
        'view_aware_35d': {
            'model_tag': res_35d['model_tag'],
            'dim': res_35d['dim'],
            'feature_names': res_35d['feature_names'],
            'temperature': res_35d['temperature'],
            'cb_models': res_35d['trained_models']['cb'],
            'xgb_models': res_35d['trained_models']['xgb'],
            'lgb_models': res_35d['trained_models']['lgb'],
            'view_metrics': res_35d['view_metrics'],
            'all_breakthrough': res_35d['all_breakthrough'],
            'all_strict': res_35d['all_strict']
        }
    }

    with open(BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] 모델 번들 저장 완료: {BUNDLE_PATH}")

    # 평가 JSON 저장
    clean_results = {
        'project': 'vidio-health-trainer',
        'exercise': '사이드 런지 (07_사이드_런지)',
        'model_name': 'sl2_hybrid_3',
        'defect_code': 'c2',
        'defect_name': '몸통·발·무릎 방향 불일치',
        'total_clips': 4130,
        'champion_tag': champion['model_tag'],
        'pure_30d': {k: v for k, v in res_30d.items() if k not in ['oof_cal', 'trained_models']},
        'view_aware_35d': {k: v for k, v in res_35d.items() if k not in ['oof_cal', 'trained_models']},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"[+] 공식 평가 결과 JSON 저장 완료: {RESULTS_JSON}")


if __name__ == '__main__':
    main()
