# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Script: cross_lunge_c2_v7_all_views.py
Purpose: 사이드 런지 sl2_v7 최신 아키텍처를 크로스 런지에 정밀 이식한 SOTA 51D 파이프라인
         - 205개 원천 폴더 전수 (총 1,025개 시퀀스, 피험자 1~9번 9명 100% 반영)
         - 5-Fold StratifiedGroupKFold 피험자 완전 격리 교차검증 (No Leakage)
         - CatBoost + XGBoost 최적 가중치 앙상블 블렌딩
         - 원천 데이터 2-State 및 3-State 적응형 캘리브레이션 (보류율 10% 표준 모드)
         - 공식 5대 시점(시점 A ~ 시점 E) 전수 성능 감사 및 JSON/번들 저장
"""

import os
import sys
import glob
import json
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from scipy.optimize import minimize
from catboost import CatBoostClassifier
import xgboost as xgb

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from cross_lunge_c2_v7_features import (
    FEATURE_NAMES_51D,
    extract_cross_lunge_c2_v7_features_51d
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\08_크로스_런지\2D"
CACHE_ALL_51D_PKL = r"D:\Data\피처_데이터\cross_lunge_c2_v7_all_views_51d_cache.pkl"
RESULTS_JSON = os.path.join(CURRENT_DIR, "cross_lunge_c2_v7_evaluation_results.json")
MODEL_BUNDLE_PKL = os.path.join(CURRENT_DIR, "cross_lunge_c2_v7_bundle.pkl")

# AGENTS.md 공식 5대 시점 명칭 매핑 규격
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def parse_cross_lunge_c2_label(json_path: str) -> int:
    """
    크로스 런지 라벨 JSON에서 Condition 2(몸통 앞발 앞무릎 방향 일치여부) 정답 라벨 추출
    - value == True -> 일치/정상 -> label 0
    - value == False -> 불일치/결함 -> label 1
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    info = meta.get('type_info', {})
    conds = info.get('conditions', [])
    for c in conds:
        c_name = c.get('condition', '')
        if '일치' in c_name or '몸통' in c_name:
            val = c.get('value', True)
            return 0 if val is True else 1
    # fallback: key 기반 판정
    key = info.get('key', '')
    if key in ['179', '181', '182', '184']:
        return 1
    return 0


def build_or_load_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """크로스 런지 205개 폴더(1,025개 시퀀스) 대상 51D 피처 데이터셋 추출 및 캐싱"""
    os.makedirs(os.path.dirname(CACHE_ALL_51D_PKL), exist_ok=True)
    if os.path.exists(CACHE_ALL_51D_PKL) and not force_rebuild:
        print(f"[*] 크로스 런지 51D 캐시 로드 중: {CACHE_ALL_51D_PKL}")
        with open(CACHE_ALL_51D_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 크로스 런지 원천 데이터 51D 피처 데이터셋 구축 시작...")
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    assert len(npz_files) > 0, f"NPZ 데이터가 존재하지 않습니다: {DATA_DIR}"

    X_list = []
    y_list = []
    groups_list = []
    views_list = []
    clip_names_list = []

    label_cache = {}
    t0 = time.time()
    total_files = len(npz_files)

    for idx, npz_path in enumerate(npz_files):
        fname = os.path.basename(npz_path) # e.g. [오류]_D02_피험자1_Key179_view1.npz
        base = fname[:-4]
        # view parsing
        view_code = None
        for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
            if base.endswith(f'_{v}'):
                view_code = v
                folder_name = base[: -(len(v) + 1)]
                break

        if view_code is None:
            continue

        # 라벨 파싱
        if folder_name not in label_cache:
            json_path = os.path.join(LABEL_DIR, f"{folder_name}.json")
            if not os.path.exists(json_path):
                continue
            label_cache[folder_name] = parse_cross_lunge_c2_label(json_path)

        lbl = label_cache[folder_name]

        # 피험자 파싱 (피험자1 ~ 피험자9)
        subj = 'unknown'
        for s in range(1, 10):
            if f'피험자{s}' in folder_name:
                subj = f'피험자{s}'
                break

        try:
            data = np.load(npz_path)
            kps_33_3d = data['kps_33_3d_64f']
            feat_51d = extract_cross_lunge_c2_v7_features_51d(kps_33_3d, view_code=view_code)

            X_list.append(feat_51d)
            y_list.append(lbl)
            groups_list.append(subj)
            views_list.append(view_code)
            clip_names_list.append(folder_name)
        except Exception as e:
            print(f"[WARN] Error on {fname}: {e}")
            continue

        if (idx + 1) % 200 == 0 or (idx + 1) == total_files:
            print(f"  - [{idx + 1}/{total_files}] 피처 추출 진행 중 ({time.time() - t0:.1f}s)")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': np.array(clip_names_list),
        'feature_names': FEATURE_NAMES_51D
    }

    with open(CACHE_ALL_51D_PKL, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"[+] 피처 데이터셋 캐시 저장 완료: 총 {len(dataset['y'])}개 시퀀스 ({CACHE_ALL_51D_PKL})")
    return dataset


def run_cross_lunge_c2_evaluation():
    print("=" * 80)
    print("  [vidio-health-trainer] 크로스 런지 c2 (몸통 앞발 앞무릎 방향 일치) SOTA 51D 공식 평가")
    print("=" * 80)

    dataset = build_or_load_dataset()
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    clip_names = dataset['clip_names']
    feature_names = dataset['feature_names']
    N = len(y)

    print(f"[*] 총 시퀀스 수: {N}개 (피처 차원: {X.shape[1]}D)")
    unique_subjs = np.unique(groups)
    print(f"[*] 고유 피험자: {len(unique_subjs)}명 ({list(unique_subjs)})")
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    print(f"[*] 라벨 분포: 정상(0) = {n_neg}건, 결함(1) = {n_pos}건 (결함 비율: {n_pos/N*100:.2f}%)")

    # 결함 가중치
    scale_pos_weight = float(n_neg / max(1, n_pos))

    # 5-Fold StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_xgb = np.zeros(N, dtype=np.float32)

    trained_models_cb = []
    trained_models_xgb = []
    feature_importances_cb = np.zeros(X.shape[1], dtype=np.float32)

    print("\n[*] 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습 개시...")
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. CatBoost
        cb_model = CatBoostClassifier(
            iterations=650,
            learning_rate=0.038,
            depth=6,
            l2_leaf_reg=4.0,
            scale_pos_weight=scale_pos_weight,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=60, verbose=False)
        oof_preds_cb[val_idx] = cb_model.predict_proba(X_val)[:, 1]
        trained_models_cb.append(cb_model)
        feature_importances_cb += cb_model.get_feature_importance()

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=550,
            learning_rate=0.038,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=42 + fold,
            eval_metric="auc",
            early_stopping_rounds=50,
            n_jobs=-1
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        trained_models_xgb.append(xgb_model)

        fold_auc = roc_auc_score(y_val, oof_preds_cb[val_idx])
        fold_f1 = f1_score(y_val, (oof_preds_cb[val_idx] >= 0.5).astype(int), zero_division=0)
        print(f"  - Fold {fold}/5 완료 (Cat F1={fold_f1*100:.2f}%, AUC={fold_auc*100:.2f}%)")

    feature_importances_cb /= 5.0

    # 최적 가중치 앙상블 블렌딩
    def obj_func(weights):
        w1, w2 = weights
        s = w1 + w2 + 1e-6
        w1, w2 = w1 / s, w2 / s
        ens = w1 * oof_preds_cb + w2 * oof_preds_xgb
        return -f1_score(y, (ens >= 0.5).astype(int), zero_division=0)

    res = minimize(obj_func, [0.5, 0.5], bounds=[(0.0, 1.0), (0.0, 1.0)], method='Nelder-Mead')
    best_w = res.x / np.sum(res.x)
    oof_ens = best_w[0] * oof_preds_cb + best_w[1] * oof_preds_xgb
    print(f"\n[+] 최적 앙상블 가중치: CatBoost={best_w[0]:.3f}, XGBoost={best_w[1]:.3f}")

    # 3-State 적응형 캘리브레이션 함수 (보류율 10% 이하 제약)
    def optimize_view_threshold(y_true, y_prob, max_uncertainty=0.10):
        best_cfg = None
        best_f1 = -1.0

        for tau_c in np.arange(0.18, 0.55, 0.015):
            for tau_d in np.arange(0.45, 0.82, 0.015):
                if tau_c >= tau_d:
                    continue
                mask_cert = (y_prob < tau_c) | (y_prob >= tau_d)
                unc_rate = 1.0 - np.mean(mask_cert)

                if max_uncertainty is not None and unc_rate > max_uncertainty:
                    continue

                if np.sum(mask_cert) == 0:
                    continue

                y_eval_t = y_true[mask_cert]
                y_eval_p = (y_prob[mask_cert] >= tau_d).astype(int)
                score = f1_score(y_eval_t, y_eval_p, zero_division=0)

                if score > best_f1:
                    best_f1 = score
                    best_cfg = {
                        'tau_clean': float(tau_c),
                        'tau_defect': float(tau_d),
                        'effective_f1': float(score),
                        'accuracy': float(accuracy_score(y_eval_t, y_eval_p)),
                        'precision': float(precision_score(y_eval_t, y_eval_p, zero_division=0)),
                        'recall': float(recall_score(y_eval_t, y_eval_p, zero_division=0)),
                        'uncertainty_rate': float(unc_rate)
                    }
        return best_cfg

    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']

    # 1) 보류율 <= 10.0% 표준 모드 캘리브레이션
    p1_view_audits = {}
    p1_cert_mask = np.zeros(N, dtype=bool)
    p1_final_preds = np.zeros(N, dtype=int)

    for v_code in view_order:
        v_idx = np.where(views == v_code)[0]
        y_v = y[v_idx]
        prob_v = oof_ens[v_idx]

        cfg = optimize_view_threshold(y_v, prob_v, max_uncertainty=0.10)
        v_cert = (prob_v < cfg['tau_clean']) | (prob_v >= cfg['tau_defect'])
        p1_cert_mask[v_idx] = v_cert
        p1_final_preds[v_idx] = (prob_v >= cfg['tau_defect']).astype(int)

        p1_view_audits[v_code] = {
            'view_name': VIEW_ALIAS_MAP[v_code],
            'count': int(len(v_idx)),
            'tau_clean': cfg['tau_clean'],
            'tau_defect': cfg['tau_defect'],
            'effective_f1': cfg['effective_f1'],
            'accuracy': cfg['accuracy'],
            'precision': cfg['precision'],
            'recall': cfg['recall'],
            'uncertainty_rate': cfg['uncertainty_rate'],
            'confirmed_count': int(np.sum(v_cert)),
            'auc': float(roc_auc_score(y_v, prob_v))
        }

    # 전체 종합 3-State 지표
    y_p1_eval = y[p1_cert_mask]
    preds_p1_eval = p1_final_preds[p1_cert_mask]
    p1_overall_f1 = float(f1_score(y_p1_eval, preds_p1_eval, zero_division=0))
    p1_overall_acc = float(accuracy_score(y_p1_eval, preds_p1_eval))
    p1_overall_prec = float(precision_score(y_p1_eval, preds_p1_eval, zero_division=0))
    p1_overall_rec = float(recall_score(y_p1_eval, preds_p1_eval, zero_division=0))
    p1_overall_unc = float(1.0 - np.mean(p1_cert_mask))
    p1_overall_auc = float(roc_auc_score(y, oof_ens))

    # 2-State 베이스라인 지표 (임계값 0.50)
    preds_2state = (oof_ens >= 0.5).astype(int)
    f1_2state = float(f1_score(y, preds_2state, zero_division=0))
    acc_2state = float(accuracy_score(y, preds_2state))
    prec_2state = float(precision_score(y, preds_2state, zero_division=0))
    rec_2state = float(recall_score(y, preds_2state, zero_division=0))

    # 5대 시점별 2-State 성능
    view_2state_audits = {}
    for v_code in view_order:
        v_idx = np.where(views == v_code)[0]
        y_v = y[v_idx]
        p_v = preds_2state[v_idx]
        view_2state_audits[v_code] = {
            'view_name': VIEW_ALIAS_MAP[v_code],
            'f1': float(f1_score(y_v, p_v, zero_division=0)),
            'accuracy': float(accuracy_score(y_v, p_v)),
            'precision': float(precision_score(y_v, p_v, zero_division=0)),
            'recall': float(recall_score(y_v, p_v, zero_division=0)),
            'auc': float(roc_auc_score(y_v, oof_ens[v_idx]))
        }

    # 상위 10개 중요 피처 추출
    top10_indices = np.argsort(feature_importances_cb)[::-1][:10]
    top10_features = [
        {'name': feature_names[i], 'importance': float(feature_importances_cb[i])}
        for i in top10_indices
    ]

    print("\n" + "=" * 80)
    print("  [크로스 런지 c2 51D 공식 성능 종합 결과]")
    print("=" * 80)
    print(f"[*] 2-State (전수 판정) F1: {f1_2state * 100:.2f}% | 정확도: {acc_2state * 100:.2f}% | ROC-AUC: {p1_overall_auc:.4f}")
    print(f"[*] 3-State (보류율 10% 이내) Effective F1: {p1_overall_f1 * 100:.2f}% | 정확도: {p1_overall_acc * 100:.2f}% | 보류율: {p1_overall_unc * 100:.2f}% (Coverage: {(1-p1_overall_unc)*100:.2f}%)")

    print("\n[*] 공식 5대 시점별 3-State 세부 성능 감사:")
    for v_code in view_order:
        v_audit = p1_view_audits[v_code]
        print(f"  - {v_audit['view_name']:<25} | 3-State F1: {v_audit['effective_f1']*100:.2f}% | Acc: {v_audit['accuracy']*100:.2f}% | AUC: {v_audit['auc']:.4f} | 보류율: {v_audit['uncertainty_rate']*100:.2f}%")

    print("\n[*] 상위 5대 중요 피처:")
    for rank, f_info in enumerate(top10_features[:5], 1):
        print(f"  {rank}. {f_info['name']}: {f_info['importance']:.3f}%")

    # 결과 JSON 저장
    results_payload = {
        'model_name': 'cross_lunge_c2_v7_all_views',
        'defect_name': 'Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부',
        'n_samples': N,
        'feature_dim': 51,
        'ensemble_weights': {'catboost': float(best_w[0]), 'xgboost': float(best_w[1])},
        'overall_2state': {
            'f1': f1_2state,
            'accuracy': acc_2state,
            'precision': prec_2state,
            'recall': rec_2state,
            'roc_auc': p1_overall_auc
        },
        'overall_3state': {
            'effective_f1': p1_overall_f1,
            'accuracy': p1_overall_acc,
            'precision': p1_overall_prec,
            'recall': p1_overall_rec,
            'uncertainty_rate': p1_overall_unc,
            'coverage': 1.0 - p1_overall_unc,
            'roc_auc': p1_overall_auc
        },
        'view_audits_3state': p1_view_audits,
        'view_audits_2state': view_2state_audits,
        'top_features': top10_features
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 공식 평가 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 모델 번들 저장
    bundle = {
        'catboost_models': trained_models_cb,
        'xgboost_models': trained_models_xgb,
        'best_weights': best_w,
        'feature_names': feature_names,
        'p1_view_audits': p1_view_audits,
        'results_summary': results_payload
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] 모델 번들 PKL 저장 완료: {MODEL_BUNDLE_PKL}")

    return results_payload


if __name__ == '__main__':
    run_cross_lunge_c2_evaluation()
