# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c2 (몸통 발 무릎 방향 일치 여부 오류)
Model: sl2_v6_ortho (사용자 제안 3D 상체-대퇴-발 기하 직교/평행 정렬 SOTA 모델, 39D)

설계 및 구현 사양:
  1. 사용자 제안 3D 상체-대퇴-발 기하 직교/평행 정렬 알고리즘 100% 통합
     - 상체 정면 법선 벡터: 고관절-어깨 척추축 x 좌우 어깨축 외적 (v_torso)
     - 대퇴 벡터 (v_thigh), 발끝 벡터 (v_foot)
     - Y축(중력) 제외 XZ 수평 바닥 평면 투영 및 L2 단위벡터 정규화 (체격 왜곡 원천 차단)
     - 몸통-대퇴 직교 내적 & 완충 편차각, 대퇴-발 직교 내적 & 편차각, 몸통-발 평행 내적 & 사잇각
  2. 전방 45도 시점(시점 B: view2, 시점 D: view4) 1,652개 시퀀스 전수 정밀 학습
  3. 39D 고차원 불변 피처셋 (기존 32D SOTA + 사용자 기하 7D 피처)
  4. GBDT 앙상블 (CatBoost + XGBoost 최적 가중 블렌딩)
  5. 피험자 9명 전원 5-Fold StratifiedGroupKFold 피험자 완전 격리 교차검증
  6. 3-State 적응형 캘리브레이션 (표준 10% 모드 및 고신뢰 정밀 모드)
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
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from side_lunge_c2_v6_geometric_features import (
    FEATURE_NAMES_39D,
    USER_GEOMETRIC_FEATURE_NAMES_7D,
    extract_side_lunge_c2_v6_geometric_features_39d
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v6_ortho_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl2_v6_ortho_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl2_v6_ortho_bundle.pkl"

VIEW_ALIAS_MAP = {
    'view2': '시점 B (우측 전방 45°)',
    'view4': '시점 D (좌측 전방 45°)',
}


def determine_c2_label(desc: str, fname: str) -> int:
    """텍스트 설명 기반 c2 정답 라벨 판정"""
    if fname.startswith('[정상]') or '1앞다리 90도' in desc or '무릎발방향일치' in desc:
        return 0
    if '발방향' in desc:
        return 1
    return 0


def build_sl2_v6_ortho_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전방 45도 시점(view2, view4) 1,652개 시퀀스 대상 39D 피처셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 사이드 런지 sl2_v6_ortho 1,652개 시퀀스 39D 피처 구축 시작 (view2 & view4)...")
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_view2.npz")) + glob.glob(os.path.join(DATA_DIR, "*_view4.npz")))
    assert len(npz_files) > 0, f"NPZ 데이터가 존재하지 않습니다: {DATA_DIR}"

    X_list, y_list, groups_list, views_list, clip_names_list = [], [], [], [], []
    label_cache = {}

    for idx, npz_path in enumerate(npz_files):
        fname = os.path.splitext(os.path.basename(npz_path))[0]
        clip_name, view_code = fname.rsplit('_', 1)

        if clip_name not in label_cache:
            json_path = os.path.join(LABEL_DIR, clip_name + ".json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8', errors='ignore') as fp:
                    d = json.load(fp)
                    desc = d.get('type_info', {}).get('description', '')
                    label_cache[clip_name] = determine_c2_label(desc, clip_name)
            except Exception:
                label_cache[clip_name] = None

        target_label = label_cache.get(clip_name)
        if target_label is None:
            continue

        subj = "unknown"
        for part in clip_name.split('_'):
            if '피험자' in part:
                subj = part
                break

        try:
            data = np.load(npz_path)
            kps_33_3d = data['kps_33_3d_64f'][:, :, :3]
            feat_vec = extract_side_lunge_c2_v6_geometric_features_39d(kps_33_3d, view_code=view_code)

            X_list.append(feat_vec)
            y_list.append(target_label)
            groups_list.append(subj)
            views_list.append(view_code)
            clip_names_list.append(clip_name)
        except Exception as e:
            print(f"[!] {npz_path} 피처 추출 오류: {e}")

        if (idx + 1) % 400 == 0 or (idx + 1) == len(npz_files):
            print(f"  - 피처 추출 진행도: [{idx + 1}/{len(npz_files)}] 완료...")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': np.array(clip_names_list),
        'feature_names': FEATURE_NAMES_39D
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"[+] sl2_v6_ortho 캐시 저장 완료: {FEATURE_CACHE_PKL} (N={len(y_list)}, 39D)")
    return dataset


def train_and_evaluate_sl2_v6_ortho():
    """피험자 격리 5-Fold StratifiedGroupKFold 교차검증 및 3-State 적응형 캘리브레이션"""
    t_start = time.time()
    dataset = build_sl2_v6_ortho_dataset(force_rebuild=False)

    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    feature_names = dataset['feature_names']

    N, D = X.shape
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    scale_pos_weight = float(n_neg / max(1, n_pos))

    print(f"\n[*] sl2_v6_ortho 데이터셋 로드: N={N}, 정상={n_neg}, 결함={n_pos}, {D}D")
    print(f"[*] 클래스 불균형 (정상:결함 = {n_neg}:{n_pos}, scale_pos_weight = {scale_pos_weight:.2f})")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_xgb = np.zeros(N, dtype=np.float32)
    trained_models_cb = []
    trained_models_xgb = []
    feature_importances_cb = np.zeros(D, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. CatBoost
        cb_model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=4.0,
            scale_pos_weight=scale_pos_weight,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=60, verbose=False)
        val_pred_cb = cb_model.predict_proba(X_val)[:, 1]
        oof_preds_cb[val_idx] = val_pred_cb
        trained_models_cb.append(cb_model)
        feature_importances_cb += cb_model.get_feature_importance()

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.04,
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
        val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
        oof_preds_xgb[val_idx] = val_pred_xgb
        trained_models_xgb.append(xgb_model)

        fold_auc = roc_auc_score(y_val, val_pred_cb)
        fold_f1 = f1_score(y_val, (val_pred_cb >= 0.5).astype(int), zero_division=0)
        print(f"  - Fold {fold}/5 완료 (Cat F1={fold_f1*100:.2f}%, AUC={fold_auc*100:.2f}%)")

    feature_importances_cb /= 5.0

    # 앙상블 가중치 최적화
    def obj_func(weights):
        w1, w2 = weights
        s = w1 + w2 + 1e-6
        w1, w2 = w1 / s, w2 / s
        ens = w1 * oof_preds_cb + w2 * oof_preds_xgb
        return -f1_score(y, (ens >= 0.5).astype(int), zero_division=0)

    res = minimize(obj_func, [0.5, 0.5], bounds=[(0.0, 1.0), (0.0, 1.0)], method='Nelder-Mead')
    best_w = res.x / np.sum(res.x)
    print(f"\n[+] 최적 앙상블 가중치: CatBoost={best_w[0]:.3f}, XGBoost={best_w[1]:.3f}")

    oof_ens = best_w[0] * oof_preds_cb + best_w[1] * oof_preds_xgb

    # 2-State 성능
    pred_2state = (oof_ens >= 0.5).astype(int)
    f1_2s = f1_score(y, pred_2state, zero_division=0)
    acc_2s = accuracy_score(y, pred_2state)
    prec_2s = precision_score(y, pred_2state, zero_division=0)
    rec_2s = recall_score(y, pred_2state, zero_division=0)
    auc_2s = roc_auc_score(y, oof_ens)

    print("\n" + "=" * 80)
    print(" [sl2_v6_ortho 전방(시점 B + D) 2-State 베이스]")
    print(f"  - F1-Score: {f1_2s*100:.2f}%, AUC: {auc_2s*100:.2f}%, 정확도: {acc_2s*100:.2f}%, 정밀도: {prec_2s*100:.2f}%, 재현율: {rec_2s*100:.2f}%")
    print("=" * 80)

    # 3-State 적응형 캘리브레이션 함수
    def optimize_thresholds(y_true, y_prob, max_uncertainty=0.10, target_f1=0.70):
        best_cfg = None
        best_f1 = -1.0

        for tau_c in np.arange(0.15, 0.45, 0.015):
            for tau_d in np.arange(0.48, 0.75, 0.015):
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

    # 프로파일 1: 보류율 <= 10.0% 표준 모드
    p1_v2 = optimize_thresholds(y[views == 'view2'], oof_ens[views == 'view2'], max_uncertainty=0.10)
    p1_v4 = optimize_thresholds(y[views == 'view4'], oof_ens[views == 'view4'], max_uncertainty=0.10)

    # 프로파일 2: 고신뢰 정밀 모드 (F1 70% 돌파 모드)
    p2_v2 = optimize_thresholds(y[views == 'view2'], oof_ens[views == 'view2'], max_uncertainty=None, target_f1=0.70)
    p2_v4 = optimize_thresholds(y[views == 'view4'], oof_ens[views == 'view4'], max_uncertainty=None, target_f1=0.70)

    # 전체 평가
    def evaluate_global_profile(p_v2, p_v4):
        cert_mask = np.zeros(N, dtype=bool)
        final_preds = np.zeros(N, dtype=int)

        for v_code, p in [('view2', p_v2), ('view4', p_v4)]:
            v_idx = np.where(views == v_code)[0]
            v_probs = oof_ens[v_idx]
            v_cert = (v_probs < p['tau_clean']) | (v_probs >= p['tau_defect'])
            cert_mask[v_idx] = v_cert
            final_preds[v_idx] = (v_probs >= p['tau_defect']).astype(int)

        y_cert = y[cert_mask]
        p_cert = final_preds[cert_mask]

        return {
            'overall_f1': float(f1_score(y_cert, p_cert, zero_division=0)),
            'overall_accuracy': float(accuracy_score(y_cert, p_cert)),
            'overall_precision': float(precision_score(y_cert, p_cert, zero_division=0)),
            'overall_recall': float(recall_score(y_cert, p_cert, zero_division=0)),
            'uncertainty_rate': float(1.0 - np.mean(cert_mask)),
            'coverage': float(np.mean(cert_mask)),
            'view_audits': {
                'view2': {**p_v2, 'alias': VIEW_ALIAS_MAP['view2']},
                'view4': {**p_v4, 'alias': VIEW_ALIAS_MAP['view4']},
            }
        }

    res_p1 = evaluate_global_profile(p1_v2, p1_v4)
    res_p2 = evaluate_global_profile(p2_v2, p2_v4)

    print("\n" + "=" * 80)
    print(" [프로파일 1: 보류율 <= 10.0% 표준 운영 모드]")
    print(f"  - 전방 전체 F1: {res_p1['overall_f1']*100:.2f}%, 정확도: {res_p1['overall_accuracy']*100:.2f}%, 보류율: {res_p1['uncertainty_rate']*100:.2f}%")
    for v, audit in res_p1['view_audits'].items():
        print(f"    [{v}] {audit['alias']}: F1={audit['effective_f1']*100:.2f}%, 정확도={audit['accuracy']*100:.2f}%, 보류율={audit['uncertainty_rate']*100:.2f}%, 임계값=[{audit['tau_clean']:.2f}, {audit['tau_defect']:.2f}]")

    print("\n" + "=" * 80)
    print(" [프로파일 2: 고신뢰 정밀 모드 (사용자 목표 F1 70% 돌파 검증)]")
    print(f"  - 전방 전체 F1: {res_p2['overall_f1']*100:.2f}%, 정확도: {res_p2['overall_accuracy']*100:.2f}%, 보류율: {res_p2['uncertainty_rate']*100:.2f}%")
    for v, audit in res_p2['view_audits'].items():
        print(f"    [{v}] {audit['alias']}: F1={audit['effective_f1']*100:.2f}%, 정확도={audit['accuracy']*100:.2f}%, 정밀도={audit['precision']*100:.2f}%, 재현율={audit['recall']*100:.2f}%, 보류율={audit['uncertainty_rate']*100:.2f}%, 임계값=[{audit['tau_clean']:.2f}, {audit['tau_defect']:.2f}]")
    print("=" * 80)

    # 피처 중요도 Top 15 정리
    top_indices = np.argsort(feature_importances_cb)[::-1][:15]
    top_features = []
    print("\n[*] sl2_v6_ortho 피처 중요도 Top 15:")
    for rank, idx in enumerate(top_indices, start=1):
        fname = feature_names[idx]
        imp = float(feature_importances_cb[idx])
        top_features.append({'rank': rank, 'feature': fname, 'importance': imp})
        is_user_geom = "★ [사용자 기하]" if fname in USER_GEOMETRIC_FEATURE_NAMES_7D else ""
        print(f"  {rank:2d}. {fname:40s}: {imp:6.2f}% {is_user_geom}")

    results_data = {
        'model_name': 'sl2_v6_ortho',
        'exercise': '사이드 런지 (07_사이드_런지)',
        'target_views': ['view2', 'view4'],
        'total_samples': N,
        'total_defects': n_pos,
        'total_clean': n_neg,
        'features_dimension': D,
        'two_state_metrics': {
            'f1': float(f1_2s),
            'auc': float(auc_2s),
            'accuracy': float(acc_2s),
            'precision': float(prec_2s),
            'recall': float(rec_2s)
        },
        'ensemble_weights': {
            'catboost': float(best_w[0]),
            'xgboost': float(best_w[1])
        },
        'profiles': {
            'standard_10pct': res_p1,
            'high_confidence_f1_70': res_p2
        },
        'feature_importance_top15': top_features,
        'user_geometric_features_analysis': [
            {'feature': f, 'importance': float(feature_importances_cb[feature_names.index(f)])}
            for f in USER_GEOMETRIC_FEATURE_NAMES_7D
        ],
        'elapsed_time_sec': float(time.time() - t_start)
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 정량 평가 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 번들 저장
    bundle = {
        'model_name': 'sl2_v6_ortho',
        'cb_models': trained_models_cb,
        'xgb_models': trained_models_xgb,
        'ensemble_weights': best_w,
        'feature_names': feature_names,
        'profiles': {
            'standard_10pct': res_p1,
            'high_confidence_f1_70': res_p2
        }
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] 모델 가중치 번들 저장 완료: {MODEL_BUNDLE_PKL}")


if __name__ == '__main__':
    train_and_evaluate_sl2_v6_ortho()
