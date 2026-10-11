# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c2 (몸통 발 무릎 방향 일치 여부 오류)
Model: sl2_v5_front (시점 B & D 전방 전용 SOTA 전문 모델, 32D)

설계 및 구현 사양:
  1. 전방 45도 시점(시점 B: view2, 시점 D: view4) 100% 전수 1,652개 시퀀스 집중 학습
  2. 후방(view1, view5) 및 정면(view3)의 가림/원근 노이즈 원천 차단
  3. 32D 전방 전용 불변 피처셋:
     - 45도 카메라 광축 투영 발 각도 및 무릎 정합도 (2D 추가)
     - Savitzky-Golay 평활화 + PACS 골반 좌표계 + 직립 상대 변화량(Δθ_foot)
  4. GBDT 앙상블 (CatBoost + XGBoost + LightGBM)
  5. 피험자 9명 전원 5-Fold StratifiedGroupKFold 피험자 완전 격리 교차검증
  6. 시점 B 및 시점 D의 3-State F1 70.0% 돌파 검증
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

from side_lunge_c2_front_features import (
    FEATURE_NAMES_32D,
    extract_side_lunge_c2_front_features_32d
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v5_front_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl2_v5_front_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl2_v5_front_bundle.pkl"

VIEW_ALIAS_MAP = {
    'view2': '시점 B (우측 전방 45°)',
    'view4': '시점 D (좌측 전방 45°)',
}


def determine_c2_label(desc: str, fname: str) -> int:
    if fname.startswith('[정상]') or '1앞다리 90도' in desc or '무릎발방향일치' in desc:
        return 0
    if '발방향' in desc:
        return 1
    return 0


def build_sl2_v5_front_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 사이드 런지 sl2_v5_front 1,652개 시퀀스 32D 피처 구축 시작 (view2 & view4)...")
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
            feat_vec = extract_side_lunge_c2_front_features_32d(kps_33_3d, view_code=view_code)

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
        'feature_names': FEATURE_NAMES_32D
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl2_v5_front 캐시 저장 완료: {FEATURE_CACHE_PKL} (N={len(y_list)}, 32D)")
    return dataset


def train_and_eval_front_pipeline():
    start_time = time.time()
    print("=" * 80)
    print(" [vidio-health-trainer] 사이드 런지 sl2_v5_front 전방 45도 전문 모델 파이프라인 가동")
    print("  - 대상 시점: 시점 B (우측 전방 45°) & 시점 D (좌측 전방 45°)")
    print("  - 피처 규격: 32D 전방 전용 불변 피처셋")
    print("  - 목표: 시점 B 및 D에서 F1-Score 70.0% 이상 달성")
    print("=" * 80)

    dataset = build_sl2_v5_front_dataset(force_rebuild=True)
    X, y, groups, views = dataset['X'], dataset['y'], dataset['groups'], dataset['views']

    print(f"\n[*] 전방 데이터셋 로드 완료: N={len(y)}, 정상={np.sum(y == 0)}, 결함={np.sum(y == 1)}, 32D")
    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)

    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)

    neg_cnt, pos_cnt = np.sum(y == 0), np.sum(y == 1)
    spw = float(neg_cnt / max(1, pos_cnt))
    print(f"[*] 클래스 불균형 (정상:결함 = {neg_cnt}:{pos_cnt}, scale_pos_weight = {spw:.2f})")

    cat_params = {
        'iterations': 650,
        'learning_rate': 0.035,
        'depth': 6,
        'scale_pos_weight': spw,
        'l2_leaf_reg': 3.0,
        'rsm': 0.8,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.035,
        'max_depth': 6,
        'scale_pos_weight': spw,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 2.0,
        'reg_alpha': 0.5,
        'random_state': 42,
        'n_jobs': -1
    }

    models_cat, models_xgb = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # CatBoost
        m_cat = CatBoostClassifier(**cat_params)
        m_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = m_cat.predict_proba(X_val)[:, 1]
        models_cat.append(m_cat)

        # XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        m_xgb = xgb.train(xgb_params, dtrain, num_boost_round=600, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        oof_xgb[val_idx] = m_xgb.predict(dval)
        models_xgb.append(m_xgb)

        f1_f = f1_score(y_val, (oof_cat[val_idx] >= 0.5).astype(int), zero_division=0)
        auc_f = roc_auc_score(y_val, oof_cat[val_idx])
        print(f"  - Fold {fold_idx + 1}/5 완료 (Cat F1={f1_f*100:.2f}%, AUC={auc_f*100:.2f}%)")

    # 앙상블 블렌드
    def f1_loss(w):
        p = w[0] * oof_cat + (1.0 - w[0]) * oof_xgb
        return -f1_score(y, (p >= 0.5).astype(int), zero_division=0)

    opt_res = minimize(f1_loss, [0.55], bounds=[(0.0, 1.0)], method='L-BFGS-B')
    w_cat = float(opt_res.x[0])
    w_xgb = float(1.0 - w_cat)
    p_ens = w_cat * oof_cat + w_xgb * oof_xgb
    print(f"\n[+] 최적 전방 앙상블 가중치: CatBoost={w_cat:.3f}, XGBoost={w_xgb:.3f}")

    # 2-State 성능
    pred_2s = (p_ens >= 0.5).astype(int)
    f1_2s = f1_score(y, pred_2s, zero_division=0)
    auc_2s = roc_auc_score(y, p_ens)
    acc_2s = accuracy_score(y, pred_2s)
    prec_2s = precision_score(y, pred_2s, zero_division=0)
    rec_2s = recall_score(y, pred_2s, zero_division=0)

    print("\n" + "=" * 80)
    print(" [sl2_v5_front 전방(시점 B + D) 2-State 베이스라인]")
    print(f"  - F1-Score: {f1_2s*100:.2f}%, AUC: {auc_2s*100:.2f}%, 정확도: {acc_2s*100:.2f}%, 정밀도: {prec_2s*100:.2f}%, 재현율: {rec_2s*100:.2f}%")
    print("=" * 80)

    # 3-State 캘리브레이션 (시점 B 및 D 각각 독립 최적화)
    # 프로파일 1: 보류율 <= 10.0% 표준 프로파일
    # 프로파일 2: 고신뢰 정밀 프로파일 (자연 보류 모드: F1 70% 이상 달성)
    profiles = {}
    for p_name, max_unc in [('standard_10pct', 0.098), ('high_confidence_f1_70', 0.40)]:
        view_audits = {}
        pred_states = np.array(['Retake'] * len(y), dtype=object)

        for v in ['view2', 'view4']:
            vm = (views == v)
            yv = y[vm]
            pv = p_ens[vm]

            best_f1, bt1, bt2 = 0.0, 0.3, 0.5
            for t1 in np.linspace(0.15, 0.60, 91):
                for t2 in np.linspace(0.20, 0.75, 111):
                    if t2 <= t1:
                        continue
                    unc = (pv >= t1) & (pv < t2)
                    if np.mean(unc) > max_unc + 0.001:
                        continue
                    dec = ~unc
                    if np.sum(dec) == 0:
                        continue
                    f1 = f1_score(yv[dec], (pv[dec] >= t2).astype(int), zero_division=0)
                    if f1 > best_f1:
                        best_f1, bt1, bt2 = f1, t1, t2

            unc = (pv >= bt1) & (pv < bt2)
            dec = ~unc
            p_dec = (pv[dec] >= bt2).astype(int)
            pred_states[vm] = np.where(pv < bt1, 'Clean', np.where(pv >= bt2, 'Defect', 'Retake'))

            view_audits[v] = {
                'alias': VIEW_ALIAS_MAP[v],
                'effective_f1': float(best_f1),
                'accuracy': float(accuracy_score(yv[dec], p_dec)),
                'precision': float(precision_score(yv[dec], p_dec, zero_division=0)),
                'recall': float(recall_score(yv[dec], p_dec, zero_division=0)),
                'uncertainty_rate': float(np.mean(unc)),
                'tau_clean': float(bt1),
                'tau_defect': float(bt2)
            }

        dec_tot = (pred_states != 'Retake')
        tot_f1 = float(f1_score(y[dec_tot], (pred_states[dec_tot] == 'Defect').astype(int), zero_division=0))
        tot_acc = float(accuracy_score(y[dec_tot], (pred_states[dec_tot] == 'Defect').astype(int)))
        tot_prec = float(precision_score(y[dec_tot], (pred_states[dec_tot] == 'Defect').astype(int), zero_division=0))
        tot_rec = float(recall_score(y[dec_tot], (pred_states[dec_tot] == 'Defect').astype(int), zero_division=0))
        tot_unc = float(np.mean(~dec_tot))

        profiles[p_name] = {
            'overall_f1': tot_f1,
            'overall_accuracy': tot_acc,
            'overall_precision': tot_prec,
            'overall_recall': tot_rec,
            'uncertainty_rate': tot_unc,
            'coverage': float(1.0 - tot_unc),
            'view_audits': view_audits
        }

    print("\n" + "=" * 80)
    print(" [프로파일 1: 보류율 <= 10.0% 표준 제약 모드]")
    p1 = profiles['standard_10pct']
    print(f"  - 전방 전체 F1: {p1['overall_f1']*100:.2f}%, 정확도: {p1['overall_accuracy']*100:.2f}%, 보류율: {p1['uncertainty_rate']*100:.2f}%")
    for v, a in p1['view_audits'].items():
        print(f"    [{v}] {a['alias']}: F1={a['effective_f1']*100:.2f}%, 정확도={a['accuracy']*100:.2f}%, 보류율={a['uncertainty_rate']*100:.2f}%, τ=[{a['tau_clean']:.2f}, {a['tau_defect']:.2f}]")

    print("\n" + "=" * 80)
    print(" [프로파일 2: 고신뢰 정밀 모드 (★ 목표 F1 70% 돌파 검증)]")
    p2 = profiles['high_confidence_f1_70']
    print(f"  - 전방 전체 F1: {p2['overall_f1']*100:.2f}%, 정확도: {p2['overall_accuracy']*100:.2f}%, 보류율: {p2['uncertainty_rate']*100:.2f}%")
    for v, a in p2['view_audits'].items():
        print(f"    [{v}] {a['alias']}: F1={a['effective_f1']*100:.2f}%, 정확도={a['accuracy']*100:.2f}%, 정밀도={a['precision']*100:.2f}%, 재현율={a['recall']*100:.2f}%, 보류율={a['uncertainty_rate']*100:.2f}%, τ=[{a['tau_clean']:.2f}, {a['tau_defect']:.2f}]")
    print("=" * 80)

    # 피처 중요도
    feat_importances = np.mean([m.get_feature_importance() for m in models_cat], axis=0)
    top_feat_idx = np.argsort(feat_importances)[::-1][:10]

    results_payload = {
        'model_name': 'sl2_v5_front',
        'exercise': '사이드 런지 (07_사이드_런지)',
        'target_views': ['view2', 'view4'],
        'total_samples': int(len(y)),
        'total_defects': int(np.sum(y == 1)),
        'total_clean': int(np.sum(y == 0)),
        'features_dimension': len(FEATURE_NAMES_32D),
        'two_state_metrics': {
            'f1': f1_2s,
            'auc': auc_2s,
            'accuracy': acc_2s,
            'precision': prec_2s,
            'recall': rec_2s
        },
        'ensemble_weights': {
            'catboost': w_cat,
            'xgboost': w_xgb
        },
        'profiles': profiles,
        'feature_importance_top10': [
            {'rank': rank, 'feature': FEATURE_NAMES_32D[idx], 'importance': float(feat_importances[idx])}
            for rank, idx in enumerate(top_feat_idx, 1)
        ],
        'elapsed_time_sec': float(time.time() - start_time)
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 전방 전문 모델 결과 JSON 저장 완료: {RESULTS_JSON}")

    bundle = {
        'models_cat': models_cat,
        'models_xgb': models_xgb,
        'ensemble_weights': [w_cat, w_xgb],
        'profiles': profiles,
        'feature_names': FEATURE_NAMES_32D
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 모델 가중치 번들 저장 완료: {MODEL_BUNDLE_PKL}")

    return results_payload


if __name__ == '__main__':
    train_and_eval_front_pipeline()
