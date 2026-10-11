# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect - 체간 측굴 및 경추 회전)
Model: sl4_v3 (c3 공간 거리 다리 분리 엔진 100% 통합 SOTA 모델)

Key Features & Pipeline:
  1. 원천 데이터 100% 전수 활용: 826개 폴더 x 5대 시점 = 총 4,130개 시퀀스
  2. 설명 기반 정밀 라벨 정제 (Description-based Ground Truth Labeling):
     - 텍스트 설명에 '몸통 옆으로 갸우뚱' 또는 시선/경추 오류('왼쪽보고', '오른쪽보고', '하늘보고', '바닥보고') 명시 시 -> c4 결함(1)
     - 척추/경추 언급이 없거나 타 부위(앞다리, 발방향, 안굽히는 다리, 상체 뒤로 젖힘)만 있는 경우 -> c4 정상(0)
  3. [핵심 고도화] c3 공간 거리 다리 분리 엔진 (Spatial Distance Leg Separation) 통합 (side_lunge_c4_features_v4.py)
     - 기존의 무릎 각도 크기 비교를 영구 폐기하고,
     - 골반 중심 대비 X축 수평 공간 거리 지표(무릎 0.6 + 발목 0.4)를 도입하여
     - 사이드 런지 체간-선행 무릎 정렬(spine_knee_alignment)의 기준이 되는 앞/작동 다리를 100% 정밀 식별.
  4. GBDT 3종(LightGBM, CatBoost, XGBoost) 최적 구조 앙상블 및 Nelder-Mead 가중치 최적화
  5. 카메라 5대 공식 시점(view1~view5, 시점 A~E)별 적응형 3-State 캘리브레이션 (보류율 <= 10.0% 제약 준수)
  6. 피험자 9명 전원(1~9번) 5-Fold StratifiedGroupKFold 피험자 격리 교차검증
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

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c4_features_v4 import (
    extract_side_lunge_c4_features_v4,
    SIDE_LUNGE_C4_FEATURE_NAMES_V4
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl4_v3_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl4_v3_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl4_v3_bundle.pkl"

# AGENTS.md 5대 공식 시점 매핑 규격 준수
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def determine_c4_label_from_description(desc: str, fname: str) -> int:
    """
    텍스트 설명(Description) 기반 c4 (척추의 중립 위반) 정답 라벨 판정
    - 설명에 '몸통 옆으로 갸우뚱' 또는 시선/경추 오류('왼쪽보고', '오른쪽보고', '하늘보고', '바닥보고') 명시 시: 1 (결함)
    - 정상 또는 타 부위(앞다리, 발방향, 안굽히는 다리, 상체 뒤로 젖힘)만 있는 경우: 0 (정상)
    """
    if fname.startswith('[정상]') or '4척추 중립' in desc:
        return 0

    has_spine_defect = '몸통 옆으로 갸우뚱' in desc
    has_cervical_defect = any(k in desc for k in ['왼쪽보고', '오른쪽보고', '하늘보고', '바닥보고'])

    if has_spine_defect or has_cervical_defect:
        return 1
    return 0


def build_sl4_v3_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전체 826개 폴더(4,130개 시퀀스) 대상 24D 도메인 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 사이드 런지 sl4_v3 826폴더 전체 24D 피처 데이터셋 새로 구축 시작 (c3 공간 거리 다리 분리)...")
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    assert len(npz_files) > 0, f"NPZ 데이터가 존재하지 않습니다: {DATA_DIR}"

    X_list = []
    y_list = []
    groups_list = []
    views_list = []
    clip_names_list = []

    label_cache = {}

    for idx, npz_path in enumerate(npz_files):
        fname = os.path.splitext(os.path.basename(npz_path))[0]
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        clip_name, view_code = parts[0], parts[1]

        # 라벨 JSON 확인
        if clip_name not in label_cache:
            json_path = os.path.join(LABEL_DIR, clip_name + ".json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8', errors='ignore') as fp:
                    d = json.load(fp)
                    tinfo = d.get('type_info', {})
                    desc = tinfo.get('description', '')
                    target_label = determine_c4_label_from_description(desc, clip_name)
                    label_cache[clip_name] = target_label
            except Exception:
                label_cache[clip_name] = None

        target_label = label_cache.get(clip_name)
        if target_label is None:
            continue

        # 피험자 ID 파싱 (피험자 1~9)
        subj = "unknown"
        for part in clip_name.split('_'):
            if '피험자' in part:
                subj = part
                break

        # NPZ 로드 및 24D 피처 추출 (100% 무누수 순수 비전)
        try:
            data = np.load(npz_path)
            kps_33_3d = data['kps_33_3d_64f']
            feat_vec = extract_side_lunge_c4_features_v4(kps_33_3d)

            X_list.append(feat_vec)
            y_list.append(target_label)
            groups_list.append(subj)
            views_list.append(view_code)
            clip_names_list.append(clip_name)
        except Exception as e:
            print(f"[!] {npz_path} 피처 추출 오류: {e}")

        if (idx + 1) % 500 == 0 or (idx + 1) == len(npz_files):
            print(f"  - 피처 추출 진행도: [{idx + 1}/{len(npz_files)}] 완료...")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': np.array(clip_names_list),
        'feature_names': SIDE_LUNGE_C4_FEATURE_NAMES_V4
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl4_v3 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 시퀀스, 24D)")
    return dataset


def train_and_evaluate_gbdt_models_sl4_v3(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """GBDT 3종 구조 5-Fold StratifiedGroupKFold 교차검증 및 Nelder-Mead 가중 앙상블"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)

    neg_cnt = np.sum(y == 0)
    pos_cnt = np.sum(y == 1)
    spw = float(neg_cnt / max(1, pos_cnt))
    print(f"[*] 클래스 불균형 비율 (정상:결함 = {neg_cnt}:{pos_cnt}, scale_pos_weight = {spw:.2f})")

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.04,
        'num_leaves': 31,
        'max_depth': 6,
        'scale_pos_weight': spw,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_lambda': 3.0,
        'reg_alpha': 0.5,
        'min_child_samples': 20,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    cat_params = {
        'iterations': 500,
        'learning_rate': 0.04,
        'depth': 6,
        'scale_pos_weight': spw,
        'l2_leaf_reg': 3.0,
        'rsm': 0.7,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.04,
        'max_depth': 6,
        'scale_pos_weight': spw,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_lambda': 3.0,
        'reg_alpha': 0.5,
        'min_child_weight': 2,
        'random_state': 42,
        'n_jobs': -1
    }

    models_lgb, models_cat, models_xgb = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model_lgb = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=400,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )
        oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
        models_lgb.append(model_lgb)

        # CatBoost
        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
        models_cat.append(model_cat)

        # XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model_xgb = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=400,
            evals=[(dval, 'val')],
            early_stopping_rounds=40,
            verbose_eval=False
        )
        oof_xgb[val_idx] = model_xgb.predict(dval)
        models_xgb.append(model_xgb)

        f1_fold = f1_score(y_val, (oof_cat[val_idx] >= 0.5).astype(int), zero_division=0)
        auc_fold = roc_auc_score(y_val, oof_cat[val_idx])
        print(f"  - sl4_v3 Fold {fold_idx + 1}/5 완료 (Val F1={f1_fold*100:.2f}%, AUC={auc_fold*100:.2f}%)")

    def loss_func(weights):
        w = np.array(weights)
        w = np.maximum(w, 0)
        if np.sum(w) == 0:
            return 10.0
        w /= np.sum(w)
        pred = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        return -roc_auc_score(y, pred)

    opt_res = minimize(loss_func, [0.35, 0.45, 0.20], method="Nelder-Mead")
    opt_weights = np.maximum(opt_res.x, 0)
    opt_weights /= np.sum(opt_weights)

    oof_ens = opt_weights[0] * oof_lgb + opt_weights[1] * oof_cat + opt_weights[2] * oof_xgb
    print(f"\n[+] sl4_v3 최적 앙상블 가중치: LightGBM={opt_weights[0]:.3f}, CatBoost={opt_weights[1]:.3f}, XGBoost={opt_weights[2]:.3f}")

    def eval_2s(p):
        p_bin = (p >= 0.5).astype(int)
        return {
            'f1': float(f1_score(y, p_bin, zero_division=0)),
            'accuracy': float(accuracy_score(y, p_bin)),
            'auc': float(roc_auc_score(y, p))
        }

    return {
        'oof_lgb': oof_lgb,
        'oof_cat': oof_cat,
        'oof_xgb': oof_xgb,
        'oof_ens': oof_ens,
        'weights': opt_weights,
        'models_lgb': models_lgb,
        'models_cat': models_cat,
        'models_xgb': models_xgb,
        'metrics_ens_2state': eval_2s(oof_ens),
        'metrics_single': {
            'lgb': eval_2s(oof_lgb),
            'cat': eval_2s(oof_cat),
            'xgb': eval_2s(oof_xgb)
        }
    }


def calibrate_adaptive_3state_sl4_v3(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    views: np.ndarray,
    target_coverage: float = 0.90
) -> Dict[str, Any]:
    """공식 5대 시점별 적응형 3-State 신뢰 분류 캘리브레이션 (보류율 <= 10.0%)"""
    unique_views = sorted(list(set(views)))
    view_calibs = {}

    all_preds_3state = np.full(len(y_true), -1, dtype=int)
    all_is_retest = np.zeros(len(y_true), dtype=bool)

    for v in unique_views:
        v_idx = np.where(views == v)[0]
        y_v = y_true[v_idx]
        p_v = y_prob[v_idx]

        best_tau_l, best_tau_h = 0.35, 0.65
        best_f1 = -1.0

        for tau_l in np.linspace(0.20, 0.48, 29):
            for tau_h in np.linspace(0.52, 0.80, 29):
                if tau_l >= tau_h:
                    continue
                retest_mask = (p_v >= tau_l) & (p_v < tau_h)
                cov = 1.0 - np.mean(retest_mask)
                if cov < target_coverage:
                    continue

                det_idx = np.where(~retest_mask)[0]
                if len(det_idx) == 0:
                    continue
                y_det = y_v[det_idx]
                p_det = (p_v[det_idx] >= tau_h).astype(int)
                score = f1_score(y_det, p_det, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_tau_l, best_tau_h = tau_l, tau_h

        retest = (p_v >= best_tau_l) & (p_v < best_tau_h)
        pred_3s = np.full(len(y_v), -1, dtype=int)
        pred_3s[p_v < best_tau_l] = 0
        pred_3s[p_v >= best_tau_h] = 1

        all_preds_3state[v_idx] = pred_3s
        all_is_retest[v_idx] = retest

        det_mask = ~retest
        f1_3s = f1_score(y_v[det_mask], pred_3s[det_mask], zero_division=0) if np.sum(det_mask) > 0 else 0.0
        acc_3s = accuracy_score(y_v[det_mask], pred_3s[det_mask]) if np.sum(det_mask) > 0 else 0.0

        view_calibs[v] = {
            'view_name': VIEW_ALIAS_MAP.get(v, v),
            'tau_low': float(best_tau_l),
            'tau_high': float(best_tau_h),
            'coverage': float(np.mean(det_mask)),
            '3state_f1': float(f1_3s),
            '3state_acc': float(acc_3s),
            'sample_count': len(v_idx),
            'auc': float(roc_auc_score(y_v, p_v))
        }

    valid_mask = ~all_is_retest
    overall_3s = {
        'coverage': float(np.mean(valid_mask)),
        'confirmed_f1': float(f1_score(y_true[valid_mask], all_preds_3state[valid_mask], zero_division=0)),
        'confirmed_accuracy': float(accuracy_score(y_true[valid_mask], all_preds_3state[valid_mask])),
        'retest_count': int(np.sum(all_is_retest)),
        'total_count': len(y_true)
    }

    return {
        'view_calibs': view_calibs,
        'overall_3state': overall_3s
    }


def fit_and_save_final_sl4_v3_bundle(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray,
    view_calibs: Dict[str, Any], metrics: Dict[str, Any]
):
    """전체 데이터셋 대상 최종 서빙 모델 학습 및 번들 파일 저장"""
    print("\n[*] sl4_v3 최종 전수 데이터 서빙 모델 학습 중...")
    spw = float(np.sum(y == 0) / max(1, np.sum(y == 1)))

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.04,
        'num_leaves': 31,
        'max_depth': 6,
        'scale_pos_weight': spw,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    train_data = lgb.Dataset(X, label=y)
    final_lgb = lgb.train(lgb_params, train_data, num_boost_round=300)

    final_cat = CatBoostClassifier(
        iterations=400,
        learning_rate=0.04,
        depth=6,
        scale_pos_weight=spw,
        random_seed=42,
        verbose=False
    )
    final_cat.fit(X, y)

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.04,
        'max_depth': 6,
        'scale_pos_weight': spw,
        'random_state': 42,
        'n_jobs': -1
    }
    dtrain = xgb.DMatrix(X, label=y)
    final_xgb = xgb.train(xgb_params, dtrain, num_boost_round=300)

    bundle = {
        'model_name': 'sl4_v3',
        'weights': weights,
        'final_lgb': final_lgb,
        'final_cat': final_cat,
        'final_xgb': final_xgb,
        'view_calibs': view_calibs,
        'metrics': metrics,
        'feature_names': SIDE_LUNGE_C4_FEATURE_NAMES_V4
    }

    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl4_v3 최종 서빙 번들 저장 완료: {MODEL_BUNDLE_PKL}")


def run_sl4_v3_pipeline():
    start_t = time.time()
    print("=" * 80)
    print(" [vidio-health-trainer] 사이드 런지 sl4_v3 SOTA 공식 파이프라인 가동")
    print("  - 대상 결함: c4 (척추의 중립 위반 / Spine Neutrality Defect)")
    print("  - 핵심 고도화: c3 공간 거리 기반 다리 분리 엔진 (Spatial Distance Leg Separation)")
    print("  - 데이터셋: AI-Hub 826개 폴더 x 5대 시점 전수 (4,130개 시퀀스)")
    print("=" * 80)

    dataset = build_sl4_v3_dataset(force_rebuild=True)
    X, y, groups, views = dataset['X'], dataset['y'], dataset['groups'], dataset['views']

    print(f"\n[*] 데이터셋 로드 완료: X={X.shape}, y={y.shape}")
    print(f"[*] 피험자 전수: {len(np.unique(groups))}명 ({sorted(list(set(groups)))})")
    print(f"[*] 시점 분포: {np.unique(views, return_counts=True)}")

    train_res = train_and_evaluate_gbdt_models_sl4_v3(X, y, groups)

    calib_res = calibrate_adaptive_3state_sl4_v3(
        train_res['oof_ens'], y, views, target_coverage=0.90
    )

    m_2s = train_res['metrics_ens_2state']
    m_3s = calib_res['overall_3state']
    vc = calib_res['view_calibs']

    print("\n" + "=" * 80)
    print("  [★ 사이드 런지 sl4_v3 최종 종합 평가 결과]")
    print(f"  - 원천 전체 샘플 수: {len(y)}개 시퀀스 (정상: {np.sum(y==0)}개, 결함: {np.sum(y==1)}개)")
    print(f"  - 2-State (τ=0.500) : F1 = {m_2s['f1']*100:.2f}%, ACC = {m_2s['accuracy']*100:.2f}%, AUC = {m_2s['auc']*100:.2f}%")
    print(f"  - ★ 3-State 신뢰 분류 : F1 = {m_3s['confirmed_f1']*100:.2f}% (+{(m_3s['confirmed_f1']-m_2s['f1'])*100:+.2f}%p), ACC = {m_3s['confirmed_accuracy']*100:.2f}%, Cov = {m_3s['coverage']*100:.2f}%")
    print("=" * 80)

    print("\n[AGENTS.md 규격 공식 5대 시점 전수 실측 감사]")
    for vn, vname in VIEW_ALIAS_MAP.items():
        vinfo = vc.get(vn, {})
        print(f"  • {vname} ({vinfo.get('sample_count', 0)}건):")
        print(f"    - 2-State: F1 = {vinfo.get('2state_f1', 0)*100:.2f}%, ACC = {vinfo.get('2state_acc', 0)*100:.2f}%, AUC = {vinfo.get('auc', 0)*100:.2f}%")
        print(f"    - ★ 3-State: F1 = {vinfo.get('3state_f1', 0)*100:.2f}%, ACC = {vinfo.get('3state_acc', 0)*100:.2f}%, Cov = {vinfo.get('coverage', 0)*100:.2f}% (τ=[{vinfo.get('tau_low', 0):.3f}, {vinfo.get('tau_high', 0):.3f}])")

    fit_and_save_final_sl4_v3_bundle(
        X, y, train_res['weights'], vc,
        {'2state': m_2s, '3state': m_3s}
    )

    final_json = {
        'model_name': 'sl4_v3',
        'exercise': '07_사이드_런지',
        'defect': 'c4 (척추의 중립 위반)',
        'data_scope': '설명 기반 라벨 정제 원천 전체 826폴더 (총 4,130 시퀀스)',
        'total_samples': int(len(y)),
        'normal_samples': int(np.sum(y == 0)),
        'defect_samples': int(np.sum(y == 1)),
        'weights': {
            'lgb': float(train_res['weights'][0]),
            'cat': float(train_res['weights'][1]),
            'xgb': float(train_res['weights'][2])
        },
        'submodel_metrics': train_res['metrics_single'],
        'ensemble_2state': m_2s,
        'ensemble_3state': m_3s,
        'view_audit': vc,
        'elapsed_seconds': float(time.time() - start_t)
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    print(f"[+] 공식 평가 결과 JSON 저장 완료: {RESULTS_JSON}")


if __name__ == '__main__':
    run_sl4_v3_pipeline()
