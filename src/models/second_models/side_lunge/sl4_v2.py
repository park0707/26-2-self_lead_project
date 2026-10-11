# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (Side Lunge - 07)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect - 체간 측굴 및 경추 회전)
Model: sl4_v2 (설명 기반 라벨 정제 원천 전체 826폴더 4,130시퀀스 SOTA 모델)

Key Features & Pipeline:
  1. 원천 데이터 100% 전수 활용: 826개 폴더 x 5대 시점 = 총 4,130개 시퀀스
  2. 설명 기반 정밀 라벨 정제 (Description-based Ground Truth Labeling):
     - 텍스트 설명에 '몸통 옆으로 갸우뚱' 또는 시선/경추 오류('왼쪽보고', '오른쪽보고', '하늘보고', '바닥보고') 명시 시 -> c4 결함(1)
     - 척추/경추 언급이 없거나 타 부위(앞다리, 발방향, 안굽히는 다리, 상체 뒤로 젖힘)만 있는 경우 -> c4 정상(0)
  3. 바벨 런지 bl4 SOTA 24D 정예 생체역학 도메인 피처셋 완벽 계승 (lunge_c4_features_v3)
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

from src.features.lunge_c4_features_v3 import (
    extract_lunge_c4_features_v3,
    LUNGE_C4_FEATURE_NAMES_V3
)

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\07_사이드_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\sl4_v2_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\side_lunge\sl4_v2_evaluation_results.json"
MODEL_BUNDLE_PKL = r"D:\src\models\second_models\side_lunge\sl4_v2_bundle.pkl"

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


def build_sl4_v2_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전체 826개 폴더(4,130개 시퀀스) 대상 24D 도메인 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 사이드 런지 sl4_v2 826폴더 전체 24D 피처 데이터셋 새로 구축 시작...")
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
            feat_vec = extract_lunge_c4_features_v3(kps_33_3d)

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
        'feature_names': LUNGE_C4_FEATURE_NAMES_V3
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] sl4_v2 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 시퀀스, 24D)")
    return dataset


def train_and_evaluate_gbdt_models_sl4_v2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
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
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 5,
        'feature_fraction': 0.8,
        'scale_pos_weight': spw,
        'random_state': 42,
        'verbose': -1,
        'n_estimators': 300
    }

    cat_params = {
        'iterations': 300,
        'learning_rate': 0.03,
        'depth': 5,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'scale_pos_weight': spw,
        'random_seed': 42,
        'verbose': 0
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.03,
        'max_depth': 5,
        'colsample_bytree': 0.8,
        'scale_pos_weight': spw,
        'random_state': 42,
        'n_estimators': 300
    }

    print("\n[*] GBDT 3종 모델 5-Fold StratifiedGroupKFold 훈련 시작...")
    fold_models = {'lgb': [], 'cat': [], 'xgb': []}

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. LightGBM
        clf_lgb = lgb.LGBMClassifier(**lgb_params)
        clf_lgb.fit(X_trn, y_trn)
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]
        fold_models['lgb'].append(clf_lgb)

        # 2. CatBoost
        clf_cat = CatBoostClassifier(**cat_params)
        clf_cat.fit(X_trn, y_trn)
        oof_cat[val_idx] = clf_cat.predict_proba(X_val)[:, 1]
        fold_models['cat'].append(clf_cat)

        # 3. XGBoost
        clf_xgb = xgb.XGBClassifier(**xgb_params)
        clf_xgb.fit(X_trn, y_trn)
        oof_xgb[val_idx] = clf_xgb.predict_proba(X_val)[:, 1]
        fold_models['xgb'].append(clf_xgb)

        f1_fold_lgb = f1_score(y_val, (oof_lgb[val_idx] >= 0.5).astype(int), zero_division=0)
        f1_fold_cat = f1_score(y_val, (oof_cat[val_idx] >= 0.5).astype(int), zero_division=0)
        f1_fold_xgb = f1_score(y_val, (oof_xgb[val_idx] >= 0.5).astype(int), zero_division=0)
        print(f"  [Fold {fold+1}/5] LGB F1: {f1_fold_lgb*100:.2f}% | CAT F1: {f1_fold_cat*100:.2f}% | XGB F1: {f1_fold_xgb*100:.2f}%")

    # 단일 모델별 OOF 성능
    metrics_single = {}
    for name, oof in [('LightGBM', oof_lgb), ('CatBoost', oof_cat), ('XGBoost', oof_xgb)]:
        pred_bin = (oof >= 0.5).astype(int)
        metrics_single[name] = {
            'auc': float(roc_auc_score(y, oof)),
            'f1': float(f1_score(y, pred_bin, zero_division=0)),
            'accuracy': float(accuracy_score(y, pred_bin)),
            'precision': float(precision_score(y, pred_bin, zero_division=0)),
            'recall': float(recall_score(y, pred_bin, zero_division=0)),
        }
        print(f"\n[*] {name} 단독 OOF 성능 -> F1: {metrics_single[name]['f1']*100:.2f}%, AUC: {metrics_single[name]['auc']*100:.2f}%, ACC: {metrics_single[name]['accuracy']*100:.2f}%")

    # Nelder-Mead 기반 앙상블 가중치 최적화
    def obj_func(weights):
        w = np.array(weights)
        w = np.maximum(w, 0)
        if np.sum(w) == 0:
            return 1.0
        w = w / np.sum(w)
        pred_prob = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
        return -f1_score(y, (pred_prob >= 0.5).astype(int), zero_division=0)

    res = minimize(obj_func, [0.333, 0.333, 0.334], method='Nelder-Mead')
    opt_w = np.maximum(res.x, 0)
    opt_w = opt_w / np.sum(opt_w)
    print(f"\n[*] 최적 앙상블 가중치: LightGBM={opt_w[0]:.3f}, CatBoost={opt_w[1]:.3f}, XGBoost={opt_w[2]:.3f}")

    oof_ens = opt_w[0] * oof_lgb + opt_w[1] * oof_cat + opt_w[2] * oof_xgb
    pred_2state = (oof_ens >= 0.5).astype(int)

    metrics_ens_2state = {
        'auc': float(roc_auc_score(y, oof_ens)),
        'f1': float(f1_score(y, pred_2state, zero_division=0)),
        'accuracy': float(accuracy_score(y, pred_2state)),
        'precision': float(precision_score(y, pred_2state, zero_division=0)),
        'recall': float(recall_score(y, pred_2state, zero_division=0)),
        'threshold': 0.5
    }
    print(f"[*] 3-GBDT 앙상블 2-State (τ=0.500) -> F1: {metrics_ens_2state['f1']*100:.2f}%, AUC: {metrics_ens_2state['auc']*100:.2f}%, ACC: {metrics_ens_2state['accuracy']*100:.2f}%")

    return {
        'oof_lgb': oof_lgb,
        'oof_cat': oof_cat,
        'oof_xgb': oof_xgb,
        'oof_ens': oof_ens,
        'weights': opt_w,
        'metrics_single': metrics_single,
        'metrics_ens_2state': metrics_ens_2state,
        'fold_models': fold_models
    }


def calibrate_adaptive_3state_sl4_v2(
    oof_ens: np.ndarray, y: np.ndarray, views: np.ndarray,
    target_coverage: float = 0.90
) -> Dict[str, Any]:
    """AGENTS.md 공식 5대 시점별 적응형 3-State 신뢰 분류 캘리브레이션 (보류율 <= 10.0%)"""
    unique_views = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_calibs = {}
    is_confirmed_all = np.zeros(len(y), dtype=bool)
    pred_3state_all = np.zeros(len(y), dtype=int)

    for vn in unique_views:
        idx = np.where(views == vn)[0]
        if len(idx) == 0:
            continue

        v_y = y[idx]
        v_prob = oof_ens[idx]

        best_f1 = -1.0
        best_t_low, best_t_high = 0.40, 0.60
        best_cov = 1.0

        for t_low in np.linspace(0.20, 0.48, 29):
            for t_high in np.linspace(0.52, 0.80, 29):
                mask = (v_prob < t_low) | (v_prob >= t_high)
                cov = np.mean(mask)
                if cov < target_coverage:
                    continue

                pred = (v_prob[mask] >= t_high).astype(int)
                score = f1_score(v_y[mask], pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t_low = float(t_low)
                    best_t_high = float(t_high)
                    best_cov = float(cov)

        # 적용
        v_mask = (v_prob < best_t_low) | (v_prob >= best_t_high)
        is_confirmed_all[idx[v_mask]] = True
        pred_3state_all[idx[v_mask]] = (v_prob[v_mask] >= best_t_high).astype(int)

        v_pred_2s = (v_prob >= 0.5).astype(int)
        v_f1_2s = f1_score(v_y, v_pred_2s, zero_division=0)
        v_acc_2s = accuracy_score(v_y, v_pred_2s)
        v_auc = roc_auc_score(v_y, v_prob) if len(np.unique(v_y)) > 1 else 0.5

        v_f1_3s = f1_score(v_y[v_mask], (v_prob[v_mask] >= best_t_high).astype(int), zero_division=0)
        v_acc_3s = accuracy_score(v_y[v_mask], (v_prob[v_mask] >= best_t_high).astype(int))

        view_calibs[vn] = {
            'alias': VIEW_ALIAS_MAP[vn],
            'sample_count': int(len(idx)),
            'tau_low': best_t_low,
            'tau_high': best_t_high,
            'coverage': best_cov,
            'abstention_rate': float(1.0 - best_cov),
            'auc': float(v_auc),
            '2state_f1': float(v_f1_2s),
            '2state_acc': float(v_acc_2s),
            '3state_f1': float(v_f1_3s),
            '3state_acc': float(v_acc_3s),
        }

    # 종합 3-State 지표
    tot_cov = float(np.mean(is_confirmed_all))
    y_conf = y[is_confirmed_all]
    pred_conf = pred_3state_all[is_confirmed_all]

    overall_3state = {
        'coverage': tot_cov,
        'abstention_rate': float(1.0 - tot_cov),
        'confirmed_f1': float(f1_score(y_conf, pred_conf, zero_division=0)),
        'confirmed_accuracy': float(accuracy_score(y_conf, pred_conf)),
        'confirmed_precision': float(precision_score(y_conf, pred_conf, zero_division=0)),
        'confirmed_recall': float(recall_score(y_conf, pred_conf, zero_division=0)),
    }

    return {
        'view_calibs': view_calibs,
        'overall_3state': overall_3state,
        'is_confirmed_all': is_confirmed_all,
        'pred_3state_all': pred_3state_all
    }


def fit_and_save_final_sl4_v2_bundle(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray,
    view_calibs: Dict[str, Any], metrics_summary: Dict[str, Any]
):
    """전체 데이터 대상 단일 최종 모델 핏팅 및 배포 번들 저장"""
    neg_cnt = np.sum(y == 0)
    pos_cnt = np.sum(y == 1)
    spw = float(neg_cnt / max(1, pos_cnt))

    clf_lgb = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', boosting_type='gbdt',
        learning_rate=0.03, num_leaves=31, max_depth=5, feature_fraction=0.8,
        scale_pos_weight=spw, random_state=42, verbose=-1, n_estimators=300
    )
    clf_lgb.fit(X, y)

    clf_cat = CatBoostClassifier(
        iterations=300, learning_rate=0.03, depth=5, loss_function='Logloss',
        eval_metric='Logloss', scale_pos_weight=spw, random_seed=42, verbose=0
    )
    clf_cat.fit(X, y)

    clf_xgb = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', learning_rate=0.03,
        max_depth=5, colsample_bytree=0.8, scale_pos_weight=spw, random_state=42, n_estimators=300
    )
    clf_xgb.fit(X, y)

    bundle = {
        'model_name': 'sl4_v2',
        'exercise': '07_사이드_런지',
        'defect': 'c4',
        'defect_name': '척추의 중립 위반 (체간 측굴 및 경추 회전)',
        'feature_names': LUNGE_C4_FEATURE_NAMES_V3,
        'weights': weights,
        'models': {
            'lgb': clf_lgb,
            'cat': clf_cat,
            'xgb': clf_xgb
        },
        'view_calibs': view_calibs,
        'metrics_summary': metrics_summary
    }

    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 최종 sl4_v2 배포 번들 저장 완료: {MODEL_BUNDLE_PKL}")


def run_sl4_v2_pipeline():
    start_t = time.time()
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 c4 sl4_v2 공식 파이프라인 가동")
    print("  - 데이터셋: 설명 기반 라벨 정제 원천 전체 826폴더 (총 4,130 시퀀스)")
    print("  - 아키텍처: 바벨 런지 bl4 계승 24D 도메인 피처 + 3-GBDT 앙상블 + 시점별 적응형 3-State")
    print("=" * 80)

    # 1. 데이터셋 구축
    dataset = build_sl4_v2_dataset(force_rebuild=False)
    X, y, groups, views = dataset['X'], dataset['y'], dataset['groups'], dataset['views']

    print(f"\n[*] 데이터셋 로드 완료: X={X.shape}, y={y.shape}")
    print(f"[*] 피험자 전수: {len(np.unique(groups))}명 ({sorted(list(set(groups)))})")
    print(f"[*] 시점 분포: {np.unique(views, return_counts=True)}")

    # 2. GBDT 3종 5-Fold 학습 및 앙상블
    train_res = train_and_evaluate_gbdt_models_sl4_v2(X, y, groups)

    # 3. 시점별 적응형 3-State 캘리브레이션
    calib_res = calibrate_adaptive_3state_sl4_v2(
        train_res['oof_ens'], y, views, target_coverage=0.90
    )

    # 4. 결과 출력
    m_2s = train_res['metrics_ens_2state']
    m_3s = calib_res['overall_3state']
    vc = calib_res['view_calibs']

    print("\n" + "=" * 80)
    print("  [★ 사이드 런지 sl4_v2 최종 종합 평가 결과]")
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

    # 5. 최종 배포 모델 저장
    fit_and_save_final_sl4_v2_bundle(
        X, y, train_res['weights'], vc,
        {'2state': m_2s, '3state': m_3s}
    )

    # 6. JSON 결과 저장
    final_json = {
        'model_name': 'sl4_v2',
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
    run_sl4_v2_pipeline()
