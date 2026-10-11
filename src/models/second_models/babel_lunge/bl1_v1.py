# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: bl1_v1.py
Purpose: 바벨 런지 첫 번째 세부 오류인 c1(앞다리 무릎 각도 90도) 모델 구축 및 Two-Tier 3-State 성능 평가
         - 피험자 9명 전수 피험자 격리 교차검증 (StratifiedGroupKFold 5-Fold)
         - 공식 5대 시점(A~E, view1~view5) 전수 대조 감사
         - GBDT 3종(LightGBM, CatBoost, XGBoost) 앙상블 토너먼트
         - 3-State(Clean / Defect / Retake) 의사결정: 최대 보류율(Uncertainty Rate) <= 10.0% 제약 준수
         - 프로덕션 단일 추론 인터페이스 BL1V1Classifier 클래스 제공
"""

import os
import sys
import glob
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

# GBDT 라이브러리 로드
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.lunge_c1_features import (
    extract_lunge_c1_feature_vector,
    LUNGE_C1_FEATURE_NAMES
)

NPZ_CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
LBL_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
FEATURE_CACHE_PKL = r"D:\Data\피처_데이터\bl1_v1_dataset_cache.pkl"
RESULTS_JSON = r"D:\src\models\second_models\babel_lunge\bl1_v1_evaluation_results.json"
BUNDLE_PKL = r"D:\src\models\second_models\babel_lunge\bl1_v1_bundle.pkl"

# 공식 시점 매핑 (AGENTS.md 표준)
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def build_bl1_dataset(force_rebuild: bool = False) -> Dict[str, Any]:
    """전처리된 MediaPipe 33관절 64F 시퀀스 및 라벨로부터 32D 피처 데이터셋 구축 및 캐싱"""
    os.makedirs(os.path.dirname(FEATURE_CACHE_PKL), exist_ok=True)
    if os.path.exists(FEATURE_CACHE_PKL) and not force_rebuild:
        print(f"[*] 피처 캐시 로드 중: {FEATURE_CACHE_PKL}")
        with open(FEATURE_CACHE_PKL, 'rb') as f:
            return pickle.load(f)

    print("[*] 피처 데이터셋 새로 구축 시작...")
    npz_files = sorted(glob.glob(os.path.join(NPZ_CACHE_DIR, "*.npz")))
    assert len(npz_files) > 0, f"NPZ 데이터가 존재하지 않습니다: {NPZ_CACHE_DIR}"

    X_list = []
    y_list = []
    groups_list = []
    views_list = []
    clip_names_list = []

    # 라벨 JSON 캐싱
    label_cache = {}

    for idx, npz_path in enumerate(npz_files):
        fname = os.path.splitext(os.path.basename(npz_path))[0]
        # 형식: {folder_name}_{view}.npz
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        clip_name, view_code = parts[0], parts[1]

        # 라벨 JSON 확인
        if clip_name not in label_cache:
            json_path = os.path.join(LBL_DIR, clip_name + ".json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as fp:
                    d = json.load(fp)
                    tinfo = d.get('type_info', {})
                    conds = tinfo.get('conditions', [])
                    c1_val = None
                    for c in conds:
                        if '앞다리' in c.get('condition', '') and '무릎' in c.get('condition', ''):
                            c1_val = c.get('value')
                            break
                    label_cache[clip_name] = c1_val
            except Exception:
                label_cache[clip_name] = None

        c1_val = label_cache.get(clip_name)
        if c1_val is None:
            continue

        # 결함(False) -> 1, 정상(True) -> 0
        target_label = 1 if c1_val is False else 0

        # 피험자 ID 파싱
        subj = "unknown"
        for part in clip_name.split('_'):
            if '피험자' in part:
                subj = part
                break

        # NPZ 로드 및 피처 추출
        try:
            data = np.load(npz_path)
            kps_33_3d = data['kps_33_3d_64f']
            vis_64f = data['vis_64f'] if 'vis_64f' in data else None
            feat_vec = extract_lunge_c1_feature_vector(kps_33_3d, vis_64f)

            X_list.append(feat_vec)
            y_list.append(target_label)
            groups_list.append(subj)
            views_list.append(view_code)
            clip_names_list.append(clip_name)
        except Exception as e:
            print(f"[!] {npz_path} 피처 추출 오류: {e}")

        if (idx + 1) % 200 == 0 or (idx + 1) == len(npz_files):
            print(f"  - 진행도: [{idx + 1}/{len(npz_files)}] 완료...")

    dataset = {
        'X': np.array(X_list, dtype=np.float32),
        'y': np.array(y_list, dtype=np.int32),
        'groups': np.array(groups_list),
        'views': np.array(views_list),
        'clip_names': np.array(clip_names_list),
        'feature_names': LUNGE_C1_FEATURE_NAMES
    }

    with open(FEATURE_CACHE_PKL, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 피처 데이터셋 캐시 저장 완료: {FEATURE_CACHE_PKL} (총 {len(y_list)}개 샘플)")
    return dataset


def train_and_evaluate_gbdt_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """GBDT 3종(LightGBM, CatBoost, XGBoost) 5-Fold StratifiedGroupKFold 교차검증 및 앙상블"""
    sgkf = StratifiedGroupKFold(n_splits=5)
    n_samples = len(y)

    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_cat = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)

    lgb_models = []
    cat_models = []
    xgb_models = []

    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    scale_weight = float(neg_count) / max(1.0, float(pos_count))
    print(f"[*] 클래스 불균형 비율 (정상:결함 = {neg_count}:{pos_count}, scale_pos_weight = {scale_weight:.2f})")

    fold_idx = 0
    for train_idx, val_idx in sgkf.split(X, y, groups=groups):
        fold_idx += 1
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # 1. LightGBM
        if lgb is not None:
            lgb_train = lgb.Dataset(X_tr, label=y_tr)
            lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
            params_lgb = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
                'scale_pos_weight': scale_weight,
                'verbose': -1,
                'random_state': 42 + fold_idx
            }
            model_lgb = lgb.train(
                params_lgb,
                lgb_train,
                num_boost_round=400,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            oof_lgb[val_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
            lgb_models.append(model_lgb)

        # 2. CatBoost
        if CatBoostClassifier is not None:
            model_cat = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                scale_pos_weight=scale_weight,
                verbose=False,
                random_seed=42 + fold_idx
            )
            model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
            oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]
            cat_models.append(model_cat)

        # 3. XGBoost
        if xgb is not None:
            model_xgb = xgb.XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=scale_weight,
                eval_metric='logloss',
                early_stopping_rounds=50,
                random_state=42 + fold_idx,
                tree_method='hist'
            )
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
            xgb_models.append(model_xgb)

        print(f"  - Fold {fold_idx}/5 학습 완료 (Train: {len(train_idx)}, Val: {len(val_idx)})")

    # 앙상블 연성 투표 (LightGBM 0.35 + CatBoost 0.40 + XGBoost 0.25)
    oof_ensemble = 0.35 * oof_lgb + 0.40 * oof_cat + 0.25 * oof_xgb

    return {
        'oof_lgb': oof_lgb,
        'oof_cat': oof_cat,
        'oof_xgb': oof_xgb,
        'oof_ensemble': oof_ensemble,
        'lgb_models': lgb_models,
        'cat_models': cat_models,
        'xgb_models': xgb_models,
    }


def optimize_3state_thresholds(y_true: np.ndarray, oof_probs: np.ndarray, max_uncertainty: float = 0.10) -> Dict[str, Any]:
    """
    최대 보류율(Uncertainty Rate) <= max_uncertainty 제약 하에서 3-State 확정 판정 F1-Score 극대화
    State 0: Clean (P < tau_clean)
    State 1: Defect (P >= tau_defect)
    State 2: Retake (tau_clean <= P < tau_defect)
    """
    best_f1 = -1.0
    best_tau_clean = 0.40
    best_tau_defect = 0.50
    best_metrics = {}

    # 그리드 탐색
    for t_c in np.linspace(0.25, 0.48, 24):
        for t_d in np.linspace(0.49, 0.75, 27):
            if t_c >= t_d:
                continue

            # 분류 판정
            is_clean = (oof_probs < t_c)
            is_defect = (oof_probs >= t_d)
            is_retake = (~is_clean) & (~is_defect)

            retake_rate = np.mean(is_retake)
            if retake_rate > max_uncertainty:
                continue

            # 확정 판정(Clean or Defect) 샘플만으로 F1 산출
            valid_mask = ~is_retake
            if np.sum(valid_mask) < len(y_true) * 0.8:
                continue

            y_eval_true = y_true[valid_mask]
            y_eval_pred = np.where(oof_probs[valid_mask] >= t_d, 1, 0)

            cur_f1 = f1_score(y_eval_true, y_eval_pred, zero_division=0)
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                best_tau_clean = float(t_c)
                best_tau_defect = float(t_d)
                best_metrics = {
                    '3state_f1': float(cur_f1),
                    '3state_acc': float(accuracy_score(y_eval_true, y_eval_pred)),
                    '3state_prec': float(precision_score(y_eval_true, y_eval_pred, zero_division=0)),
                    '3state_recall': float(recall_score(y_eval_true, y_eval_pred, zero_division=0)),
                    'uncertainty_rate': float(retake_rate),
                    'coverage': float(1.0 - retake_rate),
                    'tau_clean': best_tau_clean,
                    'tau_defect': best_tau_defect
                }

    # 기본 이진 분류 (0.5 기준) 지표
    bin_pred = (oof_probs >= 0.5).astype(int)
    base_metrics = {
        'binary_f1': float(f1_score(y_true, bin_pred, zero_division=0)),
        'binary_acc': float(accuracy_score(y_true, bin_pred)),
        'binary_prec': float(precision_score(y_true, bin_pred, zero_division=0)),
        'binary_recall': float(recall_score(y_true, bin_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, oof_probs))
    }

    return {
        'best_3state': best_metrics,
        'baseline_binary': base_metrics
    }


def evaluate_5_views(y_true: np.ndarray, oof_probs: np.ndarray, views: np.ndarray, tau_clean: float, tau_defect: float) -> Dict[str, Any]:
    """공식 5대 시점(A~E, view1~view5) 전수 대조 감사"""
    view_results = {}
    for v_code, v_name in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        if np.sum(v_mask) == 0:
            continue

        y_v = y_true[v_mask]
        p_v = oof_probs[v_mask]

        is_clean = (p_v < tau_clean)
        is_defect = (p_v >= tau_defect)
        is_retake = (~is_clean) & (~is_defect)
        retake_rate = float(np.mean(is_retake))

        valid_mask = ~is_retake
        if np.sum(valid_mask) > 0:
            f1_3s = float(f1_score(y_v[valid_mask], (p_v[valid_mask] >= tau_defect).astype(int), zero_division=0))
            acc_3s = float(accuracy_score(y_v[valid_mask], (p_v[valid_mask] >= tau_defect).astype(int)))
            prec_3s = float(precision_score(y_v[valid_mask], (p_v[valid_mask] >= tau_defect).astype(int), zero_division=0))
            rec_3s = float(recall_score(y_v[valid_mask], (p_v[valid_mask] >= tau_defect).astype(int), zero_division=0))
        else:
            f1_3s, acc_3s, prec_3s, rec_3s = 0.0, 0.0, 0.0, 0.0

        try:
            auc = float(roc_auc_score(y_v, p_v))
        except Exception:
            auc = 0.5

        view_results[v_code] = {
            'view_name': v_name,
            'samples': int(np.sum(v_mask)),
            '3state_f1': f1_3s,
            'accuracy': acc_3s,
            'precision': prec_3s,
            'recall': rec_3s,
            'roc_auc': auc,
            'uncertainty_rate': retake_rate,
            'coverage': float(1.0 - retake_rate)
        }
    return view_results


class BL1V1Classifier:
    """바벨 런지 c1(앞다리 무릎 각도 90도) 실서비스 추론 인터페이스 클래스"""

    def __init__(self, bundle_path: str = BUNDLE_PKL):
        assert os.path.exists(bundle_path), f"가중치 번들 파일이 없습니다: {bundle_path}"
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.lgb_models = bundle['lgb_models']
        self.cat_models = bundle['cat_models']
        self.xgb_models = bundle['xgb_models']
        self.tau_clean = bundle['tau_clean']
        self.tau_defect = bundle['tau_defect']
        self.feature_names = bundle['feature_names']

    def predict_sequence(self, kps_33_3d_64f: np.ndarray, vis_64f: Optional[np.ndarray] = None, view: str = 'view2') -> Dict[str, Any]:
        """
        단일 64프레임 런지 시퀀스 추론
        - kps_33_3d_64f: (64, 33, 3) 관절 좌표
        - view: 카메라 시점 코드 ('view1' ~ 'view5')
        반환:
          - state: 0 (CLEAN), 1 (DEFECT), 2 (RETAKE)
          - defect_prob: float (0.0 ~ 1.0)
          - margin: float (|p - 0.5|)
          - feedback_msg: str
          - key_angles: Dict[str, float]
        """
        # Tier 0: 센서 온전성 체크
        if vis_64f is not None:
            mean_vis = np.mean(vis_64f[:, [25, 26, 27, 28]])  # 양 무릎, 양 발목
            if mean_vis < 0.40:
                return {
                    'state': 2,
                    'defect_prob': 0.50,
                    'margin': 0.0,
                    'feedback_msg': "[판정보류: RETAKE] 하체가 화면에 가려졌습니다. 전신이 다 보이도록 2걸음 뒤로 물러서세요.",
                    'lead_knee_min_angle': 0.0
                }

        # 32D 피처 추출
        feat_vec = extract_lunge_c1_feature_vector(kps_33_3d_64f, vis_64f).reshape(1, -1)

        # 앙상블 추론
        preds_lgb = [m.predict(feat_vec)[0] for m in self.lgb_models] if self.lgb_models else [0.5]
        preds_cat = [m.predict_proba(feat_vec)[0, 1] for m in self.cat_models] if self.cat_models else [0.5]
        preds_xgb = [m.predict_proba(feat_vec)[0, 1] for m in self.xgb_models] if self.xgb_models else [0.5]

        p_lgb = np.mean(preds_lgb)
        p_cat = np.mean(preds_cat)
        p_xgb = np.mean(preds_xgb)

        # 연성 투표 가중 융합
        p_defect = float(0.35 * p_lgb + 0.40 * p_cat + 0.25 * p_xgb)
        margin = float(abs(p_defect - 0.5))

        # 3-State 의사결정
        if p_defect < self.tau_clean:
            state = 0
            msg = "[정상: CLEAN] 완벽한 런지입니다! 앞다리 무릎 각도 90도를 올바르게 준수하였습니다."
        elif p_defect >= self.tau_defect:
            state = 1
            msg = "[결함: DEFECT] 앞다리 무릎 각도 90도 미준수! 하강 시 앞다리 무릎을 90도까지 더 깊게 굽혀주세요."
        else:
            state = 2
            v_name = VIEW_ALIAS_MAP.get(view, '대각선 45도')
            msg = f"[판정보류: RETAKE] 무릎 각도가 애매하거나 가려졌습니다. {v_name}에서 1회 다시 수행해 주세요."

        # 주요 관절 지표
        lead_min_angle = float(feat_vec[0, 0])
        return {
            'state': state,
            'defect_prob': p_defect,
            'margin': margin,
            'feedback_msg': msg,
            'lead_knee_min_angle': lead_min_angle,
            'tau_clean': self.tau_clean,
            'tau_defect': self.tau_defect
        }


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 런지 bl1_v1 (c1: 앞다리 무릎 90도) 모델 학습 및 평가")
    print("=" * 80)

    t0 = time.time()
    # 1. 데이터셋 빌드 / 캐시 로드
    dataset = build_bl1_dataset(force_rebuild=False)
    X = dataset['X']
    y = dataset['y']
    groups = dataset['groups']
    views = dataset['views']
    feat_names = dataset['feature_names']

    print(f"\n[+] 데이터셋 규격: X={X.shape}, y={y.shape} (정상={np.sum(y==0)}, 결함={np.sum(y==1)})")
    print(f"[+] 고유 피험자 수: {len(np.unique(groups))}명 ({sorted(np.unique(groups).tolist())})")
    print(f"[+] 고유 시점 수: {len(np.unique(views))}개 ({sorted(np.unique(views).tolist())})")

    # 2. GBDT 앙상블 학습 및 OOF 예측
    print("\n[*] GBDT 앙상블 교차검증 (StratifiedGroupKFold 5-Fold) 가동...")
    models_dict = train_and_evaluate_gbdt_models(X, y, groups)
    oof_ensemble = models_dict['oof_ensemble']

    # 3. 3-State 동적 게이팅 임계치 최적화 (보류율 <= 10.0% 제약)
    print("\n[*] 3-State 임계치 최적화 (최대 보류율 <= 10.0% 제약)...")
    opt_res = optimize_3state_thresholds(y, oof_ensemble, max_uncertainty=0.10)
    best_3s = opt_res['best_3state']
    base_bin = opt_res['baseline_binary']

    print(f"[+] 최적화 결과:")
    print(f"  - 베이스라인 이진 F1: {base_bin['binary_f1']*100:.2f}% | 정확도: {base_bin['binary_acc']*100:.2f}% | AUC: {base_bin['roc_auc']*100:.2f}%")
    print(f"  - 3-State 확정 판정 F1: {best_3s['3state_f1']*100:.2f}% | 정확도: {best_3s['3state_acc']*100:.2f}%")
    print(f"  - 판단 보류율(Uncertainty): {best_3s['uncertainty_rate']*100:.2f}% (커버리지: {best_3s['coverage']*100:.2f}%)")
    print(f"  - 최적 임계치: tau_clean={best_3s['tau_clean']:.3f}, tau_defect={best_3s['tau_defect']:.3f}")

    # 4. 공식 5대 시점 전수 대조 감사
    print("\n[*] 공식 5대 시점 전수 성능 감사 (AGENTS.md 표준)...")
    view_audit = evaluate_5_views(y, oof_ensemble, views, best_3s['tau_clean'], best_3s['tau_defect'])
    for v_code, v_info in view_audit.items():
        print(f"  - {v_info['view_name']} ({v_code}): 3-State F1={v_info['3state_f1']*100:.2f}%, ACC={v_info['accuracy']*100:.2f}%, AUC={v_info['roc_auc']*100:.2f}%, 보류율={v_info['uncertainty_rate']*100:.2f}% (샘플: {v_info['samples']})")

    # 5. 결과 JSON 및 번들 저장
    out_results = {
        'metadata': {
            'project_name': 'vidio-health-trainer',
            'exercise': '04_바벨_런지',
            'target_defect': 'c1 (앞다리 무릎 각도 90도)',
            'model_name': 'bl1_v1',
            'total_samples': len(y),
            'subjects': sorted(np.unique(groups).tolist()),
            'elapsed_sec': time.time() - t0
        },
        'baseline_binary': base_bin,
        'best_3state': best_3s,
        'view_audit': view_audit
    }

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 최종 평가 결과 JSON 저장: {RESULTS_JSON}")

    # 프로덕션 배포 가중치 번들 저장
    bundle = {
        'lgb_models': models_dict['lgb_models'],
        'cat_models': models_dict['cat_models'],
        'xgb_models': models_dict['xgb_models'],
        'tau_clean': best_3s['tau_clean'],
        'tau_defect': best_3s['tau_defect'],
        'feature_names': feat_names
    }
    with open(BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 프로덕션 번들 저장: {BUNDLE_PKL}")
    print(f"[+] 전체 모델링 소요 시간: {time.time()-t0:.1f}초")


if __name__ == "__main__":
    main()
