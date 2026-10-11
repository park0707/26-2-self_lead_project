# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Model: sl1_hybrid_1 (MediaPipe Pseudo-3D + RTMPose 2D 상호보완 융합 최적화 모델)
Baseline: sl1_rtmp_1 & sl1_v11 계승 (5-Fold StratifiedGroupKFold, 3-GBDT 앙상블, 온도 스케일링, 3-State 선택적 분류)

규정 준수 사항 (RULE[d:\AGENTS.md]):
  1. 공식 프로젝트명: vidio-health-trainer (NovelFlow 명칭 절대 금지)
  2. 피험자 9명 전원(1~9번) 5-Fold StratifiedGroupKFold 완전 격리 교차검증 (Zero-Leakage Invariant)
  3. 100% 순수 비전 추출 포인트 기반 피처 투입 (AI-Hub JSON 무누수 0.00%)
  4. 공식 5대 시점 규격 준수:
     - 시점 A: 우측 후방 (45도) [view5]
     - 시점 B: 우측 전방 (45도) [view2]
     - 시점 C: 정면 (0도) [view3]
     - 시점 D: 좌측 전방 (45도) [view4]
     - 시점 E: 좌측 후방 (45도) [view1]
  5. 3-State 선택적 분류: 불확실성 보류율(Uncertainty Rate) <= 10.0% 엄격 준수
"""

import os
import sys
import json
import pickle
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings('ignore')

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, log_loss, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DATA_CACHE_NPZ = r"D:\Data\피처_데이터\sl1_hybrid_features.npz"
RESULTS_JSON = os.path.join(CURRENT_DIR, "sl1_hybrid_1_evaluation_results.json")
MODEL_BUNDLE_PKL = os.path.join(CURRENT_DIR, "sl1_hybrid_1_bundle.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면 (0도)',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)',
}
VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (probs == bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc = np.mean(labels[in_bin])
            conf = np.mean(probs[in_bin])
            ece += np.abs(conf - acc) * prop_in_bin
    return float(ece)


def optimize_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """NLL 최소화 1D 온도 스케일링 최적화"""
    def nll_func(T):
        p_t = 1.0 / (1.0 + np.exp(-z / T[0]))
        p_t = np.clip(p_t, 1e-7, 1.0 - 1e-7)
        return log_loss(y, p_t)

    res = minimize(nll_func, [1.0], bounds=[(0.05, 5.0)], method='L-BFGS-B')
    return float(res.x[0])


def search_view_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    views: np.ndarray,
    max_uncertainty: float = 0.10
) -> Dict[str, Dict[str, float]]:
    """5대 시점별 보류율 <= 10.0% 제약 하에서 F1 최대화 3-State 임계값 그리드 탐색"""
    view_th = {}
    for vn in VIEW_ORDER:
        v_mask = (views == vn)
        yt_v = y_true[v_mask]
        yp_v = y_prob[v_mask]

        best_f1 = -1.0
        best_th = {'tau_clean': 0.40, 'tau_defect': 0.60}

        for tc in np.linspace(0.20, 0.48, 20):
            for td in np.linspace(0.52, 0.80, 20):
                if td <= tc:
                    continue
                unc = (yp_v >= tc) & (yp_v < td)
                u_rate = np.mean(unc)
                if u_rate > max_uncertainty:
                    continue

                dec = ~unc
                if np.sum(dec) == 0:
                    continue
                f1_val = f1_score(yt_v[dec], (yp_v[dec] >= td).astype(int), zero_division=0)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_th = {'tau_clean': round(float(tc), 3), 'tau_defect': round(float(td), 3)}

        view_th[vn] = best_th
    return view_th


class SL1Hybrid1Classifier:
    """
    사이드 런지 c1 (앞다리 무릎 각도 90도 오류) MediaPipe + RTMPose 하이브리드 공식 추론 엔진
    """
    __module__ = "src.models.second_models.side_lunge.sl1_hybrid_1"
    def __init__(
        self,
        models_lgb: List[Any],
        models_cat: List[Any],
        models_xgb: List[Any],
        temperature: float,
        view_thresholds: Dict[str, Dict[str, float]],
        feature_names: List[str]
    ):
        self.models_lgb = models_lgb
        self.models_cat = models_cat
        self.models_xgb = models_xgb
        self.temperature = float(temperature)
        self.view_thresholds = view_thresholds
        self.feature_names = feature_names
        self._dict = {
            'model_name': 'sl1_final (sl1_hybrid_1)',
            'base_model': 'MediaPipe Pseudo-3D + RTMPose 2D Full Complementary Concat',
            'features_dimension': len(feature_names),
            'models_lgb': models_lgb,
            'models_cat': models_cat,
            'models_xgb': models_xgb,
            'temperature': self.temperature,
            'view_thresholds': view_thresholds,
            'feature_names': feature_names,
            'VIEW_ALIAS_MAP': VIEW_ALIAS_MAP,
            'default_thresholds': {'tau_clean': 0.35, 'tau_defect': 0.55}
        }

    def __getitem__(self, item):
        return self._dict[item]

    def __contains__(self, item):
        return item in self._dict

    def get(self, item, default=None):
        return self._dict.get(item, default)

    def keys(self):
        return self._dict.keys()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)

        p_lgb = np.mean([m.predict_proba(X)[:, 1] for m in self.models_lgb], axis=0)
        p_cat = np.mean([m.predict_proba(X)[:, 1] for m in self.models_cat], axis=0)
        p_xgb = np.mean([m.predict_proba(X)[:, 1] for m in self.models_xgb], axis=0)

        p_ens = (p_lgb + p_cat + p_xgb) / 3.0
        p_ens = np.clip(p_ens, 1e-6, 1.0 - 1e-6)

        logits = np.log(p_ens / (1.0 - p_ens))
        logits_scaled = logits / self.temperature
        p_cal = 1.0 / (1.0 + np.exp(-logits_scaled))
        return p_cal

    def predict_3state(self, X: np.ndarray, view_code: str) -> Dict[str, Any]:
        """
        단일 시퀀스 3-State 판정:
        - 0: CLEAN (정상: 앞다리 무릎 90도 양호)
        - 1: DEFECT (결함: 앞다리 무릎 90도 오류)
        - 2: RETAKE (보류: 불확실성 판정 보류)
        """
        p_cal = float(self.predict_proba(X)[0])
        th = self.view_thresholds.get(view_code, {'tau_clean': 0.38, 'tau_defect': 0.55})
        tc = th['tau_clean']
        td = th['tau_defect']

        if p_cal < tc:
            state = 0
            label = "CLEAN (정상)"
            detail = "앞다리 무릎이 90도 기준에 알맞게 안정적으로 굴곡되었습니다."
        elif p_cal >= td:
            state = 1
            label = "DEFECT (결함)"
            detail = "앞다리 무릎 각도가 90도 기준에 미달하거나 과도하게 굴곡되었습니다. 체중을 충분히 싣고 무릎 높이를 맞추세요."
        else:
            state = 2
            label = "RETAKE (보류)"
            detail = "시점 또는 가림으로 인해 무릎 굴곡 심도 판정이 모호합니다. 신체 전신이 보이도록 재촬영해 주세요."

        return {
            'defect_code': 'c1',
            'defect_name': '앞다리 무릎 각도 90도 오류',
            'calibrated_prob': round(p_cal, 4),
            'state': state,
            'state_label': label,
            'view_code': view_code,
            'view_alias': VIEW_ALIAS_MAP.get(view_code, view_code),
            'tau_clean': tc,
            'tau_defect': td,
            'feedback': detail
        }


def run_sl1_hybrid_training_and_evaluation():
    print("=" * 80)
    print("  [vidio-health-trainer] sl1_hybrid_1: 사이드 런지 c1 MediaPipe + RTMPose")
    print("  상호보완 융합 3-GBDT 앙상블, 온도 스케일링, 3-State 공식 최적화 파이프라인")
    print("=" * 80)

    # 1. 데이터 로드
    assert os.path.exists(DATA_CACHE_NPZ), f"캐시 파일 부재: {DATA_CACHE_NPZ}"
    data = np.load(DATA_CACHE_NPZ)
    X_55d = data['X_hybrid_55d'].astype(np.float32)
    X_110d = data['X_hybrid_110d'].astype(np.float32)
    y = data['y'].astype(int)
    groups = data['groups']
    views = data['views']
    names_55d = list(data['feature_names_55d'])
    names_110d = list(data['feature_names_110d'])

    n_samples = len(y)
    print(f"[*] 총 시퀀스 수: {n_samples}개")
    print(f"[*] 피험자 분포: {len(np.unique(groups))}명, 5대 시점 분포: {len(np.unique(views))}개 시점")

    # 2. 5-Fold StratifiedGroupKFold 교차검증 비교 (55D vs 110D)
    sgkf = StratifiedGroupKFold(n_splits=5)

    experiments = {
        'Hybrid-55D (RTMP 2D + MP 3D Replaced)': (X_55d, names_55d),
        'Hybrid-110D (Full Complementary Concat)': (X_110d, names_110d)
    }

    results = {}
    best_exp_name = None
    best_f1_val = -1.0
    best_bundle_info = None

    for exp_name, (X_curr, feat_names) in experiments.items():
        print("\n" + "-" * 70)
        print(f"[*] 교차검증 실행: {exp_name} (피처 차원: {X_curr.shape[1]}D)")
        print("-" * 70)

        oof_probs_lgb = np.zeros(n_samples)
        oof_probs_cat = np.zeros(n_samples)
        oof_probs_xgb = np.zeros(n_samples)

        all_lgb = []
        all_cat = []
        all_xgb = []

        start_time = time.time()
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_curr, y, groups), 1):
            X_tr, y_tr = X_curr[train_idx], y[train_idx]
            X_va, y_va = X_curr[val_idx], y[val_idx]

            # Model 1: LightGBM
            m_lgb = lgb.LGBMClassifier(
                n_estimators=120, max_depth=4, num_leaves=15,
                learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                random_state=42 + fold, verbose=-1, n_jobs=-1
            )
            m_lgb.fit(X_tr, y_tr)
            all_lgb.append(m_lgb)
            oof_probs_lgb[val_idx] = m_lgb.predict_proba(X_va)[:, 1]

            # Model 2: CatBoost
            m_cat = CatBoostClassifier(
                iterations=150, depth=4, learning_rate=0.04,
                random_seed=42 + fold, verbose=0, thread_count=2
            )
            m_cat.fit(X_tr, y_tr)
            all_cat.append(m_cat)
            oof_probs_cat[val_idx] = m_cat.predict_proba(X_va)[:, 1]

            # Model 3: XGBoost
            m_xgb = xgb.XGBClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42 + fold, eval_metric='logloss', n_jobs=-1
            )
            m_xgb.fit(X_tr, y_tr)
            all_xgb.append(m_xgb)
            oof_probs_xgb[val_idx] = m_xgb.predict_proba(X_va)[:, 1]

        elapsed = time.time() - start_time
        print(f"[*] 5-Fold 앙상블 학습 완료 ({elapsed:.1f}초)")

        oof_probs_raw = (oof_probs_lgb + oof_probs_cat + oof_probs_xgb) / 3.0
        oof_probs_clipped = np.clip(oof_probs_raw, 1e-6, 1.0 - 1e-6)
        oof_logits = np.log(oof_probs_clipped / (1.0 - oof_probs_clipped))

        # Level 1: NLL 최소화 온도 스케일링
        T_opt = optimize_temperature(oof_logits, y)
        oof_probs_cal = 1.0 / (1.0 + np.exp(-oof_logits / T_opt))
        ece_pre = compute_ece(oof_probs_raw, y)
        ece_post = compute_ece(oof_probs_cal, y)
        brier_pre = float(np.mean((oof_probs_raw - y) ** 2))
        brier_post = float(np.mean((oof_probs_cal - y) ** 2))

        # 2-State (강제 결정) 성능
        auc_score = float(roc_auc_score(y, oof_probs_cal)) * 100
        y_pred_2s = (oof_probs_cal >= 0.50).astype(int)
        f1_2s = float(f1_score(y, y_pred_2s)) * 100
        acc_2s = float(accuracy_score(y, y_pred_2s)) * 100
        prec_2s = float(precision_score(y, y_pred_2s, zero_division=0)) * 100
        rec_2s = float(recall_score(y, y_pred_2s, zero_division=0)) * 100

        # Level 2: 5대 시점별 3-State 적응형 신뢰 임계값 최적화
        view_th = search_view_thresholds(y, oof_probs_cal, views, max_uncertainty=0.10)

        # 3-State 판정 산출
        decisions_3s = np.zeros(n_samples, dtype=int)
        retake_mask = np.zeros(n_samples, dtype=bool)

        for vn in VIEW_ORDER:
            v_idx = (views == vn)
            tc = view_th[vn]['tau_clean']
            td = view_th[vn]['tau_defect']
            u_v = (oof_probs_cal[v_idx] >= tc) & (oof_probs_cal[v_idx] < td)
            retake_mask[v_idx] = u_v
            d_v = np.zeros(np.sum(v_idx), dtype=int)
            d_v[oof_probs_cal[v_idx] >= td] = 1
            decisions_3s[v_idx] = d_v

        valid_mask = ~retake_mask
        retake_rate = float(np.mean(retake_mask)) * 100
        coverage_rate = 100.0 - retake_rate
        f1_3s = float(f1_score(y[valid_mask], decisions_3s[valid_mask])) * 100
        acc_3s = float(accuracy_score(y[valid_mask], decisions_3s[valid_mask])) * 100
        prec_3s = float(precision_score(y[valid_mask], decisions_3s[valid_mask], zero_division=0)) * 100
        rec_3s = float(recall_score(y[valid_mask], decisions_3s[valid_mask], zero_division=0)) * 100

        # 시점별 세부 성능
        view_metrics = {}
        for vn in VIEW_ORDER:
            v_idx = (views == vn)
            y_v = y[v_idx]
            p_v = oof_probs_cal[v_idx]
            dec_v = decisions_3s[v_idx]
            ret_v = retake_mask[v_idx]
            val_v = ~ret_v

            view_metrics[vn] = {
                'alias': VIEW_ALIAS_MAP[vn],
                'auc': round(float(roc_auc_score(y_v, p_v)) * 100, 2),
                'f1_2s': round(float(f1_score(y_v, (p_v >= 0.50).astype(int))) * 100, 2),
                'f1_3s': round(float(f1_score(y_v[val_v], dec_v[val_v])) * 100, 2),
                'retake': round(float(np.mean(ret_v)) * 100, 2),
                'coverage': round(float(100.0 - np.mean(ret_v) * 100), 2),
                'tau_clean': view_th[vn]['tau_clean'],
                'tau_defect': view_th[vn]['tau_defect']
            }

        exp_result = {
            'auc': round(auc_score, 2),
            'f1_2s': round(f1_2s, 2),
            'prec_2s': round(prec_2s, 2),
            'rec_2s': round(rec_2s, 2),
            'acc_2s': round(acc_2s, 2),
            'f1_3s': round(f1_3s, 2),
            'prec_3s': round(prec_3s, 2),
            'rec_3s': round(rec_3s, 2),
            'acc_3s': round(acc_3s, 2),
            'retake_rate': round(retake_rate, 2),
            'coverage': round(coverage_rate, 2),
            'opt_temperature': round(T_opt, 4),
            'ece_pre': round(ece_pre * 100, 2),
            'ece_post': round(ece_post * 100, 2),
            'brier_pre': round(brier_pre, 4),
            'brier_post': round(brier_post, 4),
            'view_thresholds': view_th,
            'view_metrics': view_metrics
        }
        results[exp_name] = exp_result

        print(f"[{exp_name}]")
        print(f"  - ROC-AUC: {auc_score:.2f}%")
        print(f"  - 2-State F1: {f1_2s:.2f}%, Acc: {acc_2s:.2f}%")
        print(f"  - 3-State F1: {f1_3s:.2f}%, Retake: {retake_rate:.2f}%, Coverage: {coverage_rate:.2f}%")
        print(f"  - ECE: {ece_pre*100:.2f}% -> {ece_post*100:.2f}%, T: {T_opt:.4f}")

        if f1_3s > best_f1_val:
            best_f1_val = f1_3s
            best_exp_name = exp_name
            best_bundle_info = {
                'models_lgb': all_lgb,
                'models_cat': all_cat,
                'models_xgb': all_xgb,
                'temperature': T_opt,
                'view_thresholds': view_th,
                'feature_names': feat_names,
                'X_data': X_curr
            }

    print("\n" + "=" * 80)
    print(f"  [★ 챔피언 선정] {best_exp_name} (3-State F1: {best_f1_val:.2f}%)")
    print("=" * 80)

    # 3. 모델 번들 pkl 저장
    classifier = SL1Hybrid1Classifier(
        models_lgb=best_bundle_info['models_lgb'],
        models_cat=best_bundle_info['models_cat'],
        models_xgb=best_bundle_info['models_xgb'],
        temperature=best_bundle_info['temperature'],
        view_thresholds=best_bundle_info['view_thresholds'],
        feature_names=best_bundle_info['feature_names']
    )
    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"[+] 공식 모델 번들 저장 완료: {MODEL_BUNDLE_PKL}")

    # 4. 종합 평가 결과 json 저장
    final_output = {
        'project': 'vidio-health-trainer',
        'exercise': '07_사이드_런지',
        'model_name': 'sl1_hybrid_1',
        'defect_name': '앞다리 무릎 각도 90도 오류',
        'champion_architecture': best_exp_name,
        'total_sequences': n_samples,
        'subjects_count': 9,
        'views_count': 5,
        'baselines': {
            'mediapipe_sl1_v11': {
                'auc': 85.47,
                'f1_2s': 65.78,
                'f1_3s': 71.93,
                'retake': 17.89
            },
            'rtmp_sl1_rtmp_1': {
                'auc': 84.50,
                'f1_2s': 63.82,
                'f1_3s': 69.58,
                'retake': 9.49
            }
        },
        'experiments': results,
        'champion_metrics': results[best_exp_name]
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"[+] 공식 평가 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 5. 단일 시퀀스 추론 테스트 (Inference Invariant Verification)
    test_feat = best_bundle_info['X_data'][0]
    test_view = str(views[0])
    res_infer = classifier.predict_3state(test_feat, test_view)
    print(f"[*] 단일 시퀀스 추론 테스트 결과: {res_infer}")


if __name__ == '__main__':
    run_sl1_hybrid_training_and_evaluation()
