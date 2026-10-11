# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c1 (앞다리 무릎 각도 90도 오류)
Model: sl1_rtmp_1 (RTMPose-m 전용 최적화 모델)
Baseline: sl1_v11 계승 (55D 도메인 피처, GBDT 3종 앙상블, 온도 스케일링, 3-State 선택적 분류)

규정 준수 사항 (RULE[d:\AGENTS.md]):
  1. 공식 프로젝트명: vidio-health-trainer (NovelFlow 명칭 절대 금지)
  2. 피험자 9명 전원(1~9번) 5-Fold StratifiedGroupKFold 완전 격리 교차검증 (Zero Leakage)
  3. 100% 순수 비전 RTMPose 관절 기반 피처 투입 (AI-Hub JSON 피처 배제)
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

from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_feature_vector_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)

DATA_CACHE_NPZ = r"D:\Data\피처_데이터\side_lunge_rtmp_features.npz"
RESULTS_JSON = os.path.join(CURRENT_DIR, "sl1_rtmp_1_evaluation_results.json")
MODEL_BUNDLE_PKL = os.path.join(CURRENT_DIR, "sl1_rtmp_1_bundle.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면 (0도)',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)',
}
VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Guo et al. ICML 2017 Expected Calibration Error (ECE)"""
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

        for tc in np.linspace(0.20, 0.48, 15):
            for td in np.linspace(0.52, 0.80, 15):
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


class SL1RTMP1Classifier:
    """
    사이드 런지 c1 (앞다리 무릎 각도 90도 오류) 공식 RTMPose 추론 엔진
    """
    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or MODEL_BUNDLE_PKL
        self.is_loaded = False
        self.models_lgb = []
        self.models_cat = []
        self.models_xgb = []
        self.weights = [0.15, 0.65, 0.20]
        self.temperature = 1.0
        self.view_thresholds = {}
        self.default_thresholds = {'tau_clean': 0.35, 'tau_defect': 0.65}
        self.feature_names = SIDE_LUNGE_C1_FEATURE_NAMES_V11

        if os.path.exists(self.bundle_path):
            self._load_bundle(self.bundle_path)

    def _load_bundle(self, path: str):
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        self.models_lgb = bundle.get('models_lgb', [])
        self.models_cat = bundle.get('models_cat', [])
        self.models_xgb = bundle.get('models_xgb', [])
        self.weights = bundle.get('weights', [0.15, 0.65, 0.20])
        self.temperature = bundle.get('temperature', 1.0)
        self.view_thresholds = bundle.get('view_thresholds', {})
        self.default_thresholds = bundle.get('default_thresholds', {'tau_clean': 0.35, 'tau_defect': 0.65})
        self.feature_names = bundle.get('feature_names', SIDE_LUNGE_C1_FEATURE_NAMES_V11)
        self.is_loaded = True

    def predict_proba(self, feat_vec: np.ndarray, calibrate: bool = True) -> float:
        """단일 55D 피처 벡터 확률 추론"""
        if feat_vec.ndim == 1:
            feat_vec = feat_vec.reshape(1, -1)

        p_lgb = np.mean([m.predict_proba(feat_vec)[:, 1] for m in self.models_lgb], axis=0)[0]
        p_cat = np.mean([m.predict_proba(feat_vec)[:, 1] for m in self.models_cat], axis=0)[0]
        p_xgb = np.mean([m.predict_proba(feat_vec)[:, 1] for m in self.models_xgb], axis=0)[0]

        raw_p = self.weights[0] * p_lgb + self.weights[1] * p_cat + self.weights[2] * p_xgb
        if not calibrate:
            return float(raw_p)

        clipped_p = np.clip(raw_p, 1e-6, 1.0 - 1e-6)
        logit = np.log(clipped_p / (1.0 - clipped_p))
        cal_p = 1.0 / (1.0 + np.exp(-logit / max(1e-4, self.temperature)))
        return float(cal_p)

    def predict_3state(self, feat_vec: np.ndarray, view_code: Optional[str] = None) -> Dict[str, Any]:
        """3-State 선택적 분류: 0(정상), 1(결함), 2(보류)"""
        prob = self.predict_proba(feat_vec, calibrate=True)
        th = self.view_thresholds.get(view_code, self.default_thresholds)
        tc = th.get('tau_clean', 0.35)
        td = th.get('tau_defect', 0.65)

        if prob < tc:
            state = 0  # Clean (Normal)
            status_str = "정상"
        elif prob >= td:
            state = 1  # Defect
            status_str = "결함 (무릎 각도 90도 미달 또는 초과)"
        else:
            state = 2  # Retake / Uncertain
            status_str = "판정 보류 (자세 모호 또는 가림)"

        return {
            'state': state,
            'status': status_str,
            'calibrated_prob': prob,
            'tau_clean': tc,
            'tau_defect': td,
            'view_code': view_code
        }


def run_sl1_rtmp_1_training_and_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("  vidio-health-trainer | 사이드 런지 c1 sl1_rtmp_1 (RTMPose 전용 최적화 모델)")
    print("  - 가중치: LightGBM 0.15 : CatBoost 0.65 : XGBoost 0.20 (CatBoost 강화)")
    print("  - Level 1: NLL 최소화 온도 스케일링 (ECE 최적화)")
    print("  - Level 2: 5대 시점별 적응형 3-State (보류율 <= 10.0%)")
    print("  - 데이터: 4,130개 시퀀스 (피험자 9명 전수, 5대 시점 전수)")
    print("=" * 80)

    assert os.path.exists(DATA_CACHE_NPZ), f"캐시 파일 누락: {DATA_CACHE_NPZ}"
    ds = np.load(DATA_CACHE_NPZ, allow_pickle=True)
    X = ds['X_c1']
    y = ds['y_c1']
    subjects = ds['subjects']
    views = ds['views']

    print(f"[*] 피처 로드 완료: X={X.shape} (55D), y={y.shape}, 피험자={len(set(subjects))}명")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    weights = [0.15, 0.65, 0.20]

    oof_ens = np.zeros(len(y), dtype=np.float32)
    models_lgb = []
    models_cat = []
    models_xgb = []

    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, subjects)):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va = X[val_idx]

        m_lgb = lgb.LGBMClassifier(n_estimators=160, max_depth=6, learning_rate=0.05, num_leaves=31, n_jobs=1, verbose=-1, random_state=42 + fold)
        m_lgb.fit(X_tr, y_tr)
        p_lgb = m_lgb.predict_proba(X_va)[:, 1]

        m_cat = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, thread_count=1, verbose=0, random_seed=42 + fold)
        m_cat.fit(X_tr, y_tr)
        p_cat = m_cat.predict_proba(X_va)[:, 1]

        m_xgb = xgb.XGBClassifier(n_estimators=160, max_depth=6, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=42 + fold)
        m_xgb.fit(X_tr, y_tr)
        p_xgb = m_xgb.predict_proba(X_va)[:, 1]

        oof_ens[val_idx] = weights[0] * p_lgb + weights[1] * p_cat + weights[2] * p_xgb

        models_lgb.append(m_lgb)
        models_cat.append(m_cat)
        models_xgb.append(m_xgb)

    train_time = time.time() - t0
    print(f"[*] 5-Fold 학습 완료 ({train_time:.1f}초)")

    # 1. 2-State 기본 지표
    y_pred_2s = (oof_ens >= 0.50).astype(int)
    auc_2s = float(roc_auc_score(y, oof_ens))
    f1_2s = float(f1_score(y, y_pred_2s, zero_division=0))
    acc_2s = float(accuracy_score(y, y_pred_2s))
    prec_2s = float(precision_score(y, y_pred_2s, zero_division=0))
    rec_2s = float(recall_score(y, y_pred_2s, zero_division=0))
    raw_ece = compute_ece(oof_ens, y)
    raw_brier = float(np.mean((oof_ens - y) ** 2))

    # 2. Level 1: 온도 스케일링
    p_clip = np.clip(oof_ens, 1e-6, 1.0 - 1e-6)
    logits = np.log(p_clip / (1.0 - p_clip))
    opt_T = optimize_temperature(logits, y)
    oof_cal = 1.0 / (1.0 + np.exp(-logits / opt_T))
    cal_ece = compute_ece(oof_cal, y)
    cal_brier = float(np.mean((oof_cal - y) ** 2))

    # 3. Level 2: 5대 시점별 3-State 적응형 탐색 (보류율 <= 10.0%)
    view_th_10 = search_view_thresholds(y, oof_cal, views, max_uncertainty=0.10)
    view_audits_10 = {}
    total_dec_10 = []
    total_yp_10 = []
    total_yt_10 = []

    for vn in VIEW_ORDER:
        v_mask = (views == vn)
        yt_v = y[v_mask]
        yp_v = oof_cal[v_mask]
        tc = view_th_10[vn]['tau_clean']
        td = view_th_10[vn]['tau_defect']

        unc_v = (yp_v >= tc) & (yp_v < td)
        dec_v = ~unc_v
        n_dec = int(np.sum(dec_v))
        u_rate = float(np.mean(unc_v))
        cov = float(1.0 - u_rate)

        if n_dec > 0:
            eff_f1 = float(f1_score(yt_v[dec_v], (yp_v[dec_v] >= td).astype(int), zero_division=0))
            eff_acc = float(accuracy_score(yt_v[dec_v], (yp_v[dec_v] >= td).astype(int)))
            eff_prec = float(precision_score(yt_v[dec_v], (yp_v[dec_v] >= td).astype(int), zero_division=0))
            eff_rec = float(recall_score(yt_v[dec_v], (yp_v[dec_v] >= td).astype(int), zero_division=0))
        else:
            eff_f1, eff_acc, eff_prec, eff_rec = 0.0, 0.0, 0.0, 0.0

        view_audits_10[vn] = {
            'alias': VIEW_ALIAS_MAP[vn],
            'tau_clean': tc,
            'tau_defect': td,
            'effective_f1': eff_f1,
            'accuracy': eff_acc,
            'precision': eff_prec,
            'recall': eff_rec,
            'uncertainty_rate': u_rate,
            'coverage': cov,
            'samples': int(len(yt_v)),
            'decided_samples': n_dec
        }

        total_dec_10.extend(dec_v)
        total_yp_10.extend((yp_v[dec_v] >= td).astype(int))
        total_yt_10.extend(yt_v[dec_v])

    tot_dec_arr = np.array(total_dec_10)
    f1_3s_adaptive = float(f1_score(total_yt_10, total_yp_10, zero_division=0))
    acc_3s_adaptive = float(accuracy_score(total_yt_10, total_yp_10))
    prec_3s_adaptive = float(precision_score(total_yt_10, total_yp_10, zero_division=0))
    rec_3s_adaptive = float(recall_score(total_yt_10, total_yp_10, zero_division=0))
    cov_3s_adaptive = float(np.mean(tot_dec_arr))
    retake_3s_adaptive = float(1.0 - cov_3s_adaptive)

    # 4. 고정 정책 [0.35, 0.65] 3-State 지표
    fixed_unc = (oof_cal >= 0.35) & (oof_cal < 0.65)
    fixed_dec = ~fixed_unc
    f1_3s_fixed = float(f1_score(y[fixed_dec], (oof_cal[fixed_dec] >= 0.65).astype(int), zero_division=0))
    acc_3s_fixed = float(accuracy_score(y[fixed_dec], (oof_cal[fixed_dec] >= 0.65).astype(int)))
    retake_fixed = float(np.mean(fixed_unc))
    cov_fixed = float(1.0 - retake_fixed)

    print("\n" + "=" * 80)
    print("  [sl1_rtmp_1 최적화 성능 평가 결과]")
    print(f"  - 2-State ROC-AUC : {auc_2s:.4f} | F1: {f1_2s*100:.2f}% | Acc: {acc_2s*100:.2f}%")
    print(f"  - 온도 최적화 (T)  : {opt_T:.4f} (ECE: {raw_ece*100:.2f}% -> {cal_ece*100:.2f}%)")
    print(f"  - [적응형 3-State] : F1 {f1_3s_adaptive*100:.2f}% | Acc {acc_3s_adaptive*100:.2f}% | 보류율 {retake_3s_adaptive*100:.2f}% (커버리지 {cov_3s_adaptive*100:.2f}%)")
    print(f"  - [고정 [0.35,0.65]] : F1 {f1_3s_fixed*100:.2f}% | Acc {acc_3s_fixed*100:.2f}% | 보류율 {retake_fixed*100:.2f}%")
    print("=" * 80)

    results = {
        'model_name': 'sl1_rtmp_1',
        'exercise': '07_사이드_런지',
        'defect': 'c1 (앞다리 무릎 각도 90도 오류)',
        'features_dimension': 55,
        'total_samples': len(y),
        'two_state_metrics': {
            'roc_auc': auc_2s,
            'f1': f1_2s,
            'accuracy': acc_2s,
            'precision': prec_2s,
            'recall': rec_2s,
            'raw_ece': raw_ece,
            'raw_brier': raw_brier
        },
        'calibration': {
            'temperature': opt_T,
            'cal_ece': cal_ece,
            'cal_brier': cal_brier
        },
        'three_state_adaptive': {
            'effective_f1': f1_3s_adaptive,
            'accuracy': acc_3s_adaptive,
            'precision': prec_3s_adaptive,
            'recall': rec_3s_adaptive,
            'uncertainty_rate': retake_3s_adaptive,
            'coverage': cov_3s_adaptive,
            'view_thresholds': view_th_10,
            'view_audits': view_audits_10
        },
        'three_state_fixed_035_065': {
            'effective_f1': f1_3s_fixed,
            'accuracy': acc_3s_fixed,
            'uncertainty_rate': retake_fixed,
            'coverage': cov_fixed
        },
        'oof_calibrated_probs': oof_cal.tolist()
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"[+] 평가 결과 JSON 저장: {RESULTS_JSON}")

    bundle = {
        'model_name': 'sl1_rtmp_1',
        'models_lgb': models_lgb,
        'models_cat': models_cat,
        'models_xgb': models_xgb,
        'weights': weights,
        'temperature': opt_T,
        'view_thresholds': view_th_10,
        'default_thresholds': {'tau_clean': 0.35, 'tau_defect': 0.65},
        'feature_names': SIDE_LUNGE_C1_FEATURE_NAMES_V11,
        'view_alias_map': VIEW_ALIAS_MAP
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as fp:
        pickle.dump(bundle, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 모델 번들 저장: {MODEL_BUNDLE_PKL}")

    return results


if __name__ == '__main__':
    run_sl1_rtmp_1_training_and_evaluation()
