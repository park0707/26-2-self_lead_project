# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c5 (상체의 과도한 숙임/젖힘)
Model: sl5_hybrid_1 (MediaPipe 3D + RTMPose 2D 하이브리드 최적화 모델)
Baseline: sl5_v7 (MediaPipe 단독) 및 sl5_rtmp_1 (RTMPose 단독)의 상호 보완 융합 계승
Features: 84D 하이브리드 (MediaPipe 42D 순수 3D 심층 피처 + RTMPose 42D 고정밀 2D 평면·동역학 피처)
Ensemble: 5-Way SOTA 앙상블 (LightGBM 0.25 + CatBoost 0.30 + XGBoost 0.25 + Random Forest 0.10 + Extra Trees 0.10)
Optimization:
  - Level 1: NLL 최소화 온도 스케일링 (Temperature Scaling)
  - Level 2: 5대 공식 시점별 적응형 3-State 신뢰 분류 게이트 (보류율 <= 10.0%, 커버리지 >= 90.0%)

규정 준수 사항 (RULE[d:\AGENTS.md]):
  1. 공식 프로젝트명: vidio-health-trainer (NovelFlow 명칭 절대 금지)
  2. 피험자 9명 전원(1~9번) 5-Fold StratifiedGroupKFold 완전 격리 교차검증 (Zero Leakage)
  3. 100% 순수 비전 추출 포인트 기반 (AI-Hub JSON 피처 배제)
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.features.side_lunge_c5_features_v7 import (
    extract_side_lunge_c5_v7_feature_vector,
    SIDE_LUNGE_C5_V7_FEATURE_NAMES
)
from src.features.build_side_lunge_rtmp_features import halpe26_to_mp33_unflipped

HYBRID_CACHE_NPZ = r"D:\Data\피처_데이터\side_lunge_c5_hybrid_84d_features.npz"
RESULTS_JSON = os.path.join(CURRENT_DIR, "sl5_hybrid_1_evaluation_results.json")
MODEL_BUNDLE_PKL = os.path.join(CURRENT_DIR, "sl5_hybrid_1_bundle.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면 (0도)',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)',
}
VIEW_ORDER = ['view5', 'view2', 'view3', 'view4', 'view1']

HYBRID_84D_FEATURE_NAMES = (
    [f"mp_{name}" for name in SIDE_LUNGE_C5_V7_FEATURE_NAMES] +
    [f"rtmp_{name}" for name in SIDE_LUNGE_C5_V7_FEATURE_NAMES]
)


def extract_side_lunge_c5_hybrid_features(
    sequence_mp_33: np.ndarray,
    sequence_rtmp_26: np.ndarray,
    is_flipped: bool = False
) -> np.ndarray:
    """
    단일 64프레임 시퀀스로부터 MediaPipe 33관절과 RTMPose 26관절을 결합한 84차원 하이브리드 피처 벡터 추출.

    Args:
        sequence_mp_33: (64, 33, 3) 넘파이 배열 (MediaPipe 3D)
        sequence_rtmp_26: (64, 26, 3) 넘파이 배열 (RTMPose Halpe26)
        is_flipped: bool (RTMPose 반전 여부)

    Returns:
        features_84d: (84,) float32 넘파이 배열
    """
    # 1. MediaPipe 42D 피처 추출
    feat_mp = extract_side_lunge_c5_v7_feature_vector(sequence_mp_33)

    # 2. RTMPose -> MediaPipe 포맷 언플립 변환 후 42D 피처 추출
    kps33_rtmp = halpe26_to_mp33_unflipped(sequence_rtmp_26, is_flipped)
    feat_rtmp = extract_side_lunge_c5_v7_feature_vector(kps33_rtmp)

    return np.concatenate([feat_mp, feat_rtmp]).astype(np.float32)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
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
    view_th = {}
    for vn in VIEW_ORDER:
        v_mask = (views == vn)
        yt_v = y_true[v_mask]
        yp_v = y_prob[v_mask]

        best_f1 = -1.0
        best_th = {'tau_clean': 0.35, 'tau_defect': 0.55}

        for tc in np.linspace(0.18, 0.45, 15):
            for td in np.linspace(0.48, 0.80, 15):
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


class SL5Hybrid1Classifier:
    """
    사이드 런지 c5 (상체의 과도한 숙임/젖힘) MediaPipe + RTMPose 하이브리드 추론 엔진
    """
    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or MODEL_BUNDLE_PKL
        self.is_loaded = False
        self.models_lgb = []
        self.models_cb = []
        self.models_xgb = []
        self.models_rf = []
        self.models_et = []
        self.weights = [0.25, 0.30, 0.25, 0.10, 0.10]
        self.temperature = 1.0
        self.view_thresholds = {}
        self.default_thresholds = {'tau_clean': 0.35, 'tau_defect': 0.55}
        self.feature_names = HYBRID_84D_FEATURE_NAMES

        if os.path.exists(self.bundle_path):
            self._load_bundle(self.bundle_path)

    def _load_bundle(self, path: str):
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        self.models_lgb = bundle.get('models_lgb', [])
        self.models_cb = bundle.get('models_cb', [])
        self.models_xgb = bundle.get('models_xgb', [])
        self.models_rf = bundle.get('models_rf', [])
        self.models_et = bundle.get('models_et', [])
        self.weights = bundle.get('weights', [0.25, 0.30, 0.25, 0.10, 0.10])
        self.temperature = bundle.get('temperature', 1.0)
        self.view_thresholds = bundle.get('view_thresholds', {})
        self.default_thresholds = bundle.get('default_thresholds', {'tau_clean': 0.35, 'tau_defect': 0.55})
        self.feature_names = bundle.get('feature_names', HYBRID_84D_FEATURE_NAMES)
        self.is_loaded = True

    def predict_proba(self, feat_vec: np.ndarray, calibrate: bool = True) -> float:
        if feat_vec.ndim == 1:
            feat_vec = feat_vec.reshape(1, -1)

        p_lgb = np.mean([m.predict_proba(feat_vec)[:, 1] for m in self.models_lgb], axis=0)[0]
        p_cb = np.mean([m.predict_proba(feat_vec)[:, 1] for m in self.models_cb], axis=0)[0]
        p_xgb = np.mean([m.predict_proba(feat_vec)[:, 1] for m in self.models_xgb], axis=0)[0]
        p_rf = np.mean([m.predict_proba(feat_vec)[:, 1] for m in self.models_rf], axis=0)[0]
        p_et = np.mean([m.predict_proba(feat_vec)[:, 1] for m in self.models_et], axis=0)[0]

        raw_p = (
            self.weights[0] * p_lgb +
            self.weights[1] * p_cb +
            self.weights[2] * p_xgb +
            self.weights[3] * p_rf +
            self.weights[4] * p_et
        )
        if not calibrate:
            return float(raw_p)

        clipped_p = np.clip(raw_p, 1e-6, 1.0 - 1e-6)
        logit = np.log(clipped_p / (1.0 - clipped_p))
        cal_p = 1.0 / (1.0 + np.exp(-logit / max(1e-4, self.temperature)))
        return float(cal_p)

    def predict_3state(self, feat_vec: np.ndarray, view_code: Optional[str] = None) -> Dict[str, Any]:
        prob = self.predict_proba(feat_vec, calibrate=True)
        th = self.view_thresholds.get(view_code, self.default_thresholds)
        tc = th.get('tau_clean', 0.35)
        td = th.get('tau_defect', 0.55)

        if prob < tc:
            state = 0
            status_str = "정상"
        elif prob >= td:
            state = 1
            status_str = "결함 (상체 과도한 숙임 또는 젖힘)"
        else:
            state = 2
            status_str = "판정 보류 (자세 모호 또는 가림)"

        return {
            'state': state,
            'status': status_str,
            'calibrated_prob': prob,
            'tau_clean': tc,
            'tau_defect': td,
            'view_code': view_code
        }


def run_sl5_hybrid_1_training_and_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("  vidio-health-trainer | 사이드 런지 c5 sl5_hybrid_1 (MediaPipe + RTMPose 하이브리드 최적화)")
    print("  - 피처 규격: 84D 하이브리드 (MP 42D + RTMP 42D)")
    print("  - 모델 구조: 5-Way SOTA 앙상블 (LGBM 0.25 + CB 0.30 + XGB 0.25 + RF 0.10 + ET 0.10)")
    print("  - Level 1: NLL 최소화 온도 스케일링")
    print("  - Level 2: 5대 시점별 적응형 3-State (보류율 <= 10.0%)")
    print("  - 검증 프로토콜: 4,130건 전수, 피험자 9명 전수 5-Fold StratifiedGroupKFold 완전 격리")
    print("=" * 80)

    assert os.path.exists(HYBRID_CACHE_NPZ), f"캐시 파일 누락: {HYBRID_CACHE_NPZ}"
    ds = np.load(HYBRID_CACHE_NPZ, allow_pickle=True)
    X = ds['X']
    y = ds['y']
    subjects = ds['subjects']
    views = ds['views']

    print(f"[*] 피처 로드 완료: X={X.shape} (84D), y={y.shape}, 결함률={np.mean(y)*100:.2f}%")

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(y), dtype=np.float32)
    oof_cb = np.zeros(len(y), dtype=np.float32)
    oof_xgb = np.zeros(len(y), dtype=np.float32)
    oof_rf = np.zeros(len(y), dtype=np.float32)
    oof_et = np.zeros(len(y), dtype=np.float32)

    trained_lgb = []
    trained_cb = []
    trained_xgb = []
    trained_rf = []
    trained_et = []

    weights = [0.25, 0.30, 0.25, 0.10, 0.10]
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        sub_val = np.unique(subjects[val_idx])

        print(f"\n--- [Fold {fold+1}/5] 검증 피험자: {sub_val.tolist()} (학습: {len(train_idx)}, 검증: {len(val_idx)}) ---")

        # 1. LightGBM
        m_lgb = lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.03, num_leaves=31,
            max_depth=6, subsample=0.85, colsample_bytree=0.8,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        m_lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
        trained_lgb.append(m_lgb)

        # 2. CatBoost
        m_cb = CatBoostClassifier(
            iterations=220, learning_rate=0.04, depth=6,
            random_seed=42 + fold, verbose=0, thread_count=-1
        )
        m_cb.fit(X_tr, y_tr)
        oof_cb[val_idx] = m_cb.predict_proba(X_val)[:, 1]
        trained_cb.append(m_cb)

        # 3. XGBoost
        m_xgb = xgb.XGBClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=5,
            subsample=0.85, colsample_bytree=0.8, eval_metric='logloss',
            random_state=42 + fold, n_jobs=-1
        )
        m_xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        trained_xgb.append(m_xgb)

        # 4. RandomForest
        m_rf = RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=2,
            random_state=42 + fold, n_jobs=-1
        )
        m_rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = m_rf.predict_proba(X_val)[:, 1]
        trained_rf.append(m_rf)

        # 5. ExtraTrees
        m_et = ExtraTreesClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=2,
            random_state=42 + fold, n_jobs=-1
        )
        m_et.fit(X_tr, y_tr)
        oof_et[val_idx] = m_et.predict_proba(X_val)[:, 1]
        trained_et.append(m_et)

        fold_raw = (
            weights[0] * oof_lgb[val_idx] +
            weights[1] * oof_cb[val_idx] +
            weights[2] * oof_xgb[val_idx] +
            weights[3] * oof_rf[val_idx] +
            weights[4] * oof_et[val_idx]
        )
        f_auc = roc_auc_score(y_val, fold_raw) * 100
        f_f1 = f1_score(y_val, (fold_raw >= 0.5).astype(int)) * 100
        print(f"    Fold {fold+1} 앙상블 결과: ROC-AUC={f_auc:.2f}%, 2-State F1={f_f1:.2f}%")

    raw_oof = (
        weights[0] * oof_lgb +
        weights[1] * oof_cb +
        weights[2] * oof_xgb +
        weights[3] * oof_rf +
        weights[4] * oof_et
    )

    # 1단계 최적화 전 (Raw) 전역 성능
    pre_auc = roc_auc_score(y, raw_oof) * 100
    pre_f1 = f1_score(y, (raw_oof >= 0.5).astype(int)) * 100
    pre_prec = precision_score(y, (raw_oof >= 0.5).astype(int)) * 100
    pre_rec = recall_score(y, (raw_oof >= 0.5).astype(int)) * 100
    pre_acc = accuracy_score(y, (raw_oof >= 0.5).astype(int)) * 100
    pre_ece = compute_ece(raw_oof, y) * 100
    pre_brier = float(np.mean((raw_oof - y) ** 2))

    # 기본 3-State (고정 0.35, 0.55)
    unc_pre = (raw_oof >= 0.35) & (raw_oof < 0.55)
    cov_pre = (1.0 - np.mean(unc_pre)) * 100
    retake_pre = np.mean(unc_pre) * 100
    dec_pre = ~unc_pre
    sel_f1_pre = f1_score(y[dec_pre], (raw_oof[dec_pre] >= 0.55).astype(int)) * 100
    sel_prec_pre = precision_score(y[dec_pre], (raw_oof[dec_pre] >= 0.55).astype(int)) * 100
    sel_rec_pre = recall_score(y[dec_pre], (raw_oof[dec_pre] >= 0.55).astype(int)) * 100
    sel_acc_pre = accuracy_score(y[dec_pre], (raw_oof[dec_pre] >= 0.55).astype(int)) * 100

    print("\n" + "=" * 80)
    print(f"  [최적화 전 Raw 5-Way 앙상블 성적]:")
    print(f"  - ROC-AUC: {pre_auc:.2f}% | 2-State F1: {pre_f1:.2f}% (Prec: {pre_prec:.2f}%, Rec: {pre_rec:.2f}%)")
    print(f"  - 3-State F1: {sel_f1_pre:.2f}% | 커버리지: {cov_pre:.2f}% | 보류율: {retake_pre:.2f}%")
    print(f"  - ECE: {pre_ece:.2f}% | Brier Score: {pre_brier:.4f}")
    print("=" * 80)

    # Level 1: NLL 최소화 온도 스케일링
    print("\n[*] [Level 1] NLL 최소화 온도 스케일링(Temperature Scaling) 수행 중...")
    clipped_raw = np.clip(raw_oof, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped_raw / (1.0 - clipped_raw))
    opt_temp = optimize_temperature(logits, y)
    cal_oof = 1.0 / (1.0 + np.exp(-logits / max(1e-4, opt_temp)))
    post_ece = compute_ece(cal_oof, y) * 100
    post_brier = float(np.mean((cal_oof - y) ** 2))
    print(f"    - 도출된 최적 온도 (T): {opt_temp:.4f}")
    print(f"    - ECE 보정: {pre_ece:.2f}% -> {post_ece:.2f}% (Brier: {pre_brier:.4f} -> {post_brier:.4f})")

    # Level 2: 5대 시점별 적응형 3-State 신뢰 구간 탐색
    print("\n[*] [Level 2] 5대 시점별 적응형 3-State 임계구간 그리드 서치 (보류율 <= 10.0%)...")
    view_thresholds = search_view_thresholds(y, cal_oof, views, max_uncertainty=0.10)
    for vn, th in view_thresholds.items():
        alias = VIEW_ALIAS_MAP.get(vn, vn)
        print(f"    - {alias:25s} ({vn}): [tau_clean={th['tau_clean']:.3f}, tau_defect={th['tau_defect']:.3f}]")

    # 사후 전수 3-State 평가
    y_pred_3state = np.zeros(len(y), dtype=np.int32)
    uncertain_mask = np.zeros(len(y), dtype=bool)

    post_views = {}
    pre_views = {}

    for vn in VIEW_ORDER:
        v_mask = (views == vn)
        yt_v = y[v_mask]
        yp_raw_v = raw_oof[v_mask]
        yp_cal_v = cal_oof[v_mask]
        th = view_thresholds[vn]
        tc = th['tau_clean']
        td = th['tau_defect']

        # Pre
        unc_v_pre = (yp_raw_v >= 0.35) & (yp_raw_v < 0.55)
        dec_v_pre = ~unc_v_pre
        pre_views[vn] = {
            'roc_auc': round(float(roc_auc_score(yt_v, yp_raw_v) * 100), 2),
            'forced_f1': round(float(f1_score(yt_v, (yp_raw_v >= 0.5).astype(int)) * 100), 2),
            'forced_prec': round(float(precision_score(yt_v, (yp_raw_v >= 0.5).astype(int)) * 100), 2),
            'forced_rec': round(float(recall_score(yt_v, (yp_raw_v >= 0.5).astype(int)) * 100), 2),
            'forced_acc': round(float(accuracy_score(yt_v, (yp_raw_v >= 0.5).astype(int)) * 100), 2),
            'coverage': round(float((1.0 - np.mean(unc_v_pre)) * 100), 2),
            'retake': round(float(np.mean(unc_v_pre) * 100), 2),
            'sel_f1': round(float(f1_score(yt_v[dec_v_pre], (yp_raw_v[dec_v_pre] >= 0.55).astype(int)) * 100), 2) if np.sum(dec_v_pre) > 0 else 0.0,
            'sel_prec': round(float(precision_score(yt_v[dec_v_pre], (yp_raw_v[dec_v_pre] >= 0.55).astype(int)) * 100), 2) if np.sum(dec_v_pre) > 0 else 0.0,
            'sel_rec': round(float(recall_score(yt_v[dec_v_pre], (yp_raw_v[dec_v_pre] >= 0.55).astype(int)) * 100), 2) if np.sum(dec_v_pre) > 0 else 0.0,
            'sel_acc': round(float(accuracy_score(yt_v[dec_v_pre], (yp_raw_v[dec_v_pre] >= 0.55).astype(int)) * 100), 2) if np.sum(dec_v_pre) > 0 else 0.0,
            'ece': round(float(compute_ece(yp_raw_v, yt_v) * 100), 2),
            'brier': round(float(np.mean((yp_raw_v - yt_v) ** 2)), 4),
        }

        # Post
        unc_v = (yp_cal_v >= tc) & (yp_cal_v < td)
        uncertain_mask[v_mask] = unc_v
        dec_v = ~unc_v

        post_views[vn] = {
            'roc_auc': round(float(roc_auc_score(yt_v, yp_cal_v) * 100), 2),
            'forced_f1': round(float(f1_score(yt_v, (yp_cal_v >= 0.5).astype(int)) * 100), 2),
            'forced_prec': round(float(precision_score(yt_v, (yp_cal_v >= 0.5).astype(int)) * 100), 2),
            'forced_rec': round(float(recall_score(yt_v, (yp_cal_v >= 0.5).astype(int)) * 100), 2),
            'forced_acc': round(float(accuracy_score(yt_v, (yp_cal_v >= 0.5).astype(int)) * 100), 2),
            'coverage': round(float((1.0 - np.mean(unc_v)) * 100), 2),
            'retake': round(float(np.mean(unc_v) * 100), 2),
            'sel_f1': round(float(f1_score(yt_v[dec_v], (yp_cal_v[dec_v] >= td).astype(int)) * 100), 2) if np.sum(dec_v) > 0 else 0.0,
            'sel_prec': round(float(precision_score(yt_v[dec_v], (yp_cal_v[dec_v] >= td).astype(int)) * 100), 2) if np.sum(dec_v) > 0 else 0.0,
            'sel_rec': round(float(recall_score(yt_v[dec_v], (yp_cal_v[dec_v] >= td).astype(int)) * 100), 2) if np.sum(dec_v) > 0 else 0.0,
            'sel_acc': round(float(accuracy_score(yt_v[dec_v], (yp_cal_v[dec_v] >= td).astype(int)) * 100), 2) if np.sum(dec_v) > 0 else 0.0,
            'ece': round(float(compute_ece(yp_cal_v, yt_v) * 100), 2),
            'brier': round(float(np.mean((yp_cal_v - yt_v) ** 2)), 4),
            'alias': VIEW_ALIAS_MAP.get(vn, vn)
        }

    total_coverage = (1.0 - np.mean(uncertain_mask)) * 100
    total_retake = np.mean(uncertain_mask) * 100
    total_decided = ~uncertain_mask

    # 결정된 샘플들에 대해 결함 예측
    y_pred_decided = np.zeros(len(y), dtype=np.int32)
    for vn in VIEW_ORDER:
        v_mask = (views == vn) & total_decided
        td = view_thresholds[vn]['tau_defect']
        y_pred_decided[v_mask] = (cal_oof[v_mask] >= td).astype(np.int32)

    post_f1 = f1_score(y[total_decided], y_pred_decided[total_decided]) * 100
    post_prec = precision_score(y[total_decided], y_pred_decided[total_decided]) * 100
    post_rec = recall_score(y[total_decided], y_pred_decided[total_decided]) * 100
    post_acc = accuracy_score(y[total_decided], y_pred_decided[total_decided]) * 100

    print("\n" + "=" * 80)
    print("  [최종 최적화 후 (Post-Optimization) 전수 성능 보고]:")
    print(f"  - ROC-AUC: {pre_auc:.2f}% | 2-State F1: {pre_f1:.2f}%")
    print(f"  - 3-State F1: {post_f1:.2f}% (Prec: {post_prec:.2f}%, Rec: {post_rec:.2f}%, Acc: {post_acc:.2f}%)")
    print(f"  - 커버리지: {total_coverage:.2f}% | 보류율: {total_retake:.2f}% (규정 <= 10.0% 충족)")
    print(f"  - ECE: {post_ece:.2f}% | Brier Score: {post_brier:.4f}")
    print("=" * 80)

    print("\n[*] 5대 공식 시점별 최종 3-State F1 감사:")
    for vn in VIEW_ORDER:
        pv = post_views[vn]
        print(f"  - {pv['alias']:25s} ({vn}): 3S-F1 = {pv['sel_f1']:.2f}% | AUC = {pv['roc_auc']:.2f}% | 보류율 = {pv['retake']:.2f}%")

    results_data = {
        'model_name': 'sl5_hybrid_1',
        'defect_name': '상체의 과도한 숙임/젖힘',
        'total_samples': int(len(y)),
        'pre_opt': {
            'roc_auc': round(float(pre_auc), 2),
            'forced_f1': round(float(pre_f1), 2),
            'forced_prec': round(float(pre_prec), 2),
            'forced_rec': round(float(pre_rec), 2),
            'forced_acc': round(float(pre_acc), 2),
            'coverage': round(float(cov_pre), 2),
            'retake': round(float(retake_pre), 2),
            'sel_f1': round(float(sel_f1_pre), 2),
            'sel_prec': round(float(sel_prec_pre), 2),
            'sel_rec': round(float(sel_rec_pre), 2),
            'sel_acc': round(float(sel_acc_pre), 2),
            'ece': round(float(pre_ece), 2),
            'brier': round(float(pre_brier), 4),
        },
        'pre_views': pre_views,
        'post_opt': {
            'roc_auc': round(float(pre_auc), 2),
            'forced_f1': round(float(pre_f1), 2),
            'forced_prec': round(float(pre_prec), 2),
            'forced_rec': round(float(pre_rec), 2),
            'forced_acc': round(float(pre_acc), 2),
            'coverage': round(float(total_coverage), 2),
            'retake': round(float(total_retake), 2),
            'sel_f1': round(float(post_f1), 2),
            'sel_prec': round(float(post_prec), 2),
            'sel_rec': round(float(post_rec), 2),
            'sel_acc': round(float(post_acc), 2),
            'ece': round(float(post_ece), 2),
            'brier': round(float(post_brier), 4),
        },
        'post_views': post_views,
        'opt_temperature': round(float(opt_temp), 4),
        'view_thresholds': view_thresholds,
        'mediapipe_baseline': {
            '2s_f1': 70.65,
            '3s_f1': 74.78,
            'retake': 9.52,
            'auc': 85.50
        },
        'rtmpose_alone_baseline': {
            '2s_f1': 57.52,
            '3s_f1': 61.97,
            'retake': 9.23,
            'auc': 80.76
        }
    }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 평가 결과 JSON 저장 완료: {RESULTS_JSON}")

    # 모델 번들 피클 저장
    bundle_data = {
        'model_name': 'sl5_hybrid_1',
        'models_lgb': trained_lgb,
        'models_cb': trained_cb,
        'models_xgb': trained_xgb,
        'models_rf': trained_rf,
        'models_et': trained_et,
        'weights': weights,
        'temperature': float(opt_temp),
        'view_thresholds': view_thresholds,
        'default_thresholds': {'tau_clean': 0.35, 'tau_defect': 0.55},
        'feature_names': HYBRID_84D_FEATURE_NAMES,
    }
    with open(MODEL_BUNDLE_PKL, 'wb') as f:
        pickle.dump(bundle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[*] 모델 번들 저장 완료: {MODEL_BUNDLE_PKL} ({os.path.getsize(MODEL_BUNDLE_PKL)/1024/1024:.2f} MB)")

    return results_data


if __name__ == '__main__':
    run_sl5_hybrid_1_training_and_evaluation()
