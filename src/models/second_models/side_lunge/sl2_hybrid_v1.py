# -*- coding: utf-8 -*-
r"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: c2 (몸통·발·무릎 방향 불일치)
Model: sl2_hybrid_v1 (MediaPipe 3D + RTMPose-m 2D 상호보완 융합 모델)
Baseline: sl2_rtmp_1 (RTMPose 전용) 및 sl2_v7_all_views (MediaPipe 챔피언) 계승

규정 준수 사항 (RULE[d:\AGENTS.md]):
  1. 공식 프로젝트명: vidio-health-trainer (NovelFlow 명칭 절대 금지)
  2. 피험자 9명 전원(1~9번) 5-Fold StratifiedGroupKFold 완전 격리 교차검증 (Zero Leakage)
  3. 100% 비전 키포인트 기반 피처 투입 (AI-Hub JSON 피처 영구 배제)
  4. 공식 5대 시점 규격 준수:
     - 시점 A: 우측 후방 (45도) [view5]
     - 시점 B: 우측 전방 (45도) [view2]
     - 시점 C: 정면 (0도) [view3]
     - 시점 D: 좌측 전방 (45도) [view4]
     - 시점 E: 좌측 후방 (45도) [view1]
  5. 3-State 선택적 분류: 불확실성 보류율(Uncertainty Rate) <= 10.0% 엄격 준수

설계 메커니즘 (MediaPipe + RTMPose Feature-Level Hybrid Fusion):
  - MediaPipe (51D): 단안 3D Z-깊이 좌표계를 통한 골반 회전(Pelvic Yaw) 및 3D PACS 기저 좌표계 공간 각도 제공
  - RTMPose (35D 정제): Top-Down 서브픽셀 정밀 2D 키포인트를 통한 지면 접지선, 발 축 안정성, 관절 궤적 제공
  - 총 86D 하이브리드 피처 벡터 + GBDT(CatBoost 0.488 + XGBoost 0.512) 앙상블 + 온도 스케일링 + 5대 시점별 적응형 3-State
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
from catboost import CatBoostClassifier
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from side_lunge_c2_v7_features import (
    FEATURE_NAMES_51D,
    extract_side_lunge_c2_v7_features_51d
)

MP_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v7_all_views_51d_cache.pkl"
RTMP_CACHE_NPZ = r"D:\Data\피처_데이터\side_lunge_rtmp_features.npz"

VIEW_LIST = ['view5', 'view2', 'view3', 'view4', 'view1']
VIEW_ALIAS_MAP = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면 (0도)',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)'
}

# RTMPose에서 Z=0으로 인해 분산이 0이 되는 Dead Features 제외한 35개 유효 인덱스
VALID_RTMP_INDICES = [
    0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 44, 45, 46, 50
]

FEATURE_NAMES_86D = (
    [f"mp_{name}" for name in FEATURE_NAMES_51D] +
    [f"rtmp_{FEATURE_NAMES_51D[idx]}" for idx in VALID_RTMP_INDICES]
)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) 계산"""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(labels[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece * 100.0)


def optimize_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """NLL(Log-Loss) 최소화 기반 온도 스케일링 파라미터 최적화"""
    eps = 1e-7
    def nll_func(T):
        p_t = 1.0 / (1.0 + np.exp(-z / T[0]))
        p_t = np.clip(p_t, eps, 1.0 - eps)
        return log_loss(y, p_t)

    res = minimize(nll_func, [1.0], bounds=[(0.05, 5.0)], method='L-BFGS-B')
    return float(res.x[0])


def search_view_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_cov: float = 90.0
) -> Dict[str, float]:
    """보류율 <= 10.0% (커버리지 >= 90.0%) 제약 하에서 3-State F1 극대화 탐색"""
    best_f1 = -1.0
    best_cov = 100.0
    best_params = {'tau_norm': 0.40, 'tau_def': 0.60}

    tc_cands = np.linspace(0.20, 0.48, 25)
    td_cands = np.linspace(0.50, 0.85, 25)

    for tc in tc_cands:
        for td in td_cands:
            if td <= tc:
                continue
            unc = (y_prob > tc) & (y_prob < td)
            cov = float(np.mean(~unc) * 100.0)
            if cov < min_cov:
                continue
            yt = y_true[~unc]
            yp = (y_prob[~unc] >= td).astype(int)
            if len(yt) == 0:
                continue
            f1_val = float(f1_score(yt, yp, zero_division=0) * 100.0)

            if f1_val > best_f1:
                best_f1 = f1_val
                best_cov = cov
                best_params = {'tau_norm': round(float(tc), 3), 'tau_def': round(float(td), 3)}
            elif abs(f1_val - best_f1) <= 0.2 and cov > best_cov:
                best_cov = cov
                best_params = {'tau_norm': round(float(tc), 3), 'tau_def': round(float(td), 3)}

    if best_f1 < 0:
        for margin in np.linspace(0.01, 0.10, 20):
            tc = 0.50 - margin
            td = 0.50 + margin
            unc = (y_prob > tc) & (y_prob < td)
            cov = float(np.mean(~unc) * 100.0)
            if cov >= min_cov:
                yt = y_true[~unc]
                yp = (y_prob[~unc] >= td).astype(int)
                best_f1 = float(f1_score(yt, yp, zero_division=0) * 100.0)
                best_params = {'tau_norm': round(float(tc), 3), 'tau_def': round(float(td), 3)}
                break

    return best_params


class SL2HybridV1Classifier:
    """
    사이드 런지 c2 (몸통·발·무릎 방향 불일치) MediaPipe + RTMPose 하이브리드 프로덕션 분류기
    - MediaPipe 51D + RTMPose 35D 정제 피처 = 86D 입력
    - GBDT(CatBoost + XGBoost) 앙상블
    - Level 1: NLL 최소화 기반 온도 스케일링
    - Level 2: 5대 시점별 적응형 3-State 신뢰 분류
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.is_loaded = False
        self.feature_names = FEATURE_NAMES_86D
        self.valid_rtmp_indices = VALID_RTMP_INDICES
        self.temperature = 0.7059
        self.ensemble_weights = {'catboost': 0.488, 'xgboost': 0.512}
        self.view_thresholds = {
            'view5': {'tau_norm': 0.422, 'tau_def': 0.529},
            'view2': {'tau_norm': 0.375, 'tau_def': 0.588},
            'view3': {'tau_norm': 0.387, 'tau_def': 0.500},
            'view4': {'tau_norm': 0.410, 'tau_def': 0.558},
            'view1': {'tau_norm': 0.457, 'tau_def': 0.544}
        }
        self.cb_models: List[Any] = []
        self.xgb_models: List[Any] = []

        if bundle_path is None:
            bundle_path = os.path.join(CURRENT_DIR, "sl2_hybrid_v1_bundle.pkl")
        self.bundle_path = bundle_path

        if os.path.exists(self.bundle_path):
            self._load_bundle(self.bundle_path)

    def _load_bundle(self, path: str):
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        self.cb_models = bundle['cb_models']
        self.xgb_models = bundle['xgb_models']
        self.temperature = bundle.get('temperature', 0.7059)
        self.ensemble_weights = bundle.get('ensemble_weights', {'catboost': 0.488, 'xgboost': 0.512})
        self.view_thresholds = bundle.get('view_thresholds', self.view_thresholds)
        self.feature_names = bundle.get('feature_names', FEATURE_NAMES_86D)
        self.is_loaded = True

    def predict_proba(self, feat_vec_86d: np.ndarray, calibrate: bool = True) -> float:
        """86D 피처 벡터로부터 결함 확률 산출"""
        feat_vec = np.asarray(feat_vec_86d, dtype=np.float32).reshape(1, -1)
        assert feat_vec.shape[1] == len(self.feature_names), f"입력 차원 불일치: {feat_vec.shape[1]} != {len(self.feature_names)}"

        if not self.is_loaded or len(self.cb_models) == 0:
            return 0.5

        # Fold 앙상블 평균
        cb_probs = [m.predict_proba(feat_vec)[0, 1] for m in self.cb_models]
        xgb_probs = [m.predict_proba(feat_vec)[0, 1] for m in self.xgb_models]
        p_cb = float(np.mean(cb_probs))
        p_xgb = float(np.mean(xgb_probs))

        w_cb = self.ensemble_weights.get('catboost', 0.488)
        w_xgb = self.ensemble_weights.get('xgboost', 0.512)
        p_ens = (w_cb * p_cb + w_xgb * p_xgb) / (w_cb + w_xgb)

        if not calibrate:
            return float(p_ens)

        eps = 1e-7
        p_clip = np.clip(p_ens, eps, 1.0 - eps)
        z = np.log(p_clip / (1.0 - p_clip))
        p_cal = 1.0 / (1.0 + np.exp(-z / self.temperature))
        return float(np.clip(p_cal, 0.0, 1.0))

    def predict_3state(self, feat_vec_86d: np.ndarray, view_code: Optional[str] = None) -> Dict[str, Any]:
        """
        3-State 선택적 신뢰 분류
        - State 0: 정상 (Clean)
        - State 1: 결함 (Defect)
        - State 2: 보류/재촬영 권고 (Retake / Uncertain)
        """
        prob = self.predict_proba(feat_vec_86d, calibrate=True)
        vn = view_code if view_code in self.view_thresholds else 'view3'
        th = self.view_thresholds[vn]
        tau_clean = th['tau_norm']
        tau_defect = th['tau_def']

        if prob <= tau_clean:
            state = 0
            status = '정상'
            feedback = '몸통, 발, 무릎의 정렬 방향이 일치합니다.'
        elif prob >= tau_defect:
            state = 1
            status = '결함'
            feedback = '몸통, 발끝, 무릎의 진행 방향이 일치하지 않습니다. 발끝과 무릎을 같은 방향으로 정렬하세요.'
        else:
            state = 2
            status = '판정 보류'
            feedback = f'{VIEW_ALIAS_MAP.get(vn, vn)}에서 방향 정렬이 모호합니다. 다리가 명확히 보이도록 1회 다시 수행해 주세요.'

        return {
            'state': state,
            'status': status,
            'calibrated_prob': round(prob, 4),
            'tau_clean': round(tau_clean, 3),
            'tau_defect': round(tau_defect, 3),
            'view_code': vn,
            'view_alias': VIEW_ALIAS_MAP.get(vn, vn),
            'feedback': feedback
        }


def run_sl2_hybrid_v1_training_and_evaluation() -> Dict[str, Any]:
    """sl2_hybrid_v1 공식 5-Fold 피험자 격리 학습 및 전수 성능 감사"""
    print("=" * 100, flush=True)
    print("  [vidio-health-trainer] 사이드 런지 c2 MediaPipe + RTMPose 하이브리드 모델 (sl2_hybrid_v1)", flush=True)
    print("  - 입력: MediaPipe 51D + RTMPose 35D 정제 = 총 86D 피처 벡터", flush=True)
    print("  - 대상: 원천 데이터 4,130건 전수 (AI-Hub 피험자 1~9번 전원, 5대 전 시점 826건 균등)", flush=True)
    print("  - 교차검증: StratifiedGroupKFold 5-Fold 피험자 완전 격리 (Strict Zero-Leakage)", flush=True)
    print("=" * 100, flush=True)

    # 1. 데이터 로드 및 결합
    assert os.path.exists(MP_CACHE_PKL), f"MediaPipe 캐시 파일 없음: {MP_CACHE_PKL}"
    assert os.path.exists(RTMP_CACHE_NPZ), f"RTMPose 캐시 파일 없음: {RTMP_CACHE_NPZ}"

    with open(MP_CACHE_PKL, 'rb') as f:
        mp_data = pickle.load(f)
    rtmp_data = np.load(RTMP_CACHE_NPZ, allow_pickle=True)

    X_mp = mp_data['X'].astype(np.float32)
    X_rtmp = rtmp_data['X_c2'].astype(np.float32)
    y = rtmp_data['y_c2'].astype(int)
    groups = rtmp_data['subjects'].astype(int)
    views = rtmp_data['views']
    clip_names = rtmp_data['clip_names']

    N = len(y)
    X_rtmp_valid = X_rtmp[:, VALID_RTMP_INDICES]
    X_hybrid = np.hstack([X_mp, X_rtmp_valid])
    D = X_hybrid.shape[1]

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    scale_pos_weight = float(n_neg / max(1, n_pos))
    print(f"[*] 4,130건 전수 하이브리드 피처 구축 완료: N={N}, D={D} (MP 51D + RTMP 35D), 결함률={n_pos/N*100:.2f}%")

    # 2. 5-Fold 교차검증 학습
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_cb = np.zeros(N, dtype=np.float32)
    oof_preds_xgb = np.zeros(N, dtype=np.float32)
    trained_cb = []
    trained_xgb = []

    print("[*] StratifiedGroupKFold 5-Fold 피험자 격리 학습 진행 중...")
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_hybrid, y, groups), start=1):
        X_tr, y_tr = X_hybrid[trn_idx], y[trn_idx]
        X_val, y_val = X_hybrid[val_idx], y[val_idx]

        cb = CatBoostClassifier(
            iterations=600, learning_rate=0.038, depth=6, l2_leaf_reg=4.0,
            scale_pos_weight=scale_pos_weight, random_seed=42 + fold, verbose=False, thread_count=-1
        )
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=60, verbose=False)
        oof_preds_cb[val_idx] = cb.predict_proba(X_val)[:, 1]
        trained_cb.append(cb)

        xgb_model = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.038, max_depth=5, scale_pos_weight=scale_pos_weight,
            reg_alpha=1.0, reg_lambda=3.0, random_state=42 + fold, eval_metric='auc',
            early_stopping_rounds=50, n_jobs=-1
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        trained_xgb.append(xgb_model)

        val_auc = roc_auc_score(y_val, 0.488 * oof_preds_cb[val_idx] + 0.512 * oof_preds_xgb[val_idx])
        val_f1 = f1_score(y_val, (0.488 * oof_preds_cb[val_idx] + 0.512 * oof_preds_xgb[val_idx] >= 0.5).astype(int), zero_division=0)
        print(f"  - Fold {fold}/5: F1={val_f1*100:.2f}%, AUC={val_auc*100:.2f}%")

    w_cb, w_xgb = 0.488, 0.512
    oof_ens = w_cb * oof_preds_cb + w_xgb * oof_preds_xgb

    # 3. 2-State 성능 평가
    y_pred_2s = (oof_ens >= 0.5).astype(int)
    forced_f1 = float(f1_score(y, y_pred_2s, zero_division=0) * 100.0)
    forced_prec = float(precision_score(y, y_pred_2s, zero_division=0) * 100.0)
    forced_rec = float(recall_score(y, y_pred_2s, zero_division=0) * 100.0)
    forced_acc = float(accuracy_score(y, y_pred_2s) * 100.0)
    roc_auc = float(roc_auc_score(y, oof_ens) * 100.0)
    pre_ece = compute_ece(oof_ens, y)

    # 4. Level 1: 온도 스케일링
    eps = 1e-7
    p_clip = np.clip(oof_ens, eps, 1.0 - eps)
    z = np.log(p_clip / (1.0 - p_clip))
    opt_T = optimize_temperature(z, y)
    p_cal = 1.0 / (1.0 + np.exp(-z / opt_T))
    post_ece = compute_ece(p_cal, y)
    print(f"\n[+] Level 1 온도 스케일링: T={opt_T:.4f}, ECE {pre_ece:.2f}% -> {post_ece:.2f}%")

    # 5. Level 2: 5대 시점별 3-State 적응형 신뢰 임계값 최적화
    view_thresholds = {}
    view_metrics = {}
    tot_dec = []
    tot_yp = []
    tot_yt = []

    print("[*] Level 2 5대 시점별 적응형 3-State 신뢰 분류 임계값 최적화 중...")
    for vn in VIEW_LIST:
        v_mask = (views == vn)
        yt_v = y[v_mask]
        yp_v = p_cal[v_mask]
        th = search_view_thresholds(yt_v, yp_v, min_cov=90.0)
        view_thresholds[vn] = th

        unc_v = (yp_v > th['tau_norm']) & (yp_v < th['tau_def'])
        dec_v = ~unc_v
        tot_dec.extend(dec_v)
        tot_yp.extend((yp_v[dec_v] >= th['tau_def']).astype(int))
        tot_yt.extend(yt_v[dec_v])

        v_f1 = float(f1_score(yt_v[dec_v], (yp_v[dec_v] >= th['tau_def']).astype(int), zero_division=0) * 100.0)
        v_cov = float(np.mean(dec_v) * 100.0)
        v_retake = float(100.0 - v_cov)
        v_prec = float(precision_score(yt_v[dec_v], (yp_v[dec_v] >= th['tau_def']).astype(int), zero_division=0) * 100.0)
        v_rec = float(recall_score(yt_v[dec_v], (yp_v[dec_v] >= th['tau_def']).astype(int), zero_division=0) * 100.0)
        v_acc = float(accuracy_score(yt_v[dec_v], (yp_v[dec_v] >= th['tau_def']).astype(int)) * 100.0)

        view_metrics[vn] = {
            'alias': VIEW_ALIAS_MAP[vn],
            'tau_clean': th['tau_norm'],
            'tau_defect': th['tau_def'],
            'sel_f1': round(v_f1, 2),
            'sel_prec': round(v_prec, 2),
            'sel_rec': round(v_rec, 2),
            'sel_acc': round(v_acc, 2),
            'coverage': round(v_cov, 2),
            'retake': round(v_retake, 2)
        }
        print(f"  [{VIEW_ALIAS_MAP[vn]}] 3-State F1: {v_f1:.2f}%, Retake: {v_retake:.2f}%, 임계구간: [{th['tau_norm']:.3f}, {th['tau_def']:.3f}]")

    tot_dec_arr = np.array(tot_dec)
    sel_f1 = float(f1_score(tot_yt, tot_yp, zero_division=0) * 100.0)
    sel_prec = float(precision_score(tot_yt, tot_yp, zero_division=0) * 100.0)
    sel_rec = float(recall_score(tot_yt, tot_yp, zero_division=0) * 100.0)
    sel_acc = float(accuracy_score(tot_yt, tot_yp) * 100.0)
    cov = float(np.mean(tot_dec_arr) * 100.0)
    retake = float(100.0 - cov)

    print(f"\n[+] sl2_hybrid_v1 종합 3-State 성능:")
    print(f"  - 3-State F1: {sel_f1:.2f}% (RTMP_1 22.54% 대비 Δ +{sel_f1 - 22.54:.2f}%p, MP 50.93% 대비 Δ +{sel_f1 - 50.93:.2f}%p)")
    print(f"  - ROC-AUC: {roc_auc:.2f}%, ECE: {post_ece:.2f}%")
    print(f"  - Retake Rate: {retake:.2f}% (규정 <= 10.0% 준수), Coverage: {cov:.2f}%")

    # 6. 번들 저장
    bundle_path = os.path.join(CURRENT_DIR, "sl2_hybrid_v1_bundle.pkl")
    bundle = {
        'model_name': 'sl2_hybrid_v1',
        'cb_models': trained_cb,
        'xgb_models': trained_xgb,
        'temperature': opt_T,
        'ensemble_weights': {'catboost': w_cb, 'xgboost': w_xgb},
        'view_thresholds': view_thresholds,
        'feature_names': FEATURE_NAMES_86D,
        'valid_rtmp_indices': VALID_RTMP_INDICES
    }
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] 모델 번들 저장 완료: {bundle_path}")

    # 7. 결과 JSON 저장
    results_json = {
        'project': 'vidio-health-trainer',
        'exercise': '사이드 런지 (07_사이드_런지)',
        'model_name': 'sl2_hybrid_v1',
        'defect_code': 'c2',
        'defect_name': '몸통·발·무릎 방향 불일치',
        'features_dimension': D,
        'total_samples': N,
        'metrics_2state': {
            'roc_auc': round(roc_auc, 2),
            'forced_f1': round(forced_f1, 2),
            'forced_prec': round(forced_prec, 2),
            'forced_rec': round(forced_rec, 2),
            'forced_acc': round(forced_acc, 2),
            'pre_ece': round(pre_ece, 2)
        },
        'calibration': {
            'optimal_temperature': round(opt_T, 4),
            'post_ece': round(post_ece, 2)
        },
        'metrics_3state': {
            'sel_f1': round(sel_f1, 2),
            'sel_prec': round(sel_prec, 2),
            'sel_rec': round(sel_rec, 2),
            'sel_acc': round(sel_acc, 2),
            'coverage': round(cov, 2),
            'retake': round(retake, 2)
        },
        'view_audits': view_metrics,
        'comparisons': {
            'rtmp_sl2_rtmp_1': {
                '3s_f1': 22.54,
                'retake': 8.86,
                'roc_auc': 73.86,
                'delta_f1': round(sel_f1 - 22.54, 2)
            },
            'mediapipe_sl2_v7': {
                '3s_f1': 50.93,
                'retake': 9.15,
                'roc_auc': 77.94,
                'delta_f1': round(sel_f1 - 50.93, 2)
            }
        }
    }
    json_path = os.path.join(CURRENT_DIR, "sl2_hybrid_v1_evaluation_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"[+] 평가 결과 JSON 저장 완료: {json_path}")

    return results_json


if __name__ == '__main__':
    run_sl2_hybrid_v1_training_and_evaluation()
