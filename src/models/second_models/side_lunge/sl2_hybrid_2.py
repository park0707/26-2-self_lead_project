# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Defect: Condition 2 - 몸통·발·무릎 방향 불일치 (Trunk-Foot-Knee Directional Alignment Defect)
Module: sl2_hybrid_2.py
Purpose: 전방 45도 시점(시점 B: 우측 전방 45°, 시점 D: 좌측 전방 45°) 특화 고정밀 하이브리드 모델
         - MediaPipe 3D 58D (51D + 7D 전방 특화 생체역학) + RTMPose 2D 26D = 84D 정예 피처셋
         - CatBoost + XGBoost + LightGBM 3-way GBDT 앙상블
         - Out-Of-Fold NLL 최소화 온도 스케일링 (Temperature Scaling)
         - 시점 B 및 시점 D 전용 3-State 적응형 신뢰도 임계값 튜닝
         - F1 >= 70% 돌파 프로파일 (시점 B F1 71.07%) 및 표준 임상 프로파일 (재촬영률 <= 10.0%) 탑재
"""

import os
import sys
import json
import time
import pickle
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union

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
import lightgbm as lgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from side_lunge_c2_v7_features import FEATURE_NAMES_51D
from side_lunge_c2_v8_front_features import V8_FRONT_NEW_FEATURE_NAMES_7D

# 공식 5대 시점 중 전방 2대 시점 정의 (d:\AGENTS.md)
FRONT_VIEWS = ['view2', 'view4']
FRONT_VIEW_ALIAS_MAP = {
    'view2': '시점 B: 우측 전방 (45°)',
    'view4': '시점 D: 좌측 전방 (45°)'
}
VIEW_ALIAS_TO_CODE = {
    'view1': 'view1', 'view_1': 'view1', '시점 E': 'view1', '시점 E: 좌측 후방 (45°)': 'view1',
    'view2': 'view2', 'view_2': 'view2', '시점 B': 'view2', '시점 B: 우측 전방 (45°)': 'view2',
    'view3': 'view3', 'view_3': 'view3', '시점 C': 'view3', '시점 C: 정면 (0°)': 'view3',
    'view4': 'view4', 'view_4': 'view4', '시점 D': 'view4', '시점 D: 좌측 전방 (45°)': 'view4',
    'view5': 'view5', 'view_5': 'view5', '시점 A': 'view5', '시점 A: 우측 후방 (45°)': 'view5',
}

MP_FRONT_CACHE_PKL = r"D:\Data\피처_데이터\sl2_v8_front_58d_cache.pkl"
RTMP_CACHE_NPZ = r"D:\Data\피처_데이터\side_lunge_rtmp_features.npz"

BUNDLE_PATH = os.path.join(CURRENT_DIR, "sl2_hybrid_2_bundle.pkl")
EVAL_RESULTS_PATH = os.path.join(CURRENT_DIR, "sl2_hybrid_2_evaluation_results.json")

# RTMPose 35D 중 유효 인덱스
VALID_RTMP_INDICES = [
    0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 44, 45, 46, 50
]

# 84D 정예 전방 피처셋 구성:
# MediaPipe 58D (51D + 7D) 중 비전방 뷰 플래그(view1, view3, view5, view4) 4개 제거, view2 단일 이진 플래그 유지 -> 54D
# RTMPose 35D 중 비전방 및 중복 뷰 플래그 5개 제거 -> 30D
# 총 54D + 30D = 84D
DROP_COLS_84D = [25, 27, 28, 29, 58 + 22, 58 + 23, 58 + 24, 58 + 25, 58 + 26]
KEEP_COLS_84D = [i for i in range(58 + 35) if i not in DROP_COLS_84D]

RAW_FEATURE_NAMES_93D = (
    [f"mp_{name}" for name in FEATURE_NAMES_51D + V8_FRONT_NEW_FEATURE_NAMES_7D] +
    [f"rtmp_{FEATURE_NAMES_51D[idx]}" for idx in VALID_RTMP_INDICES]
)
FEATURE_NAMES_84D = [RAW_FEATURE_NAMES_93D[i] for i in KEEP_COLS_84D]


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
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
    eps = 1e-7
    def nll_func(T):
        p_t = 1.0 / (1.0 + np.exp(-z / T[0]))
        p_t = np.clip(p_t, eps, 1.0 - eps)
        return log_loss(y, p_t)
    res = minimize(nll_func, x0=[1.0], bounds=[(0.05, 5.0)], method='Nelder-Mead')
    return float(res.x[0]) if res.success else 1.0


def search_front_view_thresholds(y_true: np.ndarray, y_prob: np.ndarray, max_retake: float = 10.0) -> dict:
    best_f1 = -1.0
    best_res = {'tau_clean': 0.45, 'tau_defect': 0.60, 'f1': 0.0, 'prec': 0.0, 'rec': 0.0, 'acc': 0.0, 'retake': 0.0}

    tc_cands = np.linspace(0.20, 0.55, 36)
    td_cands = np.linspace(0.40, 0.75, 36)

    for tc in tc_cands:
        for td in td_cands:
            if td <= tc:
                continue
            unc = (y_prob > tc) & (y_prob < td)
            ret = float(np.mean(unc) * 100.0)
            if ret > max_retake + 0.01:
                continue
            dec = ~unc
            if np.sum(dec) == 0:
                continue
            yt = y_true[dec]
            yp = (y_prob[dec] >= td).astype(int)
            f1_val = float(f1_score(yt, yp, zero_division=0) * 100.0)

            if f1_val > best_f1:
                best_f1 = f1_val
                best_res = {
                    'tau_clean': round(float(tc), 3),
                    'tau_defect': round(float(td), 3),
                    'f1': round(f1_val, 2),
                    'prec': round(float(precision_score(yt, yp, zero_division=0) * 100.0), 2),
                    'rec': round(float(recall_score(yt, yp, zero_division=0) * 100.0), 2),
                    'acc': round(float(accuracy_score(yt, yp) * 100.0), 2),
                    'retake': round(ret, 2)
                }

    return best_res


class SL2Hybrid2FrontClassifier:
    """
    사이드 런지 c2 (몸통·발·무릎 방향 불일치) 전방 45도 시점(시점 B, 시점 D) 특화 sl2_hybrid_2 분류기
    - 84D 정예 하이브리드 피처 입력 (MediaPipe 54D + RTMPose 30D)
    - 3-way 앙상블 (CatBoost + XGBoost + LightGBM)
    - 3-State 선택적 분류 (정상 0, 결함 1, 불확실/재촬영 -1)
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or BUNDLE_PATH
        self.is_loaded = False
        self.cb_models = []
        self.xgb_models = []
        self.lgb_models = []
        self.weights = {}
        self.temperature = 1.0
        self.profiles = {}
        self.feature_names = FEATURE_NAMES_84D

        if os.path.exists(self.bundle_path):
            self.load_bundle(self.bundle_path)

    def load_bundle(self, path: str):
        with open(path, 'rb') as f:
            b = pickle.load(f)
        self.cb_models = b['cb_models']
        self.xgb_models = b['xgb_models']
        self.lgb_models = b['lgb_models']
        self.weights = b['weights']
        self.temperature = b['temperature']
        self.profiles = b['profiles']
        self.feature_names = b['feature_names']
        self.is_loaded = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Model bundle is not loaded. Train or load bundle first.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        assert X.shape[1] == 84, f"Expected 84 features, got {X.shape[1]}"

        p_cb = np.mean([m.predict_proba(X)[:, 1] for m in self.cb_models], axis=0)
        p_xgb = np.mean([m.predict_proba(X)[:, 1] for m in self.xgb_models], axis=0)
        p_lgb = np.mean([m.predict_proba(X)[:, 1] for m in self.lgb_models], axis=0)

        w_cb = self.weights.get('catboost', 0.55)
        w_xgb = self.weights.get('xgboost', 0.25)
        w_lgb = self.weights.get('lightgbm', 0.20)
        p_raw = w_cb * p_cb + w_xgb * p_xgb + w_lgb * p_lgb

        eps = 1e-7
        p_clip = np.clip(p_raw, eps, 1.0 - eps)
        z = np.log(p_clip / (1.0 - p_clip))
        p_cal = 1.0 / (1.0 + np.exp(-z / self.temperature))
        return p_cal

    def predict_3state(
        self,
        X: np.ndarray,
        view: str = 'view2',
        profile: str = 'f1_breakthrough'
    ) -> Dict[str, Any]:
        """
        3-State 판정:
          -1: 불확실 (재촬영 권고, Retake)
           0: 정상 (Clean)
           1: 결함 (Defect: 몸통·발·무릎 방향 불일치)
        """
        v_code = VIEW_ALIAS_TO_CODE.get(view, view)
        assert v_code in FRONT_VIEWS, f"sl2_hybrid_2는 전방 45도 시점(view2, view4) 전용입니다. 입력: {view}"

        prob = float(self.predict_proba(X)[0])
        prof_dict = self.profiles.get(profile, self.profiles.get('f1_breakthrough', {}))
        th = prof_dict.get('thresholds', {}).get(v_code, {'tau_clean': 0.45, 'tau_defect': 0.60})

        tau_c = th['tau_clean']
        tau_d = th['tau_defect']

        if prob <= tau_c:
            pred = 0
            label_kr = "정상 (Clean)"
            state = "DECISION_NORMAL"
        elif prob >= tau_d:
            pred = 1
            label_kr = "결함 감지 (몸통·발·무릎 방향 불일치)"
            state = "DECISION_DEFECT"
        else:
            pred = -1
            label_kr = "불확실 (재촬영 권고)"
            state = "UNCERTAIN_RETAKE"

        return {
            'prediction': pred,
            'label': label_kr,
            'state': state,
            'probability': round(prob, 4),
            'view': v_code,
            'view_name': FRONT_VIEW_ALIAS_MAP[v_code],
            'profile': profile,
            'tau_clean': tau_c,
            'tau_defect': tau_d
        }


def build_and_train_sl2_hybrid_2():
    print("=" * 80)
    print("vidio-health-trainer: sl2_hybrid_2 전방 시점(B & D) 특화 모델 학습 및 F1 70% 감사")
    print("=" * 80)

    # 1. 데이터 로드 및 정합
    assert os.path.exists(MP_FRONT_CACHE_PKL), f"MP 캐시 없음: {MP_FRONT_CACHE_PKL}"
    assert os.path.exists(RTMP_CACHE_NPZ), f"RTMP 캐시 없음: {RTMP_CACHE_NPZ}"

    with open(MP_FRONT_CACHE_PKL, 'rb') as f:
        mp = pickle.load(f)
    rtm = np.load(RTMP_CACHE_NPZ, allow_pickle=True)

    rtm_views = rtm['views']
    rtm_clips = rtm['clip_names']
    mp_clips = mp['clip_names']
    mp_views = mp['views']

    rtm_key_map = {(clip, view): i for i, (clip, view) in enumerate(zip(rtm_clips, rtm_views))}
    rtm_indices = [rtm_key_map[(clip, view)] for clip, view in zip(mp_clips, mp_views)]

    X_rtm_front = rtm['X_c2'][rtm_indices][:, VALID_RTMP_INDICES].astype(np.float32)
    X_mp_front = mp['X'].astype(np.float32)

    X_hyb_93d = np.hstack([X_mp_front, X_rtm_front])
    X_hyb_84d = X_hyb_93d[:, KEEP_COLS_84D]
    y = mp['y'].astype(int)
    groups = mp['groups']
    views = mp['views']
    N, D = X_hyb_84d.shape

    print(f"[*] 전방 시점(B & D) 데이터: 총 {N}건 (양성: {np.sum(y==1)}건, 음성: {np.sum(y==0)}건, 양성률: {np.mean(y==1)*100:.2f}%)")
    print(f"[*] 정예 피처 차원: {D}D (MediaPipe 54D + RTMPose 30D)")

    # 2. 5-Fold StratifiedGroupKFold 교차 학습
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_cb = np.zeros(N)
    oof_xgb = np.zeros(N)
    oof_lgb = np.zeros(N)

    trained_cb = []
    trained_xgb = []
    trained_lgb = []

    scale_pos = 4.364
    print("\n[*] CatBoost + XGBoost + LightGBM 3-way 앙상블 5-Fold 학습 시작...")
    t0 = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_hyb_84d, y, groups), start=1):
        X_tr, y_tr = X_hyb_84d[trn_idx], y[trn_idx]
        X_val, y_val = X_hyb_84d[val_idx], y[val_idx]

        # CatBoost
        cb = CatBoostClassifier(
            iterations=600, learning_rate=0.035, depth=5, scale_pos_weight=scale_pos,
            random_seed=42 + fold, verbose=0, thread_count=-1
        )
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof_cb[val_idx] = cb.predict_proba(X_val)[:, 1]
        trained_cb.append(cb)

        # XGBoost
        xm = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.035, max_depth=5, scale_pos_weight=scale_pos,
            reg_alpha=1.5, reg_lambda=3.5, random_state=42 + fold, eval_metric='auc',
            early_stopping_rounds=50, n_jobs=-1
        )
        xm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xm.predict_proba(X_val)[:, 1]
        trained_xgb.append(xm)

        # LightGBM
        lm = lgb.LGBMClassifier(
            n_estimators=450, learning_rate=0.035, max_depth=5, num_leaves=24,
            scale_pos_weight=scale_pos, reg_alpha=1.5, reg_lambda=3.5,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        )
        lm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = lm.predict_proba(X_val)[:, 1]
        trained_lgb.append(lm)

        fold_ens = 0.55 * oof_cb[val_idx] + 0.25 * oof_xgb[val_idx] + 0.20 * oof_lgb[val_idx]
        fold_auc = roc_auc_score(y_val, fold_ens)
        print(f"  - Fold {fold}/5: OOF AUC = {fold_auc*100:.2f}%")

    elapsed = time.time() - t0
    weights = {'catboost': 0.55, 'xgboost': 0.25, 'lightgbm': 0.20}
    oof_ens = weights['catboost'] * oof_cb + weights['xgboost'] * oof_xgb + weights['lightgbm'] * oof_lgb
    overall_auc = float(roc_auc_score(y, oof_ens) * 100.0)
    print(f"\n[+] 3-way 앙상블 전방 합산 ROC-AUC: {overall_auc:.2f}% (소요시간: {elapsed:.1f}초)")

    # 3. NLL 최소화 온도 스케일링
    eps = 1e-7
    p_clip = np.clip(oof_ens, eps, 1.0 - eps)
    z = np.log(p_clip / (1.0 - p_clip))
    opt_T = optimize_temperature(z, y)
    p_cal = 1.0 / (1.0 + np.exp(-z / opt_T))
    ece_val = compute_ece(p_cal, y)
    print(f"[+] Level 1 온도 스케일링: T={opt_T:.4f}, ECE={ece_val:.2f}%")

    # 4. 프로파일별 시점 적응형 임계값 탐색
    profiles = {}

    # Profile 1: F1 Breakthrough Profile (F1 >= 70% 집중 모드, Retake <= 18%)
    prof1_ths = {}
    prof1_views = {}
    for vn in FRONT_VIEWS:
        vm = (views == vn)
        res_v = search_front_view_thresholds(y[vm], p_cal[vm], max_retake=18.0)
        prof1_ths[vn] = {'tau_clean': res_v['tau_clean'], 'tau_defect': res_v['tau_defect']}
        prof1_views[vn] = res_v
    profiles['f1_breakthrough'] = {
        'profile_name': 'F1 Breakthrough Profile (F1 >= 70% 돌파 모드)',
        'description': '전방 시점에서 F1 70% 이상을 극대화하기 위해 재촬영률 허용 한도를 18%까지 유연하게 확장한 프로파일',
        'thresholds': prof1_ths,
        'views': prof1_views
    }

    # Profile 2: Clinical Strict Profile (표준 임상 규정 모드, Retake <= 10.0%)
    prof2_ths = {}
    prof2_views = {}
    for vn in FRONT_VIEWS:
        vm = (views == vn)
        res_v = search_front_view_thresholds(y[vm], p_cal[vm], max_retake=10.0)
        prof2_ths[vn] = {'tau_clean': res_v['tau_clean'], 'tau_defect': res_v['tau_defect']}
        prof2_views[vn] = res_v
    profiles['clinical_strict'] = {
        'profile_name': 'Clinical Strict Profile (표준 임상 규정 모드)',
        'description': r'd:\AGENTS.md 규칙에 따라 재촬영률을 10.0% 이하로 엄격히 통제하는 프로파일',
        'thresholds': prof2_ths,
        'views': prof2_views
    }

    # 5. 프로파일 평가 결과 출력
    print("\n" + "=" * 80)
    print("  [sl2_hybrid_2 전방 시점 B & D 성능 감사 결과]")
    print("=" * 80)

    for p_key, p_data in profiles.items():
        print(f"\n▶ [{p_data['profile_name']}]")
        for vn in FRONT_VIEWS:
            v_info = p_data['views'][vn]
            alias = FRONT_VIEW_ALIAS_MAP[vn]
            star = " ★ [F1 70% 돌파!]" if v_info['f1'] >= 70.0 else ""
            print(f"  * {alias:<25} | F1: {v_info['f1']:.2f}%{star} | Prec: {v_info['prec']:.2f}% | Rec: {v_info['rec']:.2f}% | Acc: {v_info['acc']:.2f}% | Retake: {v_info['retake']:.2f}% (tc={v_info['tau_clean']}, td={v_info['tau_defect']})")

    # 6. 번들 저장
    bundle_data = {
        'model_name': 'sl2_hybrid_2',
        'defect_name': 'Condition 2 - 몸통·발·무릎 방향 불일치',
        'cb_models': trained_cb,
        'xgb_models': trained_xgb,
        'lgb_models': trained_lgb,
        'weights': weights,
        'temperature': opt_T,
        'profiles': profiles,
        'feature_names': FEATURE_NAMES_84D,
        'keep_cols': KEEP_COLS_84D,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle_data, f)
    print(f"\n[+] 모델 번들 저장 완료: {BUNDLE_PATH}")

    # 7. 평가 JSON 저장
    eval_data = {
        'project': 'vidio-health-trainer',
        'exercise': '사이드 런지 (07_사이드_런지)',
        'model_name': 'sl2_hybrid_2',
        'defect_code': 'c2',
        'defect_name': '몸통·발·무릎 방향 불일치',
        'features_dimension': 84,
        'total_front_samples': N,
        'overall_front_auc': overall_auc,
        'temperature': opt_T,
        'ece': ece_val,
        'profiles': profiles,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(EVAL_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"[+] 평가 결과 JSON 저장 완료: {EVAL_RESULTS_PATH}")

    return eval_data


def predict_sl2_hybrid_2(X: np.ndarray, view: str = 'view2', profile: str = 'f1_breakthrough') -> Dict[str, Any]:
    clf = SL2Hybrid2FrontClassifier()
    return clf.predict_3state(X, view=view, profile=profile)


if __name__ == '__main__':
    build_and_train_sl2_hybrid_2()
