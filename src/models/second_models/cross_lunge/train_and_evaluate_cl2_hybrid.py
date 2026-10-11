# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 크로스 런지 (Cross Lunge, 08_크로스_런지)
Defect: Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부 (Trunk-LeadFoot-LeadKnee Directional Alignment Defect)
Script: train_and_evaluate_cl2_hybrid.py
Purpose: MediaPipe 3D (51D) + RTMPose Halpe26 (29D) + 크로스모달 상호작용 (6D) = 86D 하이브리드 파이프라인
         공식 5-Fold StratifiedGroupKFold 앙상블 학습, 온도 스케일링, 3-State 감사 및 번들 배포
"""

import os
import sys
import time
import json
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, log_loss
)
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from cl2_hybrid_1 import (
    CL2_HYBRID_1_FEATURE_NAMES,
    normalize_2d_vec
)

MP_CACHE_PATH = r"D:\Data\피처_데이터\cross_lunge_c2_v7_all_views_51d_cache.pkl"
RTM_CACHE_PATH = r"D:\Data\피처_데이터\cross_lunge_rtmp_optimized_features_cache.npz"
HYBRID_CACHE_PATH = r"D:\Data\피처_데이터\cross_lunge_c2_hybrid_86d_cache.npz"

MP_NPZ_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"
RTM_NPZ_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\08_크로스_런지"

BUNDLE_OUT = os.path.join(CURRENT_DIR, "cl2_hybrid_1_bundle.pkl")
EVAL_OUT = os.path.join(CURRENT_DIR, "cl2_hybrid_1_evaluation_results.json")

VIEW_CODE_TO_NAME = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)',
}


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (probs == bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc = np.mean(labels[in_bin])
            conf = np.mean(probs[in_bin])
            ece += np.abs(conf - acc) * prop_in_bin
    return float(ece)


def build_or_load_hybrid_dataset(force_rebuild: bool = False):
    if os.path.exists(HYBRID_CACHE_PATH) and not force_rebuild:
        print(f"[*] 하이브리드 캐시 로드 중: {HYBRID_CACHE_PATH}")
        cache = np.load(HYBRID_CACHE_PATH, allow_pickle=True)
        return cache['X'], cache['y'], cache['groups'], cache['views'], list(cache['seq_ids'])

    print("[*] 하이브리드 데이터셋 새로 구축 시작...")
    # 1. MediaPipe 51D 로드
    with open(MP_CACHE_PATH, 'rb') as f:
        mp_data = pickle.load(f)
    X_mp_51d = mp_data['X'] # (1025, 51)
    y = mp_data['y']
    groups = mp_data['groups']
    views = mp_data['views']
    seq_ids = [f"{c}_{v}" for c, v in zip(mp_data['clip_names'], mp_data['views'])]

    # 2. RTMPose 29D 로드 (82D 중 인덱스 48:77 에 해당하는 29D)
    rtm_data = np.load(RTM_CACHE_PATH)
    X_rtm_29d = rtm_data['X_c2'][:, 48:77] # (1025, 29)

    # 3. 크로스모달 6D 피처 고속 추출
    print("[*] 크로스모달 상호작용 피처(6D) 고속 추출 중...")
    X_cross_6d = []
    for sid in seq_ids:
        mp_path = os.path.join(MP_NPZ_DIR, f"{sid}.npz")
        rtm_path = os.path.join(RTM_NPZ_DIR, f"{sid}.npz")

        d_mp = np.load(mp_path)['kps_33_3d_64f']
        d_rtm = np.load(rtm_path)['kps_26_64f']

        kps_2d = d_rtm[:, :, :2]
        conf = d_rtm[:, :, 2]
        l_knee_y = kps_2d[:, 13, 1]
        r_knee_y = kps_2d[:, 14, 1]
        t_l = int(np.argmax(l_knee_y))
        t_r = int(np.argmax(r_knee_y))
        is_left_lead_rtm = (l_knee_y[t_l] >= r_knee_y[t_r])
        t_bot_rtm = t_l if is_left_lead_rtm else t_r
        t_bot_rtm = max(4, min(59, t_bot_rtm))

        win = slice(max(0, t_bot_rtm - 3), min(64, t_bot_rtm + 4))
        k_bot = np.mean(kps_2d[win], axis=0)

        pelvis_bot = k_bot[19] if conf[t_bot_rtm, 19] > 0.3 else (k_bot[11] + k_bot[12]) * 0.5
        neck_bot = k_bot[18] if conf[t_bot_rtm, 18] > 0.3 else (k_bot[5] + k_bot[6]) * 0.5
        trunk_len_rtm = np.linalg.norm(neck_bot - pelvis_bot) + 1e-6

        toe_big_l = 20 if is_left_lead_rtm else 21
        toe_sml_l = 22 if is_left_lead_rtm else 23
        heel_l = 24 if is_left_lead_rtm else 25
        knee_l = 13 if is_left_lead_rtm else 14
        mid_toe_bot = (k_bot[toe_big_l] + k_bot[toe_sml_l]) * 0.5
        v_foot_rtm = normalize_2d_vec(mid_toe_bot - k_bot[heel_l])

        # MP
        sh_mid_mp = (d_mp[t_bot_rtm, 11, :2] + d_mp[t_bot_rtm, 12, :2]) * 0.5
        hip_mid_mp = (d_mp[t_bot_rtm, 23, :2] + d_mp[t_bot_rtm, 24, :2]) * 0.5
        v_trunk_mp = normalize_2d_vec(sh_mid_mp - hip_mid_mp)

        # 앞다리 MP
        l_knee_y_mp = d_mp[:, 25, 1]
        r_knee_y_mp = d_mp[:, 26, 1]
        is_left_lead_mp = (l_knee_y_mp[int(np.argmax(l_knee_y_mp))] >= r_knee_y_mp[int(np.argmax(r_knee_y_mp))])
        hip_l_mp = 23 if is_left_lead_mp else 24
        knee_l_mp = 25 if is_left_lead_mp else 26
        v_thigh_mp = normalize_2d_vec(d_mp[t_bot_rtm, knee_l_mp, :2] - d_mp[t_bot_rtm, hip_l_mp, :2])

        c1 = float(np.dot(v_trunk_mp, v_foot_rtm))
        c2 = float(v_trunk_mp[0]*v_foot_rtm[1] - v_trunk_mp[1]*v_foot_rtm[0])
        c3 = float(np.dot(v_thigh_mp, v_foot_rtm))
        c4 = float(v_thigh_mp[0]*v_foot_rtm[1] - v_thigh_mp[1]*v_foot_rtm[0])

        mp_knee_x_norm = float(d_mp[t_bot_rtm, knee_l_mp, 0] - hip_mid_mp[0])
        rtm_knee_x_norm = float(k_bot[knee_l, 0] - pelvis_bot[0]) / trunk_len_rtm
        c5 = float(mp_knee_x_norm - rtm_knee_x_norm)
        c6 = 1.0 if (is_left_lead_mp == is_left_lead_rtm) else 0.0

        X_cross_6d.append([c1, c2, c3, c4, c5, c6])

    X_cross_6d = np.array(X_cross_6d, dtype=np.float32)

    X_hybrid = np.concatenate([X_mp_51d, X_rtm_29d, X_cross_6d], axis=1)
    X_hybrid = np.nan_to_num(X_hybrid, nan=0.0)

    print(f"[*] 하이브리드 피처 구축 완료: {X_hybrid.shape}")
    os.makedirs(os.path.dirname(HYBRID_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        HYBRID_CACHE_PATH,
        X=X_hybrid, y=y, groups=groups, views=views, seq_ids=seq_ids
    )
    return X_hybrid, y, groups, views, seq_ids


def main():
    print("=" * 80)
    print("vidio-health-trainer: cl2_hybrid_1 앙상블 학습 및 최적화 파이프라인")
    print("=" * 80)

    X, y, groups, views, seq_ids = build_or_load_hybrid_dataset()
    N, D = X.shape
    assert D == 86, f"피처 차원이 86이어야 합니다: 현재 {D}"

    neg_cnt = np.sum(y == 0)
    pos_cnt = np.sum(y == 1)
    spw = float(neg_cnt / max(1, pos_cnt))
    print(f"[*] 총 시퀀스: {N}, 정상: {neg_cnt}, 결함: {pos_cnt}, scale_pos_weight: {spw:.2f}")

    # 1. 5-Fold StratifiedGroupKFold 교차검증
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.035,
        'num_leaves': 31,
        'max_depth': 6,
        'scale_pos_weight': spw,
        'subsample': 0.8,
        'colsample_bytree': 0.75,
        'reg_lambda': 2.5,
        'reg_alpha': 0.5,
        'min_child_samples': 15,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 1
    }

    cat_params = {
        'iterations': 550,
        'learning_rate': 0.035,
        'depth': 6,
        'scale_pos_weight': spw,
        'l2_leaf_reg': 3.0,
        'rsm': 0.75,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50,
        'thread_count': 1
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.035,
        'max_depth': 6,
        'scale_pos_weight': spw,
        'subsample': 0.8,
        'colsample_bytree': 0.75,
        'reg_lambda': 2.5,
        'reg_alpha': 0.5,
        'random_state': 42,
        'n_jobs': 1
    }

    models_cat = []
    models_lgb = []
    models_xgb = []

    oof_preds_cat = np.zeros(N, dtype=np.float32)
    oof_preds_lgb = np.zeros(N, dtype=np.float32)
    oof_preds_xgb = np.zeros(N, dtype=np.float32)

    print("[*] 5-Fold 학습 시작...")
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # LightGBM
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        m_lgb = lgb.train(lgb_params, trn_data, valid_sets=[val_data], num_boost_round=600,
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)
        models_lgb.append(m_lgb)
        oof_preds_lgb[val_idx] = p_lgb

        # CatBoost
        m_cat = CatBoostClassifier(**cat_params)
        m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        p_cat = m_cat.predict_proba(X_va)[:, 1]
        models_cat.append(m_cat)
        oof_preds_cat[val_idx] = p_cat

        # XGBoost
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        m_xgb = xgb.train(xgb_params, dtrain, num_boost_round=600,
                          evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        p_xgb = m_xgb.predict(dval)
        models_xgb.append(m_xgb)
        oof_preds_xgb[val_idx] = p_xgb

        print(f"    - Fold {fold+1}/5 완료: AUC (Cat: {roc_auc_score(y_va, p_cat):.4f}, LGB: {roc_auc_score(y_va, p_lgb):.4f}, XGB: {roc_auc_score(y_va, p_xgb):.4f})")

    # 2. 최적 앙상블 가중치 탐색 (Nelder-Mead NLL 최소화)
    def obj_weights(w):
        w = np.maximum(0, w)
        s = np.sum(w)
        if s == 0:
            return 999.0
        w = w / s
        p = w[0]*oof_preds_cat + w[1]*oof_preds_lgb + w[2]*oof_preds_xgb
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return log_loss(y, p)

    res_w = minimize(obj_weights, [0.34, 0.33, 0.33], method='Nelder-Mead')
    opt_w = np.maximum(0, res_w.x)
    opt_w = (opt_w / np.sum(opt_w)).tolist()
    print(f"[*] 최적 앙상블 가중치: Cat={opt_w[0]:.4f}, LGB={opt_w[1]:.4f}, XGB={opt_w[2]:.4f}")

    oof_blend = opt_w[0]*oof_preds_cat + opt_w[1]*oof_preds_lgb + opt_w[2]*oof_preds_xgb

    # 3. 온도 스케일링 (Temperature Scaling) 최적화
    def obj_temp(t):
        temp = max(0.05, float(t[0]))
        eps = 1e-7
        p = np.clip(oof_blend, eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p))
        p_cal = 1.0 / (1.0 + np.exp(-logit / temp))
        p_cal = np.clip(p_cal, eps, 1.0 - eps)
        return log_loss(y, p_cal)

    res_t = minimize(obj_temp, [1.0], method='Nelder-Mead')
    best_temp = float(max(0.1, res_t.x[0]))

    eps = 1e-7
    p_b = np.clip(oof_blend, eps, 1.0 - eps)
    logit = np.log(p_b / (1.0 - p_b))
    oof_calibrated = 1.0 / (1.0 + np.exp(-logit / best_temp))
    oof_calibrated = np.clip(oof_calibrated, 0.0, 1.0)

    ece_pre = compute_ece(oof_blend, y) * 100
    ece_post = compute_ece(oof_calibrated, y) * 100
    print(f"[*] 온도 스케일링(T={best_temp:.4f}): ECE {ece_pre:.2f}% -> {ece_post:.2f}%")

    # 4. 2-State 성능 평가
    auc_2s = roc_auc_score(y, oof_calibrated)
    p_bin_2s = (oof_calibrated >= 0.5).astype(int)
    f1_2s = f1_score(y, p_bin_2s)
    acc_2s = accuracy_score(y, p_bin_2s)
    prec_2s = precision_score(y, p_bin_2s)
    rec_2s = recall_score(y, p_bin_2s)

    print(f"\n[2-State 강제 판정 Baseline]")
    print(f"  - ROC-AUC: {auc_2s*100:.2f}%")
    print(f"  - F1-Score: {f1_2s*100:.2f}%")
    print(f"  - Accuracy: {acc_2s*100:.2f}%")
    print(f"  - Precision: {prec_2s*100:.2f}%, Recall: {rec_2s*100:.2f}%")

    # 5. AGENTS.md 공식 5대 시점별 적응형 3-State 신뢰도 게이팅 최적화
    unique_views = ['view5', 'view2', 'view3', 'view4', 'view1']
    view_thresholds = {}
    view_audits = {}

    sel_mask_all = np.zeros(N, dtype=bool)
    pred_3s_all = np.zeros(N, dtype=int)

    for v_code in unique_views:
        idx_v = np.where(views == v_code)[0]
        p_v = oof_calibrated[idx_v]
        y_v = y[idx_v]

        best_f1 = 0.0
        best_tau = (0.45, 0.55)
        best_mask = None

        # 커버리지 >= 90% (보류율 <= 10%) 제약 하에서 F1 최대화
        for tc in np.linspace(0.20, 0.49, 30):
            for td in np.linspace(0.51, 0.80, 30):
                sel_mask = (p_v <= tc) | (p_v >= td)
                cov = np.mean(sel_mask)
                if cov < 0.90:
                    continue
                pred_bin = (p_v[sel_mask] >= td).astype(int)
                f1_val = f1_score(y_v[sel_mask], pred_bin)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_tau = (float(tc), float(td))
                    best_mask = sel_mask

        if best_mask is None:
            best_tau = (0.49, 0.51)
            best_mask = np.ones(len(y_v), dtype=bool)
            pred_bin = (p_v >= 0.5).astype(int)
            best_f1 = f1_score(y_v, pred_bin)
        else:
            pred_bin = (p_v[best_mask] >= best_tau[1]).astype(int)

        sel_mask_all[idx_v] = best_mask
        pred_3s_all[idx_v[best_mask]] = pred_bin

        v_auc = roc_auc_score(y_v, p_v) * 100
        v_acc = accuracy_score(y_v[best_mask], pred_bin) * 100
        v_prec = precision_score(y_v[best_mask], pred_bin) * 100
        v_rec = recall_score(y_v[best_mask], pred_bin) * 100
        v_cov = np.mean(best_mask) * 100

        view_thresholds[v_code] = {
            'tau_clean': round(best_tau[0], 2),
            'tau_defect': round(best_tau[1], 2)
        }
        view_audits[v_code] = {
            'view_name': VIEW_CODE_TO_NAME[v_code],
            'tau_clean': round(best_tau[0], 2),
            'tau_defect': round(best_tau[1], 2),
            'sel_f1': round(best_f1 * 100, 2),
            'sel_acc': round(v_acc, 2),
            'sel_prec': round(v_prec, 2),
            'sel_rec': round(v_rec, 2),
            'coverage': round(v_cov, 2),
            'retake': round(100.0 - v_cov, 2),
            'roc_auc': round(v_auc, 2)
        }

    overall_sel_f1 = f1_score(y[sel_mask_all], pred_3s_all[sel_mask_all]) * 100
    overall_sel_acc = accuracy_score(y[sel_mask_all], pred_3s_all[sel_mask_all]) * 100
    overall_sel_prec = precision_score(y[sel_mask_all], pred_3s_all[sel_mask_all]) * 100
    overall_sel_rec = recall_score(y[sel_mask_all], pred_3s_all[sel_mask_all]) * 100
    overall_cov = np.mean(sel_mask_all) * 100
    overall_retake = 100.0 - overall_cov

    print(f"\n[3-State 선택적 분류 Audit]")
    print(f"  - 3-State F1-Score: {overall_sel_f1:.2f}%")
    print(f"  - 3-State Accuracy: {overall_sel_acc:.2f}%")
    print(f"  - 3-State Precision: {overall_sel_prec:.2f}%, Recall: {overall_sel_rec:.2f}%")
    print(f"  - 유효 판정 커버리지: {overall_cov:.2f}% (재촬영 보류율: {overall_retake:.2f}%)")

    print("\n[시점별 3-State 상세 지표]")
    for v_code in unique_views:
        va = view_audits[v_code]
        print(f"  {va['view_name']}: 3S F1={va['sel_f1']}%, AUC={va['roc_auc']}%, ACC={va['sel_acc']}%, Retake={va['retake']}% (tau: {va['tau_clean']}, {va['tau_defect']})")

    # 6. 공식 번들 저장
    bundle_data = {
        'model_name': 'cl2_hybrid_1',
        'defect_name': 'Condition 2 - 몸통 앞발 앞무릎 방향 일치 여부',
        'models_cat': models_cat,
        'models_lgb': models_lgb,
        'models_xgb': models_xgb,
        'weights': opt_w,
        'temperature': best_temp,
        'view_thresholds': view_thresholds,
        'feature_names': CL2_HYBRID_1_FEATURE_NAMES,
        'metadata': {
            'feature_dim': 86,
            'n_samples': N,
            'subjects_count': 9,
            'views_count': 5,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics_2state': {
                'f1': round(f1_2s * 100, 2),
                'acc': round(acc_2s * 100, 2),
                'prec': round(prec_2s * 100, 2),
                'rec': round(rec_2s * 100, 2),
                'roc_auc': round(auc_2s * 100, 2)
            },
            'metrics_3state': {
                'sel_f1': round(overall_sel_f1, 2),
                'sel_acc': round(overall_sel_acc, 2),
                'sel_prec': round(overall_sel_prec, 2),
                'sel_rec': round(overall_sel_rec, 2),
                'coverage': round(overall_cov, 2),
                'retake': round(overall_retake, 2)
            }
        }
    }

    with open(BUNDLE_OUT, 'wb') as f:
        pickle.dump(bundle_data, f)
    print(f"\n[*] 번들 저장 완료: {BUNDLE_OUT}")

    # 7. 평가 결과 JSON 저장
    eval_results = {
        'model_name': 'cl2_hybrid_1',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'feature_dim': 86,
        'n_samples': N,
        'weights': opt_w,
        'temperature': best_temp,
        'ece_pre': round(ece_pre, 2),
        'ece_post': round(ece_post, 2),
        'metrics_2state': bundle_data['metadata']['metrics_2state'],
        'metrics_3state': bundle_data['metadata']['metrics_3state'],
        'views_metrics': view_audits
    }

    with open(EVAL_OUT, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"[*] 평가 결과 JSON 저장 완료: {EVAL_OUT}")


if __name__ == '__main__':
    main()
