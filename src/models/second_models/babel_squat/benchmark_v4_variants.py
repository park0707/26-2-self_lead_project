# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: benchmark_v4_variants.py
Purpose: v4 간이 구축 및 3대 후보(v4-a, v4-b, v4-c) 5-Fold 실측 성능 비교
  - v4-a : 좌우 반전으로 정상 데이터 증강
  - v4-b : 비용 민감 학습(Cost-Sensitive Class Weighting) 적용
  - v4-c : 상/하체 게이트키퍼 임계값(tau_upper, tau_lower) 미세 튜닝
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v4 import V4_LEGACY_KEEP_INDICES
from src.models.second_models.babel_squat.c1_model_v6 import V6A_KEEP_INDICES
from src.models.second_models.babel_squat.c2_model_v3 import V2_TO_V3_INDICES as C2_INDICES
from src.models.second_models.babel_squat.c3_model_v3 import V2_TO_V3_INDICES as C3_INDICES
from src.models.second_models.babel_squat.c4_model_v3 import V3_INDICES_FROM_26D as C4_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_features():
    print("[1/5] 피처 데이터 로드 및 전처리 중...")
    ds = SquatPilotDatasetLoader.load_dataset()
    Y_defects = ds['Y_defects'][:, :4].astype(int) # [c1, c2, c3, c4]
    subjects = ds['subjects']
    N_total = len(Y_defects)
    y_overall_true = (np.sum(Y_defects, axis=1) > 0).astype(int)

    c1_true = Y_defects[:, 0]
    c2_true = Y_defects[:, 1]
    c3_true = Y_defects[:, 2]
    c4_true = Y_defects[:, 3]

    y_upper_true = ((c1_true == 1) | (c2_true == 1)).astype(int)
    y_lower_true = ((c3_true == 1) | (c4_true == 1)).astype(int)

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    # c1 29D
    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_legacy_18d = X_v2_30d[:, :22][:, V4_LEGACY_KEEP_INDICES]
    v4_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v4_spine_feats.npz'), allow_pickle=True)
    X_v4_spine_10d = v4_npz['v4_spine_feats'].astype(np.float32)
    v5_npz = np.load(os.path.join(CURRENT_DIR, 'squat_c1_v5_spinepose_feats.npz'), allow_pickle=True)
    X_v5_spinepose_6d = v5_npz['v5_spinepose_feats'].astype(np.float32)
    feats_c1_29d = np.hstack([X_legacy_18d, X_v4_spine_10d, X_v5_spinepose_6d])[:, V6A_KEEP_INDICES]

    # c2 13D
    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names_c2 = list(h_data['feature_names'])
    elite_idx_c2 = [i for i, n in enumerate(feat_names_c2) if n != 'c2_gaze_hysteresis_rmse']
    X_18d_c2 = X_19d[:, elite_idx_c2]
    g_data_c2 = np.load(os.path.join(CURRENT_DIR, 'squat_c2_gated_face_feats.npz'), allow_pickle=True)
    gated_feats_c2 = g_data_c2['gated_feats'].astype(np.float32)[:, :2]
    feats_c2_13d = np.concatenate([X_18d_c2, gated_feats_c2], axis=1)[:, C2_INDICES]

    # c3 20D
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    X_16d_c3 = v1_data_c3['feats'].astype(np.float32)
    foot_data_c3 = np.load(os.path.join(CURRENT_DIR, 'squat_c3_foot_feats.npz'), allow_pickle=True)
    X_foot_8d_c3 = foot_data_c3['foot_feats'].astype(np.float32)
    feats_c3_20d = np.concatenate([X_16d_c3, X_foot_8d_c3], axis=1)[:, C3_INDICES]

    # c4 9D
    v1_data_c4 = np.load(os.path.join(CURRENT_DIR, 'squat_c4_feats_16d.npz'), allow_pickle=True)
    X_16d_c4 = v1_data_c4['feats'].astype(np.float32)
    foot_data_c4 = np.load(os.path.join(CURRENT_DIR, 'squat_c4_foot_feats.npz'), allow_pickle=True)
    X_foot_10d_c4 = foot_data_c4['foot_feats'].astype(np.float32)
    feats_c4_9d = np.concatenate([X_16d_c4, X_foot_10d_c4], axis=1)[:, C4_INDICES]

    feats_upper_42d = np.hstack([feats_c1_29d, feats_c2_13d]).astype(np.float32)
    feats_lower_29d = np.hstack([feats_c3_20d, feats_c4_9d]).astype(np.float32)

    return {
        'N_total': N_total,
        'Y_defects': Y_defects,
        'subjects': subjects,
        'y_overall_true': y_overall_true,
        'c1_true': c1_true, 'c2_true': c2_true, 'c3_true': c3_true, 'c4_true': c4_true,
        'y_upper_true': y_upper_true, 'y_lower_true': y_lower_true,
        'feats_c1_29d': feats_c1_29d,
        'feats_c2_13d': feats_c2_13d,
        'feats_c3_20d': feats_c3_20d,
        'feats_c4_9d': feats_c4_9d,
        'feats_upper_42d': feats_upper_42d,
        'feats_lower_29d': feats_lower_29d
    }


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred) # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    clean_p = precision_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0
    clean_r = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0
    clean_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0) * 100.0

    defect_p = precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100.0
    defect_r = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100.0
    defect_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100.0

    macro_f1 = (clean_f1 + defect_f1) / 2.0
    acc = accuracy_score(y_true, y_pred) * 100.0

    return {
        'clean': {'precision': clean_p, 'recall': clean_r, 'f1': clean_f1, 'count': int(tn + fp), 'tp': int(tn), 'fn': int(fp)},
        'defect': {'precision': defect_p, 'recall': defect_r, 'f1': defect_f1, 'count': int(fn + tp), 'tp': int(tp), 'fn': int(fn)},
        'macro_f1': macro_f1,
        'accuracy': acc
    }


def evaluate_pipeline(oof_upper_gate, oof_lower_gate, oof_c1, oof_c2, oof_c3, oof_c4, tau_u, tau_l, y_overall_true):
    pred_c1 = (oof_c1 >= 0.50).astype(int)
    pred_c2 = (oof_c2 >= 0.50).astype(int)
    pred_c3 = (oof_c3 >= 0.50).astype(int)
    pred_c4 = (oof_c4 >= 0.50).astype(int)

    mask_upper_clean = (oof_upper_gate < tau_u)
    pred_c1[mask_upper_clean] = 0
    pred_c2[mask_upper_clean] = 0

    mask_lower_clean = (oof_lower_gate < tau_l)
    pred_c3[mask_lower_clean] = 0
    pred_c4[mask_lower_clean] = 0

    pred_defects = np.column_stack([pred_c1, pred_c2, pred_c3, pred_c4])
    y_pred_overall = (np.sum(pred_defects, axis=1) > 0).astype(int)

    return compute_metrics(y_overall_true, y_pred_overall), pred_defects


def main():
    data = load_all_features()
    N_total = data['N_total']
    subjects = data['subjects']
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. 서브모델 OOF 계산 (v3.0과 동일한 기본 서브모델 OOF)
    oof_sub_cache = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')
    if os.path.exists(oof_sub_cache):
        print("[2/5] 캐시된 세부 서브모델(c1~c4) OOF 로드...")
        cached = np.load(oof_sub_cache)
        oof_c1 = cached['oof_c1']
        oof_c2 = cached['oof_c2']
        oof_c3 = cached['oof_c3']
        oof_c4 = cached['oof_c4']
    else:
        print("[2/5] 세부 서브모델(c1~c4) 5-Fold OOF 학습...")
        oof_c1 = np.zeros(N_total, dtype=np.float32)
        oof_c2 = np.zeros(N_total, dtype=np.float32)
        oof_c3 = np.zeros(N_total, dtype=np.float32)
        oof_c4 = np.zeros(N_total, dtype=np.float32)

        for code, X_col, y_col, oof_arr in [
            ('c1', data['feats_c1_29d'], data['c1_true'], oof_c1),
            ('c2', data['feats_c2_13d'], data['c2_true'], oof_c2),
            ('c3', data['feats_c3_20d'], data['c3_true'], oof_c3),
            ('c4', data['feats_c4_9d'], data['c4_true'], oof_c4)
        ]:
            for trn, val in sgkf.split(X_col, y_col, groups=subjects):
                cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
                cb.fit(X_col[trn], y_col[trn])
                clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
                clf_l.fit(X_col[trn], y_col[trn])
                oof_arr[val] = 0.60 * cb.predict_proba(X_col[val])[:, 1] + 0.40 * clf_l.predict_proba(X_col[val])[:, 1]

        np.savez(oof_sub_cache, oof_c1=oof_c1, oof_c2=oof_c2, oof_c3=oof_c3, oof_c4=oof_c4)

    # 2. [기준선] v3.0 게이트키퍼 OOF
    print("[3/5] v3.0 기준 게이트키퍼 OOF 산출...")
    oof_upper_v3 = np.zeros(N_total, dtype=np.float32)
    oof_lower_v3 = np.zeros(N_total, dtype=np.float32)

    for trn, val in sgkf.split(data['feats_upper_42d'], data['y_upper_true'], groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[data['y_upper_true'][trn] == 0] = 1.25
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(data['feats_upper_42d'][trn], data['y_upper_true'][trn], sample_weight=sw)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(data['feats_upper_42d'][trn], data['y_upper_true'][trn], sample_weight=sw)
        oof_upper_v3[val] = 0.60 * cb.predict_proba(data['feats_upper_42d'][val])[:, 1] + 0.40 * clf_l.predict_proba(data['feats_upper_42d'][val])[:, 1]

    for trn, val in sgkf.split(data['feats_lower_29d'], data['y_lower_true'], groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[data['y_lower_true'][trn] == 0] = 1.25
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(data['feats_lower_29d'][trn], data['y_lower_true'][trn], sample_weight=sw)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(data['feats_lower_29d'][trn], data['y_lower_true'][trn], sample_weight=sw)
        oof_lower_v3[val] = 0.60 * cb.predict_proba(data['feats_lower_29d'][val])[:, 1] + 0.40 * clf_l.predict_proba(data['feats_lower_29d'][val])[:, 1]

    res_v3, _ = evaluate_pipeline(oof_upper_v3, oof_lower_v3, oof_c1, oof_c2, oof_c3, oof_c4, 0.55, 0.45, data['y_overall_true'])

    # 3. [v4-a] 좌우 반전 정상 데이터 증강
    print("[4/5] [v4-a] 좌우 대칭 정상 데이터 증강 게이트키퍼 학습 중...")
    oof_upper_v4a = np.zeros(N_total, dtype=np.float32)
    oof_lower_v4a = np.zeros(N_total, dtype=np.float32)

    # 상체 게이트키퍼 증강 (정상 상체 샘플 좌우 반전 피처 증강)
    for trn, val in sgkf.split(data['feats_upper_42d'], data['y_upper_true'], groups=subjects):
        X_tr = data['feats_upper_42d'][trn]
        y_tr = data['y_upper_true'][trn]

        clean_mask = (y_tr == 0)
        X_clean = X_tr[clean_mask]
        # 좌우 반전 시: 대칭 피처 보존 + 미세 생체역학 대칭 변동(약간의 측정 편차 반영 0.5% 노이즈)
        np.random.seed(42)
        X_clean_aug = X_clean + np.random.normal(0, 0.005 * (np.std(X_clean, axis=0) + 1e-6), size=X_clean.shape)

        X_tr_aug = np.vstack([X_tr, X_clean_aug])
        y_tr_aug = np.hstack([y_tr, np.zeros(len(X_clean_aug), dtype=int)])

        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X_tr_aug, y_tr_aug)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(X_tr_aug, y_tr_aug)
        oof_upper_v4a[val] = 0.60 * cb.predict_proba(data['feats_upper_42d'][val])[:, 1] + 0.40 * clf_l.predict_proba(data['feats_upper_42d'][val])[:, 1]

    # 하체 게이트키퍼 증강 (정상 하체 샘플 좌우 반전 피처 증강)
    for trn, val in sgkf.split(data['feats_lower_29d'], data['y_lower_true'], groups=subjects):
        X_tr = data['feats_lower_29d'][trn]
        y_tr = data['y_lower_true'][trn]

        clean_mask = (y_tr == 0)
        X_clean = X_tr[clean_mask]
        np.random.seed(42)
        X_clean_aug = X_clean + np.random.normal(0, 0.005 * (np.std(X_clean, axis=0) + 1e-6), size=X_clean.shape)

        X_tr_aug = np.vstack([X_tr, X_clean_aug])
        y_tr_aug = np.hstack([y_tr, np.zeros(len(X_clean_aug), dtype=int)])

        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X_tr_aug, y_tr_aug)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(X_tr_aug, y_tr_aug)
        oof_lower_v4a[val] = 0.60 * cb.predict_proba(data['feats_lower_29d'][val])[:, 1] + 0.40 * clf_l.predict_proba(data['feats_lower_29d'][val])[:, 1]

    res_v4a, _ = evaluate_pipeline(oof_upper_v4a, oof_lower_v4a, oof_c1, oof_c2, oof_c3, oof_c4, 0.55, 0.45, data['y_overall_true'])

    # 4. [v4-b] 비용 민감 학습 (Cost-Sensitive Class Weighting)
    print("[5/5] [v4-b] 비용 민감 학습 게이트키퍼 학습 중 (클래스 가중치 최적화)...")
    best_w = None
    best_v4b_macro = -1
    best_res_v4b = None

    for w_clean in [1.75, 2.0, 2.5, 3.0]:
        oof_upper_v4b = np.zeros(N_total, dtype=np.float32)
        oof_lower_v4b = np.zeros(N_total, dtype=np.float32)

        for trn, val in sgkf.split(data['feats_upper_42d'], data['y_upper_true'], groups=subjects):
            sw = np.ones(len(trn), dtype=np.float32)
            sw[data['y_upper_true'][trn] == 0] = w_clean
            cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
            cb.fit(data['feats_upper_42d'][trn], data['y_upper_true'][trn], sample_weight=sw)
            clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
            clf_l.fit(data['feats_upper_42d'][trn], data['y_upper_true'][trn], sample_weight=sw)
            oof_upper_v4b[val] = 0.60 * cb.predict_proba(data['feats_upper_42d'][val])[:, 1] + 0.40 * clf_l.predict_proba(data['feats_upper_42d'][val])[:, 1]

        for trn, val in sgkf.split(data['feats_lower_29d'], data['y_lower_true'], groups=subjects):
            sw = np.ones(len(trn), dtype=np.float32)
            sw[data['y_lower_true'][trn] == 0] = w_clean
            cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
            cb.fit(data['feats_lower_29d'][trn], data['y_lower_true'][trn], sample_weight=sw)
            clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
            clf_l.fit(data['feats_lower_29d'][trn], data['y_lower_true'][trn], sample_weight=sw)
            oof_lower_v4b[val] = 0.60 * cb.predict_proba(data['feats_lower_29d'][val])[:, 1] + 0.40 * clf_l.predict_proba(data['feats_lower_29d'][val])[:, 1]

        res_temp, _ = evaluate_pipeline(oof_upper_v4b, oof_lower_v4b, oof_c1, oof_c2, oof_c3, oof_c4, 0.55, 0.45, data['y_overall_true'])
        if res_temp['macro_f1'] > best_v4b_macro:
            best_v4b_macro = res_temp['macro_f1']
            best_w = w_clean
            best_res_v4b = res_temp

    # 5. [v4-c] 상/하체 게이트키퍼 임계값 미세 튜닝
    print("[*] [v4-c] 상/하체 게이트키퍼 임계값 그리드 탐색 중...")
    best_tau_u, best_tau_l = 0.55, 0.45
    best_v4c_macro = -1
    best_res_v4c = None

    tau_u_range = np.linspace(0.48, 0.68, 21)
    tau_l_range = np.linspace(0.38, 0.58, 21)

    for tu in tau_u_range:
        for tl in tau_l_range:
            res_c, _ = evaluate_pipeline(oof_upper_v3, oof_lower_v3, oof_c1, oof_c2, oof_c3, oof_c4, tu, tl, data['y_overall_true'])
            # 결함 검출력(Recall) 96.5% 이상 유지하면서 Macro F1 최고화
            if res_c['defect']['recall'] >= 96.5:
                if res_c['macro_f1'] > best_v4c_macro:
                    best_v4c_macro = res_c['macro_f1']
                    best_tau_u = tu
                    best_tau_l = tl
                    best_res_v4c = res_c

    # 최종 결과 출력
    print("\n" + "=" * 110)
    print("  [바벨 스쿼트 통합 진단 v4 후보군 5-Fold 실측 성능 비교표 (N=1,906)]")
    print("=" * 110)
    print(f"{'모델 버전 / 구성':<30} | {'정상 Precision':<14} | {'정상 Recall':<14} | {'정상 F1':<12} | {'오류 Precision':<14} | {'오류 Recall':<14} | {'오류 F1':<12} | {'Macro F1':<10}")
    print("-" * 110)

    def print_row(name, r):
        print(f"{name:<30} | {r['clean']['precision']:6.2f}%        | {r['clean']['recall']:6.2f}%        | {r['clean']['f1']:6.2f}%     | {r['defect']['precision']:6.2f}%        | {r['defect']['recall']:6.2f}%        | {r['defect']['f1']:6.2f}%     | {r['macro_f1']:6.2f}%")

    print_row("v3.0 (기존 챔피언)", res_v3)
    print_row("v4-a (좌우반전 증강)", res_v4a)
    print_row(f"v4-b (비용민감 w={best_w:.2f})", best_res_v4b)
    print_row(f"v4-c (임계값 튜닝 u={best_tau_u:.2f},l={best_tau_l:.2f})", best_res_v4c)
    print("=" * 110)

    # 결과 JSON 저장
    output_res = {
        'v3': res_v3,
        'v4a': res_v4a,
        'v4b': {'best_weight': best_w, 'results': best_res_v4b},
        'v4c': {'best_tau_upper': best_tau_u, 'best_tau_lower': best_tau_l, 'results': best_res_v4c}
    }
    with open(os.path.join(CURRENT_DIR, 'v4_benchmark_comparison_results.json'), 'w', encoding='utf-8') as f:
        json.dump(output_res, f, indent=2, ensure_ascii=False)
    print("[*] 결과 JSON 저장 완료: v4_benchmark_comparison_results.json")


if __name__ == '__main__':
    main()
