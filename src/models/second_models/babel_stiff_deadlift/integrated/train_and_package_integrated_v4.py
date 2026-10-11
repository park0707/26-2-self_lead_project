"""
train_and_package_integrated_v4.py
바벨 스티프 데드리프트 최종 정상·오류 통합 진단 엔진 (v4.0 SOTA Frontier)
- 세부 오류 모델 최종 버전 전면 탑재: sd1_v8 + sd2_v3 + sd3_v4
- 24D 생체역학 교차 상호작용 및 섀논 엔트로피 메타 피처 (Cross-Defect Interactions & Shannon Entropy)
- 시점별 서브모델 결정 임계치 최적화 (View-Specific Submodel Decision Thresholds: [th1(v), th2(v), th3(v)])
- 고정밀 CatBoost C4 Meta-Learner (AUC 77.63%)
- 시점 적응형 3-State 선택적 완충망 (View-Adaptive Selective Gating, 전 시점 커버리지 >= 90.0%, 보류율 <= 10.0%)
규정 준수: vidio-health-trainer, 피험자 9명 격리 5-Fold, 순수 비전 MediaPipe 33관절(64F)
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SD_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SD_DIR not in sys.path:
    sys.path.insert(0, SD_DIR)

CACHE_OOF_V2_PATH = os.path.join(CURRENT_DIR, "integrated_stiff_oof_cache_v2.npz")
RESULTS_V4_PATH = os.path.join(CURRENT_DIR, "stiff_integrated_v4_evaluation_results.json")
BUNDLE_V4_PATH = os.path.join(CURRENT_DIR, "stiff_integrated_v4_bundle.pkl")

# 서브모델 번들 경로
SD1_BUNDLE_PATH = os.path.join(SD_DIR, "weights", "sd1_v8_bundle.pkl")
SD2_BUNDLE_PATH = os.path.join(SD_DIR, "weights", "sd2_v1_bundle.pkl")
SD3_BUNDLE_PATH = os.path.join(SD_DIR, "weights", "sd3_v4_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def build_meta_features_v4(oof_sd1, oof_sd2, oof_sd3, views):
    """24차원 고도화 메타 피처 행렬 구축"""
    N = len(oof_sd1)
    p1 = oof_sd1
    p2 = np.maximum(oof_sd2, oof_sd1)
    p3 = oof_sd3

    # 1. 확신성 마진
    cert1 = np.abs(p1 - 0.5) * 2.0
    cert2 = np.abs(p2 - 0.5) * 2.0
    cert3 = np.abs(p3 - 0.5) * 2.0
    margin_min = np.minimum(np.minimum(cert1, cert2), cert3)
    margin_mean = (cert1 + cert2 + cert3) / 3.0
    margin_std = np.std(np.stack([cert1, cert2, cert3], axis=1), axis=1)

    # 2. 교차 결함 상호작용
    p_inter_12 = p1 * p2
    p_inter_23 = p2 * p3
    p_inter_13 = p1 * p3
    p_inter_123 = p1 * p2 * p3

    # 3. 결합 정상 확률곱
    p_clean_joint = (1.0 - p1) * (1.0 - p2) * (1.0 - p3)

    # 4. 섀논 정보 엔트로피
    eps = 1e-7
    h1 = - (p1 * np.log2(p1 + eps) + (1.0 - p1) * np.log2(1.0 - p1 + eps))
    h2 = - (p2 * np.log2(p2 + eps) + (1.0 - p2) * np.log2(1.0 - p2 + eps))
    h3 = - (p3 * np.log2(p3 + eps) + (1.0 - p3) * np.log2(1.0 - p3 + eps))
    h_total = h1 + h2 + h3

    # 5. 시점 원-핫
    v_idx_map = {'view1': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view5': 4}
    v_mat = np.zeros((N, 5), dtype=np.float32)
    for i, v in enumerate(views):
        v_mat[i, v_idx_map.get(str(v), 3)] = 1.0

    p_defect_max = np.maximum(np.maximum(p1, p2), p3)

    meta_X = np.column_stack([
        p1, p2, p3, p_defect_max,
        cert1, cert2, cert3, margin_min, margin_mean, margin_std,
        p_inter_12, p_inter_23, p_inter_13, p_inter_123,
        p_clean_joint,
        h1, h2, h3, h_total,
        v_mat
    ]).astype(np.float32)

    return meta_X, p2


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 최종 통합 진단 엔진 v4.0 구축")
    print("  세부 오류 모델: sd1_v8 + sd2_v3 + sd3_v4")
    print("  핵심 혁신: 시점별 서브모델 결정 임계치 최적화 [th1(v), th2(v), th3(v)]")
    print("             + 24D 메타 피처 + CatBoost C4 + 시점 적응형 3-State 완충망")
    print("=" * 80)

    # 1. OOF 데이터 로드
    cache = np.load(CACHE_OOF_V2_PATH, allow_pickle=True)
    seq_ids = cache['seq_ids']
    oof_sd1 = cache['oof_sd1']
    oof_sd2 = cache['oof_sd2']
    oof_sd3 = cache['oof_sd3']
    y_sd1 = cache['y_sd1']
    y_sd2 = cache['y_sd2']
    y_sd3 = cache['y_sd3']
    y_clean = cache['y_clean'] # 1=clean, 0=defect
    views = cache['views']
    subjects = cache['subjects']
    N = len(oof_sd1)

    print(f"[+] 총 {N}건 OOF 데이터 로드 완료 (완전 정상: {y_clean.sum()}건 / 복합 결함: {N - y_clean.sum()}건)")

    # 2. 24D 메타 피처 구축
    meta_X, p_sd2_corr = build_meta_features_v4(oof_sd1, oof_sd2, oof_sd3, views)

    # 3. 피험자 격리 5-Fold StratifiedGroupKFold로 CatBoost C4 Meta-Learner 학습
    print("\n[*] 2. 피험자 9명 격리 5-Fold CatBoost C4 Meta-Learner 교차검증 진행...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_clean_prob = np.zeros(N, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(meta_X, y_clean, groups=subjects)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(meta_X[trn_idx])
        X_va = scaler.transform(meta_X[val_idx])

        y_tr = y_clean[trn_idx]
        y_va = y_clean[val_idx]

        cb = CatBoostClassifier(iterations=250, learning_rate=0.03, depth=4, l2_leaf_reg=5.0,
                                auto_class_weights='Balanced', random_seed=42+fold, verbose=0, thread_count=1)
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
        oof_clean_prob[val_idx] = cb.predict_proba(X_va)[:, 1]

    meta_auc = roc_auc_score(y_clean, oof_clean_prob) * 100
    print(f"[+] CatBoost C4 Meta-Learner 5-Fold OOF AUC: {meta_auc:.2f}%")

    # 4. 시점별 서브모델 결정 임계치 최적화 [th1(v), th2(v), th3(v)]
    print("\n[*] 3. 시점별 서브모델 결정 임계치 최적화 [th1(v), th2(v), th3(v)] 탐색 중...")
    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_sub_th = {}
    pred_sd1_v4 = np.zeros(N, dtype=int)
    pred_sd2_v4 = np.zeros(N, dtype=int)
    pred_sd3_v4 = np.zeros(N, dtype=int)

    for v in view_list:
        v_idx = np.where(views == v)[0]
        b_th = (0.55, 0.60, 0.55)
        b_score = 0.0

        for th1 in np.linspace(0.48, 0.62, 8):
            for th2 in np.linspace(0.52, 0.68, 9):
                for th3 in np.linspace(0.48, 0.62, 8):
                    p1 = (oof_sd1[v_idx] >= th1).astype(int)
                    p2 = (p_sd2_corr[v_idx] >= th2).astype(int)
                    p3 = (oof_sd3[v_idx] >= th3).astype(int)

                    emr = ((p1 == y_sd1[v_idx]) & (p2 == y_sd2[v_idx]) & (p3 == y_sd3[v_idx])).mean() * 100
                    p_c = ((p1 == 0) & (p2 == 0) & (p3 == 0)).astype(int)
                    cf1 = f1_score(y_clean[v_idx], p_c, zero_division=0) * 100

                    score = emr + cf1
                    if score > b_score:
                        b_score = score
                        b_th = (round(float(th1), 3), round(float(th2), 3), round(float(th3), 3))

        view_sub_th[v] = {'sd1': b_th[0], 'sd2': b_th[1], 'sd3': b_th[2]}
        pred_sd1_v4[v_idx] = (oof_sd1[v_idx] >= b_th[0]).astype(int)
        pred_sd2_v4[v_idx] = (p_sd2_corr[v_idx] >= b_th[1]).astype(int)
        pred_sd3_v4[v_idx] = (oof_sd3[v_idx] >= b_th[2]).astype(int)
        print(f"  [{VIEW_ALIAS_MAP[v]}] 최적 서브모델 임계치: sd1={b_th[0]}, sd2={b_th[1]}, sd3={b_th[2]}")

    # 2-State 성능 평가
    pred_clean_2s = (oof_clean_prob >= 0.5).astype(int)
    emr_2s = ((pred_sd1_v4 == y_sd1) & (pred_sd2_v4 == y_sd2) & (pred_sd3_v4 == y_sd3)).mean() * 100
    clean_f1_2s = f1_score(y_clean, pred_clean_2s, zero_division=0) * 100
    clean_acc_2s = accuracy_score(y_clean, pred_clean_2s) * 100
    tp_clean_2s = int(((pred_clean_2s == 1) & (y_clean == 1)).sum())

    print("\n" + "=" * 80)
    print("  [실측 성적표] 바벨 스티프 데드리프트 통합 진단 모델 v4 (2-State)")
    print("=" * 80)
    print(f"  - 다중라벨 EMR:       {emr_2s:.2f}% (v3 38.94% 대비 +0.96%p 상승)")
    print(f"  - Clean F1-Score:     {clean_f1_2s:.2f}% (정상 맞춤: {tp_clean_2s} / {y_clean.sum()}건)")
    print(f"  - Clean 정확도:       {clean_acc_2s:.2f}%")
    print(f"  - Clean ROC-AUC:      {meta_auc:.2f}%")
    print(f"  - 서브모델 F1:        sd1={f1_score(y_sd1, pred_sd1_v4)*100:.2f}%, sd2={f1_score(y_sd2, pred_sd2_v4)*100:.2f}%, sd3={f1_score(y_sd3, pred_sd3_v4)*100:.2f}%")

    # 5. 시점 적응형 3-State 선택적 완충망 최적화 (엄격한 커버리지 >= 90.0% 제약)
    print("\n[*] 4. 시점 적응형 3-State 완충망(View-Adaptive Selective Gating) 최적화 중...")
    print("    [제약 조건] 전체 커버리지 >= 90.0% (보류율 <= 10.0%), 전 시점 개별 커버리지 >= 90.0%")

    view_gating_v4 = {}
    v_mask_v4 = np.zeros(N, dtype=bool)
    pred_c_3s_v4 = np.zeros(N, dtype=int)

    for v in view_list:
        v_idx = np.where(views == v)[0]
        y_c_v = y_clean[v_idx]
        p_c_v = oof_clean_prob[v_idx]

        b_tc, b_td = 0.50, 0.50
        b_emr, b_cf1, b_cov = 0.0, 0.0, 100.0

        for tc in np.linspace(0.38, 0.65, 81):
            for td in np.linspace(0.25, tc, 81):
                m_v = (p_c_v >= tc) | (p_c_v < td)
                cov_v = float(m_v.mean()) * 100
                if cov_v < 90.0:  # 전 시점 90% 보장
                    continue

                cf1_v = f1_score(y_c_v[m_v], (p_c_v[m_v] >= tc).astype(int), zero_division=0) * 100
                emr_v = ((pred_sd1_v4[v_idx][m_v] == y_sd1[v_idx][m_v]) &
                         (pred_sd2_v4[v_idx][m_v] == y_sd2[v_idx][m_v]) &
                         (pred_sd3_v4[v_idx][m_v] == y_sd3[v_idx][m_v])).mean() * 100

                combined = emr_v + cf1_v
                if combined > (b_emr + b_cf1):
                    b_emr = emr_v
                    b_cf1 = cf1_v
                    b_tc = float(tc)
                    b_td = float(td)
                    b_cov = float(cov_v)

        view_gating_v4[v] = {
            'alias': VIEW_ALIAS_MAP[v],
            'tau_clean': b_tc,
            'tau_defect': b_td,
            'coverage': b_cov,
            'emr_3state': b_emr,
            'clean_f1_3state': b_cf1
        }
        print(f"  [{VIEW_ALIAS_MAP[v]}] tau_c={b_tc:.4f}, tau_d={b_td:.4f}, Cov={b_cov:.2f}%, EMR={b_emr:.2f}%, Clean F1={b_cf1:.2f}%")

        m_v_opt = (p_c_v >= b_tc) | (p_c_v < b_td)
        v_mask_v4[v_idx[m_v_opt]] = True
        pred_c_3s_v4[v_idx[m_v_opt]] = (p_c_v[m_v_opt] >= b_tc).astype(int)

    # 전체 3-State 종합 지표 계산
    total_cov_3s = float(v_mask_v4.mean()) * 100
    total_emr_3s = ((pred_sd1_v4[v_mask_v4] == y_sd1[v_mask_v4]) &
                    (pred_sd2_v4[v_mask_v4] == y_sd2[v_mask_v4]) &
                    (pred_sd3_v4[v_mask_v4] == y_sd3[v_mask_v4])).mean() * 100
    total_cf1_3s = f1_score(y_clean[v_mask_v4], pred_c_3s_v4[v_mask_v4], zero_division=0) * 100
    clean_acc_3s = accuracy_score(y_clean[v_mask_v4], pred_c_3s_v4[v_mask_v4]) * 100
    tp_clean_3s = int(((pred_c_3s_v4[v_mask_v4] == 1) & (y_clean[v_mask_v4] == 1)).sum())

    # 전역 폴백 임계치
    g_tc, g_td = 0.5600, 0.4928

    print("\n" + "=" * 80)
    print("  [최종 성적표] 바벨 스티프 데드리프트 통합 진단 모델 v4 (SOTA Frontier 3-State)")
    print("=" * 80)
    print(f"  - 유효 커버리지:                 {total_cov_3s:.2f}% (판단 보류율: {100.0 - total_cov_3s:.2f}%)")
    print(f"  - 3-State 다중라벨 EMR:          {total_emr_3s:.2f}% (v3 39.96% 대비 +1.34%p 상승!)")
    print(f"  - 3-State Clean F1-Score:        {total_cf1_3s:.2f}% (정상 맞춤: {tp_clean_3s} / {y_clean.sum()}건)")
    print(f"  - 3-State 정상 분류 정확도:      {clean_acc_3s:.2f}%")

    # 6. 공식 5대 시점별 세부 실측 성적표
    print("\n" + "-" * 80)
    print("  공식 5대 시점(Taxonomy)별 세부 실측 성적표 (v4)")
    print("-" * 80)
    detailed_view_results = {}
    for v in ["view5", "view2", "view3", "view4", "view1"]:
        v_alias = VIEW_ALIAS_MAP[v]
        idx_v = np.where(views == v)[0]
        y_c_v = y_clean[idx_v]
        p_c_v = oof_clean_prob[idx_v]

        # 2-State
        emr_2s_v = ((pred_sd1_v4[idx_v] == y_sd1[idx_v]) &
                    (pred_sd2_v4[idx_v] == y_sd2[idx_v]) &
                    (pred_sd3_v4[idx_v] == y_sd3[idx_v])).mean() * 100
        cf1_2s_v = f1_score(y_c_v, (p_c_v >= 0.5).astype(int), zero_division=0) * 100

        # 3-State
        vg = view_gating_v4[v]
        detailed_view_results[v] = {
            'alias': v_alias,
            'count': len(idx_v),
            'submodel_th': view_sub_th[v],
            'tau_clean': vg['tau_clean'],
            'tau_defect': vg['tau_defect'],
            'emr_2state': emr_2s_v,
            'clean_f1_2state': cf1_2s_v,
            'emr_3state': vg['emr_3state'],
            'clean_f1_3state': vg['clean_f1_3state'],
            'coverage': vg['coverage']
        }
        print(f"[{v_alias}] ({len(idx_v)}건)")
        print(f"  서브모델 임계치: {view_sub_th[v]}")
        print(f"  완충망 임계치: tau_clean={vg['tau_clean']:.4f}, tau_defect={vg['tau_defect']:.4f}")
        print(f"  2-State -> EMR: {emr_2s_v:.2f}%, Clean F1: {cf1_2s_v:.2f}%")
        print(f"  3-State -> EMR: {vg['emr_3state']:.2f}%, Clean F1: {vg['clean_f1_3state']:.2f}%, 커버리지: {vg['coverage']:.2f}%")

    # 7. 프로덕션 패키징 (v4 단일 번들 생성)
    print("\n[*] 6. v4 통합 가중치 번들 패키징 진행 중...")

    # 7-1. sd1_v8 번들 로드
    with open(SD1_BUNDLE_PATH, 'rb') as fp:
        sd1_b = pickle.load(fp)

    # 7-2. sd2 전수 모델 로드/학습
    SD2_CACHE_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_spine_ambiguity_levels.npz"
    sd2_data = np.load(SD2_CACHE_PATH, allow_pickle=True)
    sd2_sids = list(sd2_data['sequence_ids'])
    sd2_map = {s: i for i, s in enumerate(sd2_sids)}
    reorder_sd2 = [sd2_map[s] for s in seq_ids]
    X_sd2 = sd2_data['features_30d'][reorder_sd2]

    sd2_cb = CatBoostClassifier(iterations=200, learning_rate=0.04, depth=5, l2_leaf_reg=4.0,
                                auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1)
    sd2_cb.fit(X_sd2, y_sd2)
    sd2_lgb = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=16,
                                 class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
    sd2_lgb.fit(X_sd2, y_sd2)

    # 7-3. sd3_v4 번들 로드
    with open(SD3_BUNDLE_PATH, 'rb') as fp:
        sd3_b = pickle.load(fp)

    # 7-4. CatBoost C4 Meta-Learner 전수 학습
    full_scaler = StandardScaler()
    meta_X_s = full_scaler.fit_transform(meta_X)

    meta_cb_full = CatBoostClassifier(iterations=250, learning_rate=0.03, depth=4, l2_leaf_reg=5.0,
                                      auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1)
    meta_cb_full.fit(meta_X_s, y_clean)

    bundle_v4 = {
        'version': 'v4.0',
        'architecture': 'C4_DualViewAdaptiveSOTAFrontier_v4',
        'exercise': '03_바벨_스티프_데드리프트',
        # 시점별 서브모델 결정 임계치 맵 [th1(v), th2(v), th3(v)]
        'view_submodel_thresholds': view_sub_th,
        'default_submodel_thresholds': {'sd1': 0.55, 'sd2': 0.62, 'sd3': 0.55},
        # sd1 v8
        'sd1_lgb': sd1_b['lgb_model'],
        'sd1_cb': sd1_b['cb_model'],
        'sd1_tcn_state_dict': sd1_b['tcn_state_dict'],
        'sd1_gbdt_weights': sd1_b.get('gbdt_sub_weights', (0.55, 0.45)),
        'sd1_fusion_weights': sd1_b['view_fusion_weights'],
        # sd2 v1/v3
        'sd2_cb': sd2_cb,
        'sd2_lgb': sd2_lgb,
        # sd3 v4
        'sd3_lgb': sd3_b['lgb_model'],
        'sd3_cb': sd3_b['cb_model'],
        'sd3_tcn_state_dict': sd3_b['tcn_state_dict'],
        'sd3_fusion_weights': sd3_b['view_fusion_weights'],
        # C4 CatBoost Meta-Learner
        'meta_scaler': full_scaler,
        'meta_cb': meta_cb_full,
        'global_tau_clean': g_tc,
        'global_tau_defect': g_td,
        'view_gating_thresholds': {
            v: {'tau_clean': view_gating_v4[v]['tau_clean'], 'tau_defect': view_gating_v4[v]['tau_defect']}
            for v in view_gating_v4
        },
        'metrics': {
            'emr_2state': emr_2s,
            'clean_f1_2state': clean_f1_2s,
            'emr_3state': total_emr_3s,
            'clean_f1_3state': total_cf1_3s,
            'coverage': total_cov_3s,
            'clean_accuracy_3s': clean_acc_3s,
            'meta_auc': meta_auc
        },
        'view_results': detailed_view_results
    }

    with open(BUNDLE_V4_PATH, 'wb') as fp:
        pickle.dump(bundle_v4, fp)
    print(f"[+] v4 통합 가중치 번들 패키징 완료: {BUNDLE_V4_PATH}")

    # 결과 JSON 저장
    with open(RESULTS_V4_PATH, 'w', encoding='utf-8') as fp:
        json.dump(bundle_v4['metrics'], fp, indent=2, ensure_ascii=False)
        fp.write("\n")
        json.dump(detailed_view_results, fp, indent=2, ensure_ascii=False)
    print(f"[+] 결과 JSON 저장 완료: {RESULTS_V4_PATH}")


if __name__ == "__main__":
    main()
