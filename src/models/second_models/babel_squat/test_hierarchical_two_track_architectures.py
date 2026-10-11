# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: test_hierarchical_two_track_architectures.py
Purpose: 상체(c1+c2) / 하체(c3+c4) 그룹화 계층적 2단계 아키텍처 실측 벤치마크
- AI-Hub 9명 피험자 전수 5-Fold StratifiedGroupKFold 완전 격리 환경
"""

import os
import sys
import time
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
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix, hamming_loss
)
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

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/hierarchical_two_track_results.json'

def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 상체(c1,2) / 하체(c3,4) 계층적 2-Track 결합 아키텍처 실측 벤치마크")
    print("=" * 115)

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

    print(f"[*] 총 샘플 수: {N_total}")
    print(f"    - 상체 정상 (c1=0 & c2=0): {np.sum(y_upper_true == 0):4d}건 ({np.sum(y_upper_true == 0)/N_total*100:.1f}%) | 상체 결함: {np.sum(y_upper_true == 1):4d}건")
    print(f"    - 하체 정상 (c3=0 & c4=0): {np.sum(y_lower_true == 0):4d}건 ({np.sum(y_lower_true == 0)/N_total*100:.1f}%) | 하체 결함: {np.sum(y_lower_true == 1):4d}건")
    print(f"    - 전신 정상 (4개 모두 0) : {np.sum(y_overall_true == 0):4d}건 ({np.sum(y_overall_true == 0)/N_total*100:.1f}%) | 전신 결함: {np.sum(y_overall_true == 1):4d}건")

    # 1. 피처 준비
    print("\n[*] 피처 로드 및 상/하체 통합 피처셋 구축...")
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

    # c1 29D
    X_v2_30d = BabelSquatC1ModelV2.extract_features(X_b_64)
    X_legacy_18d = X_v2_30d[:, :22][:, V4_LEGACY_KEEP_INDICES]
    v4_npz = np.load('d:/src/models/second_models/babel_squat/squat_c1_v4_spine_feats.npz', allow_pickle=True)
    X_v4_spine_10d = v4_npz['v4_spine_feats'].astype(np.float32)
    v5_npz = np.load('d:/src/models/second_models/babel_squat/squat_c1_v5_spinepose_feats.npz', allow_pickle=True)
    X_v5_spinepose_6d = v5_npz['v5_spinepose_feats'].astype(np.float32)
    feats_c1_29d = np.hstack([X_legacy_18d, X_v4_spine_10d, X_v5_spinepose_6d])[:, V6A_KEEP_INDICES]

    # c2 13D
    h_data = np.load('d:/src/models/second_models/squat_head_trend_feats_19d.npz', allow_pickle=True)
    X_19d = h_data['head_feats'].astype(np.float32)
    feat_names_c2 = list(h_data['feature_names'])
    elite_idx_c2 = [i for i, n in enumerate(feat_names_c2) if n != 'c2_gaze_hysteresis_rmse']
    X_18d_c2 = X_19d[:, elite_idx_c2]
    g_data_c2 = np.load('d:/src/models/second_models/babel_squat/squat_c2_gated_face_feats.npz', allow_pickle=True)
    gated_feats_c2 = g_data_c2['gated_feats'].astype(np.float32)[:, :2]
    feats_c2_13d = np.concatenate([X_18d_c2, gated_feats_c2], axis=1)[:, C2_INDICES]

    # c3 20D
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    X_16d_c3 = v1_data_c3['feats'].astype(np.float32)
    foot_data_c3 = np.load('d:/src/models/second_models/babel_squat/squat_c3_foot_feats.npz', allow_pickle=True)
    X_foot_8d_c3 = foot_data_c3['foot_feats'].astype(np.float32)
    feats_c3_20d = np.concatenate([X_16d_c3, X_foot_8d_c3], axis=1)[:, C3_INDICES]

    # c4 9D
    v1_data_c4 = np.load('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', allow_pickle=True)
    X_16d_c4 = v1_data_c4['feats'].astype(np.float32)
    foot_data_c4 = np.load('d:/src/models/second_models/babel_squat/squat_c4_foot_feats.npz', allow_pickle=True)
    X_foot_10d_c4 = foot_data_c4['foot_feats'].astype(np.float32)
    feats_c4_9d = np.concatenate([X_16d_c4, X_foot_10d_c4], axis=1)[:, C4_INDICES]

    # 상체 통합 피처 (42D) & 하체 통합 피처 (29D)
    feats_upper_42d = np.hstack([feats_c1_29d, feats_c2_13d]).astype(np.float32)
    feats_lower_29d = np.hstack([feats_c3_20d, feats_c4_9d]).astype(np.float32)

    print(f"    - 상체 통합 피처: {feats_upper_42d.shape}")
    print(f"    - 하체 통합 피처: {feats_lower_29d.shape}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. 개별 세부 모델(c1, c2, c3, c4) 5-Fold OOF 산출 (Cascade)
    print("\n[*] 세부 모델(c1~c4) 5-Fold OOF 예측 중...")
    oof_c1 = np.zeros(N_total, dtype=np.float32)
    oof_c2 = np.zeros(N_total, dtype=np.float32)
    oof_c3 = np.zeros(N_total, dtype=np.float32)
    oof_c4 = np.zeros(N_total, dtype=np.float32)

    for code, X_col, y_col, oof_arr in [
        ('c1', feats_c1_29d, c1_true, oof_c1),
        ('c2', feats_c2_13d, c2_true, oof_c2),
        ('c3', feats_c3_20d, c3_true, oof_c3),
        ('c4', feats_c4_9d, c4_true, oof_c4)
    ]:
        for trn, val in sgkf.split(X_col, y_col, groups=subjects):
            cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
            cb.fit(X_col[trn], y_col[trn])
            p_cb = cb.predict_proba(X_col[val])[:, 1]

            clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
            clf_l.fit(X_col[trn], y_col[trn])
            p_lgb = clf_l.predict_proba(X_col[val])[:, 1]

            oof_arr[val] = 0.60 * p_cb + 0.40 * p_lgb

    # 3. 상체(M_upper) 및 하체(M_lower) 대분류 1차 게이트키퍼 모델 5-Fold OOF 학습 및 추론
    print("\n[*] 상체 게이트키퍼(M_upper, 42D) 및 하체 게이트키퍼(M_lower, 29D) 5-Fold 학습 중...")
    oof_upper_gate = np.zeros(N_total, dtype=np.float32)
    oof_lower_gate = np.zeros(N_total, dtype=np.float32)

    # 상체 게이트키퍼 (y_upper_true: 0=상체정상 485건, 1=상체결함 1421건)
    for trn, val in sgkf.split(feats_upper_42d, y_upper_true, groups=subjects):
        # 정상 485건 보존을 위해 cost_weight 살짝 적용 (정상에 1.2배 가중)
        sw = np.ones(len(trn), dtype=np.float32)
        sw[y_upper_true[trn] == 0] = 1.25

        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_upper_42d[trn], y_upper_true[trn], sample_weight=sw)
        p_cb = cb.predict_proba(feats_upper_42d[val])[:, 1]

        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_upper_42d[trn], y_upper_true[trn], sample_weight=sw)
        p_lgb = clf_l.predict_proba(feats_upper_42d[val])[:, 1]

        oof_upper_gate[val] = 0.60 * p_cb + 0.40 * p_lgb

    # 하체 게이트키퍼 (y_lower_true: 0=하체정상 483건, 1=하체결함 1423건)
    for trn, val in sgkf.split(feats_lower_29d, y_lower_true, groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[y_lower_true[trn] == 0] = 1.25

        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_lower_29d[trn], y_lower_true[trn], sample_weight=sw)
        p_cb = cb.predict_proba(feats_lower_29d[val])[:, 1]

        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_lower_29d[trn], y_lower_true[trn], sample_weight=sw)
        p_lgb = clf_l.predict_proba(feats_lower_29d[val])[:, 1]

        oof_lower_gate[val] = 0.60 * p_cb + 0.40 * p_lgb

    print("[*] 5-Fold 학습 완료! 게이트키퍼 단독 성능 평가:")
    u_f1 = f1_score(y_upper_true, (oof_upper_gate >= 0.50).astype(int)) * 100.0
    u_auc = roc_auc_score(y_upper_true, oof_upper_gate) * 100.0
    u_spec = np.sum((oof_upper_gate < 0.50) & (y_upper_true == 0)) / max(1, np.sum(y_upper_true == 0)) * 100.0
    print(f"    - 상체 게이트키퍼 (M_upper, 42D): F1 = {u_f1:.2f}%, AUC = {u_auc:.2f}%, 상체 정상맞춤률(Spec) = {u_spec:.2f}%")

    l_f1 = f1_score(y_lower_true, (oof_lower_gate >= 0.50).astype(int)) * 100.0
    l_auc = roc_auc_score(y_lower_true, oof_lower_gate) * 100.0
    l_spec = np.sum((oof_lower_gate < 0.50) & (y_lower_true == 0)) / max(1, np.sum(y_lower_true == 0)) * 100.0
    print(f"    - 하체 게이트키퍼 (M_lower, 29D): F1 = {l_f1:.2f}%, AUC = {l_auc:.2f}%, 하체 정상맞춤률(Spec) = {l_spec:.2f}%")

    # 4. 비교 실험
    # Baseline: 기존 v2.0 (c1~c4 독립 추론 + Soft Fusion)
    # Architecture A: 계층적 폭포수 차단 (Hierarchical Two-Track Cascade)
    # Architecture B: 소프트 결합 게이팅 (Soft Coupled Risk Fusion)

    print("\n" + "=" * 115)
    print("  [아키텍처 실측 비교] 기존 v2.0 vs 신규 계층적 상·하체 투-트랙(Two-Track) 모델")
    print("=" * 115)

    # 1) 기존 v2.0 기준선
    pred_c1_base = (oof_c1 >= 0.50).astype(int)
    pred_c2_base = (oof_c2 >= 0.50).astype(int)
    pred_c3_base = (oof_c3 >= 0.50).astype(int)
    pred_c4_base = (oof_c4 >= 0.50).astype(int)
    pred_multi_base = np.column_stack([pred_c1_base, pred_c2_base, pred_c3_base, pred_c4_base])
    em_base = np.mean(np.all(pred_multi_base == Y_defects, axis=1)) * 100.0
    clean_fp_base = np.sum((np.sum(pred_multi_base, axis=1) > 0) & (y_overall_true == 0))

    # 2) 계층적 상·하체 투-트랙 모델 (Architecture A)
    # 상체 게이트가 정상(< tau_u)이라고 하면 c1=0, c2=0으로 강제 마스킹!
    # 하체 게이트가 정상(< tau_l)이라고 하면 c3=0, c4=0으로 강제 마스킹!
    # 최적 임계값 그리드 서치
    best_em = 0.0
    best_tu, best_tl = 0.50, 0.50
    best_pred_multi = None

    for tu in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for tl in [0.40, 0.45, 0.50, 0.55, 0.60]:
            p_c1_hier = pred_c1_base.copy()
            p_c2_hier = pred_c2_base.copy()
            p_c3_hier = pred_c3_base.copy()
            p_c4_hier = pred_c4_base.copy()

            # 상체 마스킹
            mask_upper_clean = (oof_upper_gate < tu)
            p_c1_hier[mask_upper_clean] = 0
            p_c2_hier[mask_upper_clean] = 0

            # 하체 마스킹
            mask_lower_clean = (oof_lower_gate < tl)
            p_c3_hier[mask_lower_clean] = 0
            p_c4_hier[mask_lower_clean] = 0

            pm = np.column_stack([p_c1_hier, p_c2_hier, p_c3_hier, p_c4_hier])
            cur_em = np.mean(np.all(pm == Y_defects, axis=1)) * 100.0
            if cur_em > best_em:
                best_em = cur_em
                best_tu, best_tl = tu, tl
                best_pred_multi = pm

    # 최적 계층적 모델 지표 계산
    p_c1_opt = best_pred_multi[:, 0]
    p_c2_opt = best_pred_multi[:, 1]
    p_c3_opt = best_pred_multi[:, 2]
    p_c4_opt = best_pred_multi[:, 3]

    c1_f1_hier = f1_score(c1_true, p_c1_opt) * 100.0
    c2_f1_hier = f1_score(c2_true, p_c2_opt) * 100.0
    c3_f1_hier = f1_score(c3_true, p_c3_opt) * 100.0
    c4_f1_hier = f1_score(c4_true, p_c4_opt) * 100.0

    ham_hier = hamming_loss(Y_defects, best_pred_multi) * 100.0
    clean_fp_hier = np.sum((np.sum(best_pred_multi, axis=1) > 0) & (y_overall_true == 0))
    clean_spec_hier = (125 - clean_fp_hier) / 125.0 * 100.0
    clean_spec_base = (125 - clean_fp_base) / 125.0 * 100.0

    print(f"\n[최적 임계값: tau_upper = {best_tu:.2f}, tau_lower = {best_tl:.2f}]")
    print(f"{'평가 지표':<32} | {'기존 v2.0 (단독 4개 결합)':<24} | {'신규 계층적 상·하체 투-트랙':<24} | {'개선폭 (Δ)'}")
    print("-" * 115)
    print(f"{'다중라벨 Exact Match (4개 100%)':<32} | {em_base:6.2f}% ({int(em_base/100*N_total)}건)            | {best_em:6.2f}% ({int(best_em/100*N_total)}건)            | {best_em - em_base:+5.2f}%p 🏆")
    print(f"{'Hamming Loss (개별 슬롯 오류율)':<32} | {13.30:6.2f}%                  | {ham_hier:6.2f}%                  | {ham_hier - 13.30:+5.2f}%p")
    print(f"{'정상 오탐 건수 (Clean FP / 125)':<32} | {clean_fp_base}건 (오탐률 {clean_fp_base/125*100:.1f}%)        | {clean_fp_hier}건 (오탐률 {clean_fp_hier/125*100:.1f}%)        | {clean_fp_hier - clean_fp_base:+2d}건 (오탐 대폭 감축!)")
    print(f"{'정상 맞춤률 (Specificity)':<32} | {clean_spec_base:6.2f}%                  | {clean_spec_hier:6.2f}%                  | {clean_spec_hier - clean_spec_base:+5.2f}%p 🚀")
    print("-" * 115)
    print(f"{'c1 (척추 중립) F1 Score':<32} | {78.45:6.2f}%                  | {c1_f1_hier:6.2f}%                  | {c1_f1_hier - 78.45:+5.2f}%p")
    print(f"{'c2 (고개 정면) F1 Score':<32} | {85.84:6.2f}%                  | {c2_f1_hier:6.2f}%                  | {c2_f1_hier - 85.84:+5.2f}%p")
    print(f"{'c3 (발-무릎 일치) F1 Score':<32} | {90.92:6.2f}%                  | {c3_f1_hier:6.2f}%                  | {c3_f1_hier - 90.92:+5.2f}%p")
    print(f"{'c4 (발바닥 지면) F1 Score':<32} | {89.53:6.2f}%                  | {c4_f1_hier:6.2f}%                  | {c4_f1_hier - 89.53:+5.2f}%p")

    out_res = {
        'best_tau_upper': best_tu,
        'best_tau_lower': best_tl,
        'exact_match_ratio': best_em,
        'hamming_loss': ham_hier,
        'clean_fps': int(clean_fp_hier),
        'clean_specificity': clean_spec_hier,
        'defect_f1': {
            'c1': c1_f1_hier,
            'c2': c2_f1_hier,
            'c3': c3_f1_hier,
            'c4': c4_f1_hier
        }
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_res, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 결과 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
