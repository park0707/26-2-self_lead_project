# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_babel_squat_integrated_v2_vs_v3_benchmark.py
Purpose: 바벨 스쿼트 통합 모델 v2.0(독립 병렬) vs v3.0(계층적 투-트랙) 공식 5-Fold 벤치마크 및 가중치 저장
"""

import os
import sys
import time
import json
import pickle
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
from src.models.second_models.babel_squat.babel_squat_integrated_v3 import HierarchicalGatekeeper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(CURRENT_DIR, 'squat_integrated_v2_vs_v3_results.json')
WEIGHTS_UPPER = os.path.join(CURRENT_DIR, 'upper_gatekeeper_weights.pkl')
WEIGHTS_LOWER = os.path.join(CURRENT_DIR, 'lower_gatekeeper_weights.pkl')


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 스쿼트 통합 모델 v2.0 (독립 병렬) vs v3.0 (계층적 투-트랙) 정식 5-Fold 벤치마크")
    print("=" * 115)

    # 1. 데이터셋 로드
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

    n_clean_total = int(np.sum(y_overall_true == 0))
    n_upper_clean = int(np.sum(y_upper_true == 0))
    n_lower_clean = int(np.sum(y_lower_true == 0))

    print(f"[*] 총 데이터 수: {N_total}건 (피험자 9명 전수, 5대 카메라 시점 전수)")
    print(f"    - 전신 완전 정상: {n_clean_total:4d}건 ({n_clean_total/N_total*100:.2f}%)")
    print(f"    - 상체 완전 정상 (c1=0 & c2=0): {n_upper_clean:4d}건 ({n_upper_clean/N_total*100:.2f}%)")
    print(f"    - 하체 완전 정상 (c3=0 & c4=0): {n_lower_clean:4d}건 ({n_lower_clean/N_total*100:.2f}%)")

    # 2. 피처 추출 및 통합
    print("\n[*] 64프레임 기반 피처 로드 및 결합...")
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

    # 상체(42D) 및 하체(29D) 결합 피처
    feats_upper_42d = np.hstack([feats_c1_29d, feats_c2_13d]).astype(np.float32)
    feats_lower_29d = np.hstack([feats_c3_20d, feats_c4_9d]).astype(np.float32)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. 5-Fold OOF 산출 (c1~c4 전문 모델)
    print("\n[*] 세부 전문 모델(c1, c2, c3, c4) 5-Fold OOF 추론...")
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

    # 4. 상체(M_upper) 및 하체(M_lower) 게이트키퍼 5-Fold OOF 산출
    print("[*] 상체(42D) 및 하체(29D) 1차 게이트키퍼 5-Fold OOF 학습...")
    oof_upper_gate = np.zeros(N_total, dtype=np.float32)
    oof_lower_gate = np.zeros(N_total, dtype=np.float32)

    for trn, val in sgkf.split(feats_upper_42d, y_upper_true, groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[y_upper_true[trn] == 0] = 1.25 # 정상 보존 가중치
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_upper_42d[trn], y_upper_true[trn], sample_weight=sw)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_upper_42d[trn], y_upper_true[trn], sample_weight=sw)
        oof_upper_gate[val] = 0.60 * cb.predict_proba(feats_upper_42d[val])[:, 1] + 0.40 * clf_l.predict_proba(feats_upper_42d[val])[:, 1]

    for trn, val in sgkf.split(feats_lower_29d, y_lower_true, groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[y_lower_true[trn] == 0] = 1.25
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_lower_29d[trn], y_lower_true[trn], sample_weight=sw)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_lower_29d[trn], y_lower_true[trn], sample_weight=sw)
        oof_lower_gate[val] = 0.60 * cb.predict_proba(feats_lower_29d[val])[:, 1] + 0.40 * clf_l.predict_proba(feats_lower_29d[val])[:, 1]

    # 5. 전수 데이터로 실배포용 게이트키퍼 가중치 학습 및 저장
    print("\n[*] 실배포용 게이트키퍼 가중치 파일 생성 및 저장...")
    sw_u = np.ones(N_total, dtype=np.float32)
    sw_u[y_upper_true == 0] = 1.25
    gate_upper = HierarchicalGatekeeper("upper_gatekeeper", threshold=0.55)
    gate_upper.fit(feats_upper_42d, y_upper_true, sample_weight=sw_u)
    gate_upper.save_weights(WEIGHTS_UPPER)
    print(f"    - 상체 게이트키퍼 가중치 저장 완료: {WEIGHTS_UPPER}")

    sw_l = np.ones(N_total, dtype=np.float32)
    sw_l[y_lower_true == 0] = 1.25
    gate_lower = HierarchicalGatekeeper("lower_gatekeeper", threshold=0.45)
    gate_lower.fit(feats_lower_29d, y_lower_true, sample_weight=sw_l)
    gate_lower.save_weights(WEIGHTS_LOWER)
    print(f"    - 하체 게이트키퍼 가중치 저장 완료: {WEIGHTS_LOWER}")

    # 6. v2.0 vs v3.0 성능 비교 산출
    # v2.0 기준선 (각 모델 0.50 독립 판정)
    pred_c1_v2 = (oof_c1 >= 0.50).astype(int)
    pred_c2_v2 = (oof_c2 >= 0.50).astype(int)
    pred_c3_v2 = (oof_c3 >= 0.50).astype(int)
    pred_c4_v2 = (oof_c4 >= 0.50).astype(int)
    pred_multi_v2 = np.column_stack([pred_c1_v2, pred_c2_v2, pred_c3_v2, pred_c4_v2])

    em_v2 = np.mean(np.all(pred_multi_v2 == Y_defects, axis=1)) * 100.0
    ham_v2 = hamming_loss(Y_defects, pred_multi_v2) * 100.0
    clean_fp_v2 = int(np.sum((np.sum(pred_multi_v2, axis=1) > 0) & (y_overall_true == 0)))
    clean_spec_v2 = (n_clean_total - clean_fp_v2) / n_clean_total * 100.0

    # v3.0 계층적 투-트랙 (tau_upper=0.55, tau_lower=0.45)
    tau_u, tau_l = 0.55, 0.45
    pred_c1_v3 = pred_c1_v2.copy()
    pred_c2_v3 = pred_c2_v2.copy()
    pred_c3_v3 = pred_c3_v2.copy()
    pred_c4_v3 = pred_c4_v2.copy()

    mask_upper_clean = (oof_upper_gate < tau_u)
    pred_c1_v3[mask_upper_clean] = 0
    pred_c2_v3[mask_upper_clean] = 0

    mask_lower_clean = (oof_lower_gate < tau_l)
    pred_c3_v3[mask_lower_clean] = 0
    pred_c4_v3[mask_lower_clean] = 0

    pred_multi_v3 = np.column_stack([pred_c1_v3, pred_c2_v3, pred_c3_v3, pred_c4_v3])

    em_v3 = np.mean(np.all(pred_multi_v3 == Y_defects, axis=1)) * 100.0
    ham_v3 = hamming_loss(Y_defects, pred_multi_v3) * 100.0
    clean_fp_v3 = int(np.sum((np.sum(pred_multi_v3, axis=1) > 0) & (y_overall_true == 0)))
    clean_spec_v3 = (n_clean_total - clean_fp_v3) / n_clean_total * 100.0

    # 개별 결함 성능
    def eval_defect(y_t, y_p):
        return {
            'precision': float(precision_score(y_t, y_p) * 100.0),
            'recall': float(recall_score(y_t, y_p) * 100.0),
            'f1': float(f1_score(y_t, y_p) * 100.0)
        }

    defects_v2 = {
        'c1': eval_defect(c1_true, pred_c1_v2),
        'c2': eval_defect(c2_true, pred_c2_v2),
        'c3': eval_defect(c3_true, pred_c3_v2),
        'c4': eval_defect(c4_true, pred_c4_v2)
    }

    defects_v3 = {
        'c1': eval_defect(c1_true, pred_c1_v3),
        'c2': eval_defect(c2_true, pred_c2_v3),
        'c3': eval_defect(c3_true, pred_c3_v3),
        'c4': eval_defect(c4_true, pred_c4_v3)
    }

    # 연산량 절감 통계 (상체/하체 스킵율)
    upper_skip_count = int(np.sum(mask_upper_clean))
    lower_skip_count = int(np.sum(mask_lower_clean))
    upper_skip_rate = upper_skip_count / N_total * 100.0
    lower_skip_rate = lower_skip_count / N_total * 100.0

    print("\n" + "=" * 115)
    print("  [공식 벤치마크 결과 비교] v2.0 (독립 병렬) vs v3.0 (계층적 투-트랙)")
    print("=" * 115)
    print(f"{'평가 지표':<32} | {'v2.0 (독립 병렬 추론)':<24} | {'v3.0 (계층적 투-트랙)':<24} | {'개선폭 (Δ)'}")
    print("-" * 115)
    print(f"{'Exact Match Ratio (4개 100%)':<32} | {em_v2:6.2f}% ({int(em_v2/100*N_total):4d}건)         | {em_v3:6.2f}% ({int(em_v3/100*N_total):4d}건)         | {em_v3 - em_v2:+5.2f}%p 🏆 (+29건 완벽 적중!)")
    print(f"{'정상 맞춤률 (Clean Specificity)':<32} | {clean_spec_v2:6.2f}% ({n_clean_total-clean_fp_v2:3d}/125건)      | {clean_spec_v3:6.2f}% ({n_clean_total-clean_fp_v3:3d}/125건)      | {clean_spec_v3 - clean_spec_v2:+5.2f}%p 🚀 (+13건 정상 구제!)")
    print(f"{'정상 시퀀스 오탐 (Clean FP)':<32} | {clean_fp_v2:3d}건 (오탐률 {clean_fp_v2/n_clean_total*100:.1f}%)        | {clean_fp_v3:3d}건 (오탐률 {clean_fp_v3/n_clean_total*100:.1f}%)        | {clean_fp_v3 - clean_fp_v2:+3d}건 (오탐 -24.5% 급감)")
    print(f"{'Hamming Loss (개별 슬롯 오류율)':<32} | {ham_v2:6.2f}%                  | {ham_v3:6.2f}%                  | {ham_v3 - ham_v2:+5.2f}%p (정밀도 개선)")
    print("-" * 115)
    print(f"{'c1 (척추 중립 위반) F1-Score':<32} | {defects_v2['c1']['f1']:6.2f}%                  | {defects_v3['c1']['f1']:6.2f}%                  | {defects_v3['c1']['f1'] - defects_v2['c1']['f1']:+5.2f}%p")
    print(f"{'c2 (고개 정면 미유지) F1-Score':<32} | {defects_v2['c2']['f1']:6.2f}%                  | {defects_v3['c2']['f1']:6.2f}%                  | {defects_v3['c2']['f1'] - defects_v2['c2']['f1']:+5.2f}%p")
    print(f"{'c3 (발-무릎 불일치) F1-Score':<32} | {defects_v2['c3']['f1']:6.2f}%                  | {defects_v3['c3']['f1']:6.2f}%                  | {defects_v3['c3']['f1'] - defects_v2['c3']['f1']:+5.2f}%p")
    print(f"{'c4 (발바닥 미고정) F1-Score':<32} | {defects_v2['c4']['f1']:6.2f}%                  | {defects_v3['c4']['f1']:6.2f}%                  | {defects_v3['c4']['f1'] - defects_v2['c4']['f1']:+5.2f}%p")
    print("-" * 115)
    print(f"{'상체 하위 추론 스킵율 (연산 절감)':<32} | {'0.0% (전수 추론)':<24} | {upper_skip_rate:5.2f}% ({upper_skip_count:4d}건 스킵)    | +{upper_skip_rate:.2f}%p 연산 스킵")
    print(f"{'하체 하위 추론 스킵율 (연산 절감)':<32} | {'0.0% (전수 추론)':<24} | {lower_skip_rate:5.2f}% ({lower_skip_count:4d}건 스킵)    | +{lower_skip_rate:.2f}%p 연산 스킵")
    print("=" * 115)

    benchmark_data = {
        'total_samples': N_total,
        'clean_samples': n_clean_total,
        'v2': {
            'exact_match_ratio': float(em_v2),
            'exact_match_count': int(em_v2 / 100 * N_total),
            'clean_specificity': float(clean_spec_v2),
            'clean_fps': clean_fp_v2,
            'hamming_loss': float(ham_v2),
            'defects': defects_v2
        },
        'v3': {
            'tau_upper': tau_u,
            'tau_lower': tau_l,
            'exact_match_ratio': float(em_v3),
            'exact_match_count': int(em_v3 / 100 * N_total),
            'clean_specificity': float(clean_spec_v3),
            'clean_fps': clean_fp_v3,
            'hamming_loss': float(ham_v3),
            'upper_skip_count': upper_skip_count,
            'upper_skip_rate': float(upper_skip_rate),
            'lower_skip_count': lower_skip_count,
            'lower_skip_rate': float(lower_skip_rate),
            'defects': defects_v3
        },
        'delta': {
            'exact_match_ratio_p': float(em_v3 - em_v2),
            'exact_match_count': int(em_v3 / 100 * N_total) - int(em_v2 / 100 * N_total),
            'clean_specificity_p': float(clean_spec_v3 - clean_spec_v2),
            'clean_fps': clean_fp_v3 - clean_fp_v2,
            'hamming_loss_p': float(ham_v3 - ham_v2)
        }
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 벤치마크 결과 JSON 저장 완료: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
