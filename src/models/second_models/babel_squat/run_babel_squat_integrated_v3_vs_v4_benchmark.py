# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_babel_squat_integrated_v3_vs_v4_benchmark.py
Purpose: 바벨 스쿼트 통합 모델 v3.0 vs v4.0(c3 딥러닝 시계열 임베더 탑재) 공식 5-Fold 벤치마크 및 성능 재평가
         - 전신 정상(Clean) 및 전신 오류(Defect) 각각의 Precision, Recall, F1-Score 도출
         - 4대 결함(c1~c4) 세부 정밀도/재현율/F1 재평가
         - 전체 1,906건 및 Level 4 정제 1,854건 이원화 실측
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
from src.models.second_models.babel_squat.babel_squat_integrated_v4 import HierarchicalGatekeeperV4

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(CURRENT_DIR, 'squat_integrated_v3_vs_v4_results.json')
WEIGHTS_LOWER_V4 = os.path.join(CURRENT_DIR, 'lower_gatekeeper_v4_weights.pkl')


def calculate_binary_metrics(y_true, y_pred, pos_label=1):
    """지정된 클래스를 양성(Positive)으로 하는 Precision, Recall, F1 계산"""
    p = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0) * 100.0
    r = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0) * 100.0
    f = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0) * 100.0
    return {'precision': float(p), 'recall': float(r), 'f1': float(f)}


def run_benchmark():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 스쿼트 통합 모델 v3.0 vs v4.0(c3 딥러닝 임베더 탑재) 공식 5-Fold 벤치마크")
    print("=" * 115)

    # 1. 데이터셋 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    Y_defects = ds['Y_defects'][:, :4].astype(int)  # [c1, c2, c3, c4]
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
    n_defect_total = int(np.sum(y_overall_true == 1))

    # Level 4 마스크 (사용자가 검증 확정한 52건 가짜 정상 배제)
    clean_mask = np.ones(N_total, dtype=bool)
    l4_path = 'd:/src/models/second_models/squat_c3_ambiguity_levels.npz'
    if os.path.exists(l4_path):
        l4_data = np.load(l4_path)
        if 'levels' in l4_data:
            l4_indices = np.where((l4_data['levels'] == 4) & (c3_true == 0))[0]
            clean_mask[l4_indices] = False
    N_clean_dataset = int(np.sum(clean_mask))

    print(f"[*] 총 데이터 수: {N_total}건 (피험자 9명 전수, 5대 카메라 시점 전수)")
    print(f"    - 전신 완전 정상: {n_clean_total}건 ({n_clean_total/N_total*100:.2f}%)")
    print(f"    - 전신 결함 존재: {n_defect_total}건 ({n_defect_total/N_total*100:.2f}%)")
    print(f"    - Level 4(가짜 정상 52건) 배제 후 실무 데이터셋: {N_clean_dataset}건")

    # 2. 피처 로드
    print("\n[*] 상체 및 하체 피처셋 로드...")
    # c1 29D
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)
    f_interp = interp1d(np.linspace(0, 1, 16), X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(np.linspace(0, 1, 64)).astype(np.float32)

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

    # c3 v3 20D
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    X_16d_c3 = v1_data_c3['feats'].astype(np.float32)
    foot_data_c3 = np.load(os.path.join(CURRENT_DIR, 'squat_c3_foot_feats.npz'), allow_pickle=True)
    X_foot_8d_c3 = foot_data_c3['foot_feats'].astype(np.float32)
    feats_c3_20d = np.concatenate([X_16d_c3, X_foot_8d_c3], axis=1)[:, C3_INDICES]

    # c3 v4 딥러닝 피처 (16D 임베딩 + 1D 확률 = 17D)
    deep_data = np.load(os.path.join(CURRENT_DIR, 'squat_c3_deep_temporal_feats.npz'), allow_pickle=True)
    deep_embeddings = deep_data['deep_embeddings'].astype(np.float32)
    deep_probs = deep_data['deep_probs'].astype(np.float32)
    feats_c3_v4_17d = np.hstack([deep_embeddings, deep_probs[:, None]])

    # c4 9D
    v1_data_c4 = np.load(os.path.join(CURRENT_DIR, 'squat_c4_feats_16d.npz'), allow_pickle=True)
    X_16d_c4 = v1_data_c4['feats'].astype(np.float32)
    foot_data_c4 = np.load(os.path.join(CURRENT_DIR, 'squat_c4_foot_feats.npz'), allow_pickle=True)
    X_foot_10d_c4 = foot_data_c4['foot_feats'].astype(np.float32)
    feats_c4_9d = np.concatenate([X_16d_c4, X_foot_10d_c4], axis=1)[:, C4_INDICES]

    # 결합 피처
    feats_upper_42d = np.hstack([feats_c1_29d, feats_c2_13d]).astype(np.float32)
    feats_lower_v3_29d = np.hstack([feats_c3_20d, feats_c4_9d]).astype(np.float32)
    feats_lower_v4_26d = np.hstack([feats_c3_v4_17d, feats_c4_9d]).astype(np.float32)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. OOF 캐시 로드 및 생성
    print("\n[*] 세부 전문 모델 OOF 확률 산출...")
    cache_path = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')
    cached = np.load(cache_path)

    oof_c1 = cached['oof_c1']
    oof_c2 = cached['oof_c2']
    oof_c3_v3 = cached.get('oof_c3_v3', None)
    if oof_c3_v3 is None:
        # v3 OOF 재계산
        oof_c3_v3 = np.zeros(N_total, dtype=np.float32)
        for trn, val in sgkf.split(feats_c3_20d, c3_true, groups=subjects):
            cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
            cb.fit(feats_c3_20d[trn], c3_true[trn])
            clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
            clf_l.fit(feats_c3_20d[trn], c3_true[trn])
            oof_c3_v3[val] = 0.60 * cb.predict_proba(feats_c3_20d[val])[:, 1] + 0.40 * clf_l.predict_proba(feats_c3_20d[val])[:, 1]

    oof_c3_v4 = cached['oof_c3']  # 딥러닝 v4 확률
    oof_c4 = cached['oof_c4']

    # 4. 게이트키퍼 5-Fold OOF 산출
    print("[*] 1차 게이트키퍼 5-Fold OOF 산출 및 실배포 가중치 패키징...")
    oof_upper_gate = np.zeros(N_total, dtype=np.float32)
    oof_lower_gate_v3 = np.zeros(N_total, dtype=np.float32)
    oof_lower_gate_v4 = np.zeros(N_total, dtype=np.float32)

    for trn, val in sgkf.split(feats_upper_42d, y_upper_true, groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[y_upper_true[trn] == 0] = 1.25
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_upper_42d[trn], y_upper_true[trn], sample_weight=sw)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_upper_42d[trn], y_upper_true[trn], sample_weight=sw)
        oof_upper_gate[val] = 0.60 * cb.predict_proba(feats_upper_42d[val])[:, 1] + 0.40 * clf_l.predict_proba(feats_upper_42d[val])[:, 1]

    for trn, val in sgkf.split(feats_lower_v3_29d, y_lower_true, groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[y_lower_true[trn] == 0] = 1.25
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_lower_v3_29d[trn], y_lower_true[trn], sample_weight=sw)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_lower_v3_29d[trn], y_lower_true[trn], sample_weight=sw)
        oof_lower_gate_v3[val] = 0.60 * cb.predict_proba(feats_lower_v3_29d[val])[:, 1] + 0.40 * clf_l.predict_proba(feats_lower_v3_29d[val])[:, 1]

    for trn, val in sgkf.split(feats_lower_v4_26d, y_lower_true, groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[y_lower_true[trn] == 0] = 1.25
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_lower_v4_26d[trn], y_lower_true[trn], sample_weight=sw)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_lower_v4_26d[trn], y_lower_true[trn], sample_weight=sw)
        oof_lower_gate_v4[val] = 0.60 * cb.predict_proba(feats_lower_v4_26d[val])[:, 1] + 0.40 * clf_l.predict_proba(feats_lower_v4_26d[val])[:, 1]

    # 실배포 하체 게이트키퍼 v4 가중치 저장
    sw_l = np.ones(N_total, dtype=np.float32)
    sw_l[y_lower_true == 0] = 1.25
    gate_lower_v4 = HierarchicalGatekeeperV4("lower_gatekeeper_v4", threshold=0.45)
    gate_lower_v4.fit(feats_lower_v4_26d, y_lower_true, sample_weight=sw_l)
    gate_lower_v4.save_weights(WEIGHTS_LOWER_V4)
    print(f"[*] 하체 게이트키퍼 v4 가중치 저장 완료: {WEIGHTS_LOWER_V4}")

    # 5. 파이프라인 예측 실행 (v3 vs v4)
    tau_u, tau_l = 0.55, 0.45

    def run_pipeline(oof_c3_arr, oof_l_gate):
        p1 = (oof_c1 >= 0.50).astype(int)
        p2 = (oof_c2 >= 0.50).astype(int)
        p3 = (oof_c3_arr >= 0.50).astype(int)
        p4 = (oof_c4 >= 0.50).astype(int)

        # 게이트키퍼 스킵 적용
        mask_u_clean = (oof_upper_gate < tau_u)
        mask_l_clean = (oof_l_gate < tau_l)

        p1[mask_u_clean] = 0
        p2[mask_u_clean] = 0
        p3[mask_l_clean] = 0
        p4[mask_l_clean] = 0

        pred_matrix = np.column_stack([p1, p2, p3, p4])
        # 전신 결함 여부 (1개 이상 결함이면 1, 전부 0이면 0)
        overall_pred = (np.sum(pred_matrix, axis=1) > 0).astype(int)

        return pred_matrix, overall_pred, mask_u_clean, mask_l_clean

    preds_v3, overall_v3, u_skip_v3, l_skip_v3 = run_pipeline(oof_c3_v3, oof_lower_gate_v3)
    preds_v4, overall_v4, u_skip_v4, l_skip_v4 = run_pipeline(oof_c3_v4, oof_lower_gate_v4)

    # 6. 정량 지표 산출 함수
    def compute_all_metrics(Y_true_mat, Y_pred_mat, y_ov_true, y_ov_pred):
        # 1) 전신 정상 (Clean Class: label 0) 성능
        clean_metrics = calculate_binary_metrics(y_ov_true, y_ov_pred, pos_label=0)
        # 2) 전신 오류 (Defect Class: label 1) 성능
        defect_metrics = calculate_binary_metrics(y_ov_true, y_ov_pred, pos_label=1)

        # 3) 개별 4대 결함 성능
        defects_m = {}
        for idx, code in enumerate(['c1', 'c2', 'c3', 'c4']):
            defects_m[code] = calculate_binary_metrics(Y_true_mat[:, idx], Y_pred_mat[:, idx], pos_label=1)

        # 4) Exact Match & Hamming
        em = float(np.mean(np.all(Y_pred_mat == Y_true_mat, axis=1)) * 100.0)
        ham = float(hamming_loss(Y_true_mat, Y_pred_mat) * 100.0)

        # Confusion Matrix
        cm = confusion_matrix(y_ov_true, y_ov_pred) # [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()

        return {
            'clean': clean_metrics,
            'defect': defect_metrics,
            'defects': defects_m,
            'exact_match': em,
            'hamming_loss': ham,
            'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }

    # 전체 데이터셋 (N=1,906)
    m_all_v3 = compute_all_metrics(Y_defects, preds_v3, y_overall_true, overall_v3)
    m_all_v4 = compute_all_metrics(Y_defects, preds_v4, y_overall_true, overall_v4)

    # Clean 데이터셋 (N=1,854)
    m_clean_v3 = compute_all_metrics(Y_defects[clean_mask], preds_v3[clean_mask], y_overall_true[clean_mask], overall_v3[clean_mask])
    m_clean_v4 = compute_all_metrics(Y_defects[clean_mask], preds_v4[clean_mask], y_overall_true[clean_mask], overall_v4[clean_mask])

    # 7. 출력 보고서 작성
    print("\n" + "=" * 120)
    print("  [1. 전신 '정상(Clean)' vs '오류(Defect)' 종합 성능 재평가 성적표 (N=1,906 전체 데이터셋)]")
    print("=" * 120)
    print(f"{'분류 클래스':<28} | {'평가 지표':<12} | {'통합 엔진 v3.0':<18} | {'통합 엔진 v4.0':<18} | {'개선폭 (Δ)'}")
    print("-" * 120)
    # 정상 클래스 (Clean)
    print(f"{'전신 정상 (Clean Class)':<28} | {'Precision':<12} | {m_all_v3['clean']['precision']:6.2f}%            | {m_all_v4['clean']['precision']:6.2f}%            | {m_all_v4['clean']['precision'] - m_all_v3['clean']['precision']:+5.2f}%p")
    print(f"{'(4대 결함 모두 없음)':<28} | {'Recall':<12} | {m_all_v3['clean']['recall']:6.2f}% (맞춤률)     | {m_all_v4['clean']['recall']:6.2f}% (맞춤률)     | {m_all_v4['clean']['recall'] - m_all_v3['clean']['recall']:+5.2f}%p 🚀")
    print(f"{'':<28} | {'F1-Score':<12} | {m_all_v3['clean']['f1']:6.2f}%            | {m_all_v4['clean']['f1']:6.2f}%            | {m_all_v4['clean']['f1'] - m_all_v3['clean']['f1']:+5.2f}%p 🏆")
    print("-" * 120)
    # 오류 클래스 (Defect)
    print(f"{'전신 오류 (Defect Class)':<28} | {'Precision':<12} | {m_all_v3['defect']['precision']:6.2f}%            | {m_all_v4['defect']['precision']:6.2f}%            | {m_all_v4['defect']['precision'] - m_all_v3['defect']['precision']:+5.2f}%p")
    print(f"{'(1개 이상 결함 존재)':<28} | {'Recall':<12} | {m_all_v3['defect']['recall']:6.2f}% (적중률)     | {m_all_v4['defect']['recall']:6.2f}% (적중률)     | {m_all_v4['defect']['recall'] - m_all_v3['defect']['recall']:+5.2f}%p")
    print(f"{'':<28} | {'F1-Score':<12} | {m_all_v3['defect']['f1']:6.2f}%            | {m_all_v4['defect']['f1']:6.2f}%            | {m_all_v4['defect']['f1'] - m_all_v3['defect']['f1']:+5.2f}%p 🏆")
    print("=" * 120)

    # 4대 결함별 정밀 비교
    print("\n" + "=" * 120)
    print("  [2. 4대 결함별(c1~c4) 세부 Precision, Recall, F1-Score 재평가 성적표]")
    print("=" * 120)
    print(f"{'결함 항목':<34} | {'지표':<10} | {'통합 엔진 v3.0':<18} | {'통합 엔진 v4.0':<18} | {'개선폭 (Δ)'}")
    print("-" * 120)
    for code, desc in [
        ('c1', 'c1 (척추 중립 위반)'),
        ('c2', 'c2 (고개 정면 미유지)'),
        ('c3', 'c3 (발과 무릎 방향 불일치)'),
        ('c4', 'c4 (발바닥 지면 미고정)')
    ]:
        v3_f = m_all_v3['defects'][code]
        v4_f = m_all_v4['defects'][code]
        print(f"{desc:<34} | Precision  | {v3_f['precision']:6.2f}%            | {v4_f['precision']:6.2f}%            | {v4_f['precision'] - v3_f['precision']:+5.2f}%p")
        print(f"{'':<34} | Recall     | {v3_f['recall']:6.2f}%            | {v4_f['recall']:6.2f}%            | {v4_f['recall'] - v3_f['recall']:+5.2f}%p")
        print(f"{'':<34} | F1-Score   | {v3_f['f1']:6.2f}%            | {v4_f['f1']:6.2f}%            | {v4_f['f1'] - v3_f['f1']:+5.2f}%p 🏆")
        print("-" * 120)

    # 전체 다중 라벨 일치율
    print(f"{'Exact Match Ratio (4개 슬롯 100% 완전 일치)':<40} | v3.0: {m_all_v3['exact_match']:6.2f}% ({int(m_all_v3['exact_match']/100*N_total)}건) | v4.0: {m_all_v4['exact_match']:6.2f}% ({int(m_all_v4['exact_match']/100*N_total)}건) | {m_all_v4['exact_match'] - m_all_v3['exact_match']:+5.2f}%p 🏆 (+{int(m_all_v4['exact_match']/100*N_total) - int(m_all_v3['exact_match']/100*N_total)}건 완벽 적중)")
    print(f"{'Hamming Loss (개별 슬롯 오류율)':<40} | v3.0: {m_all_v3['hamming_loss']:6.2f}%           | v4.0: {m_all_v4['hamming_loss']:6.2f}%           | {m_all_v4['hamming_loss'] - m_all_v3['hamming_loss']:+5.2f}%p")
    print(f"{'하체 하위 추론 스킵율 (연산 절감)':<40} | v3.0: {np.sum(l_skip_v3)/N_total*100:6.2f}%           | v4.0: {np.sum(l_skip_v4)/N_total*100:6.2f}%           | {np.sum(l_skip_v4)/N_total*100 - np.sum(l_skip_v3)/N_total*100:+5.2f}%p")
    print("=" * 120)

    # 3. Level 4 정제 데이터셋 (N=1,854) 성적표
    print("\n" + "=" * 120)
    print("  [3. 가짜 정상(Level 4, 52건) 배제 후 순수 클린 실무 데이터셋 재평가 (N=1,854)]")
    print("=" * 120)
    print(f"{'분류 클래스 / 항목':<34} | {'지표':<10} | {'통합 엔진 v3.0':<18} | {'통합 엔진 v4.0':<18} | {'개선폭 (Δ)'}")
    print("-" * 120)
    print(f"{'전신 정상 (Clean Class)':<34} | Precision  | {m_clean_v3['clean']['precision']:6.2f}%            | {m_clean_v4['clean']['precision']:6.2f}%            | {m_clean_v4['clean']['precision'] - m_clean_v3['clean']['precision']:+5.2f}%p")
    print(f"{'(73건 순수 정상)':<34} | Recall     | {m_clean_v3['clean']['recall']:6.2f}%            | {m_clean_v4['clean']['recall']:6.2f}%            | {m_clean_v4['clean']['recall'] - m_clean_v3['clean']['recall']:+5.2f}%p")
    print(f"{'':<34} | F1-Score   | {m_clean_v3['clean']['f1']:6.2f}%            | {m_clean_v4['clean']['f1']:6.2f}%            | {m_clean_v4['clean']['f1'] - m_clean_v3['clean']['f1']:+5.2f}%p")
    print("-" * 120)
    print(f"{'전신 오류 (Defect Class)':<34} | Precision  | {m_clean_v3['defect']['precision']:6.2f}%            | {m_clean_v4['defect']['precision']:6.2f}%            | {m_clean_v4['defect']['precision'] - m_clean_v3['defect']['precision']:+5.2f}%p")
    print(f"{'(1,781건 결함)':<34} | Recall     | {m_clean_v3['defect']['recall']:6.2f}%            | {m_clean_v4['defect']['recall']:6.2f}%            | {m_clean_v4['defect']['recall'] - m_clean_v3['defect']['recall']:+5.2f}%p")
    print(f"{'':<34} | F1-Score   | {m_clean_v3['defect']['f1']:6.2f}%            | {m_clean_v4['defect']['f1']:6.2f}%            | {m_clean_v4['defect']['f1'] - m_clean_v3['defect']['f1']:+5.2f}%p")
    print("-" * 120)
    print(f"{'c3 (발-무릎 방향 불일치) F1':<34} | F1-Score   | {m_clean_v3['defects']['c3']['f1']:6.2f}%            | {m_clean_v4['defects']['c3']['f1']:6.2f}%            | {m_clean_v4['defects']['c3']['f1'] - m_clean_v3['defects']['c3']['f1']:+5.2f}%p 🏆")
    print(f"{'Exact Match Ratio (완전 적중률)':<34} | Accuracy   | {m_clean_v3['exact_match']:6.2f}%            | {m_clean_v4['exact_match']:6.2f}%            | {m_clean_v4['exact_match'] - m_clean_v3['exact_match']:+5.2f}%p 🏆")
    print("=" * 120)

    # JSON 저장
    result_data = {
        'total_samples': N_total,
        'clean_dataset_samples': N_clean_dataset,
        'all_dataset_metrics': {'v3': m_all_v3, 'v4': m_all_v4},
        'clean_dataset_metrics': {'v3': m_clean_v3, 'v4': m_clean_v4}
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 공식 벤치마크 결과 JSON 저장 완료: {OUTPUT_JSON}")


if __name__ == '__main__':
    run_benchmark()
