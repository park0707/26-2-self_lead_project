# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_babel_squat_integrated_v4_vs_v4_1_benchmark.py
Purpose: 바벨 스쿼트 통합 모델 v4.0 (c1 v6.0 탑재) vs v4.1 (c1 v7.1 45D 최적 다이어트 융합 모델 탑재)
         공식 5-Fold 벤치마크 및 완전 적중률(Exact Match Ratio) 정밀 평가
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
from src.models.second_models.babel_squat.c4_model_v3 import V3_INDICES_FROM_26D as C4_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(CURRENT_DIR, 'squat_integrated_v4_vs_v4_1_results.json')


def calculate_binary_metrics(y_true, y_pred, pos_label=1):
    p = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0) * 100.0
    r = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0) * 100.0
    f = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0) * 100.0
    return {'precision': float(p), 'recall': float(r), 'f1': float(f)}


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 스쿼트 통합 엔진 v4.0 (c1 v6.0) vs v4.1 (c1 v7.1 탑재) 공식 5-Fold 벤치마크")
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

    # Level 4 마스크
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
    print(f"    - Level 4 배제 후 실무 데이터셋: {N_clean_dataset}건")

    # 2. 피처 로드
    print("\n[*] 상체 및 하체 게이트키퍼 피처셋 로드...")
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

    # 게이트키퍼 결합 피처
    feats_upper_42d = np.hstack([feats_c1_29d, feats_c2_13d]).astype(np.float32)
    feats_lower_v4_26d = np.hstack([feats_c3_v4_17d, feats_c4_9d]).astype(np.float32)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. OOF 로드
    cache_path = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')
    cached = np.load(cache_path)

    oof_c1_v6 = cached['oof_c1_v6']
    oof_c1_v71 = cached['oof_c1']  # v7.1
    oof_c2 = cached['oof_c2']
    oof_c3_v4 = cached['oof_c3']  # v4.0
    oof_c4 = cached['oof_c4']

    # 4. 게이트키퍼 5-Fold OOF 산출
    print("[*] 1차 게이트키퍼 5-Fold OOF 산출...")
    oof_upper_gate = np.zeros(N_total, dtype=np.float32)
    oof_lower_gate_v4 = np.zeros(N_total, dtype=np.float32)

    for trn, val in sgkf.split(feats_upper_42d, y_upper_true, groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[y_upper_true[trn] == 0] = 1.25
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_upper_42d[trn], y_upper_true[trn], sample_weight=sw)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_upper_42d[trn], y_upper_true[trn], sample_weight=sw)
        oof_upper_gate[val] = 0.60 * cb.predict_proba(feats_upper_42d[val])[:, 1] + 0.40 * clf_l.predict_proba(feats_upper_42d[val])[:, 1]

    for trn, val in sgkf.split(feats_lower_v4_26d, y_lower_true, groups=subjects):
        sw = np.ones(len(trn), dtype=np.float32)
        sw[y_lower_true[trn] == 0] = 1.25
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(feats_lower_v4_26d[trn], y_lower_true[trn], sample_weight=sw)
        clf_l = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(feats_lower_v4_26d[trn], y_lower_true[trn], sample_weight=sw)
        oof_lower_gate_v4[val] = 0.60 * cb.predict_proba(feats_lower_v4_26d[val])[:, 1] + 0.40 * clf_l.predict_proba(feats_lower_v4_26d[val])[:, 1]

    # 5. 파이프라인 예측 실행 (v4.0 vs v4.1)
    tau_u, tau_l = 0.55, 0.45

    def run_pipeline(c1_oof_arr):
        p1 = (c1_oof_arr >= 0.50).astype(int)
        p2 = (oof_c2 >= 0.50).astype(int)
        p3 = (oof_c3_v4 >= 0.50).astype(int)
        p4 = (oof_c4 >= 0.50).astype(int)

        mask_u_clean = (oof_upper_gate < tau_u)
        mask_l_clean = (oof_lower_gate_v4 < tau_l)

        p1[mask_u_clean] = 0
        p2[mask_u_clean] = 0
        p3[mask_l_clean] = 0
        p4[mask_l_clean] = 0

        pred_matrix = np.column_stack([p1, p2, p3, p4])
        overall_pred = (np.sum(pred_matrix, axis=1) > 0).astype(int)
        return pred_matrix, overall_pred

    preds_v40, overall_v40 = run_pipeline(oof_c1_v6)
    preds_v41, overall_v41 = run_pipeline(oof_c1_v71)

    # 6. 정량 지표 산출
    def compute_all_metrics(Y_true_mat, Y_pred_mat, y_ov_true, y_ov_pred):
        clean_metrics = calculate_binary_metrics(y_ov_true, y_ov_pred, pos_label=0)
        defect_metrics = calculate_binary_metrics(y_ov_true, y_ov_pred, pos_label=1)

        defects_m = {}
        for idx, code in enumerate(['c1', 'c2', 'c3', 'c4']):
            defects_m[code] = calculate_binary_metrics(Y_true_mat[:, idx], Y_pred_mat[:, idx], pos_label=1)

        em = float(np.mean(np.all(Y_pred_mat == Y_true_mat, axis=1)) * 100.0)
        ham = float(hamming_loss(Y_true_mat, Y_pred_mat) * 100.0)
        cm = confusion_matrix(y_ov_true, y_ov_pred)
        tn, fp, fn, tp = cm.ravel()

        return {
            'clean': clean_metrics,
            'defect': defect_metrics,
            'defects': defects_m,
            'exact_match': em,
            'hamming_loss': ham,
            'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }

    m_all_v40 = compute_all_metrics(Y_defects, preds_v40, y_overall_true, overall_v40)
    m_all_v41 = compute_all_metrics(Y_defects, preds_v41, y_overall_true, overall_v41)

    m_clean_v40 = compute_all_metrics(Y_defects[clean_mask], preds_v40[clean_mask], y_overall_true[clean_mask], overall_v40[clean_mask])
    m_clean_v41 = compute_all_metrics(Y_defects[clean_mask], preds_v41[clean_mask], y_overall_true[clean_mask], overall_v41[clean_mask])

    # 7. 출력 보고서 작성
    print("\n" + "=" * 120)
    print("  [1. 전신 '정상(Clean)' vs '오류(Defect)' 종합 성능 재평가 성적표 (N=1,906 전체 데이터셋)]")
    print("=" * 120)
    print(f"{'분류 클래스':<28} | {'평가 지표':<12} | {'통합 엔진 v4.0':<18} | {'통합 엔진 v4.1':<18} | {'개선폭 (Δ)'}")
    print("-" * 120)
    print(f"{'전신 정상 (Clean Class)':<28} | {'Precision':<12} | {m_all_v40['clean']['precision']:6.2f}%            | {m_all_v41['clean']['precision']:6.2f}%            | {m_all_v41['clean']['precision'] - m_all_v40['clean']['precision']:+5.2f}%p")
    print(f"{'(4대 결함 모두 없음)':<28} | {'Recall':<12} | {m_all_v40['clean']['recall']:6.2f}% (맞춤률)     | {m_all_v41['clean']['recall']:6.2f}% (맞춤률)     | {m_all_v41['clean']['recall'] - m_all_v40['clean']['recall']:+5.2f}%p")
    print(f"{'':<28} | {'F1-Score':<12} | {m_all_v40['clean']['f1']:6.2f}%            | {m_all_v41['clean']['f1']:6.2f}%            | {m_all_v41['clean']['f1'] - m_all_v40['clean']['f1']:+5.2f}%p 🏆")
    print("-" * 120)
    print(f"{'전신 오류 (Defect Class)':<28} | {'Precision':<12} | {m_all_v40['defect']['precision']:6.2f}%            | {m_all_v41['defect']['precision']:6.2f}%            | {m_all_v41['defect']['precision'] - m_all_v40['defect']['precision']:+5.2f}%p")
    print(f"{'(1개 이상 결함 존재)':<28} | {'Recall':<12} | {m_all_v40['defect']['recall']:6.2f}% (적중률)     | {m_all_v41['defect']['recall']:6.2f}% (적중률)     | {m_all_v41['defect']['recall'] - m_all_v40['defect']['recall']:+5.2f}%p")
    print(f"{'':<28} | {'F1-Score':<12} | {m_all_v40['defect']['f1']:6.2f}%            | {m_all_v41['defect']['f1']:6.2f}%            | {m_all_v41['defect']['f1'] - m_all_v40['defect']['f1']:+5.2f}%p 🏆")
    print("=" * 120)

    # 4대 결함별 정밀 비교
    print("\n" + "=" * 120)
    print("  [2. 4대 결함별(c1~c4) 세부 Precision, Recall, F1-Score 재평가 성적표]")
    print("=" * 120)
    print(f"{'결함 항목':<34} | {'지표':<10} | {'통합 엔진 v4.0':<18} | {'통합 엔진 v4.1':<18} | {'개선폭 (Δ)'}")
    print("-" * 120)
    for code, desc in [
        ('c1', '척추 중립 위반 (Spine Neutrality)'),
        ('c2', '고개 정면 미유지 (Head Alignment)'),
        ('c3', '무릎 방향 불일치 (Knee Valgus)'),
        ('c4', '발바닥 미고정 (Foot Grounding)')
    ]:
        v40_m = m_all_v40['defects'][code]
        v41_m = m_all_v41['defects'][code]
        print(f"{desc:<32} | {'Precision':<10} | {v40_m['precision']:6.2f}%            | {v41_m['precision']:6.2f}%            | {v41_m['precision'] - v40_m['precision']:+5.2f}%p")
        print(f"{code:<34} | {'Recall':<10} | {v40_m['recall']:6.2f}%            | {v41_m['recall']:6.2f}%            | {v41_m['recall'] - v40_m['recall']:+5.2f}%p")
        print(f"{'':<34} | {'F1-Score':<10} | {v40_m['f1']:6.2f}%            | {v41_m['f1']:6.2f}%            | {v41_m['f1'] - v40_m['f1']:+5.2f}%p 🏆")
        print("-" * 120)

    # 종합 진단 완벽 적중률 (Exact Match Ratio)
    em_v40 = m_all_v40['exact_match']
    em_v41 = m_all_v41['exact_match']
    delta_em = em_v41 - em_v40
    correct_v40 = int(round(em_v40 / 100.0 * N_total))
    correct_v41 = int(round(em_v41 / 100.0 * N_total))

    em_c_v40 = m_clean_v40['exact_match']
    em_c_v41 = m_clean_v41['exact_match']
    delta_em_c = em_c_v41 - em_c_v40

    print("\n" + "=" * 120)
    print("  [3. 4대 결함 100% 완전 동시 적중률 (Exact Match Ratio: 4개 결함 라벨 완벽 일치율)]")
    print("=" * 120)
    print(f"[*] 전체 데이터셋 (N=1,906):")
    print(f"    - 통합 엔진 v4.0 (c1 v6.0 탑재): {em_v40:.2f}% ({correct_v40} / {N_total}건 완벽 적중)")
    print(f"    - 🏆 통합 엔진 v4.1 (c1 v7.1 탑재): {em_v41:.2f}% ({correct_v41} / {N_total}건 완벽 적중)")
    print(f"    - 🚀 순증 개선: {delta_em:+5.2f}%p (4대 결함 완벽 적중 건수 {correct_v41 - correct_v40:+d}건 추가 적중!)")
    print("-" * 120)
    print(f"[*] Level 4(가짜 정상 52건) 배제 후 실무 데이터셋 (N=1,854):")
    print(f"    - 통합 엔진 v4.0 (c1 v6.0 탑재): {em_c_v40:.2f}%")
    print(f"    - 🏆 통합 엔진 v4.1 (c1 v7.1 탑재): {em_c_v41:.2f}% ({delta_em_c:+5.2f}%p 추가 상승)")
    print("=" * 120)

    # JSON 저장
    results_payload = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'N_total': N_total,
        'N_clean_dataset': N_clean_dataset,
        'all_dataset': {
            'v4_0': m_all_v40,
            'v4_1': m_all_v41
        },
        'clean_dataset': {
            'v4_0': m_clean_v40,
            'v4_1': m_clean_v41
        }
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[*] 벤치마크 결과 JSON 저장 완료: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
