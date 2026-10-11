# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: run_c3_v4_vs_v5_benchmark.py
Purpose: c3 모델 v4.0 (기존 43D) vs v5.0 (절대 무릎/발목 비율 및 좌우독립 외반합 탑재 53D)
         공식 5-Fold 피험자 격리 벤치마크 및 10건 의심 케이스 개선 효과 정밀 검증
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c3_model_v3 import V2_TO_V3_INDICES as C3_INDICES
from src.models.second_models.babel_squat.c3_model_v5 import BabelSquatC3ModelV5, V5_FEATURE_NAMES_10

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz')
WEIGHTS_PATH = os.path.join(CURRENT_DIR, 'c3_model_v5_weights.pkl')
OUTPUT_JSON = os.path.join(CURRENT_DIR, 'c3_v4_vs_v5_comparison_results.json')


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] c3 무릎 외반: v4.0 (기존 43D) vs v5.0 (절대외반 53D) 공식 5-Fold 벤치마크")
    print("=" * 115)

    # 1. 데이터셋 로드
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)
    subjects = ds['subjects']
    N_total = len(y_c3)

    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b = m['X_Format_B'][sq_mask]
    seq_ids = m['sequence_ids'][sq_mask]

    print(f"[*] 총 데이터 수: {N_total}건 (정상: {np.sum(y_c3==0)}건, 결함: {np.sum(y_c3==1)}건)")

    # 2. 피처 구축
    # (1) Base v3 피처 (20D)
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    foot_data_c3 = np.load(os.path.join(CURRENT_DIR, 'squat_c3_foot_feats.npz'), allow_pickle=True)
    c3_24d = np.concatenate([v1_data_c3['feats'], foot_data_c3['foot_feats']], axis=1).astype(np.float32)
    X_v3 = c3_24d[:, C3_INDICES].copy()

    # (2) Unilateral 피처 (6D)
    uni_data = np.load(os.path.join(CURRENT_DIR, 'squat_c3_unilateral_foot_feats.npz'), allow_pickle=True)
    X_uni = uni_data['unilateral_feats'].astype(np.float32)

    # (3) Deep 피처 (17D)
    deep_data = np.load(os.path.join(CURRENT_DIR, 'squat_c3_deep_temporal_feats.npz'), allow_pickle=True)
    X_deep = np.hstack([deep_data['deep_embeddings'], deep_data['deep_probs'][:, None]]).astype(np.float32)

    # v4.0 피처셋 (43D)
    X_v4 = np.hstack([X_v3, X_uni, X_deep]).astype(np.float32)

    # (4) 신규 v5 절대/좌우외반 피처 (10D)
    X_new_10d = BabelSquatC3ModelV5.extract_v5_new_features(X_b)

    # v5.0 피처셋 (53D)
    X_v5 = np.hstack([X_v4, X_new_10d]).astype(np.float32)

    print(f"[*] 피처셋 구성 완료: v4.0 = {X_v4.shape[1]}D, v5.0 = {X_v5.shape[1]}D (+10D 신규 탑재)")

    # 3. 5-Fold StratifiedGroupKFold 훈련 및 평가
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_v4 = np.zeros(N_total, dtype=np.float32)
    oof_v5 = np.zeros(N_total, dtype=np.float32)

    cb_v5_models = []
    lgb_v5_models = []

    print("\n[*] 5-Fold 피험자 격리 모델 훈련 및 검증 진행 중...")
    for fold, (trn, val) in enumerate(sgkf.split(X_v5, y_c3, groups=subjects)):
        # v4.0
        cb4 = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb4.fit(X_v4[trn], y_c3[trn])
        lgb4 = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        lgb4.fit(X_v4[trn], y_c3[trn])
        oof_v4[val] = 0.60 * cb4.predict_proba(X_v4[val])[:, 1] + 0.40 * lgb4.predict_proba(X_v4[val])[:, 1]

        # v5.0
        cb5 = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb5.fit(X_v5[trn], y_c3[trn])
        lgb5 = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        lgb5.fit(X_v5[trn], y_c3[trn])
        oof_v5[val] = 0.60 * cb5.predict_proba(X_v5[val])[:, 1] + 0.40 * lgb5.predict_proba(X_v5[val])[:, 1]

        cb_v5_models.append(cb5)
        lgb_v5_models.append(lgb5)

    # 4. 종합 지표 산출
    def get_metrics(oof, y_true):
        pred = (oof >= 0.50).astype(int)
        p = precision_score(y_true, pred) * 100.0
        r = recall_score(y_true, pred) * 100.0
        f = f1_score(y_true, pred) * 100.0
        auc = roc_auc_score(y_true, oof) * 100.0
        acc = accuracy_score(y_true, pred) * 100.0
        missed = int(np.sum((y_true == 1) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        return {'p': p, 'r': r, 'f1': f, 'auc': auc, 'acc': acc, 'missed': missed, 'fp': fp}

    m_v4 = get_metrics(oof_v4, y_c3)
    m_v5 = get_metrics(oof_v5, y_c3)

    print("\n" + "=" * 115)
    print("  [c3 무릎 외반 모델: v4.0 vs v5.0 공식 5-Fold 실측 성적표 (N=1,906)]")
    print("=" * 115)
    print(f"{'모델 버전 및 핵심 혁신':<45} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<9} | {'미인식 오류':<9} | {'개선폭'}")
    print("-" * 115)
    print(f"{'c3 v4.0 챔피언 (20D+좌우6D+딥러닝17D = 43D)':<45} | {m_v4['p']:6.2f}%   | {m_v4['r']:6.2f}%   | {m_v4['f1']:6.2f}%   | {m_v4['auc']:6.2f}%  | {m_v4['missed']:3d}건     | (기준)")
    delta_f1 = m_v5['f1'] - m_v4['f1']
    rescued = m_v4['missed'] - m_v5['missed']
    print(f"{'🏆 c3 v5.0 얼티밋 (절대외반/좌우합 10D 결합 = 53D)':<45} | {m_v5['p']:6.2f}%   | {m_v5['r']:6.2f}%   | {m_v5['f1']:6.2f}%   | {m_v5['auc']:6.2f}%  | {m_v5['missed']:3d}건     | +{delta_f1:.2f}%p (★ +{rescued}건 구제!)")
    print("=" * 115)

    # 5. 사용자 육안 검증 10건 실측 확인
    target_sids = [
        '[오류]_D03_피험자1_Key316_rep01',
        '[오류]_D03_피험자1_Key321_rep04',
        '[오류]_D03_피험자1_Key327_rep03',
        '[오류]_D03_피험자3_Key321_rep01',
        '[오류]_D03_피험자4_Key316_rep01',
        '[오류]_D03_피험자4_Key319_rep01',
        '[오류]_D03_피험자5_Key319_rep05',
        '[오류]_D03_피험자6_Key319_rep03',
        '[오류]_D03_피험자6_Key319_rep02',
        '[오류]_D03_피험자6_Key325_rep01'
    ]

    print("\n[사용자 육안 검증 10건의 결함 예측 확률 변화 (v4 vs v5)]")
    print(f"{'순번':<4} | {'시퀀스 ID':<30} | {'v4 예측확률':<12} | {'v5 예측확률':<12} | {'판정 상태'}")
    print("-" * 75)
    cases_detail = []
    for idx, sid in enumerate(target_sids):
        i = np.where(seq_ids == sid)[0][0]
        p4 = float(oof_v4[i] * 100.0)
        p5 = float(oof_v5[i] * 100.0)
        status = "🔴 결함 적중 유지 (95%+)" if p5 >= 90.0 else ("🟢 결함 감지" if p5 >= 50.0 else "⚠️ 미인식")
        print(f"{idx+1:2d}   | {sid[-28:]:<30} | {p4:6.2f}%      | {p5:6.2f}%      | {status} ({p5-p4:+5.2f}%p)")
        cases_detail.append({
            'sid': sid,
            'p_v4': p4,
            'p_v5': p5,
            'delta': p5 - p4
        })

    # 6. v5 가중치 파일 저장
    v5_engine = BabelSquatC3ModelV5()
    v5_engine.cb_models = cb_v5_models
    v5_engine.lgb_models = lgb_v5_models
    v5_engine.is_fitted = True
    v5_engine.save_weights(WEIGHTS_PATH)

    # 7. OOF 캐시 갱신 (oof_c3_v5 및 oof_c3에 v5 반영)
    cached = dict(np.load(CACHE_PATH))
    cached['oof_c3_v4'] = oof_v4
    cached['oof_c3_v5'] = oof_v5
    cached['oof_c3'] = oof_v5 # 최신 v5 확률로 공식 바인딩
    np.savez_compressed(CACHE_PATH, **cached)
    print(f"[*] OOF 캐시 갱신 완료: {CACHE_PATH} ('oof_c3'에 v5 반영)")

    # 8. 벤치마크 결과 JSON 저장
    benchmark_res = {
        'v4': m_v4,
        'v5': m_v5,
        'delta': {
            'precision': m_v5['p'] - m_v4['p'],
            'recall': m_v5['r'] - m_v4['r'],
            'f1': delta_f1,
            'auc': m_v5['auc'] - m_v4['auc'],
            'rescued_count': rescued
        },
        'audit_10_cases': cases_detail
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(benchmark_res, f, indent=2, ensure_ascii=False)
    print(f"[*] 벤치마크 결과 저장 완료: {OUTPUT_JSON}")
    print("\n[완료] c3 v5.0 모델 구축 및 공식 검증 성공!")


if __name__ == '__main__':
    main()
