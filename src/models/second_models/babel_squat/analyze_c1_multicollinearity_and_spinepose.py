# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: analyze_c1_multicollinearity_and_spinepose.py
Purpose: c1(척추의 중립 위반) 피처 간 다중공선성/간섭 관계 전수 분석
         및 SpinePose 피처셋의 성능 기여도(Ablation Study) 실측 검증
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c1_model_v6 import V6A_KEEP_INDICES, C1_V6A_FEATURE_NAMES_29
from src.models.second_models.babel_squat.c1_model_v2 import BabelSquatC1ModelV2
from src.models.second_models.babel_squat.c1_model_v4 import V4_LEGACY_KEEP_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_c1_features():
    """c1 전체 46D 피처 로드 (v6 29D + 딥러닝 17D)"""
    ds = SquatPilotDatasetLoader.load_dataset()
    y_c1 = ds['Y_defects'][:, 0].astype(int)
    subjects = ds['subjects']
    N_total = len(y_c1)

    # 1. 29D v6 기하학 피처
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

    # 34D 원본 (v6A 적용 전)
    X_34d = np.hstack([X_legacy_18d, X_v4_spine_10d, X_v5_spinepose_6d])
    X_v6_29d = X_34d[:, V6A_KEEP_INDICES]

    # 2. 17D 딥러닝 시계열 임베딩 피처
    deep_npz = os.path.join(CURRENT_DIR, 'squat_c1_deep_temporal_feats.npz')
    deep_data = np.load(deep_npz, allow_pickle=True)
    deep_embeddings = deep_data['deep_embeddings'].astype(np.float32)
    deep_probs = deep_data['deep_probs'].astype(np.float32)
    X_deep_17d = np.hstack([deep_embeddings, deep_probs[:, None]])

    X_46d = np.hstack([X_v6_29d, X_deep_17d]).astype(np.float32)

    c1_deep_names = [f"c1_deep_emb_{i}" for i in range(16)] + ["c1_deep_prob"]
    feature_names = C1_V6A_FEATURE_NAMES_29 + c1_deep_names

    return {
        'X_46d': X_46d,
        'X_legacy_18d': X_legacy_18d,
        'X_v4_spine_10d': X_v4_spine_10d,
        'X_v5_spinepose_6d': X_v5_spinepose_6d,
        'X_v6_29d': X_v6_29d,
        'X_deep_17d': X_deep_17d,
        'y_c1': y_c1,
        'subjects': subjects,
        'feature_names': feature_names
    }


def analyze_multicollinearity(X, feature_names):
    """피처 간 상관계수 전수 분석 (|r| >= 0.80 고상관 쌍 추출)"""
    print("\n" + "=" * 110)
    print("  [Step 1] c1 피처 간 상관계수(Correlation) 및 다중공선성(Multicollinearity) 전수 분석")
    print("=" * 110)

    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()

    high_corr_pairs = []
    num_feats = len(feature_names)

    for i in range(num_feats):
        for j in range(i + 1, num_feats):
            r = corr_matrix.iloc[i, j]
            if r >= 0.70:
                high_corr_pairs.append((feature_names[i], feature_names[j], r))

    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"[*] 총 {num_feats}개 피처 중 |r| >= 0.70 이상인 고상관 쌍: 총 {len(high_corr_pairs)}쌍 발견!")
    print("-" * 110)
    print(f"{'피처 A':<38} <---> {'피처 B':<38} | {'상관계수 |r|':<12} | {'간섭 위험도'}")
    print("-" * 110)

    for f_a, f_b, r in high_corr_pairs:
        risk = "🚨 극심한 중복/간섭 (|r|>=0.90)" if r >= 0.90 else "⚠️ 높은 상관 (|r|>=0.80)" if r >= 0.80 else "주의 (0.70~0.80)"
        print(f"{f_a:<38} <---> {f_b:<38} | r = {r:6.3f}     | {risk}")
    print("=" * 110)
    return high_corr_pairs


def evaluate_5fold(X, y, subjects, desc=""):
    """5-Fold StratifiedGroupKFold 엄격 실측 평가"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    cb_importances = np.zeros(X.shape[1], dtype=np.float32)

    for trn, val in sgkf.split(X, y, groups=subjects):
        cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb.fit(X[trn], y[trn])
        clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X[trn], y[trn])

        p_cb = cb.predict_proba(X[val])[:, 1]
        p_lgb = clf.predict_proba(X[val])[:, 1]
        oof[val] = 0.60 * p_cb + 0.40 * p_lgb
        cb_importances += cb.get_feature_importance() / 5.0

    pred = (oof >= 0.50).astype(int)
    p = precision_score(y, pred) * 100
    r = recall_score(y, pred) * 100
    f = f1_score(y, pred) * 100
    auc = roc_auc_score(y, oof) * 100
    missed = int(np.sum((y == 1) & (pred == 0)))

    return {
        'desc': desc,
        'precision': p, 'recall': r, 'f1': f, 'auc': auc,
        'missed': missed, 'importances': cb_importances
    }


def ablation_study_spinepose(data):
    """SpinePose 기여도 및 피처군별 소거(Ablation) 실측 검증"""
    print("\n" + "=" * 115)
    print("  [Step 2] SpinePose 피처군 소거 실험 (Ablation Study) 및 실질적 기여도 검증")
    print("=" * 115)

    y = data['y_c1']
    subj = data['subjects']

    # 1. Full 46D (v7.0 기준선)
    res_full = evaluate_5fold(data['X_46d'], y, subj, "v7.0 전체 피처셋 (46D: SpinePose 포함)")

    # 2. SpinePose 5D 완전 제거 (41D: v6 29D 중 v5 5개 제거 -> 24D + 딥러닝 17D = 41D)
    # v6_29d에서 v5 피처가 어디에 있는지 확인
    # v6_29d는 X_legacy_18d(18) + v4(8) + v5(3 or 5)
    spinepose_names = [n for n in data['feature_names'] if 'c1_v5_' in n]
    non_spinepose_idx = [i for i, n in enumerate(data['feature_names']) if 'c1_v5_' not in n]
    X_no_spinepose = data['X_46d'][:, non_spinepose_idx]
    res_no_spinepose = evaluate_5fold(X_no_spinepose, y, subj, "SpinePose 제거 모델 (41D: SpinePose 전면 배제)")

    # 3. SpinePose 피처만 단독 사용했을 때 (SpinePose 5D)
    spinepose_idx = [i for i, n in enumerate(data['feature_names']) if 'c1_v5_' in n]
    X_spinepose_only = data['X_46d'][:, spinepose_idx]
    res_spinepose_only = evaluate_5fold(X_spinepose_only, y, subj, f"SpinePose 단독 모델 ({len(spinepose_idx)}D)")

    # 4. v4 척추 피처군(10D) 제거 실험 (SpinePose v5 + 레거시 18D + 딥러닝 17D)
    v4_idx = [i for i, n in enumerate(data['feature_names']) if 'c1_v4_' in n]
    non_v4_idx = [i for i, n in enumerate(data['feature_names']) if 'c1_v4_' not in n]
    X_no_v4 = data['X_46d'][:, non_v4_idx]
    res_no_v4 = evaluate_5fold(X_no_v4, y, subj, f"v4 척추기하 제거 모델 ({len(non_v4_idx)}D)")

    # 5. 순수 레거시 18D (v2 시절 피처)
    res_legacy = evaluate_5fold(data['X_legacy_18d'], y, subj, "순수 레거시 18D (SpinePose 도입 이전 기준선)")

    ablations = [res_full, res_no_spinepose, res_no_v4, res_spinepose_only, res_legacy]

    print(f"{'실험군 / 모델 구성':<48} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<9} | {'미인식 오류'}")
    print("-" * 115)
    base_f = res_full['f1']
    for r in ablations:
        delta = f"({r['f1'] - base_f:+5.2f}%p)" if r != res_full else "(기준)"
        print(f"{r['desc']:<48} | {r['precision']:6.2f}%   | {r['recall']:6.2f}%   | {r['f1']:6.2f}% {delta} | {r['auc']:6.2f}%  | {r['missed']:3d}건")
    print("=" * 115)

    # SpinePose 기여도 결론
    delta_sp = res_full['f1'] - res_no_spinepose['f1']
    print(f"\n[*] SpinePose 순수 기여도 실측 판정: F1 {delta_sp:+5.2f}%p")
    if delta_sp > 0:
        print(f"    ▶ SpinePose를 탑재했을 때 F1이 +{delta_sp:.2f}%p 유의미하게 상승하며, 결함 {res_no_spinepose['missed'] - res_full['missed']}건을 추가 구제함을 실측 확인!")
    else:
        print(f"    ▶ SpinePose 피처가 기존 피처와 중복되거나 간섭을 일으켜 F1이 {delta_sp:.2f}%p 저해되고 있음을 확인!")

    # 6. 피처 중요도(Feature Importance) TOP 15 분석
    print("\n" + "=" * 115)
    print("  [Step 3] c1 46D 전체 피처 중요도(Feature Importance) TOP 15 랭킹")
    print("=" * 115)
    imp = res_full['importances']
    imp_ranking = sorted(zip(data['feature_names'], imp), key=lambda x: x[1], reverse=True)

    for rank, (name, score) in enumerate(imp_ranking[:15], 1):
        is_sp = "★ [SpinePose]" if 'c1_v5_' in name else "[Deep Net]" if 'c1_deep_' in name else "[v4 척추]" if 'c1_v4_' in name else "[Legacy]"
        print(f"  #{rank:02d} {is_sp:<14} {name:<42} | 중요도: {score:6.2f}%")
    print("=" * 115)

    return imp_ranking, ablations


def optimize_c1_features_diet(data, high_corr_pairs, imp_ranking):
    """
    고상관 간섭 피처 소거(Diet)를 통한 c1 최적화 탐색
    - |r| >= 0.85인 쌍 중에서 중요도가 더 낮은 피처를 1개씩 소거하며 F1 상승 여부 실측
    """
    print("\n" + "=" * 115)
    print("  [Step 4] 고상관 간섭 피처 선별 소거(Diet)를 통한 c1 최적화 실측 탐색")
    print("=" * 115)

    y = data['y_c1']
    subj = data['subjects']
    names = data['feature_names']
    imp_dict = dict(imp_ranking)

    # 0.85 이상 고상관 쌍 중 중요도 낮은 쪽 소거 후보군 추출
    candidates_to_drop = set()
    for fa, fb, r in high_corr_pairs:
        if r >= 0.85:
            # 둘 중 중요도가 더 낮은 피처 선택
            worse = fa if imp_dict.get(fa, 0) < imp_dict.get(fb, 0) else fb
            candidates_to_drop.add(worse)

    print(f"[*] 다중공선성(|r|>=0.85) 간섭 피처 중 중요도 하위 소거 후보: {len(candidates_to_drop)}개 피처")
    for c in candidates_to_drop:
        print(f"    - 후보: {c} (중요도: {imp_dict.get(c, 0):.2f}%)")

    # 소거 실험
    keep_indices = [i for i, n in enumerate(names) if n not in candidates_to_drop]
    X_diet = data['X_46d'][:, keep_indices]
    res_diet = evaluate_5fold(X_diet, y, subj, f"최적 다이어트 모델 ({len(keep_indices)}D: {len(candidates_to_drop)}개 간섭 소거)")

    res_base = evaluate_5fold(data['X_46d'], y, subj, "v7.0 기존 모델 (46D 전체)")

    print("\n" + "-" * 115)
    print(f"{'모델 버전':<48} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<9} | {'미인식 오류'}")
    print("-" * 115)
    print(f"{res_base['desc']:<48} | {res_base['precision']:6.2f}%   | {res_base['recall']:6.2f}%   | {res_base['f1']:6.2f}% (기준)  | {res_base['auc']:6.2f}%  | {res_base['missed']:3d}건")
    delta_f = res_diet['f1'] - res_base['f1']
    delta_str = f"({delta_f:+5.2f}%p)"
    print(f"{res_diet['desc']:<48} | {res_diet['precision']:6.2f}%   | {res_diet['recall']:6.2f}%   | {res_diet['f1']:6.2f}% {delta_str} | {res_diet['auc']:6.2f}%  | {res_diet['missed']:3d}건")
    print("-" * 115)

    if delta_f > 0:
        print(f"🏆 [최적화 대성공] 간섭 피처 {len(candidates_to_drop)}개를 덜어내자 F1이 {delta_f:+5.2f}%p 추가 상승했습니다!")
    else:
        print(f"[*] 현재 피처셋이 이미 앙상블 트리에서 유의미하게 활용되고 있으며, 과도한 소거는 정보를 일부 손실시킴을 확인.")


if __name__ == '__main__':
    data = load_all_c1_features()
    high_corr = analyze_multicollinearity(data['X_46d'], data['feature_names'])
    ranking, ablations = ablation_study_spinepose(data)
    optimize_c1_features_diet(data, high_corr, ranking)
