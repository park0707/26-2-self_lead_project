# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: compare_c2_v4_vs_v5_benchmark.py
Purpose: c2 v5.0 안면 포인트 가드(Face Gatekeeper) 장착 전 vs 장착 후 실측 성능 비교 및 다차원 감사
         - 규정 준수:
           1) 9명 피험자 완전 격리 (StratifiedGroupKFold, 5-Fold) 동일 분할
           2) 5대 전 시점(view1~view5) 2,605개 원천 64F 전수 대상
           3) 다차원 성능 감사: 서브모델 F1/AUC 및 5대 시점별 F1 대조 감사
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.babel_squat.c2_model_v5 import (
    extract_c2_v5_9d_features_from_seq,
    FEATURE_NAMES_9_C2,
    BabelSquatC2ModelV5
)

RAW_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\01_바벨_스쿼트"
MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
C2_UNGUARDED_PATH = r"D:\Data\피처_데이터\squat_c2_v5_pure_9d_feats.npz"
C2_GATED_PATH = r"D:\Data\피처_데이터\squat_c2_v5_gated_9d_feats.npz"
C2_V5_WEIGHTS_PATH = r"D:\src\models\second_models\babel_squat\c2_model_v5_weights.pkl"


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] c2(고개 정면) v5.0 얼굴 포인트 가드 장착 전 vs 장착 후 1:1 실측 성능 비교", flush=True)
    print("  ★ 가드 장착 전: 후방 시점(view1, view5)에서 MediaPipe 뒤통수 가짜 코(환각) 노이즈 그대로 허용", flush=True)
    print("  ★ 가드 장착 후: 체간 법선 벡터(n_z >= 0) 감지 시 후방 가짜 안면 피처 0.0 강제 차단 및 경추 협응 주력화", flush=True)
    print("  ★ 9명 피험자 완전 격리 (StratifiedGroupKFold, 5-Fold) & 5대 전 시점 2,605개 전수", flush=True)
    print("=" * 115, flush=True)

    master_d = np.load(MASTER_PATH, allow_pickle=True)
    seq_ids = master_d['sequence_ids']
    y_c2 = master_d['Y_defects_golden'][:, 1].astype(int)
    subjects = master_d['subjects']
    views = master_d['views']
    N = len(seq_ids)

    # 1. 가드 장착 전 피처 로드
    d_ungarded = np.load(C2_UNGUARDED_PATH, allow_pickle=True)
    X_ungarded = d_ungarded['X_c2_v5_9d']
    print(f"[*] [1/2] 가드 장착 전 피처 로드 완료: {X_ungarded.shape}", flush=True)

    # 2. 가드 장착 후 피처 전수 추출 및 저장
    print(f"[*] [2/2] 가드 장착 후 피처 추출 시작 (체간 법선 벡터 검증 가드 발동)...", flush=True)
    t0 = time.time()
    gated_list = []
    blocked_count = 0

    for i, sid in enumerate(seq_ids, 1):
        fp = os.path.join(RAW_DIR, f"{sid}.npz")
        kps_33 = np.load(fp)['kps_33_3d_64f']
        f9 = extract_c2_v5_9d_features_from_seq(kps_33)
        gated_list.append(f9)
        # 피처 8번이 0.0이면 가드 차단된 것
        if f9[7] == 0.0:
            blocked_count += 1
        if i % 500 == 0 or i == N:
            print(f"    - 진행률: {i:4d}/{N} ({i/N*100:5.1f}%) | 경과: {time.time()-t0:.1f}초", flush=True)

    X_gated = np.array(gated_list, dtype=np.float32)
    np.savez_compressed(
        C2_GATED_PATH,
        X_c2_v5_gated_9d=X_gated,
        feature_names=np.array(FEATURE_NAMES_9_C2),
        sequence_ids=seq_ids
    )
    print(f"[+] 가드 장착 후 피처 저장 완료: {C2_GATED_PATH} (총 {blocked_count}개 후방 샘플 가짜 안면 차단 성공!)", flush=True)

    # 3. 동일한 5-Fold 분할 상에서 두 모델 학습 및 OOF 사후확률 추론
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_model(X_data, label_name):
        print(f"\n[*] 5-Fold 학습 시작: {label_name}", flush=True)
        oof_preds = np.zeros(N, dtype=np.float32)
        trained_cb = []
        trained_lgb = []

        for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_data, y_c2, groups=subjects), 1):
            cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42 + fold, verbose=0)
            cb.fit(X_data[trn_idx], y_c2[trn_idx])

            clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42 + fold, verbose=-1, n_jobs=-1)
            clf.fit(X_data[trn_idx], y_c2[trn_idx])

            p_cb = cb.predict_proba(X_data[val_idx])[:, 1]
            p_lgb = clf.predict_proba(X_data[val_idx])[:, 1]
            p_fold = 0.60 * p_cb + 0.40 * p_lgb
            oof_preds[val_idx] = p_fold

            trained_cb.append(cb)
            trained_lgb.append(clf)

        return oof_preds, trained_cb, trained_lgb

    oof_ungarded, _, _ = evaluate_model(X_ungarded, "가드 장착 전 (가짜 코 노이즈 허용)")
    oof_gated, cb_gated, lgb_gated = evaluate_model(X_gated, "가드 장착 후 (후방 가짜 코 차단 클린)")

    # 4. 가중치 저장 (가드 장착 버전 공식 등록)
    v5_engine = BabelSquatC2ModelV5()
    v5_engine.cb_models = cb_gated
    v5_engine.lgb_models = lgb_gated
    v5_engine.is_fitted = True
    v5_engine.save_weights(C2_V5_WEIGHTS_PATH)

    # 5. 성능 평가
    def get_metrics(oof):
        p_bin = (oof >= 0.50).astype(int)
        return {
            'f1': f1_score(y_c2, p_bin, zero_division=0) * 100.0,
            'prec': precision_score(y_c2, p_bin, zero_division=0) * 100.0,
            'rec': recall_score(y_c2, p_bin, zero_division=0) * 100.0,
            'auc': roc_auc_score(y_c2, oof) * 100.0,
            'acc': accuracy_score(y_c2, p_bin) * 100.0
        }

    m_ung = get_metrics(oof_ungarded)
    m_gat = get_metrics(oof_gated)

    print("\n" + "=" * 115, flush=True)
    print("  [1] c2 v5.0 얼굴 포인트 가드 장착 전 vs 장착 후 실측 지표 비교", flush=True)
    print("=" * 115, flush=True)
    print(f"{'평가 지표':<20} | {'가드 장착 전 (노이즈 방치)':<25} | {'★ 가드 장착 후 (클린 마스킹)':<25} | {'변화폭':<15}", flush=True)
    print("-" * 115, flush=True)
    print(f"{'F1-Score':<20} | {m_ung['f1']:>23.2f}% | {m_gat['f1']:>23.2f}% | {m_gat['f1'] - m_ung['f1']:>+13.2f}%p", flush=True)
    print(f"{'ROC-AUC':<20} | {m_ung['auc']:>23.2f}% | {m_gat['auc']:>23.2f}% | {m_gat['auc'] - m_ung['auc']:>+13.2f}%p", flush=True)
    print(f"{'Precision (정밀도)':<20} | {m_ung['prec']:>23.2f}% | {m_gat['prec']:>23.2f}% | {m_gat['prec'] - m_ung['prec']:>+13.2f}%p", flush=True)
    print(f"{'Recall (재현율)':<20} | {m_ung['rec']:>23.2f}% | {m_gat['rec']:>23.2f}% | {m_gat['rec'] - m_ung['rec']:>+13.2f}%p", flush=True)
    print(f"{'Accuracy (정확도)':<20} | {m_ung['acc']:>23.2f}% | {m_gat['acc']:>23.2f}% | {m_gat['acc'] - m_ung['acc']:>+13.2f}%p", flush=True)

    print("\n" + "=" * 115, flush=True)
    print("  [2] 다차원 성능 감사: 5대 시점(view1~view5)별 세부 F1 1:1 대조 (가드 효과 검증)", flush=True)
    print("=" * 115, flush=True)
    view_list = ['view1', 'view2', 'view3', 'view4', 'view5']
    view_names = ['view1 (완전 후방)', 'view2 (우측 45도)', 'view3 (완전 정면)', 'view4 (좌측 45도)', 'view5 (우측 후방 45도)']

    p_ung_bin = (oof_ungarded >= 0.50).astype(int)
    p_gat_bin = (oof_gated >= 0.50).astype(int)

    print(f"{'시점 (카메라 앵글)':<25} | {'가드 전 F1':<15} | {'★ 가드 후 F1':<15} | {'F1 변화폭':<15} | {'안면 노이즈 가드 상태':<20}", flush=True)
    print("-" * 115, flush=True)

    for v, vn in zip(view_list, view_names):
        mask = (views == v)
        f_u = f1_score(y_c2[mask], p_ung_bin[mask], zero_division=0) * 100.0
        f_g = f1_score(y_c2[mask], p_gat_bin[mask], zero_division=0) * 100.0
        diff = f_g - f_u
        diff_str = f"{diff:>+13.2f}%p"
        guard_status = "★ 가드 차단 (환각 방지)" if v in ['view1', 'view5'] else "정상 안면 시선 통과"
        print(f"{vn:<25} | {f_u:>13.2f}% | {f_g:>13.2f}% | {diff_str} | {guard_status}", flush=True)

    print("\n" + "=" * 115, flush=True)
    print("[+] 가드 장착 비교 벤치마크 완료 및 공식 가중치 저장 완료:")
    print(f"  • 공식 가중치 파일: {C2_V5_WEIGHTS_PATH}")
    print("=" * 115, flush=True)


if __name__ == '__main__':
    main()
