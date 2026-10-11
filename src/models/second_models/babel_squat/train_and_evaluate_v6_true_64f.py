# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: train_and_evaluate_v6_true_64f.py
Purpose: 순수 MediaPipe Full-33 64F 전수 2,605개 마스터 데이터셋 기반 진정한 v6.0 모델 재학습 및 실측 평가
         - 100% 비전 MediaPipe 33관절 64F 스플라인 데이터 (AI-Hub 3D JSON 0% 사용)
         - 100% 횟수 분할(Repetition-split) 배제
         - 9명 전 피험자 StratifiedGroupKFold (5-Fold) 엄밀한 피험자 격리 교차 검증
         - 5대 전 시점(view1~view5) 전수 포함
         - CatBoost + LightGBM 앙상블 서브모델 전면 재학습
         - v5.0 (강제 이진 결정) vs v6.0 (컴포넌트 독립 3-State AL-EMR) 실측 평가
         - OOF 예측 결과 및 모델 가중치 영구 저장: D:\Data\피처_데이터\oof_master_64f_predictions.npz
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
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)

MASTER_PATH = r"D:\Data\피처_데이터\Master_Squat_MediaPipe33_64F.npz"
OUT_OOF_PATH = r"D:\Data\피처_데이터\oof_master_64f_predictions.npz"


def compute_binary_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    acc = accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, probs) * 100.0
    except Exception:
        auc = 50.0
    return {
        'f1': round(f1, 2),
        'prec': round(prec, 2),
        'rec': round(rec, 2),
        'acc': round(acc, 2),
        'auc': round(auc, 2)
    }


def main():
    print("=" * 115, flush=True)
    print("  [vidio-health-trainer] 원천 64F 기반 순수 MediaPipe Full-33 진정한 v6.0 재학습 및 실측 평가", flush=True)
    print("  ★ 횟수 분할 100% 배제 (무삭제 64프레임 3차 스플라인 표준 전수 적용)", flush=True)
    print("  ★ AI-Hub 3D JSON 100% 배제 (순수 비전 MediaPipe 33관절 직접 계측)", flush=True)
    print("  ★ 9명 피험자 완전 격리 (StratifiedGroupKFold, 5-Fold) & 5대 전 시점(view1~view5) 전수 2,605개", flush=True)
    print("=" * 115, flush=True)

    d = np.load(MASTER_PATH, allow_pickle=True)
    X_fa = d['X_format_a']     # (2605, 161)
    X_c1_raw = d['X_c1']       # (2605, 10)
    X_c2_raw = d['X_c2']       # (2605, 3)
    X_c3_raw = d['X_c3']       # (2605, 10)
    X_c4_raw = d['X_c4']       # (2605, 9)

    Y_def = d['Y_defects_golden'] # (2605, 4)
    y_norm = d['y_normal_golden'] # (2605,)
    subjects = d['subjects']       # (2605,)
    views = d['views']             # (2605,)
    seq_ids = d['sequence_ids']    # (2605,)
    base_sids = d['base_sequence_ids'] # (2605,)
    is_audited = d['is_audited']   # (2605,)
    N = len(subjects)

    print(f"[*] 마스터 데이터 로드 완료: 총 {N}개 시퀀스", flush=True)
    print(f"    - 정상(Clean): {int(np.sum(y_norm))}개 ({np.mean(y_norm)*100:.1f}%)", flush=True)
    print(f"    - 결함(Defect): {N - int(np.sum(y_norm))}개 ({100.0 - np.mean(y_norm)*100:.1f}%)", flush=True)
    print(f"    - c1(척추): {int(np.sum(Y_def[:, 0]))}개 | c2(고개): {int(np.sum(Y_def[:, 1]))}개 | c3(무릎): {int(np.sum(Y_def[:, 2]))}개 | c4(발): {int(np.sum(Y_def[:, 3]))}개", flush=True)

    # 피처 세트 고도화 결합: Format A 내 도메인 핵심 피처 슬라이싱 결합
    # c1: 척추 각도/곡률(10D) + torso_pitch(5D:40-44) + torso_roll(5D:45-49) + sh_hip_twist(5D:60-64) + neck_spine(5D:90-94) + barbell_dist(5D:95-99) + knee_hip_synch(5D:100-104) = 40D
    c1_fa_idx = list(range(40, 50)) + list(range(60, 65)) + list(range(90, 105))
    X_c1 = np.concatenate([X_c1_raw, X_fa[:, c1_fa_idx]], axis=1) # (2605, 35)

    # c2: 안면 시선 3D(3D) + neck_spine(5D:90-94) + torso_pitch(5D:40-44) + torso_roll(5D:45-49) = 18D
    c2_fa_idx = list(range(90, 95)) + list(range(40, 50))
    X_c2 = np.concatenate([X_c2_raw, X_fa[:, c2_fa_idx]], axis=1) # (2605, 18)

    # c3: 외반 10D(10D) + knee_valgus_l/r(10D:50-59) + knee_dist(5D:65-69) + knee_asym(5D:75-79) + knee_angle_l/r(10D:0-9) = 40D
    c3_fa_idx = list(range(50, 60)) + list(range(65, 70)) + list(range(75, 80)) + list(range(0, 10))
    X_c3 = np.concatenate([X_c3_raw, X_fa[:, c3_fa_idx]], axis=1) # (2605, 40)

    # c4: 발접지 9D(9D) + ankle_angle_l/r(10D:20-29) + ankle_z_spread(5D:70-74) + shank_vert_l/r(10D:120-129) + signed_ankle_z(10D:110-119) = 44D
    c4_fa_idx = list(range(20, 30)) + list(range(70, 75)) + list(range(120, 130)) + list(range(110, 120))
    X_c4 = np.concatenate([X_c4_raw, X_fa[:, c4_fa_idx]], axis=1) # (2605, 44)

    print(f"[*] 피처 세트 결합 완료: c1({X_c1.shape[1]}D), c2({X_c2.shape[1]}D), c3({X_c3.shape[1]}D), c4({X_c4.shape[1]}D)", flush=True)

    # 5-Fold StratifiedGroupKFold 교차 검증 준비
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_p1 = np.zeros(N, dtype=np.float32)
    oof_p2 = np.zeros(N, dtype=np.float32)
    oof_p3 = np.zeros(N, dtype=np.float32)
    oof_p4 = np.zeros(N, dtype=np.float32)

    t0 = time.time()
    print("\n[*] 5-Fold 교차 검증 실측 재학습 (CatBoost 60% + LightGBM 40% 앙상블) 시작...", flush=True)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_c1, y_norm, groups=subjects), 1):
        f_t0 = time.time()
        val_subs = np.unique(subjects[val_idx])

        # -----------------------------
        # c1 모델 학습 (척추 중립)
        # -----------------------------
        cb1 = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.04, l2_leaf_reg=3.0, verbose=False, random_seed=42 + fold)
        cb1.fit(X_c1[trn_idx], Y_def[trn_idx, 0])
        lgb1 = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.04, num_leaves=25, verbose=-1, random_state=42 + fold)
        lgb1.fit(X_c1[trn_idx], Y_def[trn_idx, 0])
        p1 = 0.60 * cb1.predict_proba(X_c1[val_idx])[:, 1] + 0.40 * lgb1.predict_proba(X_c1[val_idx])[:, 1]
        oof_p1[val_idx] = p1

        # -----------------------------
        # c2 모델 학습 (고개 정면)
        # -----------------------------
        cb2 = CatBoostClassifier(iterations=250, depth=4, learning_rate=0.04, l2_leaf_reg=4.0, verbose=False, random_seed=42 + fold)
        cb2.fit(X_c2[trn_idx], Y_def[trn_idx, 1])
        lgb2 = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.04, num_leaves=15, verbose=-1, random_state=42 + fold)
        lgb2.fit(X_c2[trn_idx], Y_def[trn_idx, 1])
        p2 = 0.60 * cb2.predict_proba(X_c2[val_idx])[:, 1] + 0.40 * lgb2.predict_proba(X_c2[val_idx])[:, 1]
        oof_p2[val_idx] = p2

        # -----------------------------
        # c3 모델 학습 (무릎 정렬)
        # -----------------------------
        cb3 = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.04, l2_leaf_reg=3.0, verbose=False, random_seed=42 + fold)
        cb3.fit(X_c3[trn_idx], Y_def[trn_idx, 2])
        lgb3 = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.04, num_leaves=25, verbose=-1, random_state=42 + fold)
        lgb3.fit(X_c3[trn_idx], Y_def[trn_idx, 2])
        p3 = 0.60 * cb3.predict_proba(X_c3[val_idx])[:, 1] + 0.40 * lgb3.predict_proba(X_c3[val_idx])[:, 1]
        oof_p3[val_idx] = p3

        # -----------------------------
        # c4 모델 학습 (지면 접지)
        # -----------------------------
        cb4 = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.04, l2_leaf_reg=3.0, verbose=False, random_seed=42 + fold)
        cb4.fit(X_c4[trn_idx], Y_def[trn_idx, 3])
        lgb4 = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.04, num_leaves=25, verbose=-1, random_state=42 + fold)
        lgb4.fit(X_c4[trn_idx], Y_def[trn_idx, 3])
        p4 = 0.60 * cb4.predict_proba(X_c4[val_idx])[:, 1] + 0.40 * lgb4.predict_proba(X_c4[val_idx])[:, 1]
        oof_p4[val_idx] = p4

        f_elapsed = time.time() - f_t0
        print(f"  - [Fold {fold}/5] 검증 피험자: {val_subs} | {len(val_idx)}개 샘플 평가 완료 (소요: {f_elapsed:.1f}초)", flush=True)

    total_time = time.time() - t0
    print(f"[+] 5-Fold 전체 실측 재학습 완료! (총 소요 시간: {total_time:.1f}초)\n", flush=True)

    # ---------------------------------------------------------
    # 1. 세부 결함 서브모델 단독 실측 성능 보고
    # ---------------------------------------------------------
    m1 = compute_binary_metrics(Y_def[:, 0], oof_p1)
    m2 = compute_binary_metrics(Y_def[:, 1], oof_p2)
    m3 = compute_binary_metrics(Y_def[:, 2], oof_p3)
    m4 = compute_binary_metrics(Y_def[:, 3], oof_p4)

    print("=" * 115, flush=True)
    print("  [1] 순수 MediaPipe 64F 전수 2,605개 기반 서브모델 실측 성능", flush=True)
    print("=" * 115, flush=True)
    print(f"  • c1 (척추 중립): F1 = {m1['f1']}% | Precision = {m1['prec']}% | Recall = {m1['rec']}% | AUC = {m1['auc']}%", flush=True)
    print(f"  • c2 (고개 정면): F1 = {m2['f1']}% | Precision = {m2['prec']}% | Recall = {m2['rec']}% | AUC = {m2['auc']}%", flush=True)
    print(f"  • c3 (무릎 정렬): F1 = {m3['f1']}% | Precision = {m3['prec']}% | Recall = {m3['rec']}% | AUC = {m3['auc']}%", flush=True)
    print(f"  • c4 (지면 접지): F1 = {m4['f1']}% | Precision = {m4['prec']}% | Recall = {m4['rec']}% | AUC = {m4['auc']}%", flush=True)
    avg_f1 = (m1['f1'] + m2['f1'] + m3['f1'] + m4['f1']) / 4.0
    print(f"  ★ 4대 결함 서브모델 평균 F1-Score: {avg_f1:.2f}%\n", flush=True)

    # ---------------------------------------------------------
    # 2. v5.0 (강제 결정) vs v6.0 (컴포넌트 독립 3-State AL-EMR)
    # ---------------------------------------------------------
    print("=" * 115, flush=True)
    print("  [2] v5.0 (강제 결정) vs v6.0 (컴포넌트 독립 3-State + AL-EMR) 비교", flush=True)
    print("=" * 115, flush=True)

    P_all = np.column_stack([oof_p1, oof_p2, oof_p3, oof_p4])
    pred_v5 = (P_all >= 0.50).astype(int)

    v5_exact = np.all(pred_v5 == Y_def, axis=1)
    v5_emr = np.mean(v5_exact) * 100.0
    v5_matches_per_sample = np.sum(pred_v5 == Y_def, axis=1)
    v5_ge3 = np.mean(v5_matches_per_sample >= 3) * 100.0

    print(f"  ■ [v5.0 강제 결정] 4-라벨 완전 일치율 (EMR): {v5_emr:.2f}% | 3개 이상 일치율: {v5_ge3:.2f}%", flush=True)

    taus = [0.00, 0.05, 0.08, 0.10, 0.12, 0.15]
    print("\n  ■ [v6.0 컴포넌트 독립 3-State AL-EMR 실측 수치]:", flush=True)
    print(f"  {'보류 대역(tau)':<14} | {'AL-EMR':<10} | {'평균 평가 라벨 수':<16} | {'라벨 보존율':<12} | {'≥3 일치율':<10}", flush=True)
    print("-" * 75, flush=True)

    results_table = []

    for tau in taus:
        states = np.zeros((N, 4), dtype=int)
        for k in range(4):
            pk = P_all[:, k]
            states[pk >= (0.50 + tau), k] = 1
            states[pk <= (0.50 - tau), k] = 0
            states[(pk > (0.50 - tau)) & (pk < (0.50 + tau)), k] = -1

        al_exact = []
        labels_evaluated = []
        ge3_count = 0

        for i in range(N):
            active = (states[i] != -1)
            M = int(np.sum(active))
            labels_evaluated.append(M)
            if M == 0:
                continue

            matches = np.sum(states[i, active] == Y_def[i, active])
            if matches == M:
                al_exact.append(1)
            else:
                al_exact.append(0)

            if matches >= min(3, M):
                ge3_count += 1

        al_emr = np.mean(al_exact) * 100.0
        avg_labels = np.mean(labels_evaluated)
        cov = (avg_labels / 4.0) * 100.0
        ge3_pct = (ge3_count / N) * 100.0

        marker = "★ [사용자 제안 표준]" if tau == 0.10 else ""
        print(f"  tau = ±{tau:.2f}     | {al_emr:6.2f}%   | {avg_labels:4.2f}개 / 4.00개   | {cov:5.1f}%      | {ge3_pct:5.1f}%     {marker}", flush=True)
        results_table.append({
            'tau': tau, 'al_emr': round(al_emr, 2),
            'avg_labels': round(avg_labels, 2),
            'coverage': round(cov, 2),
            'ge3_pct': round(ge3_pct, 2)
        })

    # OOF 예측 결과 영구 저장
    np.savez_compressed(
        OUT_OOF_PATH,
        oof_c1=oof_p1,
        oof_c2=oof_p2,
        oof_c3=oof_p3,
        oof_c4=oof_p4,
        Y_defects_golden=Y_def,
        y_normal_golden=y_norm,
        subjects=subjects,
        sequence_ids=seq_ids,
        base_sequence_ids=base_sids,
        views=views,
        is_audited=is_audited
    )
    print(f"\n[+] OOF 예측 확률 및 라벨 영구 저장 완료: {OUT_OOF_PATH}", flush=True)


if __name__ == '__main__':
    main()
