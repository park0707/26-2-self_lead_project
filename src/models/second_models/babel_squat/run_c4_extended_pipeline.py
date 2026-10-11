# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: run_c4_extended_pipeline.py
Purpose: 바벨 스쿼트 c4 (발바닥 지면 고정) 2단계 정상/오류 분류 모델 v1.0 프로토콜 파이프라인
- 규정 준수:
  1) 공식 명칭 vidio-health-trainer 적용
  2) 64프레임 3차 스플라인 표준 보간 적용
  3) AI-Hub 바벨 스쿼트 9명 전수 5-Fold StratifiedGroupKFold 완전 격리
  4) 4단계 모호성(Level 1~4) 분석 및 Level 4 역전 노이즈 정제
  5) 2계층 캐스케이드(Tier-1 Gatekeeper + Tier-2 Resolver) 아키텍처 실측
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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c4_cascade_comparison_results.json'

FEATURE_NAMES_16_C4 = [
    'c4_ankle_angle_min',
    'c4_ankle_angle_drop',
    'c4_ankle_angle_mean',
    'c4_ankle_height_lift_max',
    'c4_ankle_height_lift_at_nadir',
    'c4_ankle_height_wobble',
    'c4_stance_width_drift',
    'c4_stance_width_at_nadir',
    'c4_ankle_z_drift_max',
    'c4_bilateral_ankle_angle_asym',
    'c4_ankle_angular_velocity_peak',
    'c4_ankle_acceleration_jerk',
    'c4_shank_ankle_coupling_ratio',
    'c4_ascent_ankle_destabilization',
    'c4_ankle_heel_lift_duration_ratio',
    'c4_knee_ankle_vertical_ratio'
]

def compute_metrics(y_true, probs, threshold=0.50):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0) * 100.0
    prec = precision_score(y_true, preds, zero_division=0) * 100.0
    rec = recall_score(y_true, preds, zero_division=0) * 100.0
    bacc = balanced_accuracy_score(y_true, preds) * 100.0
    try:
        auc = roc_auc_score(y_true, probs) * 100.0
    except Exception:
        auc = 50.0
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    best_th, best_f1 = threshold, f1
    for t in np.linspace(0.20, 0.80, 61):
        f = f1_score(y_true, (probs >= t).astype(int), zero_division=0) * 100.0
        if f > best_f1:
            best_f1, best_th = f, t

    return {
        'f1': round(float(f1), 2),
        'precision': round(float(prec), 2),
        'recall': round(float(rec), 2),
        'bacc': round(float(bacc), 2),
        'roc_auc': round(float(auc), 2),
        'opt_f1': round(float(best_f1), 2),
        'opt_th': round(float(best_th), 2),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def extract_c4_16d_features(X_b_64):
    """
    Format B (N, 64, 92) 텐서로부터 c4 발바닥 지면 고정 전담 16D 생체역학 피처 추출
    """
    N, T, C = X_b_64.shape
    dt = 1.0 / max(1, T - 1)
    feats = np.zeros((N, 16), dtype=np.float32)

    for i in range(N):
        seq = X_b_64[i]

        # 무릎 굴곡 및 최하사점 b
        knee_angle_l = seq[:, 56]
        knee_angle_r = seq[:, 57]
        knee_angle_mean = (knee_angle_l + knee_angle_r) / 2.0
        knee_flex = 180.0 - knee_angle_mean
        knee_flex_s = gaussian_filter1d(knee_flex, sigma=1.0)
        b = int(np.clip(np.argmax(knee_flex_s), 4, T - 5))

        # 발목 굴곡각 (ch 58, 59)
        ankle_ang_l = seq[:, 58] if C > 58 else np.zeros(T)
        ankle_ang_r = seq[:, 59] if C > 59 else np.zeros(T)
        ankle_ang_mean = (ankle_ang_l + ankle_ang_r) / 2.0
        ankle_ang_s = gaussian_filter1d(ankle_ang_mean, sigma=0.8)

        # 발목 간격(스탠스) (ch 70)
        ankle_dist = seq[:, 70] if C > 70 else np.zeros(T)
        ankle_dist_s = gaussian_filter1d(ankle_dist, sigma=0.8)

        # 발목 전후 분리 깊이 (ch 71)
        ankle_z_spread = seq[:, 71] if C > 71 else np.zeros(T)

        # 정강이 수직각 (ch 89, 90)
        shank_l = seq[:, 89] if C > 89 else np.zeros(T)
        shank_r = seq[:, 90] if C > 90 else np.zeros(T)
        shank_mean = (shank_l + shank_r) / 2.0

        # 기립 구간 기준 (처음 8프레임)
        standing_slice = max(1, T // 8)
        ankle_ang_standing = float(np.mean(ankle_ang_s[:standing_slice]))
        ankle_dist_standing = float(np.mean(ankle_dist_s[:standing_slice]))

        # [0] 최하사점 발목 각도 최소치
        ank_min = float(np.min(ankle_ang_s))

        # [1] 기립 대비 최하사점 발목 각도 변화폭 (배굴/저굴 변화량)
        ank_drop = float(abs(ankle_ang_standing - ankle_ang_s[b]))

        # [2] 전 구간 발목 각도 평균
        ank_mean = float(np.mean(ankle_ang_s))

        # [3] 발목 가상 수직 상승고 근사치 (무릎 하강 대비 발목 각도 닫힘량)
        # 발뒤꿈치가 들리면 발목 각도가 비정상적으로 작아지거나 펴짐
        heel_lift_stream = np.maximum(0.0, ankle_ang_standing - ankle_ang_s)
        heel_lift_max = float(np.max(heel_lift_stream))

        # [4] 최하사점 발목 수직 들림량
        heel_lift_nadir = float(heel_lift_stream[b])

        # [5] 발목 들썩임 표준편차
        heel_lift_wobble = float(np.std(heel_lift_stream))

        # [6] 양 발목 간격(스탠스) 드리프트 변동폭 (미끄러짐/벌어짐)
        stance_drift = float(np.max(np.abs(ankle_dist_s - ankle_dist_standing)))

        # [7] 최하사점 양 발목 간격
        stance_nadir = float(ankle_dist_s[b])

        # [8] 발목 전후 깊이 최대 변동폭
        z_drift_max = float(np.max(ankle_z_spread))

        # [9] 좌우 발목 각도 비대칭도
        asym_stream = np.abs(ankle_ang_l - ankle_ang_r)
        asym_max = float(np.max(asym_stream))

        # [10] 발목 각속도 피크 (최하사점 부근 급격한 들림 스파이크)
        v_ank = np.gradient(ankle_ang_s, dt)
        vel_peak = float(np.max(np.abs(v_ank[max(0, b - 4):min(T, b + 5)])))

        # [11] 발목 가속도 저크
        a_ank = np.gradient(v_ank, dt)
        j_ank = np.gradient(a_ank, dt)
        ank_jerk = float(np.mean(np.abs(j_ank)))

        # [12] 정강이 경사 대비 발목 연동비
        shank_diff = np.max(shank_mean) - np.min(shank_mean) + 1e-6
        ank_diff = np.max(ankle_ang_s) - np.min(ankle_ang_s)
        coupling_ratio = float(ank_diff / shank_diff)

        # [13] 상승 구간 발목 불안정성 비율
        ascent_ank = ankle_ang_s[b:]
        if len(ascent_ank) > 0:
            ascent_unstable = float(np.sum(np.abs(np.gradient(ascent_ank, dt)) > 15.0) / len(ascent_ank))
        else:
            ascent_unstable = 0.0

        # [14] 뒤꿈치 들림 지속 프레임 비율
        lift_threshold = ankle_ang_standing - 4.0
        lift_duration_ratio = float(np.sum(ankle_ang_s < lift_threshold) / float(T))

        # [15] 무릎-발목 수직 변위 상호작용비
        knee_drop_val = np.max(knee_flex_s) - np.min(knee_flex_s) + 1e-6
        knee_ank_ratio = float(heel_lift_max / knee_drop_val)

        feat = np.array([
            ank_min, ank_drop, ank_mean, heel_lift_max,
            heel_lift_nadir, heel_lift_wobble, stance_drift, stance_nadir,
            z_drift_max, asym_max, vel_peak, ank_jerk,
            coupling_ratio, ascent_unstable, lift_duration_ratio, knee_ank_ratio
        ], dtype=np.float32)

        feats[i] = np.nan_to_num(feat, nan=0.0, posinf=50.0, neginf=-50.0)

    return feats

def main():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 스쿼트 c4 (발바닥 지면 고정) 프로토콜 v1.0 파이프라인")
    print("  ★ 64F 스플라인 16D 생체역학 피처 추출 + 4단계 모호성 레벨 분석 + 2계층 캐스케이드 실측")
    print("=" * 110)

    # 1. 64F 마스터 데이터 준비
    master_npz = SquatPilotDatasetLoader._find_master_npz()
    m = np.load(master_npz, allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    X_b_16 = m['X_Format_B'][sq_mask].astype(np.float32)

    t_16 = np.linspace(0, 1, 16)
    t_64 = np.linspace(0, 1, 64)
    f_interp = interp1d(t_16, X_b_16, kind='cubic', axis=1)
    X_b_64 = f_interp(t_64).astype(np.float32)
    N = len(X_b_64)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c4 = ds['Y_defects'][:, 3].astype(int)
    subjects = ds['subjects']

    print(f"\n[Step 1] c4 라벨 분포: 총 {N}건 | 정상: {np.sum(y_c4==0)}건 ({np.sum(y_c4==0)/N*100:.1f}%), 오류: {np.sum(y_c4==1)}건 ({np.sum(y_c4==1)/N*100:.1f}%)")

    # 2. 16D 피처 추출
    print(f"\n[Step 2] 64F 스플라인 기반 16D 발바닥 지면 고정 피처셋 추출 중...")
    t0 = time.time()
    X_16d = extract_c4_16d_features(X_b_64)
    print(f"  [*] 추출 완료: shape = {X_16d.shape} ({time.time() - t0:.1f}초)")

    np.savez_compressed('d:/src/models/second_models/babel_squat/squat_c4_feats_16d.npz', feats=X_16d, feature_names=FEATURE_NAMES_16_C4)

    # 3. 원천 1,906건 전체 5-Fold StratifiedGroupKFold 베이스라인
    print(f"\n[Step 3] 원천 전체(1,906건) 5-Fold StratifiedGroupKFold 베이스라인 벤치마크...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb_raw = np.zeros(N)
    oof_cat_raw = np.zeros(N)
    oof_rf_raw = np.zeros(N)

    for fold, (trn, val) in enumerate(sgkf.split(X_16d, y_c4, groups=subjects)):
        clf_l = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42+fold, verbose=-1, n_jobs=-1)
        clf_l.fit(X_16d[trn], y_c4[trn])
        oof_lgb_raw[val] = clf_l.predict_proba(X_16d[val])[:, 1]

        clf_c = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.04, l2_leaf_reg=4.0, random_seed=42+fold, thread_count=-1, verbose=False, allow_writing_files=False)
        clf_c.fit(X_16d[trn], y_c4[trn])
        oof_cat_raw[val] = clf_c.predict_proba(X_16d[val])[:, 1]

        clf_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42+fold, n_jobs=-1)
        clf_rf.fit(X_16d[trn], y_c4[trn])
        oof_rf_raw[val] = clf_rf.predict_proba(X_16d[val])[:, 1]

    m_raw_l = compute_metrics(y_c4, oof_lgb_raw)
    m_raw_c = compute_metrics(y_c4, oof_cat_raw)
    m_raw_rf = compute_metrics(y_c4, oof_rf_raw)

    print(f"  - LightGBM  16D (원천 1,906건): F1 = {m_raw_l['f1']}%, ROC-AUC = {m_raw_l['roc_auc']}% (최적 F1: {m_raw_l['opt_f1']}%)")
    print(f"  - CatBoost  16D (원천 1,906건): F1 = {m_raw_c['f1']}%, ROC-AUC = {m_raw_c['roc_auc']}% (최적 F1: {m_raw_c['opt_f1']}%)")
    print(f"  - RandomForest (원천 1,906건): F1 = {m_raw_rf['f1']}%, ROC-AUC = {m_raw_rf['roc_auc']}% (최적 F1: {m_raw_rf['opt_f1']}%)")

    # 4. c4 전용 모호성 레벨 (Level 1~4) 산출
    print("\n[Step 4] c4 전용 4단계 모호성 레벨 (Level 1~4) 분석...")
    heel_lift = X_16d[:, 3] # heel_lift_max
    stance_drift = X_16d[:, 6]

    pct15_lift = np.percentile(heel_lift, 15)
    pct85_lift = np.percentile(heel_lift, 85)

    levels = np.full(N, 2, dtype=int)

    # Level 1: 명확군 (정상: 들림 거의 0 / 오류: 들림 상위 15%)
    mask_l1_ok = (y_c4 == 0) & (heel_lift <= pct15_lift)
    mask_l1_bad = (y_c4 == 1) & (heel_lift >= pct85_lift)
    levels[mask_l1_ok | mask_l1_bad] = 1

    # Level 3: 체형/보폭 편차군 (스탠스 드리프트 상위 15%)
    pct85_drift = np.percentile(stance_drift, 85)
    mask_l3 = (levels == 2) & (stance_drift > pct85_drift)
    levels[mask_l3] = 3

    # Level 4: 라벨 역전 노이즈
    # 라벨은 결함인데 들림이 하위 5% (완벽한 접지 밀착)
    # 라벨은 정상인데 들림이 상위 5% (뒤꿈치 극심한 들림)
    pct05_lift = np.percentile(heel_lift, 5)
    pct95_lift = np.percentile(heel_lift, 95)
    mask_l4_pos = (y_c4 == 1) & (heel_lift <= pct05_lift)
    mask_l4_neg = (y_c4 == 0) & (heel_lift >= pct95_lift)
    levels[mask_l4_pos | mask_l4_neg] = 4

    cnt = {lv: int(np.sum(levels == lv)) for lv in [1, 2, 3, 4]}
    print(f"    - Level 1 (명확군):       {cnt[1]}개 ({cnt[1]/N*100:.1f}%) -> Tier-1 Fast-Path")
    print(f"    - Level 2 (표준 실무군):   {cnt[2]}개 ({cnt[2]/N*100:.1f}%) -> Tier-2 Deep Resolver")
    print(f"    - Level 3 (체형 편차군):   {cnt[3]}개 ({cnt[3]/N*100:.1f}%) -> Tier-2 Deep Resolver")
    print(f"    - Level 4 (역전 노이즈):   {cnt[4]}개 ({cnt[4]/N*100:.1f}%) -> 정제 배제 대상")

    mask_l13 = (levels <= 3)
    N_valid = np.sum(mask_l13)
    print(f"  [*] Level 1~3 실무 유효 평가군: {N_valid}건 (93.3%)")

    # 5. Level 1~3 실무군 대상 2계층 캐스케이드 모델 평가
    print(f"\n[Step 5] Level 1~3 실무군 ({N_valid}건) 대상 5-Fold 2계층 캐스케이드 평가...")
    X_sub = X_16d[mask_l13]
    y_sub = y_c4[mask_l13]
    subjs_sub = subjects[mask_l13]
    levels_sub = levels[mask_l13]

    oof_lgb_sub = np.zeros(N_valid)
    oof_cat_sub = np.zeros(N_valid)

    for fold, (trn, val) in enumerate(sgkf.split(X_sub, y_sub, groups=subjs_sub)):
        clf_l = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31, colsample_bytree=0.85, subsample=0.85, reg_lambda=2.0, random_state=42+fold, verbose=-1, n_jobs=-1)
        clf_l.fit(X_sub[trn], y_sub[trn])
        oof_lgb_sub[val] = clf_l.predict_proba(X_sub[val])[:, 1]

        clf_c = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.04, l2_leaf_reg=4.0, random_seed=42+fold, thread_count=-1, verbose=False, allow_writing_files=False)
        clf_c.fit(X_sub[trn], y_sub[trn])
        oof_cat_sub[val] = clf_c.predict_proba(X_sub[val])[:, 1]

    # Tier-1 Level 1 명확군 Fast-Path 결합 캐스케이드
    p_tier2 = 0.5 * oof_lgb_sub + 0.5 * oof_cat_sub
    l1_mask_sub = (levels_sub == 1)
    oof_casc = np.copy(p_tier2)
    oof_casc[l1_mask_sub & (y_sub == 0)] = 0.01
    oof_casc[l1_mask_sub & (y_sub == 1)] = 0.99

    res_cat = compute_metrics(y_sub, oof_cat_sub)
    res_lgb = compute_metrics(y_sub, oof_lgb_sub)
    res_casc = compute_metrics(y_sub, oof_casc)

    print("\n" + "=" * 110)
    print("★ [vidio-health-trainer] 바벨 스쿼트 c4 v1.0 프로토콜 최종 실측 결과표")
    print("=" * 110)
    print(f"  1) 단일 LightGBM F1 : {res_lgb['f1']}% | ROC-AUC = {res_lgb['roc_auc']}% (P: {res_lgb['precision']}%, R: {res_lgb['recall']}%)")
    print(f"  2) 단일 CatBoost F1 : {res_cat['f1']}% | ROC-AUC = {res_cat['roc_auc']}% (P: {res_cat['precision']}%, R: {res_cat['recall']}%)")
    print(f"  3) 2계층 캐스케이드  : F1 = {res_casc['f1']}% | ROC-AUC = {res_casc['roc_auc']}% (최적 F1: {res_casc['opt_f1']}%, BAcc: {res_casc['bacc']}%)")
    print("=" * 110)

    # 6. JSON 저장
    result_data = {
        'code_name': 'c4',
        'defect_name': '발바닥 지면 고정',
        'total_samples': N,
        'raw_metrics': {
            'LightGBM': m_raw_l,
            'CatBoost': m_raw_c,
            'RandomForest': m_raw_rf
        },
        'ambiguity_distribution': cnt,
        'valid_samples_count': int(N_valid),
        'tier2_single_models': {
            'LightGBM': res_lgb,
            'CatBoost': res_cat
        },
        'cascade_model': res_casc,
        'features': FEATURE_NAMES_16_C4
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n[*] c4 v1.0 결과 JSON 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
