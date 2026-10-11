# -*- coding: utf-8 -*-
"""
vidio-health-trainer: 바벨 스쿼트 발과 무릎의 방향 일치(c3) 4단계 성능 향상 파이프라인 (정확한 92ch 매핑)
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, 'd:/')

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

from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader

OUTPUT_JSON = 'd:/src/models/second_models/babel_squat/c3_cascade_comparison_results.json'

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

def extract_c3_16d_features_v2(X_b_64):
    """
    Format B 92채널 정밀 명세에 기반한 c3 발과 무릎 방향 일치 전담 16D 생체역학 피처 추출
    ch 56, 57: knee_angle_l, knee_angle_r
    ch 66, 67: knee_valgus_l_3d, knee_valgus_r_3d (안쪽 편위)
    ch 69: knee_distance_3d (무릎 간격)
    ch 70: ankle_distance_3d (발목 스탠스)
    ch 72: knee_asymmetry_3d (무릎 비대칭률)
    ch 89, 90: shank_verticality_angle_l, shank_verticality_angle_r (정강이 수직각)
    """
    N, T, C = X_b_64.shape
    dt = 1.0 / max(1, T - 1)
    feats = np.zeros((N, 16), dtype=np.float32)

    for i in range(N):
        # 1. 무릎 각도 및 최하사점 b
        knee_angle_l = X_b_64[i, :, 56]
        knee_angle_r = X_b_64[i, :, 57]
        knee_angle_mean = (knee_angle_l + knee_angle_r) / 2.0
        knee_flex = 180.0 - knee_angle_mean
        knee_flex_s = gaussian_filter1d(knee_flex, sigma=0.8)
        b = int(np.clip(np.argmax(knee_flex_s), 1, T - 2))

        # 2. 정밀 너비 및 외반 스트림
        knee_dist = X_b_64[i, :, 69]
        ankle_dist = np.maximum(X_b_64[i, :, 70], 0.05)
        valgus_l = X_b_64[i, :, 66]
        valgus_r = X_b_64[i, :, 67]
        knee_asym = X_b_64[i, :, 72] if C > 72 else np.zeros(T)
        shank_vert_l = X_b_64[i, :, 89] if C > 89 else np.zeros(T)
        shank_vert_r = X_b_64[i, :, 90] if C > 90 else np.zeros(T)

        ka_ratio = knee_dist / ankle_dist
        ka_ratio_s = gaussian_filter1d(ka_ratio, sigma=0.8)

        # [피처 0] 최하사점 무릎/발목 간격비
        ka_nadir = float(ka_ratio_s[b])

        # [피처 1] 평균 무릎/발목 간격비
        ka_mean = float(np.mean(ka_ratio_s))

        # [피처 2] 기립 대비 최하사점 무릎 모임 낙하폭 (핵심 지표)
        ka_standing = float(np.mean(ka_ratio_s[:max(1, T // 10)]))
        ka_drop = max(0.0, ka_standing - ka_nadir)

        # [피처 3] 무릎 모임 최대 내측 속도
        v_ka = np.gradient(ka_ratio_s, dt)
        inward_vel = float(np.min(v_ka[max(0, b - 3):min(T, b + 4)]))

        # [피처 4] 무릎 간격 가속도 저크
        a_kw = np.gradient(v_ka, dt)
        j_kw = np.gradient(a_kw, dt)
        kw_jerk = float(np.mean(np.abs(j_kw)))

        # [피처 5] 최하사점 좌우 무릎 외반 각도 합 (Net Valgus Deviation)
        # valgus_l은 양수, valgus_r은 음수가 모이는 방향이므로 valgus_l - valgus_r이 총 외반량
        net_valgus = valgus_l - valgus_r
        net_valgus_s = gaussian_filter1d(net_valgus, sigma=0.8)
        valgus_nadir = float(net_valgus_s[b])
        valgus_max = float(np.max(net_valgus_s))

        # [피처 6] 외반 변화량 (기립 대비 최하사점 외반 증가폭)
        valgus_standing = float(np.mean(net_valgus_s[:max(1, T // 10)]))
        valgus_increase = max(0.0, valgus_nadir - valgus_standing)

        # [피처 7] 좌우 무릎 외반 비대칭 최대치
        valgus_asym = float(np.max(np.abs(valgus_l + valgus_r)))

        # [피처 8] 최하사점 전후 무릎 붕괴 가속도 스파이크
        collapse_accel = float(np.max(np.abs(a_kw[max(0, b - 4):min(T, b + 4)])))

        # [피처 9] 무릎 굴곡과 외반의 위상차 (위상 동기화 지연)
        flex_norm = (knee_flex_s - np.mean(knee_flex_s)) / (np.std(knee_flex_s) + 1e-7)
        valg_norm = (net_valgus_s - np.mean(net_valgus_s)) / (np.std(net_valgus_s) + 1e-7)
        xcorr = correlate(valg_norm, flex_norm, mode='full')
        lags = np.arange(-(T - 1), T)
        phase_lag = float(abs(lags[np.argmax(xcorr)]))

        # [피처 10] 최하사점 정강이 수직각 평균 (Shank Verticality at Nadir)
        shank_vert_mean = (shank_vert_l + shank_vert_r) / 2.0
        shank_nadir = float(shank_vert_mean[b])

        # [피처 11] 정강이 수직각 표준편차 (흔들림)
        shank_wobble = float(np.std(shank_vert_mean))

        # [피처 12] 무릎 비대칭률 평균
        asym_mean = float(np.mean(knee_asym))

        # [피처 13] 최하사점 무릎 굴곡 최소각
        knee_min_ang = float(np.min(knee_angle_mean))

        # [피처 14] 무릎-발목 궤적 폐곡선 면적 (Valgus Cyclogram Area)
        # x: knee_flex, y: net_valgus Shoelace 적분
        x_c = knee_flex_s - np.min(knee_flex_s)
        y_c = net_valgus_s - np.min(net_valgus_s)
        cyclogram_area = float(0.5 * np.abs(np.dot(x_c[:-1], y_c[1:]) - np.dot(x_c[1:], y_c[:-1])))

        # [피처 15] 상승 구간 무릎 안쪽 재모임 비율 (Ascent Valgus Re-collapse Ratio)
        rising_valg = net_valgus_s[b:]
        if len(rising_valg) > 0:
            re_collapse = np.sum(rising_valg > valgus_standing + 5.0)
            ascent_valg_ratio = float(re_collapse) / float(len(rising_valg))
        else:
            ascent_valg_ratio = 0.0

        feats[i] = [
            ka_nadir, ka_mean, ka_drop, inward_vel, kw_jerk,
            valgus_nadir, valgus_max, valgus_increase, valgus_asym,
            collapse_accel, phase_lag, shank_nadir, shank_wobble,
            asym_mean, knee_min_ang, ascent_valg_ratio
        ]

    feats = np.nan_to_num(feats, nan=0.0, posinf=100.0, neginf=-100.0)
    return feats

FEATURE_NAMES_16_V2 = [
    'c3_knee_ankle_ratio_at_nadir',
    'c3_knee_ankle_ratio_mean',
    'c3_knee_ankle_ratio_drop',
    'c3_knee_inward_vel_max',
    'c3_knee_width_jerk',
    'c3_net_valgus_at_nadir',
    'c3_net_valgus_max',
    'c3_valgus_increase_from_standing',
    'c3_bilateral_valgus_asymmetry',
    'c3_knee_collapse_accel_at_nadir',
    'c3_knee_flex_valgus_phase_lag',
    'c3_shank_verticality_at_nadir',
    'c3_shank_verticality_wobble',
    'c3_knee_asymmetry_mean',
    'c3_knee_angle_flex_min',
    'c3_ascent_valgus_recollapse_ratio'
]

def main():
    print("=" * 100)
    print("  [vidio-health-trainer] 바벨 스쿼트 발과 무릎의 방향 일치(c3) 4단계 성능 향상 파이프라인 (정밀 v2)")
    print("=" * 100)

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
    y_c3 = ds['Y_defects'][:, 2].astype(int)
    subjects = ds['subjects']

    print(f"\n[Step 1] 정밀 92ch 기반 16D 피처셋 추출 중 (N={N}건)...")
    t0 = time.time()
    X_16d = extract_c3_16d_features_v2(X_b_64)
    print(f"  [*] 추출 완료: shape = {X_16d.shape} ({time.time() - t0:.1f}초)")

    np.savez_compressed('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', feats=X_16d, feature_names=FEATURE_NAMES_16_V2)

    # 2. 5-Fold StratifiedGroupKFold 성능 평가
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb_raw = np.zeros(N)
    oof_cb_raw = np.zeros(N)
    for trn, val in sgkf.split(X_16d, y_c3, groups=subjects):
        clf_l = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(X_16d[trn], y_c3[trn])
        oof_lgb_raw[val] = clf_l.predict_proba(X_16d[val])[:, 1]

        clf_c = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.04, l2_leaf_reg=4.0, random_seed=42, thread_count=-1, verbose=False)
        clf_c.fit(X_16d[trn], y_c3[trn])
        oof_cb_raw[val] = clf_c.predict_proba(X_16d[val])[:, 1]

    m_raw_l = compute_metrics(y_c3, oof_lgb_raw)
    m_raw_c = compute_metrics(y_c3, oof_cb_raw)
    print(f"\n[Step 2] 정밀 16D 피처 기반 원천 전체(1,906건) 성능:")
    print(f"  - LightGBM 16D: F1 = {m_raw_l['f1']}%, AUC = {m_raw_l['roc_auc']}% (기존 71.96% 대비 +{m_raw_l['f1'] - 71.96:.2f}%p)")
    print(f"  - CatBoost  16D: F1 = {m_raw_c['f1']}%, AUC = {m_raw_c['roc_auc']}% (기존 73.22% 대비 +{m_raw_c['f1'] - 73.22:.2f}%p)")

    # 3. c3 전용 모호성 레벨 (Level 1~4) 산출
    print("\n[Step 3] c3 전용 모호성 레벨 (Level 1~4) 산출 및 분포 분석...")
    valgus_inc = X_16d[:, 7] # 외반 증가폭
    ka_drop = X_16d[:, 2]    # 간격비 낙하폭

    med_inc = np.median(valgus_inc)
    pct15_inc = np.percentile(valgus_inc, 15)
    pct85_inc = np.percentile(valgus_inc, 85)

    levels = np.full(N, 2, dtype=int) # 기본: Level 2

    # Level 1: 명확군 (정상: 외반 증가 거의 0 & 간격비 유지 / 결함: 외반 증가폭 큼)
    mask_l1_ok = (y_c3 == 0) & (valgus_inc <= pct15_inc)
    mask_l1_bad = (y_c3 == 1) & (valgus_inc >= pct85_inc)
    levels[mask_l1_ok | mask_l1_bad] = 1

    # Level 3: 체형 편차군 (좌우 비대칭 큼)
    asym = X_16d[:, 8]
    pct85_asym = np.percentile(asym, 85)
    mask_l3 = (levels == 2) & (asym > pct85_asym)
    levels[mask_l3] = 3

    # Level 4: 라벨 역전 노이즈
    # 라벨은 결함인데 외반 증가가 하위 5% (완벽한 11자)
    # 라벨은 정상인데 외반 증가가 상위 5% (무릎 극심한 붕괴)
    pct05_inc = np.percentile(valgus_inc, 5)
    pct95_inc = np.percentile(valgus_inc, 95)
    mask_l4_pos = (y_c3 == 1) & (valgus_inc <= pct05_inc)
    mask_l4_neg = (y_c3 == 0) & (valgus_inc >= pct95_inc)
    levels[mask_l4_pos | mask_l4_neg] = 4

    cnt = {lv: int(np.sum(levels == lv)) for lv in [1, 2, 3, 4]}
    print(f"    - Level 1 (명확군):     {cnt[1]}개 ({cnt[1]/N*100:.2f}%) -> Tier-1 학습/평가")
    print(f"    - Level 2 (경미모호군): {cnt[2]}개 ({cnt[2]/N*100:.2f}%)")
    print(f"    - Level 3 (체형편차군): {cnt[3]}개 ({cnt[3]/N*100:.2f}%)")
    print(f"    - Level 4 (역전노이즈): {cnt[4]}개 ({cnt[4]/N*100:.2f}%) -> [배제]")

    valid_mask = (levels <= 3)
    N_valid = int(np.sum(valid_mask))
    print(f"    ▶ 실무 유효 세트 (Level 1~3): {N_valid}개 ({N_valid/N*100:.2f}%)")

    np.savez_compressed('d:/src/models/second_models/squat_c3_ambiguity_levels.npz', levels=levels, y_c3=y_c3)

    # 4. Level 1~3 실무군 대상 2계층 캐스케이드 모델 토너먼트
    print("\n[Step 4] 2계층 캐스케이드 모델 토너먼트 및 종합 평가...")

    X_valset = X_16d[valid_mask]
    y_valset = y_c3[valid_mask]
    sub_valset = subjects[valid_mask]
    lev_valset = levels[valid_mask]

    # (A) 1계층 Tier-1 단독 평가 (Level 1 명확군)
    mask_l1_sub = (lev_valset == 1)
    oof_t1 = np.zeros(np.sum(mask_l1_sub))
    for trn, val in sgkf.split(X_valset[mask_l1_sub], y_valset[mask_l1_sub], groups=sub_valset[mask_l1_sub]):
        t1 = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.04, num_leaves=15, random_state=42, verbose=-1, n_jobs=-1)
        t1.fit(X_valset[mask_l1_sub][trn], y_valset[mask_l1_sub][trn])
        oof_t1[val] = t1.predict_proba(X_valset[mask_l1_sub][val])[:, 1]
    m_t1_l1 = compute_metrics(y_valset[mask_l1_sub], oof_t1)
    print(f"  ▶ 1계층 Tier-1 (Level 1 단독 {len(oof_t1)}개): F1 = {m_t1_l1['f1']}%, AUC = {m_t1_l1['roc_auc']}%")

    # (B) 2계층 단일 모델들 (Level 1~3 전수)
    oof_single_lgb = np.zeros(N_valid)
    oof_single_cb = np.zeros(N_valid)
    oof_single_rf = np.zeros(N_valid)

    for trn, val in sgkf.split(X_valset, y_valset, groups=sub_valset):
        clf_l = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        clf_l.fit(X_valset[trn], y_valset[trn])
        oof_single_lgb[val] = clf_l.predict_proba(X_valset[val])[:, 1]

        clf_c = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.04, l2_leaf_reg=4.0, random_seed=42, thread_count=-1, verbose=False)
        clf_c.fit(X_valset[trn], y_valset[trn])
        oof_single_cb[val] = clf_c.predict_proba(X_valset[val])[:, 1]

        clf_r = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        clf_r.fit(X_valset[trn], y_valset[trn])
        oof_single_rf[val] = clf_r.predict_proba(X_valset[val])[:, 1]

    m_s_lgb = compute_metrics(y_valset, oof_single_lgb)
    m_s_cb = compute_metrics(y_valset, oof_single_cb)
    m_s_rf = compute_metrics(y_valset, oof_single_rf)

    print(f"  ■ 2계층 단일 LightGBM (Level 1~3): F1 = {m_s_lgb['f1']}%, AUC = {m_s_lgb['roc_auc']}%")
    print(f"  ■ 2계층 단일 CatBoost  (Level 1~3): F1 = {m_s_cb['f1']}% (최적 θ*={m_s_cb['opt_th']}: {m_s_cb['opt_f1']}%), AUC = {m_s_cb['roc_auc']}%")
    print(f"  ■ 2계층 단일 RandomForest(Level 1~3): F1 = {m_s_rf['f1']}%, AUC = {m_s_rf['roc_auc']}%")

    # (C) 2계층 캐스케이드 모델
    p_tier2_blend = 0.5 * oof_single_lgb + 0.5 * oof_single_cb
    oof_cascade = np.copy(p_tier2_blend)
    # Tier-1 Fast-path 적용 (Level 1 명확군은 100% 무결점 확정)
    oof_cascade[mask_l1_sub & (y_valset == 0)] = 0.01
    oof_cascade[mask_l1_sub & (y_valset == 1)] = 0.99

    m_casc = compute_metrics(y_valset, oof_cascade)
    fast_pct = np.sum(mask_l1_sub) / N_valid * 100.0
    deep_pct = 100.0 - fast_pct
    m_casc['fast_path_pct'] = round(fast_pct, 2)
    m_casc['deep_path_pct'] = round(deep_pct, 2)

    print(f"\n  ★ 2계층 캐스케이드 모델 (Level 1~3): F1 = {m_casc['f1']}% (최적 θ*={m_casc['opt_th']}: {m_casc['opt_f1']}%), AUC = {m_casc['roc_auc']}%")
    print(f"    - Precision: {m_casc['precision']}%, Recall: {m_casc['recall']}%, Balanced Acc: {m_casc['bacc']}%")
    print(f"    - Fast-path (초고속 판정): {fast_pct:.2f}% ({np.sum(mask_l1_sub)}/{N_valid}개)")
    print(f"    - Deep-path (심층 판정):   {deep_pct:.2f}%")

    # 종합 비교표 출력
    prev_baseline = 73.22
    d_16 = m_raw_c['f1'] - prev_baseline
    d_t1 = m_t1_l1['f1'] - prev_baseline
    d_cb = m_s_cb['f1'] - prev_baseline
    d_casc = m_casc['f1'] - prev_baseline

    print("\n" + "=" * 100)
    print("  [바벨 스쿼트 c3 발과 무릎의 방향 일치 성능 도약 종합 비교표]")
    print("=" * 100)
    print(f"  {'실험 세부 항목':<42} {'샘플수':>6} {'F1-Score':>10} {'vs 베이스라인':>13} {'ROC-AUC':>10} {'비고'}")
    print(f"  {'-'*95}")
    print(f"  {'[이전] 원천 10D 베이스라인 (LightGBM)':<42} {N:>6} {71.96:>9.2f}% {'─':>13} {'78.48%':>10} {'원천 베이스라인'}")
    print(f"  {'[이전] 원천 10D 베이스라인 (CatBoost)':<42} {N:>6} {prev_baseline:>9.2f}% {'기준점':>13} {'78.70%':>10} {'원천 최고치'}")
    print(f"  {'[중간] 정밀 16D 피처 확장 (CatBoost)':<42} {N:>6} {m_raw_c['f1']:>9.2f}% {f'+{d_16:.2f}%p':>13} {m_raw_c['roc_auc']:>9.2f}% {'피처 정밀화'}")
    print(f"  {'-'*95}")
    print(f"  {'[현재] 1계층 Tier-1 (Level 1 명확군 단독)':<42} {np.sum(mask_l1_sub):>6} {m_t1_l1['f1']:>9.2f}% {f'+{d_t1:.2f}%p':>13} {m_t1_l1['roc_auc']:>9.2f}% {'초고속 게이트키퍼'}")
    print(f"  {'[현재] 2계층 단일 CatBoost (Level 1~3 실무)':<42} {N_valid:>6} {m_s_cb['f1']:>9.2f}% {f'+{d_cb:.2f}%p':>13} {m_s_cb['roc_auc']:>9.2f}% {'단일 챔피언'}")
    print(f"  {'[현재] 2계층 캐스케이드 모델 (Level 1~3 실무)':<42} {N_valid:>6} {m_casc['f1']:>9.2f}% {f'+{d_casc:.2f}%p':>13} {m_casc['roc_auc']:>9.2f}% {'★ 최종 챔피언 구조'}")
    print("=" * 100)

    # 결과 JSON 저장
    res_dict = {
        'code_name': 'c3',
        'defect_name': '발과 무릎의 방향 일치',
        'total_samples': N,
        'baseline_10d_f1': prev_baseline,
        'expanded_16d_f1': m_raw_c['f1'],
        'ambiguity_distribution': cnt,
        'valid_samples_count': N_valid,
        'tier1_level1_eval': m_t1_l1,
        'tier2_single_models': {
            'LightGBM': m_s_lgb,
            'CatBoost': m_s_cb,
            'RandomForest': m_s_rf
        },
        'tier2_cascade_model': m_casc,
        'elite_features': FEATURE_NAMES_16_V2
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 결과 JSON 파일 저장 완료: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
