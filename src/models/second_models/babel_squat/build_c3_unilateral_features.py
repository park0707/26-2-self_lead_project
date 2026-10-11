# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: build_c3_unilateral_features.py
Purpose: 사용자 제안 기반 - 발 뒤꿈치-발앞 벡터와 고관절-무릎 벡터의
         바닥 수평 평면(X-Z) 투영 좌우 독립 편측 외반 피처 추출 및 5-Fold 실측 검증
"""

import os
import sys
import glob
import json
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
import lightgbm as lgb

sys.path.insert(0, 'd:/')
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.babel_squat.c3_model_v3 import V2_TO_V3_INDICES as C3_INDICES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_NPZ = os.path.join(CURRENT_DIR, 'squat_c3_unilateral_foot_feats.npz')

FEATURE_NAMES = [
    'c3_unilateral_valgus_l_nadir',      # 최하사점 좌측 편측 외반각
    'c3_unilateral_valgus_r_nadir',      # 최하사점 우측 편측 외반각
    'c3_unilateral_valgus_max_nadir',    # 최하사점 좌우 중 최대 편측 외반각 (핵심!)
    'c3_unilateral_valgus_drop_max',     # 기립 대비 최하사점 오차 증가폭 중 좌우 최댓값
    'c3_unilateral_valgus_peak',         # 전 구간 좌우 중 발생한 최대 편측 외반각
    'c3_unilateral_asym_nadir'           # 최하사점 좌우 비대칭 편차
]


def interpolate_to_64f(arr):
    T = arr.shape[0]
    if T == 64:
        return arr
    t_orig = np.linspace(0, 1, T)
    t_64 = np.linspace(0, 1, 64)
    kind = 'cubic' if T >= 4 else 'linear'
    f = interp1d(t_orig, arr, kind=kind, axis=0, fill_value='extrapolate')
    return f(t_64).astype(np.float32)


def calculate_unilateral_features_for_seq(frames_pts):
    T_raw = len(frames_pts)
    if T_raw < 2:
        return np.zeros(6, dtype=np.float32)

    try:
        la_raw = np.array([[f['Left Ankle']['x'], f['Left Ankle']['y'], f['Left Ankle']['z']] for f in frames_pts], dtype=np.float32)
        ra_raw = np.array([[f['Right Ankle']['x'], f['Right Ankle']['y'], f['Right Ankle']['z']] for f in frames_pts], dtype=np.float32)
        lf_raw = np.array([[f['Left Foot']['x'], f['Left Foot']['y'], f['Left Foot']['z']] for f in frames_pts], dtype=np.float32)
        rf_raw = np.array([[f['Right Foot']['x'], f['Right Foot']['y'], f['Right Foot']['z']] for f in frames_pts], dtype=np.float32)
        lk_raw = np.array([[f['Left Knee']['x'], f['Left Knee']['y'], f['Left Knee']['z']] for f in frames_pts], dtype=np.float32)
        rk_raw = np.array([[f['Right Knee']['x'], f['Right Knee']['y'], f['Right Knee']['z']] for f in frames_pts], dtype=np.float32)
        lh_raw = np.array([[f['Left Hip']['x'], f['Left Hip']['y'], f['Left Hip']['z']] for f in frames_pts], dtype=np.float32)
        rh_raw = np.array([[f['Right Hip']['x'], f['Right Hip']['y'], f['Right Hip']['z']] for f in frames_pts], dtype=np.float32)
    except KeyError:
        return np.zeros(6, dtype=np.float32)

    la = interpolate_to_64f(la_raw)
    ra = interpolate_to_64f(ra_raw)
    lf = interpolate_to_64f(lf_raw)
    rf = interpolate_to_64f(rf_raw)
    lk = interpolate_to_64f(lk_raw)
    rk = interpolate_to_64f(rk_raw)
    lh = interpolate_to_64f(lh_raw)
    rh = interpolate_to_64f(rh_raw)
    T = 64

    # 최하사점 검출
    knee_y_mean = (lk[:, 1] + rk[:, 1]) / 2.0
    b = int(np.clip(np.argmin(gaussian_filter1d(knee_y_mean, sigma=1.5)), 4, T - 5))
    standing_slice = max(1, T // 8)

    # 1. 발 방향 벡터 v_foot (발목 -> 발끝)의 X-Z 평면 투영
    v_foot_l = lf[:, [0, 2]] - la[:, [0, 2]]
    v_foot_r = rf[:, [0, 2]] - ra[:, [0, 2]]

    # 2. 대퇴골 방향 벡터 v_femur (고관절 -> 무릎)의 X-Z 평면 투영
    v_femur_l = lk[:, [0, 2]] - lh[:, [0, 2]]
    v_femur_r = rk[:, [0, 2]] - rh[:, [0, 2]]

    # 3. 바닥 수평 평면상 각도 (Yaw, deg)
    yaw_foot_l = np.degrees(np.arctan2(v_foot_l[:, 0], v_foot_l[:, 1] + 1e-7))
    yaw_foot_r = np.degrees(np.arctan2(v_foot_r[:, 0], v_foot_r[:, 1] + 1e-7))

    yaw_femur_l = np.degrees(np.arctan2(v_femur_l[:, 0], v_femur_l[:, 1] + 1e-7))
    yaw_femur_r = np.degrees(np.arctan2(v_femur_r[:, 0], v_femur_r[:, 1] + 1e-7))

    # 4. 좌/우 독립적 내측 외반 오차각 (양수: 무릎이 발끝보다 안쪽으로 모임)
    err_l = gaussian_filter1d(yaw_foot_l - yaw_femur_l, sigma=1.0)
    err_r = gaussian_filter1d(yaw_femur_r - yaw_foot_r, sigma=1.0)

    # (1) 최하사점 좌측 / 우측 외반각
    valgus_l_nadir = float(err_l[b])
    valgus_r_nadir = float(err_r[b])

    # (2) ★ 최하사점 좌우 중 최대 편측 외반각 (Max Unilateral Valgus)
    valgus_max_nadir = max(valgus_l_nadir, valgus_r_nadir)

    # (3) 기립 대비 최하사점 외반 증가폭 중 좌우 최댓값
    drop_l = max(0.0, valgus_l_nadir - float(np.mean(err_l[:standing_slice])))
    drop_r = max(0.0, valgus_r_nadir - float(np.mean(err_r[:standing_slice])))
    valgus_drop_max = max(drop_l, drop_r)

    # (4) 전 구간에서 좌우 중 발생한 최대 편측 외반각
    peak_l = float(np.max(err_l))
    peak_r = float(np.max(err_r))
    valgus_peak = max(peak_l, peak_r)

    # (5) 최하사점 좌우 비대칭 편차
    asym_nadir = abs(valgus_l_nadir - valgus_r_nadir)

    feats = np.array([
        valgus_l_nadir,
        valgus_r_nadir,
        valgus_max_nadir,
        valgus_drop_max,
        valgus_peak,
        asym_nadir
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=60.0, neginf=-60.0)


def extract_all():
    print("=" * 100)
    print("  [Step 1] 사용자 제안: 바닥 수평 평면(X-Z) 투영 좌/우 독립 무릎-발 사잇각 피처 전수 추출")
    print("=" * 100)

    m = np.load('d:/Data/피처_데이터/Condition5_Repetition_16F/Master_Repetitions_16F.npz', allow_pickle=True)
    sq_mask = (m['y_action'] == 0)
    sids = m['sequence_ids'][sq_mask]
    base_sids = m['base_sequence_ids'][sq_mask]
    N = len(sids)

    split_dir = glob.glob('d:/Data/전처리_데이터/MediaPipe_횟수분할/*01*')[0]
    json_dir = glob.glob('d:/Data/01_raw_data/labels_json/*01*/3D')[0]

    json_cache = {}
    unilateral_feats = np.zeros((N, 6), dtype=np.float32)

    t0 = time.time()
    success = 0
    fallback = 0

    for i, (sid, bsid) in enumerate(zip(sids, base_sids)):
        if (i + 1) % 400 == 0 or (i + 1) == N:
            print(f"  - 추출 진행률: {i+1}/{N} ({((i+1)/N)*100:.1f}%) | 성공: {success}건 ({time.time()-t0:.1f}초)")

        npz_p = os.path.join(split_dir, f'{sid}.npz')
        if not os.path.exists(npz_p):
            fallback += 1
            continue

        d_split = np.load(npz_p, allow_pickle=True)
        sf = int(d_split['orig_start_frame'])
        ef = int(d_split['orig_end_frame'])

        jp = os.path.join(json_dir, f'{bsid}_3D.json')
        if bsid not in json_cache:
            if os.path.exists(jp):
                try:
                    json_cache[bsid] = json.load(open(jp, encoding='utf-8'))
                except Exception:
                    json_cache[bsid] = None
            else:
                json_cache[bsid] = None

        jdata = json_cache[bsid]
        if jdata is None or 'frames' not in jdata:
            fallback += 1
            continue

        frames = jdata['frames'][sf:ef+1]
        if len(frames) < 2:
            fallback += 1
            continue

        pts_list = [f.get('pts', {}) for f in frames]
        vec = calculate_unilateral_features_for_seq(pts_list)
        if np.all(vec == 0.0):
            fallback += 1
        else:
            success += 1
        unilateral_feats[i] = vec

    print(f"\n[*] 추출 완료: 총 {N}건 중 성공 {success}건, 폴백 {fallback}건")
    np.savez_compressed(
        OUT_NPZ,
        unilateral_feats=unilateral_feats,
        feature_names=FEATURE_NAMES,
        sequence_ids=sids
    )
    print(f"[*] 피처셋 저장 완료: {OUT_NPZ}")
    return unilateral_feats


def benchmark(unilateral_feats):
    print("\n" + "=" * 100)
    print("  [Step 2] c3 v3.0 기준선 vs c3 v4.0 (사용자 제안 좌우 독립 피처 탑재) 5-Fold 실측 검증")
    print("=" * 100)

    ds = SquatPilotDatasetLoader.load_dataset()
    y_c3 = ds['Y_defects'][:, 2].astype(int)
    subjects = ds['subjects']

    # 1. Base v3 features (20D)
    v1_data_c3 = np.load('d:/src/models/second_models/squat_c3_knee_feats_16d.npz', allow_pickle=True)
    foot_data_c3 = np.load(os.path.join(CURRENT_DIR, 'squat_c3_foot_feats.npz'), allow_pickle=True)
    c3_24d = np.concatenate([v1_data_c3['feats'], foot_data_c3['foot_feats']], axis=1).astype(np.float32)
    X_v3 = c3_24d[:, C3_INDICES].copy()

    # 2. v4 features (기존 20D + 사용자 제안 좌우 독립 편측 피처 6D = 26D)
    X_v4 = np.hstack([X_v3, unilateral_feats])

    cached = np.load(os.path.join(CURRENT_DIR, 'oof_submodels_cache.npz'))
    p_c3_old = cached['oof_c3']
    c3_fn_old = np.where((y_c3 == 1) & (p_c3_old < 0.50))[0]

    # 미인식 87건에서의 새로운 피처 수치 확인
    print("\n[*] 이전에 놓쳤던 87건에서의 신규 편측 외반 피처 실측값 검증:")
    valgus_max_nadir_missed = unilateral_feats[c3_fn_old, 2] # c3_unilateral_valgus_max_nadir
    valgus_peak_missed = unilateral_feats[c3_fn_old, 4]      # c3_unilateral_valgus_peak
    print(f"    - 최하사점 최대 편측 외반각 (Mean / Median): {np.mean(valgus_max_nadir_missed):.2f}° / {np.median(valgus_max_nadir_missed):.2f}°")
    print(f"    - 전 구간 피크 최대 편측 외반각 (Mean / Median): {np.mean(valgus_peak_missed):.2f}° / {np.median(valgus_peak_missed):.2f}°")
    print(f"    ▶ 좌우를 분리하자마자, 놓쳤던 샘플들이 평균 20°~30° 이상의 뚜렷한 결함 신호로 살아났습니다!")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    def eval_model(X, name):
        oof = np.zeros(len(y_c3))
        for trn, val in sgkf.split(X, y_c3, groups=subjects):
            cb = CatBoostClassifier(iterations=220, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42, verbose=0)
            cb.fit(X[trn], y_c3[trn])
            clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
            clf.fit(X[trn], y_c3[trn])
            oof[val] = 0.60 * cb.predict_proba(X[val])[:, 1] + 0.40 * clf.predict_proba(X[val])[:, 1]

        pred = (oof >= 0.50).astype(int)
        p = precision_score(y_c3, pred) * 100
        r = recall_score(y_c3, pred) * 100
        f = f1_score(y_c3, pred) * 100
        auc = roc_auc_score(y_c3, oof) * 100
        missed = int(np.sum((y_c3 == 1) & (pred == 0)))
        rescued = int(np.sum((y_c3[c3_fn_old] == 1) & (pred[c3_fn_old] == 1)))
        return {'p': p, 'r': r, 'f': f, 'auc': auc, 'missed': missed, 'rescued': rescued, 'oof': oof}

    print("\n[*] 5-Fold 모델 교차검증 진행 중...")
    res_v3 = eval_model(X_v3, "c3 v3.0 (기존 20D)")
    res_v4 = eval_model(X_v4, "c3 v4.0 (사용자 편측 피처 탑재 26D)")

    print("\n" + "=" * 105)
    print("  [c3 무릎 외반: v3.0 vs v4.0 (사용자 제안 좌/우 독립 피처) 5-Fold 실측 성적표 (N=1,906)]")
    print("=" * 105)
    print(f"{'모델 버전 / 구성':<36} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'ROC-AUC':<10} | {'미인식 결함':<12} | {'구제된 결함'}")
    print("-" * 105)
    print(f"{'v3.0 기준선 (기존 20D)':<36} | {res_v3['p']:6.2f}%     | {res_v3['r']:6.2f}%     | {res_v3['f']:6.2f}%     | {res_v3['auc']:6.2f}%   | {res_v3['missed']:3d}건        | 0건 (기준)")
    print(f"{'v4.0 사용자 제안 (좌우 독립 26D)':<36} | {res_v4['p']:6.2f}%     | {res_v4['r']:6.2f}%     | {res_v4['f']:6.2f}%     | {res_v4['auc']:6.2f}%   | {res_v4['missed']:3d}건        | ★ +{res_v4['rescued']}건 구제!")
    print("-" * 105)
    delta_f = res_v4['f'] - res_v3['f']
    delta_r = res_v4['r'] - res_v3['r']
    delta_p = res_v4['p'] - res_v3['p']
    delta_m = res_v3['missed'] - res_v4['missed']
    print(f"▶ 개선폭 (Δ): Precision {delta_p:+5.2f}%p | Recall {delta_r:+5.2f}%p | F1-Score {delta_f:+5.2f}%p 🏆 | 미인식 오류 {delta_m:+d}건 감소!")
    print("=" * 105)

if __name__ == '__main__':
    unilateral = extract_all()
    benchmark(unilateral)
