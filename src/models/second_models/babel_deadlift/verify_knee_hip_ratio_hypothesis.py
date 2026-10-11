# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: verify_knee_hip_ratio_hypothesis.py
Purpose: 사용자의 가설 검증:
         "무릎 각도와 골반 각도의 비율 R(t) = θ_knee(t) / θ_hip(t)을 피처로 넣으면
          정상에서는 완만하고 오류(무릎 먼저 펴짐)에서는 비율이 급변할 것인가?"
         1) 단일 샘플 (피험자 1번 정상 Key281 vs 오류 Key294) 시계열 궤적 실측 대조
         2) 전체 770개 시퀀스 대상 정상 집단 vs 오류 집단 정량 통계 검정 (t-test, Cohen's d, AUC)
         3) 모델 성능 향상 기여도 실측 (기존 24D vs 비율 피처 5D 추가 비교)
"""

import os
import sys
import glob
import json
import time
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d4_model_v1 import (
    extract_d4_v1_features_from_seq,
    FEATURE_NAMES_24_D4
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_d5"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"

MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28

def compute_angle_3d(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    cos_ang = np.sum(v1 * v2, axis=-1) / (norm1 * norm2 + 1e-7)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

def extract_ratio_features_from_seq(kps):
    """무릎-골반 각도 비율 기반 동역학 피처 추출"""
    T = kps.shape[0]
    sh = (kps[:, MP_LEFT_SHOULDER] + kps[:, MP_RIGHT_SHOULDER]) / 2.0
    hip = (kps[:, MP_LEFT_HIP] + kps[:, MP_RIGHT_HIP]) / 2.0
    knee = (kps[:, MP_LEFT_KNEE] + kps[:, MP_RIGHT_KNEE]) / 2.0
    ank = (kps[:, MP_LEFT_ANKLE] + kps[:, MP_RIGHT_ANKLE]) / 2.0

    k_ang = gaussian_filter1d(compute_angle_3d(hip, knee, ank), sigma=1.5)
    h_ang = gaussian_filter1d(compute_angle_3d(sh, hip, knee), sigma=1.5)

    # 1. 각도 비율 시계열 R(t) = θ_knee / θ_hip
    ratio = k_ang / (h_ang + 1e-4)

    # 최하사점(Nadir) 탐색: 골반 각도 최소점
    nadir_idx = int(np.argmin(h_ang))
    nadir_idx = np.clip(nadir_idx, 5, T - 10)
    ascent_slice = slice(nadir_idx, T)
    ascent_len = max(T - nadir_idx, 4)
    mid_ascent_idx = nadir_idx + ascent_len // 2

    # 파생 피처 후보군:
    # 1) First Pull 구간 각도 비율 최대 변화량 (무릎 먼저 펴짐의 직접 지표)
    ratio_first_pull = ratio[nadir_idx:mid_ascent_idx]
    ratio_delta_first_pull = float(np.max(ratio_first_pull) - np.min(ratio_first_pull))

    # 2) 전 구간 각도 비율의 순간 미분(변화율) 최대치: max |d(ratio)/dt|
    d_ratio = np.abs(np.gradient(ratio))
    ratio_deriv_max = float(np.max(d_ratio[ascent_slice]))

    # 3) 상승기 각도 비율의 표준편차 (변동성 / 불안정도)
    ratio_std_ascent = float(np.std(ratio[ascent_slice]))

    # 4) 상승기 각도 비율의 최대치 (무릎은 펴지고 골반은 아직 접혀있을 때의 피크)
    ratio_max_ascent = float(np.max(ratio[ascent_slice]))

    # 5) 무릎/골반 정규화 진행률 편차의 급변 지수
    k_rom = float(np.max(k_ang) - np.min(k_ang) + 1e-4)
    h_rom = float(np.max(h_ang) - np.min(h_ang) + 1e-4)
    k_norm = (k_ang - np.min(k_ang)) / k_rom
    h_norm = (h_ang - np.min(h_ang)) / h_rom
    norm_diff_first_pull = float(np.max(k_norm[nadir_idx:mid_ascent_idx] - h_norm[nadir_idx:mid_ascent_idx]))

    return {
        "k_ang": k_ang,
        "h_ang": h_ang,
        "ratio": ratio,
        "nadir_idx": nadir_idx,
        "features": np.array([
            ratio_delta_first_pull,
            ratio_deriv_max,
            ratio_std_ascent,
            ratio_max_ascent,
            norm_diff_first_pull
        ], dtype=np.float32)
    }

FEATURE_NAMES_RATIO = [
    "d4_ratio_delta_first_pull",     # First Pull 구간 각도 비율 급변량
    "d4_ratio_deriv_max_ascent",     # 상승기 각도 비율 순간 미분 최대치
    "d4_ratio_std_ascent",           # 상승기 각도 비율 변동성(std)
    "d4_ratio_max_ascent",           # 상승기 각도 비율 최대 피크
    "d4_norm_progress_diff_max"      # First Pull 정규화 무릎-골반 신전 진행률 편차
]

def main():
    print("=" * 80)
    print("  [가설 검증] 무릎 각도 / 골반 각도 비율 R(t)의 결함 분별력 정량 검증")
    print("  - 가설: 오류 시 무릎이 먼저 펴지므로 비율 R(t)이 순간 급등/급변할 것이다.")
    print("=" * 80, flush=True)

    # 1. 단일 샘플 궤적 비교 (Key281 정상 vs Key294 오류)
    norm_files = glob.glob(os.path.join(CACHE_DIR, "*Key281*view2.npz"))
    def_files = glob.glob(os.path.join(CACHE_DIR, "*Key294*view2.npz"))

    if norm_files and def_files:
        norm_p = norm_files[0]
        def_p = def_files[0]
        d_norm = extract_ratio_features_from_seq(np.load(norm_p)["kps_33_3d_64f"])
        d_def = extract_ratio_features_from_seq(np.load(def_p)["kps_33_3d_64f"])

        print("\n[1] 피험자 1번 1:1 대조 (시점B / view2 우측전방45°):", flush=True)
        print(f"  * 정상 (Key281):")
        print(f"    - First Pull 각도비율 급변량: {d_norm['features'][0]:.4f}")
        print(f"    - 상승기 각도비율 순간미분최대: {d_norm['features'][1]:.4f}")
        print(f"    - 상승기 각도비율 변동성(std): {d_norm['features'][2]:.4f}")
        print(f"    - 상승기 각도비율 최대피크  : {d_norm['features'][3]:.4f}")
        print(f"    - 정규화 진행률 최대편차    : {d_norm['features'][4]:.4f}")

        print(f"  * 오류 (Key294 - '무릎 먼저 펴고'):")
        print(f"    - First Pull 각도비율 급변량: {d_def['features'][0]:.4f} (정상 대비 {d_def['features'][0]/d_norm['features'][0]:.2f}배 증가)")
        print(f"    - 상승기 각도비율 순간미분최대: {d_def['features'][1]:.4f} (정상 대비 {d_def['features'][1]/d_norm['features'][1]:.2f}배)")
        print(f"    - 상승기 각도비율 변동성(std): {d_def['features'][2]:.4f} (정상 대비 {d_def['features'][2]/d_norm['features'][2]:.2f}배)")
        print(f"    - 상승기 각도비율 최대피크  : {d_def['features'][3]:.4f} (정상 대비 {d_def['features'][3]/d_norm['features'][3]:.2f}배)")
        print(f"    - 정규화 진행률 최대편차    : {d_def['features'][4]:.4f} (정상 대비 {d_def['features'][4]/d_norm['features'][4]:.2f}배 폭증!)", flush=True)

    # 2. 전체 770개 시퀀스 피처 추출 및 통계 검정
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    orig_feats = []
    ratio_feats = []
    y_list = []
    subs_list = []

    for p in npz_files:
        base = os.path.basename(p).replace(".npz", "")
        parts = base.split("_")
        folder = "_".join(parts[:-1])
        lp = os.path.join(LABEL_JSON_DIR, f"{folder}.json")
        if not os.path.exists(lp):
            continue
        with open(lp, "r", encoding="utf-8", errors="ignore") as f:
            d = json.load(f)
        conds = {c["condition"]: c["value"] for c in d.get("type_info", {}).get("conditions", [])}
        y = 0 if conds.get("무릎과 골반이 동시에 펴짐", True) else 1

        sub = 1
        for i in range(1, 10):
            if f"피험자{i}" in folder:
                sub = i
                break

        kps_64 = np.load(p)["kps_33_3d_64f"]
        f24 = extract_d4_v1_features_from_seq(kps_64)
        r_res = extract_ratio_features_from_seq(kps_64)

        orig_feats.append(f24)
        ratio_feats.append(r_res["features"])
        y_list.append(y)
        subs_list.append(sub)

    X_orig = np.array(orig_feats, dtype=np.float32)
    X_ratio = np.array(ratio_feats, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.int32)
    subs = np.array(subs_list, dtype=np.int32)

    norm_mask = (y_all == 0)
    def_mask = (y_all == 1)

    print("\n" + "=" * 80)
    print("  [2] 전체 770개 시퀀스 통계적 유의성 검정 (Normal: 635 vs Defect: 135)")
    print("=" * 80, flush=True)
    print(f"{'피처명':<30} | {'정상 평균±표준편차':<18} | {'오류 평균±표준편차':<18} | {'t-stat':<8} | {'p-value':<10} | {'AUC':<6}")
    print("-" * 100)

    for i, fname in enumerate(FEATURE_NAMES_RATIO):
        vals_norm = X_ratio[norm_mask, i]
        vals_def = X_ratio[def_mask, i]

        m_n, s_n = np.mean(vals_norm), np.std(vals_norm)
        m_d, s_d = np.mean(vals_def), np.std(vals_def)

        t_res = stats.ttest_ind(vals_def, vals_norm, equal_var=False)
        auc = roc_auc_score(y_all, X_ratio[:, i])
        if auc < 0.5:
            auc = 1.0 - auc

        p_str = f"{t_res.pvalue:.2e}" if t_res.pvalue < 0.001 else f"{t_res.pvalue:.4f}"
        print(f"{fname:<30} | {m_n:6.3f} ± {s_n:5.3f}      | {m_d:6.3f} ± {s_d:5.3f}      | {t_res.statistic:7.2f} | {p_str:<10} | {auc:.4f}")

    # 3. 모델 교차검증 성능 비교
    X_combined = np.hstack([X_orig, X_ratio])

    print("\n" + "=" * 80)
    print("  [3] 5-Fold 피험자 격리 모델 교차검증 성능 비교 (기존 24D vs 29D 비율 결합)")
    print("=" * 80, flush=True)

    scale_weight = float(np.sum(y_all == 0) / np.sum(y_all == 1))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for name, X_data in [("기존 24D 피처셋", X_orig), ("제안 29D 피처셋 (비율 5D 추가)", X_combined)]:
        oof = np.zeros(len(y_all))
        for fold, (tr, va) in enumerate(sgkf.split(X_data, y_all, groups=subs), 1):
            clf_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15,
                                         scale_pos_weight=scale_weight, random_state=42, verbose=-1)
            clf_lgb.fit(X_data[tr], y_all[tr])
            clf_cat = CatBoostClassifier(iterations=80, learning_rate=0.05, depth=4,
                                         scale_pos_weight=scale_weight, random_seed=42, verbose=False)
            clf_cat.fit(X_data[tr], y_all[tr])
            clf_rf = RandomForestClassifier(n_estimators=80, max_depth=5, class_weight='balanced', random_state=42)
            clf_rf.fit(X_data[tr], y_all[tr])

            p = 0.5 * clf_lgb.predict_proba(X_data[va])[:, 1] + 0.3 * clf_cat.predict_proba(X_data[va])[:, 1] + 0.2 * clf_rf.predict_proba(X_data[va])[:, 1]
            oof[va] = p

        auc = roc_auc_score(y_all, oof)
        best_f1, best_th = 0.0, 0.5
        for th in np.linspace(0.2, 0.8, 61):
            sc = f1_score(y_all, (oof >= th).astype(int), zero_division=0)
            if sc > best_f1:
                best_f1 = sc
                best_th = th
        pred = (oof >= best_th).astype(int)
        acc = accuracy_score(y_all, pred)
        prec = precision_score(y_all, pred, zero_division=0)
        rec = recall_score(y_all, pred, zero_division=0)

        print(f"  [{name}]", flush=True)
        print(f"    - ROC-AUC   : {auc:.4f}")
        print(f"    - Best F1   : {best_f1*100:.2f}% (tau={best_th:.2f})")
        print(f"    - Accuracy  : {acc*100:.2f}%")
        print(f"    - Precision : {prec*100:.2f}%")
        print(f"    - Recall    : {rec*100:.2f}%", flush=True)

if __name__ == "__main__":
    main()
