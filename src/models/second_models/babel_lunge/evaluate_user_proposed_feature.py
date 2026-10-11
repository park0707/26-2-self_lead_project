# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: evaluate_user_proposed_feature.py
Purpose: 사용자가 제안한 [앞다리 고관절-무릎 수직(Y) 거리] 피처의 타당성 실측 전수 평가
         1. 이론적/생체역학적 정합성 분석
         2. AI-Hub 1,081개 전수 데이터 대상 실측 통계 (정상 vs 결함)
            - 원천 픽셀/좌표 Y거리 vs 체간 정규화 Y거리 vs 대퇴골 정규화 Y거리
            - 전체 데이터셋 및 5대 시점별 (특히 정면 view3) 분별력 (Cohen's d, ROC-AUC, T-검정 p-value)
         3. v3 피처셋과의 상관관계 및 다중공선성 검사
         4. bl1_v3에 해당 피처 추가 투입 시의 실측 5-Fold F1/AUC/정확도 영향 검증
"""

import os
import sys
import glob
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

# 경로
NPZ_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\04_바벨_런지\2D"
V3_CACHE_PATH = r"D:\Data\피처_데이터\bl1_v3_dataset_cache.pkl"
EVAL_OUT_JSON = r"D:\src\models\second_models\babel_lunge\user_feature_evaluation.json"

sys.path.append(r"D:\src\features")
from lunge_c1_features_v3 import (
    detect_lead_leg_robust,
    find_dual_anchor_inflection_point,
    safe_savgol_filter,
    vector_angle_3d,
    MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE,
    MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER
)


def evaluate_proposed_feature():
    print("=" * 80)
    print(" [vidio-health-trainer] 사용자 제안 피처: 앞다리 고관절-무릎 수직(Y) 거리 타당성 평가")
    print("=" * 80)

    # v3 캐시 로드 (라벨 및 메타데이터 확보)
    with open(V3_CACHE_PATH, "rb") as f:
        v3_cache = pickle.load(f)

    y = v3_cache["y"]
    groups = v3_cache.get("groups", v3_cache.get("subjects"))
    views = v3_cache["views"]
    clip_names = v3_cache["clip_names"]
    X_v3 = v3_cache["X"]
    v3_feature_names = v3_cache["feature_names"]

    n_samples = len(y)
    print(f"[*] 분석 대상 샘플 수: {n_samples}개 (정상: {np.sum(y==0)}건, 결함: {np.sum(y==1)}건)")

    # 1. 1,081개 NPZ 전수에서 고관절-무릎 수직 거리 피처 변형 3종 산출
    # Form 1: raw_delta_y = |y_knee - y_hip| (최저점 프레임)
    # Form 2: torso_norm_delta_y = |y_knee - y_hip| / L_torso (체간 정규화)
    # Form 3: femur_norm_delta_y = |y_knee - y_hip| / L_femur (대퇴골 정규화)
    # Form 4: signed_delta_y = (y_knee - y_hip) / L_torso (부호 있는 수직 거리: 무릎이 고관절보다 아래면 +)
    feat_raw_dy = np.zeros(n_samples, dtype=np.float32)
    feat_torso_dy = np.zeros(n_samples, dtype=np.float32)
    feat_femur_dy = np.zeros(n_samples, dtype=np.float32)
    feat_signed_dy = np.zeros(n_samples, dtype=np.float32)

    print("[*] 1,081개 시퀀스에서 제안 피처 계산 중...")
    for idx, cname in enumerate(clip_names):
        view_code = views[idx]
        npz_path = os.path.join(NPZ_DIR, f"{cname}_{view_code}.npz")
        data = np.load(npz_path)
        kps_64f = data["kps_33_3d_64f"]

        lead_side = detect_lead_leg_robust(kps_64f)
        hip_idx = MP_LEFT_HIP if lead_side == "left" else MP_RIGHT_HIP
        knee_idx = MP_LEFT_KNEE if lead_side == "left" else MP_RIGHT_KNEE
        ankle_idx = MP_LEFT_ANKLE if lead_side == "left" else MP_RIGHT_ANKLE

        # 3D 각도 궤적 및 변곡점 탐색
        lead_angles = np.zeros(64, dtype=np.float32)
        for f in range(64):
            v_fem = kps_64f[f, hip_idx] - kps_64f[f, knee_idx]
            v_tib = kps_64f[f, ankle_idx] - kps_64f[f, knee_idx]
            lead_angles[f] = vector_angle_3d(v_fem, v_tib)
        angles_sg = safe_savgol_filter(lead_angles, window=7, poly=2)
        inflection_idx, _, _ = find_dual_anchor_inflection_point(kps_64f, angles_sg)

        # 최저점에서의 고관절과 무릎 위치
        hip_pt = kps_64f[inflection_idx, hip_idx]
        knee_pt = kps_64f[inflection_idx, knee_idx]

        # 체간 길이 및 대퇴골 길이 스케일
        torso_len = np.linalg.norm(kps_64f[inflection_idx, MP_LEFT_SHOULDER] - kps_64f[inflection_idx, MP_LEFT_HIP])
        if torso_len < 1e-4:
            torso_len = 1.0
        femur_len = np.linalg.norm(hip_pt - knee_pt)
        if femur_len < 1e-4:
            femur_len = 1.0

        # MediaPipe에서 Y축은 아래쪽이 + 방향:
        # 무릎이 고관절보다 아래에 있으면 knee_pt[1] > hip_pt[1]
        delta_y_abs = abs(knee_pt[1] - hip_pt[1])
        delta_y_signed = knee_pt[1] - hip_pt[1]

        feat_raw_dy[idx] = delta_y_abs
        feat_torso_dy[idx] = delta_y_abs / torso_len
        feat_femur_dy[idx] = delta_y_abs / femur_len
        feat_signed_dy[idx] = delta_y_signed / torso_len

    # 2. 통계적 분별력 분석 (정상 vs 결함)
    print("\n" + "=" * 60)
    print(" [통계 검증 1] 전체 데이터셋(1,081건) 대상 집단 간 분별력")
    print("=" * 60)

    clean_mask = (y == 0)
    defect_mask = (y == 1)

    candidate_dict = {
        "1. 원천 절대 수직거리 |Δy|": feat_raw_dy,
        "2. 체간 정규화 수직거리 (|Δy| / L_torso) [추천]": feat_torso_dy,
        "3. 대퇴골 정규화 수직거리 (|Δy| / L_femur)": feat_femur_dy,
        "4. 부호 있는 수직거리 ((y_k - y_h) / L_torso)": feat_signed_dy
    }

    stat_results = {}
    for cname, arr in candidate_dict.items():
        val_clean = arr[clean_mask]
        val_defect = arr[defect_mask]

        m_clean, s_clean = float(np.mean(val_clean)), float(np.std(val_clean))
        m_defect, s_defect = float(np.mean(val_defect)), float(np.std(val_defect))

        # T-검정 및 Cohen's d (효과 크기)
        t_stat, p_val = stats.ttest_ind(val_clean, val_defect, equal_var=False)
        pooled_std = np.sqrt((s_clean**2 + s_defect**2) / 2.0)
        cohen_d = float(abs(m_clean - m_defect) / (pooled_std + 1e-8))

        # 단독 피처로서의 ROC-AUC (결함=1 분별력)
        # 결함일수록 delta_y가 커지므로 arr 자체가 positive score
        auc = roc_auc_score(y, arr)
        if auc < 0.5:
            auc = 1.0 - auc

        stat_results[cname] = {
            "mean_clean": round(m_clean, 4),
            "std_clean": round(s_clean, 4),
            "mean_defect": round(m_defect, 4),
            "std_defect": round(s_defect, 4),
            "t_statistic": round(float(t_stat), 3),
            "p_value": f"{p_val:.2e}",
            "cohens_d": round(cohen_d, 3),
            "roc_auc": round(float(auc), 4)
        }

        print(f"\n* {cname}:")
        print(f"  - 정상 (Clean): 평균 {m_clean:.4f} (±{s_clean:.4f})")
        print(f"  - 결함 (Defect): 평균 {m_defect:.4f} (±{s_defect:.4f})")
        print(f"  - 분별 효과 크기 (Cohen's d): {cohen_d:.3f} | p-value: {p_val:.2e}")
        print(f"  - 단독 피처 ROC-AUC: {auc*100:.2f}%")

    # 3. 5대 시점별, 특히 사용자가 강조한 [정면 view3]에서의 실측 검증
    print("\n" + "=" * 60)
    print(" [통계 검증 2] 5대 카메라 시점별 단독 ROC-AUC (정면 view3 집중 감사)")
    print("=" * 60)

    view_auc_results = {}
    VIEW_ALIAS_MAP = {
        'view5': '시점 A (우측 후방 45°)',
        'view2': '시점 B (우측 전방 45°)',
        'view3': '시점 C (정면 0°)',
        'view4': '시점 D (좌측 전방 45°)',
        'view1': '시점 E (좌측 후방 45°)',
    }

    target_feat = feat_torso_dy  # 체간 정규화 수직거리

    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_idx = (views == v_code)
        y_v = y[v_idx]
        f_v = target_feat[v_idx]

        m_c = float(np.mean(f_v[y_v == 0]))
        m_d = float(np.mean(f_v[y_v == 1]))
        auc_v = float(roc_auc_score(y_v, f_v))
        if auc_v < 0.5:
            auc_v = 1.0 - auc_v

        view_auc_results[v_code] = {
            "alias": v_alias,
            "mean_clean": round(m_c, 4),
            "mean_defect": round(m_d, 4),
            "roc_auc": round(auc_v, 4)
        }
        print(f"  - {v_alias} ({v_code}): 정상 평균={m_c:.4f} vs 결함 평균={m_d:.4f} -> 단독 AUC: {auc_v*100:.2f}%")

    # 4. v3 기존 피처들과의 상관관계 (다중공선성) 점검
    print("\n" + "=" * 60)
    print(" [상관관계 점검 3] v3 기존 피처셋과의 공선성 분석")
    print("=" * 60)
    df_v3 = pd.DataFrame(X_v3, columns=v3_feature_names)
    s_feat = pd.Series(target_feat)
    corrs = df_v3.apply(lambda col: float(col.corr(s_feat)))
    top_corrs = corrs.abs().sort_values(ascending=False).head(5)

    print("* 제안 피처와 상관관계가 높은 v3 기존 피처 Top 5:")
    corr_records = []
    for col_name in top_corrs.index:
        r_val = float(corrs[col_name])
        corr_records.append({"feature": col_name, "corr": round(r_val, 4)})
        print(f"  - {col_name:<30}: r = {r_val:+.4f}")

    # 5. v3 모델에 제안 피처를 추가했을 때의 5-Fold 실측 성능 검증 (Ablation/Addition)
    print("\n" + "=" * 60)
    print(" [실측 검증 4] v3(26D) + 제안 피처(1D) = 27D 모델 5-Fold 교차검증")
    print("=" * 60)

    X_new = np.hstack([X_v3, target_feat.reshape(-1, 1)])
    sgkf = StratifiedGroupKFold(n_splits=5)
    spw = float(np.sum(y == 0) / np.sum(y == 1))

    oof_lgb = np.zeros(n_samples)
    oof_cat = np.zeros(n_samples)

    for tr, val in sgkf.split(X_new, y, groups=groups):
        td, vd = lgb.Dataset(X_new[tr], label=y[tr]), lgb.Dataset(X_new[val], label=y[val])
        ml = lgb.train(
            {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.04, 'num_leaves': 23, 'max_depth': 5, 'scale_pos_weight': spw, 'colsample_bytree': 0.7, 'reg_lambda': 3.0, 'reg_alpha': 0.5, 'min_child_samples': 20, 'verbose': -1, 'n_jobs': -1},
            td, num_boost_round=350, valid_sets=[vd], callbacks=[lgb.early_stopping(35, verbose=False)]
        )
        oof_lgb[val] = ml.predict(X_new[val], num_iteration=ml.best_iteration)

        mc = CatBoostClassifier(iterations=400, learning_rate=0.04, depth=5, scale_pos_weight=spw, l2_leaf_reg=3.0, rsm=0.7, random_seed=42, verbose=False, early_stopping_rounds=40)
        mc.fit(X_new[tr], y[tr], eval_set=(X_new[val], y[val]), verbose=False)
        oof_cat[val] = mc.predict_proba(X_new[val])[:, 1]

    oof_ens = 0.35 * oof_lgb + 0.65 * oof_cat
    auc_new = roc_auc_score(y, oof_ens)

    # 3-State 그리드 서치 (보류율 <= 10.0%)
    best_f1, best_acc, best_unc = 0, 0, 0
    best_tc, best_td = 0, 0
    for tc in np.linspace(0.25, 0.48, 47):
        for td in np.linspace(tc + 0.04, 0.65, 43):
            unc = np.mean((oof_ens >= tc) & (oof_ens < td))
            if unc > 0.10:
                continue
            dec = (oof_ens < tc) | (oof_ens >= td)
            yp = (oof_ens[dec] >= td).astype(int)
            f1 = f1_score(y[dec], yp)
            if f1 > best_f1:
                best_f1 = f1
                best_acc = accuracy_score(y[dec], yp)
                best_unc = unc
                best_tc, best_td = tc, td

    print(f"\n[+] v3 (기존 26D): AUC 88.52% | 3-State F1 75.66% | 정확도 85.73% | 보류율 9.90%")
    print(f"[+] v3 + 제안 피처(27D): AUC {auc_new*100:.2f}% | 3-State F1 {best_f1*100:.2f}% | 정확도 {best_acc*100:.2f}% | 보류율 {best_unc*100:.2f}%")

    eval_summary = {
        "statistical_metrics": stat_results,
        "viewpoint_aucs": view_auc_results,
        "top_correlations": corr_records,
        "model_impact": {
            "v3_base_auc": 0.8852,
            "v3_base_3state_f1": 0.7566,
            "v3_base_acc": 0.8573,
            "v3_plus_proposed_auc": float(auc_new),
            "v3_plus_proposed_3state_f1": float(best_f1),
            "v3_plus_proposed_acc": float(best_acc),
            "v3_plus_proposed_unc": float(best_unc)
        }
    }

    os.makedirs(os.path.dirname(EVAL_OUT_JSON), exist_ok=True)
    with open(EVAL_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[+] 평가 결과 JSON 저장 완료: {EVAL_OUT_JSON}")


if __name__ == "__main__":
    evaluate_proposed_feature()
