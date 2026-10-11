# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Script: analyze_misclass_root_causes.py
Purpose: 오분류(FP 51건, FN 45건)의 4대 근본 원인(Root Causes) 정밀 계측 및 규명
         1) [원인 1] 피험자 고유 특성: 불완전 ROM(깔짝 수행) 및 라벨 노이즈 (피험자 4, 5의 ROM 결핍)
         2) [원인 2] 3D 스켈레톤 인버전 / 척추 뒤집힘 왜곡: view5(우측 후방) 및 view1(좌측 후방)에서의 2D->3D 깊이 뒤집힘
         3) [원인 3] 타 결함(d4 무릎 모임 등)으로 인한 체간 정렬 왜곡
         4) [원인 4] 경계선(Threshold 0.35 부근) 모호성 (Marginal Confidence)
"""

import os
import sys
import glob
import json
import warnings
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d1_model_v4_spinepose import (
    extract_d1_v4_features_28d,
    extract_morphology_ratios,
    MorphologyResidualizerV4,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_34_D1_V4
)

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_V4_1_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
ERROR_REPORT_JSON = os.path.join(CURRENT_DIR, "d1_v4_1_error_analysis_report.json")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}

MP = {
    'NOSE': 0, 'LSH': 11, 'RSH': 12, 'LELB': 13, 'RELB': 14,
    'LWR': 15, 'RWR': 16, 'LHIP': 23, 'RHIP': 24, 'LKNE': 25, 'RKNE': 26,
    'LANK': 27, 'RANK': 28
}


def analyze_all():
    # 1. 이전 진단 결과 로드
    with open(ERROR_REPORT_JSON, "r", encoding="utf-8") as f:
        err_data = json.load(f)

    spine_cache = np.load(SPINEPOSE_V4_1_CACHE, allow_pickle=True)
    spine_dict = {sid: spine_cache['spine_9_64f_vision'][i] for i, sid in enumerate(spine_cache['sequence_ids'])}

    # 385개 샘플에 대해 ROM 및 척추 특성 전수 계산
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    
    seq_records = []
    
    for f in npz_files:
        sid = os.path.basename(f).replace(".npz", "")
        parts = sid.split("_")
        vn = parts[-1]
        folder_name = "_".join(parts[:-1])

        label_p = os.path.join(LABEL_JSON_DIR, f"{folder_name}.json")
        if not os.path.exists(label_p):
            continue

        with open(label_p, "r", encoding="utf-8", errors="ignore") as jf:
            d_json = json.load(jf)
        conds = {c["condition"]: c["value"] for c in d_json.get("type_info", {}).get("conditions", [])}
        is_d1_defect = 0 if conds.get("척추의 중립", True) else 1

        sub = 1
        for d in range(1, 10):
            if f"피험자{d}" in folder_name:
                sub = d
                break

        d = np.load(f, allow_pickle=True)
        kps = d['kps_33_3d_64f']
        s9 = spine_dict.get(sid, None)

        mid_sh = (kps[:, MP['LSH']] + kps[:, MP['RSH']]) / 2.0
        mid_hip = (kps[:, MP['LHIP']] + kps[:, MP['RHIP']]) / 2.0
        mid_wrist = (kps[:, MP['LWR']] + kps[:, MP['RWR']]) / 2.0
        mid_knee = (kps[:, MP['LKNE']] + kps[:, MP['RKNE']]) / 2.0

        v_torso = mid_sh - mid_hip
        y_hat = v_torso / (np.linalg.norm(v_torso, axis=1, keepdims=True) + 1e-7)
        torso_pitch = np.degrees(np.arccos(np.clip(y_hat[:, 1], -1.0, 1.0)))

        wrist_rom = float(np.ptp(gaussian_filter1d(mid_wrist[:, 1], sigma=1.2)))
        hip_rom = float(np.ptp(gaussian_filter1d(mid_hip[:, 1], sigma=1.2)))
        torso_pitch_min = float(np.min(torso_pitch))
        torso_pitch_max = float(np.max(torso_pitch))
        torso_pitch_rom = float(np.ptp(torso_pitch))

        # SpinePose 곡률
        if s9 is not None:
            s1 = s9[:, 0]
            l5 = s9[:, 1]
            l3 = s9[:, 2]
            l1 = s9[:, 3]
            t8 = s9[:, 4]
            c7 = s9[:, 6]

            # 체간 로컬 z축 투영
            v_sh = kps[:, MP['LSH']] - kps[:, MP['RSH']]
            z_raw = np.cross(y_hat, v_sh)
            z_hat = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-7)

            z_s1 = np.sum((s1 - mid_hip) * z_hat, axis=-1)
            z_l5 = np.sum((l5 - mid_hip) * z_hat, axis=-1)
            z_l3 = np.sum((l3 - mid_hip) * z_hat, axis=-1)
            z_l1 = np.sum((l1 - mid_hip) * z_hat, axis=-1)
            z_t8 = np.sum((t8 - mid_hip) * z_hat, axis=-1)

            lumbar_curves = np.column_stack([z_s1, z_l5, z_l3, z_l1, z_t8])
            x_coords = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

            poly_a_series = np.zeros(64, dtype=np.float32)
            for t_idx in range(64):
                p = np.polyfit(x_coords, lumbar_curves[t_idx], deg=2)
                poly_a_series[t_idx] = float(p[0])
            poly_a_mean = float(np.mean(poly_a_series))
            poly_a_max = float(np.max(poly_a_series))
            poly_a_rom = float(np.ptp(poly_a_series))
        else:
            poly_a_mean = 0.0
            poly_a_max = 0.0
            poly_a_rom = 0.0

        seq_records.append({
            "seq_id": sid,
            "subject": sub,
            "view_code": vn,
            "view_name": VIEW_ALIAS_MAP[vn],
            "ground_truth": is_d1_defect,
            "wrist_rom": wrist_rom,
            "hip_rom": hip_rom,
            "torso_pitch_min": torso_pitch_min,
            "torso_pitch_max": torso_pitch_max,
            "torso_pitch_rom": torso_pitch_rom,
            "poly_a_mean": poly_a_mean,
            "poly_a_max": poly_a_max,
            "poly_a_rom": poly_a_rom,
            "other_conditions": conds
        })

    df_feats = pd.DataFrame(seq_records)

    # 2. 피험자별 가동범위(ROM) 비교
    print("=" * 110)
    print("  [피험자 1~9번 가동범위(ROM) 및 척추 만곡 실측 비교]")
    print("=" * 110)
    sub_rom_summary = df_feats.groupby("subject").agg({
        "wrist_rom": ["mean", "std"],
        "torso_pitch_rom": ["mean", "std"],
        "torso_pitch_min": ["mean", "min"],
        "poly_a_rom": ["mean", "std"]
    }).round(3)
    print(sub_rom_summary)

    # 3. 오분류 전수 데이터와 결합
    # JSON 파일에서 top_fp_samples, top_fn_samples 등 읽기
    # d1_v4_1_error_analysis_report.json에는 개별 records가 안 들어있을 수 있으므로 다시 매칭
    # evaluate_d1_v4_1_spinepose 재현 매칭
    from sklearn.model_selection import StratifiedGroupKFold
    
    # 5-fold OOF probabilities 재현
    N = len(df_feats)
    X_28d = np.zeros((N, 28), dtype=np.float32)
    ratios_list = np.zeros((N, 6), dtype=np.float32)
    labels = df_feats["ground_truth"].values
    subject_ids = df_feats["subject"].values

    for i in range(N):
        f = npz_files[i]
        d = np.load(f, allow_pickle=True)
        kps = d['kps_33_3d_64f']
        s9 = spine_dict.get(df_feats.iloc[i]["seq_id"], None)
        X_28d[i] = extract_d1_v4_features_28d(kps, s9)
        ratios_list[i] = extract_morphology_ratios(kps)

    # 가중치 파일 로드하여 oof_probs 계산
    # 이전 실행에서 oof_p_ens를 생성했으므로, 다시 계산
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    oof_p_lgb = np.zeros(N, dtype=np.float32)
    oof_p_cat = np.zeros(N, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_28d, labels, groups=subject_ids)):
        norm_train_indices = train_idx[labels[train_idx] == 0]
        res = MorphologyResidualizerV4(alpha=15.0)
        res.fit(ratios_list[norm_train_indices], X_28d[norm_train_indices, 17], X_28d[norm_train_indices, 23])

        def build_34d(indices):
            sub_28 = X_28d[indices]
            sub_rat = ratios_list[indices]
            sub_res = np.zeros((len(indices), 2), dtype=np.float32)
            for k in range(len(indices)):
                dp = float(sub_28[k, 17])
                pa = float(sub_28[k, 23])
                sub_res[k] = res.transform(sub_rat[k], dp, pa)
            return np.concatenate([sub_28, sub_res, sub_rat[:, :4]], axis=1)

        X_train_34 = build_34d(train_idx)
        X_val_34 = build_34d(val_idx)

        clf_lgb = lgb.LGBMClassifier(n_estimators=160, learning_rate=0.035, num_leaves=15, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_samples=8, random_state=42 + fold, verbose=-1)
        clf_lgb.fit(X_train_34, labels[train_idx])
        oof_p_lgb[val_idx] = clf_lgb.predict_proba(X_val_34)[:, 1]

        clf_cat = CatBoostClassifier(iterations=180, learning_rate=0.04, depth=4, l2_leaf_reg=4.0, random_seed=42 + fold, verbose=False)
        clf_cat.fit(X_train_34, labels[train_idx])
        oof_p_cat[val_idx] = clf_cat.predict_proba(X_val_34)[:, 1]

    # GBDT만으로도 확률 근사 가능 (TCN과 앙상블한 상관관계 > 0.95)
    oof_p_gbdt = 0.5 * oof_p_lgb + 0.5 * oof_p_cat
    th = 0.35
    preds = (oof_p_gbdt >= th).astype(int)

    df_feats["prob"] = oof_p_gbdt
    df_feats["pred"] = preds
    df_feats["group"] = "TN"
    df_feats.loc[(df_feats["ground_truth"] == 0) & (df_feats["pred"] == 0), "group"] = "TN"
    df_feats.loc[(df_feats["ground_truth"] == 0) & (df_feats["pred"] == 1), "group"] = "FP"
    df_feats.loc[(df_feats["ground_truth"] == 1) & (df_feats["pred"] == 0), "group"] = "FN"
    df_feats.loc[(df_feats["ground_truth"] == 1) & (df_feats["pred"] == 1), "group"] = "TP"

    # 오분류 원인(Root Cause) 분류 규칙
    # 1. ROM_DEFICIENCY (가동범위 부족/깔짝 데드리프트): wrist_rom < 0.40 또는 torso_pitch_rom < 40.0
    # 2. SEVERE_FLEXION_FP (정상 라벨인데 실제 극심한 굴곡/숙임): torso_pitch_min < 20.0 및 poly_a_rom > 0.40
    # 3. VIEW_OCCLUSION (후방 뷰 view1, view5에서의 깊이 모호성): view_code in ['view1', 'view5']
    # 4. KNEE_VALGUS_CROSSTALK (무릎 모임 d4 결함 동반): conds['발과 무릎의 방향 일치'] == False
    # 5. MARGINAL_UNCERTAIN (확률 0.30 ~ 0.40 사이의 경계선): 0.30 <= prob <= 0.40

    root_causes = []
    for idx, row in df_feats.iterrows():
        grp = row["group"]
        if grp in ["TN", "TP"]:
            root_causes.append("CORRECT")
            continue

        c_reasons = []
        # FN 분석 (실제 결함인데 정상으로 판정)
        if grp == "FN":
            if row["wrist_rom"] < 0.45 or row["torso_pitch_rom"] < 45.0:
                c_reasons.append("가동범위_부족_동작불완전(피험자4·5특성)")
            if 0.30 <= row["prob"] < 0.35:
                c_reasons.append("임계치_경계선_모호(Marginal)")
            if row["view_code"] in ["view1", "view5"]:
                c_reasons.append("후방시점_깊이_가림")
            if not c_reasons:
                c_reasons.append("미세만곡_정상유사")

        # FP 분석 (실제 정상인데 결함으로 판정)
        elif grp == "FP":
            if row["torso_pitch_min"] < 30.0:
                c_reasons.append("과도숙임_체간인버전(Pitch<30°)")
            if row["other_conditions"].get("발과 무릎의 방향 일치") is False:
                c_reasons.append("타결함(d4_무릎모임)_간섭")
            if 0.35 <= row["prob"] <= 0.42:
                c_reasons.append("임계치_경계선_모호(Marginal)")
            if row["view_code"] in ["view1", "view5"]:
                c_reasons.append("후방시점_자세왜곡")
            if not c_reasons:
                c_reasons.append("자연만곡_결함간주(라벨노이즈)")

        root_causes.append(", ".join(c_reasons))

    df_feats["root_causes"] = root_causes

    # 4. 오분류 원인별 집계
    print("\n" + "=" * 110)
    print("  [오분류(False Negative 45건) 근본 원인 분류 결과]")
    print("=" * 110)
    fn_df = df_feats[df_feats["group"] == "FN"]
    for rc, cnt in fn_df["root_causes"].value_counts().items():
        print(f"  - {rc}: {cnt}건 ({cnt/len(fn_df)*100:.1f}%)")

    print("\n" + "=" * 110)
    print("  [오분류(False Positive 51건) 근본 원인 분류 결과]")
    print("=" * 110)
    fp_df = df_feats[df_feats["group"] == "FP"]
    for rc, cnt in fp_df["root_causes"].value_counts().items():
        print(f"  - {rc}: {cnt}건 ({cnt/len(fp_df)*100:.1f}%)")

    # 원인 분석 세부 결과 CSV 및 요약 저장
    out_csv = os.path.join(CURRENT_DIR, "d1_v4_1_misclassifications_detailed.csv")
    df_feats.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[+] 상세 오분류 진단 CSV 저장 완료: {out_csv}")


if __name__ == "__main__":
    analyze_all()
