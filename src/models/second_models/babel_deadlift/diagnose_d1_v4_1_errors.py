# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: diagnose_d1_v4_1_errors.py
Purpose: d1 v4.1 (SpinePose v4.1 정제 모델) 5-Fold OOF 예측 기반 오분류(FP 50건, FN 43건) 전수 정밀 진단
         1) FP / FN 전수 목록화 및 확률 마진 분석
         2) 피험자별 편향 및 원천 데이터 라벨 노이즈 규명 (피험자 4, 5, 8 FN 집중 및 1, 6, 7, 9 FP 집중 원인)
         3) 복합 결함(d2 시선, d3 바벨이격, d4 무릎모임, d5 전방쏠림)과의 교차 간섭(Cross-talk) 분석
         4) 공식 5대 시점(시점 A~E)별 오분류 메커니즘 (가림 및 시점 편향)
         5) 34D 생체역학 피처별 4대 집단(TN, FP, FN, TP) 통계 검정 및 원인 피처 도출
"""

import os
import sys
import glob
import json
import warnings
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_deadlift.d1_model_v4_spinepose import (
    extract_d1_v4_features_28d,
    extract_morphology_ratios,
    MorphologyResidualizerV4,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_34_D1_V4
)

warnings.filterwarnings("ignore")

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
SPINEPOSE_V4_1_CACHE = r"D:\Data\피처_데이터\deadlift_spinepose_denoised_v4_1_64f.npz"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUTPUT_ANALYSIS_JSON = os.path.join(CURRENT_DIR, "d1_v4_1_error_analysis_report.json")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


def run_error_diagnosis():
    print("=" * 110)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v4.1 오분류(Misclassification) 원인 심층 진단 하네스 가동")
    print("=" * 110)

    # 1. 데이터 로드 및 메타데이터 파싱
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    spine_cache = np.load(SPINEPOSE_V4_1_CACHE, allow_pickle=True)
    spine_dict = {sid: spine_cache['spine_9_64f_vision'][i] for i, sid in enumerate(spine_cache['sequence_ids'])}

    sequence_ids = []
    labels = []
    subject_ids = []
    view_ids = []
    view_names = []
    other_conditions_list = []
    kps_list = []
    spine_list = []

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
        kps_33_64f = d['kps_33_3d_64f']

        sequence_ids.append(sid)
        labels.append(is_d1_defect)
        subject_ids.append(sub)
        view_ids.append(vn)
        view_names.append(VIEW_ALIAS_MAP.get(vn, vn))
        other_conditions_list.append(conds)
        kps_list.append(kps_33_64f)
        spine_list.append(spine_dict.get(sid, None))

    labels = np.array(labels, dtype=int)
    subject_ids = np.array(subject_ids, dtype=int)
    view_ids = np.array(view_ids)
    N = len(labels)

    # 피처 추출
    print(f"[*] 385개 시퀀스 피처 추출 및 5-Fold OOF 재생성...")
    X_28d = np.zeros((N, 28), dtype=np.float32)
    ratios_list = np.zeros((N, 6), dtype=np.float32)
    X_temporal = np.zeros((N, 64, 24), dtype=np.float32)

    for i in range(N):
        kps = kps_list[i]
        s9 = spine_list[i]
        X_28d[i] = extract_d1_v4_features_28d(kps, s9)
        ratios_list[i] = extract_morphology_ratios(kps)
        X_temporal[i] = extract_temporal_24d_with_spine(kps, s9)

    # 5-Fold OOF 재생성
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p_lgb = np.zeros(N, dtype=np.float32)
    oof_p_cat = np.zeros(N, dtype=np.float32)
    oof_p_tcn = np.zeros(N, dtype=np.float32)
    X_34d_all = np.zeros((N, 34), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_28d, labels, groups=subject_ids)):
        norm_train_indices = train_idx[labels[train_idx] == 0]
        res = MorphologyResidualizerV4(alpha=15.0)
        res.fit(
            ratios_list[norm_train_indices],
            X_28d[norm_train_indices, 17],
            X_28d[norm_train_indices, 23]
        )

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
        X_34d_all[val_idx] = X_val_34
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=160, learning_rate=0.035, num_leaves=15, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=8,
            random_state=42 + fold, verbose=-1
        )
        clf_lgb.fit(X_train_34, y_train)
        oof_p_lgb[val_idx] = clf_lgb.predict_proba(X_val_34)[:, 1]

        clf_cat = CatBoostClassifier(
            iterations=180, learning_rate=0.04, depth=4, l2_leaf_reg=4.0,
            random_seed=42 + fold, verbose=False
        )
        clf_cat.fit(X_train_34, y_train)
        oof_p_cat[val_idx] = clf_cat.predict_proba(X_val_34)[:, 1]

        tcn_model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
        optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        X_tr_t = torch.tensor(X_temporal[train_idx].transpose(0, 2, 1), dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_va_t = torch.tensor(X_temporal[val_idx].transpose(0, 2, 1), dtype=torch.float32)

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        tcn_model.train()
        for epoch in range(30):
            for bx, by in loader:
                optimizer.zero_grad()
                out = tcn_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        tcn_model.eval()
        with torch.no_grad():
            val_out = tcn_model(X_va_t).squeeze(1)
            oof_p_tcn[val_idx] = torch.sigmoid(val_out).numpy()

    oof_p_gbdt = 0.5 * oof_p_lgb + 0.5 * oof_p_cat
    oof_p_ens = 0.5 * oof_p_gbdt + 0.5 * oof_p_tcn

    th = 0.35
    preds_binary = (oof_p_ens >= th).astype(int)

    # 4대 그룹 분류
    # TN: y=0, pred=0
    # FP: y=0, pred=1 (오분류 정상)
    # FN: y=1, pred=0 (오분류 결함)
    # TP: y=1, pred=1
    group_labels = []
    for y, p in zip(labels, preds_binary):
        if y == 0 and p == 0:
            group_labels.append("TN")
        elif y == 0 and p == 1:
            group_labels.append("FP")
        elif y == 1 and p == 0:
            group_labels.append("FN")
        else:
            group_labels.append("TP")
    group_labels = np.array(group_labels)

    tn_cnt = np.sum(group_labels == "TN")
    fp_cnt = np.sum(group_labels == "FP")
    fn_cnt = np.sum(group_labels == "FN")
    tp_cnt = np.sum(group_labels == "TP")

    print(f"[*] 그룹별 분포: TN={tn_cnt}, FP={fp_cnt}, FN={fn_cnt}, TP={tp_cnt} (총 오분류 {fp_cnt + fn_cnt}건)")

    # -------------------------------------------------------------
    # 1. 오분류 전수 데이터 세부 테이블 구축
    # -------------------------------------------------------------
    records = []
    for i in range(N):
        rec = {
            "seq_idx": i,
            "seq_id": sequence_ids[i],
            "subject": int(subject_ids[i]),
            "view_code": view_ids[i],
            "view_name": view_names[i],
            "ground_truth": int(labels[i]),
            "prediction": int(preds_binary[i]),
            "group": group_labels[i],
            "prob_ens": float(oof_p_ens[i]),
            "prob_lgb": float(oof_p_lgb[i]),
            "prob_cat": float(oof_p_cat[i]),
            "prob_tcn": float(oof_p_tcn[i]),
            "margin_from_th": float(oof_p_ens[i] - th),
            "other_conditions": other_conditions_list[i]
        }
        records.append(rec)

    df_all = pd.DataFrame(records)
    df_fp = df_all[df_all["group"] == "FP"].copy().sort_values(by="prob_ens", ascending=False)
    df_fn = df_all[df_all["group"] == "FN"].copy().sort_values(by="prob_ens", ascending=True)

    print("\n" + "=" * 90)
    print("  [1. 피험자별 오분류(FP/FN) 분포 분석]")
    print("=" * 90)
    sub_breakdown = {}
    for sub in range(1, 10):
        sub_mask = (df_all["subject"] == sub)
        sub_df = df_all[sub_mask]
        n_total = len(sub_df)
        n_norm = int(np.sum(sub_df["ground_truth"] == 0))
        n_def = int(np.sum(sub_df["ground_truth"] == 1))
        n_tn = int(np.sum(sub_df["group"] == "TN"))
        n_fp = int(np.sum(sub_df["group"] == "FP"))
        n_fn = int(np.sum(sub_df["group"] == "FN"))
        n_tp = int(np.sum(sub_df["group"] == "TP"))
        sub_breakdown[f"피험자_{sub}"] = {
            "total": n_total, "normal": n_norm, "defect": n_def,
            "TN": n_tn, "FP": n_fp, "FN": n_fn, "TP": n_tp,
            "fp_rate": round(n_fp / n_norm * 100, 2) if n_norm > 0 else 0.0,
            "fn_rate": round(n_fn / n_def * 100, 2) if n_def > 0 else 0.0,
            "mean_prob_ens_norm": round(float(sub_df[sub_df["ground_truth"] == 0]["prob_ens"].mean()), 4),
            "mean_prob_ens_def": round(float(sub_df[sub_df["ground_truth"] == 1]["prob_ens"].mean()), 4) if n_def > 0 else 0.0
        }
        print(f"피험자 {sub}번: Total {n_total} (정상 {n_norm}, 결함 {n_def}) | TN={n_tn}, FP={n_fp} ({sub_breakdown[f'피험자_{sub}']['fp_rate']}%), FN={n_fn} ({sub_breakdown[f'피험자_{sub}']['fn_rate']}%), TP={n_tp} | P_norm_mean={sub_breakdown[f'피험자_{sub}']['mean_prob_ens_norm']:.3f}, P_def_mean={sub_breakdown[f'피험자_{sub}']['mean_prob_ens_def']:.3f}")

    print("\n" + "=" * 90)
    print("  [2. 공식 5대 시점별 오분류(FP/FN) 분포 분석]")
    print("=" * 90)
    view_breakdown = {}
    for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
        v_name = VIEW_ALIAS_MAP[vn]
        v_df = df_all[df_all["view_code"] == vn]
        n_total = len(v_df)
        n_norm = int(np.sum(v_df["ground_truth"] == 0))
        n_def = int(np.sum(v_df["ground_truth"] == 1))
        n_tn = int(np.sum(v_df["group"] == "TN"))
        n_fp = int(np.sum(v_df["group"] == "FP"))
        n_fn = int(np.sum(v_df["group"] == "FN"))
        n_tp = int(np.sum(v_df["group"] == "TP"))
        view_breakdown[v_name] = {
            "view_code": vn, "total": n_total, "normal": n_norm, "defect": n_def,
            "TN": n_tn, "FP": n_fp, "FN": n_fn, "TP": n_tp,
            "fp_rate": round(n_fp / n_norm * 100, 2) if n_norm > 0 else 0.0,
            "fn_rate": round(n_fn / n_def * 100, 2) if n_def > 0 else 0.0,
            "accuracy": round((n_tn + n_tp) / n_total * 100, 2)
        }
        print(f"{v_name}: Total {n_total} | TN={n_tn}, FP={n_fp} ({view_breakdown[v_name]['fp_rate']}%), FN={n_fn} ({view_breakdown[v_name]['fn_rate']}%), TP={n_tp} | Acc: {view_breakdown[v_name]['accuracy']}%")

    print("\n" + "=" * 90)
    print("  [3. 타 결함(d2, d3, d4, d5)과의 교차 간섭 (Cross-talk) 분석]")
    print("=" * 90)
    # AI-Hub 라벨 상 타 조건: 시선, 바벨의 위치(이격), 무릎의 방향(모임), 시작/마무리 자세(전방쏠림) 등
    # 정상 동작(y=0)인데 FP가 된 샘플들에서 다른 결함이 발생했는지 점검!
    other_defect_keys = {
        "시선": "d2_gaze",
        "바벨": "d3_barbell_distance",
        "무릎": "d4_knee_valgus",
        "시작/마무리": "d5_setup_finish"
    }

    crosstalk_analysis = {}
    for group_name in ["TN", "FP", "FN", "TP"]:
        grp_df = df_all[df_all["group"] == group_name]
        g_len = len(grp_df)
        cond_counts = {}
        for cond_name, col_code in other_defect_keys.items():
            # condition value가 False인 경우(결함) 카운트
            defect_cnt = 0
            for oc in grp_df["other_conditions"]:
                for k, v in oc.items():
                    if cond_name in k and v is False:
                        defect_cnt += 1
                        break
            cond_counts[col_code] = {
                "defect_count": defect_cnt,
                "rate": round(defect_cnt / g_len * 100, 2) if g_len > 0 else 0.0
            }
        crosstalk_analysis[group_name] = cond_counts
        print(f"[{group_name} (N={g_len})] 타 결함 발생률: "
              f"시선 결함(d2): {cond_counts['d2_gaze']['rate']}%, "
              f"바벨 이격(d3): {cond_counts['d3_barbell_distance']['rate']}%, "
              f"무릎 모임(d4): {cond_counts['d4_knee_valgus']['rate']}%, "
              f"시작/마무리(d5): {cond_counts['d5_setup_finish']['rate']}%")

    print("\n" + "=" * 90)
    print("  [4. 34D 생체역학 피처별 통계적 유의차 검정 (TN vs FP, TP vs FN)]")
    print("=" * 90)
    # TN vs FP: 실제 정상인데 모델이 결함으로 오판한 원인 피처 (왜 척추가 굽었다고 판단했는가?)
    # TP vs FN: 실제 결함인데 모델이 정상으로 오판한 원인 피처 (왜 척추가 굽지 않았다고 판단했는가?)
    feature_stats = {}
    feature_names = FEATURE_NAMES_34_D1_V4

    tn_idx = np.where(group_labels == "TN")[0]
    fp_idx = np.where(group_labels == "FP")[0]
    fn_idx = np.where(group_labels == "FN")[0]
    tp_idx = np.where(group_labels == "TP")[0]

    for fid, fname in enumerate(feature_names):
        val_tn = X_34d_all[tn_idx, fid]
        val_fp = X_34d_all[fp_idx, fid]
        val_fn = X_34d_all[fn_idx, fid]
        val_tp = X_34d_all[tp_idx, fid]

        # Mann-Whitney U test
        u_fp, p_fp = stats.mannwhitneyu(val_tn, val_fp, alternative='two-sided')
        u_fn, p_fn = stats.mannwhitneyu(val_tp, val_fn, alternative='two-sided')

        feature_stats[fname] = {
            "feature_id": fid,
            "TN_mean": float(np.mean(val_tn)),
            "TN_std": float(np.std(val_tn)),
            "FP_mean": float(np.mean(val_fp)),
            "FP_std": float(np.std(val_fp)),
            "FN_mean": float(np.mean(val_fn)),
            "FN_std": float(np.std(val_fn)),
            "TP_mean": float(np.mean(val_tp)),
            "TP_std": float(np.std(val_tp)),
            "FP_vs_TN_pval": float(p_fp),
            "FN_vs_TP_pval": float(p_fn)
        }

    # FP 유발 상위 피처 (p-value 가장 작은 피처)
    sorted_fp_feats = sorted(feature_stats.items(), key=lambda x: x[1]["FP_vs_TN_pval"])
    print("\n[*] FP(정상을 결함으로 오판)를 유도한 유의 피처 Top 7 (p < 0.01):")
    for fname, st in sorted_fp_feats[:7]:
        print(f"  - {fname:35s}: TN mean={st['TN_mean']:8.3f} vs FP mean={st['FP_mean']:8.3f} (p={st['FP_vs_TN_pval']:.2e})")

    # FN 유발 상위 피처 (p-value 가장 작은 피처)
    sorted_fn_feats = sorted(feature_stats.items(), key=lambda x: x[1]["FN_vs_TP_pval"])
    print("\n[*] FN(결함을 정상으로 오판)을 유도한 유의 피처 Top 7 (p < 0.01):")
    for fname, st in sorted_fn_feats[:7]:
        print(f"  - {fname:35s}: TP mean={st['TP_mean']:8.3f} vs FN mean={st['FN_mean']:8.3f} (p={st['FN_vs_TP_pval']:.2e})")

    # -------------------------------------------------------------
    # 5. 극단적 오분류 케이스 Top 10 상세 분석
    # -------------------------------------------------------------
    top_fp_samples = df_fp.head(10).to_dict(orient="records")
    top_fn_samples = df_fn.head(10).to_dict(orient="records")

    print("\n" + "=" * 90)
    print("  [5. 가장 확신에 차서 오분류한 극단 케이스 (Top FP & Top FN)]")
    print("=" * 90)
    print("[*] 극단적 False Positive (실제 정상인데 모델 확률 > 0.60):")
    for r in top_fp_samples[:5]:
        print(f"  - [{r['seq_id']}] 피험자 {r['subject']} ({r['view_name']}): Prob={r['prob_ens']:.3f} (GBDT={r['prob_lgb']:.2f}/{r['prob_cat']:.2f}, TCN={r['prob_tcn']:.2f}) | 타조건: {r['other_conditions']}")

    print("\n[*] 극단적 False Negative (실제 결함인데 모델 확률 < 0.15):")
    for r in top_fn_samples[:5]:
        print(f"  - [{r['seq_id']}] 피험자 {r['subject']} ({r['view_name']}): Prob={r['prob_ens']:.3f} (GBDT={r['prob_lgb']:.2f}/{r['prob_cat']:.2f}, TCN={r['prob_tcn']:.2f}) | 타조건: {r['other_conditions']}")

    # -------------------------------------------------------------
    # 6. JSON 보고서 저장
    # -------------------------------------------------------------
    summary_data = {
        "total_samples": N,
        "confusion_matrix": {
            "TN": int(tn_cnt), "FP": int(fp_cnt),
            "FN": int(fn_cnt), "TP": int(tp_cnt)
        },
        "subject_breakdown": sub_breakdown,
        "view_breakdown": view_breakdown,
        "crosstalk_analysis": crosstalk_analysis,
        "feature_stats": feature_stats,
        "top_fp_samples": top_fp_samples,
        "top_fn_samples": top_fn_samples
    }

    with open(OUTPUT_ANALYSIS_JSON, "w", encoding="utf-8") as jf:
        json.dump(summary_data, jf, indent=2, ensure_ascii=False)
    print(f"\n[+] 정밀 진단 결과 JSON 저장 완료: {OUTPUT_ANALYSIS_JSON}")


if __name__ == "__main__":
    run_error_diagnosis()
