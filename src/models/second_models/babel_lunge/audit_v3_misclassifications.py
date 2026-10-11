# -*- coding: utf-8 -*-
"""
바벨 런지 Condition 2: bl2_v3 5-Fold OOF 오분류 데이터 전수 역추적 및 원인 정밀 분석
1. False Positive (FP, 정상인데 결함으로 오판): 몇 건, 어느 시점, 어느 피험자, 주요 피처 편차
2. False Negative (FN, 결함인데 정상으로 오판): 몇 건, 어느 시점, 어느 피험자, 주요 피처 편차
3. Retake (판정 보류): 108건의 특성
4. AI-Hub 메타데이터 및 클립명 대조
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

sys.stdout.reconfigure(encoding='utf-8')

def audit_v3_errors():
    v3_cache = r"D:\src\models\second_models\babel_lunge\lunge_c2_features_v3_cache.npz"
    v3_json = r"D:\src\models\second_models\babel_lunge\bl2_v3_benchmark_results.json"
    
    if not os.path.exists(v3_cache) or not os.path.exists(v3_json):
        print("[-] Cache or JSON not found!")
        return

    data = np.load(v3_cache, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]
    views = data["views"]
    clips = data["clip_names"]

    with open(v3_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tau_n = meta["tau_normal"]
    tau_d = meta["tau_defect"]

    from bl2_v3 import BabelLungeCondition2ModelV3, FEATURE_NAMES_17D
    bundle_path = r"D:\src\models\second_models\babel_lunge\weights\bl2_v3_bundle.pkl"
    model = BabelLungeCondition2ModelV3.load_bundle(bundle_path)

    # 전체 확률 예측
    probs = model.predict_proba(X)

    # 3-State 예측 판정
    # 0: CLEAN, 1: DEFECT, 2: RETAKE
    states = np.zeros(len(y), dtype=int)
    states[probs <= tau_n] = 0
    states[probs >= tau_d] = 1
    states[(probs > tau_n) & (probs < tau_d)] = 2

    decided_mask = (states != 2)
    y_dec = y[decided_mask]
    pred_dec = states[decided_mask]
    prob_dec = probs[decided_mask]
    sub_dec = subjects[decided_mask]
    view_dec = views[decided_mask]
    clip_dec = clips[decided_mask]
    X_dec = X[decided_mask]

    tp_mask = (y_dec == 1) & (pred_dec == 1)
    tn_mask = (y_dec == 0) & (pred_dec == 0)
    fp_mask = (y_dec == 0) & (pred_dec == 1)
    fn_mask = (y_dec == 1) & (pred_dec == 0)
    retake_mask = ~decided_mask

    print("=" * 70)
    print("[1] bl2_v3 전체 3-State 판정 현황 요약 (전체 1,081건)")
    print(f"  - 전체 샘플 수: {len(y)}건 (정상: {np.sum(y==0)}건, 결함: {np.sum(y==1)}건)")
    print(f"  - 유효 판정(Coverage): {np.sum(decided_mask)}건 ({np.mean(decided_mask)*100:.2f}%)")
    print(f"  - 판정 보류(Retake)  : {np.sum(retake_mask)}건 ({np.mean(retake_mask)*100:.2f}%)")
    print(f"  - True Positive (TP) : {np.sum(tp_mask)}건")
    print(f"  - True Negative (TN) : {np.sum(tn_mask)}건")
    print(f"  - False Positive(FP) : {np.sum(fp_mask)}건 (정상인데 결함으로 오진 -> 과잉 경보)")
    print(f"  - False Negative(FN) : {np.sum(fn_mask)}건 (결함인데 정상으로 오진 -> 치명적 놓침)")
    print("=" * 70)

    # 시점별 오분류 분포
    view_summary = []
    for vn in sorted(list(np.unique(views))):
        v_idx = np.where(views == vn)[0]
        v_dec_idx = np.where(view_dec == vn)[0]
        v_fp = np.sum(fp_mask[v_dec_idx])
        v_fn = np.sum(fn_mask[v_dec_idx])
        v_tp = np.sum(tp_mask[v_dec_idx])
        v_tn = np.sum(tn_mask[v_dec_idx])
        v_retake = np.sum(retake_mask[v_idx])
        view_summary.append({
            "시점": vn,
            "전체건수": len(v_idx),
            "TP": v_tp, "TN": v_tn,
            "FP(과잉경보)": v_fp, "FN(놓침)": v_fn,
            "Retake": v_retake,
            "결함F1": f"{2*v_tp/(2*v_tp + v_fp + v_fn)*100:.1f}%" if (2*v_tp + v_fp + v_fn) > 0 else "0.0%"
        })
    print("\n[2] 공식 시점별 오분류(FP/FN) 및 보류 분포:")
    print(pd.DataFrame(view_summary).to_string(index=False))

    # 피험자별 오분류 분포
    sub_summary = []
    for s in sorted(list(np.unique(subjects))):
        s_idx = np.where(subjects == s)[0]
        s_dec_idx = np.where(sub_dec == s)[0]
        s_fp = np.sum(fp_mask[s_dec_idx])
        s_fn = np.sum(fn_mask[s_dec_idx])
        s_tp = np.sum(tp_mask[s_dec_idx])
        s_tn = np.sum(tn_mask[s_dec_idx])
        sub_summary.append({
            "피험자": f"피험자{s}",
            "전체건수": len(s_idx),
            "TP": s_tp, "TN": s_tn,
            "FP": s_fp, "FN": s_fn,
            "정확도": f"{(s_tp+s_tn)/(len(s_dec_idx))*100:.1f}%" if len(s_dec_idx) > 0 else "0.0%"
        })
    print("\n[3] 피험자별 오분류 분포 (개인별 체형/관절 유연성 편차 감사):")
    print(pd.DataFrame(sub_summary).to_string(index=False))

    # 주요 피처별 편차 분석: FP(정상인데 결함 판정) vs TN(정상이고 정상 판정)
    print("\n[4] FP(과잉경보)를 유발한 주요 피처 편차 분석 (FP vs TN t-test):")
    fp_features = X_dec[fp_mask]
    tn_features = X_dec[tn_mask]
    diff_records = []
    for j, fname in enumerate(FEATURE_NAMES_17D):
        m_fp = np.mean(fp_features[:, j])
        m_tn = np.mean(tn_features[:, j])
        stat, pval = ttest_ind(fp_features[:, j], tn_features[:, j], equal_var=False)
        diff_records.append({
            "피처명": fname,
            "TN평균": round(m_tn, 4),
            "FP평균": round(m_fp, 4),
            "차이(FP-TN)": round(m_fp - m_tn, 4),
            "p-value": f"{pval:.2e}"
        })
    df_fp_diff = pd.DataFrame(diff_records).sort_values(by="차이(FP-TN)", key=abs, ascending=False)
    print(df_fp_diff.head(10).to_string(index=False))

    # 주요 피처별 편차 분석: FN(결함인데 정상 판정) vs TP(결함이고 결함 판정)
    print("\n[5] FN(놓침)을 유발한 주요 피처 편차 분석 (FN vs TP t-test):")
    fn_features = X_dec[fn_mask]
    tp_features = X_dec[tp_mask]
    fn_records = []
    for j, fname in enumerate(FEATURE_NAMES_17D):
        m_fn = np.mean(fn_features[:, j])
        m_tp = np.mean(tp_features[:, j])
        stat, pval = ttest_ind(fn_features[:, j], tp_features[:, j], equal_var=False)
        fn_records.append({
            "피처명": fname,
            "TP평균": round(m_tp, 4),
            "FN평균": round(m_fn, 4),
            "차이(FN-TP)": round(m_fn - m_tp, 4),
            "p-value": f"{pval:.2e}"
        })
    df_fn_diff = pd.DataFrame(fn_records).sort_values(by="차이(FN-TP)", key=abs, ascending=False)
    print(df_fn_diff.head(10).to_string(index=False))

    # 대표적인 오분류 샘플 10개 목록
    print("\n[6] 대표 False Positive(FP) 클립 목록 Top 5:")
    fp_clips = clip_dec[fp_mask]
    fp_probs = prob_dec[fp_mask]
    fp_views = view_dec[fp_mask]
    for i in range(min(5, len(fp_clips))):
        print(f"  - {fp_clips[i]} ({fp_views[i]}): 결함확률 {fp_probs[i]:.4f} (실제 정답: 정상)")

    print("\n[7] 대표 False Negative(FN) 클립 목록 Top 5:")
    fn_clips = clip_dec[fn_mask]
    fn_probs = prob_dec[fn_mask]
    fn_views = view_dec[fn_mask]
    for i in range(min(5, len(fn_clips))):
        print(f"  - {fn_clips[i]} ({fn_views[i]}): 결함확률 {fn_probs[i]:.4f} (실제 정답: 결함)")

if __name__ == "__main__":
    audit_v3_errors()
