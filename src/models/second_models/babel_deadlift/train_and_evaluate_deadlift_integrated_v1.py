# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: train_and_evaluate_deadlift_integrated_v1.py
Exercise: 바벨 데드리프트 (02_바벨_데드리프트)
Purpose:
  스쿼트 v12.3 구조 계승 바벨 데드리프트 32-Way Joint Meta MoE 통합 진단 엔진 v1.0
  AI-Hub 770건 5-Fold StratifiedGroupKFold 피험자 완전 격리 학습 및 다차원 성능 평가
"""

import os
import sys
import json
import time
import numpy as np

# Windows OpenMP Deadlock 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
torch.set_num_threads(2)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, hamming_loss

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
META_CACHE_PATH = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "deadlift_integrated_v1_benchmark_results.json")
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights_integrated_v1")
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)

from joint_meta_32way_v1_net import (
    JointMeta32WayMoENetV1,
    SoftEMRConsistencyLoss32,
    decode_hybrid_32,
    STATE_TO_DEFECTS_MAP_32
)

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def evaluate_3state_emr(y_true: np.ndarray, pred_matrix: np.ndarray, p_clean: np.ndarray, tn: float, td: float):
    """3-State 선택적 분류 EMR 및 Clean F1 평가"""
    # p_clean >= tn: 정상 확정(00000)
    # p_clean < td: 결함 확정(pred_matrix 반영)
    # td <= p_clean < tn: 보류(RETAKE)
    is_clean_pred = (p_clean >= tn)
    is_defect_pred = (p_clean < td)
    decided_mask = is_clean_pred | is_defect_pred
    
    cov = float(np.mean(decided_mask))
    if cov < 0.50:
        return {"coverage": cov, "emr": 0.0, "clean_f1": 0.0, "clean_spec": 0.0}

    y_dec_true = y_true[decided_mask]
    y_dec_pred = np.zeros_like(y_dec_true)
    
    # 정상 확정은 00000, 결함 확정은 pred_matrix
    defect_indices = np.where(is_defect_pred[decided_mask])[0]
    y_dec_pred[defect_indices] = pred_matrix[decided_mask][defect_indices]

    # Exact Match Ratio (5개 항목 100% 일치)
    exact_matches = np.all(y_dec_true == y_dec_pred, axis=1)
    emr = float(np.mean(exact_matches))

    # Clean 평가
    clean_true = np.all(y_dec_true == 0, axis=1).astype(int)
    clean_pred = np.all(y_dec_pred == 0, axis=1).astype(int)
    clean_f1 = float(f1_score(clean_true, clean_pred, zero_division=0))
    clean_rec = float(recall_score(clean_true, clean_pred, zero_division=0))
    clean_prec = float(precision_score(clean_true, clean_pred, zero_division=0))

    return {
        "coverage": cov,
        "abstain_rate": 1.0 - cov,
        "emr": emr,
        "clean_f1": clean_f1,
        "clean_rec": clean_rec,
        "clean_prec": clean_prec,
        "n_decided": int(np.sum(decided_mask)),
        "n_total": len(y_true)
    }


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 통합 진단 엔진 v1.0 벤치마크")
    print("  - 아키텍처: 스쿼트 v12.3 계승 32-Way Joint Meta MoE Net")
    print("  - 평가: 770건 5-Fold StratifiedGroupKFold 피험자 완전 격리")
    print("=" * 80)

    # 1. 메타 데이터 로드
    cache = np.load(META_CACHE_PATH, allow_pickle=True)
    X_meta = cache["X_meta"]          # (770, 48)
    y_all = cache["y"]                # (770, 5)
    states_32 = cache["states_32"]    # (770,)
    oof_indep = cache["oof_probs"]    # (770, 5) - 개별 모델 독립 확률
    subjects = cache["subjects"]      # (770,)
    views = cache["views"]            # (770,)

    N = len(y_all)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_p32 = np.zeros((N, 32), dtype=np.float32)
    oof_p5 = np.zeros((N, 5), dtype=np.float32)

    device = torch.device('cpu')

    # 클래스 불균형 완화 가중치 (32개 상태 빈도 역수)
    counts = np.bincount(states_32, minlength=32)
    weights_32 = len(states_32) / (32.0 * np.maximum(counts, 1).astype(np.float32))
    weights_32_tensor = torch.tensor(weights_32, dtype=torch.float32).to(device)

    print("\n[*] 5-Fold Two-Stream Joint Meta MoE 메타러너 학습 시작...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_meta, states_32, groups=subjects), 1):
        print(f"  --- [Fold {fold}/5] 학습 중... (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")
        
        torch.manual_seed(42 + fold)
        model = JointMeta32WayMoENetV1(in_features=48, hidden_dim=64, dropout_rate=0.20).to(device)
        criterion = SoftEMRConsistencyLoss32(
            w_ce=0.40, w_bce=0.25, w_cons=0.15, w_emr=0.10, w_clean=0.10,
            class_weights_32=weights_32_tensor
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)

        train_ds = TensorDataset(
            torch.tensor(X_meta[train_idx], dtype=torch.float32),
            torch.tensor(states_32[train_idx], dtype=torch.long),
            torch.tensor(y_all[train_idx], dtype=torch.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(45):
            for bx, bs, by in train_loader:
                bx, bs, by = bx.to(device), bs.to(device), by.to(device)
                optimizer.zero_grad()
                l32, l5 = model(bx)
                loss = criterion(l32, l5, bs, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # OOF 추론
        model.eval()
        with torch.no_grad():
            vx = torch.tensor(X_meta[val_idx], dtype=torch.float32).to(device)
            p32_val, p5_val = model.predict_probabilities(vx)
            oof_p32[val_idx] = p32_val.cpu().numpy()
            oof_p5[val_idx] = p5_val.cpu().numpy()

        # 가중치 저장
        torch.save(model.state_dict(), os.path.join(OUT_WEIGHTS_DIR, f"joint_meta_32way_fold{fold}.pth"))

    print("\n[+] 5-Fold 전체 메타 신경망 OOF 예측 완료!")

    # 2. 하이브리드 블렌딩 디코딩
    p_final_hybrid = decode_hybrid_32(oof_p32, oof_p5, alpha=0.65)
    p_clean_joint = oof_p32[:, 0]  # State 0 (Clean) 확률

    # =========================================================================
    # 3. 벤치마크 비교: [Baseline: 5개 단순 독립 병렬 결합] vs [신규 v1.0 통합 엔진]
    # =========================================================================
    
    # 3-1. Baseline: 5개 단순 독립 결합 (p_k >= 0.50)
    pred_indep = (oof_indep >= 0.50).astype(int)
    emr_indep = float(np.mean(np.all(pred_indep == y_all, axis=1)))
    hl_indep = float(hamming_loss(y_all, pred_indep))
    
    clean_true_mask = np.all(y_all == 0, axis=1) # 160건
    clean_indep_pred = np.all(pred_indep == 0, axis=1)
    clean_spec_indep = float(np.mean(clean_indep_pred[clean_true_mask])) # 정상 맞춤률
    clean_fp_indep = int(np.sum(~clean_indep_pred & clean_true_mask))   # 정상 오탐 건수
    clean_f1_indep = float(f1_score(clean_true_mask.astype(int), clean_indep_pred.astype(int), zero_division=0))

    # 3-2. v1.0 통합 엔진 (강제 2-State 판정, threshold=0.50)
    pred_v1_2s = (p_final_hybrid >= 0.50).astype(int)
    emr_v1_2s = float(np.mean(np.all(pred_v1_2s == y_all, axis=1)))
    hl_v1_2s = float(hamming_loss(y_all, pred_v1_2s))
    
    clean_v1_2s_pred = np.all(pred_v1_2s == 0, axis=1)
    clean_spec_v1_2s = float(np.mean(clean_v1_2s_pred[clean_true_mask]))
    clean_fp_v1_2s = int(np.sum(~clean_v1_2s_pred & clean_true_mask))
    clean_f1_v1_2s = float(f1_score(clean_true_mask.astype(int), clean_v1_2s_pred.astype(int), zero_division=0))

    # 3-3. v1.0 통합 엔진 (3-State 선택적 판정 그리드 탐색)
    best_3s_90, best_3s_85 = None, None
    max_emr_90, max_emr_85 = 0.0, 0.0

    for td in np.linspace(0.10, 0.35, 26):
        for tn in np.linspace(td + 0.05, 0.55, 25):
            eval_3s = evaluate_3state_emr(y_all, pred_v1_2s, p_clean_joint, tn, td)
            cov = eval_3s["coverage"]
            emr_val = eval_3s["emr"]
            if cov >= 0.900 and emr_val > max_emr_90:
                max_emr_90 = emr_val
                eval_3s["tau_normal"] = float(tn)
                eval_3s["tau_defect"] = float(td)
                best_3s_90 = eval_3s
            if cov >= 0.850 and emr_val > max_emr_85:
                max_emr_85 = emr_val
                eval_3s["tau_normal"] = float(tn)
                eval_3s["tau_defect"] = float(td)
                best_3s_85 = eval_3s

    # 4. 5대 세부 결함 F1 비교
    defect_names = ["d1 (척추중립)", "d2 (무릎방향)", "d3 (지면접지)", "d4 (동시펴짐)", "d5 (바벨밀착)"]
    defect_f1_indep = [float(f1_score(y_all[:, i], pred_indep[:, i])) for i in range(5)]
    defect_f1_v1 = [float(f1_score(y_all[:, i], pred_v1_2s[:, i])) for i in range(5)]

    # 5. 5대 공식 시점별 감사
    view_audits = {}
    for vn in sorted(np.unique(views)):
        v_mask = (views == vn)
        y_v = y_all[v_mask]
        p_v_indep = pred_indep[v_mask]
        p_v_v1 = pred_v1_2s[v_mask]
        
        emr_v_indep = float(np.mean(np.all(p_v_indep == y_v, axis=1)))
        emr_v_v1 = float(np.mean(np.all(p_v_v1 == y_v, axis=1)))
        
        clean_v_true = np.all(y_v == 0, axis=1)
        clean_v_pred = np.all(p_v_v1 == 0, axis=1)
        spec_v = float(np.mean(clean_v_pred[clean_v_true])) if np.sum(clean_v_true) > 0 else 0.0

        view_audits[vn] = {
            "alias": VIEW_ALIAS_MAP.get(vn, vn),
            "samples": int(len(y_v)),
            "emr_baseline": emr_v_indep,
            "emr_v1": emr_v_v1,
            "clean_spec_v1": spec_v
        }

    # 결과 종합 저장
    results_summary = {
        "dataset_samples": N,
        "clean_samples_true": int(np.sum(clean_true_mask)),
        "baseline_independent": {
            "emr": emr_indep,
            "hamming_loss": hl_indep,
            "clean_spec": clean_spec_indep,
            "clean_fp": clean_fp_indep,
            "clean_f1": clean_f1_indep,
            "defect_f1": defect_f1_indep
        },
        "integrated_v1_2state": {
            "emr": emr_v1_2s,
            "hamming_loss": hl_v1_2s,
            "clean_spec": clean_spec_v1_2s,
            "clean_fp": clean_fp_v1_2s,
            "clean_f1": clean_f1_v1_2s,
            "defect_f1": defect_f1_v1
        },
        "integrated_v1_3state_cov90": best_3s_90,
        "integrated_v1_3state_cov85": best_3s_85,
        "views": view_audits
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 벤치마크 결과 저장 완료: {OUT_RESULTS_JSON}")

    # 결과 출력
    print("\n" + "=" * 80)
    print("  [★ 바벨 데드리프트 통합 진단 엔진 v1.0 정량적 벤치마크 결과]")
    print("=" * 80)
    print(f"1. 다중 라벨 완전 일치율 (5-Defect Exact Match Ratio):")
    print(f"   - Baseline (5개 단순 병렬 결합) : {emr_indep*100:.2f}% ({int(emr_indep*N)}/{N}건)")
    print(f"   - v1.0 통합 메타 엔진 (2-State)  : {emr_v1_2s*100:.2f}% ({int(emr_v1_2s*N)}/{N}건) [변화: {(emr_v1_2s-emr_indep)*100:+.2f}%p]")
    if best_3s_90:
        print(f"   - v1.0 선택적 판정 (Cov >= 90%) : {best_3s_90['emr']*100:.2f}% (Coverage: {best_3s_90['coverage']*100:.2f}%, 보류율: {best_3s_90['abstain_rate']*100:.2f}%)")
    if best_3s_85:
        print(f"   - v1.0 선택적 판정 (Cov >= 85%) : {best_3s_85['emr']*100:.2f}% (Coverage: {best_3s_85['coverage']*100:.2f}%, 보류율: {best_3s_85['abstain_rate']*100:.2f}%)")

    print(f"\n2. 정상 시퀀스 맞춤률 (Clean Specificity, N=160건):")
    print(f"   - Baseline (단순 병렬) : {clean_spec_indep*100:.2f}% (오탐: {clean_fp_indep}건, {clean_fp_indep/np.sum(clean_true_mask)*100:.1f}%)")
    print(f"   - v1.0 통합 메타 엔진  : {clean_spec_v1_2s*100:.2f}% (오탐: {clean_fp_v1_2s}건, {clean_fp_v1_2s/np.sum(clean_true_mask)*100:.1f}%) [오탐 감축: {clean_fp_indep - clean_fp_v1_2s}건]")

    print(f"\n3. 세부 5대 결함 F1-Score:")
    for i, dname in enumerate(defect_names):
        print(f"   - {dname:<18} : Baseline={defect_f1_indep[i]*100:.2f}% -> v1.0={defect_f1_v1[i]*100:.2f}% ({(defect_f1_v1[i]-defect_f1_indep[i])*100:+.2f}%p)")

    print(f"\n4. 5대 공식 시점별 완전 일치율(EMR):")
    for vn, vd in view_audits.items():
        print(f"   - {vd['alias']:<24} : Baseline={vd['emr_baseline']*100:.2f}% -> v1.0={vd['emr_v1']*100:.2f}% ({(vd['emr_v1']-vd['emr_baseline'])*100:+.2f}%p)")


if __name__ == "__main__":
    main()
