# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)
Script: tune_advanced_3state_d5_v6.py
Purpose: v6 OOF 확률에 대해 시점별 정밀 임계값 그리드 탐색 및 비대칭 3-State 신뢰도 최적화
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d5_model_v6_evaluation_results.json")
V6_CACHE_PATH = os.path.join(CURRENT_DIR, "d5_features_37_v6_cache.npz")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


def main():
    with open(OUT_RESULTS_JSON, "r", encoding="utf-8") as f:
        v6_eval = json.load(f)

    cached = np.load(V6_CACHE_PATH, allow_pickle=True)
    y = cached['y']
    groups = cached['groups']
    views = cached['views']

    # Fold 모델들로부터 raw OOF 확률 재생성
    # fold 가중치 로드
    from src.models.second_models.babel_deadlift.d5_model_v6 import BabelDeadliftD5ModelV6
    from sklearn.model_selection import StratifiedGroupKFold

    X = cached['X']
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_raw_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        weight_path = os.path.join(CURRENT_DIR, "weights", f"d5_v6_fold{fold+1}.pkl")
        m = BabelDeadliftD5ModelV6.load(weight_path)
        oof_raw_probs[val_idx] = m.predict_raw_proba(X[val_idx])

    print("=" * 80)
    print(" [*] [vidio-health-trainer] d5 v6.0 시점별 정밀 임계값 및 비대칭 3-State 최적화 탐색")
    print("=" * 80)

    # 1. 시점별 정밀 F1 / Balanced Accuracy 그리드 서치
    # 단순 F1만 보면 시점 C에서 극단적 저임계값(0.20)으로 치달아 FP가 폭증하므로,
    # F1과 Accuracy를 균형 있게 고려하는 Harmonic Mean 또는 F1 기준 최적점 탐색
    tuned_thresholds = {}
    print("\n[1] 5대 시점별 정밀 임계값 스위핑:")
    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        alias = VIEW_ALIAS_MAP[vn]
        mask_v = (views == vn)
        y_v = y[mask_v]
        p_v = oof_raw_probs[mask_v]

        best_score = -1.0
        best_t = 0.50
        best_metrics = {}

        # 0.25부터 0.65까지 0.01 단위로 정밀 스위핑
        for t in np.arange(0.25, 0.65, 0.01):
            pred_t = (p_v >= t).astype(int)
            f1_t = f1_score(y_v, pred_t, zero_division=0)
            acc_t = accuracy_score(y_v, pred_t)
            prec_t = precision_score(y_v, pred_t, zero_division=0)
            rec_t = recall_score(y_v, pred_t, zero_division=0)

            # 정면 C의 경우 정밀도가 너무 떨어지지 않도록 페널티 (정밀도 >= 60% 제약)
            if vn == "view3":
                if prec_t < 0.60:
                    score = f1_t * 0.7  # 페널티
                else:
                    score = f1_t + acc_t * 0.5
            else:
                score = f1_t + acc_t * 0.3

            if score > best_score:
                best_score = score
                best_t = round(float(t), 2)
                best_metrics = {
                    "f1": f1_t, "acc": acc_t, "prec": prec_t, "rec": rec_t
                }

        tuned_thresholds[vn] = best_t
        print(f"  > {alias:<22}: 최적 Th={best_t:.2f} | F1={best_metrics['f1']*100:.2f}% | Acc={best_metrics['acc']*100:.2f}% | Prec={best_metrics['prec']*100:.2f}% | Rec={best_metrics['rec']*100:.2f}%")

    # 2. 튜닝된 임계값 적용 1차 성능
    tuned_preds = np.zeros(len(y), dtype=int)
    tuned_cal_probs = np.zeros(len(y), dtype=np.float32)

    for vn in ["view5", "view2", "view3", "view4", "view1"]:
        mask_v = (views == vn)
        th = tuned_thresholds[vn]
        p_v = oof_raw_probs[mask_v]
        p_cal_v = np.where(p_v >= th, 0.50 + 0.50 * (p_v - th) / (1.0 - th + 1e-6), 0.50 * p_v / (th + 1e-6))
        tuned_cal_probs[mask_v] = p_cal_v
        tuned_preds[mask_v] = (p_v >= th).astype(int)

    f1_all = f1_score(y, tuned_preds, zero_division=0)
    acc_all = accuracy_score(y, tuned_preds)
    prec_all = precision_score(y, tuned_preds, zero_division=0)
    rec_all = recall_score(y, tuned_preds, zero_division=0)
    auc_all = roc_auc_score(y, oof_raw_probs)
    cm_all = confusion_matrix(y, tuned_preds).tolist()

    print("\n" + "=" * 80)
    print(" [균형 튜닝 후 770건 전체 OOF 성능 - v6.0]")
    print("=" * 80)
    print(f"[*] 기본 정확도 : {acc_all*100:.2f}%")
    print(f"[*] 정밀도     : {prec_all*100:.2f}%")
    print(f"[*] 재현율     : {rec_all*100:.2f}%")
    print(f"[*] F1-Score   : ★ {f1_all*100:.2f}%")
    print(f"[*] ROC-AUC    : ★ {auc_all:.4f}")
    print(f"[*] 총 오류 수 : {cm_all[0][1] + cm_all[1][0]}건 (FP: {cm_all[0][1]}건, FN: {cm_all[1][0]}건)")

    # 3. 비대칭 3-State 신뢰도 마진 탐색
    # 시점별로 불확실성 마진을 다르게 적용:
    # 정면(시점 C)과 후방(시점 A, E)은 마진을 조금 더 넓히고, 완벽한 시점 D는 마진을 0으로 둠
    print("\n" + "=" * 80)
    print(" [비대칭 3-State 신뢰도 게이팅 최적화]")
    print("=" * 80)

    best_3s_result = None
    best_3s_f1 = -1.0

    for base_m in [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        decided_mask = np.zeros(len(y), dtype=bool)

        for vn in ["view5", "view2", "view3", "view4", "view1"]:
            mask_v = (views == vn)
            cal_p_v = tuned_cal_probs[mask_v]

            # 시점별 맞춤 마진
            if vn == "view4":
                m_v = 0.02  # 시점 D는 이미 96%+ 이므로 거의 보류 없음
            elif vn == "view3":
                m_v = base_m + 0.02  # 정면은 불확실성 조금 더 적극 보류
            elif vn in ["view5", "view1"]:
                m_v = base_m + 0.01  # 후방
            else:
                m_v = base_m  # 시점 B

            l_th = 0.50 - m_v
            h_th = 0.50 + m_v
            dec_v = (cal_p_v < l_th) | (cal_p_v >= h_th)
            decided_mask[mask_v] = dec_v

        cov = float(np.mean(decided_mask))
        y_dec = y[decided_mask]
        p_dec = (tuned_cal_probs[decided_mask] >= 0.50).astype(int)

        f1_3s = f1_score(y_dec, p_dec, zero_division=0)
        acc_3s = accuracy_score(y_dec, p_dec)
        prec_3s = precision_score(y_dec, p_dec, zero_division=0)
        rec_3s = recall_score(y_dec, p_dec, zero_division=0)

        print(f"  기준마진 {base_m:.2f} | 커버리지: {cov*100:5.2f}% ({sum(decided_mask)}/770) | 3-State F1: ★ {f1_3s*100:5.2f}% | Acc: {acc_3s*100:5.2f}%")

        if cov >= 0.90 and f1_3s > best_3s_f1:
            best_3s_f1 = f1_3s
            best_3s_result = {
                "base_margin": base_m,
                "coverage": cov,
                "decided_count": int(sum(decided_mask)),
                "f1_score": float(f1_3s),
                "accuracy": float(acc_3s),
                "precision": float(prec_3s),
                "recall": float(rec_3s)
            }

    print("\n" + "=" * 80)
    print(f"[*] 최적 3-State (커버리지 90%+ 제약): F1={best_3s_result['f1_score']*100:.2f}%, Acc={best_3s_result['accuracy']*100:.2f}%, 커버리지={best_3s_result['coverage']*100:.2f}%")
    print("=" * 80)

    # 튜닝된 결과 업데이트 저장
    v6_eval["tuned_thresholds"] = tuned_thresholds
    v6_eval["tuned_metrics"] = {
        "accuracy": float(acc_all),
        "precision": float(prec_all),
        "recall": float(rec_all),
        "f1_score": float(f1_all),
        "confusion_matrix": cm_all,
        "total_errors": int(cm_all[0][1] + cm_all[1][0])
    }
    v6_eval["optimal_3state"] = best_3s_result

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(v6_eval, f, indent=2, ensure_ascii=False)
    print(f"[+] 튜닝 결과 저장 완료: {OUT_RESULTS_JSON}")


if __name__ == "__main__":
    main()
