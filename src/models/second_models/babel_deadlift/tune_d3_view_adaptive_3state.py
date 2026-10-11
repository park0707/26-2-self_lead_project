# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: tune_d3_view_adaptive_3state.py
Purpose: 바벨 데드리프트 D3 모델의 RTMPose OOF 확률을 산출하고,
         5대 공식 시점(시점 A~E)별 3-State 최적 임계값 튜닝 및 원천 데이터 3-State F1 / 보류율 분석
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb

CURRENT_DIR = r"D:\src\models\second_models\babel_deadlift"
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.models.second_models.babel_deadlift.d3_model_v5_3_opt import extract_d3_v5_3_opt_features_from_seq

CACHE_META = os.path.join(CURRENT_DIR, "deadlift_5models_oof_cache.npz")
RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\02_바벨_데드리프트"

VIEW_MAP = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)'
}

def halpe26_to_mediapipe33(kps_26: np.ndarray) -> np.ndarray:
    T = kps_26.shape[0]
    mp33 = np.zeros((T, 33, 3), dtype=np.float32)
    mapping = {
        0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
        5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
        20: 31, 21: 32, 24: 29, 25: 30
    }
    for h_idx, m_idx in mapping.items():
        mp33[:, m_idx] = kps_26[:, h_idx]
    return mp33

def main():
    print("[*] 메타데이터 로드 중...")
    meta = np.load(CACHE_META, allow_pickle=True)
    clip_names = meta['clip_names']
    subjects = meta['subjects']
    views = meta['views']
    y_golden = meta['y']              # (770, 5)
    oof_mp_all = meta['oof_probs']    # (770, 5)
    N = len(clip_names)

    y_d3 = y_golden[:, 2] # d3 index
    oof_mp_d3 = oof_mp_all[:, 2]

    print(f"[*] 총 시퀀스 {N}건에서 D3 RTMPose 17D 피처 추출 시작...")
    feats_d3 = []
    t0 = time.time()
    for cname in clip_names:
        rtm_p = os.path.join(RTM_DIR, cname + ".npz")
        data = np.load(rtm_p)
        kps_26 = data['kps_26_64f']
        kps_33 = halpe26_to_mediapipe33(kps_26)
        f3 = extract_d3_v5_3_opt_features_from_seq(kps_33)
        feats_d3.append(f3)
    feats_d3 = np.nan_to_num(np.array(feats_d3, dtype=np.float32), nan=0.0)
    print(f"[+] 피처 추출 완료: shape={feats_d3.shape} ({time.time()-t0:.2f}초)")

    # 5-Fold 피험자 격리 학습
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(feats_d3, y_d3, groups=subjects))

    oof_rtm_d3 = np.zeros(N, dtype=np.float32)
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        X_trn, y_trn = feats_d3[trn_idx], y_d3[trn_idx]
        X_val = feats_d3[val_idx]

        cb = CatBoostClassifier(
            iterations=140, depth=5, learning_rate=0.05,
            auto_class_weights='Balanced', random_seed=42 + fold_idx,
            thread_count=1, allow_writing_files=False, verbose=False
        )
        cb.fit(X_trn, y_trn)
        p_cb = cb.predict_proba(X_val)[:, 1]

        clf = lgb.LGBMClassifier(
            n_estimators=110, max_depth=5, learning_rate=0.05,
            num_leaves=24, class_weight='balanced',
            random_state=42 + fold_idx, verbose=-1, n_jobs=1
        )
        clf.fit(X_trn, y_trn)
        p_lgb = clf.predict_proba(X_val)[:, 1]

        oof_rtm_d3[val_idx] = 0.50 * p_cb + 0.50 * p_lgb

    print("[+] D3 RTMPose OOF 확률 산출 완료.")

    # 저장해두기
    np.savez(
        os.path.join(CURRENT_DIR, "deadlift_d3_rtmpose_oof_cache.npz"),
        oof_rtm_d3=oof_rtm_d3,
        oof_mp_d3=oof_mp_d3,
        y_d3=y_d3,
        views=views,
        subjects=subjects,
        clip_names=clip_names
    )

    # 시점별 3-State 그리드 서치
    print("\n" + "="*80)
    print("  [D3 RTMPose] 5대 시점별 3-State 최적 임계값 그리드 탐색")
    print("="*80)

    # 1. 기존 균등 [0.40, 0.60] 기준선
    def calc_3state(y_t, p, t_n, t_d):
        mask = (p <= t_n) | (p >= t_d)
        n_det = int(np.sum(mask))
        cov = float(n_det / len(y_t) * 100.0) if len(y_t) > 0 else 0.0
        retake = 100.0 - cov
        if n_det > 0:
            yt_det = y_t[mask]
            yp_det = (p[mask] >= 0.50).astype(int)
            f1 = float(f1_score(yt_det, yp_det, zero_division=0) * 100.0)
            prec = float(precision_score(yt_det, yp_det, zero_division=0) * 100.0)
            rec = float(recall_score(yt_det, yp_det, zero_division=0) * 100.0)
            acc = float(accuracy_score(yt_det, yp_det) * 100.0)
        else:
            f1, prec, rec, acc = 0.0, 0.0, 0.0, 0.0
        return {'f1': f1, 'prec': prec, 'rec': rec, 'acc': acc, 'cov': cov, 'retake': retake, 'n_det': n_det}

    # 시점별 탐색
    view_list = ['view5', 'view2', 'view3', 'view4', 'view1']
    
    # 튜닝 전략 1: 보류율 <= 10.0% 하드 제약 하에서 F1 최대화
    # 튜닝 전략 2: 보류율 <= 15.0% 제약 하에서 F1 최대화
    # 튜닝 전략 3: 순수 F1 최대화 (보류율 <= 20%)

    for max_retake, strat_name in [(10.0, "상용 안전 기준 (보류율 <= 10.0%)"), 
                                   (15.0, "균형 최적화 기준 (보류율 <= 15.0%)"), 
                                   (20.0, "고신뢰 F1 극대화 (보류율 <= 20.0%)")]:
        print(f"\n>>> [{strat_name}] 시점별 임계값 튜닝 결과 <<<")
        view_best = {}
        total_det_mask = np.zeros(N, dtype=bool)
        
        for v in view_list:
            v_idx = (views == v)
            y_v = y_d3[v_idx]
            p_v = oof_rtm_d3[v_idx]

            best_res = None
            best_f1 = -1.0
            best_pair = (0.5, 0.5)

            # 그리드 탐색
            # t_normal: 0.15 ~ 0.50 (step 0.01)
            # t_defect: 0.50 ~ 0.85 (step 0.01)
            for tn in np.arange(0.15, 0.51, 0.01):
                for td in np.arange(0.50, 0.86, 0.01):
                    if tn > td:
                        continue
                    res = calc_3state(y_v, p_v, tn, td)
                    if res['retake'] <= max_retake:
                        if res['f1'] > best_f1:
                            best_f1 = res['f1']
                            best_res = res
                            best_pair = (round(float(tn), 3), round(float(td), 3))
                        elif res['f1'] == best_f1 and best_res is not None:
                            # 동일 F1이면 보류율이 더 낮은(커버리지 높은) 것 선택
                            if res['retake'] < best_res['retake']:
                                best_res = res
                                best_pair = (round(float(tn), 3), round(float(td), 3))

            if best_res is None:
                # 완화
                best_res = calc_3state(y_v, p_v, 0.40, 0.60)
                best_pair = (0.40, 0.60)

            view_best[v] = {
                'pair': best_pair,
                'metrics': best_res
            }
            # 전체 마스크 누적
            v_det = (p_v <= best_pair[0]) | (p_v >= best_pair[1])
            total_det_mask[v_idx] = v_det

            v_name = VIEW_MAP[v]
            print(f"  - {v_name:<26} ({v}): [tn={best_pair[0]:.2f}, td={best_pair[1]:.2f}] | "
                  f"F1={best_res['f1']:6.2f}% | Prec={best_res['prec']:6.2f}% | Rec={best_res['rec']:6.2f}% | "
                  f"Cov={best_res['cov']:5.1f}% | Retake={best_res['retake']:5.1f}%")

        # 전체 원천 데이터 통합 성능
        yt_all_det = y_d3[total_det_mask]
        yp_all_det = (oof_rtm_d3[total_det_mask] >= 0.50).astype(int)
        all_cov = float(np.sum(total_det_mask) / N * 100.0)
        all_retake = 100.0 - all_cov
        all_f1 = float(f1_score(yt_all_det, yp_all_det, zero_division=0) * 100.0)
        all_prec = float(precision_score(yt_all_det, yp_all_det, zero_division=0) * 100.0)
        all_rec = float(recall_score(yt_all_det, yp_all_det, zero_division=0) * 100.0)
        all_acc = float(accuracy_score(yt_all_det, yp_all_det) * 100.0)

        print(f"  >> [원천 데이터 770건 종합] <<")
        print(f"     3-State F1: {all_f1:.2f}% | Precision: {all_prec:.2f}% | Recall: {all_rec:.2f}% | "
              f"Accuracy: {all_acc:.2f}% | 커버리지: {all_cov:.2f}% | 보류율: {all_retake:.2f}%\n")

if __name__ == "__main__":
    main()
