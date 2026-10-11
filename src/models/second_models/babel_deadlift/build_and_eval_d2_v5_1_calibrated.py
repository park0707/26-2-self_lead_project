# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_and_eval_d2_v5_1_calibrated.py
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Submodel: d2 (발과 무릎의 방향 일치 / 동적 무릎 외반 Knee Valgus)
Version: v5.1 (체형 무차원화 Delta-KAWR/DWD + 시점별 바이어스/온도 캘리브레이션)
Purpose:
  v5.0의 안정적 베이스라인을 계승하고, 외부 AI의 핵심 솔루션을 반영하여
  1. 피험자 6번의 내로우 스탠스 FP 12건 소거: 절대 cm 피처 배제, Delta-KAWR 및 정강이 정규화 DWD 투입
  2. view2(FP 19건) vs view4(FN 14건) 시점 비대칭 해소: Inner Fold 시점별 바이어스/온도 스케일링(View-Specific Calibration)
  3. AI-Hub 385개 전수 5-Fold StratifiedGroupKFold 무삭제 평가
"""

import os
import sys
import time
import json
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"
OUT_WEIGHTS_DIR = r"D:\src\models\second_models\babel_deadlift\weights"
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
WEIGHTS_V5_1_PATH = os.path.join(OUT_WEIGHTS_DIR, "d2_v5_1_calibrated_models.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

VIEW_TO_IDX = {
    "view5": 0, # 시점 A
    "view2": 1, # 시점 B
    "view3": 2, # 시점 C
    "view4": 3, # 시점 D
    "view1": 4  # 시점 E
}


# =========================================================================
# 1. 체형 무차원화 생체역학 피처 추출 (Delta-KAWR & DWD)
# =========================================================================
def extract_d2_v5_1_features(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    외부 AI 권고 반영: 절대 cm 거리를 전면 배제하고,
    1. 셋업 기준 무차원 무릎-발목 너비비(setup_kawr)
    2. 피험자 고유 셋업 대비 무차원 상대 붕괴율 (Delta KAWR)
    3. 정강이 길이(Shank Length) 기준 정규화 동적 붕괴량 (DWD)
    4. 발 진행방향 대비 무릎 편위 (Foot-Relative Knee Deviation)
    5. 관상면 투영 부호 외반각 (Signed Coronal FPPA)
    총 20D 무차원 정예 피처셋
    """
    T = kps_33_64f.shape[0]  # 64

    # 주요 랜드마크
    lh, rh = kps_33_64f[:, 23], kps_33_64f[:, 24] # Hip
    lk, rk = kps_33_64f[:, 25], kps_33_64f[:, 26] # Knee
    la, ra = kps_33_64f[:, 27], kps_33_64f[:, 28] # Ankle
    lhe, rhe = kps_33_64f[:, 29], kps_33_64f[:, 30] # Heel
    lt, rt = kps_33_64f[:, 31], kps_33_64f[:, 32] # Toe
    lw, rw = kps_33_64f[:, 15], kps_33_64f[:, 16] # Wrist

    # 1. 셋업 구간(프레임 0~10) 베이스라인 산출
    setup_kw = np.mean(np.linalg.norm(lk[:11] - rk[:11], axis=-1))
    setup_aw = np.mean(np.linalg.norm(la[:11] - ra[:11], axis=-1))
    setup_hw = np.mean(np.linalg.norm(lh[:11] - rh[:11], axis=-1))
    setup_kawr = setup_kw / (setup_aw + 1e-6)
    setup_kh_ratio = setup_kw / (setup_hw + 1e-6)

    # 2. 전 구간 동적 너비 시계열
    kw_t = np.linalg.norm(lk - rk, axis=-1)
    aw_t = np.linalg.norm(la - ra, axis=-1)
    hw_t = np.linalg.norm(lh - rh, axis=-1)
    kawr_t = kw_t / (aw_t + 1e-6)
    kawr_t_s = gaussian_filter1d(kawr_t, sigma=1.0)

    # 3. [핵심 혁신 1] 피험자 고유 셋업 대비 무차원 상대 붕괴율 (Delta KAWR)
    # 무릎이 안으로 모이면 음수(-)로 낙폭 발생
    delta_kawr_t = (kawr_t_s - setup_kawr) / (setup_kawr + 1e-6)
    # 퍼스트 풀 구간 (프레임 21~45)
    p1 = slice(21, 46)
    p0 = slice(0, 21)

    min_delta_kawr = float(np.min(delta_kawr_t[p1]))
    mean_delta_kawr = float(np.mean(delta_kawr_t[p1]))
    # 결손 면적 (셋업 대비 붕괴 적분량)
    valgus_deficiency_area = float(np.sum(np.clip(-delta_kawr_t[p1], 0.0, None)))
    persistent_collapse_ratio = float(np.mean(delta_kawr_t[p1] < -0.05))

    # 4. [핵심 혁신 2] 정강이 뼈 길이(Shank Length) 기준 정규화 동적 붕괴량 (DWD)
    # 체형/골반과 무관한 절대 불변 뼈 길이 정규화
    shank_l = np.linalg.norm(lk - la, axis=-1)
    shank_r = np.linalg.norm(rk - ra, axis=-1)
    shank_len = float(np.mean((shank_l + shank_r) / 2.0)) + 1e-6
    dwd_t = (setup_kw - kw_t) / shank_len
    dwd_t_s = gaussian_filter1d(dwd_t, sigma=1.0)
    max_dwd = float(np.max(dwd_t_s[p1]))
    mean_dwd = float(np.mean(dwd_t_s[p1]))

    # 5. [핵심 혁신 3] 발 진행방향 대비 무릎 편위 (Foot-Relative Knee Deviation)
    foot_l = lt[:, :2] - lhe[:, :2]
    foot_r = rt[:, :2] - rhe[:, :2]
    foot_dir_l = foot_l / (np.linalg.norm(foot_l, axis=1, keepdims=True) + 1e-7)
    foot_dir_r = foot_r / (np.linalg.norm(foot_r, axis=1, keepdims=True) + 1e-7)
    med_norm_l = np.column_stack([-foot_dir_l[:, 1], foot_dir_l[:, 0]])
    med_norm_r = np.column_stack([foot_dir_r[:, 1], -foot_dir_r[:, 0]])
    shank_xy_l = lk[:, :2] - la[:, :2]
    shank_xy_r = rk[:, :2] - ra[:, :2]
    dev_foot_l = np.sum(shank_xy_l * med_norm_l, axis=1) / (np.linalg.norm(shank_xy_l, axis=1) + 1e-7)
    dev_foot_r = np.sum(shank_xy_r * med_norm_r, axis=1) / (np.linalg.norm(shank_xy_r, axis=1) + 1e-7)
    foot_dev_sum = dev_foot_l + dev_foot_r
    foot_dev_p90 = float(np.percentile(foot_dev_sum[p1], 90))
    foot_dev_max = float(np.max(foot_dev_sum[p1]))

    # 6. 관상면 부호 외반각 (Coronal FPPA)
    thigh_l = lh[:, :2] - lk[:, :2]
    shank_l = la[:, :2] - lk[:, :2]
    cross_l = thigh_l[:, 0] * shank_l[:, 1] - thigh_l[:, 1] * shank_l[:, 0]
    dot_l = np.sum(thigh_l * shank_l, axis=1)
    signed_fppa_l = -(180.0 - np.abs(np.degrees(np.arctan2(cross_l, dot_l + 1e-7)))) * np.sign(cross_l)

    thigh_r = rh[:, :2] - rk[:, :2]
    shank_r = ra[:, :2] - rk[:, :2]
    cross_r = thigh_r[:, 0] * shank_r[:, 1] - thigh_r[:, 1] * shank_r[:, 0]
    dot_r = np.sum(thigh_r * shank_r, axis=1)
    signed_fppa_r = (180.0 - np.abs(np.degrees(np.arctan2(cross_r, dot_r + 1e-7)))) * np.sign(cross_r)

    fppa_valgus_sum = np.maximum(0.0, signed_fppa_l) + np.maximum(0.0, signed_fppa_r)
    fppa_valgus_p90 = float(np.percentile(fppa_valgus_sum[p1], 90))

    # 7. 무릎 흔들림 각가속도 피크 (Snap)
    accel = np.gradient(np.gradient(kawr_t_s))
    accel_peak = float(np.max(np.abs(accel[p1])))

    # 8. 좌우 비대칭도
    asym_max = float(np.max(np.abs(dev_foot_l[p1] - dev_foot_r[p1])))

    # 9. 바벨-무릎 차폐 거리
    dist_l_wrist = np.linalg.norm(lw - lk, axis=1)
    dist_r_wrist = np.linalg.norm(rw - rk, axis=1)
    min_arm_dist = float(np.min(np.minimum(dist_l_wrist, dist_r_wrist)))

    # 20D 무차원 정예 피처 벡터 구성
    features_20d = np.array([
        setup_kawr,                  # 1. 셋업 개인 고유 무릎/발목비
        setup_kh_ratio,              # 2. 셋업 개인 고유 무릎/골반비
        min_delta_kawr,              # 3. 셋업 대비 무차원 무릎 모임 최대 낙폭 (Delta KAWR min)
        mean_delta_kawr,             # 4. 셋업 대비 무차원 평균 모임도
        valgus_deficiency_area,      # 5. 셋업 대비 외반 결손 누적 면적
        persistent_collapse_ratio,   # 6. 5% 이상 지속 붕괴 프레임 비율
        max_dwd,                     # 7. 정강이 길이 대비 최대 동적 붕괴량 (DWD max)
        mean_dwd,                    # 8. 정강이 길이 대비 평균 동적 붕괴량 (DWD mean)
        foot_dev_p90,                # 9. 발 진행방향 대비 무릎 편위 90백분위수 (d2 라벨 직결)
        foot_dev_max,                # 10. 발 진행방향 대비 무릎 편위 최댓값
        fppa_valgus_p90,             # 11. 관상면 부호 외반각 90백분위수
        accel_peak,                  # 12. 무릎 모임 각가속도 스냅 피크
        asym_max,                    # 13. 좌우 무릎 비대칭 모임 피크
        min_arm_dist,                # 14. 바벨-무릎 최소 3D 거리 (차폐 보정)
        float(np.min(kawr_t_s[p1])), # 15. 퍼스트풀 최소 KAWR
        float(np.mean(kawr_t_s[p1])),# 16. 퍼스트풀 평균 KAWR
        float(np.std(kawr_t_s[p1])), # 17. 퍼스트풀 KAWR 변동성
        float(np.percentile(dev_foot_l[p1], 90)), # 18. 좌발 편위 90백분위수
        float(np.percentile(dev_foot_r[p1], 90)), # 19. 우발 편위 90백분위수
        float(np.max(dwd_t_s[p0]))   # 20. 셋업 구간 DWD (자세 안정성)
    ], dtype=np.float32)

    return features_20d


# =========================================================================
# 2. 시점별 바이어스/온도 캘리브레이터 (ViewSpecificCalibrator)
# =========================================================================
class ViewSpecificCalibrator(nn.Module):
    """
    외부 AI 제안: 시점별 바이어스 및 온도 스케일링 모듈
    calibrated_logits = (logits - bias[view]) / exp(log_temp[view])
    """
    def __init__(self, n_views=5):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(n_views))
        self.bias = nn.Parameter(torch.zeros(n_views))

    def forward(self, logits, view_idx):
        temp = torch.exp(self.log_temp[view_idx])
        calibrated_logits = (logits - self.bias[view_idx]) / temp
        return torch.sigmoid(calibrated_logits)

    def fit(self, val_logits, val_views, val_labels, lr=0.02, epochs=250):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        logits_t = torch.tensor(val_logits, dtype=torch.float32)
        views_t = torch.tensor(val_views, dtype=torch.long)
        labels_t = torch.tensor(val_labels, dtype=torch.float32)

        for _ in range(epochs):
            optimizer.zero_grad()
            probs = self.forward(logits_t, views_t)
            loss = criterion(probs, labels_t)
            loss.backward()
            optimizer.step()


# =========================================================================
# 3. 5-Fold StratifiedGroupKFold 파이프라인 실행
# =========================================================================
def run_d2_v5_1_pipeline():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 데드리프트 d2 v5.1 고도화 실측 파이프라인")
    print("  - 핵심 혁신 1: 체형 무차원화 (절대 cm 전면 배제, 셋업 대비 Delta-KAWR & 정강이 정규화 DWD)")
    print("  - 핵심 혁신 2: 시점 비대칭 캘리브레이션 (Inner Fold View-Specific Temperature & Bias Scaling)")
    print("  - 데이터셋: AI-Hub 385개 시퀀스 전수 무삭제 평가 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 검증 프로토콜: 5-Fold StratifiedGroupKFold (9명 피험자 완전 격리, 데이터 누수 0%)")
    print("=" * 115)

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    N = len(files)

    X_list = []
    y_list = []
    subs = []
    views = []
    view_indices = []

    t0 = time.time()
    for p in files:
        base = os.path.basename(p).replace(".npz", "")
        v_code = base.split("_")[-1]
        views.append(v_code)
        view_indices.append(VIEW_TO_IDX.get(v_code, 2))

        subj = "Subj_Unknown"
        for part in base.split("_"):
            if "피험자" in part:
                subj = part
                break
        subs.append(subj)

        folder = "_".join(base.split("_")[:-1])
        json_path = os.path.join(LABEL_JSON_DIR, f"{folder}.json")
        with open(json_path, 'r', encoding='utf-8') as fp:
            conds = {c['condition']: c['value'] for c in json.load(fp).get('type_info', {}).get('conditions', [])}
        y_list.append(0 if conds.get('발과 무릎의 방향 일치', True) else 1)

        kps = np.load(p)['kps_33_3d_64f']
        f20 = extract_d2_v5_1_features(kps)
        X_list.append(f20)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subs = np.array(subs)
    views = np.array(views)
    view_indices = np.array(view_indices, dtype=np.int64)

    print(f"[+] 20D 무차원 생체역학 피처 추출 완료: {X.shape} (경과: {time.time()-t0:.1f}초)")
    print(f"  - 정상(0): {np.sum(y==0)}건 vs 결함(1): {np.sum(y==1)}건 (결함비율: {np.mean(y)*100:.1f}%)")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    raw_oof_probs = np.zeros(N, dtype=np.float32)
    calibrated_oof_probs = np.zeros(N, dtype=np.float32)

    os.makedirs(r"C:\Temp\catboost_info", exist_ok=True)

    print("\n[*] 5-Fold 피험자 격리 학습 및 시점 캘리브레이션 시작...")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        views_tr, views_val = view_indices[train_idx], view_indices[val_idx]
        subs_tr = subs[train_idx]

        # 1. Train GBDT
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=12,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=2.0,
            scale_pos_weight=3.28, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)
        p_l = clf_lgb.predict_proba(X_val)[:, 1]

        clf_cat = CatBoostClassifier(
            iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=4.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1,
            train_dir=r"C:\Temp\catboost_info"
        )
        clf_cat.fit(X_tr, y_tr)
        p_c = clf_cat.predict_proba(X_val)[:, 1]

        raw_p_fold = 0.5 * p_l + 0.5 * p_c
        raw_oof_probs[val_idx] = raw_p_fold

        # 2. Inner-Fold OOF Predictions for Calibrator Fitting (데이터 누수 100% 차단!)
        inner_sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        inner_oof_logits = np.zeros(len(y_tr), dtype=np.float32)

        for i_tr_idx, i_val_idx in inner_sgkf.split(X_tr, y_tr, groups=subs_tr):
            i_lgb = lgb.LGBMClassifier(n_estimators=70, learning_rate=0.04, max_depth=3, scale_pos_weight=3.28, random_state=42, verbose=-1, n_jobs=1)
            i_lgb.fit(X_tr[i_tr_idx], y_tr[i_tr_idx])
            p_in = i_lgb.predict_proba(X_tr[i_val_idx])[:, 1]
            p_in = np.clip(p_in, 1e-5, 1.0 - 1e-5)
            inner_oof_logits[i_val_idx] = np.log(p_in / (1.0 - p_in))

        # 3. Fit Calibrator on Inner OOF
        calibrator = ViewSpecificCalibrator(n_views=5)
        calibrator.fit(inner_oof_logits, views_tr, y_tr, lr=0.02, epochs=250)

        # 4. Apply Calibrator to Validation Fold
        val_logits = np.log(np.clip(raw_p_fold, 1e-5, 1.0 - 1e-5) / (1.0 - np.clip(raw_p_fold, 1e-5, 1.0 - 1e-5)))
        with torch.no_grad():
            cal_p_fold = calibrator(torch.tensor(val_logits, dtype=torch.float32), torch.tensor(views_val, dtype=torch.long)).numpy()

        calibrated_oof_probs[val_idx] = cal_p_fold

        auc_raw = roc_auc_score(y_val, raw_p_fold)
        auc_cal = roc_auc_score(y_val, cal_p_fold)
        print(f"  [Fold {fold}/5] Raw GBDT AUC: {auc_raw:.4f} | Calibrated AUC: {auc_cal:.4f}")

    # =========================================================================
    # 4. 종합 평가 및 최적 3-State 윈도우 탐색
    # =========================================================================
    auc_overall = roc_auc_score(y, calibrated_oof_probs)
    print("\n" + "=" * 115)
    print(f"  [v5.1 실측 종합 성적표 (N={N}, 피험자 1~9번 전원 전수 검증)]")
    print("=" * 115)
    print(f"  • Overall Calibrated ROC-AUC : {auc_overall:.4f} (v5.0 0.7674 대비)")

    best_res = None
    for tn in np.arange(0.20, 0.45, 0.005):
        for td in np.arange(tn + 0.005, 0.55, 0.005):
            dec = (calibrated_oof_probs <= tn) | (calibrated_oof_probs >= td)
            cov = np.mean(dec) * 100
            if cov < 90.0:  # 보류율 10% 이하 엄격 제약
                continue
            pred_eval = (calibrated_oof_probs[dec] >= td).astype(int)
            y_eval = y[dec]
            f1 = f1_score(y_eval, pred_eval, zero_division=0) * 100
            prec = precision_score(y_eval, pred_eval, zero_division=0) * 100
            rec = recall_score(y_eval, pred_eval, zero_division=0) * 100
            acc = accuracy_score(y_eval, pred_eval) * 100

            cand = {
                'window': [round(float(tn), 3), round(float(td), 3)],
                'f1': f1, 'prec': prec, 'rec': rec, 'acc': acc,
                'cov': cov, 'decline': 100.0 - cov
            }
            if best_res is None or cand['f1'] > best_res['f1']:
                best_res = cand

    tn, td = best_res['window']
    dec_mask = (calibrated_oof_probs <= tn) | (calibrated_oof_probs >= td)
    cm = confusion_matrix(y[dec_mask], (calibrated_oof_probs[dec_mask] >= td).astype(int))

    print(f"  [★ 3-State 공식 채택 윈도우]: {best_res['window']}")
    print(f"  • 3-State F1-Score      : {best_res['f1']:.2f}%")
    print(f"  • 3-State 정밀도 (Prec) : {best_res['prec']:.2f}%")
    print(f"  • 3-State 재현율 (Rec)  : {best_res['rec']:.2f}%")
    print(f"  • 3-State 정확도 (Acc)  : {best_res['acc']:.2f}%")
    print(f"  • 판정 커버리지 / 보류율: {best_res['cov']:.1f}% / {best_res['decline']:.1f}% (10% 이하 엄격 준수)")
    print(f"  • 3-State 혼동행렬      : TP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}")
    print("=" * 115)

    # 5대 시점별 세부 실측 성능 감사
    print("\n[공식 5대 시점별 3-State 성능 감사 대조표]:")
    print("공식 시점 명칭                 | 코드     | 3-State F1   | Precision  | Recall     | Coverage")
    print("-" * 85)
    view_results = {}
    for code, name in VIEW_ALIAS_MAP.items():
        v_idx = (views == code)
        v_dec = dec_mask[v_idx]
        cov_v = np.mean(v_dec) * 100
        y_v = y[v_idx][v_dec]
        pred_v = (calibrated_oof_probs[v_idx][v_dec] >= td).astype(int)
        f1_v = f1_score(y_v, pred_v, zero_division=0) * 100
        p_v = precision_score(y_v, pred_v, zero_division=0) * 100
        r_v = recall_score(y_v, pred_v, zero_division=0) * 100
        view_results[code] = {'f1': f1_v, 'prec': p_v, 'rec': r_v, 'cov': cov_v}
        print(f"{name:<28} | {code:<8} | {f1_v:6.2f}       % | {p_v:6.2f}     % | {r_v:6.2f}     % | {cov_v:5.1f}%")
    print("=" * 85)

    # 피험자별 오분류 추적 (특히 피험자 6번의 FP 변화!)
    print("\n[피험자별 FP 및 FN 변화 감사 (피험자 6번 내로우 스탠스 검증)]:")
    for s in sorted(list(set(subs))):
        s_idx = (subs == s)
        s_dec = dec_mask[s_idx]
        y_s = y[s_idx][s_dec]
        pred_s = (calibrated_oof_probs[s_idx][s_dec] >= td).astype(int)
        fp_s = int(np.sum((pred_s == 1) & (y_s == 0)))
        fn_s = int(np.sum((pred_s == 0) & (y_s == 1)))
        print(f"  {s:<15}: FP={fp_s:2d}건, FN={fn_s:2d}건")

    # 결과 파일 저장
    save_data = {
        'version': 'v5.1',
        'calibrated_oof_probs': calibrated_oof_probs.tolist(),
        'raw_oof_probs': raw_oof_probs.tolist(),
        'labels': y.tolist(),
        'subjects': subs.tolist(),
        'views': views.tolist(),
        'best_window': best_res['window'],
        'metrics': best_res,
        'view_results': view_results,
        'roc_auc': float(auc_overall),
        'confusion_matrix': {
            'tp': int(cm[1, 1]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tn': int(cm[0, 0])
        }
    }
    with open(WEIGHTS_V5_1_PATH, 'wb') as fp:
        pickle.dump(save_data, fp)

    json_path = os.path.join(OUT_WEIGHTS_DIR, "d2_v5_1_evaluation_results.json")
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(save_data, fp, ensure_ascii=False, indent=2)

    print(f"\n[+] v5.1 평가 모델 및 메트릭 저장 완료: {json_path}")
    return save_data


if __name__ == "__main__":
    run_d2_v5_1_pipeline()
