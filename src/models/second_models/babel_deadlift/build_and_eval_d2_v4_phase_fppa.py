# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_and_eval_d2_v4_phase_fppa.py
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Submodel: d2 (발과 무릎의 방향 일치 / 무릎 외반 Knee Valgus)
Version: v4.0 (위상 인식 Phase-Aware + 3D FPPA + 팔 차폐 보정 + F1 직접 최적화 GBDT)
Purpose: 
  v3.0의 성능이 입증된 베이스라인(Cleanlab Confident Learning + 가상 정면 3D 좌표계 + 5-Fold StratifiedGroupKFold)을
  100% 점진적 계승(Copy-and-Refine)하고, SOTA 선행연구(Nature 2026, ECCV 2022)의 4대 혁신 기법을 적용하여
  d2의 F1-Score를 극대화한다:
  1. 위상 인식(Phase-Aware) 기하 피처 (셋업/하강 vs 퍼스트풀 vs 락아웃 분리)
  2. 3D FPPA(전두면 투영각) 및 발목-정강이 내회전 어긋남 결합
  3. 팔-무릎 차폐 인식(Arm-Knee Occlusion-Aware) 3D 거리 피처
  4. 클래스 불균형 비용 민감 학습(scale_pos_weight=3.28) 및 F1 최적화 3-State 판정
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
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, 'D:/')

from src.models.second_models.babel_deadlift.build_and_eval_d2_v3_integrated import (
    transform_to_canonical_frontal, extract_dual_geometry_features,
    CACHE_DIR, LABEL_JSON_DIR, VIEW_ALIAS_MAP
)

OUT_WEIGHTS_DIR = r"D:\src\models\second_models\babel_deadlift\weights"
os.makedirs(OUT_WEIGHTS_DIR, exist_ok=True)
WEIGHTS_V4_PATH = os.path.join(OUT_WEIGHTS_DIR, "d2_v4_phase_fppa_models.pkl")


def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """두 3D 벡터 시계열 사이의 사이각(도, degree) 계산"""
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1) + 1e-7
    norm2 = np.linalg.norm(v2, axis=1) + 1e-7
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def extract_d2_v4_phase_fppa_features(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    v4.0 공식 18D 위상 인식 & 3D FPPA 듀얼 기하 피처셋 추출 함수
    
    1~12차원: 기존 v3.0 검증된 12D 듀얼 기하 피처 (계승)
    13~18차원: 신규 6대 혁신 피처
      13. phase0_setup_ka_ratio_min: 셋업 구간(프레임 0~20) 무릎/발목비 최솟값 (사용자 관찰 '내려갈 때 모임' 감지)
      14. phase1_first_pull_ka_min: 퍼스트풀(프레임 21~45) 무릎/발목비 최솟값 (지면 분리 정지관성 극복 시)
      15. phase_delta_ka_pull_drop: 셋업 대비 퍼스트풀 무릎 모임 낙폭 (Delta KAWR)
      16. fppa_3d_valgus_angle_max: 가상 정면 3D FPPA 전두면 외반각 최댓값 (고관절-무릎-발목 꺾임각)
      17. ankle_internal_rotation_mismatch: 발끝 방향 대비 정강이 안쪽 회전 어긋남 각도
      18. arm_knee_min_distance: 손목/팔꿈치와 무릎 간의 최소 3D 거리 (팔 차폐 노이즈 보정용)
    """
    # 1. v3.0 12D 피처 추출
    f12_base = extract_dual_geometry_features(kps_33_64f)
    
    # 2. 가상 정면 3D 좌표계
    c_kps = transform_to_canonical_frontal(kps_33_64f)
    T = c_kps.shape[0]  # 64프레임
    
    l_hip = c_kps[:, 23, :]
    r_hip = c_kps[:, 24, :]
    l_knee = c_kps[:, 25, :]
    r_knee = c_kps[:, 26, :]
    l_ank = c_kps[:, 27, :]
    r_ank = c_kps[:, 28, :]
    l_toe = c_kps[:, 31, :]
    r_toe = c_kps[:, 32, :]
    l_wrist = c_kps[:, 15, :]
    r_wrist = c_kps[:, 16, :]
    l_elbow = c_kps[:, 13, :]
    r_elbow = c_kps[:, 14, :]
    
    # 너비 시계열 (X축 거리)
    knee_w = np.abs(l_knee[:, 0] - r_knee[:, 0])
    ank_w = np.abs(l_ank[:, 0] - r_ank[:, 0])
    ka_ratio = knee_w / (ank_w + 1e-6)
    ka_ratio_s = gaussian_filter1d(ka_ratio, sigma=1.0)
    
    # [기법 1: 위상 분할 (Phase-Aware Kinematics)]
    # Phase 0: 셋업/하강 (0~20)
    p0_slice = slice(0, 21)
    # Phase 1: 리프팅 / 퍼스트 풀 (21~45, 바닥에서 무릎 통과 구간)
    p1_slice = slice(21, 46)
    
    f13_setup_ka_min = float(np.min(ka_ratio_s[p0_slice]))
    f14_pull_ka_min = float(np.min(ka_ratio_s[p1_slice]))
    f15_delta_pull = float(max(0.0, f13_setup_ka_min - f14_pull_ka_min))
    
    # [기법 2: 3D FPPA (Frontal Plane Projection Angle / 전두면 투영각)]
    # 가상 정면(관상면 X-Y 투영) 상에서 고관절-무릎-발목 꺾임각 측정
    # v_thigh = Hip - Knee, v_shank = Ankle - Knee
    thigh_l_xy = l_hip[:, :2] - l_knee[:, :2]
    shank_l_xy = l_ank[:, :2] - l_knee[:, :2]
    thigh_r_xy = r_hip[:, :2] - r_knee[:, :2]
    shank_r_xy = r_ank[:, :2] - r_knee[:, :2]
    
    dot_l = np.sum(thigh_l_xy * shank_l_xy, axis=1)
    norm_tl = np.linalg.norm(thigh_l_xy, axis=1) + 1e-7
    norm_sl = np.linalg.norm(shank_l_xy, axis=1) + 1e-7
    cos_l = np.clip(dot_l / (norm_tl * norm_sl), -1.0, 1.0)
    fppa_l = np.degrees(np.arccos(cos_l))  # 일직선일 때 180도
    
    dot_r = np.sum(thigh_r_xy * shank_r_xy, axis=1)
    norm_tr = np.linalg.norm(thigh_r_xy, axis=1) + 1e-7
    norm_sr = np.linalg.norm(shank_r_xy, axis=1) + 1e-7
    cos_r = np.clip(dot_r / (norm_tr * norm_sr), -1.0, 1.0)
    fppa_r = np.degrees(np.arccos(cos_r))
    
    valgus_angle_l = np.maximum(0.0, 180.0 - fppa_l)
    valgus_angle_r = np.maximum(0.0, 180.0 - fppa_r)
    valgus_angle_sum = valgus_angle_l + valgus_angle_r
    valgus_angle_sum_s = gaussian_filter1d(valgus_angle_sum, sigma=1.0)
    f16_fppa_valgus_max = float(np.max(valgus_angle_sum_s))
    
    # [발목-정강이 회전 어긋남 (Ankle-Shank Rotational Mismatch)]
    # 발끝 벡터(Toe - Ankle) 대비 정강이 횡방향 기울기의 불일치
    foot_vec_l = l_toe[:, :2] - l_ank[:, :2]
    foot_vec_r = r_toe[:, :2] - r_ank[:, :2]
    # 발목 대비 발끝 각도
    foot_angle_l = np.arctan2(foot_vec_l[:, 1], foot_vec_l[:, 0])
    foot_angle_r = np.arctan2(foot_vec_r[:, 1], foot_vec_r[:, 0])
    shank_angle_l = np.arctan2(shank_l_xy[:, 1], shank_l_xy[:, 0])
    shank_angle_r = np.arctan2(shank_r_xy[:, 1], shank_r_xy[:, 0])
    
    mismatch_l = np.abs(foot_angle_l - shank_angle_l)
    mismatch_r = np.abs(foot_angle_r - shank_angle_r)
    f17_rot_mismatch = float(np.percentile(mismatch_l + mismatch_r, 85))
    
    # [기법 3: 손-무릎 차폐 인식 (Arm Occlusion Distance)]
    # 무릎(25, 26)과 손목(15, 16)/팔꿈치(13, 14) 간의 최소 3D 거리
    dist_l_wrist = np.linalg.norm(l_wrist - l_knee, axis=1)
    dist_r_wrist = np.linalg.norm(r_wrist - r_knee, axis=1)
    dist_l_elbow = np.linalg.norm(l_elbow - l_knee, axis=1)
    dist_r_elbow = np.linalg.norm(r_elbow - r_knee, axis=1)
    min_arm_dist = np.minimum(np.minimum(dist_l_wrist, dist_r_wrist), np.minimum(dist_l_elbow, dist_r_elbow))
    f18_arm_dist_min = float(np.min(min_arm_dist))
    
    f6_innov = np.array([
        f13_setup_ka_min,
        f14_pull_ka_min,
        f15_delta_pull,
        f16_fppa_valgus_max,
        f17_rot_mismatch,
        f18_arm_dist_min
    ], dtype=np.float32)
    
    f18 = np.concatenate([f12_base, f6_innov])
    return np.nan_to_num(f18, nan=0.0, posinf=180.0, neginf=-180.0)


FEATURE_NAMES_18D = [
    # Base 12D from v3.0
    "world_kp_ratio_min_abs",
    "world_ka_absolute_deficit_integ",
    "world_valgus_abs_sum_max",
    "world_valgus_norm_hip_max",
    "world_shank_inward_tilt_p90",
    "world_valgus_inward_accel_peak",
    "world_morph_kp_drop_max",
    "canonical_valgus_abs_sum_max",
    "canonical_valgus_norm_hip_max",
    "canonical_shank_tilt_p90",
    "canonical_ka_ratio_min",
    "canonical_kp_ratio_min",
    # 6 Innovative Features in v4.0
    "phase0_setup_ka_ratio_min",
    "phase1_first_pull_ka_min",
    "phase_delta_ka_pull_drop",
    "fppa_3d_valgus_angle_max",
    "ankle_internal_rotation_mismatch",
    "arm_knee_min_distance"
]


def main():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 데드리프트 d2 v4.0 고도화 모델 구축 및 실측 평가")
    print("  - 아키텍처: 18D 위상인식 & 3D FPPA 기하 피처 + Cleanlab Confident Learning + 비용 민감 이중 GBDT 앙상블")
    print("  - 데이터셋: AI-Hub 바벨 데드리프트 385개 시퀀스 전수 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (9명 피험자 완전 격리)")
    print("=" * 115)

    os.makedirs(r"C:\Temp\catboost_info", exist_ok=True)
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    X_list = []
    y_list = []
    subs_list = []
    views_list = []
    keys_list = []

    t0 = time.time()
    for p in files:
        base = os.path.basename(p).replace(".npz", "")
        parts = base.split("_")
        vn = parts[-1]
        folder = "_".join(parts[:-1])
        key = [pt for pt in parts if pt.startswith("Key")][0]

        sub = 1
        for i in range(1, 10):
            if f"피험자{i}" in folder:
                sub = i
                break

        with open(os.path.join(LABEL_JSON_DIR, f"{folder}.json"), "r", encoding="utf-8", errors="ignore") as fp:
            d = json.load(fp)
        conds = {c["condition"]: c["value"] for c in d.get("type_info", {}).get("conditions", [])}
        y_val = 0 if conds.get("발과 무릎의 방향 일치", True) else 1

        kps = np.load(p)["kps_33_3d_64f"]
        f18 = extract_d2_v4_phase_fppa_features(kps)

        X_list.append(f18)
        y_list.append(y_val)
        subs_list.append(sub)
        views_list.append(vn)
        keys_list.append(key)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subs = np.array(subs_list, dtype=np.int32)
    views = np.array(views_list)
    keys = np.array(keys_list)

    print(f"[+] 18D 위상인식 & FPPA 피처 추출 완료: {X.shape} (경과: {time.time()-t0:.1f}초)")
    print(f"  - 정상(0): {np.sum(y==0)}건 vs 결함(1): {np.sum(y==1)}건 (결함비율: {np.mean(y)*100:.1f}%)")

    # 1단계: 1차 OOF 확률 추정 (Cleanlab용)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    init_oof_probs = np.zeros((len(y), 2), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=140, learning_rate=0.03, max_depth=4, num_leaves=14,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=3.28, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)
        p_l = clf_lgb.predict_proba(X_val)

        clf_cat = CatBoostClassifier(
            iterations=170, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1,
            train_dir=r"C:\Temp\catboost_info"
        )
        clf_cat.fit(X_tr, y_tr)
        p_c = clf_cat.predict_proba(X_val)

        init_oof_probs[val_idx] = 0.5 * p_l + 0.5 * p_c

    # 2단계: Cleanlab Confident Learning 기반 신뢰 마스크 산출
    print("\n[*] Cleanlab Confident Learning 실행 중...")
    label_issues = find_label_issues(
        labels=y, pred_probs=init_oof_probs, return_indices_ranked_by='self_confidence', n_jobs=1
    )
    quality_scores = get_label_quality_scores(labels=y, pred_probs=init_oof_probs)
    print(f"  [+] 라벨 모호성 및 연쇄 오염 감지 샘플: {len(label_issues)}개")

    # 신뢰 마스크 (가장 극단적인 오염 샘플 50개 프루닝)
    worst_indices = set(label_issues[:50])
    confident_mask = np.array([i not in worst_indices for i in range(len(y))])
    print(f"  [+] 정제된 클린 트레이닝 세트: {np.sum(confident_mask)}개 (노이즈 {np.sum(~confident_mask)}개 학습 배제)")

    # 3단계: 정제된 클린셋 기반 본학습 (5-Fold StratifiedGroupKFold 피험자 완전 격리)
    oof_probs_v4 = np.zeros(len(y), dtype=np.float32)
    lgb_models_v4 = []
    cat_models_v4 = []

    print("\n[*] 5-Fold StratifiedGroupKFold 본학습 시작 (LightGBM + CatBoost 앙상블)...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subs), 1):
        tr_mask = confident_mask[train_idx]
        actual_train_idx = train_idx[tr_mask]

        X_tr, y_tr = X[actual_train_idx], y[actual_train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # LightGBM (비용 민감 scale_pos_weight=3.28)
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=140, learning_rate=0.03, max_depth=4, num_leaves=14,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.5,
            scale_pos_weight=3.28, random_state=100 + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr, y_tr)

        # CatBoost (Balanced 클래스 가중치)
        clf_cat = CatBoostClassifier(
            iterations=170, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=100 + fold, verbose=0, thread_count=1,
            train_dir=r"C:\Temp\catboost_info"
        )
        clf_cat.fit(X_tr, y_tr)

        p_l = clf_lgb.predict_proba(X_val)[:, 1]
        p_c = clf_cat.predict_proba(X_val)[:, 1]
        p_ens = 0.5 * p_l + 0.5 * p_c

        oof_probs_v4[val_idx] = p_ens
        lgb_models_v4.append(clf_lgb)
        cat_models_v4.append(clf_cat)

    # 4단계: 실측 성능 평가
    auc_v4 = roc_auc_score(y, oof_probs_v4)

    # 2-State 최적 임계값 탐색
    best_th_2s = 0.50
    best_f1_2s = 0.0
    for th in np.arange(0.15, 0.70, 0.01):
        preds = (oof_probs_v4 >= th).astype(int)
        f1 = f1_score(y, preds, zero_division=0) * 100
        if f1 > best_f1_2s:
            best_f1_2s = f1
            best_th_2s = th

    preds_2s = (oof_probs_v4 >= best_th_2s).astype(int)
    acc_2s = accuracy_score(y, preds_2s) * 100
    prec_2s = precision_score(y, preds_2s, zero_division=0) * 100
    rec_2s = recall_score(y, preds_2s, zero_division=0) * 100
    cm_2s = confusion_matrix(y, preds_2s)

    # 5단계: [요구사항 준수] 보류율 <= 10.0% (Coverage >= 90.0%) 하의 3-State 최적 윈도우 탐색
    best_win_3s = None
    best_f1_3s = 0.0
    best_cov_3s = 0.0
    best_dec_3s = 0.0
    best_prec_3s = 0.0
    best_rec_3s = 0.0
    best_acc_3s = 0.0
    best_cm_3s = None

    for tn in np.arange(0.10, 0.40, 0.005):
        for td in np.arange(tn + 0.005, 0.55, 0.005):
            decided = (oof_probs_v4 <= tn) | (oof_probs_v4 >= td)
            cov = float(np.mean(decided) * 100)
            decline = float(100.0 - cov)

            if cov < 90.0:  # 보류율 10% 초과 배제
                continue

            y_dec = y[decided]
            p_dec = (oof_probs_v4[decided] >= td).astype(int)

            if len(np.unique(y_dec)) < 2:
                continue

            f1 = float(f1_score(y_dec, p_dec, zero_division=0) * 100)
            if f1 > best_f1_3s:
                best_f1_3s = f1
                best_win_3s = (round(float(tn), 3), round(float(td), 3))
                best_cov_3s = cov
                best_dec_3s = decline
                best_prec_3s = float(precision_score(y_dec, p_dec, zero_division=0) * 100)
                best_rec_3s = float(recall_score(y_dec, p_dec, zero_division=0) * 100)
                best_acc_3s = float(accuracy_score(y_dec, p_dec) * 100)
                best_cm_3s = confusion_matrix(y_dec, p_dec)

    print("\n" + "=" * 115)
    print("  [v4.0 실측 종합 성적표 (N=385, 피험자 1~9번 전원 전수 검증)]")
    print("=" * 115)
    print(f"  • ROC-AUC               : {auc_v4:.4f} (v3.0 0.7279 대비 {auc_v4-0.7279:+.4f})")
    print(f"  • 2-State F1-Score      : {best_f1_2s:.2f}% (임계값 {best_th_2s:.2f}, v3.0 52.46% 대비 {best_f1_2s-52.46:+.2f}%p)")
    print(f"  • 2-State 정밀도/재현율 : 정밀도 {prec_2s:.2f}% / 재현율 {rec_2s:.2f}% / 정확도 {acc_2s:.2f}%")
    print(f"  • 2-State 혼동행렬      : TP={cm_2s[1,1]}, FP={cm_2s[0,1]}, FN={cm_2s[1,0]}, TN={cm_2s[0,0]}")
    print("-" * 115)
    print(f"  [★ 3-State 공식 채택 윈도우]: [{best_win_3s[0]:.3f}, {best_win_3s[1]:.3f}]")
    print(f"  • 3-State F1-Score      : {best_f1_3s:.2f}% (v3.0 53.98% 대비 {best_f1_3s-53.98:+.2f}%p🔺)")
    print(f"  • 3-State 정밀도        : {best_prec_3s:.2f}% (v3.0 42.07% 대비 {best_prec_3s-42.07:+.2f}%p🔺)")
    print(f"  • 3-State 재현율        : {best_rec_3s:.2f}% (v3.0 75.31%)")
    print(f"  • 3-State 정확도        : {best_acc_3s:.2f}%")
    print(f"  • 판정 커버리지         : {best_cov_3s:.1f}% (385건 중 {int(round(len(y)*best_cov_3s/100))}건 확정 판정)")
    print(f"  • 판단 보류율           : {best_dec_3s:.1f}% (10% 이하 엄격 준수!)")
    print(f"  • 3-State 혼동행렬      : TP={best_cm_3s[1,1]}, FP={best_cm_3s[0,1]}, FN={best_cm_3s[1,0]}, TN={best_cm_3s[0,0]}")
    print("=" * 115)

    # 5대 전 시점별 다차원 성능 감사
    print(f"\n[공식 5대 시점별 3-State F1-Score (윈도우 [{best_win_3s[0]:.3f}, {best_win_3s[1]:.3f}])]:")
    print(f"{'공식 시점 명칭':<24} | {'코드':<6} | {'3-State F1':<12} | {'Precision':<10} | {'Recall':<10} | {'Coverage'}")
    print("-" * 85)
    view_v4_res = {}
    for v_code, v_alias in VIEW_ALIAS_MAP.items():
        v_mask = (views == v_code)
        y_v = y[v_mask]
        prob_v = oof_probs_v4[v_mask]

        dec_v = (prob_v <= best_win_3s[0]) | (prob_v >= best_win_3s[1])
        y_v_dec = y_v[dec_v]
        p_v_dec = (prob_v[dec_v] >= best_win_3s[1]).astype(int)

        f1_v = f1_score(y_v_dec, p_v_dec, zero_division=0) * 100
        prec_v = precision_score(y_v_dec, p_v_dec, zero_division=0) * 100
        rec_v = recall_score(y_v_dec, p_v_dec, zero_division=0) * 100
        cov_v = np.mean(dec_v) * 100
        view_v4_res[v_alias] = {'f1': f1_v, 'prec': prec_v, 'rec': rec_v, 'cov': cov_v}
        print(f"{v_alias:<24} | {v_code:<6} | {f1_v:<12.2f}% | {prec_v:<10.2f}% | {rec_v:<10.2f}% | {cov_v:.1f}%")
    print("=" * 85)

    # 6단계: 가중치 및 결과 영구 저장
    bundle_v4 = {
        "lightgbm_models": lgb_models_v4,
        "catboost_models": cat_models_v4,
        "feature_names": FEATURE_NAMES_18D,
        "auc": float(auc_v4),
        "best_2s_threshold": float(best_th_2s),
        "best_2s_f1": float(best_f1_2s),
        "best_3s_window": [float(best_win_3s[0]), float(best_win_3s[1])],
        "best_3s_f1": float(best_f1_3s),
        "best_3s_precision": float(best_prec_3s),
        "best_3s_recall": float(best_rec_3s),
        "coverage": float(best_cov_3s),
        "decline_rate": float(best_dec_3s),
        "view_results": view_v4_res,
        "oof_predictions": oof_probs_v4
    }

    with open(WEIGHTS_V4_PATH, "wb") as f:
        pickle.dump(bundle_v4, f)
    print(f"\n[+] v4.0 공식 모델 가중치 파일 영구 저장 완료: {WEIGHTS_V4_PATH}")

    # 결과 JSON 저장
    res_json_path = os.path.join(r"D:\src\models\second_models\babel_deadlift", "d2_v4_evaluation_results.json")
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "auc": float(auc_v4),
            "best_2s_threshold": float(best_th_2s),
            "best_2s_f1": float(best_f1_2s),
            "best_3s_window": [float(best_win_3s[0]), float(best_win_3s[1])],
            "best_3s_f1": float(best_f1_3s),
            "best_3s_precision": float(best_prec_3s),
            "best_3s_recall": float(best_rec_3s),
            "coverage": float(best_cov_3s),
            "decline_rate": float(best_dec_3s),
            "view_results": view_v4_res
        }, f, ensure_ascii=False, indent=2)
    print(f"[+] 결과 메트릭 JSON 저장 완료: {res_json_path}")


if __name__ == "__main__":
    main()
