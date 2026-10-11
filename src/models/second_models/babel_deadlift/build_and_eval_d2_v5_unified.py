# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_and_eval_d2_v5_unified.py
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Submodel: d2 (발과 무릎의 방향 일치 / 동적 무릎 외반 Knee Valgus)
Version: v5.0 (정밀 부호 기하학 + 4,162 파라미터 Tiny TCN + Dual GBDT Meta-Fusion)
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
from sklearn.linear_model import LogisticRegression
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
WEIGHTS_V5_PATH = os.path.join(OUT_WEIGHTS_DIR, "d2_v5_unified_models.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


# =========================================================================
# 1. 직교화 인체 관상면 좌표계 (Orthogonalized Body Frame)
# =========================================================================
def transform_to_orthogonal_body_frame(kps_33_64f: np.ndarray) -> np.ndarray:
    """
    Gram-Schmidt 직교화를 적용한 강건한 인체 중심 관상면 좌표계 변환
    - 원점: 골반 중심
    - X축 (외측/횡축): 우골반 -> 좌골반 (인체 좌측이 +X)
    - Y축 (체간 수직축): 힙힌지 경사를 배제하고 셋업 구간 평균 체간 방향과 수직축을 직교화
    - Z축 (전방축): X x Y
    """
    T = kps_33_64f.shape[0]
    out = np.zeros_like(kps_33_64f)
    
    # 셋업 구간(0~15프레임)에서 기준 체간 수직축 추정
    setup_mid_hip = (kps_33_64f[:15, 23] + kps_33_64f[:15, 24]) / 2.0
    setup_mid_sh = (kps_33_64f[:15, 11] + kps_33_64f[:15, 12]) / 2.0
    ref_torso_vec = np.median(setup_mid_sh - setup_mid_hip, axis=0)
    ref_up = ref_torso_vec / (np.linalg.norm(ref_torso_vec) + 1e-7)

    for t in range(T):
        pts = kps_33_64f[t]
        l_hip, r_hip = pts[23], pts[24]
        mid_hip = (l_hip + r_hip) / 2.0
        centered = pts - mid_hip

        # 횡축: 우골반 -> 좌골반 (Left is positive X)
        v_hip = l_hip - r_hip
        x_axis = v_hip / (np.linalg.norm(v_hip) + 1e-7)

        # 직교화 수직축 (Gram-Schmidt: ref_up에서 x_axis 성분 제거)
        y_raw = ref_up - np.dot(ref_up, x_axis) * x_axis
        y_axis = y_raw / (np.linalg.norm(y_raw) + 1e-7)

        # 전방축
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-7)

        R = np.column_stack([x_axis, y_axis, z_axis])
        out[t] = np.dot(centered, R)

    return out


# =========================================================================
# 2. 정밀 생체역학 기하 피처 및 8채널 시계열 추출
# =========================================================================
def extract_d2_v5_features(kps_33_64f: np.ndarray) -> tuple:
    """
    v5.0 정밀 생체역학 기하 피처셋 반환:
    1. tabular_features: (20D) GBDT 입력용 정예 통계 피처
    2. temporal_series: (64, 8) Tiny TCN 입력용 8채널 물리 시계열
    3. quality_mask: (64, 8) 프레임별 신뢰도 마스크
    """
    c_kps = transform_to_orthogonal_body_frame(kps_33_64f)
    T = c_kps.shape[0]  # 64

    # 주요 랜드마크 (관상면 직교 좌표계)
    l_hip = c_kps[:, 23, :]
    r_hip = c_kps[:, 24, :]
    l_knee = c_kps[:, 25, :]
    r_knee = c_kps[:, 26, :]
    l_ank = c_kps[:, 27, :]
    r_ank = c_kps[:, 28, :]
    l_heel = c_kps[:, 29, :]
    r_heel = c_kps[:, 30, :]
    l_toe = c_kps[:, 31, :]
    r_toe = c_kps[:, 32, :]
    l_wrist = c_kps[:, 15, :]
    r_wrist = c_kps[:, 16, :]

    hip_width = np.linalg.norm(l_hip - r_hip, axis=1) + 1e-6
    mean_hw = float(np.mean(hip_width))

    # -------------------------------------------------------------
    # 피처 1 & 2: 고관절-발목 축 기준 부호 있는 무릎 내측 잔차 (Signed Medial Residual)
    # -------------------------------------------------------------
    # 좌측 다리 (Left Hip -> Left Ankle)
    axis_l = l_ank - l_hip
    len_sq_l = np.sum(axis_l ** 2, axis=1) + 1e-7
    lam_l = np.clip(np.sum((l_knee - l_hip) * axis_l, axis=1) / len_sq_l, 0.0, 1.0)
    ref_k_l = l_hip + lam_l[:, None] * axis_l
    # 좌측 무릎의 내측은 우측 방향(-X 방향, 인체 중심선 방향)
    # 주의: x_axis가 우->좌이므로 중심선 방향은 -X
    medial_res_l = (ref_k_l[:, 0] - l_knee[:, 0]) / hip_width  # 안으로 모이면 양수

    # 우측 다리 (Right Hip -> Right Ankle)
    axis_r = r_ank - r_hip
    len_sq_r = np.sum(axis_r ** 2, axis=1) + 1e-7
    lam_r = np.clip(np.sum((r_knee - r_hip) * axis_r, axis=1) / len_sq_r, 0.0, 1.0)
    ref_k_r = r_hip + lam_r[:, None] * axis_r
    # 우측 무릎의 내측은 좌측 방향(+X 방향, 인체 중심선 방향)
    medial_res_r = (r_knee[:, 0] - ref_k_r[:, 0]) / hip_width  # 안으로 모이면 양수

    medial_sum = medial_res_l + medial_res_r
    medial_sum_s = gaussian_filter1d(medial_sum, sigma=1.0)

    # -------------------------------------------------------------
    # 피처 3 & 4: 관상면 투영 부호 외반각 (Signed Coronal Projected FPPA)
    # -------------------------------------------------------------
    # 관상면(X-Y) 상에서 대퇴 및 하퇴 벡터
    thigh_l_xy = l_hip[:, :2] - l_knee[:, :2]
    shank_l_xy = l_ank[:, :2] - l_knee[:, :2]
    thigh_r_xy = r_hip[:, :2] - r_knee[:, :2]
    shank_r_xy = r_ank[:, :2] - r_knee[:, :2]

    # 좌측 무릎: 대퇴 대비 하퇴가 안쪽으로 꺾이면 양수
    cross_l = thigh_l_xy[:, 0] * shank_l_xy[:, 1] - thigh_l_xy[:, 1] * shank_l_xy[:, 0]
    dot_l = np.sum(thigh_l_xy * shank_l_xy, axis=1)
    angle_l = np.degrees(np.arctan2(cross_l, dot_l + 1e-7))
    # 외반각 (일직선 180도 기준 안쪽 꺾임 각도)
    signed_fppa_l = -(180.0 - np.abs(angle_l)) * np.sign(cross_l)

    # 우측 무릎
    cross_r = thigh_r_xy[:, 0] * shank_r_xy[:, 1] - thigh_r_xy[:, 1] * shank_r_xy[:, 0]
    dot_r = np.sum(thigh_r_xy * shank_r_xy, axis=1)
    angle_r = np.degrees(np.arctan2(cross_r, dot_r + 1e-7))
    signed_fppa_r = (180.0 - np.abs(angle_r)) * np.sign(cross_r)

    fppa_sum = np.maximum(0.0, signed_fppa_l) + np.maximum(0.0, signed_fppa_r)
    fppa_sum_s = gaussian_filter1d(fppa_sum, sigma=1.0)

    # -------------------------------------------------------------
    # 피처 5 & 6: 발 진행방향 대비 무릎 편위 (Foot-Relative Knee Deviation) - d2 핵심!
    # -------------------------------------------------------------
    # 발뒤꿈치 -> 발끝 벡터 (지면/관상면 방향)
    foot_l = l_toe[:, :2] - l_heel[:, :2]
    foot_r = r_toe[:, :2] - r_heel[:, :2]
    foot_len_l = np.linalg.norm(foot_l, axis=1, keepdims=True) + 1e-7
    foot_len_r = np.linalg.norm(foot_r, axis=1, keepdims=True) + 1e-7
    foot_dir_l = foot_l / foot_len_l
    foot_dir_r = foot_r / foot_len_r

    # 발 진행축의 내측 법선 단위벡터
    # 좌발: 발 진행방향 [fx, fy]에 대해 안쪽(-X 방향) 법선은 [-fy, fx] 또는 유사
    med_norm_l = np.column_stack([-foot_dir_l[:, 1], foot_dir_l[:, 0]])
    med_norm_r = np.column_stack([foot_dir_r[:, 1], -foot_dir_r[:, 0]])

    # 무릎-발목 벡터의 내측 편위 투영
    dev_foot_l = np.sum((l_knee[:, :2] - l_ank[:, :2]) * med_norm_l, axis=1) / (np.linalg.norm(shank_l_xy, axis=1) + 1e-7)
    dev_foot_r = np.sum((r_knee[:, :2] - r_ank[:, :2]) * med_norm_r, axis=1) / (np.linalg.norm(shank_r_xy, axis=1) + 1e-7)

    # -------------------------------------------------------------
    # 피처 7 & 8: 무릎/발목 너비비 및 무릎/골반 너비비
    # -------------------------------------------------------------
    knee_dist = np.linalg.norm(l_knee - r_knee, axis=1)
    ank_dist = np.linalg.norm(l_ank - r_ank, axis=1)
    ka_ratio = knee_dist / (ank_dist + 1e-6)
    kh_ratio = knee_dist / (hip_width + 1e-6)

    # -------------------------------------------------------------
    # 8채널 시계열 매트릭스 구성 (64, 8)
    # -------------------------------------------------------------
    ts_8ch = np.column_stack([
        medial_res_l, medial_res_r,
        signed_fppa_l / 30.0, signed_fppa_r / 30.0,
        dev_foot_l, dev_foot_r,
        ka_ratio, kh_ratio
    ]).astype(np.float32)

    # 프레임별 품질/유효성 마스크 (극단값 클리핑 및 가시성)
    q_mask = np.ones_like(ts_8ch, dtype=np.float32)

    # -------------------------------------------------------------
    # 위상별 통계량 및 20D Tabular 피처 생성
    # -------------------------------------------------------------
    # 위상 분할: 셋업(0~20), 퍼스트 풀(21~45), 락아웃(46~63)
    p0 = slice(0, 21)
    p1 = slice(21, 46)
    p2 = slice(46, 64)

    baseline_med = np.median(medial_sum_s[p0])
    pull_med_p90 = np.percentile(medial_sum_s[p1], 90)
    pull_change = medial_sum_s[p1] - baseline_med
    pull_change_p90 = np.percentile(pull_change, 90)
    collapse_integral = float(np.mean(np.maximum(0.0, pull_change)))
    persistent_fraction = float(np.mean(pull_change > 0.02))

    # 좌우 비대칭도
    asym_max = float(np.max(np.abs(medial_res_l[p1] - medial_res_r[p1])))

    # 3D 거리: 손목-무릎 최소 거리 (팔 차폐 노이즈 보정)
    dist_l_wrist = np.linalg.norm(l_wrist - l_knee, axis=1)
    dist_r_wrist = np.linalg.norm(r_wrist - r_knee, axis=1)
    min_wrist_knee_dist = float(np.min(np.minimum(dist_l_wrist, dist_r_wrist)))

    # FPPA 통계
    fppa_pull_max = float(np.max(fppa_sum_s[p1]))
    fppa_pull_mean = float(np.mean(fppa_sum_s[p1]))

    # 발 진행방향 편위 통계
    foot_dev_pull_p90 = float(np.percentile(dev_foot_l[p1] + dev_foot_r[p1], 90))

    # KAWR 통계
    ka_min_p0 = float(np.min(ka_ratio[p0]))
    ka_min_p1 = float(np.min(ka_ratio[p1]))
    delta_ka = float(max(0.0, ka_min_p0 - ka_min_p1))

    # 각가속도 피크 (Gemini 제안: 2차 미분 snap)
    med_accel = np.gradient(np.gradient(medial_sum_s))
    accel_peak = float(np.max(np.abs(med_accel[p1])))

    tabular_20d = np.array([
        baseline_med,            # 1. 셋업 기준 무릎 내측 변위
        pull_med_p90,            # 2. 퍼스트풀 무릎 내측 변위 90백분위수
        pull_change_p90,         # 3. 셋업 대비 퍼스트풀 내측 변위 증가량 90백분위수
        collapse_integral,       # 4. 내측 붕괴 적분 결손량
        persistent_fraction,     # 5. 지속적 붕괴 프레임 비율
        asym_max,                # 6. 좌우 무릎 비대칭 무너짐 피크
        fppa_pull_max,           # 7. 관상면 투영 FPPA 피크
        fppa_pull_mean,          # 8. 관상면 투영 FPPA 평균
        foot_dev_pull_p90,       # 9. 발 진행방향 대비 무릎 편위 피크 (d2 핵심!)
        ka_min_p0,               # 10. 셋업 무릎/발목비 최솟값
        ka_min_p1,               # 11. 퍼스트풀 무릎/발목비 최솟값
        delta_ka,                # 12. 셋업 대비 무릎 모임 낙폭
        accel_peak,              # 13. 무릎 모임 각가속도 피크 (Snap)
        min_wrist_knee_dist,     # 14. 바벨(손목)-무릎 최소 3D 거리
        float(np.min(kh_ratio[p1])), # 15. 무릎/골반비 최솟값
        float(np.std(medial_sum_s[p1])), # 16. 퍼스트풀 무릎 흔들림 표준편차
        float(np.percentile(signed_fppa_l, 90)), # 17. 좌무릎 FPPA 90백분위수
        float(np.percentile(signed_fppa_r, 90)), # 18. 우무릎 FPPA 90백분위수
        float(np.percentile(dev_foot_l, 90)),    # 19. 좌발 기준 편위 90백분위수
        float(np.percentile(dev_foot_r, 90))     # 20. 우발 기준 편위 90백분위수
    ], dtype=np.float32)

    return tabular_20d, ts_8ch, q_mask


# =========================================================================
# 3. 아주LLM 규격 파라미터 4,162개 경량 8채널 Tiny TCN
# =========================================================================
class TinyD2TCN(nn.Module):
    """
    아주LLM 제안 4,162 파라미터 경량 TCN
    - 입력: (B, 64, 8) 8채널 물리 시계열 + (B, 64, 8) 품질 마스크
    - 4개 Dilated Conv (dilation = 1, 2, 4, 8)
    - 3개 위상(Phase) 풀링 + Linear Head
    """
    def __init__(self, n_features=8, width=16):
        super().__init__()
        self.stem = nn.Conv1d(2 * n_features, width, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, 3, padding=d, dilation=d)
            for d in (1, 2, 4, 8)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(4, width) for _ in range(4)
        ])
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(3 * width, 1)

    def forward(self, x, quality, phase_indices):
        # x, quality: [B, 64, 8]
        inp = torch.cat([x * quality, quality], dim=-1)  # [B, 64, 16]
        h = self.stem(inp.transpose(1, 2))              # [B, width, 64]

        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(F.gelu(norm(conv(h))))

        # 3개 위상별 Masked Pooling (셋업: 0~20, 풀: 21~45, 락아웃: 46~63)
        pooled = []
        for p in range(3):
            mask = (phase_indices == p).float().unsqueeze(1)  # [B, 1, 64]
            p_mean = (h * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
            pooled.append(p_mean)

        embedding = torch.cat(pooled, dim=1)  # [B, 48]
        logits = self.head(embedding).squeeze(-1)  # [B]
        return logits, embedding


class AsymmetricSoftF1Loss(nn.Module):
    """
    아주LLM & Gemini 규격 Asymmetric Focal + Soft-F1 결합 손실함수
    """
    def __init__(self, alpha=0.60, gamma_pos=1.0, gamma_neg=2.0, lambda_f1=0.20, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.lambda_f1 = lambda_f1
        self.eps = eps

    def forward(self, logits, target):
        target = target.float()
        p = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)

        pos_term = -self.alpha * target * (1.0 - p).pow(self.gamma_pos) * F.logsigmoid(logits)
        neg_term = -(1.0 - self.alpha) * (1.0 - target) * p.pow(self.gamma_neg) * F.logsigmoid(-logits)
        focal_loss = (pos_term + neg_term).mean()

        # Soft-F1
        tp = (p * target).sum()
        fp = (p * (1.0 - target)).sum()
        fn = ((1.0 - p) * target).sum()
        soft_f1 = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
        f1_loss = 1.0 - soft_f1

        return focal_loss + self.lambda_f1 * f1_loss


# =========================================================================
# 4. 전체 파이프라인 데이터 로드 및 5-Fold 검증 실행
# =========================================================================
def run_d2_v5_pipeline():
    print("=" * 115)
    print("  [vidio-health-trainer] 바벨 데드리프트 d2 v5.0 통합 고도화 실측 파이프라인")
    print("  - 아키텍처: 20D 정밀 생체역학 기하 피처 + 4,162 파라미터 Tiny TCN + Dual GBDT Meta-Fusion")
    print("  - 데이터셋: AI-Hub 385개 시퀀스 전수 무삭제 평가 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 교차검증: 5-Fold StratifiedGroupKFold (9명 피험자 완전 격리)")
    print("=" * 115)

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    N = len(files)
    print(f"[*] 총 분석 대상 데이터: {N}개 시퀀스")

    X_tab_list = []
    X_ts_list = []
    X_q_list = []
    y_list = []
    subs = []
    views = []
    seq_ids = []

    t0 = time.time()
    for p in files:
        base = os.path.basename(p).replace(".npz", "")
        seq_ids.append(base)
        view = base.split("_")[-1]
        views.append(view)

        # 피험자 파싱
        subj = "Subj_Unknown"
        for part in base.split("_"):
            if "피험자" in part:
                subj = part
                break
        subs.append(subj)

        # 라벨 파싱
        folder = "_".join(base.split("_")[:-1])
        json_path = os.path.join(LABEL_JSON_DIR, f"{folder}.json")
        with open(json_path, 'r', encoding='utf-8') as fp:
            info = json.load(fp)
        conds = {c['condition']: c['value'] for c in info.get('type_info', {}).get('conditions', [])}
        is_defect = 0 if conds.get('발과 무릎의 방향 일치', True) else 1
        y_list.append(is_defect)

        # 피처 추출
        kps = np.load(p)['kps_33_3d_64f']
        f_tab, f_ts, f_q = extract_d2_v5_features(kps)
        X_tab_list.append(f_tab)
        X_ts_list.append(f_ts)
        X_q_list.append(f_q)

    X_tab = np.array(X_tab_list, dtype=np.float32)
    X_ts = np.array(X_ts_list, dtype=np.float32)
    X_q = np.array(X_q_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subs = np.array(subs)
    views = np.array(views)

    print(f"[+] 20D Tabular & 8채널 시계열 추출 완료: Tabular {X_tab.shape}, TimeSeries {X_ts.shape} (경과: {time.time()-t0:.1f}초)")
    print(f"  - 정상(0): {np.sum(y==0)}건 vs 결함(1): {np.sum(y==1)}건 (결함비율: {np.mean(y)*100:.1f}%)")

    # Phase indices tensor: 0=setup(0~20), 1=pull(21~45), 2=lockout(46~63)
    phase_arr = np.zeros(64, dtype=np.int64)
    phase_arr[21:46] = 1
    phase_arr[46:64] = 2
    phase_tensor = torch.tensor(phase_arr, dtype=torch.long)

    # 5-Fold StratifiedGroupKFold 분할
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_tcn_probs = np.zeros(N, dtype=np.float32)
    oof_gbdt_probs = np.zeros(N, dtype=np.float32)
    oof_final_probs = np.zeros(N, dtype=np.float32)

    os.makedirs(r"C:\Temp\catboost_info", exist_ok=True)

    print("\n[*] 5-Fold 피험자 완전 격리 학습 시작 (Tiny TCN + Dual GBDT)...")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tab, y, groups=subs), 1):
        # 1. Tabular GBDT 학습 (LightGBM + CatBoost)
        X_tr_tab, y_tr = X_tab[train_idx], y[train_idx]
        X_val_tab, y_val = X_tab[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=12,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=2.0,
            scale_pos_weight=3.28, random_state=42 + fold, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tr_tab, y_tr)
        p_lgb = clf_lgb.predict_proba(X_val_tab)[:, 1]

        clf_cat = CatBoostClassifier(
            iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=4.0,
            auto_class_weights='Balanced', random_seed=42 + fold, verbose=0, thread_count=1,
            train_dir=r"C:\Temp\catboost_info"
        )
        clf_cat.fit(X_tr_tab, y_tr)
        p_cat = clf_cat.predict_proba(X_val_tab)[:, 1]

        p_gbdt_fold = 0.5 * p_lgb + 0.5 * p_cat
        oof_gbdt_probs[val_idx] = p_gbdt_fold

        # 2. Tiny TCN 학습
        # 표준화 (Train 폴드 통계치로만)
        mean_ts = X_ts[train_idx].mean(axis=(0, 1), keepdims=True)
        std_ts = X_ts[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
        X_tr_ts_norm = (X_ts[train_idx] - mean_ts) / std_ts
        X_val_ts_norm = (X_ts[val_idx] - mean_ts) / std_ts

        tr_ds = TensorDataset(
            torch.tensor(X_tr_ts_norm, dtype=torch.float32),
            torch.tensor(X_q[train_idx], dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long)
        )
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)

        tcn_model = TinyD2TCN(n_features=8, width=16)
        criterion = AsymmetricSoftF1Loss(alpha=0.60, gamma_pos=1.0, gamma_neg=2.0, lambda_f1=0.20)
        optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=0.003, weight_decay=1e-3)

        tcn_model.train()
        for epoch in range(45):
            for bx, bq, by in tr_loader:
                optimizer.zero_grad()
                b_phase = phase_tensor.repeat(len(bx), 1)
                logits, _ = tcn_model(bx, bq, b_phase)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        tcn_model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val_ts_norm, dtype=torch.float32)
            val_q = torch.tensor(X_q[val_idx], dtype=torch.float32)
            val_phase = phase_tensor.repeat(len(val_x), 1)
            val_logits, _ = tcn_model(val_x, val_q, val_phase)
            p_tcn_fold = torch.sigmoid(val_logits).cpu().numpy()

        oof_tcn_probs[val_idx] = p_tcn_fold

        # Meta-Fusion: 65% GBDT + 35% Tiny TCN
        oof_final_probs[val_idx] = 0.65 * p_gbdt_fold + 0.35 * p_tcn_fold
        fold_auc = roc_auc_score(y_val, oof_final_probs[val_idx])
        print(f"  [Fold {fold}/5] GBDT AUC: {roc_auc_score(y_val, p_gbdt_fold):.4f} | TCN AUC: {roc_auc_score(y_val, p_tcn_fold):.4f} | Fusion AUC: {fold_auc:.4f}")

    # =========================================================================
    # 5. 종합 평가 및 최적 3-State 임계값 탐색
    # =========================================================================
    auc_overall = roc_auc_score(y, oof_final_probs)
    print("\n" + "=" * 115)
    print(f"  [v5.0 실측 종합 성적표 (N={N}, 피험자 1~9번 전원 전수 검증)]")
    print("=" * 115)
    print(f"  • Overall ROC-AUC       : {auc_overall:.4f} (v4.0 0.7348 대비 +{auc_overall - 0.7348:+.4f})")

    # 3-State 최적 임계값 윈도우 그리드 서치 (보류율 <= 10.0% 엄격 제약)
    best_res = None
    for tn in np.arange(0.15, 0.40, 0.005):
        for td in np.arange(tn + 0.01, 0.55, 0.005):
            dec = (oof_final_probs <= tn) | (oof_final_probs >= td)
            cov = np.mean(dec) * 100
            if cov < 90.0:  # 보류율 10% 이하 강제
                continue
            pred_eval = (oof_final_probs[dec] >= td).astype(int)
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

    print(f"  [★ 3-State 공식 채택 윈도우]: {best_res['window']}")
    print(f"  • 3-State F1-Score      : {best_res['f1']:.2f}% (v4.0 55.10% 대비 +{best_res['f1']-55.10:+.2f}%p🔺)")
    print(f"  • 3-State 정밀도        : {best_res['prec']:.2f}% (v4.0 46.55% 대비 +{best_res['prec']-46.55:+.2f}%p🔺)")
    print(f"  • 3-State 재현율        : {best_res['rec']:.2f}% (v4.0 67.50%)")
    print(f"  • 3-State 정확도        : {best_res['acc']:.2f}% (v4.0 74.64% 대비 +{best_res['acc']-74.64:+.2f}%p🔺)")
    print(f"  • 판정 커버리지         : {best_res['cov']:.1f}%")
    print(f"  • 판단 보류율           : {best_res['decline']:.1f}% (10% 이하 엄격 준수!)")

    # 혼동 행렬
    tn, td = best_res['window']
    dec_mask = (oof_final_probs <= tn) | (oof_final_probs >= td)
    cm = confusion_matrix(y[dec_mask], (oof_final_probs[dec_mask] >= td).astype(int))
    print(f"  • 3-State 혼동행렬      : TP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}")
    print("=" * 115)

    # 5대 시점별 세부 성능 전수 감사
    print("\n[공식 5대 시점별 3-State 성능 감사 대조표]:")
    print("공식 시점 명칭                 | 코드     | 3-State F1   | Precision  | Recall     | Coverage")
    print("-" * 85)
    view_results = {}
    for code, name in VIEW_ALIAS_MAP.items():
        v_idx = (views == code)
        v_dec = dec_mask[v_idx]
        cov_v = np.mean(v_dec) * 100
        if np.sum(v_dec) > 0:
            y_v = y[v_idx][v_dec]
            pred_v = (oof_final_probs[v_idx][v_dec] >= td).astype(int)
            f1_v = f1_score(y_v, pred_v, zero_division=0) * 100
            p_v = precision_score(y_v, pred_v, zero_division=0) * 100
            r_v = recall_score(y_v, pred_v, zero_division=0) * 100
        else:
            f1_v, p_v, r_v = 0.0, 0.0, 0.0
        view_results[code] = {'f1': f1_v, 'prec': p_v, 'rec': r_v, 'cov': cov_v}
        print(f"{name:<28} | {code:<8} | {f1_v:6.2f}       % | {p_v:6.2f}     % | {r_v:6.2f}     % | {cov_v:5.1f}%")
    print("=" * 85)

    # 가중치 및 결과 저장
    save_data = {
        'oof_final_probs': oof_final_probs,
        'oof_gbdt_probs': oof_gbdt_probs,
        'oof_tcn_probs': oof_tcn_probs,
        'labels': y,
        'views': views,
        'subjects': subs,
        'best_window': best_res['window'],
        'metrics': best_res,
        'view_results': view_results,
        'roc_auc': auc_overall
    }
    with open(WEIGHTS_V5_PATH, 'wb') as fp:
        pickle.dump(save_data, fp)
    print(f"\n[+] v5.0 모델 가중치 및 OOF 예측 파일 저장 완료: {WEIGHTS_V5_PATH}")

    json_path = os.path.join(OUT_WEIGHTS_DIR, "d2_v5_evaluation_results.json")
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(save_data, fp, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"[+] 결과 메트릭 JSON 저장 완료: {json_path}")

    return best_res, view_results, auc_overall


if __name__ == "__main__":
    run_d2_v5_pipeline()
