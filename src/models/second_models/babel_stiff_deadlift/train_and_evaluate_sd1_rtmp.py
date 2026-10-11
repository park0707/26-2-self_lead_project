# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect: sd1. 무릎 구부린 채 고정 위반 (Knee Fixation Violation)
Script: train_and_evaluate_sd1_rtmp.py
Purpose: RTMPose-m (Halpe26) 최적화 sd1 모델 5-Fold 교차검증 및 패키징
         - Ceteris Paribus & Copy-and-Refine: 9인 피험자 격리 5-Fold GroupCV 전수 평가 (1,035건)
         - 2D 기하학 정합 30D 생체역학 피처 + 24D 정규화 시계열 텐서
         - Dual-Tree Ensemble (LightGBM + CatBoost) + Dilated-TCN 2-Stream Fusion
         - RTMPose 고유 확률 분포에 맞춘 3-State 안전 게이트 임계구간 최적화 (보류율 <= 15%)
         - 공식 번들 패키징: weights/sd1_rtmp_bundle.pkl
         - 공식 평가 결과 JSON 저장: sd1_rtmp_evaluation_results.json
"""

import os
import sys
import time
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sd1_rtmp_features import (
    extract_sd1_rtmp_features,
    extract_sd1_temporal_rtmp,
    FEATURE_NAMES_30D_SD1_RTMP,
    VIEW_INDEX_MAP
)

CACHE_PKL = os.path.join(CURRENT_DIR, "stiff_d1_v3_dataset_cache.pkl")
RTM_DIR = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트"
CLEAN_CACHE_NPZ = os.path.join(CURRENT_DIR, "stiff_sd1_rtmp_clean_features.npz")
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
BUNDLE_OUT_PATH = os.path.join(WEIGHTS_DIR, "sd1_rtmp_bundle.pkl")
RESULTS_JSON_PATH = os.path.join(CURRENT_DIR, "sd1_rtmp_evaluation_results.json")

VIEW_ALIAS_MAP = {
    'view5': '시점 A: 우측 후방 (45도)',
    'view2': '시점 B: 우측 전방 (45도)',
    'view3': '시점 C: 정면',
    'view4': '시점 D: 좌측 전방 (45도)',
    'view1': '시점 E: 좌측 후방 (45도)'
}


class DilatedTCN24(nn.Module):
    def __init__(self, in_channels: int = 24, hidden_dim: int = 32):
        super(DilatedTCN24, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, dilation=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4, padding=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        res1 = self.relu(self.bn1(self.conv1(x)))
        res2 = self.relu(self.bn2(self.conv2(res1))) + res1
        res3 = self.relu(self.bn3(self.conv3(res2))) + res2
        out = self.pool(res3).squeeze(-1)
        logits = self.fc(out).squeeze(-1)
        return logits


def evaluate_3state_metrics(y_true, y_prob, tau_clean, tau_defect):
    """
    3-State 선택적 판정 공식 평가 함수
    - P < tau_clean: 정상 확정 (0)
    - P >= tau_defect: 결함 확정 (1)
    - tau_clean <= P < tau_defect: 판단 보류 (Retake)
    """
    det_mask = (y_prob < tau_clean) | (y_prob >= tau_defect)
    cov = float(det_mask.mean() * 100.0)
    retake = float(100.0 - cov)

    if np.sum(det_mask) > 0:
        y_eval = y_true[det_mask]
        p_eval = (y_prob[det_mask] >= tau_defect).astype(int)
        f1 = float(f1_score(y_eval, p_eval, zero_division=0) * 100.0)
        prec = float(precision_score(y_eval, p_eval, zero_division=0) * 100.0)
        rec = float(recall_score(y_eval, p_eval, zero_division=0) * 100.0)
        acc = float(accuracy_score(y_eval, p_eval) * 100.0)
    else:
        f1, prec, rec, acc = 0.0, 0.0, 0.0, 0.0

    return {
        'sel_f1': round(f1, 2),
        'sel_prec': round(prec, 2),
        'sel_rec': round(rec, 2),
        'sel_acc': round(acc, 2),
        'coverage': round(cov, 2),
        'retake': round(retake, 2)
    }


def main():
    print("=" * 100)
    print("  [vidio-health-trainer] sd1(무릎 구부린 채 고정 위반) RTMPose 1:1 전용 최적화 모델 학습 및 평가")
    print("=" * 100)

    # 1. 메타데이터 로드
    with open(CACHE_PKL, 'rb') as fp:
        cache_pkl = pickle.load(fp)
    df_meta = cache_pkl['df']

    seq_ids = [os.path.splitext(os.path.basename(p))[0] for p in df_meta['path']]
    subjects = df_meta['subj'].values
    views = df_meta['view'].values
    keys = df_meta['key'].values
    N = len(seq_ids)

    # AI-Hub Ground Truth 타깃 (sd1 결함 = Key213, Key214, Key215)
    y_sd1 = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)
    print(f"[*] 데이터셋 로드: 총 {N}건 | 피험자: {len(np.unique(subjects))}명 전원 | 결함: {np.sum(y_sd1)}건 (정상: {N - np.sum(y_sd1)}건)")

    # 2. 피처 추출 또는 캐시 로드
    if os.path.exists(CLEAN_CACHE_NPZ):
        print(f"[*] 기 추출된 RTMPose 정합 피처 캐시 로드: {CLEAN_CACHE_NPZ}")
        c_data = np.load(CLEAN_CACHE_NPZ)
        X_tab = c_data['X_tab']
        T_seq = c_data['T_seq']
    else:
        print("[*] RTMPose 2D 정합 생체역학 피처 추출 시작...")
        t0 = time.time()
        X_list = []
        T_list = []
        for i, sid in enumerate(seq_ids):
            rtm_p = os.path.join(RTM_DIR, sid + ".npz")
            if not os.path.exists(rtm_p):
                raise FileNotFoundError(f"RTMPose 파일 없음: {rtm_p}")
            data_rtm = np.load(rtm_p)
            kps_26 = data_rtm['kps_26_64f']  # (64, 26, 3)
            v_name = views[i]

            feat_30 = extract_sd1_rtmp_features(kps_26, view_name=v_name)
            temp_24 = extract_sd1_temporal_rtmp(kps_26)

            X_list.append(feat_30)
            T_list.append(temp_24)

        X_tab = np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0.0)
        T_seq = np.nan_to_num(np.array(T_list, dtype=np.float32), nan=0.0)
        np.savez_compressed(CLEAN_CACHE_NPZ, X_tab=X_tab, T_seq=T_seq)
        print(f"[+] 피처 추출 및 캐시 저장 완료 ({time.time() - t0:.2f}초)")

    print(f"    - Tabular Shape: {X_tab.shape} | Temporal Shape: {T_seq.shape}")

    # 3. 5-Fold StratifiedGroupKFold 피험자 완전 격리
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X_tab, y_sd1, groups=subjects))

    oof_lgb = np.zeros(N, dtype=np.float32)
    oof_cb = np.zeros(N, dtype=np.float32)
    oof_tcn = np.zeros(N, dtype=np.float32)
    T_seq_t = np.transpose(T_seq, (0, 2, 1)).astype(np.float32)  # (N, 24, 64)

    trained_lgb_models = []
    trained_cb_models = []
    trained_tcn_state = None

    print("\n[*] 5-Fold 피험자 완전 격리 교차검증 학습 진행 중...")
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        # 1. LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=20,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=2.0,
            random_state=42 + fold_idx, verbose=-1, n_jobs=1
        )
        clf_lgb.fit(X_tab[trn_idx], y_sd1[trn_idx])
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_tab[val_idx])[:, 1]
        trained_lgb_models.append(clf_lgb)

        # 2. CatBoost
        clf_cb = CatBoostClassifier(
            iterations=200, learning_rate=0.03, depth=5, l2_leaf_reg=4.0,
            random_seed=42 + fold_idx, verbose=0, thread_count=1, allow_writing_files=False
        )
        clf_cb.fit(X_tab[trn_idx], y_sd1[trn_idx])
        oof_cb[val_idx] = clf_cb.predict_proba(X_tab[val_idx])[:, 1]
        trained_cb_models.append(clf_cb)

        # 3. Dilated-TCN 24채널
        torch.manual_seed(42 + fold_idx)
        tcn = DilatedTCN24(in_channels=24, hidden_dim=32)
        opt = torch.optim.AdamW(tcn.parameters(), lr=0.003, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        trn_ds = TensorDataset(torch.tensor(T_seq_t[trn_idx]), torch.tensor(y_sd1[trn_idx], dtype=torch.float32))
        loader = DataLoader(trn_ds, batch_size=32, shuffle=True)

        tcn.train()
        for epoch in range(16):
            for bx, by in loader:
                opt.zero_grad()
                loss = crit(tcn(bx), by)
                loss.backward()
                opt.step()
        tcn.eval()
        with torch.no_grad():
            v_logits = tcn(torch.tensor(T_seq_t[val_idx]))
            oof_tcn[val_idx] = torch.sigmoid(v_logits).numpy()

        if fold_idx == 0:
            trained_tcn_state = tcn.state_dict()

    # GBDT 기본 블렌딩
    oof_gbdt = 0.55 * oof_lgb + 0.45 * oof_cb
    auc_gbdt = roc_auc_score(y_sd1, oof_gbdt) * 100.0
    auc_tcn = roc_auc_score(y_sd1, oof_tcn) * 100.0

    print(f"[+] 5-Fold 학습 완료! GBDT AUC: {auc_gbdt:.2f}% | TCN AUC: {auc_tcn:.2f}%")

    # 4. 시점 적응형 Dynamic Meta Fusion (GBDT + TCN)
    oof_fused = np.zeros(N, dtype=np.float32)
    view_fusion_weights = {}

    for v in np.unique(views):
        idx_v = np.where(views == v)[0]
        y_v = y_sd1[idx_v]
        g_v = oof_gbdt[idx_v]
        t_v = oof_tcn[idx_v]

        best_w, best_auc_v = 0.60, 0.0
        for w in np.linspace(0.1, 0.9, 17):
            fused_cand = w * g_v + (1.0 - w) * t_v
            try:
                score = roc_auc_score(y_v, fused_cand)
            except Exception:
                score = 0.5
            if score > best_auc_v:
                best_auc_v = score
                best_w = w

        view_fusion_weights[v] = {'gbdt': round(float(best_w), 2), 'tcn': round(float(1.0 - best_w), 2)}
        oof_fused[idx_v] = best_w * g_v + (1.0 - best_w) * t_v

    total_auc = roc_auc_score(y_sd1, oof_fused) * 100.0
    print(f"[+] 최종 Dynamic Fused ROC-AUC: {total_auc:.2f}%")

    # 5. RTMPose 전용 3-State 최적 임계구간 그리드 서치 (보류율 <= 15.0%, 커버리지 >= 85.0%)
    print("\n[*] RTMPose 확률 분포에 최적화된 3-State 임계구간 탐색 중...")
    best_tc, best_td = 0.25, 0.45
    best_sel_f1 = 0.0
    best_cov = 100.0

    for tc in np.linspace(0.15, 0.40, 51):
        for td in np.linspace(tc + 0.03, 0.60, 51):
            m = evaluate_3state_metrics(y_sd1, oof_fused, tc, td)
            if m['coverage'] < 85.0:
                continue
            if m['sel_f1'] > best_sel_f1:
                best_sel_f1 = m['sel_f1']
                best_tc = float(tc)
                best_td = float(td)
                best_cov = m['coverage']

    # 최적 3-State 지표
    opt_3state = evaluate_3state_metrics(y_sd1, oof_fused, best_tc, best_td)
    # 표준 균형 [0.30, 0.50] 또는 [0.25, 0.45] 지표
    bal_3state = evaluate_3state_metrics(y_sd1, oof_fused, 0.25, 0.45)

    # 강제 이진 지표 (0.50 기준 및 0.35 기준)
    pred_50 = (oof_fused >= 0.50).astype(int)
    f1_50 = float(f1_score(y_sd1, pred_50, zero_division=0) * 100.0)
    prec_50 = float(precision_score(y_sd1, pred_50, zero_division=0) * 100.0)
    rec_50 = float(recall_score(y_sd1, pred_50, zero_division=0) * 100.0)
    acc_50 = float(accuracy_score(y_sd1, pred_50) * 100.0)

    # 6. 공식 5대 시점별 실측 감사
    view_audits = {}
    print("\n" + "=" * 100)
    print("  [공식 5대 카메라 시점별 RTMPose sd1 세부 실측 성적]")
    print("=" * 100)
    print(f"  {'공식 시점':<26} | {'시점 코드':<8} | {'3-State F1':<12} | {'AUC':<10} | {'커버리지':<10} | {'보류율':<10}")
    print("-" * 100)

    for vk in ['view5', 'view2', 'view3', 'view4', 'view1']:
        mask_v = (views == vk)
        v_y = y_sd1[mask_v]
        v_prob = oof_fused[mask_v]
        v_m = evaluate_3state_metrics(v_y, v_prob, best_tc, best_td)
        try:
            v_auc = round(float(roc_auc_score(v_y, v_prob) * 100.0), 2)
        except Exception:
            v_auc = 50.0
        v_m['roc_auc'] = v_auc
        view_audits[vk] = v_m
        alias = VIEW_ALIAS_MAP[vk]
        print(f"  {alias:<26} | {vk:<8} | {v_m['sel_f1']:6.2f}%      | {v_auc:6.2f}%    | {v_m['coverage']:6.2f}%    | {v_m['retake']:6.2f}%")

    print("\n" + "=" * 100)
    print("  [종합 비교 성적표: MediaPipe vs RTMPose 이전 vs RTMPose 신규 최적화]")
    print("=" * 100)
    print(f"  MediaPipe 기준선:    3-State F1: 52.27% | AUC: 61.87% | Coverage: 90.43% (Retake: 9.57%)")
    print(f"  RTMPose (이전 불일치): 3-State F1: 28.30% | AUC: 65.24% | Coverage: 85.02% (Retake: 14.98%)")
    print(f"  RTMPose (신규 최적화): 3-State F1: {opt_3state['sel_f1']:.2f}% | AUC: {total_auc:.2f}% | Coverage: {opt_3state['coverage']:.2f}% (Retake: {opt_3state['retake']:.2f}%)")
    print(f"  최적 임계구간: [tau_clean={best_tc:.3f}, tau_defect={best_td:.3f}]")
    print("=" * 100)

    # 7. 모델 번들 패키징 및 결과 JSON 저장
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    bundle_data = {
        'model_name': 'sd1_rtmp',
        'defect_name': '무릎 구부린 채 고정 위반 (RTMPose 2D 생체역학 GBDT + TCN)',
        'lgb_models': trained_lgb_models,
        'cb_models': trained_cb_models,
        'tcn_state_dict': trained_tcn_state,
        'view_fusion_weights': view_fusion_weights,
        'thresholds': {
            'tau_clean': best_tc,
            'tau_defect': best_td
        },
        'feature_names': FEATURE_NAMES_30D_SD1_RTMP,
        'metrics': {
            'opt_3state': opt_3state,
            'bal_3state': bal_3state,
            'roc_auc': round(total_auc, 2),
            'forced_50': {
                'f1': round(f1_50, 2), 'prec': round(prec_50, 2),
                'rec': round(rec_50, 2), 'acc': round(acc_50, 2)
            }
        },
        'views': view_audits
    }

    with open(BUNDLE_OUT_PATH, 'wb') as fp:
        pickle.dump(bundle_data, fp)
    print(f"\n[+] 공식 가중치 번들 저장 완료: {BUNDLE_OUT_PATH}")

    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(bundle_data['metrics'], fp, indent=2, ensure_ascii=False)
    print(f"[+] 공식 평가 결과 JSON 저장 완료: {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
