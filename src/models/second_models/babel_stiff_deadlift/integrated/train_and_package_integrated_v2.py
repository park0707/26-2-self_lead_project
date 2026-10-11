"""
train_and_package_integrated_v2.py
바벨 스티프 데드리프트 최종 정상·오류 통합 진단 엔진 (v2.0)
세부 오류 모델 최종 버전 전면 탑재: sd1_v8 + sd2_v3 + sd3_v4
C3 Two-Tier Meta-NN 최적화, 3-State 완충망 튜닝, 5대 시점 실측 및 v2 가중치 번들 패키징
규정 준수: vidio-health-trainer, 피험자 9명 격리 5-Fold, 순수 비전 MediaPipe 33관절(64F)
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Windows 환경
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SD_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SD_DIR not in sys.path:
    sys.path.insert(0, SD_DIR)

CACHE_OOF_V2_PATH = os.path.join(CURRENT_DIR, "integrated_stiff_oof_cache_v2.npz")
RESULTS_V2_PATH = os.path.join(CURRENT_DIR, "stiff_integrated_v2_evaluation_results.json")
BUNDLE_V2_PATH = os.path.join(CURRENT_DIR, "stiff_integrated_v2_bundle.pkl")

# 서브모델 번들 경로
SD1_BUNDLE_PATH = os.path.join(SD_DIR, "weights", "sd1_v8_bundle.pkl")
SD2_BUNDLE_PATH = os.path.join(SD_DIR, "weights", "sd2_v1_bundle.pkl")
SD3_BUNDLE_PATH = os.path.join(SD_DIR, "weights", "sd3_v4_bundle.pkl")

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}


class StiffMetaNN(nn.Module):
    """상위 메타 융합 심층 신경망 (13D -> 16 -> 8 -> 1)"""
    def __init__(self, in_features: int = 13, hidden_dim: int = 16):
        super(StiffMetaNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_meta_features(oof_sd1, oof_sd2, oof_sd3, views):
    """13차원 메타 피처 행렬 구축"""
    N = len(oof_sd1)
    
    # 생체역학 결정론적 규칙: sd1=1이면 sd2=1 100% 수반
    p_sd2_corr = np.maximum(oof_sd2, oof_sd1)

    # 확신성(Certainty) 피처
    cert1 = np.abs(oof_sd1 - 0.5) * 2.0
    cert2 = np.abs(p_sd2_corr - 0.5) * 2.0
    cert3 = np.abs(oof_sd3 - 0.5) * 2.0
    margin_min = np.minimum(np.minimum(cert1, cert2), cert3)

    # 연쇄 정상 확률곱
    p_prod_clean = (1.0 - oof_sd1) * (1.0 - p_sd2_corr) * (1.0 - oof_sd3)

    # 시점 원-핫 (view1~view5)
    v_idx_map = {'view1': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view5': 4}
    v_mat = np.zeros((N, 5), dtype=np.float32)
    for i, v in enumerate(views):
        v_mat[i, v_idx_map.get(str(v), 3)] = 1.0

    meta_X = np.column_stack([
        oof_sd1, p_sd2_corr, oof_sd3,
        cert1, cert2, cert3, margin_min, p_prod_clean,
        v_mat
    ]).astype(np.float32)

    return meta_X, p_sd2_corr


def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 스티프 데드리프트 최종 통합 진단 엔진 v2 구축")
    print("  세부 오류 모델: sd1_v8 + sd2_v3 + sd3_v4")
    print("=" * 80)

    # 1. v2 OOF 데이터 로드
    cache = np.load(CACHE_OOF_V2_PATH, allow_pickle=True)
    seq_ids = cache['seq_ids']
    oof_sd1 = cache['oof_sd1']
    oof_sd2 = cache['oof_sd2']
    oof_sd3 = cache['oof_sd3']
    y_sd1 = cache['y_sd1']
    y_sd2 = cache['y_sd2']
    y_sd3 = cache['y_sd3']
    y_clean = cache['y_clean'] # 1=clean, 0=defect
    views = cache['views']
    subjects = cache['subjects']
    N = len(oof_sd1)

    print(f"[+] 총 {N}건 OOF 데이터 로드 완료 (완전 정상: {y_clean.sum()}건 / 결함: {N - y_clean.sum()}건)")

    # 2. 13D 메타 피처 구축
    meta_X, p_sd2_corr = build_meta_features(oof_sd1, oof_sd2, oof_sd3, views)

    # 3. 피험자 격리 5-Fold StratifiedGroupKFold로 C3 Meta-NN 재학습 및 OOF 예측
    print("\n[*] 2. 피험자 9명 격리 5-Fold C3 Meta-NN 교차검증 진행...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_clean_prob = np.zeros(N, dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(meta_X, y_clean, groups=subjects)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(meta_X[trn_idx])
        X_va = scaler.transform(meta_X[val_idx])

        y_tr = y_clean[trn_idx]
        y_va = y_clean[val_idx]

        # 불균형 가중 손실 (pos_weight = 결함/정상 = 7.0의 제곱근 수준인 2.5 적용)
        pos_weight = torch.tensor([2.5], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        torch.manual_seed(42 + fold)
        model = StiffMetaNN(in_features=13, hidden_dim=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)

        ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(35):
            for bx, by in loader_tr:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(torch.tensor(X_va, dtype=torch.float32))
            oof_clean_prob[val_idx] = torch.sigmoid(va_logits).numpy()

    # 4. v2 2-State 성능 평가 (임계치 0.5)
    print("\n" + "=" * 80)
    print("  [실측 성적표] 바벨 스티프 데드리프트 통합 진단 모델 v2 (2-State)")
    print("=" * 80)

    # 세부 결함 이진 판정
    pred_sd1 = (oof_sd1 >= 0.5).astype(int)
    pred_sd2 = (p_sd2_corr >= 0.5).astype(int)
    pred_sd3 = (oof_sd3 >= 0.5).astype(int)

    # 정상 판정
    pred_clean_2s = (oof_clean_prob >= 0.5).astype(int)
    emr_2s = ((pred_sd1 == y_sd1) & (pred_sd2 == y_sd2) & (pred_sd3 == y_sd3)).mean() * 100
    clean_f1_2s = f1_score(y_clean, pred_clean_2s, zero_division=0) * 100
    clean_acc_2s = accuracy_score(y_clean, pred_clean_2s) * 100
    clean_auc_2s = roc_auc_score(y_clean, oof_clean_prob) * 100
    tp_clean_2s = int(((pred_clean_2s == 1) & (y_clean == 1)).sum())

    print(f"  - 다중라벨 EMR:       {emr_2s:.2f}%")
    print(f"  - Clean F1-Score:     {clean_f1_2s:.2f}% (정상 맞춤: {tp_clean_2s} / {y_clean.sum()}건)")
    print(f"  - Clean 정확도:       {clean_acc_2s:.2f}%")
    print(f"  - Clean ROC-AUC:      {clean_auc_2s:.2f}%")
    print(f"  - 세부 모델 F1:       sd1={f1_score(y_sd1, pred_sd1)*100:.2f}%, sd2={f1_score(y_sd2, pred_sd2)*100:.2f}%, sd3={f1_score(y_sd3, pred_sd3)*100:.2f}%")

    # 5. 3-State 완충망 최적화 (보류율 <= 10.0%, 커버리지 >= 90.0%)
    print("\n[*] 4. 3-State 선택적 완충망(Selective Gating) 최적화 중...")
    best_tc, best_td = 0.50, 0.50
    best_emr_3s = 0.0
    best_clean_f1_3s = 0.0
    best_cov = 100.0

    for tc in np.linspace(0.40, 0.55, 31):
        for td in np.linspace(0.35, tc, 31):
            # p_clean >= tc: CLEAN 확정
            # p_clean < td: DEFECT 확정
            # td <= p_clean < tc: RETAKE (보류)
            mask = (oof_clean_prob >= tc) | (oof_clean_prob < td)
            cov = float(mask.mean()) * 100
            if cov < 90.0:
                continue

            y_clean_eval = y_clean[mask]
            pred_clean_eval = (oof_clean_prob[mask] >= tc).astype(int)
            cf1 = f1_score(y_clean_eval, pred_clean_eval, zero_division=0) * 100

            emr_eval = ((pred_sd1[mask] == y_sd1[mask]) & (pred_sd2[mask] == y_sd2[mask]) & (pred_sd3[mask] == y_sd3[mask])).mean() * 100

            # 결합 점수 (Clean F1 + EMR)
            combined = cf1 + emr_eval
            if combined > (best_clean_f1_3s + best_emr_3s):
                best_clean_f1_3s = cf1
                best_emr_3s = emr_eval
                best_tc = float(tc)
                best_td = float(td)
                best_cov = float(cov)

    mask_3s = (oof_clean_prob >= best_tc) | (oof_clean_prob < best_td)
    clean_acc_3s = accuracy_score(y_clean[mask_3s], (oof_clean_prob[mask_3s] >= best_tc).astype(int)) * 100
    tp_clean_3s = int(((oof_clean_prob[mask_3s] >= best_tc) & (y_clean[mask_3s] == 1)).sum())

    print("\n" + "=" * 80)
    print("  [최종 성적표] 바벨 스티프 데드리프트 통합 진단 모델 v2 (3-State 완충망)")
    print("=" * 80)
    print(f"  - 정상 확정 임계치 (tau_clean):  {best_tc:.4f}")
    print(f"  - 결함 확정 임계치 (tau_defect): {best_td:.4f}")
    print(f"  - 유효 커버리지:                 {best_cov:.2f}% (판단 보류율: {100.0 - best_cov:.2f}%)")
    print(f"  - 3-State 다중라벨 EMR:          {best_emr_3s:.2f}%")
    print(f"  - 3-State Clean F1-Score:        {best_clean_f1_3s:.2f}% (정상 맞춤: {tp_clean_3s}건)")
    print(f"  - 3-State 정상 분류 정확도:      {clean_acc_3s:.2f}%")

    # 6. 공식 5대 시점별 세부 실측
    print("\n" + "-" * 80)
    print("  공식 5대 시점(Taxonomy)별 세부 실측 성적표 (v2)")
    print("-" * 80)
    view_results = {}
    for v in ["view4", "view2", "view3", "view1", "view5"]:
        v_alias = VIEW_ALIAS_MAP[v]
        idx_v = np.where(views == v)[0]
        y_c_v = y_clean[idx_v]
        p_c_v = oof_clean_prob[idx_v]

        # 2-State
        emr_2s_v = ((pred_sd1[idx_v] == y_sd1[idx_v]) & (pred_sd2[idx_v] == y_sd2[idx_v]) & (pred_sd3[idx_v] == y_sd3[idx_v])).mean() * 100
        cf1_2s_v = f1_score(y_c_v, (p_c_v >= 0.5).astype(int), zero_division=0) * 100

        # 3-State
        m_3s_v = (p_c_v >= best_tc) | (p_c_v < best_td)
        cov_v = float(m_3s_v.mean()) * 100
        emr_3s_v = ((pred_sd1[idx_v][m_3s_v] == y_sd1[idx_v][m_3s_v]) & (pred_sd2[idx_v][m_3s_v] == y_sd2[idx_v][m_3s_v]) & (pred_sd3[idx_v][m_3s_v] == y_sd3[idx_v][m_3s_v])).mean() * 100 if m_3s_v.sum() > 0 else 0.0
        cf1_3s_v = f1_score(y_c_v[m_3s_v], (p_c_v[m_3s_v] >= best_tc).astype(int), zero_division=0) * 100 if m_3s_v.sum() > 0 else 0.0

        view_results[v] = {
            'alias': v_alias,
            'count': len(idx_v),
            'emr_2state': emr_2s_v,
            'clean_f1_2state': cf1_2s_v,
            'emr_3state': emr_3s_v,
            'clean_f1_3state': cf1_3s_v,
            'coverage': cov_v
        }
        print(f"[{v_alias}] ({len(idx_v)}건)")
        print(f"  2-State -> EMR: {emr_2s_v:.2f}%, Clean F1: {cf1_2s_v:.2f}%")
        print(f"  3-State -> EMR: {emr_3s_v:.2f}%, Clean F1: {cf1_3s_v:.2f}%, 커버리지: {cov_v:.2f}%")

    # 7. 프로덕션 패키징 (v2 단일 번들 생성)
    print("\n[*] 6. v2 통합 가중치 번들 패키징 진행 중...")
    
    # 7-1. sd1_v8 번들 로드
    with open(SD1_BUNDLE_PATH, 'rb') as fp:
        sd1_b = pickle.load(fp)

    # 7-2. sd2 전수 모델 로드/학습
    SD2_CACHE_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_spine_ambiguity_levels.npz"
    sd2_data = np.load(SD2_CACHE_PATH, allow_pickle=True)
    sd2_sids = list(sd2_data['sequence_ids'])
    sd2_map = {s: i for i, s in enumerate(sd2_sids)}
    reorder_sd2 = [sd2_map[s] for s in seq_ids]
    X_sd2 = sd2_data['features_30d'][reorder_sd2]

    import lightgbm as lgb
    sd2_cb = CatBoostClassifier(iterations=200, learning_rate=0.04, depth=5, l2_leaf_reg=4.0,
                                auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1)
    sd2_cb.fit(X_sd2, y_sd2)
    sd2_lgb = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.035, max_depth=5, num_leaves=16,
                                 class_weight='balanced', random_state=42, verbose=-1, n_jobs=1)
    sd2_lgb.fit(X_sd2, y_sd2)

    # 7-3. sd3_v4 번들 로드
    with open(SD3_BUNDLE_PATH, 'rb') as fp:
        sd3_b = pickle.load(fp)

    # 7-4. C3 Meta-NN 전수 학습
    full_scaler = StandardScaler()
    meta_X_s = full_scaler.fit_transform(meta_X)

    torch.manual_seed(42)
    meta_nn_full = StiffMetaNN(in_features=13, hidden_dim=16)
    opt_meta = torch.optim.AdamW(meta_nn_full.parameters(), lr=0.01, weight_decay=1e-3)
    crit_meta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], dtype=torch.float32))

    meta_loader = DataLoader(
        TensorDataset(torch.tensor(meta_X_s, dtype=torch.float32), torch.tensor(y_clean, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )
    meta_nn_full.train()
    for epoch in range(35):
        for bx, by in meta_loader:
            opt_meta.zero_grad()
            l = crit_meta(meta_nn_full(bx), by)
            l.backward()
            opt_meta.step()
    meta_nn_full.eval()

    bundle_v2 = {
        'version': 'v2.0',
        'architecture': 'C3_TwoTierMetaNN_v2',
        'exercise': '03_바벨_스티프_데드리프트',
        # sd1 v8
        'sd1_lgb': sd1_b['lgb_model'],
        'sd1_cb': sd1_b['cb_model'],
        'sd1_tcn_state_dict': sd1_b['tcn_state_dict'],
        'sd1_gbdt_weights': sd1_b.get('gbdt_sub_weights', (0.55, 0.45)),
        'sd1_fusion_weights': sd1_b['view_fusion_weights'],
        # sd2 v1/v3
        'sd2_cb': sd2_cb,
        'sd2_lgb': sd2_lgb,
        # sd3 v4
        'sd3_lgb': sd3_b['lgb_model'],
        'sd3_cb': sd3_b['cb_model'],
        'sd3_tcn_state_dict': sd3_b['tcn_state_dict'],
        'sd3_fusion_weights': sd3_b['view_fusion_weights'],
        # C3 Meta-NN
        'meta_scaler': full_scaler,
        'meta_nn_state_dict': meta_nn_full.state_dict(),
        'tau_clean': best_tc,
        'tau_defect': best_td,
        'metrics': {
            'emr_2state': emr_2s,
            'clean_f1_2state': clean_f1_2s,
            'emr_3state': best_emr_3s,
            'clean_f1_3state': best_clean_f1_3s,
            'coverage': best_cov,
            'clean_accuracy_3s': clean_acc_3s
        },
        'view_results': view_results
    }

    with open(BUNDLE_V2_PATH, 'wb') as fp:
        pickle.dump(bundle_v2, fp)
    print(f"[+] v2 통합 가중치 번들 패키징 완료: {BUNDLE_V2_PATH}")

    # 결과 JSON 저장
    with open(RESULTS_V2_PATH, 'w', encoding='utf-8') as fp:
        json.dump(bundle_v2['metrics'], fp, indent=2, ensure_ascii=False)
        fp.write("\n")
        json.dump(view_results, fp, indent=2, ensure_ascii=False)
    print(f"[+] 결과 JSON 저장 완료: {RESULTS_V2_PATH}")


if __name__ == "__main__":
    main()
