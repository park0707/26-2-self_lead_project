# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 통합 (05_스텝_포워드_다이나믹_런지 + 06_스텝_백워드_다이나믹_런지)
Defect: c4 (척추의 중립 위반 / Spine Neutrality Defect)
Model Name: dl4_v1 (바벨 런지 bl4 24D+14ch 아키텍처 계승)
Script: train_and_eval_dl4.py
Purpose:
  1. 다이나믹 런지 c4 데이터셋 대상 학습 및 평가:
     - MultiScale-SE-TCN: 14D 순수 시계열 텐서 학습
     - GBDT 3총사(LightGBM, XGBoost, CatBoost): 24D 정예 도메인 피처 학습
  2. 5-Fold StratifiedGroupKFold 피험자 9명 완전 격리 (Data Leakage 0.00%)
  3. 전체 데이터셋 대상 3-State Conformal Selective F1-Score 실측 및 임계치 최적화
  4. AGENTS.md 규격 공식 5대 시점(시점 A~E / view1~5) 전수 세부 성능 감사
  5. 배포용 최적화 모델 번들(dl4_weights.pkl) 및 평가 결과 json 저장
"""

import os
import sys
import time
import json
import pickle
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dl4 import MultiScaleSETCN, MultiScaleConvBlock, SEBlock1D

CACHE_PATH = r"D:\Data\피처_데이터\dynamic_lunge_c4_dataset_cache.npz"
OUTPUT_JSON_PATH = os.path.join(CURRENT_DIR, "dl4_evaluation_results.json")
BUNDLE_PATH = os.path.join(CURRENT_DIR, "dl4_weights.pkl")

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.45, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def evaluate_3state_conformal(probs: np.ndarray, labels: np.ndarray, tau_clean: float, tau_def: float) -> Dict[str, Any]:
    preds_3state = []
    det_mask = []

    for p in probs:
        if p <= tau_clean:
            preds_3state.append(0)
            det_mask.append(True)
        elif p >= tau_def:
            preds_3state.append(1)
            det_mask.append(True)
        else:
            preds_3state.append(-1)
            det_mask.append(False)

    preds_3state = np.array(preds_3state)
    det_mask = np.array(det_mask)
    n_total = len(labels)
    n_det = int(np.sum(det_mask))
    coverage = n_det / n_total if n_total > 0 else 0.0
    abstain_rate = 1.0 - coverage

    if n_det > 0:
        y_true_det = labels[det_mask]
        y_pred_det = preds_3state[det_mask]
        f1 = float(f1_score(y_true_det, y_pred_det, pos_label=1, zero_division=0))
        prec = float(precision_score(y_true_det, y_pred_det, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true_det, y_pred_det, pos_label=1, zero_division=0))
        acc = float(accuracy_score(y_true_det, y_pred_det))
    else:
        f1, prec, rec, acc = 0.0, 0.0, 0.0, 0.0

    score = f1 * (coverage ** 0.5)

    return {
        'score': score,
        'tau_clean': tau_clean,
        'tau_def': tau_def,
        'coverage': coverage,
        'abstain_rate': abstain_rate,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'accuracy': acc,
        'n_determined': n_det,
        'n_abstained': n_total - n_det,
        'preds_3state': preds_3state
    }


def find_optimal_thresholds(probs: np.ndarray, labels: np.ndarray, max_abstain: float = 0.10) -> Dict[str, Any]:
    min_cov = 1.0 - max_abstain
    best_res = None
    best_f1 = -1.0
    best_acc = -1.0

    candidates = []
    for tc in np.linspace(0.32, 0.48, 33):
        for td in np.linspace(0.48, 0.64, 33):
            if tc >= td:
                continue
            res = evaluate_3state_conformal(probs, labels, tc, td)
            if res['coverage'] >= min_cov:
                candidates.append(res)
                if (res['f1'] > best_f1) or (abs(res['f1'] - best_f1) < 1e-4 and res['accuracy'] > best_acc):
                    best_f1 = res['f1']
                    best_acc = res['accuracy']
                    best_res = res

    if best_res is None:
        best_res = evaluate_3state_conformal(probs, labels, 0.42, 0.52)

    candidates_sorted = sorted(candidates, key=lambda x: (x['f1'], x['accuracy']), reverse=True)
    print("\n[Threshold Tuning] 상위 5대 임계치 후보 (보류율 <= 10.0%):")
    for i, c in enumerate(candidates_sorted[:5]):
        print(f"  Top {i+1}: tc={c['tau_clean']:.3f}, td={c['tau_def']:.3f} | Cov={c['coverage']*100:.2f}% (Abstain={c['abstain_rate']*100:.2f}%) | ★ F1={c['f1']*100:.2f}%, Prec={c['precision']*100:.2f}%, Rec={c['recall']*100:.2f}%, Acc={c['accuracy']*100:.2f}%")

    return best_res


def train_and_eval_dl4():
    print("=" * 85)
    print("  [vidio-health-trainer] 다이나믹 런지 c4 세부 오류 모델 dl4 학습 및 성능 평가")
    print("=" * 85)

    t_start = time.time()

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"캐시 파일이 존재하지 않습니다: {CACHE_PATH}")

    data = np.load(CACHE_PATH, allow_pickle=True)
    X_24d = data['X_24d']
    X_temp_14d = data['X_temp_14d']
    y = data['y']
    subjects = data['subjects']
    views = data['views']
    feature_names = list(data['feature_names'])

    N = len(y)
    n_def = int(np.sum(y == 1))
    n_clean = int(np.sum(y == 0))
    print(f"[*] 총 데이터 수: {N}개 (정상: {n_clean}개, 결함: {n_def}개, 결함비: {n_def/N*100:.2f}%)")
    print(f"[*] GBDT 피처 차원: {X_24d.shape[1]}차원 (정예 24D)")
    print(f"[*] TCN 시계열 차원: {X_temp_14d.shape[2]}차원 (14ch 시계열 텐서)")
    print(f"[*] 고유 피험자 수: {len(np.unique(subjects))}명 ({list(np.unique(subjects))})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] TCN 학습 디바이스: {device}")

    # 5-Fold StratifiedGroupKFold 설정
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_tcn_probs = np.zeros(N, dtype=np.float32)
    oof_lgb_probs = np.zeros(N, dtype=np.float32)
    oof_xgb_probs = np.zeros(N, dtype=np.float32)
    oof_cat_probs = np.zeros(N, dtype=np.float32)

    fold_models = {
        'tcn': [],
        'lgb': [],
        'xgb': [],
        'cat': [],
        'scaler': []
    }

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_24d, y, groups=subjects), 1):
        tr_subj = np.unique(subjects[train_idx])
        va_subj = np.unique(subjects[val_idx])
        print(f"\n--- [Fold {fold}/5] Train: {len(train_idx)}건 (피험자 {len(tr_subj)}명) | Val: {len(val_idx)}건 (피험자 {len(va_subj)}명) ---")

        # 정규화
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_24d[train_idx])
        X_va_sc = scaler.transform(X_24d[val_idx])
        fold_models['scaler'].append(scaler)

        y_tr, y_va = y[train_idx], y[val_idx]
        T_tr, T_va = X_temp_14d[train_idx], X_temp_14d[val_idx]

        # A. MultiScale-SE-TCN
        tcn_model = MultiScaleSETCN(in_features=14, num_classes=2).to(device)
        criterion = FocalLoss(alpha=0.45, gamma=1.5)
        optimizer = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)

        ds_tr = TensorDataset(torch.from_numpy(T_tr).float(), torch.from_numpy(y_tr).long())
        loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)

        best_tcn_f1 = -1.0
        best_tcn_state = None

        for epoch in range(1, 36):
            tcn_model.train()
            for bx, by in loader_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = tcn_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            tcn_model.eval()
            with torch.no_grad():
                val_x = torch.from_numpy(T_va).float().to(device)
                v_out = tcn_model(val_x)
                v_prob = F.softmax(v_out, dim=-1)[:, 1].cpu().numpy()
                v_pred = (v_prob >= 0.5).astype(int)
                f1_ep = f1_score(y_va, v_pred, pos_label=1, zero_division=0)
                if f1_ep > best_tcn_f1:
                    best_tcn_f1 = f1_ep
                    best_tcn_state = {k: v.cpu().clone() for k, v in tcn_model.state_dict().items()}

        tcn_model.load_state_dict(best_tcn_state)
        tcn_model.eval()
        with torch.no_grad():
            v_prob = F.softmax(tcn_model(torch.from_numpy(T_va).float().to(device)), dim=-1)[:, 1].cpu().numpy()
        oof_tcn_probs[val_idx] = v_prob
        fold_models['tcn'].append(tcn_model.cpu())

        # B. LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=0.3,
            class_weight='balanced',
            random_state=42 + fold,
            verbose=-1
        )
        lgb_model.fit(X_tr_sc, y_tr)
        p_lgb = lgb_model.predict_proba(X_va_sc)[:, 1]
        oof_lgb_probs[val_idx] = p_lgb
        fold_models['lgb'].append(lgb_model)

        # C. XGBoost
        pos_weight = float(np.sum(y_tr == 0)) / float(max(1, np.sum(y_tr == 1)))
        xgb_model = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.80,
            gamma=0.15,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=42 + fold
        )
        xgb_model.fit(X_tr_sc, y_tr)
        p_xgb = xgb_model.predict_proba(X_va_sc)[:, 1]
        oof_xgb_probs[val_idx] = p_xgb
        fold_models['xgb'].append(xgb_model)

        # D. CatBoost
        cat_model = CatBoostClassifier(
            iterations=120,
            learning_rate=0.04,
            depth=4,
            l2_leaf_reg=3.0,
            auto_class_weights='Balanced',
            verbose=0,
            random_seed=42 + fold
        )
        cat_model.fit(X_tr_sc, y_tr)
        p_cat = cat_model.predict_proba(X_va_sc)[:, 1]
        oof_cat_probs[val_idx] = p_cat
        fold_models['cat'].append(cat_model)

        print(f"  Fold {fold} F1: TCN={best_tcn_f1*100:.1f}%, LGB={f1_score(y_va, (p_lgb>=0.5).astype(int))*100:.1f}%, XGB={f1_score(y_va, (p_xgb>=0.5).astype(int))*100:.1f}%, CAT={f1_score(y_va, (p_cat>=0.5).astype(int))*100:.1f}%")

    t_elapsed = time.time() - t_start
    print(f"\n[*] 5-Fold 교차검증 완료 (소요시간: {t_elapsed:.1f}초)")

    # 최적 앙상블 블렌딩 가중치 탐색
    best_weights = (0.25, 0.25, 0.25, 0.25)
    best_ens_f1 = -1.0

    weight_candidates = [
        (0.25, 0.25, 0.25, 0.25),
        (0.40, 0.20, 0.20, 0.20),
        (0.10, 0.30, 0.30, 0.30),
        (0.30, 0.25, 0.25, 0.20),
        (0.20, 0.25, 0.25, 0.30),
        (0.35, 0.25, 0.20, 0.20),
        (0.15, 0.35, 0.25, 0.25)
    ]

    for (wt, wl, wx, wc) in weight_candidates:
        p_comb = wt * oof_tcn_probs + wl * oof_lgb_probs + wx * oof_xgb_probs + wc * oof_cat_probs
        f1_c = f1_score(y, (p_comb >= 0.5).astype(int), pos_label=1, zero_division=0)
        if f1_c > best_ens_f1:
            best_ens_f1 = f1_c
            best_weights = (wt, wl, wx, wc)

    wt, wl, wx, wc = best_weights
    print(f"\n[*] 최적 블렌딩 가중치: TCN={wt:.2f}, LGB={wl:.2f}, XGB={wx:.2f}, CAT={wc:.2f} (2-State F1={best_ens_f1*100:.2f}%)")

    oof_final_probs = wt * oof_tcn_probs + wl * oof_lgb_probs + wx * oof_xgb_probs + wc * oof_cat_probs
    auc_total = float(roc_auc_score(y, oof_final_probs))

    # 2-State 기준 지표
    bin_preds = (oof_final_probs >= 0.50).astype(int)
    f1_2s = float(f1_score(y, bin_preds, pos_label=1, zero_division=0))
    prec_2s = float(precision_score(y, bin_preds, pos_label=1, zero_division=0))
    rec_2s = float(recall_score(y, bin_preds, pos_label=1, zero_division=0))
    acc_2s = float(accuracy_score(y, bin_preds))

    # 3-State Conformal 최적화
    opt_3state = find_optimal_thresholds(oof_final_probs, y, max_abstain=0.10)
    tc = opt_3state['tau_clean']
    td = opt_3state['tau_def']

    # 서브모델별 개별 AUC/F1
    sub_metrics = {
        'TCN': {
            'auc': float(roc_auc_score(y, oof_tcn_probs)),
            'f1': float(f1_score(y, (oof_tcn_probs >= 0.5).astype(int), pos_label=1, zero_division=0)),
            'acc': float(accuracy_score(y, (oof_tcn_probs >= 0.5).astype(int)))
        },
        'LightGBM': {
            'auc': float(roc_auc_score(y, oof_lgb_probs)),
            'f1': float(f1_score(y, (oof_lgb_probs >= 0.5).astype(int), pos_label=1, zero_division=0)),
            'acc': float(accuracy_score(y, (oof_lgb_probs >= 0.5).astype(int)))
        },
        'XGBoost': {
            'auc': float(roc_auc_score(y, oof_xgb_probs)),
            'f1': float(f1_score(y, (oof_xgb_probs >= 0.5).astype(int), pos_label=1, zero_division=0)),
            'acc': float(accuracy_score(y, (oof_xgb_probs >= 0.5).astype(int)))
        },
        'CatBoost': {
            'auc': float(roc_auc_score(y, oof_cat_probs)),
            'f1': float(f1_score(y, (oof_cat_probs >= 0.5).astype(int), pos_label=1, zero_division=0)),
            'acc': float(accuracy_score(y, (oof_cat_probs >= 0.5).astype(int)))
        }
    }

    # 5대 시점별 세부 실측
    view_performance = {}
    for vn, alias in VIEW_ALIAS_MAP.items():
        v_idx = np.where(views == vn)[0]
        if len(v_idx) == 0:
            continue
        v_y = y[v_idx]
        v_probs = oof_final_probs[v_idx]

        # 2-State
        v_bin = (v_probs >= 0.50).astype(int)
        v_f1_2s = float(f1_score(v_y, v_bin, pos_label=1, zero_division=0))
        v_acc_2s = float(accuracy_score(v_y, v_bin))
        try:
            v_auc = float(roc_auc_score(v_y, v_probs))
        except:
            v_auc = 0.5

        # 3-State
        v_3s = evaluate_3state_conformal(v_probs, v_y, tc, td)

        view_performance[vn] = {
            'alias': alias,
            'n_samples': len(v_idx),
            'n_defect': int(np.sum(v_y == 1)),
            'auc': v_auc,
            'f1_2state': v_f1_2s,
            'acc_2state': v_acc_2s,
            'f1_3state': v_3s['f1'],
            'acc_3state': v_3s['accuracy'],
            'coverage': v_3s['coverage'],
            'abstain_rate': v_3s['abstain_rate']
        }

    # 피험자별 세부 실측
    subj_performance = {}
    for subj in np.unique(subjects):
        s_idx = np.where(subjects == subj)[0]
        s_y = y[s_idx]
        s_probs = oof_final_probs[s_idx]

        s_bin = (s_probs >= 0.50).astype(int)
        s_f1_2s = float(f1_score(s_y, s_bin, pos_label=1, zero_division=0))
        s_acc_2s = float(accuracy_score(s_y, s_bin))
        try:
            s_auc = float(roc_auc_score(s_y, s_probs))
        except:
            s_auc = 0.5

        s_3s = evaluate_3state_conformal(s_probs, s_y, tc, td)

        subj_performance[str(subj)] = {
            'n_samples': len(s_idx),
            'n_defect': int(np.sum(s_y == 1)),
            'auc': s_auc,
            'f1_2state': s_f1_2s,
            'acc_2state': s_acc_2s,
            'f1_3state': s_3s['f1'],
            'acc_3state': s_3s['accuracy'],
            'coverage': s_3s['coverage'],
            'abstain_rate': s_3s['abstain_rate']
        }

    # 결과 종합
    evaluation_summary = {
        'model_name': 'dl4_v1',
        'exercise': '다이나믹 런지 (05_포워드 + 06_백워드 통합)',
        'defect': 'c4 (척추의 중립 위반)',
        'n_total': N,
        'n_defect': n_def,
        'n_clean': n_clean,
        'weights': {'w_tcn': wt, 'w_lgb': wl, 'w_xgb': wx, 'w_cat': wc},
        'thresholds': {'tau_clean': tc, 'tau_def': td, 'tau_binary': 0.50},
        'overall_metrics_2state': {
            'auc': auc_total,
            'f1': f1_2s,
            'precision': prec_2s,
            'recall': rec_2s,
            'accuracy': acc_2s
        },
        'overall_metrics_3state': {
            'f1': opt_3state['f1'],
            'precision': opt_3state['precision'],
            'recall': opt_3state['recall'],
            'accuracy': opt_3state['accuracy'],
            'coverage': opt_3state['coverage'],
            'abstain_rate': opt_3state['abstain_rate'],
            'n_determined': opt_3state['n_determined'],
            'n_abstained': opt_3state['n_abstained']
        },
        'submodel_metrics': sub_metrics,
        'view_performance': view_performance,
        'subj_performance': subj_performance,
        'feature_names': feature_names
    }

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[+] 평가 결과 저장 완료: {OUTPUT_JSON_PATH}")

    # 가중치 번들 저장
    bundle_data = {
        'feature_names': feature_names,
        'weights': {'w_tcn': wt, 'w_lgb': wl, 'w_xgb': wx, 'w_cat': wc},
        'thresholds': {'tau_clean': tc, 'tau_def': td, 'tau_binary': 0.50},
        'fold_models': fold_models
    }
    with open(BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle_data, f)
    print(f"[+] 모델 가중치 번들 저장 완료: {BUNDLE_PATH}")

    # 최종 결과 출력
    print("\n" + "=" * 85)
    print("  [다이나믹 런지 dl4 최종 종합 성능 요약]")
    print("=" * 85)
    print(f"  * 2-State AUC: {auc_total*100:.2f}% | F1: {f1_2s*100:.2f}% | Acc: {acc_2s*100:.2f}% | Prec: {prec_2s*100:.2f}% | Rec: {rec_2s*100:.2f}%")
    print(f"  * 3-State Conformal Selective (tc={tc:.3f}, td={td:.3f}):")
    print(f"    - Coverage: {opt_3state['coverage']*100:.2f}% (판정보류율 Abstain: {opt_3state['abstain_rate']*100:.2f}%)")
    print(f"    - ★ Selective F1-Score: {opt_3state['f1']*100:.2f}% (정밀도: {opt_3state['precision']*100:.2f}%, 재현율: {opt_3state['recall']*100:.2f}%)")
    print(f"    - ★ Selective Accuracy: {opt_3state['accuracy']*100:.2f}%")
    print("\n  [4대 서브모델별 단독 기여도 (OOF AUC)]")
    for sm, m in sub_metrics.items():
        print(f"    - {sm:10s}: AUC = {m['auc']*100:.2f}%, F1 = {m['f1']*100:.2f}%, Acc = {m['acc']*100:.2f}%")
    print("\n  [AGENTS.md 공식 5대 시점 전수 성능]")
    for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
        if vn in view_performance:
            vp = view_performance[vn]
            print(f"    - {vp['alias']:24s} (N={vp['n_samples']:3d}): AUC={vp['auc']*100:.1f}%, 2S-F1={vp['f1_2state']*100:.1f}%, ★ 3S-F1={vp['f1_3state']*100:.1f}%, Cov={vp['coverage']*100:.1f}%")
    print("=" * 85)


if __name__ == '__main__':
    train_and_eval_dl4()
