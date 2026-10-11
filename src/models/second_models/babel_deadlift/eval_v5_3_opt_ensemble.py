# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_v5_3_opt_ensemble.py
Purpose: 피처 다이어트로 도출된 17D 정예 피처셋 + ST-SeqNet 57D
         v5.3-Opt 최종 앙상블 5-Fold 피험자 완전 격리 실측 평가
"""

import os
import sys

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

from src.models.second_models.babel_deadlift.train_and_eval_d3_v5_3 import (
    prepare_v5_3_clean_dataset,
    find_best_view_threshold_f1_priority,
    VIEW_CODE_LIST,
    VIEW_ALIAS_MAP
)
from src.models.second_models.babel_deadlift.d3_model_v5_3 import DeadliftSTSequenceNetV5_3

# 최적 17D 정예 피처 인덱스
OPTIMAL_17_INDICES = [1, 4, 5, 6, 7, 8, 9, 11, 13, 19, 20, 21, 23, 24, 27, 29, 30]

def run_opt_evaluation():
    print("=" * 80)
    print("vidio-health-trainer 바벨 데드리프트 d3 v5.3-Opt (17D 정예셋 + ST-SeqNet 57D) 평가")
    print("=" * 80)

    X_seq_57, X_feats_32, y, subjects, views, clip_names = prepare_v5_3_clean_dataset()
    X_feats_17 = X_feats_32[:, OPTIMAL_17_INDICES]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_tree_17 = np.zeros(len(y), dtype=np.float32)
    oof_dl = np.zeros(len(y), dtype=np.float32)

    print(f"\n[*] 5-Fold 피험자 완전 격리 학습 개시 (피처 차원: 32D -> 17D 다이어트 정예화)...")
    for fold, (tr, val) in enumerate(sgkf.split(X_feats_17, y, groups=subjects)):
        pos_w = float(np.sum(y[tr] == 0)) / max(1, float(np.sum(y[tr] == 1)))

        # 1. Tree 17D 정예 학습
        cb = CatBoostClassifier(iterations=350, depth=4, learning_rate=0.03, auto_class_weights='Balanced', verbose=False, random_seed=42+fold)
        cb.fit(X_feats_17[tr], y[tr])
        p_cb = cb.predict_proba(X_feats_17[val])[:, 1]

        ds = lgb.Dataset(X_feats_17[tr], label=y[tr])
        params = {'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42+fold}
        lgb_m = lgb.train(params, ds, num_boost_round=230)
        p_lgb = lgb_m.predict(X_feats_17[val])

        rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42+fold, n_jobs=-1)
        rf.fit(X_feats_17[tr], y[tr])
        p_rf = rf.predict_proba(X_feats_17[val])[:, 1]

        oof_tree_17[val] = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf

        # 2. ST-SeqNet 57D (Clean KAPW)
        net = DeadliftSTSequenceNetV5_3(in_features=57, hidden_dim=64)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w_t = torch.tensor([pos_w])
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)

        t_x = torch.tensor(X_seq_57[tr], dtype=torch.float32)
        t_y = torch.tensor(y[tr], dtype=torch.float32)

        net.train()
        for ep in range(35):
            indices = np.random.permutation(len(tr))
            for i in range(0, len(tr), 32):
                b = indices[i:i+32]
                opt.zero_grad()
                logits = net(t_x[b])
                loss = bce(logits, t_y[b])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            oof_dl[val] = torch.sigmoid(net(torch.tensor(X_seq_57[val], dtype=torch.float32))).numpy()

    print(f"[+] Tree 17D OOF AUC: {roc_auc_score(y, oof_tree_17):.4f}")
    print(f"[+] ST-SeqNet 57D OOF AUC: {roc_auc_score(y, oof_dl):.4f}")

    # 시점 적응형 앙상블 블렌드 (후방 0.55 DL, 전방/정면 0.40 DL)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        v = views[i]
        w_dl = 0.55 if v in ['view5', 'view1'] else 0.40
        oof_probs[i] = w_dl * oof_dl[i] + (1.0 - w_dl) * oof_tree_17[i]

    total_auc = float(roc_auc_score(y, oof_probs))
    print(f"[+] 최종 통합 앙상블 전체 ROC-AUC: {total_auc:.4f}")

    # 3-State 안전 게이트 최적화 (보류율 <= 10.0% 엄격)
    print("\n" + "=" * 80)
    print("5대 공식 시점별 3-State 안전 게이트 실측 결과 (보류율 <= 10.0% 엄격)")
    print("=" * 80)

    view_results = {}
    det_mask_total = np.zeros(len(y), dtype=bool)
    pred_total = np.zeros(len(y), dtype=int)

    for vcode in VIEW_CODE_LIST:
        v_idx = np.where(views == vcode)[0]
        y_v = y[v_idx]
        p_v = oof_probs[v_idx]

        tn, td, metrics = find_best_view_threshold_f1_priority(y_v, p_v, min_coverage=0.90)

        det_v = (p_v <= tn) | (p_v >= td)
        det_mask_total[v_idx] = det_v
        pred_total[v_idx] = np.where(p_v >= td, 1, 0)

        alias = VIEW_ALIAS_MAP[vcode]
        view_results[vcode] = {
            'alias': alias,
            'total_samples': len(y_v),
            'defect_samples': int(np.sum(y_v == 1)),
            'normal_samples': int(np.sum(y_v == 0)),
            't_normal': float(tn),
            't_defect': float(td),
            'coverage': metrics['coverage'],
            'abstain_rate': metrics['abstain_rate'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': float(roc_auc_score(y_v, p_v))
        }

        print(f"[{alias}]")
        print(f"  - 임계값: 정상 <= {tn:.3f}, 결함 >= {td:.3f} (보류: {tn:.3f} ~ {td:.3f})")
        print(f"  - 커버리지: {metrics['coverage']*100:.2f}% (보류율: {metrics['abstain_rate']*100:.2f}%)")
        print(f"  - 확정 F1: {metrics['f1']*100:.2f}% | Precision: {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | AUC: {view_results[vcode]['roc_auc']:.4f}")

    y_det = y[det_mask_total]
    pred_det = pred_total[det_mask_total]

    overall_cov = float(np.mean(det_mask_total))
    overall_abstain = float(1.0 - overall_cov)
    overall_f1 = float(f1_score(y_det, pred_det, zero_division=0))
    overall_acc = float(accuracy_score(y_det, pred_det))
    overall_prec = float(precision_score(y_det, pred_det, zero_division=0))
    overall_rec = float(recall_score(y_det, pred_det, zero_division=0))
    cm = confusion_matrix(y_det, pred_det).tolist()

    print("\n" + "=" * 80)
    print("★ [v5.3-Opt (17D 다이어트) 최종 확정 성능 요약] ★")
    print(f"  - 전체 확정 F1-Score: {overall_f1*100:.2f}%")
    print(f"  - 전체 확정 Recall: {overall_rec*100:.2f}% | Precision: {overall_prec*100:.2f}%")
    print(f"  - 전체 확정 Accuracy: {overall_acc*100:.2f}% | ROC-AUC: {total_auc:.4f}")
    print(f"  - 전체 유효 커버리지: {overall_cov*100:.2f}% (판단 보류율: {overall_abstain*100:.2f}% <= 10.0% 엄격 충족)")
    print(f"  - 혼동 행렬 (TN, FP / FN, TP): {cm}")
    print("=" * 80)

    # JSON 저장
    opt_summary = {
        'model_version': 'v5.3-Opt',
        'n_features': 17,
        'feature_indices': OPTIMAL_17_INDICES,
        'overall_metrics': {
            'coverage': overall_cov,
            'abstain_rate': overall_abstain,
            'f1': overall_f1,
            'accuracy': overall_acc,
            'precision': overall_prec,
            'recall': overall_rec,
            'roc_auc': total_auc,
            'confusion_matrix': cm
        },
        'view_metrics': view_results
    }
    with open(os.path.join(CURRENT_DIR, "d3_v5_3_opt_evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(opt_summary, f, indent=2, ensure_ascii=False)
    print("[+] v5.3-Opt 결과 저장 완료")

if __name__ == '__main__':
    try:
        run_opt_evaluation()
    except Exception as e:
        import traceback
        traceback.print_exc()
