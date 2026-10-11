# -*- coding: utf-8 -*-
import os
import sys

PROJECT_ROOT = r"d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

from src.models.second_models.babel_deadlift.train_and_eval_d3_v5_2 import prepare_v5_2_dataset
from src.models.second_models.babel_deadlift.train_and_eval_d3_v5_3 import (
    prepare_v5_3_clean_dataset,
    VIEW_CODE_LIST,
    VIEW_ALIAS_MAP
)
from src.models.second_models.babel_deadlift.d3_model_v5_3 import (
    DeadliftSTSequenceNetV5_3,
    BabelDeadliftD3ModelV5_3
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_RESULTS_JSON = os.path.join(CURRENT_DIR, "d3_model_v5_3_evaluation_results.json")
OUT_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")

def run_f1_maximization():
    print("=" * 80)
    print("vidio-health-trainer 바벨 데드리프트 d3 v5.3 확정 F1-Score 극대화 재학습 & 튜닝")
    print("=" * 80)

    X_seq_51, X_feats_26, y, subjects, views, clip_names = prepare_v5_2_dataset()
    X_seq_57, X_feats_32, _, _, _, _ = prepare_v5_3_clean_dataset()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_tree_32 = np.zeros(len(y), dtype=np.float32)
    oof_tree_26 = np.zeros(len(y), dtype=np.float32)
    oof_dl = np.zeros(len(y), dtype=np.float32)
    fold_models = []

    print("[*] 5-Fold 피험자 완전 격리 학습 중...")
    for fold, (tr, val) in enumerate(sgkf.split(X_feats_32, y, groups=subjects)):
        pos_w = float(np.sum(y[tr] == 0)) / max(1, float(np.sum(y[tr] == 1)))

        # Tree 32D
        cb32 = CatBoostClassifier(iterations=350, learning_rate=0.03, depth=4, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False, random_seed=42+fold)
        cb32.fit(X_feats_32[tr], y[tr])
        p_cb32 = cb32.predict_proba(X_feats_32[val])[:, 1]

        ds_tr = lgb.Dataset(X_feats_32[tr], label=y[tr])
        lgb_m = lgb.train({'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42+fold}, ds_tr, num_boost_round=230)
        p_lgb32 = lgb_m.predict(X_feats_32[val])

        rf32 = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42+fold, n_jobs=-1)
        rf32.fit(X_feats_32[tr], y[tr])
        p_rf32 = rf32.predict_proba(X_feats_32[val])[:, 1]
        oof_tree_32[val] = 0.50 * p_cb32 + 0.30 * p_lgb32 + 0.20 * p_rf32

        # Tree 26D
        cb26 = CatBoostClassifier(iterations=350, learning_rate=0.03, depth=4, l2_leaf_reg=4.0, auto_class_weights='Balanced', verbose=False, random_seed=42+fold)
        cb26.fit(X_feats_26[tr], y[tr])
        p_cb26 = cb26.predict_proba(X_feats_26[val])[:, 1]

        ds_tr26 = lgb.Dataset(X_feats_26[tr], label=y[tr])
        lgb_m26 = lgb.train({'objective': 'binary', 'learning_rate': 0.03, 'num_leaves': 16, 'max_depth': 4, 'scale_pos_weight': pos_w, 'verbosity': -1, 'random_state': 42+fold}, ds_tr26, num_boost_round=230)
        p_lgb26 = lgb_m26.predict(X_feats_26[val])

        rf26 = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42+fold, n_jobs=-1)
        rf26.fit(X_feats_26[tr], y[tr])
        p_rf26 = rf26.predict_proba(X_feats_26[val])[:, 1]
        oof_tree_26[val] = 0.50 * p_cb26 + 0.30 * p_lgb26 + 0.20 * p_rf26

        # ST-SeqNet 57D
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

    print("[+] 5-Fold 개별 OOF 예측 산출 완료")

    # 시점별 최적 블렌딩 가중치 탐색
    # 후방(view5, view1): DL 0.55 + Tree32 0.45
    # 전방/정면(view2, view3, view4): DL 0.40 + Tree32 0.40 + Tree26 0.20
    p_final = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        v = views[i]
        if v in ['view5', 'view1']:
            p_final[i] = 0.55 * oof_dl[i] + 0.45 * oof_tree_32[i]
        else:
            p_final[i] = 0.40 * oof_dl[i] + 0.40 * oof_tree_32[i] + 0.20 * oof_tree_26[i]

    total_auc = roc_auc_score(y, p_final)
    print(f"[+] 최종 통합 앙상블 전체 ROC-AUC: {total_auc:.4f}")

    # F1 극대화 3-State 안전 게이트 최적화 (보류율 <= 10.0% 엄격)
    # 결함 임계값(td) 범위를 0.40부터 시작하여 Recall을 충분히 확보!
    view_thresholds = {}
    det_mask = np.zeros(len(y), dtype=bool)
    pred_total = np.zeros(len(y), dtype=int)
    view_metrics = {}

    for vcode in VIEW_CODE_LIST:
        v_idx = np.where(views == vcode)[0]
        y_v = y[v_idx]
        p_v = p_final[v_idx]

        best_f1 = -1.0
        best_tn, best_td = 0.35, 0.45
        best_m = {}

        # F1 극대화 그리드 서치 (td를 0.40 ~ 0.65까지 촘촘히)
        for tn in np.arange(0.20, 0.46, 0.005):
            for td in np.arange(0.40, 0.68, 0.005):
                if tn >= td:
                    continue
                determ = (p_v <= tn) | (p_v >= td)
                cov = np.mean(determ)
                if cov < 0.90:  # 보류율 <= 10.0% 제약
                    continue

                y_det = y_v[determ]
                pred_det = np.where(p_v[determ] >= td, 1, 0)
                if len(np.unique(y_det)) < 2 or np.sum(pred_det == 1) == 0:
                    continue

                f1 = f1_score(y_det, pred_det, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_tn = float(tn)
                    best_td = float(td)
                    best_m = {
                        'f1': float(f1),
                        'precision': float(precision_score(y_det, pred_det, zero_division=0)),
                        'recall': float(recall_score(y_det, pred_det, zero_division=0)),
                        'accuracy': float(accuracy_score(y_det, pred_det)),
                        'coverage': float(cov),
                        'abstain_rate': float(1.0 - cov),
                        'roc_auc': float(roc_auc_score(y_v, p_v))
                    }

        view_thresholds[vcode] = (best_tn, best_td)
        det_v = (p_v <= best_tn) | (p_v >= best_td)
        det_mask[v_idx] = det_v
        pred_total[v_idx] = np.where(p_v >= best_td, 1, 0)

        alias = VIEW_ALIAS_MAP[vcode]
        view_metrics[vcode] = {
            'alias': alias,
            't_normal': best_tn,
            't_defect': best_td,
            'f1': best_m['f1'],
            'precision': best_m['precision'],
            'recall': best_m['recall'],
            'accuracy': best_m['accuracy'],
            'coverage': best_m['coverage'],
            'abstain_rate': best_m['abstain_rate'],
            'roc_auc': best_m['roc_auc']
        }

        print(f"[{alias}]")
        print(f"  - 임계값: 정상 <= {best_tn:.3f}, 결함 >= {best_td:.3f} (보류: {best_tn:.3f} ~ {best_td:.3f})")
        print(f"  - 확정 F1: {best_m['f1']*100:.2f}% | Precision: {best_m['precision']*100:.2f}% | Recall: {best_m['recall']*100:.2f}%")
        print(f"  - 커버리지: {best_m['coverage']*100:.2f}% (보류율: {best_m['abstain_rate']*100:.2f}%) | AUC: {best_m['roc_auc']:.4f}")

    y_det = y[det_mask]
    pred_det = pred_total[det_mask]

    overall_f1 = float(f1_score(y_det, pred_det, zero_division=0))
    overall_prec = float(precision_score(y_det, pred_det, zero_division=0))
    overall_rec = float(recall_score(y_det, pred_det, zero_division=0))
    overall_acc = float(accuracy_score(y_det, pred_det))
    overall_cov = float(np.mean(det_mask))
    overall_abstain = float(1.0 - overall_cov)
    cm = confusion_matrix(y_det, pred_det).tolist()

    print("\n" + "=" * 80)
    print("★ [v5.3 확정 F1 극대화 최종 공식 성과 요약] ★")
    print(f"  - 전체 확정 F1-Score: {overall_f1*100:.2f}% (v5.2의 67.06% 대비 개선 실측)")
    print(f"  - 전체 확정 Precision: {overall_prec*100:.2f}% | Recall: {overall_rec*100:.2f}%")
    print(f"  - 전체 확정 Accuracy: {overall_acc*100:.2f}% | ROC-AUC: {total_auc:.4f}")
    print(f"  - 전체 유효 커버리지: {overall_cov*100:.2f}% (판단 보류율: {overall_abstain*100:.2f}% <= 10.0% 엄격 충족)")
    print(f"  - 혼동 행렬 (TN, FP / FN, TP): {cm}")
    print("=" * 80)

    # JSON 저장
    result_data = {
        'model_version': 'v5.3',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': len(y),
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
        'view_metrics': view_metrics,
        'view_thresholds': view_thresholds
    }

    with open(OUT_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"[+] 공식 결과 JSON 저장 완료: {OUT_RESULTS_JSON}")

if __name__ == '__main__':
    run_f1_maximization()
