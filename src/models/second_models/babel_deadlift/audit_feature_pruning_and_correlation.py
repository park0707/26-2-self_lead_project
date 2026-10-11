# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: audit_feature_pruning_and_correlation.py
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Submodel: d2 (발과 무릎의 방향 일치 / 동적 무릎 외반 Knee Valgus)
Purpose: 
  v5.0 25개 피처의 다중공선성(Correlation > 0.85) 전수 조사 및
  5-Fold StratifiedGroupKFold 기반 LOFO(Leave-One-Feature-Out) 기여도 평가
  -> F1/AUC를 저해하거나 노이즈를 유발하는 피처 전수 식별
"""

import os
import sys
import json
import glob
import time
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, 'D:/')
from src.models.second_models.babel_deadlift.d2_model_v5_final import (
    extract_d2_v5_final_features, FEATURE_NAMES_25_D2
)

CACHE_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\02_바벨_데드리프트_test"
LABEL_JSON_DIR = r"D:\Data\01_raw_data\labels_json\02_바벨_데드리프트\2D"

def run_feature_audit():
    print("=" * 115)
    print("  [vidio-health-trainer] d2 v5.0 25D 피처셋 정밀 감사 및 다중공선성/저해 피처 색출")
    print("  - 데이터셋: AI-Hub 385개 시퀀스 전수 (피험자 1~9번 전원, 5대 전 시점)")
    print("  - 검증 프로토콜: 5-Fold StratifiedGroupKFold (9명 피험자 완전 격리)")
    print("=" * 115)

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    X_list, y_list, subs, views = [], [], [], []

    for p in files:
        base = os.path.basename(p).replace(".npz", "")
        v_code = base.split("_")[-1]
        views.append(v_code)

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
        f25, _, _ = extract_d2_v5_final_features(kps)
        X_list.append(f25)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    subs = np.array(subs)
    views = np.array(views)

    print(f"[+] 25D 피처 매트릭스 로드 완료: {X.shape}")

    # 1. 다중공선성 (Correlation > 0.85) 쌍 식별
    print("\n" + "=" * 80)
    print("  [1] 초고상관(High Multicollinearity, |r| >= 0.85) 피처 쌍 전수 조사")
    print("=" * 80)
    corr = np.corrcoef(X, rowvar=False)
    high_corr_pairs = []
    for i in range(len(FEATURE_NAMES_25_D2)):
        for j in range(i + 1, len(FEATURE_NAMES_25_D2)):
            r_val = corr[i, j]
            if abs(r_val) >= 0.85:
                high_corr_pairs.append((FEATURE_NAMES_25_D2[i], FEATURE_NAMES_25_D2[j], r_val))
                print(f"  • {FEATURE_NAMES_25_D2[i]:<30} <---> {FEATURE_NAMES_25_D2[j]:<30} : r = {r_val:+.4f}")

    print(f"\n  [!] 총 {len(high_corr_pairs)}개의 고상관 피처 쌍 발견 (중복 정보 및 다중공선성 노이즈 유발)")

    # 2. 베이스라인 성능 (25D 전수)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs(r"C:\Temp\catboost_info", exist_ok=True)

    def eval_features(feat_indices):
        oof_p = np.zeros(len(y), dtype=np.float32)
        X_sub = X[:, feat_indices]
        for tr_idx, val_idx in sgkf.split(X_sub, y, groups=subs):
            X_tr, y_tr = X_sub[tr_idx], y[tr_idx]
            X_val = X_sub[val_idx]
            clf_lgb = lgb.LGBMClassifier(
                n_estimators=130, learning_rate=0.03, max_depth=4, num_leaves=12,
                subsample=0.85, colsample_bytree=0.80, reg_alpha=0.15, reg_lambda=2.0,
                scale_pos_weight=3.28, random_state=42, verbose=-1, n_jobs=1
            )
            clf_lgb.fit(X_tr, y_tr)
            p_l = clf_lgb.predict_proba(X_val)[:, 1]

            clf_cat = CatBoostClassifier(
                iterations=160, learning_rate=0.03, depth=4, l2_leaf_reg=4.0,
                auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=1,
                train_dir=r"C:\Temp\catboost_info"
            )
            clf_cat.fit(X_tr, y_tr)
            p_c = clf_cat.predict_proba(X_val)[:, 1]
            oof_p[val_idx] = 0.5 * p_l + 0.5 * p_c

        auc = roc_auc_score(y, oof_p)
        # Search best 3-State F1
        best_f1 = 0.0
        best_prec = 0.0
        best_rec = 0.0
        best_win = None
        for tn in np.arange(0.20, 0.40, 0.005):
            for td in np.arange(tn + 0.005, 0.50, 0.005):
                dec = (oof_p <= tn) | (oof_p >= td)
                if np.mean(dec) < 0.90: continue
                yy = y[dec]
                pred = (oof_p[dec] >= td).astype(int)
                f1 = f1_score(yy, pred, zero_division=0) * 100
                if f1 > best_f1:
                    best_f1 = f1
                    best_prec = precision_score(yy, pred, zero_division=0) * 100
                    best_rec = recall_score(yy, pred, zero_division=0) * 100
                    best_win = [round(float(tn), 3), round(float(td), 3)]
        return auc, best_f1, best_prec, best_rec, best_win

    print("\n" + "=" * 80)
    print("  [2] 베이스라인 25D 전수 성능 평가 (GBDT OOF)")
    print("=" * 80)
    all_indices = list(range(25))
    base_auc, base_f1, base_p, base_r, base_win = eval_features(all_indices)
    print(f"  • 베이스라인 25D -> ROC-AUC: {base_auc:.4f} | 3-State F1: {base_f1:.2f}% (Prec: {base_p:.2f}%, Rec: {base_r:.2f}%, Window: {base_win})")

    # 3. Leave-One-Feature-Out (LOFO) 정밀 평가
    print("\n" + "=" * 80)
    print("  [3] LOFO (Leave-One-Feature-Out) 피처별 단독 제거 영향도 평가")
    print("      (특정 피처를 뺐을 때 AUC/F1이 오히려 상승하면 -> '저해/노이즈 피처'로 판정!)")
    print("=" * 80)

    lofo_results = []
    for drop_idx in range(25):
        drop_name = FEATURE_NAMES_25_D2[drop_idx]
        sub_indices = [i for i in range(25) if i != drop_idx]
        auc_drop, f1_drop, p_drop, r_drop, win_drop = eval_features(sub_indices)
        delta_auc = auc_drop - base_auc
        delta_f1 = f1_drop - base_f1
        lofo_results.append({
            'index': drop_idx,
            'name': drop_name,
            'auc_without': auc_drop,
            'f1_without': f1_drop,
            'delta_auc': delta_auc,
            'delta_f1': delta_f1
        })
        status = "❌ 저해 피처 (제거 시 성능 상승!)" if delta_f1 > 0 or delta_auc > 0.005 else "✅ 필수 피처 (제거 시 성능 하락)"
        print(f"  [{drop_idx+1:2d}/25] {drop_name:<30} 제외 시: AUC={auc_drop:.4f} ({delta_auc:+.4f}) | F1={f1_drop:5.2f}% ({delta_f1:+5.2f}%p) -> {status}")

    # 저해 피처 랭킹
    harmful_features = [r for r in lofo_results if r['delta_f1'] > 0 or r['delta_auc'] > 0]
    harmful_features = sorted(harmful_features, key=lambda x: (x['delta_f1'], x['delta_auc']), reverse=True)

    print("\n" + "=" * 80)
    print("  [4] F1-Score를 저해하고 있던 노이즈/역효과 피처 순위")
    print("=" * 80)
    for h in harmful_features:
        print(f"  • {h['name']:<30}: 제거 시 F1 {h['delta_f1']:+5.2f}%p 상승! (제거 후 F1 {h['f1_without']:.2f}%), AUC {h['delta_auc']:+.4f}")

    # 결과를 JSON으로 저장
    out_json = r"D:\src\models\second_models\babel_deadlift\weights\d2_lofo_audit_results.json"
    with open(out_json, 'w', encoding='utf-8') as fp:
        json.dump({
            'base_auc': base_auc, 'base_f1': base_f1,
            'high_corr_pairs': high_corr_pairs,
            'lofo_results': lofo_results,
            'harmful_features': harmful_features
        }, fp, ensure_ascii=False, indent=2)
    print(f"\n[+] LOFO 감사 결과 JSON 저장 완료: {out_json}")

if __name__ == "__main__":
    run_feature_audit()
