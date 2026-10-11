# -*- coding: utf-8 -*-
import sys, os, json
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pickle
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

eval_json_p = os.path.join(CURRENT_DIR, "d5_model_v7_two_tier_evaluation_results.json")
with open(eval_json_p, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

cache = np.load(os.path.join(CURRENT_DIR, "d5_features_37_v6_cache.npz"), allow_pickle=True)
amb = np.load(os.path.join(CURRENT_DIR, "d5_ambiguity_levels.npz"), allow_pickle=True)
l4_mask = amb['l4_mask']

clean_mask = (~l4_mask)
X = cache['X'][clean_mask]
y = cache['y'][clean_mask]
groups = cache['groups'][clean_mask]
views = cache['views'][clean_mask]

sgkf = StratifiedGroupKFold(n_splits=5)
weights_dir = os.path.join(CURRENT_DIR, "weights")

oof_cascade = np.zeros(len(y), dtype=np.float32)
oof_t2 = np.zeros(len(y), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    w_p = os.path.join(weights_dir, f"d5_v7_two_tier_fold{fold+1}.pkl")
    with open(w_p, "rb") as f:
        fold_obj = pickle.load(f)
    t1_m = fold_obj['tier1_model']
    t2_m = fold_obj['tier2_model']
    p1 = t1_m.predict_proba(X[val_idx])
    p2 = t2_m.predict_proba(X[val_idx])

    p_c = np.zeros(len(val_idx), dtype=np.float32)
    for i in range(len(val_idx)):
        if p1[i] <= 0.25:
            p_c[i] = p1[i]
        elif p1[i] >= 0.75:
            p_c[i] = p1[i]
        else:
            p_c[i] = 0.30 * p1[i] + 0.70 * p2[i]
    oof_cascade[val_idx] = p_c
    oof_t2[val_idx] = p2

opt_th = eval_data['optimal_thresholds_level123']

cal_probs = np.zeros(len(y), dtype=np.float32)
for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
    mask_v = (views == vn)
    th = opt_th[vn]
    p_v = oof_cascade[mask_v]
    p_cal = np.where(p_v >= th, 0.50 + 0.50 * (p_v - th) / (1.0 - th + 1e-6), 0.50 * p_v / (th + 1e-6))
    cal_probs[mask_v] = p_cal

# 기준 마진 0.12 적용 3-State 시점별 성능
decided = np.zeros(len(y), dtype=bool)
for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
    mask_v = (views == vn)
    p_cal_v = cal_probs[mask_v]
    if vn == 'view4':
        m_v = 0.01
    elif vn == 'view3':
        m_v = 0.17
    elif vn in ['view5', 'view1']:
        m_v = 0.14
    else:
        m_v = 0.12
    dec_v = (p_cal_v < 0.50 - m_v) | (p_cal_v >= 0.50 + m_v)
    decided[mask_v] = dec_v

VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}

print("=" * 85)
print("  [vidio-health-trainer] d5 v7.0 Two-Tier 3-State 공식 성능 감사")
print("=" * 85)

cov_total = np.mean(decided) * 100
y_dec = y[decided]
p_dec = (cal_probs[decided] >= 0.50).astype(int)

f1_tot = f1_score(y_dec, p_dec, zero_division=0) * 100
clean_f1_tot = f1_score(1 - y_dec, 1 - p_dec, zero_division=0) * 100
acc_tot = accuracy_score(y_dec, p_dec) * 100
prec_tot = precision_score(y_dec, p_dec, zero_division=0) * 100
rec_tot = recall_score(y_dec, p_dec, zero_division=0) * 100
cm_tot = confusion_matrix(y_dec, p_dec)

print(f"\n[*] 3-State 전체 성능 (커버리지 {cov_total:.1f}%, 보류율 {100-cov_total:.1f}%):")
print(f"    - Defect F1-Score : ★ {f1_tot:.2f}%")
print(f"    - Clean F1-Score  : ★ {clean_f1_tot:.2f}%")
print(f"    - 전체 정확도(Acc) : ★ {acc_tot:.2f}%")
print(f"    - 결함 정밀도(Prec): {prec_tot:.2f}%")
print(f"    - 결함 재현율(Rec) : {rec_tot:.2f}%")
print(f"    - 보류 샘플 수    : {int(np.sum(~decided))}건 / {len(y)}건")
print(f"    - 혼동 행렬:\n      정상예측  결함예측\n      [[ {cm_tot[0][0]:4d}    {cm_tot[0][1]:4d} ]  (실제 정상)\n       [ {cm_tot[1][0]:4d}    {cm_tot[1][1]:4d} ]] (실제 결함)")

print("\n[*] 5대 공식 시점별 3-State 세부 성능:")
for vn in ['view5', 'view2', 'view3', 'view4', 'view1']:
    mask_v = (views == vn)
    mask_dec_v = mask_v & decided
    cov_v = np.sum(mask_dec_v) / np.sum(mask_v) * 100
    y_v = y[mask_dec_v]
    p_v = (cal_probs[mask_dec_v] >= 0.50).astype(int)
    f1_v = f1_score(y_v, p_v, zero_division=0) * 100
    clean_f1_v = f1_score(1 - y_v, 1 - p_v, zero_division=0) * 100
    acc_v = accuracy_score(y_v, p_v) * 100
    print(f"  ■ {VIEW_ALIAS_MAP[vn]:<22}: Defect F1={f1_v:5.2f}% | Clean F1={clean_f1_v:5.2f}% | Acc={acc_v:5.2f}% | 커버리지={cov_v:5.1f}% ({np.sum(mask_dec_v)}/{np.sum(mask_v)}건)")
