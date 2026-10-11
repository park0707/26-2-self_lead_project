# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: eval_observable_clean_v7_1.py
Purpose: Step 1 - 해결책 2 실측 검증: 5대 안전 가드레일 기반 Observable Clean 및 정상 정의 삼원화 시뮬레이션
         - 1. Verified Clean: 전면/측면 4부위 전수 검증 정상
         - 2. Observable Clean: 후방 5대 가드레일 조건부 정상 (시선 제외 고지)
         - 3. Total Combined Clean (Verified + Observable) 성능 및 안전성 정밀 측정
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

META_PATH = r"D:\Data\피처_데이터\meta_fusion_v7_dataset.npz"

print("=" * 115)
print("  [Step 1] 해결책 2 실측 시뮬레이션: 5대 가드레일 기반 Observable Clean 검증")
print("=" * 115)

data = np.load(META_PATH, allow_pickle=True)
probs_defect = data['probs_defect'] # [p1, p2, p3, p4]
c1_statuses = data['c1_statuses']
views = data['views']
subjects = data['subjects']
y_all = data['y_all']               # (2605, 4)
true_clean = data['true_clean']     # (2605,) 135건
N = len(true_clean)

p_c1 = probs_defect[:, 0]
p_c2 = probs_defect[:, 1]
p_c3 = probs_defect[:, 2]
p_c4 = probs_defect[:, 3]

# -------------------------------------------------------------
# 1. 5대 안전 가드레일 기반 시뮬레이션 그리드 탐색
# -------------------------------------------------------------
# 파라미터 탐색:
# tau_safe_normal: 후방에서 c1, c3, c4가 만족해야 하는 고신뢰 정상 임계치 (0.25, 0.30, 0.35, 0.40)
# tau_c2_defect_max: 후방에서 c2 결함 신호가 없어야 하는 상한치 (0.50, 0.55, 0.60)
# tau_clean_4part: 전면/측면에서 4대 부위 결합 정상 확률 임계치 (0.45)
# tau_clean_3part: 후방에서 3대 부위 결합 정상 확률 임계치 (0.45, 0.50, 0.55)

tau_clean_4part = 0.45

results = []

for tau_safe_normal in [0.25, 0.30, 0.35, 0.40]:
    for tau_c2_defect_max in [0.50, 0.55, 0.60]:
        for tau_clean_3part in [0.45, 0.50, 0.55]:
            
            is_verified = np.zeros(N, dtype=int)
            is_observable = np.zeros(N, dtype=int)
            is_total_clean = np.zeros(N, dtype=int)
            
            for i in range(N):
                is_rear = views[i] in ['view1', 'view5']
                
                # 4대 부위 결합 정상 확률
                p_cln_4 = (1.0 - p_c1[i]) * (1.0 - p_c2[i]) * (1.0 - p_c3[i]) * (1.0 - p_c4[i])
                
                # 1) Verified Clean (전면/사선 시점 중심)
                if (not is_rear) and (p_cln_4 >= tau_clean_4part):
                    is_verified[i] = 1
                    is_total_clean[i] = 1
                elif is_rear:
                    # 2) Observable Clean (후방 5대 가드레일)
                    # 조건 1: c1, c3, c4 결함 확률 각각 <= tau_safe_normal
                    cond_sub_safe = (p_c1[i] <= tau_safe_normal) and (p_c3[i] <= tau_safe_normal) and (p_c4[i] <= tau_safe_normal)
                    # 조건 2: c2 결함 신호 부재 (p_c2 < tau_c2_defect_max)
                    cond_c2_safe = (p_c2[i] < tau_c2_defect_max)
                    # 조건 3: 3개 부위 결합 정상 확률 >= tau_clean_3part
                    p_cln_3 = (1.0 - p_c1[i]) * (1.0 - p_c3[i]) * (1.0 - p_c4[i])
                    cond_prob_3 = (p_cln_3 >= tau_clean_3part)
                    
                    if cond_sub_safe and cond_c2_safe and cond_prob_3:
                        is_observable[i] = 1
                        is_total_clean[i] = 1
                        
            tp_tot = int(np.sum((is_total_clean == 1) & (true_clean == 1)))
            fp_tot = int(np.sum((is_total_clean == 1) & (true_clean == 0)))
            
            tp_ver = int(np.sum((is_verified == 1) & (true_clean == 1)))
            fp_ver = int(np.sum((is_verified == 1) & (true_clean == 0)))
            
            tp_obs = int(np.sum((is_observable == 1) & (true_clean == 1)))
            fp_obs = int(np.sum((is_observable == 1) & (true_clean == 0)))
            
            pre_tot = precision_score(true_clean, is_total_clean, zero_division=0) * 100
            rec_tot = recall_score(true_clean, is_total_clean, zero_division=0) * 100
            f1_tot = f1_score(true_clean, is_total_clean, zero_division=0) * 100
            fp_rate_tot = (fp_tot / (N - 135)) * 100
            
            results.append({
                'tau_safe_normal': tau_safe_normal,
                'tau_c2_defect_max': tau_c2_defect_max,
                'tau_clean_3part': tau_clean_3part,
                'tp_tot': tp_tot,
                'fp_tot': fp_tot,
                'tp_ver': tp_ver,
                'fp_ver': fp_ver,
                'tp_obs': tp_obs,
                'fp_obs': fp_obs,
                'pre_tot': pre_tot,
                'rec_tot': rec_tot,
                'f1_tot': f1_tot,
                'fp_rate_tot': fp_rate_tot,
                'is_verified': is_verified,
                'is_observable': is_observable,
                'is_total_clean': is_total_clean
            })

# F1 기준 정렬
results.sort(key=lambda x: x['f1_tot'], reverse=True)

print(f"{'가드레일 설정 (tau_safe / tau_c2 / tau_3p)':<42} | {'Verified':>10} | {'Observable':>12} | {'Total TP':>8} | {'Total FP':>8} | {'Precision':>9} | {'Recall':>8} | {'F1-Score':>8} | {'오승인율':>8}")
print("-" * 115)

for r in results[:10]:
    cfg_str = f"safe={r['tau_safe_normal']:.2f}, c2<{r['tau_c2_defect_max']:.2f}, 3p>={r['tau_clean_3part']:.2f}"
    ver_str = f"{r['tp_ver']}(FP {r['fp_ver']})"
    obs_str = f"{r['tp_obs']}(FP {r['fp_obs']})"
    print(f"{cfg_str:<42} | {ver_str:>10} | {obs_str:>12} | {r['tp_tot']:>8d} | {r['fp_tot']:>8d} | {r['pre_tot']:8.2f}% | {r['rec_tot']:7.2f}% | {r['f1_tot']:7.2f}% | {r['fp_rate_tot']:7.2f}%")
print("=" * 115)

# 최적 설정 선택 (F1 상위 중 오승인율 균형점)
best = results[0]
for r in results:
    if r['fp_tot'] <= 45 and r['f1_tot'] > 50.0:
        best = r
        break

print(f"\n[최적 v7.1 가드레일 파라미터 확정]")
print(f"  - 후방 3부위(척추·무릎·발) 고신뢰 정상 임계치: p <= {best['tau_safe_normal']}")
print(f"  - 후방 c2 결함 신호 배제 임계치              : p_c2 < {best['tau_c2_defect_max']}")
print(f"  - 후방 3부위 결합 정상 확률 임계치          : P_clean(3p) >= {best['tau_clean_3part']}")
print(f"  - 전면/사선 4부위 결합 정상 확률 임계치       : P_clean(4p) >= {tau_clean_4part}")
print("-" * 115)
print(f"  [실측 성과 요약]")
print(f"  * Verified Clean (전수 검증) : 적중 {best['tp_ver']}건 / 결함 오인 {best['fp_ver']}건")
print(f"  * Observable Clean (후방 관측): 적중 {best['tp_obs']}건 / 결함 오인 {best['fp_obs']}건")
print(f"  * Total Clean 적중 (TP)      : {best['tp_tot']}건 (기존 v6.0 19건 대비 +{best['tp_tot'] - 19}건 증가!)")
print(f"  * 총 정상 Recall              : {best['rec_tot']:.2f}% (기존 v6.0 14.07% 대비 +{best['rec_tot'] - 14.07:.2f}%p 폭등!)")
print(f"  * 총 정상 F1-Score            : {best['f1_tot']:.2f}% (기존 v6.0 23.60% 대비 +{best['f1_tot'] - 23.60:.2f}%p 대폭 향상!)")

# 시점별 재현율 분석
v_recs = []
for v in ['view1', 'view2', 'view3', 'view4', 'view5']:
    v_idx = np.where(views == v)[0]
    v_t = true_clean[v_idx]
    v_p = best['is_total_clean'][v_idx]
    rec_v = recall_score(v_t, v_p, zero_division=0) * 100
    v_recs.append(f"{v}:{rec_v:.1f}%")
print(f"  * 5대 시점별 정상 재현율: {' | '.join(v_recs)}")
print("=" * 115)
