# -*- coding: utf-8 -*-
import json
import numpy as np

with open(r"D:\data\일반화\cand1_vs_cand2_comparison.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total benchmark clips analyzed: {len(data)}")

methods = [
    ('baseline', 'Baseline (무보정)'),
    ('cand1_user_prop', '후보 1 (사용자 제안)'),
    ('cand2_blapose', '후보 2 (BLAPose FK)'),
    ('cand_hybrid_1_plus_2', '후보 1+2 융합 (Hybrid)')
]

def get_defect_dict(defects_list):
    return {d['defect_code']: d['probability'] for d in defects_list}

for ex_name in ['스쿼트', '바벨 데드', '스티프 데드']:
    print("\n" + "=" * 90)
    print(f" {ex_name} 결함별 평균 예측 확률 (%) 대조")
    print("=" * 90)
    
    ex_clips = [c for c in data if c['exercise_name'] == ex_name]
    
    defect_codes = set()
    for c in ex_clips:
        for r in c['reps']:
            for m_key, _ in methods:
                d_dict = get_defect_dict(r[m_key])
                defect_codes.update(d_dict.keys())
    d_list = sorted(list(defect_codes))
    
    header = f"{'결함 항목':<10} | " + " | ".join([f"{m_name:^18}" for _, m_name in methods])
    print(header)
    print("-" * len(header))
    
    for d_code in d_list:
        row = [f"{d_code:<10}"]
        for m_key, _ in methods:
            probs = []
            for c in ex_clips:
                for r in c['reps']:
                    d_dict = get_defect_dict(r[m_key])
                    if d_code in d_dict:
                        probs.append(d_dict[d_code])
            avg_p = np.mean(probs) if probs else 0.0
            row.append(f"{avg_p*100:15.2f}% ")
        print(" | ".join(row))

# 정상 클립 무릎 결함(c3) 반복별 상세 대조 (Clip 1)
print("\n" + "=" * 90)
print(" 정상 스쿼트(Clip 1) 무릎 결함(c3) 반복별 오탐 추이 (%)")
print("=" * 90)
clip1 = data[0]
for r in clip1['reps']:
    r_idx = r['rep_index']
    p_b = get_defect_dict(r['baseline'])['c3'] * 100
    p_c1 = get_defect_dict(r['cand1_user_prop'])['c3'] * 100
    p_c2 = get_defect_dict(r['cand2_blapose'])['c3'] * 100
    p_hy = get_defect_dict(r['cand_hybrid_1_plus_2'])['c3'] * 100
    print(f"  Rep {r_idx}: Base={p_b:5.1f}% | Cand1(사용자)={p_c1:5.1f}% | Cand2(BLAPose)={p_c2:5.1f}% | Hybrid(1+2)={p_hy:5.1f}%")

# 종합 평균
print("\n" + "=" * 90)
print(" 전체 종합 평균 결함 확률")
print("=" * 90)
for m_key, m_name in methods:
    all_probs = []
    for c in data:
        for r in c['reps']:
            d_dict = get_defect_dict(r[m_key])
            all_probs.extend(d_dict.values())
    print(f"{m_name:<28}: {np.mean(all_probs)*100:6.2f}%")
