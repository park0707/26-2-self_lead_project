# -*- coding: utf-8 -*-
import json
import numpy as np

with open(r"D:\data\일반화\3tests_comparison_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total clips analyzed: {len(data)}")

# Collect defect-by-defect stats across modes
# Modes: baseline, test1_isotropic, test2_body_centric, test3_virtual_leveling, hybrid_1_plus_3
modes = ["baseline", "test1_isotropic", "test2_body_centric", "test3_virtual_leveling", "hybrid_1_plus_3"]

defect_stats = {}

for clip in data:
    ex = clip["exercise_name"]
    intent = clip["intent"]
    
    for r in clip["reps"]:
        for m in modes:
            for d in r[m]:
                code = d["defect_code"]
                full_k = f"{ex}_{code}"
                if full_k not in defect_stats:
                    defect_stats[full_k] = {m: [] for m in modes}
                defect_stats[full_k][m].append(d["probability"])

print("\n" + "=" * 105)
print(f"{'결함 항목':18s} | {'Baseline':12s} | {'Test 1 (등방성)':14s} | {'Test 2 (로컬골격)':14s} | {'Test 3 (가상수평)':14s} | {'Hybrid (1+3)':14s}")
print("=" * 105)

for k, m_dict in sorted(defect_stats.items()):
    b_mean = np.mean(m_dict["baseline"]) * 100
    t1_mean = np.mean(m_dict["test1_isotropic"]) * 100
    t2_mean = np.mean(m_dict["test2_body_centric"]) * 100
    t3_mean = np.mean(m_dict["test3_virtual_leveling"]) * 100
    hy_mean = np.mean(m_dict["hybrid_1_plus_3"]) * 100
    
    print(f"{k:18s} | {b_mean:9.1f}%   | {t1_mean:11.1f}%    | {t2_mean:11.1f}%    | {t3_mean:11.1f}%    | {hy_mean:11.1f}%")
