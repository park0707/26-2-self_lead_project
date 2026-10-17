# -*- coding: utf-8 -*-
import json
import numpy as np

with open(r"D:\data\일반화\anatomical_benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total benchmark clips analyzed: {len(data)}")

methods = [
    ('baseline', 'Baseline'),
    ('test1_isotropic', 'Test 1(등방)'),
    ('test3_virtual_leveling', 'Test 3(수평)'),
    ('user_prop_a_thigh', '제안 A(대퇴골)'),
    ('user_prop_b_torso', '제안 B(체간)'),
    ('user_prop_a_plus_leveling', '제안 A+3(융합)')
]

first_rep = data[0]['reps'][0]
sa = first_rep['scale_info_a']
sb = first_rep['scale_info_b']
print("=" * 80)
print(" 1. 사용자 신체 계측치 및 보정 스케일 계수 확인")
print("=" * 80)
print(f"사용자 어깨 너비 (Stand Phase): {sa['user_shoulder']:.4f} m (AI-Hub 표준: 0.3117 m)")
print(f"사용자 대퇴골 길이           : {sa['user_thigh']:.4f} m (AI-Hub 표준: 0.3696 m)")
print(f"사용자 체간 길이             : {sa['user_torso']:.4f} m (AI-Hub 표준: 0.4874 m)")
print(f"제안 A (대퇴골) 스케일 계수 : s_x = {sa['s_x']:.4f}, s_y = {sa['s_y']:.4f}, 가로확장 = {(sa['aspect_correction_ratio']-1)*100:+.2f}%")
print(f"제안 B (체간)   스케일 계수 : s_x = {sb['s_x']:.4f}, s_y = {sb['s_y']:.4f}, 가로확장 = {(sb['aspect_correction_ratio']-1)*100:+.2f}%")

# Helper to get defect prob dict {defect_code: prob}
def get_defect_dict(defects_list):
    return {d['defect_code']: d['probability'] for d in defects_list}

# 2. 운동별 세부 결함 확률 분석
for ex_name in ['스쿼트', '바벨 데드', '스티프 데드']:
    print("\n" + "=" * 80)
    print(f" 2. {ex_name} 결함별 평균 예측 확률 (%)")
    print("=" * 80)
    
    ex_clips = [c for c in data if c['exercise_name'] == ex_name]
    
    # 해당 운동의 결함 코드 수집
    defect_codes = set()
    for c in ex_clips:
        for r in c['reps']:
            for m_key, _ in methods:
                d_dict = get_defect_dict(r[m_key])
                defect_codes.update(d_dict.keys())
    d_list = sorted(list(defect_codes))
    
    header = f"{'결함 코드':<10} | " + " | ".join([f"{m_name:^13}" for _, m_name in methods])
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
            row.append(f"{avg_p*100:11.2f}% ")
        print(" | ".join(row))

# 3. 전체 종합 평균
print("\n" + "=" * 80)
print(" 3. 전체 종합 평균 결함 확률 (전체 3대 운동 전수 집계)")
print("=" * 80)
for m_key, m_name in methods:
    all_probs = []
    for c in data:
        for r in c['reps']:
            d_dict = get_defect_dict(r[m_key])
            for p in d_dict.values():
                all_probs.append(p)
    print(f"{m_name:<20}: 평균 결함 확률 = {np.mean(all_probs)*100:6.2f}%")

# 4. 정상 영상에서의 오탐률(FP) vs 결함 영상에서의 탐지율(TP) 정밀 대조
print("\n" + "=" * 80)
print(" 4. 정상 의도 영상 vs 결함 의도 영상별 평균 결함 확률")
print("=" * 80)
normal_clips = [c for c in data if "정상" in c['intent']]
defect_clips = [c for c in data if "결함" in c['intent']]

print(f"[정상 의도 클립 N={len(normal_clips)}] -> 낮을수록 오탐(FP) 완화:")
for m_key, m_name in methods:
    probs = []
    for c in normal_clips:
        for r in c['reps']:
            d_dict = get_defect_dict(r[m_key])
            probs.extend(d_dict.values())
    print(f"  - {m_name:<18}: {np.mean(probs)*100:6.2f}%")

print(f"\n[결함 의도 클립 N={len(defect_clips)}] -> 높을수록 탐지(TP) 유지:")
for m_key, m_name in methods:
    probs = []
    for c in defect_clips:
        for r in c['reps']:
            d_dict = get_defect_dict(r[m_key])
            probs.extend(d_dict.values())
    print(f"  - {m_name:<18}: {np.mean(probs)*100:6.2f}%")
