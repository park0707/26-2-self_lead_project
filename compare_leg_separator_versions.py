import json
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

def load(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

print("================================================================================")
print("     [vidio-health-trainer] 사이드 런지 전 모델 다리 판별 엔진 이전 vs 신규 성능 비교")
print("================================================================================")

# 1. c1: sl1_v2 vs sl1_v3
c1_v2 = load(r"D:\src\models\second_models\side_lunge\sl1_v2_evaluation_results.json")
c1_v3 = load(r"D:\src\models\second_models\side_lunge\sl1_v3_evaluation_results.json")

f1_v2 = c1_v2["two_state_metrics"]["ensemble"]["f1"] * 100
auc_v2 = (c1_v2["two_state_metrics"]["ensemble"].get("roc_auc") or c1_v2["two_state_metrics"]["ensemble"].get("auc", 0)) * 100
f1_3s_v2 = c1_v2["three_state_metrics"]["effective_f1"] * 100
acc_3s_v2 = c1_v2["three_state_metrics"]["accuracy"] * 100
cov_v2 = c1_v2["three_state_metrics"]["coverage"] * 100

f1_v3 = c1_v3["two_state_metrics"]["ensemble"]["f1"] * 100
auc_v3 = (c1_v3["two_state_metrics"]["ensemble"].get("roc_auc") or c1_v3["two_state_metrics"]["ensemble"].get("auc", 0)) * 100
f1_3s_v3 = c1_v3["three_state_metrics"]["effective_f1"] * 100
acc_3s_v3 = c1_v3["three_state_metrics"]["accuracy"] * 100
cov_v3 = c1_v3["three_state_metrics"]["coverage"] * 100

print("\n[1] c1 (앞다리 무릎 각도 90도 오류)")
print("  - 기존 방식 (sl1_v2): 발목 전/후 Z축 깊이 변위량 (disp_z)")
print("  - 신규 방식 (sl1_v3): c3 골반 중심 대비 X축 공간 거리 분리 (Spatial Leg Separator)")
print(f"  • sl1_v2 : 2S F1 = {f1_v2:.2f}%, AUC = {auc_v2:.2f}%, ★ 3S F1 = {f1_3s_v2:.2f}%, 3S ACC = {acc_3s_v2:.2f}%, Cov = {cov_v2:.2f}%")
print(f"  • sl1_v3 : 2S F1 = {f1_v3:.2f}%, AUC = {auc_v3:.2f}%, ★ 3S F1 = {f1_3s_v3:.2f}%, 3S ACC = {acc_3s_v3:.2f}%, Cov = {cov_v3:.2f}%")
print(f"  --> 변화율: 2S F1 {f1_v3 - f1_v2:+.2f}%p | AUC {auc_v3 - auc_v2:+.2f}%p | 3S F1 {f1_3s_v3 - f1_3s_v2:+.2f}%p | 3S ACC {acc_3s_v3 - acc_3s_v2:+.2f}%p")

# 2. c2: sl2_v2 vs sl2_v3
c2_v2 = load(r"D:\src\models\second_models\side_lunge\sl2_v2_evaluation_results.json")
c2_v3 = load(r"D:\src\models\second_models\side_lunge\sl2_v3_evaluation_results.json")

f1_c2_v2 = c2_v2["gbdt_2state_metrics"]["ens"]["f1"] * 100
auc_c2_v2 = (c2_v2["gbdt_2state_metrics"]["ens"].get("roc_auc") or c2_v2["gbdt_2state_metrics"]["ens"].get("auc", 0)) * 100
f1_c2_3s_v2 = c2_v2["adaptive_3state_metrics"]["effective_f1"] * 100
acc_c2_3s_v2 = c2_v2["adaptive_3state_metrics"]["accuracy"] * 100
cov_c2_v2 = c2_v2["adaptive_3state_metrics"]["coverage"] * 100

f1_c2_v3 = c2_v3["gbdt_2state_metrics"]["ens"]["f1"] * 100
auc_c2_v3 = (c2_v3["gbdt_2state_metrics"]["ens"].get("roc_auc") or c2_v3["gbdt_2state_metrics"]["ens"].get("auc", 0)) * 100
f1_c2_3s_v3 = c2_v3["adaptive_3state_metrics"]["effective_f1"] * 100
acc_c2_3s_v3 = c2_v3["adaptive_3state_metrics"]["accuracy"] * 100
cov_c2_v3 = c2_v3["adaptive_3state_metrics"]["coverage"] * 100

print("\n[2] c2 (무릎-발 방향 일치 여부)")
print("  - 기존 방식 (sl2_v2): 최하사점 3D 무릎 굴곡 각도 비교 (더 많이 굽힌 다리)")
print("  - 신규 방식 (sl2_v3): c3 골반 중심 대비 X축 공간 거리 분리 (Spatial Leg Separator)")
print(f"  • sl2_v2 : 2S F1 = {f1_c2_v2:.2f}%, AUC = {auc_c2_v2:.2f}%, ★ 3S F1 = {f1_c2_3s_v2:.2f}%, 3S ACC = {acc_c2_3s_v2:.2f}%, Cov = {cov_c2_v2:.2f}%")
print(f"  • sl2_v3 : 2S F1 = {f1_c2_v3:.2f}%, AUC = {auc_c2_v3:.2f}%, ★ 3S F1 = {f1_c2_3s_v3:.2f}%, 3S ACC = {acc_c2_3s_v3:.2f}%, Cov = {cov_c2_v3:.2f}%")
print(f"  --> 변화율: 2S F1 {f1_c2_v3 - f1_c2_v2:+.2f}%p | AUC {auc_c2_v3 - auc_c2_v2:+.2f}%p | 3S F1 {f1_c2_3s_v3 - f1_c2_3s_v2:+.2f}%p | 3S ACC {acc_c2_3s_v3 - acc_c2_3s_v2:+.2f}%p")

# 3. c3: sl3_v1 vs sl3_v2
c3_v1 = load(r"D:\src\models\second_models\side_lunge\sl3_v1_evaluation_results.json")
c3_v2 = load(r"D:\src\models\second_models\side_lunge\sl3_v2_evaluation_results.json")

f1_c3_v1 = c3_v1["two_state_metrics"]["ensemble"]["f1"] * 100
auc_c3_v1 = (c3_v1["two_state_metrics"]["ensemble"].get("roc_auc") or c3_v1["two_state_metrics"]["ensemble"].get("auc", 0)) * 100
f1_c3_3s_v1 = c3_v1["three_state_metrics"]["effective_f1"] * 100
acc_c3_3s_v1 = c3_v1["three_state_metrics"]["accuracy"] * 100
cov_c3_v1 = c3_v1["three_state_metrics"]["coverage"] * 100

f1_c3_v2 = c3_v2["two_state_metrics"]["ensemble"]["f1"] * 100
auc_c3_v2 = (c3_v2["two_state_metrics"]["ensemble"].get("roc_auc") or c3_v2["two_state_metrics"]["ensemble"].get("auc", 0)) * 100
f1_c3_3s_v2 = c3_v2["three_state_metrics"]["effective_f1"] * 100
acc_c3_3s_v2 = c3_v2["three_state_metrics"]["accuracy"] * 100
cov_c3_v2 = c3_v2["three_state_metrics"]["coverage"] * 100

print("\n[3] c3 (안 굽히는 다리 신전 유지 여부)")
print("  - 기존 방식 (sl3_v1): 무릎 굴곡 각도 최소값 비교 (min_ang_left vs min_ang_right)")
print("  - 신규 방식 (sl3_v2): 골반 중심 대비 X축 공간 거리 분리 v2 + 22D 고도화")
print(f"  • sl3_v1 : 2S F1 = {f1_c3_v1:.2f}%, AUC = {auc_c3_v1:.2f}%, ★ 3S F1 = {f1_c3_3s_v1:.2f}%, 3S ACC = {acc_c3_3s_v1:.2f}%, Cov = {cov_c3_v1:.2f}%")
print(f"  • sl3_v2 : 2S F1 = {f1_c3_v2:.2f}%, AUC = {auc_c3_v2:.2f}%, ★ 3S F1 = {f1_c3_3s_v2:.2f}%, 3S ACC = {acc_c3_3s_v2:.2f}%, Cov = {cov_c3_v2:.2f}%")
print(f"  --> 변화율: 2S F1 {f1_c3_v2 - f1_c3_v1:+.2f}%p | AUC {auc_c3_v2 - auc_c3_v1:+.2f}%p | 3S F1 {f1_c3_3s_v2 - f1_c3_3s_v1:+.2f}%p | 3S ACC {acc_c3_3s_v2 - acc_c3_3s_v1:+.2f}%p")

# 4. c4: sl4_v2 vs sl4_v3
c4_v2 = load(r"D:\src\models\second_models\side_lunge\sl4_v2_evaluation_results.json")
c4_v3 = load(r"D:\src\models\second_models\side_lunge\sl4_v3_evaluation_results.json")

f1_c4_v2 = c4_v2["ensemble_2state"]["f1"] * 100
auc_c4_v2 = (c4_v2["ensemble_2state"].get("roc_auc") or c4_v2["ensemble_2state"].get("auc", 0)) * 100
f1_c4_3s_v2 = c4_v2["ensemble_3state"]["confirmed_f1"] * 100
acc_c4_3s_v2 = c4_v2["ensemble_3state"]["confirmed_accuracy"] * 100
cov_c4_v2 = c4_v2["ensemble_3state"]["coverage"] * 100

f1_c4_v3 = c4_v3["ensemble_2state"]["f1"] * 100
auc_c4_v3 = (c4_v3["ensemble_2state"].get("roc_auc") or c4_v3["ensemble_2state"].get("auc", 0)) * 100
f1_c4_3s_v3 = c4_v3["ensemble_3state"]["confirmed_f1"] * 100
acc_c4_3s_v3 = c4_v3["ensemble_3state"]["confirmed_accuracy"] * 100
cov_c4_v3 = c4_v3["ensemble_3state"]["coverage"] * 100

print("\n[4] c4 (척추의 중립 위반 / 체간 측굴 및 경추 회전)")
print("  - 기존 방식 (sl4_v2): 바벨 런지 계승 3D 무릎 굴곡 각도 비교")
print("  - 신규 방식 (sl4_v3): c3 골반 중심 대비 X축 공간 거리 분리 (Spatial Leg Separator)")
print(f"  • sl4_v2 : 2S F1 = {f1_c4_v2:.2f}%, AUC = {auc_c4_v2:.2f}%, ★ 3S F1 = {f1_c4_3s_v2:.2f}%, 3S ACC = {acc_c4_3s_v2:.2f}%, Cov = {cov_c4_v2:.2f}%")
print(f"  • sl4_v3 : 2S F1 = {f1_c4_v3:.2f}%, AUC = {auc_c4_v3:.2f}%, ★ 3S F1 = {f1_c4_3s_v3:.2f}%, 3S ACC = {acc_c4_3s_v3:.2f}%, Cov = {cov_c4_v3:.2f}%")
print(f"  --> 변화율: 2S F1 {f1_c4_v3 - f1_c4_v2:+.2f}%p | AUC {auc_c4_v3 - auc_c4_v2:+.2f}%p | 3S F1 {f1_c4_3s_v3 - f1_c4_3s_v2:+.2f}%p | 3S ACC {acc_c4_3s_v3 - acc_c4_3s_v2:+.2f}%p")

# 5. c5: sl5_v5 vs sl5_v6
c5_v5 = load(r"D:\src\models\second_models\side_lunge\sl5_v5_all_relabel_benchmark_results.json")
c5_v6 = load(r"D:\src\models\second_models\side_lunge\sl5_v6_benchmark_results.json")

f1_c5_v5 = c5_v5["full_coverage"]["f1"] * 100
auc_c5_v5 = (c5_v5["full_coverage"].get("roc_auc") or c5_v5["full_coverage"].get("auc", 0)) * 100
f1_c5_3s_v5 = c5_v5["best_3state"]["f1"] * 100
acc_c5_3s_v5 = c5_v5["best_3state"]["accuracy"] * 100
cov_c5_v5 = c5_v5["best_3state"]["coverage"] * 100

f1_c5_v6 = c5_v6["full_coverage"]["f1"] * 100
auc_c5_v6 = (c5_v6["full_coverage"].get("roc_auc") or c5_v6["full_coverage"].get("auc", 0)) * 100
f1_c5_3s_v6 = c5_v6["best_3state"]["f1"] * 100
acc_c5_3s_v6 = c5_v6["best_3state"]["accuracy"] * 100
cov_c5_v6 = c5_v6["best_3state"]["coverage"] * 100

print("\n[5] c5 (상체의 과도한 숙임/젖힘)")
print("  - 기존 방식 (sl5_v5): 기존 다리 분리 및 피처")
print("  - 신규 방식 (sl5_v6): c3 골반 중심 대비 X축 공간 거리 분리 (Spatial Leg Separator)")
print(f"  • sl5_v5 : 2S F1 = {f1_c5_v5:.2f}%, AUC = {auc_c5_v5:.2f}%, ★ 3S F1 = {f1_c5_3s_v5:.2f}%, 3S ACC = {acc_c5_3s_v5:.2f}%, Cov = {cov_c5_v5:.2f}%")
print(f"  • sl5_v6 : 2S F1 = {f1_c5_v6:.2f}%, AUC = {auc_c5_v6:.2f}%, ★ 3S F1 = {f1_c5_3s_v6:.2f}%, 3S ACC = {acc_c5_3s_v6:.2f}%, Cov = {cov_c5_v6:.2f}%")
print(f"  --> 변화율: 2S F1 {f1_c5_v6 - f1_c5_v5:+.2f}%p | AUC {auc_c5_v6 - auc_c5_v5:+.2f}%p | 3S F1 {f1_c5_3s_v6 - f1_c5_3s_v5:+.2f}%p | 3S ACC {acc_c5_3s_v6 - acc_c5_3s_v5:+.2f}%p")
