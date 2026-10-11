import json

def get_detailed_report():
    # 1. sl1_final
    with open(r"D:\src\models\second_models\side_lunge\sl1_final_evaluation_results.json", encoding="utf-8") as f:
        sl1 = json.load(f)
    
    # 2. sl2_v7
    with open(r"D:\src\models\second_models\side_lunge\sl2_v7_evaluation_results.json", encoding="utf-8") as f:
        sl2 = json.load(f)

    # sl2_v8_front
    with open(r"D:\src\models\second_models\side_lunge\sl2_v8_front_evaluation_results.json", encoding="utf-8") as f:
        sl2_v8 = json.load(f)
        
    # 3. sl3_v3
    with open(r"D:\src\models\second_models\side_lunge\sl3_v3_evaluation_results.json", encoding="utf-8") as f:
        sl3 = json.load(f)

    # 4. sl4_v3
    with open(r"D:\src\models\second_models\side_lunge\sl4_v3_evaluation_results.json", encoding="utf-8") as f:
        sl4 = json.load(f)

    # 5. sl5_v7
    with open(r"D:\src\models\second_models\side_lunge\sl5_v7_benchmark_results.json", encoding="utf-8") as f:
        sl5 = json.load(f)

    print("="*80)
    print("SL1 (c1):")
    p1 = sl1['primary_three_state_policy_035_065']
    print(f"  Overall 3-State F1: {p1['effective_f1']*100:.2f}% (Coverage: {p1['coverage']*100:.2f}%, Acc: {p1['accuracy']*100:.2f}%, Uncert: {p1['uncertainty_rate']*100:.2f}%)")
    for vk in ['view5', 'view2', 'view3', 'view4', 'view1']:
        va = p1['view_audits'][vk]
        print(f"    {vk}: F1={va['effective_f1']*100:.2f}%, Acc={va['accuracy']*100:.2f}%, Uncert={va['uncertainty_rate']*100:.2f}%")

    print("="*80)
    print("SL2 (c2):")
    p2 = sl2['profiles']['standard_10pct']
    print(f"  Overall 3-State (std 10%) F1: {p2['overall_f1']*100:.2f}% (Coverage: {p2['coverage']*100:.2f}%, Acc: {p2['overall_accuracy']*100:.2f}%)")
    for vk in ['view5', 'view2', 'view3', 'view4', 'view1']:
        va = p2['view_audits'][vk]
        print(f"    {vk}: F1={va['effective_f1']*100:.2f}%, Acc={va['accuracy']*100:.2f}%, Uncert={va['uncertainty_rate']*100:.2f}%")
    p2_hc = sl2['profiles']['high_confidence']
    print(f"  Overall 3-State (high conf) F1: {p2_hc['overall_f1']*100:.2f}% (Coverage: {p2_hc['coverage']*100:.2f}%, Acc: {p2_hc['overall_accuracy']*100:.2f}%)")
    for vk in ['view5', 'view2', 'view3', 'view4', 'view1']:
        va = p2_hc['view_audits'][vk]
        print(f"    {vk}: F1={va['effective_f1']*100:.2f}%, Acc={va['accuracy']*100:.2f}%, Uncert={va['uncertainty_rate']*100:.2f}%")
    print(f"  Front Views (view2, 3, 4): F1={sl2['front_views_submetrics']['f1']*100:.2f}%, Acc={sl2['front_views_submetrics']['accuracy']*100:.2f}%")
    
    print("  [Reference] sl2_v8_front (Front views only - view2, 4):")
    for prof_name in ['standard_10pct', 'high_confidence']:
        prof = sl2_v8['profiles'][prof_name]
        print(f"    Profile {prof_name}: F1={prof['overall_f1']*100:.2f}%, Coverage={prof['coverage']*100:.2f}%")
        for vk in ['view2', 'view4']:
            va = prof['view_audits'][vk]
            print(f"      {vk}: F1={va['effective_f1']*100:.2f}%, Acc={va['accuracy']*100:.2f}%")

    print("="*80)
    print("SL3 (c3):")
    p3 = sl3['three_state_metrics']
    print(f"  Overall 3-State F1: {p3['effective_f1']*100:.2f}% (Coverage: {p3['coverage']*100:.2f}%, Acc: {p3['accuracy']*100:.2f}%, Uncert: {p3['uncertainty_rate']*100:.2f}%)")
    for vk in ['view5', 'view2', 'view3', 'view4', 'view1']:
        va = p3['view_audits'][vk]
        print(f"    {vk}: F1={va['effective_f1']*100:.2f}%, Acc={va['accuracy']*100:.2f}%, Uncert={va['uncertainty_rate']*100:.2f}%")

    print("="*80)
    print("SL4 (c4):")
    p4 = sl4['ensemble_3state']
    print(f"  Overall 3-State F1: {p4['confirmed_f1']*100:.2f}% (Coverage: {p4['coverage']*100:.2f}%, Acc: {p4['confirmed_accuracy']*100:.2f}%)")
    for vk in ['view5', 'view2', 'view3', 'view4', 'view1']:
        va = sl4['view_audit'][vk]
        print(f"    {vk}: F1={va['3state_f1']*100:.2f}%, Acc={va['3state_acc']*100:.2f}%, Uncert={(1.0 - va['coverage'])*100:.2f}%, AUC={va['auc']:.4f}")

    print("="*80)
    print("SL5 (c5):")
    p5 = sl5['three_state']
    print(f"  Overall 3-State F1: {p5['f1']*100:.2f}% (Coverage: {p5['coverage']*100:.2f}%, Acc: {p5['accuracy']*100:.2f}%, Uncert={p5['rejection_rate']*100:.2f}%)")
    for vk in ['view5', 'view2', 'view3', 'view4', 'view1']:
        va = sl5['view_audits'][vk]
        print(f"    {vk}: F1={va['3state_f1']*100:.2f}%, Acc={va['accuracy']*100:.2f}%, Uncert={va['rejection_rate']*100:.2f}%, AUC={va['auc']:.4f}")

get_detailed_report()
