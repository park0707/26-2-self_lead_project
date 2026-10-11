import json

def inspect_sl():
    # sl1
    with open(r'D:\src\models\second_models\side_lunge\sl1_final_evaluation_results.json', encoding='utf-8') as f:
        sl1 = json.load(f)
    print('=== SL1 ===')
    print('Model:', sl1.get('model_name'))
    print('2-State:', sl1.get('two_state_metrics'))
    print('3-State Primary (0.35-0.65):', sl1.get('primary_three_state_policy_035_065'))
    print('3-State Adaptive:', sl1.get('three_state_metrics'))
    print('Per View (3-State Primary):')
    for v, d in sl1.get('primary_three_state_policy_035_065', {}).get('view_audits', {}).items():
        print(f'  {v}: {d}')

    # sl2_v7
    with open(r'D:\src\models\second_models\side_lunge\sl2_v7_evaluation_results.json', encoding='utf-8') as f:
        sl2 = json.load(f)
    print('\n=== SL2_v7 ===')
    print('Model:', sl2.get('model_name'))
    for prof_name, prof_data in sl2.get('profiles', {}).items():
        print(f"Profile {prof_name}: overall_f1={prof_data.get('overall_f1')}, coverage={prof_data.get('coverage')}, overall_acc={prof_data.get('overall_accuracy')}")
        for v, d in prof_data.get('view_audits', {}).items():
            print(f"    {v}: f1={d.get('effective_f1')}, acc={d.get('accuracy')}, uncert={d.get('uncertainty_rate')}")

    # sl2_v8_front
    try:
        with open(r'D:\src\models\second_models\side_lunge\sl2_v8_front_evaluation_results.json', encoding='utf-8') as f:
            sl2_v8 = json.load(f)
        print('\n=== SL2_v8_front ===')
        print('Model:', sl2_v8.get('model_name'))
        for prof_name, prof_data in sl2_v8.get('profiles', {}).items():
            print(f"Profile {prof_name}: overall_f1={prof_data.get('overall_f1')}, coverage={prof_data.get('coverage')}")
            for v, d in prof_data.get('view_audits', {}).items():
                print(f"    {v}: f1={d.get('effective_f1')}, acc={d.get('accuracy')}, uncert={d.get('uncertainty_rate')}")
    except Exception as e:
        print('sl2_v8 error:', e)

    # sl3
    with open(r'D:\src\models\second_models\side_lunge\sl3_v3_evaluation_results.json', encoding='utf-8') as f:
        sl3 = json.load(f)
    print('\n=== SL3_v3 ===')
    print('Model:', sl3.get('model_name'))
    print('2-State Ensemble:', sl3.get('two_state_metrics', {}).get('ensemble'))
    print('3-State:', sl3.get('three_state_metrics'))
    for v, d in sl3.get('three_state_metrics', {}).get('view_audits', {}).items():
        print(f'  {v}: {d}')

    # sl4
    with open(r'D:\src\models\second_models\side_lunge\sl4_v3_evaluation_results.json', encoding='utf-8') as f:
        sl4 = json.load(f)
    print('\n=== SL4_v3 ===')
    print('Model:', sl4.get('model_name'))
    print('2-State Ensemble:', sl4.get('ensemble_2state'))
    print('3-State Ensemble:', sl4.get('ensemble_3state'))
    print('View Audit:')
    for v, d in sl4.get('view_audit', {}).items():
        print(f'  {v}: {d}')

    # sl5
    with open(r'D:\src\models\second_models\side_lunge\sl5_v7_benchmark_results.json', encoding='utf-8') as f:
        sl5 = json.load(f)
    print('\n=== SL5_v7 ===')
    print('Version:', sl5.get('version'))
    print('Full Coverage:', sl5.get('full_coverage'))
    print('3-State:', sl5.get('three_state'))
    print('View Audits:')
    for v, d in sl5.get('view_audits', {}).items():
        print(f'  {v}: {d}')

if __name__ == '__main__':
    inspect_sl()
