import os

script_path = r'D:\src\models\second_models\babel_stiff_deadlift\train_and_evaluate_sd1_rtmp_2.py'
new_script_path = r'D:\src\models\second_models\babel_stiff_deadlift\train_and_evaluate_sd1_rtmp_user.py'

with open(script_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_y = "y = np.array([1 if k in ['Key213', 'Key214', 'Key215'] else 0 for k in keys], dtype=np.int32)"
new_y = """
    import pandas as pd
    df_labels = pd.read_csv(r'D:\\Data\\temp\\00_8대_Key_무릎고정결함_라벨체크시트.csv', encoding='utf-8')
    label_dict = {row.iloc[0]: int(row.iloc[5]) for i, row in df_labels.iterrows()}
    print('[*] User labels applied:', label_dict)
    y = np.array([label_dict.get(k, 0) for k in keys], dtype=np.int32)
"""

content = content.replace(old_y, new_y)
content = content.replace('sd1_rtmp_2_evaluation_results.json', 'sd1_rtmp_user_evaluation_results.json')
content = content.replace('sd1_rtmp_2_bundle.pkl', 'sd1_rtmp_user_bundle.pkl')
content = content.replace('sd1_rtmp_2', 'sd1_rtmp_user')

with open(new_script_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Created {new_script_path}")
