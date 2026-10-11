import os

src = r'D:\src\models\second_models\babel_stiff_deadlift\train_and_evaluate_sd1_rtmp_2.py'
dst = r'D:\src\models\second_models\babel_stiff_deadlift\test_sd1_twotier.py'

with open(src, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_load_dataset = False
for line in lines:
    if line.startswith('def load_dataset():'):
        in_load_dataset = True
        new_lines.append(line)
        new_lines.append("""
    print("[*] Loading dataset for Two-Tier Test...", flush=True)
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']

    # Filter out noisy keys: Key213, Key214, Key215
    mask = ~df['key'].isin(['Key213', 'Key214', 'Key215'])
    df = df[mask].copy()

    seq_ids = [os.path.splitext(os.path.basename(p))[0] for p in df['path']]
    subjects = df['subj'].values
    views = df['view'].values
    keys = df['key'].values
    N = len(seq_ids)

    # Re-label based on physical knee behavior
    # 1 (Defect): Key210, Key212, Key216 (Physical knee bending / drop)
    # 0 (Normal): Key209, Key211 (Knee fixed)
    y = np.array([1 if k in ['Key210', 'Key212', 'Key216'] else 0 for k in keys], dtype=np.int32)
    print(f"[+] Total {N} samples after Two-Tier filtering (Defect: {y.sum()} / Normal: {N - y.sum()})", flush=True)
    return seq_ids, subjects, views, keys, y
""")
    elif in_load_dataset:
        if line.startswith('def '):
            in_load_dataset = False
            new_lines.append(line)
    else:
        new_lines.append(line)

content = "".join(new_lines)
content = content.replace('sd1_rtmp_2_evaluation_results.json', 'sd1_twotier_evaluation_results.json')
content = content.replace('sd1_rtmp_2_bundle.pkl', 'sd1_twotier_bundle.pkl')
content = content.replace('sd1_rtmp_2', 'sd1_twotier')
content = content.replace('from sd1_twotier_features import (', 'from sd1_rtmp_2_features import (')
content = content.replace('FEATURE_NAMES_47D_SD1_RTMP_2', 'FEATURE_NAMES_47D_SD1_RTMP_2') # Restore this if it changed, but let's just be careful
content = content.replace('FEATURE_NAMES_47D_SD1_twotier', 'FEATURE_NAMES_47D_SD1_RTMP_2')
content = content.replace('extract_sd1_twotier_features', 'extract_sd1_rtmp_2_features')


with open(dst, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Created {dst}")
