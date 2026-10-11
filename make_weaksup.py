import os

src = r'D:\src\models\second_models\babel_stiff_deadlift\train_and_evaluate_sd1_rtmp_2.py'
dst = r'D:\src\models\second_models\babel_stiff_deadlift\train_and_evaluate_sd1_weak_supervision.py'

with open(src, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_load_dataset = False
for line in lines:
    if line.startswith('def load_dataset():'):
        in_load_dataset = True
        new_lines.append(line)
        new_lines.append("""
    print("[*] Loading dataset and applying Weak Supervision (Physical Rule) for Relabeling...", flush=True)
    with open(CACHE_PKL, 'rb') as fp:
        c = pickle.load(fp)
    df = c['df']

    seq_ids = [os.path.splitext(os.path.basename(p))[0] for p in df['path']]
    subjects = df['subj'].values
    views = df['view'].values
    keys = df['key'].values
    N = len(seq_ids)

    # --- Weak Supervision Relabeling ---
    def get_angle(A, B, C):
        ba = A - B
        bc = C - B
        cosine = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    y_clean = []
    RTM_DIR_WS = r'D:\Data\전처리_데이터\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트'
    for i, row in df.iterrows():
        sid = os.path.splitext(os.path.basename(row['path']))[0]
        p = os.path.join(RTM_DIR_WS, sid + '.npz')
        if os.path.exists(p):
            kps = np.load(p)['kps_26_64f']
            rk = get_angle(kps[:, 12, :2], kps[:, 14, :2], kps[:, 16, :2])
            lk = get_angle(kps[:, 11, :2], kps[:, 13, :2], kps[:, 15, :2])
            min_k = min(np.min(rk), np.min(lk))
            if min_k <= 155.0:
                y_clean.append(1)
            else:
                y_clean.append(0)
        else:
            y_clean.append(0)

    y = np.array(y_clean, dtype=np.int32)
    print(f"[+] Total {N} samples. Cleaned Labels: (Defect: {y.sum()} / Normal: {N - y.sum()})", flush=True)
    return seq_ids, subjects, views, keys, y
""")
    elif in_load_dataset:
        if line.startswith('def '):
            in_load_dataset = False
            new_lines.append(line)
    else:
        new_lines.append(line)

content = "".join(new_lines)
content = content.replace('sd1_rtmp_2_evaluation_results.json', 'sd1_weaksup_evaluation_results.json')
content = content.replace('sd1_rtmp_2_bundle.pkl', 'sd1_weaksup_bundle.pkl')
content = content.replace('sd1_rtmp_2', 'sd1_weaksup')

with open(dst, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Created {dst}")
