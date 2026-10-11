import os

res = []
with open('d:\\all_files.txt', 'r', encoding='utf-8') as f:
    files = [line.strip() for line in f if '바벨' in line and '데드리프트' in line and line.endswith('.md')]

for path in files:
    try:
        content = open(path, encoding='utf-8', errors='ignore').read()
        lines = content.split('\n')
        # Only keep files that mention MediaPipe and RTMPose, or have F1>70%
        table_lines = [l for l in lines if '|' in l and ('%' in l)]
        if table_lines:
            res.append(f"--- {os.path.basename(path)} ---")
            res.extend(table_lines)
    except Exception as e:
        pass

with open('d:\\final_tables.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(res))
print("DONE")
