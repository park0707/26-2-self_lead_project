import os

res = []
with open('d:\\all_files.txt', 'r', encoding='utf-8') as f:
    files = [line.strip() for line in f if ('바벨_데드리프트' in line or '바벨데드리프트' in line) and line.endswith('.md')]

for path in files:
    try:
        content = open(path, encoding='utf-8', errors='ignore').read()
        lines = content.split('\n')
        table_lines = [l.strip() for l in lines if '|' in l]
        if table_lines:
            res.append(f"\n--- {os.path.basename(path)} ---")
            res.extend(table_lines)
    except:
        pass

with open('d:\\final_tables2.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(res))
print("DONE")
