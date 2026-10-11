import os

res = []
dirs = [r'd:\자주프 문서', r'd:\7주차', r'd:\Documents']
for d in dirs:
    if not os.path.exists(d): continue
    for r, _, fs in os.walk(d):
        for f in fs:
            if '바벨' in f and '데드리프트' in f and f.endswith('.md'):
                path = os.path.join(r, f)
                try:
                    content = open(path, encoding='utf-8', errors='ignore').read()
                    lines = content.split('\n')
                    table_lines = [l.strip() for l in lines if '|' in l and 'F1' in l]
                    if table_lines:
                        res.append(f"\n--- {f} ---")
                        res.extend(table_lines)
                except Exception as e:
                    pass

with open('d:\\real_tables.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(res))
print("DONE")
