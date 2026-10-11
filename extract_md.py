import os

res = []
d = r'd:\자주프 문서\7주차'
for f in os.listdir(d):
    if f.endswith('.md') and '데드리프트' in f:
        try:
            content = open(os.path.join(d, f), encoding='utf-8').read()
            res.append(f"--- FILE: {f} ---")
            for line in content.split('\n'):
                if '|' in line and ('F1' in line or 'd1' in line.lower() or 'd2' in line.lower() or 'd3' in line.lower() or 'd4' in line.lower() or 'd5' in line.lower()):
                    res.append(line.strip())
        except Exception:
            pass

with open('d:\\md_tables.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(res))
print("DONE")
