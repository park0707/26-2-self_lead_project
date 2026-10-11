import os
res = []
for d in [r'd:\자주프 문서\7주차', r'd:\7주차', r'd:\Documents']:
    if not os.path.exists(d): continue
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.endswith('.md'):
                path = os.path.join(r, f)
                try:
                    content = open(path, encoding='utf-8').read()
                    if 'd5' in content.lower() and '70' in content:
                        res.append(path)
                except:
                    pass
with open('d:\\found.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(res))
print("DONE")
