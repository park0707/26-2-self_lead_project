import os, hashlib, collections
r = r'd:\Data\01_raw_data\images\03_바벨_스티프_데드리프트'
for key in ['[오류]_D02_피험자1_Key210', '[오류]_D02_피험자1_Key213']:
    d = os.path.join(r, key)
    files = sorted(os.listdir(d))
    by = collections.defaultdict(list)
    for f in files:
        v = [p for p in f.split('_') if p.startswith('시점')][0]
        by[v].append(f)
    print(key, {k: len(v) for k, v in by.items()})
    for v, fl in sorted(by.items()):
        hs = [hashlib.md5(open(os.path.join(d, f), 'rb').read()).hexdigest()[:6] for f in fl[:3]]
        print(' ', v, hs, [os.path.getsize(os.path.join(d, f)) for f in fl[:3]])
lr = r'd:\Data\01_raw_data\labels_json\03_바벨_스티프_데드리프트'
dd = [x for x in os.listdir(lr) if 'D02_피험자1_Key210' in x]
print(dd[:3])
if dd:
    fl = sorted(os.listdir(os.path.join(lr, dd[0]))); print(len(fl), fl[:6])
    for f in fl[:16:3]:
        print(f, hashlib.md5(open(os.path.join(lr, dd[0], f), 'rb').read()).hexdigest()[:6])
