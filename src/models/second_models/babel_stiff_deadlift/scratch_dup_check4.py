import numpy as np, hashlib, collections
def load(cond):
    out = {}
    for fn in ['Dual_Train.npz', 'Dual_Test_Unseen_View.npz', 'Dual_Valid.npz']:
        d = np.load(f'd:/Data/피처_데이터/{cond}/{fn}', allow_pickle=True)
        for s, x in zip(d['sequence_ids'], d['X_Format_B']):
            if '_D10_' in str(s): out[str(s)] = x
    return out
for cond in ['Condition4_NovelFlow_3D', 'Condition3_MediaPipe_3D']:
    o = load(cond)
    print('=====', cond, len(o), collections.Counter(k.rsplit('_', 1)[-1] for k in o))
    g = collections.defaultdict(list)
    for k, x in o.items(): g[hashlib.md5(x.tobytes()).hexdigest()].append(k.rsplit('_', 1)[-1])
    print('uniq', len(g), collections.Counter(tuple(sorted(v)) for v in g.values()).most_common(5))
    # 가까운 값 비교(동일 시퀀스의 view간 최대 절대차)
    ks = [k for k in o if k.endswith('view1')][:3]
    for k in ks:
        b = k[:-5]
        print(b, {v: float(np.abs(o[k] - o[b + v]).max()) for v in ['view2', 'view3', 'view4', 'view5'] if b + v in o})

