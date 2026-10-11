import numpy as np, hashlib, collections, os
for cond in ['Condition4_NovelFlow_3D', 'Condition3_MediaPipe_3D']:
    base = f'd:/Data/피처_데이터/{cond}'
    print('=====', cond)
    for fn in ['Dual_Train.npz', 'Dual_Test_Unseen_View.npz', 'Dual_Valid.npz', 'Dual_Test_Strict.npz', 'Dual_Test_Unseen_Subj.npz']:
        d = np.load(os.path.join(base, fn), allow_pickle=True)
        m = np.array(['_D02_' in str(s) for s in d['sequence_ids']])
        sids = d['sequence_ids'][m]; X = d['X_Format_B'][m]
        hs = [hashlib.md5(x.tobytes()).hexdigest() for x in X]
        g = collections.defaultdict(list)
        for s, h in zip(sids, hs): g[h].append(str(s).rsplit('_', 1)[-1])
        print(fn, 'N', len(sids), 'uniq', len(g), collections.Counter(tuple(sorted(v)) for v in g.values() if len(v) > 1).most_common(3))
