import numpy as np, collections
d = np.load('v7_feature_cache.npz', allow_pickle=True)
h = d['tensor_hash']
c = np.load([f for f in ['v6_dataset_cache.npz'] if False] or 'v7_rawstats_cache.npz', allow_pickle=True)
print('rawstats keys', c.files)
import v7_ablation_core as K
print([n for n in dir(K) if 'PATH' in n])
data = np.load(K.CACHE_PATH, allow_pickle=True)
print(data.files)
sids = np.array([str(s) for s in data['sequence_ids']])
print(len(sids), len(set(h)))
for s in sids[:12]: print(s)
g = collections.defaultdict(list)
for s, x in zip(sids, h): g[x].append(s)
sizes = collections.Counter(len(v) for v in g.values())
print('group size dist', sizes)
for k, v in list(g.items())[:4]: print(v)
