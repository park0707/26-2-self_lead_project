import numpy as np, glob, json, collections
fs = glob.glob(r'd:\Data\전처리_데이터\MediaPipe\03_바벨_스티프_데드리프트\*.npz')
has3d = collections.Counter(); views = collections.Counter(); raw0 = collections.Counter()
for f in fs:
    d = np.load(f, allow_pickle=True)
    has3d['keypoints_3d' in d.files] += 1
    views[json.loads(str(d['metadata'])).get('view')] += 1
    raw0[bool(np.abs(d['keypoints_2d_raw']).sum() == 0)] += 1
print(len(fs), 'has3d', dict(has3d), 'views', dict(views), 'raw2d all-zero', dict(raw0))
