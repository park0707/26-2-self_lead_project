import os, numpy as np
d = r"D:\Data\전처리_데이터\RTMPose_Halpe26_64F\04_바벨_런지"
f = os.listdir(d)[0]
data = np.load(os.path.join(d, f))
print("Keys:", data.files)
if len(data.files) > 0:
    k = data.files[0]
    print(k, data[k].shape)
