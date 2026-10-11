import numpy as np
import glob
files = glob.glob(r'D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지\*.npz')
if files:
    d = np.load(files[0])
    k3 = d['kps_33_3d_64f']
    k2 = d['kps_33_2d_64f']
    print("3D shape:", k3.shape, "range X:", k3[...,0].min(), k3[...,0].max())
    print("2D shape:", k2.shape, "range X:", k2[...,0].min(), k2[...,0].max())
else:
    print("No files found.")
