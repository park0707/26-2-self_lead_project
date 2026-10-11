# -*- coding: utf-8 -*-
import os
import glob
import json
import numpy as np

def check_all_dirs():
    print("1. Raw Images:")
    img_dir = r"D:\Data\01_raw_data\images\08_크로스_런지"
    if os.path.exists(img_dir):
        folders = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
        print(f"  Count: {len(folders)} folders")
        if folders:
            print(f"  Sample folders: {folders[:5]}")
            # check images inside sample folder
            s_files = os.listdir(os.path.join(img_dir, folders[0]))
            print(f"  Sample folder {folders[0]} files: {len(s_files)}, first 5: {s_files[:5]}")
    else:
        print("  Not found!")

    print("\n2. MediaPipe Preprocessed:")
    mp_dir = r"D:\Data\전처리_데이터\MediaPipe\08_크로스_런지"
    if os.path.exists(mp_dir):
        files = glob.glob(os.path.join(mp_dir, "*.npz"))
        print(f"  Count: {len(files)} npz files")
        if files:
            print(f"  Sample files: {[os.path.basename(f) for f in files[:5]]}")
            data = np.load(files[0], allow_pickle=True)
            print(f"  Keys: {list(data.keys())}")
            for k in data.keys():
                item = data[k]
                if hasattr(item, 'shape'):
                    print(f"    {k}: shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"    {k}: type={type(item)}, val={str(item)[:100]}")
    else:
        print("  Not found!")

    print("\n3. MediaPipe Full33 64F:")
    mp33_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"
    if os.path.exists(mp33_dir):
        files = glob.glob(os.path.join(mp33_dir, "*.npz"))
        print(f"  Count: {len(files)} npz files")
    else:
        print("  Directory does not exist yet!")

    print("\n4. AIHub Preprocessed:")
    aihub_dir = r"D:\Data\전처리_데이터\AIHub\08_크로스_런지"
    if os.path.exists(aihub_dir):
        files = glob.glob(os.path.join(aihub_dir, "**", "*.npz"), recursive=True)
        print(f"  Count: {len(files)} npz files")
        if files:
            print(f"  Sample: {files[0]}")
            data = np.load(files[0], allow_pickle=True)
            print(f"  Keys: {list(data.keys())}")
    else:
        print("  Not found!")

if __name__ == "__main__":
    check_all_dirs()
