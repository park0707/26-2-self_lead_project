import os
import glob

base = r"D:\Data\01_raw_data\images\02_바벨_데드리프트"
dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
print(f"Total image folders: {len(dirs)}")
print("Sample folders:", dirs[:5])

# 피험자 7, 피험자 5 폴더 찾기
for d in dirs:
    if "피험자7" in d or "7" in d:
        sub_dirs = os.listdir(os.path.join(base, d))
        print(f"Found folder: {d}, subdirs: {sub_dirs}")
        break
