import os
import glob
import time

d = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
files = glob.glob(os.path.join(d, "*.npz"))
print(f"[{time.strftime('%H:%M:%S')}] Extracted: {len(files)} / 1040 ({len(files)/1040*100:.1f}%)")
