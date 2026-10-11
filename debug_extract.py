import sys
sys.path.insert(0, 'd:/')
sys.path.insert(0, r'D:\src\models\second_models\babel_lunge')
import glob
import traceback
f = glob.glob(r'D:/Data/전처리_데이터/RTMPose_Halpe26_64F/04_바벨_런지/*.npz')[0]
from extract_fast import process_file
try:
    process_file(f)
    print("Success")
except Exception as e:
    traceback.print_exc()
