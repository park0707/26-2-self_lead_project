import numpy as np
import pickle
import os

path = r'D:\src\models\second_models\babel_stiff_deadlift\stiff_d1_v3_dataset_cache.pkl'
with open(path, 'rb') as f:
    c = pickle.load(f)

df = c['df']

# Since we don't have the 47D features loaded in this simple script easily,
# let's just use the feature extractor to get the knee angle for a few samples per key.
import sys
sys.path.append(r'D:\src\models\second_models\babel_stiff_deadlift')
from sd1_rtmp_2_features import extract_sd1_rtmp_2_features

import random

RTM_DIR = r"D:\Data\AIHub_Processed\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트"
RTM_DIR = r"D:\Data\스쿼트_등\RTMPose_Halpe26_64F\03_바벨_스티프_데드리프트" # let's just check the existing X matrix if possible

