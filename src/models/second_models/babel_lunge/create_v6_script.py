# -*- coding: utf-8 -*-
"""bl4_v6_flip 학습 스크립트"""
import sys
import os
import shutil

src = r"D:\src\models\second_models\babel_lunge\train_and_eval_bl4_v4b.py"
dst = r"D:\src\models\second_models\babel_lunge\train_and_eval_bl4_v6_flip.py"

with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(r"lunge_c4_dataset_cache_v3.npz", r"lunge_c4_dataset_cache_flip.npz")
content = content.replace(r"bl4_v4b_oof_", r"bl4_v6_flip_oof_")
content = content.replace(r"bl4_v4b_models_", r"bl4_v6_flip_models_")

with open(dst, 'w', encoding='utf-8') as f:
    f.write(content)
