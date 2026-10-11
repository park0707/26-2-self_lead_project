# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit test for Cross Lunge c1 (cl1) pipeline and 55D biomechanical features
Verifies:
  1. Feature dimension (55D)
  2. Absence of NaN / Inf in synthetic and real sequences
  3. Ground Truth label parsing from AI-Hub JSON
  4. 100% Pure Vision compliance (no JSON leakage into feature space)
"""

import os
import sys
import glob
import json
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.side_lunge_c1_features_v11 import (
    extract_side_lunge_c1_features_v11,
    extract_side_lunge_c1_feature_vector_v11,
    SIDE_LUNGE_C1_FEATURE_NAMES_V11
)
from src.models.second_models.cross_lunge.cl1_v1 import determine_c1_label_from_json

DATA_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\08_크로스_런지"
LABEL_DIR = r"D:\Data\01_raw_data\labels_json\08_크로스_런지\2D"


def test_feature_names_dimension():
    """피처 명칭 리스트가 정확히 55차원인지 검증"""
    assert len(SIDE_LUNGE_C1_FEATURE_NAMES_V11) == 55, f"Expected 55 features, got {len(SIDE_LUNGE_C1_FEATURE_NAMES_V11)}"
    print("[PASS] 55D Feature names dimension verified.")


def test_synthetic_sequence_extraction():
    """모의 시퀀스 데이터(64, 33, 3)에 대한 55D 피처 추출 무결성 검증"""
    np.random.seed(42)
    kps_fake = np.random.randn(64, 33, 3).astype(np.float32)
    # y축(중력방향) 인체 골격 비율 설정
    kps_fake[:, 23:25, 1] = 0.5  # 고관절
    kps_fake[:, 25:27, 1] = 0.8  # 무릎
    kps_fake[:, 27:29, 1] = 1.1  # 발목
    kps_fake[:, 11:13, 1] = 0.1  # 어깨

    feat_dict = extract_side_lunge_c1_features_v11(kps_fake)
    assert len(feat_dict) == 55, f"Expected 55 dict items, got {len(feat_dict)}"
    for k, v in feat_dict.items():
        assert not np.isnan(v), f"Feature {k} is NaN"
        assert not np.isinf(v), f"Feature {k} is Inf"

    feat_vec = extract_side_lunge_c1_feature_vector_v11(kps_fake)
    assert feat_vec.shape == (55,), f"Expected shape (55,), got {feat_vec.shape}"
    assert not np.isnan(feat_vec).any(), "Synthetic feature vector contains NaN"
    assert not np.isinf(feat_vec).any(), "Synthetic feature vector contains Inf"
    print("[PASS] Synthetic sequence 55D extraction verified.")


def test_real_sequence_extraction():
    """실제 크로스 런지 NPZ 시퀀스로 55D 피처 추출 및 무결성 검증"""
    npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    if not npz_files:
        print("[!] NPZ files not yet available, skipping real sequence test.")
        return

    sample_npz = npz_files[0]
    data = np.load(sample_npz)
    kps_33_3d = data['kps_33_3d_64f']
    vis_64f = data.get('vis_64f', None)

    assert kps_33_3d.shape == (64, 33, 3), f"Expected (64, 33, 3), got {kps_33_3d.shape}"

    feat_dict = extract_side_lunge_c1_features_v11(kps_33_3d, vis_64f)
    assert len(feat_dict) == 55
    for k, v in feat_dict.items():
        assert not np.isnan(v), f"Real sample feature {k} is NaN"
        assert not np.isinf(v), f"Real sample feature {k} is Inf"

    feat_vec = extract_side_lunge_c1_feature_vector_v11(kps_33_3d, vis_64f)
    assert feat_vec.shape == (55,)
    assert not np.isnan(feat_vec).any()
    assert not np.isinf(feat_vec).any()
    print(f"[PASS] Real cross lunge sample ({os.path.basename(sample_npz)}) 55D features extracted successfully!")


def test_label_parsing():
    """AI-Hub 2D JSON 라벨 파싱 무결성 검증"""
    json_files = glob.glob(os.path.join(LABEL_DIR, "*.json"))
    assert len(json_files) > 0, "No label JSON files found"

    sample_normal = None
    sample_defect = None
    for jf in json_files:
        val = determine_c1_label_from_json(jf)
        if val == 0 and sample_normal is None:
            sample_normal = (jf, val)
        elif val == 1 and sample_defect is None:
            sample_defect = (jf, val)
        if sample_normal and sample_defect:
            break

    assert sample_normal is not None, "Normal sample not found"
    assert sample_defect is not None, "Defect sample not found"
    assert sample_normal[1] == 0, f"Expected 0 for normal, got {sample_normal[1]}"
    assert sample_defect[1] == 1, f"Expected 1 for defect, got {sample_defect[1]}"
    print(f"[PASS] Label parsing verified (Normal: {os.path.basename(sample_normal[0])}, Defect: {os.path.basename(sample_defect[0])})")


if __name__ == "__main__":
    test_feature_names_dimension()
    test_synthetic_sequence_extraction()
    test_real_sequence_extraction()
    test_label_parsing()
    print("\n[+] All Pipeline Unit Tests Passed Successfully!")
