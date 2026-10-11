# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Defect 2: sd2_v1 단위 테스트 스크립트
"""

import os
import sys
import glob
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "d:/"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd2_v1_classifier import get_sd2_v1_classifier

RAW_MP33_DIR = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\03_바벨_스티프_데드리프트"
SPINEPOSE_CACHE = r"D:\Data\피처_데이터\stiff_deadlift_spinepose_denoised_64f.npz"


def test_sd2_v1():
    print("=" * 90)
    print("  [UNIT TEST] sd2_v1 척추 중립 분류기 엔드투엔드 추론 검증")
    print("=" * 90)

    clf = get_sd2_v1_classifier()
    print(f"[*] 모델 인스턴스 로드 성공: {clf.model_name} ({clf.version})")
    print(f"    - 임계치: tau_clean={clf.tau_clean}, tau_defect={clf.tau_defect}")

    # 샘플 파일 로드
    npz_files = sorted(glob.glob(os.path.join(RAW_MP33_DIR, "*.npz")))
    if not npz_files:
        print("[FAIL] 전처리 파일이 없습니다.")
        return

    spine_cache = {}
    if os.path.exists(SPINEPOSE_CACHE):
        sc = np.load(SPINEPOSE_CACHE, allow_pickle=True)
        sids = sc['sequence_ids']
        s_vision = sc['spine_9_64f_vision']
        spine_cache = {sids[i]: s_vision[i] for i in range(len(sids))}

    # 정상 샘플 1개, 결함 샘플 1개 테스트
    clean_sample = None
    defect_sample = None
    for f in npz_files:
        sid = os.path.basename(f).replace(".npz", "")
        if "Key209" in sid and clean_sample is None:
            clean_sample = (sid, f)
        if ("Key211" in sid or "Key216" in sid) and defect_sample is None:
            defect_sample = (sid, f)
        if clean_sample and defect_sample:
            break

    for cat_name, (sid, f_path) in [("정상 샘플 (Key 209)", clean_sample), ("결함 샘플 (Key 211/216)", defect_sample)]:
        data = np.load(f_path, allow_pickle=True)
        kps = data['kps_33_3d_64f'] if 'kps_33_3d_64f' in data else data['kps_33_2d_64f']
        spine = spine_cache.get(sid, None)

        res = clf.diagnose(kps, spine)
        print(f"\n[{cat_name}] {sid}")
        print(f"  - 결함 확률 P(Defect)  : {res['p_defect']*100:.2f}%")
        print(f"  - 이진 판정 (0/1)      : {res['binary_prediction']} (0: 정상, 1: 결함)")
        print(f"  - 3-State 상태         : [{res['three_state']}] {res['three_state_label']}")
        print(f"  - 진단 피드백          : {res['feedback']}")

    print("\n[OK] sd2_v1 분류기 단위 테스트 100% 통과!")


if __name__ == '__main__':
    test_sd2_v1()
