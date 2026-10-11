# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit test for bl5_v1.py (Barbell Lunge Defect 5 - Torso Sagittal Neutrality Classifier)
"""

import os
import sys
import glob
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_lunge.bl5_v1 import BL5V1Classifier


def test_bl5_v1():
    print("=" * 60)
    print("  [Test] bl5_v1 Production Classifier Unit Test")
    print("=" * 60)

    # 1. 모델 인스턴스화
    clf = BL5V1Classifier()
    assert clf.is_loaded, f"모델 가중치 번들이 로드되지 않았습니다: {clf.bundle_path}"
    print(f"[+] 모델 가중치 번들 정상 로드 완료: {clf.bundle_path}")
    print(f"    - LightGBM 모델 수: {len(clf.lgb_models)}")
    print(f"    - CatBoost 모델 수: {len(clf.cb_models)}")
    print(f"    - Random Forest 모델 수: {len(clf.rf_models)}")

    # 2. 테스트용 64F 시퀀스 파일 탐색
    cache_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\04_바벨_런지"
    sample_files = glob.glob(os.path.join(cache_dir, "*.npz"))
    assert len(sample_files) > 0, "테스트용 샘플 npz 파일이 없습니다."

    sample_p = sample_files[0]
    print(f"[*] 테스트 샘플 파일: {os.path.basename(sample_p)}")
    data = np.load(sample_p)
    kps_33_3d = data['kps_33_3d_64f']
    assert kps_33_3d.shape == (64, 33, 3), f"잘못된 shape: {kps_33_3d.shape}"

    # 3. 32D 피처 추출 테스트
    feats = clf.extract_features(kps_33_3d)
    assert len(feats) == 32, f"피처 수 불일치: {len(feats)} != 32"
    print(f"[+] 32차원 도메인 피처 추출 성공 (피치 윈도우: {feats['lunge_torso_pitch_bottom_win']:.2f}°)")

    # 4. 5대 공식 시점별 3-State 추론 테스트
    official_views = ['시점 A', '시점 B', '시점 C', '시점 D', '시점 E']
    for v_name in official_views:
        res = clf.predict_3state(kps_33_3d, view=v_name)
        assert res['state'] in ['Clean', 'Defect', 'Retake'], f"유효하지 않은 상태: {res['state']}"
        assert 0.0 <= res['prob'] <= 1.0, f"유효하지 않은 확률: {res['prob']}"
        assert 0.0 <= res['confidence'] <= 1.0, f"유효하지 않은 확신도: {res['confidence']}"
        assert len(res['feedback']) > 10, "피드백 메시지 누락"
        assert len(res['features_summary']) == 6, "지표 요약 누락"
        print(f"  * {v_name} 추론 결과: 상태={res['state']}, 확률={res['prob']:.4f}, 확신도={res['confidence']:.4f}")

    print("\n[+] 3-State 상세 반환 명세 확인:")
    sample_res = clf.predict_3state(kps_33_3d, view='시점 B')
    for k, v in sample_res.items():
        print(f"    - {k}: {v}")

    print("\n[SUCCESS] bl5_v1 프로덕션 추론 엔진 단위 테스트를 100% 통과하였습니다!")


if __name__ == "__main__":
    test_bl5_v1()
