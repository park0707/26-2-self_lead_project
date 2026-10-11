# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Unit Tests: tests/test_stiff_deadlift_rtmp.py
Purpose: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트) RTMPose 전용 모델 단위 테스트
         - sd2 (척추의 중립 위반 / Spine Neutrality Violation, v3.0 RTMP)
         - sd3 (바벨 궤적과 몸 밀착 위반 / Barbell Path Proximity Defect, v4.0 RTMP)
Verification Items:
1. SD2RTMPClassifier:
   - (64, 26, 3) Halpe26 정상 텐서 입력 시 extract_features, predict_proba, diagnose 검증
   - (64, 33, 3) MediaPipe 호환 텐서 입력 시 정상 동작 검증
   - 3-State 상태값 (0, 1, 2) 및 피드백 딕셔너리 규격 무결성 검증
2. SD3RTMPClassifier:
   - (64, 26, 3) 및 (64, 33, 3) 입력 시 5대 시점(view1~view5)별 predict_proba, diagnose 검증
   - 3-State 상태값 (0, 1, 2) 및 reasons, feedback 필드 무결성 검증
3. Boundary & Edge Case (Zeros, NaNs 방어) 검증
"""

import os
import sys
import unittest
from unittest.mock import patch
import numpy as np

PROJECT_ROOT = r"D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_stiff_deadlift.sd2_model_v3_rtmp import (
    SD2RTMPClassifier,
    FEATURE_NAMES_30_SD2_RTMP,
    extract_sd2_rtmp_features,
    extract_sd2_rtmp_temporal_24d,
)
from src.models.second_models.babel_stiff_deadlift.sd3_model_v4_rtmp import (
    SD3RTMPClassifier,
    FEATURE_NAMES_36_SD3_RTMP,
    VIEW_CODE_MAP,
    extract_sd3_rtmp_features,
    extract_sd3_rtmp_temporal_36d,
)


def create_synthetic_stiff_deadlift_halpe26(n_frames: int = 64) -> np.ndarray:
    """
    Halpe26 골격 구조에 맞춘 합성 바벨 스티프 데드리프트 동작 텐서 생성
    Shape: (n_frames, 26, 3)
    """
    seq = np.zeros((n_frames, 26, 3), dtype=np.float32)
    t = np.linspace(0, np.pi, n_frames)
    hip_y = 0.50 + 0.05 * np.sin(t)
    torso_tilt = 0.35 * np.sin(t)
    wrist_y = 0.55 + 0.30 * np.sin(t)

    # 기본 신뢰도 0.95
    seq[:, :, 2] = 0.95

    # NOSE
    seq[:, 0, 0] = 0.50
    seq[:, 0, 1] = 0.20 - torso_tilt * 0.4
    # HEAD
    seq[:, 17, 0] = 0.50
    seq[:, 17, 1] = 0.15 - torso_tilt * 0.4
    # NECK
    seq[:, 18, 0] = 0.50
    seq[:, 18, 1] = 0.25 - torso_tilt * 0.4
    # LSH, RSH
    seq[:, 5, 0] = 0.42
    seq[:, 5, 1] = 0.28 - torso_tilt * 0.3
    seq[:, 6, 0] = 0.58
    seq[:, 6, 1] = 0.28 - torso_tilt * 0.3
    # LELB, RELB
    seq[:, 7, 0] = 0.40
    seq[:, 7, 1] = 0.40
    seq[:, 8, 0] = 0.60
    seq[:, 8, 1] = 0.40
    # LWR, RWR
    seq[:, 9, 0] = 0.40
    seq[:, 9, 1] = wrist_y
    seq[:, 10, 0] = 0.60
    seq[:, 10, 1] = wrist_y
    # HIP (Pelvis)
    seq[:, 19, 0] = 0.50
    seq[:, 19, 1] = hip_y
    # LHIP, RHIP
    seq[:, 11, 0] = 0.44
    seq[:, 11, 1] = hip_y
    seq[:, 12, 0] = 0.56
    seq[:, 12, 1] = hip_y
    # LKNE, RKNE
    seq[:, 13, 0] = 0.44
    seq[:, 13, 1] = 0.70
    seq[:, 14, 0] = 0.56
    seq[:, 14, 1] = 0.70
    # LANK, RANK
    seq[:, 15, 0] = 0.44
    seq[:, 15, 1] = 0.90
    seq[:, 16, 0] = 0.56
    seq[:, 16, 1] = 0.90
    # BIG_TOE
    seq[:, 20, 0] = 0.44
    seq[:, 20, 1] = 0.95
    seq[:, 21, 0] = 0.56
    seq[:, 21, 1] = 0.95
    # HEEL
    seq[:, 24, 0] = 0.44
    seq[:, 24, 1] = 0.92
    seq[:, 25, 0] = 0.56
    seq[:, 25, 1] = 0.92

    return seq


def create_synthetic_stiff_deadlift_mp33(n_frames: int = 64) -> np.ndarray:
    """
    MediaPipe 33 골격 구조에 맞춘 합성 바벨 스티프 데드리프트 동작 텐서 생성
    Shape: (n_frames, 33, 3)
    """
    seq = np.zeros((n_frames, 33, 3), dtype=np.float32)
    t = np.linspace(0, np.pi, n_frames)
    hip_y = 0.50 + 0.05 * np.sin(t)
    torso_tilt = 0.35 * np.sin(t)
    wrist_y = 0.55 + 0.30 * np.sin(t)

    seq[:, :, 2] = 0.95

    # NOSE
    seq[:, 0, 0] = 0.50
    seq[:, 0, 1] = 0.20 - torso_tilt * 0.4
    # LEAR, REAR
    seq[:, 7, 0] = 0.46
    seq[:, 7, 1] = 0.22
    seq[:, 8, 0] = 0.54
    seq[:, 8, 1] = 0.22
    # LSH, RSH
    seq[:, 11, 0] = 0.42
    seq[:, 11, 1] = 0.28 - torso_tilt * 0.3
    seq[:, 12, 0] = 0.58
    seq[:, 12, 1] = 0.28 - torso_tilt * 0.3
    # LELB, RELB
    seq[:, 13, 0] = 0.40
    seq[:, 13, 1] = 0.40
    seq[:, 14, 0] = 0.60
    seq[:, 14, 1] = 0.40
    # LWR, RWR
    seq[:, 15, 0] = 0.40
    seq[:, 15, 1] = wrist_y
    seq[:, 16, 0] = 0.60
    seq[:, 16, 1] = wrist_y
    # LHIP, RHIP
    seq[:, 23, 0] = 0.44
    seq[:, 23, 1] = hip_y
    seq[:, 24, 0] = 0.56
    seq[:, 24, 1] = hip_y
    # LKNE, RKNE
    seq[:, 25, 0] = 0.44
    seq[:, 25, 1] = 0.70
    seq[:, 26, 0] = 0.56
    seq[:, 26, 1] = 0.70
    # LANK, RANK
    seq[:, 27, 0] = 0.44
    seq[:, 27, 1] = 0.90
    seq[:, 28, 0] = 0.56
    seq[:, 28, 1] = 0.90

    return seq


class TestSD2RTMPClassifier(unittest.TestCase):
    """SD2 (척추의 중립 위반) RTMPose 분류기 단위 테스트"""

    @classmethod
    def setUpClass(cls):
        cls.classifier = SD2RTMPClassifier()
        cls.halpe26_sample = create_synthetic_stiff_deadlift_halpe26(64)
        cls.mp33_sample = create_synthetic_stiff_deadlift_mp33(64)

    def test_initialization_and_metadata(self):
        """모델 초기화 및 메타데이터 정합성 검증"""
        self.assertEqual(self.classifier.defect_code, 'sd2')
        self.assertEqual(len(self.classifier.feature_names), 30)
        self.assertEqual(self.classifier.feature_names, FEATURE_NAMES_30_SD2_RTMP)
        self.assertGreater(self.classifier.tau_clean, 0.0)
        self.assertLess(self.classifier.tau_clean, self.classifier.tau_defect)
        self.assertLessEqual(self.classifier.tau_defect, 1.0)
        self.assertGreater(len(self.classifier.tcn_models), 0)
        self.assertGreater(len(self.classifier.lgb_models), 0)
        self.assertGreater(len(self.classifier.cat_models), 0)

    def test_initialization_invalid_path(self):
        """존재하지 않는 가중치 경로 입력 시 FileNotFoundError 발생 검증"""
        with self.assertRaises(FileNotFoundError):
            SD2RTMPClassifier(bundle_path="invalid_path_sd2.pkl")

    def test_extract_features_halpe26(self):
        """Halpe26 (64, 26, 3) 텐서 입력 시 피처 추출 정합성 검증"""
        features_dict = self.classifier.extract_features(self.halpe26_sample)
        self.assertIn('features_30d', features_dict)
        self.assertIn('temporal_24d', features_dict)

        f30 = features_dict['features_30d']
        t24 = features_dict['temporal_24d']

        self.assertEqual(f30.shape, (30,))
        self.assertEqual(t24.shape, (64, 24))
        self.assertEqual(f30.dtype, np.float32)
        self.assertFalse(np.isnan(f30).any(), "f30 should not contain NaN")
        self.assertFalse(np.isinf(f30).any(), "f30 should not contain Inf")
        self.assertFalse(np.isnan(t24).any(), "t24 should not contain NaN")

    def test_extract_features_mp33(self):
        """MediaPipe33 (64, 33, 3) 텐서 호환 입력 시 피처 추출 정합성 검증"""
        features_dict = self.classifier.extract_features(self.mp33_sample)
        self.assertIn('features_30d', features_dict)
        self.assertIn('temporal_24d', features_dict)

        f30 = features_dict['features_30d']
        t24 = features_dict['temporal_24d']

        self.assertEqual(f30.shape, (30,))
        self.assertEqual(t24.shape, (64, 24))
        self.assertFalse(np.isnan(f30).any(), "MP33 f30 should not contain NaN")
        self.assertFalse(np.isinf(f30).any(), "MP33 f30 should not contain Inf")
        self.assertFalse(np.isnan(t24).any(), "MP33 t24 should not contain NaN")

    def test_predict_proba_halpe26_and_mp33(self):
        """predict_proba 확률값 범위 [0.0, 1.0] 및 float 타입 검증"""
        prob_halpe = self.classifier.predict_proba(self.halpe26_sample)
        self.assertIsInstance(prob_halpe, float)
        self.assertGreaterEqual(prob_halpe, 0.0)
        self.assertLessEqual(prob_halpe, 1.0)
        self.assertFalse(np.isnan(prob_halpe))

        prob_mp = self.classifier.predict_proba(self.mp33_sample)
        self.assertIsInstance(prob_mp, float)
        self.assertGreaterEqual(prob_mp, 0.0)
        self.assertLessEqual(prob_mp, 1.0)
        self.assertFalse(np.isnan(prob_mp))

    def test_diagnose_dict_structure_halpe26(self):
        """Halpe26 입력 시 diagnose 출력 딕셔너리 규격 및 타입 검증"""
        res = self.classifier.diagnose(self.halpe26_sample)
        required_keys = {'state', 'state_label', 'probability', 'binary_prediction', 'detected_sub_defects', 'feedback'}
        self.assertTrue(required_keys.issubset(res.keys()))

        self.assertIn(res['state'], [0, 1, 2])
        self.assertIn(res['state_label'], ['CLEAN', 'DEFECT', 'RETAKE'])
        self.assertIsInstance(res['probability'], float)
        self.assertGreaterEqual(res['probability'], 0.0)
        self.assertLessEqual(res['probability'], 1.0)
        self.assertIn(res['binary_prediction'], [0, 1])
        self.assertIsInstance(res['detected_sub_defects'], list)
        self.assertIsInstance(res['feedback'], str)
        self.assertGreater(len(res['feedback']), 0)

    def test_diagnose_dict_structure_mp33(self):
        """MediaPipe33 호환 입력 시 diagnose 출력 딕셔너리 규격 검증"""
        res = self.classifier.diagnose(self.mp33_sample)
        self.assertIn(res['state'], [0, 1, 2])
        self.assertIn(res['state_label'], ['CLEAN', 'DEFECT', 'RETAKE'])
        self.assertIsInstance(res['probability'], float)
        self.assertIsInstance(res['feedback'], str)

    def test_three_state_decision_and_feedback_integrity(self):
        """3-State (CLEAN: 0, DEFECT: 1, RETAKE: 2) 의사결정 및 피드백 메시지 무결성 검증"""
        tau_c = self.classifier.tau_clean
        tau_d = self.classifier.tau_defect

        # State 0 (CLEAN): prob < tau_clean
        with patch.object(self.classifier, 'predict_proba', return_value=max(0.0, tau_c - 0.05)):
            res_clean = self.classifier.diagnose(self.halpe26_sample)
            self.assertEqual(res_clean['state'], 0)
            self.assertEqual(res_clean['state_label'], 'CLEAN')
            self.assertEqual(res_clean['binary_prediction'], 0)
            self.assertIn('척추의 중립이 전 구간에서 훌륭하게 유지', res_clean['feedback'])

        # State 1 (DEFECT): prob >= tau_defect
        with patch.object(self.classifier, 'predict_proba', return_value=min(1.0, tau_d + 0.05)):
            res_defect = self.classifier.diagnose(self.halpe26_sample)
            self.assertEqual(res_defect['state'], 1)
            self.assertEqual(res_defect['state_label'], 'DEFECT')
            self.assertEqual(res_defect['binary_prediction'], 1)
            self.assertTrue(
                ('척추 중립 위반 감지' in res_defect['feedback']) or
                ('척추가 굽거나 과도하게 젖혀졌습니다' in res_defect['feedback'])
            )

        # State 2 (RETAKE): tau_clean <= prob < tau_defect
        mid_prob = (tau_c + tau_d) / 2.0
        with patch.object(self.classifier, 'predict_proba', return_value=mid_prob):
            res_retake = self.classifier.diagnose(self.halpe26_sample)
            self.assertEqual(res_retake['state'], 2)
            self.assertEqual(res_retake['state_label'], 'RETAKE')
            self.assertIn('척추 정렬이 다소 모호하게 관측', res_retake['feedback'])

    def test_sub_defects_detection_logic(self):
        """sd2 서브 결함(바벨 좌우 기울임, 허리 과신전, 등 굽힘) 감지 조건 검증"""
        # 임의로 결함 수치가 주입된 피처로 검증
        mock_features = np.zeros(30, dtype=np.float32)
        mock_features[18] = 5.0   # barbell_roll_max >= 4.0
        mock_features[23] = 4.0   # lockout_posterior_lean >= 3.0
        mock_features[27] = 0.06  # thoracic_kyphosis_peak >= 0.04

        with patch.object(self.classifier, 'extract_features', return_value={
            'features_30d': mock_features,
            'temporal_24d': np.zeros((64, 24), dtype=np.float32)
        }):
            with patch.object(self.classifier, 'predict_proba', return_value=0.95):
                res = self.classifier.diagnose(self.halpe26_sample)
                self.assertEqual(res['state'], 1)
                sub_defects = res['detected_sub_defects']
                self.assertIn("바벨 좌우 기울임(Barbell Tilt Asymmetry)", sub_defects)
                self.assertIn("기립 락아웃 시 허리 과신전(Lumbar Hyperextension)", sub_defects)
                self.assertIn("상체 하강 시 등/어깨 굽힘(Round Back / Thoracic Kyphosis)", sub_defects)
                self.assertIn("바벨 좌우 기울임", res['feedback'])

    def test_edge_case_all_zeros(self):
        """모든 키포인트가 0인 극단적 비정상 입력 시 안정성 (NaN/예외 발생 방지) 검증"""
        zero_tensor = np.zeros((64, 26, 3), dtype=np.float32)
        feats = self.classifier.extract_features(zero_tensor)
        self.assertFalse(np.isnan(feats['features_30d']).any())
        self.assertFalse(np.isinf(feats['features_30d']).any())

        prob = self.classifier.predict_proba(zero_tensor)
        self.assertFalse(np.isnan(prob))
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

        diag = self.classifier.diagnose(zero_tensor)
        self.assertIn(diag['state'], [0, 1, 2])


class TestSD3RTMPClassifier(unittest.TestCase):
    """SD3 (바벨 궤적과 몸 밀착 위반) RTMPose 분류기 단위 테스트"""

    @classmethod
    def setUpClass(cls):
        cls.classifier = SD3RTMPClassifier()
        cls.views = ['view1', 'view2', 'view3', 'view4', 'view5']
        cls.halpe26_sample = create_synthetic_stiff_deadlift_halpe26(64)
        cls.mp33_sample = create_synthetic_stiff_deadlift_mp33(64)

    def test_initialization_and_metadata(self):
        """모델 초기화 및 멀티뷰 설정 메타데이터 검증"""
        self.assertEqual(self.classifier.defect_code, 'sd3')
        self.assertEqual(len(self.classifier.feature_names), 36)
        self.assertEqual(self.classifier.feature_names, FEATURE_NAMES_36_SD3_RTMP)
        self.assertEqual(set(VIEW_CODE_MAP.keys()), set(self.views))
        self.assertEqual(set(self.classifier.view_thresholds.keys()), set(self.views))
        self.assertEqual(set(self.classifier.view_fusion_weights.keys()), set(self.views))
        self.assertGreater(len(self.classifier.tcn_models), 0)
        self.assertGreater(len(self.classifier.lgb_models), 0)
        self.assertGreater(len(self.classifier.cb_models), 0)

    def test_initialization_invalid_path(self):
        """존재하지 않는 가중치 파일 로드시 FileNotFoundError 예외 검증"""
        with self.assertRaises(FileNotFoundError):
            SD3RTMPClassifier(bundle_path="invalid_path_sd3.pkl")

    def test_extract_features_halpe26_all_views(self):
        """Halpe26 텐서 입력 시 5대 시점별 피처 추출 및 규격 검증"""
        for v in self.views:
            with self.subTest(view=v):
                extracted = self.classifier.extract_features(self.halpe26_sample, view_name=v)
                self.assertIn('features_36d', extracted)
                self.assertIn('temporal_36d', extracted)

                f36 = extracted['features_36d']
                t36 = extracted['temporal_36d']

                self.assertEqual(f36.shape, (36,))
                self.assertEqual(t36.shape, (64, 36))
                self.assertEqual(f36.dtype, np.float32)
                self.assertFalse(np.isnan(f36).any(), f"f36 should not contain NaN for {v}")
                self.assertFalse(np.isinf(f36).any(), f"f36 should not contain Inf for {v}")
                self.assertFalse(np.isnan(t36).any(), f"t36 should not contain NaN for {v}")

                # 시점 코드 피처 [23] 검증
                expected_vcode = VIEW_CODE_MAP[v]
                self.assertEqual(f36[23], expected_vcode)

    def test_extract_features_mp33_all_views(self):
        """MediaPipe33 호환 텐서 입력 시 5대 시점별 피처 추출 검증"""
        for v in self.views:
            with self.subTest(view=v):
                extracted = self.classifier.extract_features(self.mp33_sample, view_name=v)
                self.assertIn('features_36d', extracted)
                self.assertIn('temporal_36d', extracted)

                f36 = extracted['features_36d']
                t36 = extracted['temporal_36d']

                self.assertEqual(f36.shape, (36,))
                self.assertEqual(t36.shape, (64, 36))
                self.assertFalse(np.isnan(f36).any(), f"MP33 f36 should not contain NaN for {v}")
                self.assertFalse(np.isinf(f36).any(), f"MP33 f36 should not contain Inf for {v}")
                self.assertFalse(np.isnan(t36).any(), f"MP33 t36 should not contain NaN for {v}")

    def test_multiview_predict_proba_halpe26(self):
        """Halpe26 입력 시 5대 시점(view1~view5)별 predict_proba 정상 동작 및 범위 검증"""
        for v in self.views:
            with self.subTest(view=v):
                p = self.classifier.predict_proba(self.halpe26_sample, view_name=v)
                self.assertIsInstance(p, float)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)
                self.assertFalse(np.isnan(p))

    def test_multiview_predict_proba_mp33(self):
        """MediaPipe33 호환 입력 시 5대 시점(view1~view5)별 predict_proba 검증"""
        for v in self.views:
            with self.subTest(view=v):
                p = self.classifier.predict_proba(self.mp33_sample, view_name=v)
                self.assertIsInstance(p, float)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)
                self.assertFalse(np.isnan(p))

    def test_multiview_diagnose_structure(self):
        """5대 시점별 diagnose 출력 딕셔너리 규격, reasons, feedback 필드 검증"""
        for v in self.views:
            with self.subTest(view=v):
                res = self.classifier.diagnose(self.halpe26_sample, view_name=v)
                required_keys = {'state', 'state_label', 'probability', 'binary_prediction', 'reasons', 'feedback'}
                self.assertTrue(required_keys.issubset(res.keys()))

                self.assertIn(res['state'], [0, 1, 2])
                self.assertIn(res['state_label'], ['CLEAN', 'DEFECT', 'RETAKE'])
                self.assertIsInstance(res['probability'], float)
                self.assertGreaterEqual(res['probability'], 0.0)
                self.assertLessEqual(res['probability'], 1.0)
                self.assertIn(res['binary_prediction'], [0, 1])
                self.assertIsInstance(res['reasons'], list)
                self.assertIsInstance(res['feedback'], str)
                self.assertGreater(len(res['feedback']), 0)

    def test_three_state_decision_and_feedback_integrity_all_views(self):
        """5대 시점 각각에 대한 3-State 상태값 (0, 1, 2) 및 피드백 메시지 무결성 검증"""
        for v in self.views:
            with self.subTest(view=v):
                tau_c, tau_d = self.classifier.view_thresholds[v]

                # State 0 (CLEAN): prob < tau_c
                with patch.object(self.classifier, 'predict_proba', return_value=max(0.0, tau_c - 0.05)):
                    res_clean = self.classifier.diagnose(self.halpe26_sample, view_name=v)
                    self.assertEqual(res_clean['state'], 0)
                    self.assertEqual(res_clean['state_label'], 'CLEAN')
                    self.assertEqual(res_clean['binary_prediction'], 0)
                    self.assertIn('바벨이 허벅지와 정강이를 따라 수직으로 완벽하게 밀착', res_clean['feedback'])

                # State 1 (DEFECT): prob >= tau_d
                with patch.object(self.classifier, 'predict_proba', return_value=min(1.0, tau_d + 0.05)):
                    res_defect = self.classifier.diagnose(self.halpe26_sample, view_name=v)
                    self.assertEqual(res_defect['state'], 1)
                    self.assertEqual(res_defect['state_label'], 'DEFECT')
                    self.assertEqual(res_defect['binary_prediction'], 1)
                    self.assertTrue(
                        ('바벨 이격 결함 감지' in res_defect['feedback']) or
                        ('바벨이 다리에서 멀어져' in res_defect['feedback'])
                    )

                # State 2 (RETAKE): tau_c <= prob < tau_d
                mid_prob = (tau_c + tau_d) / 2.0
                with patch.object(self.classifier, 'predict_proba', return_value=mid_prob):
                    res_retake = self.classifier.diagnose(self.halpe26_sample, view_name=v)
                    self.assertEqual(res_retake['state'], 2)
                    self.assertEqual(res_retake['state_label'], 'RETAKE')
                    self.assertIn('바벨 궤적이 카메라 앵글상 모호하게 관측', res_retake['feedback'])

    def test_reasons_field_triggers(self):
        """sd3 이격 결함 원인(정강이 이격, 허벅지 밀착 실패, 어깨-손목 경사) 트리거 검증"""
        mock_features = np.zeros(36, dtype=np.float32)
        mock_features[0] = 0.25   # shank_dist_max > 0.20
        mock_features[4] = 0.30   # thigh_dist_max > 0.25
        mock_features[12] = 22.0  # sh_wr_angle_max > 18.0

        with patch.object(self.classifier, 'extract_features', return_value={
            'features_36d': mock_features,
            'temporal_36d': np.zeros((64, 36), dtype=np.float32)
        }):
            with patch.object(self.classifier, 'predict_proba', return_value=0.90):
                res = self.classifier.diagnose(self.halpe26_sample, view_name='view2')
                self.assertEqual(res['state'], 1)
                self.assertIn("정강이 구간 바벨 과도 이격", res['reasons'])
                self.assertIn("허벅지 구간 바벨 밀착 실패", res['reasons'])
                self.assertIn("어깨-손목 전방 과도 경사 (광배근 텐션 부족)", res['reasons'])
                self.assertIn("바벨 이격 결함 감지", res['feedback'])

    def test_edge_case_all_zeros(self):
        """모든 키포인트가 0인 극단적 비정상 입력 시 안정성 검증"""
        zero_tensor = np.zeros((64, 26, 3), dtype=np.float32)
        for v in self.views:
            feats = self.classifier.extract_features(zero_tensor, view_name=v)
            self.assertFalse(np.isnan(feats['features_36d']).any())
            self.assertFalse(np.isinf(feats['features_36d']).any())

            prob = self.classifier.predict_proba(zero_tensor, view_name=v)
            self.assertFalse(np.isnan(prob))
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

            diag = self.classifier.diagnose(zero_tensor, view_name=v)
            self.assertIn(diag['state'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
