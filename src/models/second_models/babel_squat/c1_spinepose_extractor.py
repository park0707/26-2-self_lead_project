# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스쿼트 (Barbell Squat)
Module: c1_spinepose_extractor.py
Purpose: 바벨 스쿼트 c1(척추의 중립 위반) 모델 전용 SpinePose 데이터 추출기
- 핵심 기능:
  1) 실시간 영상/이미지 입력 시: DFKI SpinePoseEstimator를 사용하여 37개 관절점 및 9대 척추 분절(S1, L1, L3, L5, T8, T3, C7, C4, C1) 직접 추출
  2) 64프레임 시계열 척추 9분절로부터 고정밀 생체역학 피처(다항식 곡률 a, 요추 Cobb각, S자 조화도, 접선 저크 등) 산출
  3) AI-Hub 1,906건 시퀀스와의 완벽한 수치적 호환성 제공
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from spinepose import SpinePoseEstimator
except ImportError:
    SpinePoseEstimator = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# SpinePose 공식 37개 랜드마크 중 척추 체인 9개 인덱스
# 19: hip (S1)
# 26: spine_01 (L5)
# 27: spine_02 (L3)
# 28: spine_03 (L1)
# 29: spine_04 (T8)
# 30: spine_05 (T3)
# 18: neck (C7)
# 35: neck_02 (C4)
# 36: neck_03 (C1)
SPINEPOSE_9_INDICES = [19, 26, 27, 28, 29, 30, 18, 35, 36]
SPINEPOSE_9_NAMES = ['S1', 'L5', 'L3', 'L1', 'T8', 'T3', 'C7', 'C4', 'C1']

C1_SPINEPOSE_DENSE_FEATURE_NAMES_6 = [
    'c1_v5_lumbar_curvature_poly_a',      # L1-L3-L5-S1 요추 2차 곡률 계수 (전만 음수 vs 후만 양수)
    'c1_v5_thoracolumbar_cobb_angle',     # 최하사점 L1-L5 요추 전만각 (Cobb angle proxy, deg)
    'c1_v5_spine_s_curve_harmony',        # 흉추 후만과 요추 전만의 S자 곡률 밸런스 비율
    'c1_v5_cervicothoracic_hinge_pitch',  # C1-C7 경추선과 T3-T8 흉추선의 이행부 사잇각 (deg)
    'c1_v5_nadir_lumbar_tangent_jerk',    # 최하사점 요추 접선각의 2차 미분 각가속도 급변도
    'c1_v5_pelvis_l5_retroversion_delta'  # 골반(S1) 대비 L5 요추체의 후방 말림(Retroversion) 변위
]


class BarbellSquatC1SpinePoseExtractor:
    """
    바벨 스쿼트 c1 모델 전용 SpinePose 데이터 추출 및 척추 피처 변환기
    """

    def __init__(
        self,
        mode: str = "small",
        enable_lifting: bool = True,
        device: str = "cpu"
    ):
        self.mode = mode
        self.enable_lifting = enable_lifting
        self.device = device
        self._estimator = None

    def _ensure_estimator(self):
        """SpinePoseEstimator 지연 초기화 (온디맨드 호출)"""
        if self._estimator is None and SpinePoseEstimator is not None:
            self._estimator = SpinePoseEstimator(
                mode=self.mode,
                enable_lifting=self.enable_lifting
            )

    def extract_from_image(
        self,
        img: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        단일 이미지로부터 SpinePose 37개 관절점 및 9대 척추 좌표 추출
        """
        self._ensure_estimator()
        if self._estimator is None:
            raise RuntimeError("spinepose 라이브러리가 설치되지 않았습니다.")

        if bbox is None:
            bboxes = self._estimator.detect(img)
            if len(bboxes) == 0:
                h, w = img.shape[:2]
                bbox = np.array([[0, 0, w, h]], dtype=np.float32)
            else:
                bbox = bboxes

        keypoints = self._estimator.estimate(img, bbox)
        # keypoints: (kps_2d, scores, kps_3d)
        kps_2d = keypoints[0][0] # (37, 2)
        scores = keypoints[1][0] # (37,)
        kps_3d = keypoints[2][0] if len(keypoints) > 2 else np.zeros((37, 3), dtype=np.float32)

        spine_9_2d = kps_2d[SPINEPOSE_9_INDICES]
        spine_9_3d = kps_3d[SPINEPOSE_9_INDICES]

        return {
            'kps_2d_37': kps_2d,
            'kps_3d_37': kps_3d,
            'scores_37': scores,
            'spine_9_2d': spine_9_2d,
            'spine_9_3d': spine_9_3d
        }

    @classmethod
    def interpolate_dense_spine_9(
        cls,
        neck_seq: np.ndarray,   # (T, 3)
        back_seq: np.ndarray,   # (T, 3)
        waist_seq: np.ndarray,  # (T, 3)
        pelvis_seq: np.ndarray  # (T, 3)
    ) -> np.ndarray:
        """
        AI-Hub 4개 척추점(Pelvis, Waist, Back, Neck)으로부터
        SpinePose 규격 9개 분절(S1, L5, L3, L1, T8, T3, C7, C4, C1) 3D 좌표를
        PCHIP(Monotonic Cubic Spline)으로 고정밀 복원: (T, 9, 3)
        """
        T = len(neck_seq)
        spine_9 = np.zeros((T, 9, 3), dtype=np.float32)

        # 4개 기준 관절의 상대 척추 고도 인덱스 (0.0 ~ 1.0)
        # S1(0.0), Waist/L3(0.25), Back/T8(0.60), Neck/C7(0.85)
        # 추정 타깃 9개 고도:
        # S1=0.0, L5=0.12, L3=0.25, L1=0.42, T8=0.60, T3=0.75, C7=0.85, C4=0.92, C1=1.00
        z_src = np.array([0.0, 0.25, 0.60, 0.85], dtype=np.float32)
        z_dst = np.array([0.0, 0.12, 0.25, 0.42, 0.60, 0.75, 0.85, 0.92, 1.00], dtype=np.float32)

        for t in range(T):
            # 4개 제어점: S1, Waist, Back, Neck
            pts_4 = np.array([pelvis_seq[t], waist_seq[t], back_seq[t], neck_seq[t]], dtype=np.float32)
            
            # X, Y, Z 각각에 대해 3차 스플라인 보간
            fx = PchipInterpolator(z_src, pts_4[:, 0], extrapolate=True)
            fy = PchipInterpolator(z_src, pts_4[:, 1], extrapolate=True)
            fz = PchipInterpolator(z_src, pts_4[:, 2], extrapolate=True)

            spine_9[t, :, 0] = fx(z_dst)
            spine_9[t, :, 1] = fy(z_dst)
            spine_9[t, :, 2] = fz(z_dst)

        return spine_9

    @classmethod
    def compute_spinepose_dense_features(
        cls,
        spine_9_seq: np.ndarray,      # (64, 9, 3)
        lh_seq: np.ndarray,           # (64, 3)
        rh_seq: np.ndarray,           # (64, 3)
        knee_y_mean: np.ndarray       # (64,)
    ) -> np.ndarray:
        """
        SpinePose 9대 척추 분절 시퀀스로부터 6차원 정밀 생체역학 피처 추출
        """
        T = len(spine_9_seq)
        b = int(np.clip(np.argmin(gaussian_filter1d(knee_y_mean, sigma=1.5)), 4, T - 5))
        standing_slice = max(1, T // 8)

        # 1. 인체 좌우 횡축 정규화 벡터 (Lateral Vector)
        L = rh_seq - lh_seq
        L_norm = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-6)

        # 2. [피처 1] L1-L3-L5-S1 요추부 2차 곡률 다항식 계수 a (Quadratic Fit a)
        # 시상면(Sagittal Plane): y(체간 높이) vs z_sagittal(전후 변위)
        poly_a_list = []
        for t in range(T):
            # 요추 4개 점: S1(0), L5(1), L3(2), L1(3)
            lumbar_pts = spine_9_seq[t, :4]
            y_lumbar = lumbar_pts[:, 1]
            # 인체 횡축에 직교하는 시상면 전후 깊이
            z_lumbar = lumbar_pts[:, 2]
            try:
                poly = np.polyfit(y_lumbar, z_lumbar, 2)
                poly_a_list.append(poly[0])
            except Exception:
                poly_a_list.append(0.0)

        poly_a_s = gaussian_filter1d(np.array(poly_a_list), sigma=1.0)
        f1_poly_a_nadir = float(poly_a_s[b] * 1000.0) # 스케일링

        # 3. [피처 2] 최하사점 L1-L5 요추 전만각 (Cobb Angle Proxy, deg)
        # 벡터 1: L5 -> S1
        # 벡터 2: L3 -> L1
        v_l5_s1 = spine_9_seq[:, 0] - spine_9_seq[:, 1]
        v_l3_l1 = spine_9_seq[:, 3] - spine_9_seq[:, 2]
        dot_cobb = np.sum(v_l5_s1 * v_l3_l1, axis=1)
        norm_l5 = np.linalg.norm(v_l5_s1, axis=1) + 1e-6
        norm_l1 = np.linalg.norm(v_l3_l1, axis=1) + 1e-6
        cross_cobb = np.cross(v_l5_s1, v_l3_l1)
        sin_cobb = np.sum(cross_cobb * L_norm, axis=1) / (norm_l5 * norm_l1)
        cos_cobb = np.clip(dot_cobb / (norm_l5 * norm_l1), -1.0, 1.0)
        cobb_angles = np.degrees(np.arctan2(sin_cobb, cos_cobb))
        f2_cobb_nadir = float(gaussian_filter1d(cobb_angles, sigma=1.0)[b])

        # 4. [피처 3] 흉추 후만과 요추 전만의 S자 곡률 밸런스 비율 (S-curve Harmony)
        # 흉추 3개 점: T8(4), T3(5), C7(6)
        # 요추 곡률각 대비 흉추 곡률각의 비율
        v_t8_l1 = spine_9_seq[:, 3] - spine_9_seq[:, 4]
        v_t8_t3 = spine_9_seq[:, 5] - spine_9_seq[:, 4]
        dot_th = np.sum(v_t8_l1 * v_t8_t3, axis=1)
        norm_th1 = np.linalg.norm(v_t8_l1, axis=1) + 1e-6
        norm_th2 = np.linalg.norm(v_t8_t3, axis=1) + 1e-6
        thoracic_angle = np.degrees(np.arccos(np.clip(dot_th / (norm_th1 * norm_th2), -1.0, 1.0)))
        f3_s_harmony = float(np.abs(f2_cobb_nadir) / (np.mean(thoracic_angle[b-2:b+3]) + 1e-4))

        # 5. [피처 4] C1-C7 경추선과 T3-T8 흉추선의 이행부 사잇각 (Cervicothoracic Hinge Pitch)
        v_cervical = spine_9_seq[:, 8] - spine_9_seq[:, 6] # C1 - C7
        v_thoracic = spine_9_seq[:, 5] - spine_9_seq[:, 4] # T3 - T8
        dot_ct = np.sum(v_cervical * v_thoracic, axis=1)
        norm_c = np.linalg.norm(v_cervical, axis=1) + 1e-6
        norm_t = np.linalg.norm(v_thoracic, axis=1) + 1e-6
        cross_ct = np.cross(v_thoracic, v_cervical)
        sin_ct = np.sum(cross_ct * L_norm, axis=1) / (norm_c * norm_t)
        cos_ct = np.clip(dot_ct / (norm_c * norm_t), -1.0, 1.0)
        ct_pitch = np.degrees(np.arctan2(sin_ct, cos_ct))
        f4_ct_pitch_nadir = float(gaussian_filter1d(ct_pitch, sigma=1.0)[b])

        # 6. [피처 5] 최하사점 요추 접선각의 2차 미분 각가속도 급변도 (Lumbar Tangent Jerk)
        d_cobb = np.gradient(gaussian_filter1d(cobb_angles, sigma=1.0))
        d2_cobb = np.gradient(d_cobb)
        b_win = slice(max(0, b - 3), min(T, b + 4))
        f5_tangent_jerk = float(np.max(np.abs(d2_cobb[b_win])))

        # 7. [피처 6] 골반(S1) 대비 L5 요추체의 후방 말림(Retroversion) 변위 (cm)
        # S1 대비 L5의 전후 수평 거리
        retro_dist = np.abs(spine_9_seq[:, 1, 2] - spine_9_seq[:, 0, 2])
        f6_retro_delta = float(retro_dist[b] - np.mean(retro_dist[:standing_slice]))

        dense_feats = np.array([
            f1_poly_a_nadir,
            f2_cobb_nadir,
            f3_s_harmony,
            f4_ct_pitch_nadir,
            f5_tangent_jerk,
            f6_retro_delta
        ], dtype=np.float32)

        return np.nan_to_num(dense_feats, nan=0.0, posinf=180.0, neginf=-180.0)
