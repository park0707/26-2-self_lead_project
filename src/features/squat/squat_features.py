# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Module: src/features/squat/squat_features.py
Purpose: [Tier 3] 바벨 스쿼트 4대 세부 결함(c1~c4) 표준 피처 추출 모듈
         - c1: 척추 중립 위반 (38D, MediaPipe 33 + SpinePose 9)
         - c2: 고개 정면 위반 (19D, MediaPipe 33)
         - c3: 무릎-발 방향 일치 (22D, RTMPose 26 / MediaPipe 33)
         - c4: 발바닥 지면 고정 (25D, RTMPose 26 발 3각 지지면 직결 추출 / MediaPipe 33 안전 폴백)
         - 계약 기반: PreprocessedSkeletons 입력 수신
"""

import numpy as np
from typing import Optional

from src.web.core.contracts import PreprocessedSkeletons
from src.models.second_models.babel_squat.c1_model_v11_a import extract_c1_v11_a_features_from_seq
from src.models.second_models.babel_squat.c2_model_v7 import extract_c2_v7_19d_features_from_seq
from src.models.second_models.babel_squat.c3_model_v6 import extract_c3_v6_features_from_seq
from src.models.second_models.babel_squat.c4_model_v5_1 import extract_c4_v5_features_from_seq


def extract_squat_c1(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """c1: 척추 중립 위반 (38D)"""
    mp = skeletons.mp33_64f
    sp = skeletons.spine9_64f
    return extract_c1_v11_a_features_from_seq(mp, spine_9_64f=sp)


def extract_squat_c2(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """c2: 고개 정면 위반 (19D)"""
    mp = skeletons.mp33_64f
    return extract_c2_v7_19d_features_from_seq(mp)


def extract_squat_c3(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """c3: 무릎-발 방향 일치 (22D)"""
    # RTMPose가 있으면 RTMPose에서 추출, 없으면 MediaPipe에서 직접 추출
    if skeletons.rtm26_64f is not None and skeletons.rtm26_64f.shape[1] == 26:
        # RTMPose 26 -> MediaPipe 33 주요 관절 정합 (무릎, 발목, 고관절 1:1 보존)
        rtm = skeletons.rtm26_64f
        mp_equiv = np.zeros((64, 33, 3), dtype=np.float32)
        # 고관절: 11->23, 12->24
        mp_equiv[:, 23] = rtm[:, 11]
        mp_equiv[:, 24] = rtm[:, 12]
        # 무릎: 13->25, 14->26
        mp_equiv[:, 25] = rtm[:, 13]
        mp_equiv[:, 26] = rtm[:, 14]
        # 발목: 15->27, 16->28
        mp_equiv[:, 27] = rtm[:, 15]
        mp_equiv[:, 28] = rtm[:, 16]
        # 발끝: 20->31, 21->32
        mp_equiv[:, 31] = rtm[:, 20]
        mp_equiv[:, 32] = rtm[:, 21]
        return extract_c3_v6_features_from_seq(mp_equiv)
    else:
        return extract_c3_v6_features_from_seq(skeletons.mp33_64f)


def extract_squat_c4(skeletons: PreprocessedSkeletons) -> np.ndarray:
    """
    c4: 발바닥 지면 고정 (25D)
    [핵심 개선]
    - 어댑터 반토막 버그 영구 제거
    - RTMPose 26관절 발 3각 지지면(엄지 20, 새끼 22, 뒤꿈치 24) 직접 정합
    - 새끼발가락 결측(0.0) 시 안전하게 엄지발가락 좌표 100% 보존 (절대 2로 나누어 반토막 내지 않음)
    - Z 깊이 채널 보존
    """
    if skeletons.rtm26_64f is not None and skeletons.rtm26_64f.shape[1] == 26:
        rtm = skeletons.rtm26_64f
        mp_equiv = np.zeros((64, 33, 3), dtype=np.float32)
        
        # 고관절 및 무릎, 발목, 뒤꿈치 매핑
        mp_equiv[:, 23] = rtm[:, 11] # L Hip
        mp_equiv[:, 24] = rtm[:, 12] # R Hip
        mp_equiv[:, 25] = rtm[:, 13] # L Knee
        mp_equiv[:, 26] = rtm[:, 14] # R Knee
        mp_equiv[:, 27] = rtm[:, 15] # L Ankle
        mp_equiv[:, 28] = rtm[:, 16] # R Ankle
        mp_equiv[:, 29] = rtm[:, 24] # L Heel
        mp_equiv[:, 30] = rtm[:, 25] # R Heel

        # [발끝 정밀 보정 - 안전 가드]
        # Halpe 20: Left Big Toe, Halpe 22: Left Small Toe
        # Halpe 21: Right Big Toe, Halpe 23: Right Small Toe
        small_toe_l = rtm[:, 22]
        small_toe_r = rtm[:, 23]
        big_toe_l = rtm[:, 20]
        big_toe_r = rtm[:, 21]

        valid_small_l = (np.linalg.norm(small_toe_l[:, :2], axis=-1) > 1e-4)
        valid_small_r = (np.linalg.norm(small_toe_r[:, :2], axis=-1) > 1e-4)

        mp_equiv[:, 31] = np.where(valid_small_l[:, None], (big_toe_l + small_toe_l) * 0.5, big_toe_l)
        mp_equiv[:, 32] = np.where(valid_small_r[:, None], (big_toe_r + small_toe_r) * 0.5, big_toe_r)

        # Z 좌표가 누락된 2D인 경우 MediaPipe 3D 깊이 채널로 보강
        if np.std(mp_equiv[:, :, 2]) < 1e-5 and skeletons.mp33_64f is not None:
            mp_equiv[:, :, 2] = skeletons.mp33_64f[:, :, 2]

        return extract_c4_v5_features_from_seq(mp_equiv)
    else:
        # 순수 MediaPipe 33관절 직접 사용 (유턴 없음, 좌표 손실 없음)
        return extract_c4_v5_features_from_seq(skeletons.mp33_64f)
