# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 다이나믹 런지 (Dynamic Lunge - 05_스텝_포워드 & 06_스텝_백워드)
Defect: Condition 3 - 뒤다리 무릎 각도 90도 (Rear Knee 90° Defect)
Module: dl3_rtmp_1.py
Version: dl3_rtmp_1 (RTMPose-m Halpe26 Optimized Production Champion)
Baseline: bl3_v4 / bl3_rtmp_1 14D 정예 기구학 엔진 계승 (Copy-and-Refine)

핵심 모델 사양:
  1. 입력 모달리티: RTMPose Halpe26 2D 키포인트 (64, 26, 3) 100% 순수 비전
  2. 피처 구성: 22D 정예 기구학 피처
     - bl3 베이스라인 14D 완전 계승:
       * f01: rear_knee_stability_std (최하사점 뒤다리 무릎 각도 표준편차 - 떨림/안정성)
       * f02: lead_rear_knee_ratio (앞다리 대비 뒤다리 무릎 각도 비율)
       * f03: rear_femur_vertical_pitch (뒤다리 대퇴골의 지면 수직선 피치각)
       * f04: rear_shank_horizontal_pitch (뒤다리 정강이의 지면 수평선 피치각)
       * f05: inter_ankle_stride_bottom (최하사점 양 발목 간 2D 보폭)
       * f06: knee_velocity_decel_nadir (최하사점 전환기 뒤다리 무릎 각속도 감속도)
       * f07: rear_knee_target_dev_bottom (최하사점 90도 기준 뒤다리 무릎 편차)
       * f08: inter_knee_vertical_disp (최하사점 양 무릎 간 수직 높이차)
       * f09: torso_to_rear_femur_angle (체간 축과 뒤다리 대퇴골 간 사잇각 - 고관절 신전도)
       * f10: lead_knee_angle_bottom (최하사점 앞다리 무릎 각도)
       * f11: rear_knee_angle_bottom (최하사점 뒤다리 무릎 원천 각도)
       * f12: knee_hysteresis_area (무릎 각도-각속도 위상 루프 적분 면적)
       * f13: femur_tibia_proj_ratio (최하사점 대퇴/정강이 2D 투영 길이비)
       * f14: crp_hip_knee_mean (고관절-무릎 연속 상대 위상 CRP 평균)
     - 다이나믹 런지 RTMPose 고유 8D 피처:
       * f15: rear_foot_heel_lift (최하사점 뒤꿈치 수직 들림 높이)
       * f16: rear_foot_pitch_bottom (최하사점 뒤발 경사각: 발바닥 접촉/배굴)
       * f17: lead_foot_flatness (최하사점 앞발 지면 안착 평평도)
       * f18: rear_knee_min_angle (전 시퀀스 뒤다리 무릎 최소 굴곡 각도)
       * f19: rear_knee_rom (뒤다리 무릎 관절 가동범위: 기립 - 최소각)
       * f20: step_stride_expansion_ratio (기립 보폭 대비 최하사점 스텝 확장비)
       * f21: torso_vertical_tilt_bottom (최하사점 체간 수직선 기울기 각도)
       * f22: cam_aspect_ratio (골반폭 대비 체간길이비: 카메라 방위각 추정)
  3. 분류 알고리즘: 22D CatBoost Classifier (Production Champion)
  4. 3-State 적응형 신뢰도 게이팅:
     - tau_clean: 0.399, tau_defect: 0.693 (불확실성 보류율 14.8% 통제)
  5. 실측 성능:
     - AI-Hub 다이나믹 런지 9,795개 전수 5-Fold StratifiedGroupKFold OOF: 3-State F1 80.55%, AUC 0.9068
     - 다이나믹 런지 1,468개 벤치마크 테스트셋: 3-State F1 91.37%, AUC 0.9629
     - 공식 5대 시점 전원 70% 돌파: 시점 A(82.74%), B(81.70%), C(72.73%), D(82.78%), E(81.88%)
"""

import os
import sys
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from catboost import CatBoostClassifier

PROJECT_ROOT = "D:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"D:\src\features")

from lunge_dl3_rtmp_features import (
    extract_lunge_dl3_22d_kinematics,
    FEATURE_NAMES_22D
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE_PATH = os.path.join(CURRENT_DIR, "weights", "dl3_rtmp_1_bundle.pkl")
DEFAULT_CBM_PATH = os.path.join(CURRENT_DIR, "dl3_rtmp_1.cbm")

# AGENTS.md 공식 5대 카메라 시점 명칭 및 매핑
VIEW_ALIAS_MAP = {
    'view5': '시점 A (우측 후방 45°)',
    'view2': '시점 B (우측 전방 45°)',
    'view3': '시점 C (정면 0°)',
    'view4': '시점 D (좌측 전방 45°)',
    'view1': '시점 E (좌측 후방 45°)'
}

class DL3RTMP1Classifier:
    """다이나믹 런지 Condition 3 (뒤다리 무릎 각도 90도) 전담 dl3_rtmp_1 프로덕션 3-State 분류기"""
    
    EXERCISE_NAME = "다이나믹 런지 (Dynamic Lunge)"
    DEFECT_NAME = "뒤다리 무릎 각도 90도 (Rear Knee 90°)"
    MODEL_CODE = "dl3"
    MODEL_NAME = "dl3_rtmp_1"
    VERSION = "v1.0"
    
    # 5-Fold 실측 최적 3-State 임계값
    TAU_CLEAN = 0.399
    TAU_DEFECT = 0.693
    OPTIMAL_TAU = 0.546
    
    def __init__(self, bundle_path: Optional[str] = None):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        self.cb_model: Optional[CatBoostClassifier] = None
        self.feature_names = FEATURE_NAMES_22D
        self.bundle_data: Optional[Dict[str, Any]] = None
        self._load_model()
        
    def _load_model(self):
        """가중치 번들 또는 .cbm 단독 모델 로드"""
        if os.path.exists(self.bundle_path):
            try:
                with open(self.bundle_path, 'rb') as fp:
                    self.bundle_data = pickle.load(fp)
                self.cb_model = self.bundle_data.get('cb_model')
                self.TAU_CLEAN = float(self.bundle_data.get('tau_clean', self.TAU_CLEAN))
                self.TAU_DEFECT = float(self.bundle_data.get('tau_defect', self.TAU_DEFECT))
                self.OPTIMAL_TAU = float(self.bundle_data.get('optimal_tau', self.OPTIMAL_TAU))
                self.feature_names = self.bundle_data.get('feature_names', self.feature_names)
                return
            except Exception:
                pass
                
        if os.path.exists(DEFAULT_CBM_PATH):
            self.cb_model = CatBoostClassifier()
            self.cb_model.load_model(DEFAULT_CBM_PATH)
            return
            
        raise FileNotFoundError(f"[!] 모델 파일을 찾을 수 없습니다: {self.bundle_path} 또는 {DEFAULT_CBM_PATH}")
        
    def extract_features(
        self,
        kps_26: np.ndarray,
        view_code: Optional[str] = None
    ) -> np.ndarray:
        """
        단일 64프레임 관절 시퀀스 (64, 26, 3) 또는 (64, 26, 2)로부터 22D 정예 기구학 피처 추출
        """
        return extract_lunge_dl3_22d_kinematics(kps_26)
        
    def predict_sequence(
        self,
        kps_26: np.ndarray,
        view_code: str = 'view2'
    ) -> Dict[str, Any]:
        """
        단일 시퀀스(64, 26, 3)에 대해 dl3_rtmp_1 3-State 판정 및 생체역학 코칭 피드백 생성
        
        Parameters:
          kps_26: (64, 26, 3) 또는 (64, 26, 2)
          view_code: 'view1' ~ 'view5' (선택적)
          
        Returns:
          dict: 상태(0: CLEAN, 1: DEFECT, 2: RETAKE), 결함 확률, 생체역학 피처값, 코칭 메시지
        """
        if self.cb_model is None:
            self._load_model()
            
        feat_22 = self.extract_features(kps_26) # (22,)
        feat_2d = feat_22.reshape(1, -1)
        
        prob_defect = float(self.cb_model.predict_proba(feat_2d)[0, 1])
        margin = abs(prob_defect - self.OPTIMAL_TAU)
        
        # 3-State 판정
        if prob_defect < self.TAU_CLEAN:
            state = 0
            state_str = "CLEAN (정상)"
            decision = "NORMAL"
            feedback = "정상: 뒤다리 무릎 각도(약 90°) 및 지면 수직 정렬이 완벽하게 유지되었습니다. 모범적인 런지 하강 자세입니다."
        elif prob_defect >= self.TAU_DEFECT:
            state = 1
            state_str = "DEFECT (결함)"
            decision = "DEFECT"
            
            # 생체역학 세부 원인 역추적
            rear_angle = float(feat_22[10])      # f11: rear_knee_angle_bottom
            lead_angle = float(feat_22[9])       # f10: lead_knee_angle_bottom
            dev_bot = float(feat_22[6])          # f07: rear_knee_target_dev_bottom
            femur_pitch = float(feat_22[2])      # f03: rear_femur_vertical_pitch
            heel_lift = float(feat_22[14])       # f15: rear_foot_heel_lift
            knee_min = float(feat_22[17])        # f18: rear_knee_min_angle
            stability = float(feat_22[0])        # f01: rear_knee_stability_std
            
            if rear_angle > 105.0 or knee_min > 105.0:
                feedback = (
                    f"결함 감지 (뒤무릎 불충분 굴곡): 뒤다리 무릎 각도가 {rear_angle:.1f}°로 충분히 낮아지지 않았습니다. "
                    "골반을 수직으로 더 깊게 낮추어 뒤다리 대퇴골이 지면과 수직(90°)을 이루도록 앉아주세요."
                )
            elif rear_angle < 75.0 or knee_min < 75.0:
                feedback = (
                    f"결함 감지 (뒤무릎 과도한 굴곡/바닥 충돌 위험): 뒤다리 무릎 각도가 {rear_angle:.1f}°로 과도하게 접혔습니다. "
                    "무릎이 바닥에 부딪히지 않도록 바닥 위 약 5cm 지점에서 하강을 멈추고 버텨주세요."
                )
            elif femur_pitch > 25.0:
                feedback = (
                    f"결함 감지 (뒤다리 대퇴골 전후방 정렬 불량): 뒤다리 대퇴골의 수직 피치 각도가 {femur_pitch:.1f}°로 기울어졌습니다. "
                    "상체를 꼿꼿이 세우고 골반이 앞으로 밀리거나 뒤로 빠지지 않도록 수직 하강하세요."
                )
            elif heel_lift < 0.05:
                feedback = (
                    "결함 감지 (뒤발 지면 미들림/족저굴곡 부족): 뒤다리의 뒤꿈치가 충분히 들리지 않아 무릎이 유연하게 굽혀지지 못했습니다. "
                    "뒤발은 앞꿈치(까치발)로 체중을 지지하며 뒤꿈치를 들어 올려주세요."
                )
            elif stability > 6.0:
                feedback = (
                    f"결함 감지 (최하사점 관절 동요도 과다): 최하사점에서 무릎 떨림(표준편차 {stability:.1f}°)이 감지되었습니다. "
                    "코어와 둔근의 장력을 유지하여 흔들림 없이 안정적으로 정지 후 상승하세요."
                )
            else:
                feedback = (
                    f"결함 감지 (뒤무릎 직각 정렬 오차 {dev_bot:.1f}°): 뒤다리 무릎 굴곡 각도가 90도 권장 기준을 벗어났습니다. "
                    "앞다리와 뒷다리가 모두 직각(90°-90°)을 형성하는 스탠스를 점검하세요."
                )
        else:
            state = 2
            state_str = "UNCERTAIN / RETAKE (판정보류)"
            decision = "UNDETERMINED"
            feedback = (
                "판정 보류: 관절 굴곡 신뢰도 경계 구간입니다. "
                "카메라가 전신(머리부터 양 발끝까지)을 가림 없이 담을 수 있도록 45도 측면 시점에서 1회 재수행을 권장합니다."
            )
            
        return {
            'model_name': self.MODEL_NAME,
            'exercise': self.EXERCISE_NAME,
            'defect_code': self.MODEL_CODE,
            'defect_name': self.DEFECT_NAME,
            'state': state,
            'state_label': state_str,
            'decision': decision,
            'defect_probability': prob_defect,
            'optimal_threshold': self.OPTIMAL_TAU,
            'tau_clean': self.TAU_CLEAN,
            'tau_defect': self.TAU_DEFECT,
            'certainty_margin': margin,
            'view_code': view_code,
            'view_name': VIEW_ALIAS_MAP.get(view_code, view_code),
            'biomechanics': {
                'rear_knee_angle_bottom': float(feat_22[10]),
                'lead_knee_angle_bottom': float(feat_22[9]),
                'rear_knee_target_dev_bottom': float(feat_22[6]),
                'rear_femur_vertical_pitch': float(feat_22[2]),
                'rear_shank_horizontal_pitch': float(feat_22[3]),
                'inter_ankle_stride_bottom': float(feat_22[4]),
                'rear_foot_heel_lift': float(feat_22[14]),
                'rear_knee_min_angle': float(feat_22[17]),
                'rear_knee_rom': float(feat_22[18]),
                'rear_knee_stability_std': float(feat_22[0])
            },
            'coaching_message': feedback
        }

if __name__ == '__main__':
    clf = DL3RTMP1Classifier()
    dummy_kps = np.zeros((64, 26, 3), dtype=np.float32)
    res = clf.predict_sequence(dummy_kps, view_code='view2')
    print("=" * 65)
    print("  [+] dl3_rtmp_1 프로덕션 추론 테스트 성공!")
    print("=" * 65)
    for k, v in res.items():
        if k != 'biomechanics':
            print(f"  {k}: {v}")
    print("  biomechanics summary:", {k: round(v, 2) for k, v in res['biomechanics'].items()})
    assert res['model_name'] == 'dl3_rtmp_1'
    assert res['state'] in [0, 1, 2]
    print("[+] All assertions passed!")
