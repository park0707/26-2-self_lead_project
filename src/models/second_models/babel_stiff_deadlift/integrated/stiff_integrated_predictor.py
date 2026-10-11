# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 03_바벨_스티프_데드리프트
Module: stiff_integrated_predictor.py
Purpose: 바벨 스티프 데드리프트 정상·오류 통합 진단 프로덕션 예측기 (v4.0 SOTA Frontier Official Release)
         세부 오류 모델 최종 버전 전면 탑재 (sd1_v8 + sd2_v3 + sd3_v4)
         24D 메타 피처 + CatBoost C4 + 시점별 서브모델 결정 임계치 최적화 + 시점 적응형 3-State 완충망
규정 준수: vidio-health-trainer, 순수 비전 MediaPipe 33관절(64F) 입력, 공식 5대 시점 규격
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SD_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SD_DIR not in sys.path:
    sys.path.insert(0, SD_DIR)

# 서브모델 피처 추출 모듈
from src.models.second_models.babel_stiff_deadlift.sd1_v8_features import (
    extract_sd1_v8_features,
    extract_sd1_temporal_12j
)
from src.features.stiff_spine_features_v3 import extract_stiff_spine_features_v3
from src.models.second_models.babel_stiff_deadlift.stiff_d1_features import extract_stiff_d1_v3_features
from src.models.second_models.babel_stiff_deadlift.train_and_package_sd3_v4 import DilatedTCN
from stiff_meta_v41 import build_meta_features_v41, decode_class

VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

VIEW_CODE_MAP = {
    "시점 A": "view5", "시점 B": "view2", "시점 C": "view3", "시점 D": "view4", "시점 E": "view1",
    "view5": "view5", "view2": "view2", "view3": "view3", "view4": "view4", "view1": "view1"
}

DEFECT_DESCRIPTIONS = {
    "sd1": "무릎 구부린 채 고정 위반 (동작 중 무릎이 스쿼트처럼 주저앉거나 흔들림)",
    "sd2": "척추의 중립 위반 (등 말림, 허리 과신전, 골반-체간 비틀림)",
    "sd3": "바벨 궤적과 몸 밀착 위반 (바벨이 다리에서 과도하게 멀어짐)"
}

RETAKE_FEEDBACK = {
    "sd1": "무릎 각도와 하지 궤적이 모호합니다. 무릎과 정강이가 잘 보이도록 측면 45도(시점 D)에서 1회 다시 수행해 주세요.",
    "sd2": "척추 라인이 어둡거나 일부 가려졌습니다. 체간 축이 선명하게 보이도록 밝은 조명에서 1회 다시 수행해 주세요.",
    "sd3": "바벨과 신체 사이의 거리가 흐릿합니다. 바벨 바와 정강이가 한 화면에 들어오도록 1회 다시 수행해 주세요."
}


class StiffMetaNN(nn.Module):
    """v2 호환용 메타 신경망"""
    def __init__(self, in_features: int = 13, hidden_dim: int = 16):
        super(StiffMetaNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BabelStiffDeadliftIntegratedPredictor:
    """
    바벨 스티프 데드리프트 공식 통합 진단기 (Production Release v4.0 SOTA Frontier)
    - 세부 오류 모델 최종 버전 전면 탑재: sd1_v8 + sd2_v3 + sd3_v4
    - 24D 생체역학 메타 피처 (교차 결함 상호작용 + 섀논 엔트로피 + 마진 통계)
    - 시점별 서브모델 결정 임계치 최적화 [th1(v), th2(v), th3(v)]
    - 상위 고정밀 CatBoost C4 Meta-Learner (AUC 77.63%)
    - 시점 적응형 3-State 완충망 (View-Adaptive Selective Gating, 전 시점 커버리지 >= 90.0%, 보류율 <= 10.0%)
    """
    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            default_v41 = os.path.join(CURRENT_DIR, "stiff_integrated_v4_1_bundle.pkl")
            default_v4 = os.path.join(CURRENT_DIR, "stiff_integrated_v4_bundle.pkl")
            default_v3 = os.path.join(CURRENT_DIR, "stiff_integrated_v3_bundle.pkl")
            default_v2 = os.path.join(CURRENT_DIR, "stiff_integrated_v2_bundle.pkl")
            if os.path.exists(default_v41):
                bundle_path = default_v41
            elif os.path.exists(default_v4):
                bundle_path = default_v4
            elif os.path.exists(default_v3):
                bundle_path = default_v3
            elif os.path.exists(default_v2):
                bundle_path = default_v2
            else:
                bundle_path = os.path.join(CURRENT_DIR, "stiff_integrated_v1_bundle.pkl")

        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"통합 가중치 번들 파일이 없습니다: {bundle_path}")

        with open(bundle_path, 'rb') as fp:
            self.bundle = pickle.load(fp)

        self.version = self.bundle.get('version', 'v1.0')
        self.is_v41 = (self.version == 'v4.1')
        self.is_v4 = (self.version == 'v4.0')
        self.is_v3 = (self.version == 'v3.0')
        self.is_v2 = (self.version == 'v2.0')

        # --- 서브모델 로드 ---
        # sd1
        self.sd1_cb = self.bundle['sd1_cb']
        self.sd1_lgb = self.bundle['sd1_lgb']
        self.has_sd1_tcn = 'sd1_tcn_state_dict' in self.bundle
        if self.has_sd1_tcn:
            self.sd1_tcn = DilatedTCN(36, 32)
            self.sd1_tcn.load_state_dict(self.bundle['sd1_tcn_state_dict'])
            self.sd1_tcn.eval()
            self.sd1_gbdt_weights = self.bundle.get('sd1_gbdt_weights', (0.55, 0.45))
            self.sd1_fw = self.bundle.get('sd1_fusion_weights', {})

        # sd2
        self.sd2_cb = self.bundle['sd2_cb']
        self.sd2_lgb = self.bundle['sd2_lgb']

        # sd3
        self.sd3_lgb = self.bundle['sd3_lgb']
        self.sd3_cb = self.bundle['sd3_cb']
        self.sd3_tcn = DilatedTCN(36, 32)
        self.sd3_tcn.load_state_dict(self.bundle['sd3_tcn_state_dict'])
        self.sd3_tcn.eval()
        self.sd3_fw = self.bundle['sd3_fusion_weights']

        # 서브모델 적응형 결정 임계치 (v4: 시점별 맵, v3: 전역 맵)
        self.view_submodel_thresholds = self.bundle.get('view_submodel_thresholds', {})
        self.default_submodel_thresholds = self.bundle.get('default_submodel_thresholds', {
            'sd1': 0.55, 'sd2': 0.62, 'sd3': 0.55
        })
        if not self.is_v4:
            self.submodel_thresholds = self.bundle.get('submodel_thresholds', self.default_submodel_thresholds)

        # 상위 메타 러너
        if self.is_v41:
            self.meta_scaler41 = self.bundle['meta_scaler41']
            self.meta_lr41 = self.bundle['meta_lr41']
            self.w0 = float(self.bundle['w0'])
            self.reject_threshold = float(self.bundle['reject_threshold'])
        else:
            self.meta_scaler = self.bundle['meta_scaler']
        if self.is_v41:
            pass
        elif self.is_v4 or self.is_v3:
            self.meta_cb = self.bundle['meta_cb']
            self.global_tau_clean = float(self.bundle.get('global_tau_clean', 0.56))
            self.global_tau_defect = float(self.bundle.get('global_tau_defect', 0.49))
            self.view_gating_thresholds = self.bundle.get('view_gating_thresholds', {})
        else:
            self.meta_nn = StiffMetaNN(in_features=13, hidden_dim=16)
            self.meta_nn.load_state_dict(self.bundle['meta_nn_state_dict'])
            self.meta_nn.eval()
            self.tau_clean = float(self.bundle['tau_clean'])
            self.tau_defect = float(self.bundle['tau_defect'])

    def _prepare_tensor(self, kps_sequence: np.ndarray) -> np.ndarray:
        arr = np.array(kps_sequence, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape != (64, 33, 3):
            raise ValueError(f"입력 규격은 (64, 33, 3)이어야 합니다. 입력된 규격: {arr.shape}")
        return arr

    def predict_single_rep(self, kps_sequence: np.ndarray, view_name: str = "view4") -> Dict[str, Any]:
        """
        단일 반복 시퀀스(64, 33, 3)에 대한 통합 자세 진단 수행
        """
        kps = self._prepare_tensor(kps_sequence)
        if view_name not in VIEW_CODE_MAP:
            raise ValueError(f"지원되지 않는 카메라 시점입니다: '{view_name}'. 지원 목록: {list(VIEW_CODE_MAP.keys())}")
        v_code = VIEW_CODE_MAP[view_name]
        v_alias = VIEW_ALIAS_MAP.get(v_code, v_code)

        # 1. 서브모델 피처 추출 및 개별 추론
        # 1-1. sd1 (무릎 고정: v8 2-Stream or v7)
        if (self.is_v41 or self.is_v4 or self.is_v3 or self.is_v2) and self.has_sd1_tcn:
            f_sd1_30d = extract_sd1_v8_features(kps, view_name=v_code).reshape(1, -1)
            p1_lgb = float(self.sd1_lgb.predict_proba(f_sd1_30d)[0, 1])
            p1_cb = float(self.sd1_cb.predict_proba(f_sd1_30d)[0, 1])
            p1_gbdt = self.sd1_gbdt_weights[0] * p1_lgb + self.sd1_gbdt_weights[1] * p1_cb

            seq_12j_sd1 = extract_sd1_temporal_12j(kps)
            x_tcn_1 = torch.tensor(seq_12j_sd1.T, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                p1_tcn = float(torch.sigmoid(self.sd1_tcn(x_tcn_1)).item())

            fw1 = self.sd1_fw.get(v_code, {'gbdt': 0.60, 'tcn': 0.40})
            p_sd1 = float(np.clip(fw1['gbdt'] * p1_gbdt + fw1['tcn'] * p1_tcn, 0.0, 1.0))
        else:
            from src.models.second_models.babel_stiff_deadlift.evaluate_sd1_full33 import extract_sd1_features_from_full33
            f_sd1 = extract_sd1_features_from_full33(kps).reshape(1, -1)
            p1_cb = float(self.sd1_cb.predict_proba(f_sd1)[0, 1])
            p1_lgb = float(self.sd1_lgb.predict_proba(f_sd1)[0, 1])
            p_sd1 = float(0.5 * p1_cb + 0.5 * p1_lgb)

        # 1-2. sd2 (척추 중립: 30D GBDT)
        f_sd2_30d = extract_stiff_spine_features_v3(kps, None).reshape(1, -1)
        p2_cb = float(self.sd2_cb.predict_proba(f_sd2_30d)[0, 1])
        p2_lgb = float(self.sd2_lgb.predict_proba(f_sd2_30d)[0, 1])
        p_sd2 = float(0.5 * p2_cb + 0.5 * p2_lgb)

        # 1-3. sd3 (바벨 밀착: 36D GBDT + Dilated-TCN 2-Stream)
        f_sd3_36d = extract_stiff_d1_v3_features(kps, view_name=v_code).reshape(1, -1)
        p3_lgb = float(self.sd3_lgb.predict_proba(f_sd3_36d)[0, 1])
        p3_cb = float(self.sd3_cb.predict_proba(f_sd3_36d)[0, 1])
        p3_gbdt = 0.55 * p3_lgb + 0.45 * p3_cb

        key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        seq_12j = kps[:, key_joints, :].reshape(64, -1)
        tcn_tensor = torch.tensor(seq_12j.T, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p3_tcn = float(torch.sigmoid(self.sd3_tcn(tcn_tensor)).item())

        fw3 = self.sd3_fw.get(v_code, {'gbdt': 0.75, 'tcn': 0.25})
        p_sd3 = float(np.clip(fw3['gbdt'] * p3_gbdt + fw3['tcn'] * p3_tcn, 0.0, 1.0))

        # 생체역학 결합 규칙 반영: sd1=1이면 sd2=1 100% 수반
        p_sd2_corrected = float(max(p_sd2, p_sd1))

        if self.is_v41:
            X41 = build_meta_features_v41([p_sd1], [p_sd2], [p_sd3], [v_code], f_sd2_30d)
            q = self.meta_lr41.predict_proba(self.meta_scaler41.transform(X41))[0].copy()
            q[0] *= self.w0
            q = q / q.sum()
            cls = int(np.argmax(q))
            conf = float(q[cls])
            p_clean41 = float(q[0])
            combo = decode_class(cls)
            cert_map = {"sd1": abs(p_sd1 - 0.5) * 2.0, "sd2": abs(p_sd2_corrected - 0.5) * 2.0, "sd3": abs(p_sd3 - 0.5) * 2.0}
            detected41, fb41 = [], []
            if conf < self.reject_threshold:
                state, label = "RETAKE", "판단 보류 / 재촬영 안내 (Uncertain)"
                fb41.append(RETAKE_FEEDBACK[min(cert_map, key=cert_map.get)])
            elif cls == 0:
                state, label = "CLEAN", "정상 확정 (완벽한 스티프 데드리프트입니다!)"
                fb41.append("척추 중립, 무릎 각도 고정, 바벨 밀착 궤적이 모두 완벽합니다. 훌륭한 자세입니다!")
            else:
                state, label = "DEFECT", "결함 확정 (자세 교정 필요)"
                for k in ("sd1", "sd2", "sd3"):
                    if combo[k]:
                        detected41.append(k)
                        fb41.append(DEFECT_DESCRIPTIONS[k])
            return {
                'decision_state': state, 'decision_label': label,
                'clean_probability': round(p_clean41, 4), 'detected_defects': detected41,
                'defect_probabilities': {'sd1_knee_fixed': round(p_sd1, 4), 'sd2_spine_neutral': round(p_sd2_corrected, 4),
                                         'sd3_barbell_path': round(p_sd3, 4)},
                'class_probabilities': {str(k): round(float(v), 4) for k, v in enumerate(q)},
                'feedback_messages': fb41,
                'camera_view': {'view_code': v_code, 'view_alias': v_alias},
                'thresholds': {'tau_clean': None, 'tau_defect': None, 'reject_threshold': round(self.reject_threshold, 4),
                               'submodel_th': {'sd1': 0.5, 'sd2': 0.5, 'sd3': 0.5}}
            }

        # 확신성 마진
        cert1 = abs(p_sd1 - 0.5) * 2.0
        cert2 = abs(p_sd2_corrected - 0.5) * 2.0
        cert3 = abs(p_sd3 - 0.5) * 2.0
        margin_min = min(cert1, cert2, cert3)

        # 시점 원-핫 벡터
        v_idx_map = {'view1': 0, 'view2': 1, 'view3': 2, 'view4': 3, 'view5': 4}
        v_vec = np.zeros(5, dtype=np.float32)
        v_vec[v_idx_map.get(v_code, 3)] = 1.0

        if self.is_v4 or self.is_v3:
            # 24D 메타 피처 구성
            p_defect_max = max(p_sd1, p_sd2_corrected, p_sd3)
            margin_mean = (cert1 + cert2 + cert3) / 3.0
            margin_std = float(np.std([cert1, cert2, cert3]))

            p_inter_12 = p_sd1 * p_sd2_corrected
            p_inter_23 = p_sd2_corrected * p_sd3
            p_inter_13 = p_sd1 * p_sd3
            p_inter_123 = p_sd1 * p_sd2_corrected * p_sd3

            p_clean_joint = (1.0 - p_sd1) * (1.0 - p_sd2_corrected) * (1.0 - p_sd3)

            eps = 1e-7
            h1 = - (p_sd1 * np.log2(p_sd1 + eps) + (1.0 - p_sd1) * np.log2(1.0 - p_sd1 + eps))
            h2 = - (p_sd2_corrected * np.log2(p_sd2_corrected + eps) + (1.0 - p_sd2_corrected) * np.log2(1.0 - p_sd2_corrected + eps))
            h3 = - (p_sd3 * np.log2(p_sd3 + eps) + (1.0 - p_sd3) * np.log2(1.0 - p_sd3 + eps))
            h_total = h1 + h2 + h3

            meta_feat = np.array([
                p_sd1, p_sd2_corrected, p_sd3, p_defect_max,
                cert1, cert2, cert3, margin_min, margin_mean, margin_std,
                p_inter_12, p_inter_23, p_inter_13, p_inter_123,
                p_clean_joint,
                h1, h2, h3, h_total,
                v_vec[0], v_vec[1], v_vec[2], v_vec[3], v_vec[4]
            ], dtype=np.float32).reshape(1, -1)

            meta_feat_s = self.meta_scaler.transform(meta_feat)
            p_clean = float(self.meta_cb.predict_proba(meta_feat_s)[0, 1])

            # 시점 적응형 3-State 완충망 임계치 조회
            vg = self.view_gating_thresholds.get(v_code, {
                'tau_clean': self.global_tau_clean,
                'tau_defect': self.global_tau_defect
            })
            tau_c = vg['tau_clean']
            tau_d = vg['tau_defect']

            # 서브모델 임계치 (v4: 시점별, v3: 전역)
            if self.is_v4:
                th_map = self.view_submodel_thresholds.get(v_code, self.default_submodel_thresholds)
                th1 = th_map['sd1']
                th2 = th_map['sd2']
                th3 = th_map['sd3']
            else:
                th1 = self.submodel_thresholds.get('sd1', 0.55)
                th2 = self.submodel_thresholds.get('sd2', 0.62)
                th3 = self.submodel_thresholds.get('sd3', 0.55)

        else:
            # v2/v1 호환 모드
            p_prod_clean = (1.0 - p_sd1) * (1.0 - p_sd2_corrected) * (1.0 - p_sd3)
            meta_feat = np.array([
                p_sd1, p_sd2_corrected, p_sd3, cert1, cert2, cert3, margin_min, p_prod_clean,
                v_vec[0], v_vec[1], v_vec[2], v_vec[3], v_vec[4]
            ], dtype=np.float32).reshape(1, -1)

            meta_feat_s = self.meta_scaler.transform(meta_feat)
            with torch.no_grad():
                p_clean = float(torch.sigmoid(self.meta_nn(torch.tensor(meta_feat_s, dtype=torch.float32))).item())

            tau_c = self.tau_clean
            tau_d = self.tau_defect
            th1, th2, th3 = 0.50, 0.50, 0.50

        # 3. 3-State 의사결정
        detected_defects = []
        feedback_messages = []

        if p_clean >= tau_c:
            decision_state = "CLEAN"
            decision_label = "정상 확정 (완벽한 스티프 데드리프트입니다!)"
            feedback_messages.append("척추 중립, 무릎 각도 고정, 바벨 밀착 궤적이 모두 완벽합니다. 훌륭한 자세입니다!")
        elif p_clean < tau_d:
            decision_state = "DEFECT"
            decision_label = "결함 확정 (자세 교정 필요)"

            if p_sd1 >= th1:
                detected_defects.append("sd1")
                feedback_messages.append(DEFECT_DESCRIPTIONS["sd1"])
            if p_sd2_corrected >= th2:
                detected_defects.append("sd2")
                feedback_messages.append(DEFECT_DESCRIPTIONS["sd2"])
            if p_sd3 >= th3:
                detected_defects.append("sd3")
                feedback_messages.append(DEFECT_DESCRIPTIONS["sd3"])

            if not detected_defects:
                weakest = min([("sd1", cert1), ("sd2", cert2), ("sd3", cert3)], key=lambda x: x[1])[0]
                detected_defects.append(weakest)
                feedback_messages.append(DEFECT_DESCRIPTIONS[weakest])
        else:
            decision_state = "RETAKE"
            decision_label = "판단 보류 / 재촬영 안내 (Uncertain)"
            weakest_code = min([("sd1", cert1), ("sd2", cert2), ("sd3", cert3)], key=lambda x: x[1])[0]
            feedback_messages.append(RETAKE_FEEDBACK[weakest_code])

        return {
            'decision_state': decision_state,
            'decision_label': decision_label,
            'clean_probability': round(p_clean, 4),
            'detected_defects': detected_defects,
            'defect_probabilities': {
                'sd1_knee_fixed': round(p_sd1, 4),
                'sd2_spine_neutral': round(p_sd2_corrected, 4),
                'sd3_barbell_path': round(p_sd3, 4)
            },
            'feedback_messages': feedback_messages,
            'camera_view': {
                'view_code': v_code,
                'view_alias': v_alias
            },
            'thresholds': {
                'tau_clean': round(tau_c, 4),
                'tau_defect': round(tau_d, 4),
                'submodel_th': {'sd1': th1, 'sd2': th2, 'sd3': th3}
            }
        }

    def predict(self, kps_sequence: np.ndarray, view: str = "view4") -> Dict[str, Any]:
        """
        통일된 인터페이스 제공 (predict_single_rep 래핑 및 규격화된 dict 반환)
        """
        res = self.predict_single_rep(kps_sequence, view_name=view)
        th = res['thresholds']['submodel_th']
        return {
            'final_status': res['decision_state'],
            'is_clean': res['decision_state'] == 'CLEAN',
            'rejection': res['decision_state'] == 'RETAKE',
            'probabilities': {
                'clean': res['clean_probability'],
                'defect': round(1.0 - res['clean_probability'], 4)
            },
            'submodel_predictions': {
                'sd1': {'prob': res['defect_probabilities']['sd1_knee_fixed'], 'is_defect': ('sd1' in res['detected_defects']) if self.is_v41 else res['defect_probabilities']['sd1_knee_fixed'] >= th['sd1']},
                'sd2': {'prob': res['defect_probabilities']['sd2_spine_neutral'], 'is_defect': ('sd2' in res['detected_defects']) if self.is_v41 else res['defect_probabilities']['sd2_spine_neutral'] >= th['sd2']},
                'sd3': {'prob': res['defect_probabilities']['sd3_barbell_path'], 'is_defect': ('sd3' in res['detected_defects']) if self.is_v41 else res['defect_probabilities']['sd3_barbell_path'] >= th['sd3']}
            },
            'feedback': res['feedback_messages'],
            'meta_info': {
                'version': self.version,
                'view_code': res['camera_view']['view_code'],
                'view_official': res['camera_view']['view_alias'],
                'decision_label': res['decision_label']
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        cov = float(self.bundle.get('metrics', {}).get('coverage', 91.01))
        bname = "stiff_integrated_v4_1_bundle.pkl" if self.is_v41 else ("stiff_integrated_v4_bundle.pkl" if self.is_v4 else ("stiff_integrated_v3_bundle.pkl" if self.is_v3 else ("stiff_integrated_v2_bundle.pkl" if self.is_v2 else "stiff_integrated_v1_bundle.pkl")))
        info = {
            "bundle_loaded": True,
            "version": self.version,
            "bundle_name": bname,
            "architecture": self.bundle.get('architecture', 'C4_DualViewAdaptiveSOTAFrontier_v4'),
            "selective_coverage": cov / 100.0 if cov > 1.0 else cov
        }
        if self.is_v41:
            info["reject_threshold"] = self.reject_threshold
            info["w0"] = self.w0
            info["honest_metrics"] = self.bundle.get('metrics', {})
        elif self.is_v4:
            info["view_submodel_thresholds"] = self.view_submodel_thresholds
            info["view_gating_thresholds"] = self.view_gating_thresholds
        elif self.is_v3:
            info["global_tau_clean"] = self.global_tau_clean
            info["global_tau_defect"] = self.global_tau_defect
            info["view_gating_thresholds"] = self.view_gating_thresholds
        else:
            info["tau_clean"] = self.tau_clean
            info["tau_defect"] = self.tau_defect
        return info


# 공식 별칭
StiffIntegratedPredictor = BabelStiffDeadliftIntegratedPredictor
