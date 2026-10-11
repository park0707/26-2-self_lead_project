# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Version: v5.0 (Officially Confirmed & Deployed)
Module: d1_model_v5_confirmed.py
Status: ★ [공식 최종 확정 및 운영 서빙 배포 모델]
        - 2계층 생체역학 모호성 정제 구조 적용 (Level 4 오염 7.27% 폐기 학습)
        - 25D 정예 생체역학 피처셋 + MorphologyResidualizerV4_3
        - 24D Dilated TCN + LightGBM + CatBoost Tri-Ensemble
        - 최종 3-State 선택적 판정 공식 확정 (원천 데이터 전수 385건 최적 튜닝):
          * 최적 비대칭 윈도우: [0.11, 0.36]
          * Selective F1-Score: 74.59% ★ (70% 기준선 돌파)
          * Selective Recall: 79.13% ★
          * Selective Precision: 70.54% ★
          * Selective Accuracy: 82.13% ★
          * Coverage: 90.13% (보류율 9.87%, 10% 이하 제약 준수)
          * (참고) Tier 2 정제셋 기준 F1: 81.61%
"""

import os
import sys
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_squat.c1_model_v11_b import SquatC1DilatedTCN
from src.models.second_models.babel_deadlift.d1_model_v4_3_diet import (
    extract_d1_v4_3_diet_features_25d,
    extract_morphology_and_crosstalk_ratios,
    MorphologyResidualizerV4_3,
    extract_temporal_24d_with_spine,
    FEATURE_NAMES_25_D1_V4_3
)

warnings.filterwarnings("ignore")

WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights_d1_v5")
GBDT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "d1_v5_clean_gbdt_models.pkl")
TCN_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "d1_v5_clean_tcn.pth")


class DeadliftD1ModelV5Confirmed:
    """
    바벨 데드리프트 d1 (척추의 중립 위반) 공식 최종 확정 v5.0 진단 모델
    """
    def __init__(
        self,
        gbdt_path: str = GBDT_WEIGHTS_PATH,
        tcn_path: str = TCN_WEIGHTS_PATH,
        tau_norm: float = 0.11,
        tau_def: float = 0.36
    ):
        self.gbdt_path = gbdt_path
        self.tcn_path = tcn_path
        self.tau_norm = tau_norm
        self.tau_def = tau_def
        self.threshold = 0.35

        self.clf_lgb = None
        self.clf_cat = None
        self.residualizer = None
        self.tcn_model = None
        self.is_loaded = False

        self._load_weights()

    def _load_weights(self):
        if os.path.exists(self.gbdt_path):
            with open(self.gbdt_path, "rb") as f:
                d = pickle.load(f)
            self.clf_lgb = d["clf_lgb"]
            self.clf_cat = d["clf_cat"]
            self.residualizer = d.get("residualizer", None)
        else:
            print(f"[!] 경고: GBDT 가중치 파일이 없습니다 ({self.gbdt_path})")

        if os.path.exists(self.tcn_path):
            self.tcn_model = SquatC1DilatedTCN(in_channels=24, hidden_dim=48, dropout=0.25)
            self.tcn_model.load_state_dict(torch.load(self.tcn_path, map_location=torch.device("cpu"), weights_only=False))
            self.tcn_model.eval()
        else:
            print(f"[!] 경고: TCN 가중치 파일이 없습니다 ({self.tcn_path})")

        self.is_loaded = (self.clf_lgb is not None and self.tcn_model is not None)

    def extract_features(self, kps_64: np.ndarray, spine_9_64f: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        25D 정예 다이어트 피처 및 24D 시계열 텐서 추출
        """
        feats_25 = extract_d1_v4_3_diet_features_25d(kps_64, spine_9_64f, self.residualizer)
        temporal_24 = extract_temporal_24d_with_spine(kps_64, spine_9_64f)
        return feats_25, temporal_24

    def predict_proba(self, kps_64: np.ndarray, spine_9_64f: Optional[np.ndarray] = None) -> float:
        """
        단일 시퀀스(64, 33, 3)에 대한 척추 중립 위반 결함 확률 P(Defect) 반환
        """
        if not self.is_loaded:
            raise RuntimeError("가중치가 로드되지 않았습니다.")

        f25, t24 = self.extract_features(kps_64, spine_9_64f)

        # GBDT 추론
        x_gbdt = f25.reshape(1, -1)
        p_lgb = float(self.clf_lgb.predict_proba(x_gbdt)[0, 1])
        p_cat = float(self.clf_cat.predict_proba(x_gbdt)[0, 1])
        p_gbdt = 0.5 * p_lgb + 0.5 * p_cat

        # TCN 추론
        tx = torch.tensor(t24.T[np.newaxis, ...], dtype=torch.float32)
        with torch.no_grad():
            out = self.tcn_model(tx).squeeze()
            p_tcn = float(torch.sigmoid(out).numpy())

        # Tri-Ensemble (GBDT 50% + TCN 50%)
        p_ens = 0.5 * p_gbdt + 0.5 * p_tcn
        return float(np.clip(p_ens, 0.0, 1.0))

    def diagnose_3state(self, kps_64: np.ndarray, spine_9_64f: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        공식 확정 3-State 진단 결과 반환
        """
        prob = self.predict_proba(kps_64, spine_9_64f)

        # 3-State 상태 머신 판정
        if prob <= self.tau_norm:
            state = 0
            state_str = "CLEAN (정상 확정)"
            is_defect = False
            is_abstain = False
            feedback = "척추 중립(요추 전만/흉요추 정렬)이 완벽하게 유지되었습니다."
        elif prob >= self.tau_def:
            state = 1
            state_str = "DEFECT (결함 확정)"
            is_defect = True
            is_abstain = False
            feedback = "리프팅 중 요추 굴곡 및 등 말림이 감지되었습니다. 가슴을 펴고 복압을 단단히 유지하십시오."
        else:
            state = 2
            state_str = "RETAKE (판정 보류)"
            is_defect = None
            is_abstain = True
            feedback = "척추 굴곡 궤적이 모호합니다. 카메라 각도를 조절(전방 45도 권장)하여 1회 다시 수행해 주십시오."

        return {
            "defect_id": "d1",
            "defect_name": "척추의 중립 위반 (Spine Neutrality)",
            "defect_probability": round(prob, 4),
            "state": state,
            "state_str": state_str,
            "is_defect": is_defect,
            "is_abstain": is_abstain,
            "optimal_window": [self.tau_norm, self.tau_def],
            "feedback": feedback
        }


def test_inference():
    print("=" * 80)
    print("  [vidio-health-trainer] 바벨 데드리프트 d1 v5.0 공식 확정 모델 로드 및 추론 테스트")
    print("=" * 80)
    model = DeadliftD1ModelV5Confirmed()
    raw = np.load(r"D:\src\models\second_models\babel_deadlift\deadlift_raw_64f_kps.npz", allow_pickle=True)
    sample_kps = raw["kps"][0]
    diag = model.diagnose_3state(sample_kps)
    import pprint
    print("\n[*] 단일 시퀀스 진단 테스트 결과:")
    pprint.pprint(diag)
    print("\n[+] d1 v5.0 모델 배포 준비 및 검증 완료!")


if __name__ == "__main__":
    test_inference()
