# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Test: test_sl2_v2_pipeline.py
Purpose: 최종 확정 모델 [전략 A: Defect F1 극대화] sl2_v2 번들 로드 및 3-State E2E 단일 추론 검증
"""

import os
import sys
import glob
import pickle
import numpy as np
import xgboost as xgb

PROJECT_ROOT = "d:/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from side_lunge_c2_features import extract_side_lunge_c2_features_18d

BUNDLE_PATH = os.path.join(CURRENT_DIR, "sl2_v2_bundle.pkl")


class SideLungeCondition2ChampionV2:
    """사이드 런지 Condition 2 공식 챔피언 모델 (전략 A: Defect F1 극대화)"""
    def __init__(self, bundle_path: str = BUNDLE_PATH):
        assert os.path.exists(bundle_path), f"Bundle not found: {bundle_path}"
        with open(bundle_path, 'rb') as f:
            b = pickle.load(f)

        self.models_lgb = b['models_lgb']
        self.models_cat = b['models_cat']
        self.models_xgb = b['models_xgb']
        self.weights = b['ensemble_weights']
        self.tau_normal = b['tau_normal']
        self.tau_defect = b['tau_defect']
        self.strategy = b.get('strategy', 'Strategy A (Defect F1 Maximized)')

    def predict_single_sequence(self, kps_33_3d: np.ndarray) -> dict:
        """
        단일 64프레임 시퀀스 (64, 33, 3) 3-State 추론
        Returns:
          prob: 결함 확률 (0.0 ~ 1.0)
          state: 0 (CLEAN), 1 (DEFECT), 2 (RETAKE)
          state_label: 한글 레이블
        """
        feat = extract_side_lunge_c2_features_18d(kps_33_3d).reshape(1, -1)

        # 5-Fold 앙상블 평균 예측
        p_lgb = np.mean([m.predict(feat)[0] for m in self.models_lgb])
        p_cat = np.mean([m.predict_proba(feat)[0, 1] for m in self.models_cat])
        dmat = xgb.DMatrix(feat)
        p_xgb = np.mean([m.predict(dmat)[0] for m in self.models_xgb])

        prob = float(self.weights[0] * p_lgb + self.weights[1] * p_cat + self.weights[2] * p_xgb)

        if prob <= self.tau_normal:
            state = 0
            label = "CLEAN (정상)"
        elif prob >= self.tau_defect:
            state = 1
            label = "DEFECT (결함: 몸통-발-무릎 방향 불일치)"
        else:
            state = 2
            label = "RETAKE (판정 보류/재촬영 권고)"

        return {
            "defect_probability": prob,
            "state": state,
            "state_label": label,
            "tau_normal": self.tau_normal,
            "tau_defect": self.tau_defect,
            "strategy": self.strategy
        }


def main():
    model = SideLungeCondition2ChampionV2()
    print("=" * 80)
    print("  [vidio-health-trainer] 사이드 런지 sl2_v2 공식 챔피언 모델 추론 파이프라인 검증")
    print(f"  * 공식 전략: {model.strategy}")
    print(f"  * 임계값: [정상 <= {model.tau_normal:.4f} < 판정보류 < {model.tau_defect:.4f} <= 결함]")
    print("=" * 80)

    data_dir = r"D:\Data\전처리_데이터\MediaPipe_Full33_64F\07_사이드_런지"
    samples = glob.glob(os.path.join(data_dir, "*.npz"))[:6]

    for p in samples:
        fname = os.path.basename(p)
        data = np.load(p)
        kps_3d = data['kps_33_3d_64f'][:, :, :3]
        res = model.predict_single_sequence(kps_3d)

        print(f"파일: {fname}")
        print(f"  결함확률: {res['defect_probability']:.4f} | 상태: State {res['state']} [{res['state_label']}]")
        print("-" * 80)

    print("\n[+] sl2_v2 챔피언 모델 단일 시퀀스 추론 테스트 100% 통과 완료!")


if __name__ == '__main__':
    main()
