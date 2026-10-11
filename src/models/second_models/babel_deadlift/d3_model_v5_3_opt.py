# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d3. 발바닥 지면 고정 (Foot Ground Fixation)
Model Version: v5.3-Opt (d3_model_v5_3_opt.py)
Status: ★ 피처 전수 감사(Audit) 및 다이어트(Diet/Pruning) 최적화 프로덕션 모델
        - 기존 32D 피처셋 중 음수(-) PFI 노이즈 및 |r| >= 0.85 고상관 중복 15종 소거
        - 17D 정예 생체역학 피처셋 확정 (소표본 차원의 저주 및 과적합 해소)
        - 57D Spatio-Temporal SequenceNet (Clean KAPW 동기화) + 17D Tri-Ensemble (CatBoost/LGBM/RF)
        - 5대 공식 시점별 3-State 안전 게이트 최적화 (보류율 <= 10.0% 엄격 충족)
        - 최종 5-Fold 피험자 완전 격리 F1-Score: 71.01% (기존 68.32% 대비 +2.69%p 상승)
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

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.ensemble import RandomForestClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.second_models.babel_deadlift.d3_model_v5_3 import (
    KEY_JOINT_INDICES,
    NUM_KEY_JOINTS,
    FEATURE_NAMES_32_D3,
    DeadliftSTSequenceNetV5_3,
    extract_d3_v5_3_features_from_seq,
    apply_kapw_phase_warping,
    inpaint_lower_body_joints,
    clean_and_smooth_keypoints
)

# 17D 정예 피처 인덱스 (32D 중 엄선)
OPTIMAL_17_INDICES: List[int] = [1, 4, 5, 6, 7, 8, 9, 11, 13, 19, 20, 21, 23, 24, 27, 29, 30]

# 17D 정예 피처 명칭 목록
FEATURE_NAMES_17_D3: List[str] = [FEATURE_NAMES_32_D3[i] for i in OPTIMAL_17_INDICES]

# 소거된 15개 피처 인덱스 및 명칭 (참조용)
PRUNED_15_INDICES: List[int] = [i for i in range(32) if i not in OPTIMAL_17_INDICES]
PRUNED_15_NAMES: List[str] = [FEATURE_NAMES_32_D3[i] for i in PRUNED_15_INDICES]

# 5대 공식 시점 코드 목록
VIEW_CODE_LIST = ["view5", "view2", "view3", "view4", "view1"]


def extract_d3_v5_3_opt_features_from_seq(kps_33_64f: np.ndarray, view_code: str = "view3") -> np.ndarray:
    """
    (64, 33, 3) 시퀀스로부터 v5.3-Opt 17D 정예 피처를 직접 추출하는 캡슐화 인터페이스 함수
    - 노이즈 및 중복이 제거된 17D 정예 피처 반환
    """
    f32 = extract_d3_v5_3_features_from_seq(kps_33_64f, view_code)
    return f32[OPTIMAL_17_INDICES].astype(np.float32)


class BabelDeadliftD3ModelV5_3Opt:
    """
    바벨 데드리프트 d3 공식 프로덕션 모델 v5.3-Opt (피처 다이어트 최적화 하이브리드)
    - 57D Spatio-Temporal SequenceNet (Clean KAPW 관절 위상 동기화)
    - 17D 정예 피처 Tri-Ensemble (CatBoost + LightGBM + Random Forest)
    - 시점 적응형 앙상블 블렌드 (후방 뷰 55% DL, 전방/정면 뷰 40% DL)
    - 5대 공식 시점별 3-State 안전 게이트 (보류율 <= 10.0% 엄격 준수)
    """
    DEFECT_CODE = "d3"
    DEFECT_NAME = "발바닥 지면 고정"
    EXERCISE_NAME = "바벨 데드리프트"
    VERSION = "v5.3-Opt"
    FEATURE_DIM = 17
    FEATURE_NAMES = FEATURE_NAMES_17_D3
    SEQUENCE_FEAT_DIM = 57

    def __init__(self, view_thresholds: Optional[Dict[str, Tuple[float, float]]] = None, auto_load: bool = False):
        """
        view_thresholds: 시점별 (t_normal, t_defect) 튜플 딕셔너리
        auto_load: True일 경우 기본 프로덕션 가중치 자동 로드
        """
        # 기본 임계값 (실측 최적값 설정)
        self.view_thresholds = view_thresholds or {
            'view5': (0.260, 0.435), # 시점 A: 우측 후방 (45도)
            'view2': (0.475, 0.645), # 시점 B: 우측 전방 (45도)
            'view3': (0.375, 0.450), # 시점 C: 정면 (0도)
            'view4': (0.375, 0.575), # 시점 D: 좌측 전방 (45도)
            'view1': (0.250, 0.400)  # 시점 E: 좌측 후방 (45도)
        }
        self.dl_model: Optional[DeadliftSTSequenceNetV5_3] = None
        self.cb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.is_fitted = False

        if auto_load:
            default_weights = os.path.join(CURRENT_DIR, "d3_model_v5_3_opt_weights.pkl")
            if os.path.exists(default_weights):
                self.load_weights(default_weights)
            else:
                warnings.warn(f"기본 가중치 파일을 찾을 수 없습니다: {default_weights}")

    def fit(self, X_seq_57: np.ndarray, X_feats_17: np.ndarray, y: np.ndarray,
            epochs: int = 35, random_state: int = 42, device: str = 'cpu'):
        """
        17D 정예 피처 및 57D 시계열 데이터를 사용하여 모델 학습
        """
        if X_feats_17.shape[1] == 32:
            # 32D가 전달된 경우 17D로 자동 축소
            X_feats_17 = X_feats_17[:, OPTIMAL_17_INDICES]

        # 1. DL 모델 학습 (57D)
        dl_net = DeadliftSTSequenceNetV5_3(in_features=57, hidden_dim=64).to(device)
        optimizer = torch.optim.AdamW(dl_net.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_weight = torch.tensor([float(np.sum(y == 0)) / max(1, float(np.sum(y == 1)))]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        t_seq = torch.tensor(X_seq_57, dtype=torch.float32).to(device)
        t_y = torch.tensor(y, dtype=torch.float32).to(device)

        dl_net.train()
        batch_size = 32
        n_samples = len(y)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                b_idx = indices[i:i + batch_size]
                optimizer.zero_grad()
                logits = dl_net(t_seq[b_idx])
                loss = criterion(logits, t_y[b_idx])
                loss.backward()
                optimizer.step()

        self.dl_model = dl_net

        # 2. CatBoost 학습 (17D 정예셋)
        cb = CatBoostClassifier(
            iterations=350,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=random_state
        )
        cb.fit(X_feats_17, y)
        self.cb_model = cb

        # 3. LightGBM 학습 (17D 정예셋)
        ds_tr = lgb.Dataset(X_feats_17, label=y)
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.03,
            'num_leaves': 16,
            'max_depth': 4,
            'scale_pos_weight': float(pos_weight.item()),
            'verbosity': -1,
            'random_state': random_state
        }
        lgb_m = lgb.train(lgb_params, ds_tr, num_boost_round=230)
        self.lgb_model = lgb_m

        # 4. Random Forest 학습 (17D 정예셋)
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=random_state, n_jobs=-1)
        rf.fit(X_feats_17, y)
        self.rf_model = rf

        self.is_fitted = True

    def predict_proba(self, X_seq_57: np.ndarray, X_feats_17: np.ndarray, views: Optional[np.ndarray] = None) -> np.ndarray:
        """
        시점 적응형 앙상블 확률 추론
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if X_feats_17.shape[1] == 32:
            X_feats_17 = X_feats_17[:, OPTIMAL_17_INDICES]

        self.dl_model.eval()
        with torch.no_grad():
            t_seq = torch.tensor(X_seq_57, dtype=torch.float32)
            logits = self.dl_model(t_seq)
            p_dl = torch.sigmoid(logits).numpy()

        p_cb = self.cb_model.predict_proba(X_feats_17)[:, 1]
        p_lgb = self.lgb_model.predict(X_feats_17)
        p_rf = self.rf_model.predict_proba(X_feats_17)[:, 1]
        p_tree = 0.50 * p_cb + 0.30 * p_lgb + 0.20 * p_rf

        if views is not None:
            probs = np.zeros(len(p_dl), dtype=np.float32)
            for i, v in enumerate(views):
                w_dl = 0.55 if v in ['view5', 'view1'] else 0.40
                probs[i] = w_dl * p_dl[i] + (1.0 - w_dl) * p_tree[i]
            return probs
        else:
            return 0.45 * p_dl + 0.55 * p_tree

    def predict_3state(self, prob: float, view_code: str) -> Tuple[int, str]:
        """
        3-State 안전 게이트 판정
        반환: (state, label)
          - state: 0 (정상 확정), 1 (결함 확정), -1 (판단 보류 / Re-evaluate)
          - label: "NORMAL", "DEFECT", "ABSTAIN"
        """
        t_norm, t_def = self.view_thresholds.get(view_code, (0.35, 0.55))
        if prob <= t_norm:
            return 0, "NORMAL"
        elif prob >= t_def:
            return 1, "DEFECT"
        else:
            return -1, "ABSTAIN"

    def predict_single_sequence(self, kps_33_64f: np.ndarray, view_code: str) -> Dict[str, Any]:
        """
        단일 시퀀스(64, 33, 3)를 입력받아 최종 판정 결과를 딕셔너리로 반환
        """
        kps = clean_and_smooth_keypoints(kps_33_64f)
        kps_15 = np.zeros((64, 15, 3), dtype=np.float32)
        for idx_new, idx_mp in enumerate(KEY_JOINT_INDICES):
            kps_15[:, idx_new] = kps[:, idx_mp]

        # 45D 원본 좌표
        f_pos = kps_15.reshape(64, 45)

        # 6D 원본 발목 상대변위
        ank_mid = (kps_15[:, 9] + kps_15[:, 10]) / 2.0
        rel_orig_heel_l = kps_15[:, 11] - ank_mid
        rel_orig_heel_r = kps_15[:, 12] - ank_mid
        f_rel_orig = np.concatenate([rel_orig_heel_l, rel_orig_heel_r], axis=-1)

        # 6D KAPW 위상 정합 상대변위
        kps_kapw, _, _ = apply_kapw_phase_warping(kps_15)
        ank_mid_k = (kps_kapw[:, 9] + kps_kapw[:, 10]) / 2.0
        rel_kapw_heel_l = kps_kapw[:, 11] - ank_mid_k
        rel_kapw_heel_r = kps_kapw[:, 12] - ank_mid_k
        f_rel_kapw = np.concatenate([rel_kapw_heel_l, rel_kapw_heel_r], axis=-1)

        seq_57 = np.concatenate([f_pos, f_rel_orig, f_rel_kapw], axis=-1)[np.newaxis, ...] # (1, 64, 57)
        feat_17 = extract_d3_v5_3_opt_features_from_seq(kps_33_64f, view_code)[np.newaxis, ...] # (1, 17)

        prob = float(self.predict_proba(seq_57, feat_17, views=np.array([view_code]))[0])
        state, label = self.predict_3state(prob, view_code)

        return {
            'defect_code': self.DEFECT_CODE,
            'defect_name': self.DEFECT_NAME,
            'view_code': view_code,
            'prob': prob,
            'state': state,
            'label': label,
            'confidence': max(prob, 1.0 - prob) if state != -1 else 0.5,
            'features_17': {name: float(feat_17[0, i]) for i, name in enumerate(self.FEATURE_NAMES)}
        }

    def save_weights(self, out_path: str):
        """가중치 및 임계값 저장"""
        data = {
            'dl_state_dict': self.dl_model.state_dict() if self.dl_model else None,
            'cb_model': self.cb_model,
            'lgb_model': self.lgb_model,
            'rf_model': self.rf_model,
            'view_thresholds': self.view_thresholds,
            'feature_names': FEATURE_NAMES_17_D3,
            'feature_indices': OPTIMAL_17_INDICES
        }
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[+] v5.3-Opt 가중치 저장 완료: {out_path}")

    def load_weights(self, in_path: Optional[str] = None):
        """가중치 및 임계값 로드"""
        if in_path is None:
            in_path = os.path.join(CURRENT_DIR, "d3_model_v5_3_opt_weights.pkl")
        with open(in_path, 'rb') as f:
            data = pickle.load(f)
        if data['dl_state_dict']:
            self.dl_model = DeadliftSTSequenceNetV5_3(in_features=57, hidden_dim=64)
            self.dl_model.load_state_dict(data['dl_state_dict'])
        self.cb_model = data['cb_model']
        self.lgb_model = data['lgb_model']
        self.rf_model = data['rf_model']
        self.view_thresholds = data.get('view_thresholds', self.view_thresholds)
        self.is_fitted = True
        print(f"[+] v5.3-Opt 가중치 로드 완료: {in_path}")


# 바벨 데드리프트 d3 공식 확정 프로덕션 모델 별칭
BabelDeadliftD3Model = BabelDeadliftD3ModelV5_3Opt

