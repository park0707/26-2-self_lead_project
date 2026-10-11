# =========================================================================================================
# [모듈명]: train_hierarchical_kfold.py
# [목적]: 2-Stage 계층형(Hierarchical) 운동 분류 파이프라인 3-Fold 학습, 하이퍼파라미터 튜닝 및 4단계 다차원 평가
# [아키텍처]:
#   - Stage 1: 6대 상위 기능 그룹 3중 하이브리드 앙상블 (Bi-GRU 0.65 + XGBoost 0.20 + CatBoost 0.15)
#   - Stage 2-A: 데드리프트 패밀리 전용 2진 Expert (LightGBM + tau*_dead 그리드 튜닝)
#   - Stage 2-B: 무빙 런지 패밀리 전용 2진 Expert (LightGBM + tau*_cross 임계값 튜닝)
#   - Decision: 소프트 베이지안 확률 결합 (Soft Bayesian Marginalization)
# [실행 방법]:
#   python test_hierarchical_solution.py
# =========================================================================================================

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix
)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 프로젝트 루트 경로 등록
sys.path.insert(0, 'd:/')

from src.models.first_model.dataset import Stage1DatasetLoader, EXERCISE_NAMES_KO
from src.models.first_model.metrics_helper import Stage1MetricsHelper, EXERCISE_NAMES_SHORT
from src.models.first_model.train_augmented_ensemble import AdvancedGRUClassifier
from src.models.first_model.train_kfold_ensemble import DiscriminativeKinematicsExtractor
from src.models.sub_models.deadlift_sub_classifier import DeadliftSubClassifier
from src.models.sub_models.lunge_sub_classifier import LungeSubClassifier

OUTPUT_HIERARCHICAL_JSON = 'd:/Data/피처_데이터/tuning_results/stage1_hierarchical_ensemble_results.json'

# 8종 원본 클래스 -> 6대 상위 그룹 매핑
# 0: 스쿼트 (Class 0)
# 1: 데드리프트 패밀리 (Class 1: 일반데드, Class 2: 스티프데드)
# 2: 바벨 런지 (Class 3) - 단독 분리!
# 3: 무빙 런지 패밀리 (Class 4: 다이나믹런지, Class 6: 크로스런지)
# 4: 사이드 런지 (Class 5)
# 5: 굿모닝 (Class 7)
CLASS_TO_MACRO_GROUP = {
    0: 0, # 01. 스쿼트 -> Group 0
    1: 1, # 02. 일반데드 -> Group 1
    2: 1, # 03. 스티프데드 -> Group 1
    3: 2, # 04. 바벨런지 -> Group 2 (단독 분리)
    4: 3, # 05. 다이나믹런지 -> Group 3 (무빙 런지)
    5: 4, # 06. 사이드런지 -> Group 4
    6: 3, # 07. 크로스런지 -> Group 3 (무빙 런지)
    7: 5  # 08. 굿모닝 -> Group 5
}

MACRO_GROUP_NAMES = [
    'Group 0 (01. 바벨 스쿼트)',
    'Group 1 (데드리프트 패밀리: 02+03)',
    'Group 2 (04. 바벨 런지 - 단독)',
    'Group 3 (무빙 런지 패밀리: 05+07)',
    'Group 4 (06. 사이드 런지)',
    'Group 5 (08. 굿모닝)'
]

def _to_serializable(val: Any) -> Any:
    """numpy ndarray / int / float 객체를 표준 JSON 직렬화 가능 타입으로 재귀 변환 (NumPy 2.0+ 호환)"""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_serializable(v) for v in val]
    return val

class HierarchicalKFoldPipelineTrainer:
    """2-Stage 계층형 3-Fold 앙상블 학습, 튜닝 및 전수 평가 엔진"""

    def __init__(self,
                 data_dir: str = 'd:/Data/피처_데이터/Condition4_NovelFlow_3D' if os.path.exists('d:/Data/피처_데이터/Condition4_NovelFlow_3D/Dual_Train.npz') else 'd:/Data/피처_데이터/Condition3_MediaPipe_3D',
                 lunge_data_dir: Optional[str] = 'd:/Data/피처_데이터/Condition3_MediaPipe_3D' if os.path.exists('d:/Data/피처_데이터/Condition3_MediaPipe_3D/Dual_Train.npz') else None):
        self.data_dir = data_dir
        self.lunge_data_dir = lunge_data_dir or data_dir
        self.master_data = self._load_master_pool()

    def _load_master_pool(self) -> Dict[str, Any]:
        """마스터 풀 로드 및 판별 피처 결합 (Stage 1/2-A는 Condition 4, Stage 2-B는 초기 고정 좌표계 스트림 분기)"""
        all_X_a, all_X_b, all_y_act, all_subjs, all_views, all_sids = [], [], [], [], [], []
        all_X_b_lunge = []

        for fn in sorted(os.listdir(self.data_dir)):
            if not fn.endswith('.npz'):
                continue
            split_data = Stage1DatasetLoader.load_split_data(fn, splits_dir=self.data_dir)
            all_X_a.append(split_data['X_Format_A'])
            all_X_b.append(split_data['X_Format_B'])
            all_y_act.append(split_data['y_action'])
            all_subjs.extend(split_data['subjects'])
            all_views.extend(split_data['views'])
            all_sids.extend(split_data['sequence_ids'])

            if self.lunge_data_dir and os.path.exists(os.path.join(self.lunge_data_dir, fn)):
                split_lunge = Stage1DatasetLoader.load_split_data(fn, splits_dir=self.lunge_data_dir)
                all_X_b_lunge.append(split_lunge['X_Format_B'])
            else:
                all_X_b_lunge.append(split_data['X_Format_B'])

        def get_subj_num(s):
            s = str(s)
            for digit in ['1','2','3','4','5','6','7','8','9']:
                if f"피험자{digit}" in s or s == digit or f"_{digit}_" in s:
                    return int(digit)
            return 1

        subjs_arr = np.array([get_subj_num(s) for s in all_subjs])
        views_arr = np.array(all_views)
        raw_X_a = np.concatenate(all_X_a, axis=0)
        raw_X_b = np.concatenate(all_X_b, axis=0)
        raw_X_b_lunge = np.concatenate(all_X_b_lunge, axis=0) if all_X_b_lunge else raw_X_b
        y_act = np.concatenate(all_y_act, axis=0)

        # 특화 생체역학 피처 추출 및 Format A 융합
        print("[*] 3D 생체역학 판별 피처 융합 중...")
        discrim_feats = DiscriminativeKinematicsExtractor.extract_features(raw_X_b)
        enhanced_X_a = np.concatenate([raw_X_a, discrim_feats], axis=-1)

        # 6대 상위 그룹 라벨 생성
        y_macro = np.array([CLASS_TO_MACRO_GROUP[c] for c in y_act], dtype=np.int64)

        return {
            'X_Format_A': enhanced_X_a,
            'X_Format_B': raw_X_b,
            'X_Format_B_lunge': raw_X_b_lunge,
            'y_action': y_act,
            'y_macro': y_macro,
            'subjects': subjs_arr,
            'views': views_arr,
            'sequence_ids': np.array(all_sids),
            'num_classes_macro': len(MACRO_GROUP_NAMES),
            'num_classes_final': len(EXERCISE_NAMES_KO)
        }

    def run_hierarchical_3fold_cv(self) -> Dict[str, Any]:
        """2-Stage 계층형 3-Fold 교차 검증 및 세부 튜닝 전수 실행"""
        folds_config = [
            {
                'fold_id': 1,
                'name': 'Fold 1 [Test: 피험자 1~3 x 전방(0°,45°)]',
                'test_subjs': [1, 2, 3],
                'test_views': ['view1', 'view2'],
                'train_subjs': [4, 5, 6, 7, 8, 9],
                'train_views': ['view3', 'view4', 'view5']
            },
            {
                'fold_id': 2,
                'name': 'Fold 2 [Test: 피험자 4~6 x 측면(90°)]',
                'test_subjs': [4, 5, 6],
                'test_views': ['view3'],
                'train_subjs': [1, 2, 3, 7, 8, 9],
                'train_views': ['view1', 'view2', 'view4', 'view5']
            },
            {
                'fold_id': 3,
                'name': 'Fold 3 [Test: 피험자 7~9 x 후방(135°,180°)]',
                'test_subjs': [7, 8, 9],
                'test_views': ['view4', 'view5'],
                'train_subjs': [1, 2, 3, 4, 5, 6],
                'train_views': ['view1', 'view2', 'view3']
            }
        ]

        print("\n" + "=" * 115)
        print("★ [2-Stage 계층형 운동 분류 파이프라인]: 6대 상위 그룹 앙상블 + 2대 특화 서브 모델 + 소프트 베이지안 결합")
        print(f"  - 마스터 풀 크기: {len(self.master_data['y_action']):,}개 시퀀스 | 3-Fold 복합 교차 검증")
        print("=" * 115)

        stage1_fold_reports = []
        sub_dead_fold_reports = []
        sub_lunge_fold_reports = []
        final_fold_reports = []

        all_final_preds = []
        all_final_trues = []
        all_macro_preds = []
        all_macro_trues = []

        t_global_start = time.time()

        for fc in folds_config:
            fold_id = fc['fold_id']
            print(f"\n>>> 🚀 [{fc['name']}] 훈련 및 튜닝 시작...")

            test_mask = np.isin(self.master_data['subjects'], fc['test_subjs']) & np.isin(self.master_data['views'], fc['test_views'])
            train_mask = np.isin(self.master_data['subjects'], fc['train_subjs']) & np.isin(self.master_data['views'], fc['train_views'])

            X_tr_a, X_tr_b, y_tr_act, y_tr_mac = (
                self.master_data['X_Format_A'][train_mask],
                self.master_data['X_Format_B'][train_mask],
                self.master_data['y_action'][train_mask],
                self.master_data['y_macro'][train_mask]
            )
            X_te_a, X_te_b, y_te_act, y_te_mac = (
                self.master_data['X_Format_A'][test_mask],
                self.master_data['X_Format_B'][test_mask],
                self.master_data['y_action'][test_mask],
                self.master_data['y_macro'][test_mask]
            )
            X_tr_b_lunge = self.master_data['X_Format_B_lunge'][train_mask]
            X_te_b_lunge = self.master_data['X_Format_B_lunge'][test_mask]

            # -------------------------------------------------------------
            # [Stage 1] 6-Group 상위 그룹 3중 앙상블 학습
            # -------------------------------------------------------------
            t_s1 = time.time()
            num_macro_classes = self.master_data['num_classes_macro']
            class_counts_mac = np.bincount(y_tr_mac, minlength=num_macro_classes)
            raw_w_mac = len(y_tr_mac) / (float(num_macro_classes) * np.maximum(class_counts_mac, 1).astype(np.float32))
            smooth_w_mac = np.sqrt(raw_w_mac)
            sample_w_mac = smooth_w_mac[y_tr_mac]

            print(f"    [1/3] Stage 1 (6-Group 마스터 앙상블) 훈련 중 (Bi-GRU + XGB + CatBoost)...", flush=True)
            gru_macro = AdvancedGRUClassifier(
                input_dim=X_tr_b.shape[2],
                num_classes=num_macro_classes,
                epochs=12,
                lr=0.003,
                batch_size=512
            )
            gru_macro.fit(X_tr_b, y_tr_mac, class_weights=smooth_w_mac)

            xgb_macro = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.85, colsample_bytree=0.85, random_state=42,
                eval_metric='mlogloss', n_jobs=4
            )
            xgb_macro.fit(X_tr_a, y_tr_mac, sample_weight=sample_w_mac)

            cat_macro = CatBoostClassifier(
                iterations=100, depth=5, learning_rate=0.1,
                thread_count=4, random_seed=42, class_weights=smooth_w_mac.tolist(),
                verbose=False
            )
            cat_macro.fit(X_tr_a, y_tr_mac)

            # Stage 1 예측 확률 산출 (N, 6)
            p_gru_mac = gru_macro.predict_proba(X_te_b)
            p_xgb_mac = xgb_macro.predict_proba(X_te_a)
            p_cat_mac = cat_macro.predict_proba(X_te_a)
            P_macro_te = (0.65 * p_gru_mac) + (0.20 * p_xgb_mac) + (0.15 * p_cat_mac)
            preds_macro_te = np.argmax(P_macro_te, axis=-1)

            s1_acc = accuracy_score(y_te_mac, preds_macro_te)
            s1_f1 = f1_score(y_te_mac, preds_macro_te, average='macro', zero_division=0)
            stage1_fold_reports.append({
                'fold_name': fc['name'],
                'accuracy': float(s1_acc),
                'macro_f1': float(s1_f1),
                'cm': confusion_matrix(y_te_mac, preds_macro_te, labels=list(range(num_macro_classes))).tolist()
            })
            all_macro_preds.extend(preds_macro_te)
            all_macro_trues.extend(y_te_mac)
            print(f"      ✔ Stage 1 완료 -> 정확도: {s1_acc*100:.2f}%, Macro-F1: {s1_f1*100:.2f}% (소요: {time.time()-t_s1:.1f}초)")

            # -------------------------------------------------------------
            # [Stage 2-A] 하체 7개 관절 전용 Bi-GRU + LightGBM 2중 앙상블 Expert
            # 대상: 1 (일반데드) vs 2 (스티프데드) -> 2진 라벨 (0 vs 1)
            # -------------------------------------------------------------
            print(f"    [2/3] Stage 2-A (하체 7-Joint Bi-GRU + LightGBM 2중 앙상블) 훈련 및 튜닝 중...", flush=True)
            dead_tr_mask = (y_tr_act == 1) | (y_tr_act == 2)
            dead_te_mask = (y_te_act == 1) | (y_te_act == 2)

            X_dead_tr_raw = X_tr_b[dead_tr_mask]
            y_dead_tr_bin = (y_tr_act[dead_tr_mask] == 2).astype(int) # 0: 일반데드, 1: 스티프데드

            X_dead_te_raw_full = X_te_b # 전체 테스트용
            X_dead_te_raw_eval = X_te_b[dead_te_mask]
            y_dead_te_bin_eval = (y_te_act[dead_te_mask] == 2).astype(int)

            sub_dead_clf = DeadliftSubClassifier()
            dead_tune_res = sub_dead_clf.tune_and_fit(X_dead_tr_raw, y_dead_tr_bin)
            dead_eval_res = sub_dead_clf.evaluate(X_dead_te_raw_eval, y_dead_te_bin_eval)
            dead_eval_res['fold_name'] = fc['name']
            dead_eval_res['best_params'] = dead_tune_res['best_params']
            sub_dead_fold_reports.append(dead_eval_res)

            print(f"      ✔ Stage 2-A 완료 -> Best τ*={dead_tune_res['best_threshold']:.2f}, 스티프데드 Recall: {dead_eval_res['stiff_recall']*100:.2f}%, F1: {dead_eval_res['stiff_f1']*100:.2f}%")

            # -------------------------------------------------------------
            # [Stage 2-B] 무빙 런지 패밀리 전용 2진 Expert 튜닝 & 학습
            # 대상: 4 (다이나믹런지=0) vs 6 (크로스런지=1) (비왜곡 초기 고정 좌표계 피처 스트림 적용)
            # -------------------------------------------------------------
            print(f"    [3/3] Stage 2-B (무빙 런지 2진 Expert) 15D 최하사점 위상인식/프록시 피처 + 지능형 증강 튜닝 및 학습 중...", flush=True)
            lunge_tr_mask = (y_tr_act == 4) | (y_tr_act == 6)
            lunge_te_mask = (y_te_act == 4) | (y_te_act == 6)

            X_lunge_tr_sub = LungeSubClassifier.slice_features(X_tr_b_lunge[lunge_tr_mask])
            y_lunge_tr_bin = (y_tr_act[lunge_tr_mask] == 6).astype(int) # 0: 다이나믹, 1: 크로스

            X_lunge_te_sub_full = LungeSubClassifier.slice_features(X_te_b_lunge)
            X_lunge_te_sub_eval = LungeSubClassifier.slice_features(X_te_b_lunge[lunge_te_mask])
            y_lunge_te_bin_eval = (y_te_act[lunge_te_mask] == 6).astype(int)

            sub_lunge_clf = LungeSubClassifier()
            lunge_tune_res = sub_lunge_clf.tune_and_fit(X_lunge_tr_sub, y_lunge_tr_bin)
            lunge_eval_res = sub_lunge_clf.evaluate(X_lunge_te_sub_eval, y_lunge_te_bin_eval)
            lunge_eval_res['fold_name'] = fc['name']
            lunge_eval_res['best_params'] = lunge_tune_res['best_params']
            sub_lunge_fold_reports.append(lunge_eval_res)

            print(f"      ✔ Stage 2-B 완료 -> Best τ*={lunge_tune_res['best_threshold']:.2f}, 크로스런지 Recall: {lunge_eval_res['cross_recall']*100:.2f}%, F1: {lunge_eval_res['cross_f1']*100:.2f}%")

            # -------------------------------------------------------------
            # [통합 추론] 소프트 베이지안 확률 결합 (Soft Bayesian Marginalization)
            # -------------------------------------------------------------
            t_inf = time.time()
            P_dead_stiff_probs = sub_dead_clf.predict_proba(X_dead_te_raw_full)[:, 1] # P(Stiff)
            P_lunge_cross_probs = sub_lunge_clf.predict_proba(X_lunge_te_sub_full)[:, 1] # P(Cross)

            P_final_te = np.zeros((len(y_te_act), 8), dtype=np.float32)

            # 1. 단독 그룹 직접 매핑
            P_final_te[:, 0] = P_macro_te[:, 0] # 01. 바벨 스쿼트 (Group 0)
            P_final_te[:, 3] = P_macro_te[:, 2] # 04. 바벨 런지 (Group 2 - 단독 분리!)
            P_final_te[:, 5] = P_macro_te[:, 4] # 06. 사이드 런지 (Group 4)
            P_final_te[:, 7] = P_macro_te[:, 5] # 08. 굿모닝 (Group 5)

            # 2. 데드리프트 패밀리 확률 분배 (Group 1: 오즈비 확률 보정)
            tau_dead = np.clip(sub_dead_clf.best_threshold, 0.05, 0.95)
            eps = 1e-6
            p_stiff_safe = np.clip(P_dead_stiff_probs, eps, 1.0 - eps)
            odds_dead = ((1.0 - p_stiff_safe) / p_stiff_safe) * (tau_dead / (1.0 - tau_dead))
            stiff_calib = 1.0 / (1.0 + odds_dead)
            P_final_te[:, 1] = P_macro_te[:, 1] * (1.0 - stiff_calib) # 02. 바벨 데드리프트
            P_final_te[:, 2] = P_macro_te[:, 1] * stiff_calib         # 03. 스티프 데드리프트

            # 3. 무빙 런지 패밀리 확률 분배 (Group 3: 오즈비 확률 보정)
            tau_cross = np.clip(sub_lunge_clf.best_threshold, 0.05, 0.95)
            p_cross_safe = np.clip(P_lunge_cross_probs, eps, 1.0 - eps)
            odds_cross = ((1.0 - p_cross_safe) / p_cross_safe) * (tau_cross / (1.0 - tau_cross))
            cross_calib = 1.0 / (1.0 + odds_cross)
            P_final_te[:, 4] = P_macro_te[:, 3] * (1.0 - cross_calib) # 05. 다이나믹 런지
            P_final_te[:, 6] = P_macro_te[:, 3] * cross_calib         # 07. 크로스 런지

            # 계층형 탑다운 의사결정 게이트 (Cascaded Decision Gate):
            # Stage 1 상위 그룹 결정을 우선 고정하여 타 6종 운동 간섭을 0.00% 차단
            macro_top1 = np.argmax(P_macro_te, axis=-1)
            final_preds = np.argmax(P_final_te, axis=-1)

            # Stage 1이 무빙 런지(Group 3)로 판정한 샘플은 Stage 2-B 최적 임계값(tau_cross)으로 엄밀 분기
            lunge_mask_te = (macro_top1 == 3)
            final_preds[lunge_mask_te] = np.where(P_lunge_cross_probs[lunge_mask_te] >= tau_cross, 6, 4)

            # Stage 1이 데드리프트 패밀리(Group 1)로 판정한 샘플은 Stage 2-A 최적 임계값(tau_dead)으로 엄밀 분기
            dead_mask_te = (macro_top1 == 1)
            final_preds[dead_mask_te] = np.where(P_dead_stiff_probs[dead_mask_te] >= tau_dead, 2, 1)

            inf_time_ms = ((time.time() - t_inf) / max(1, len(y_te_act))) * 1000.0

            res_final = Stage1MetricsHelper.evaluate_predictions(
                y_true=y_te_act,
                y_pred=final_preds,
                y_proba=P_final_te,
                inference_time_ms=inf_time_ms
            )
            res_final['fold_name'] = fc['name']
            final_fold_reports.append(res_final)

            all_final_preds.extend(final_preds)
            all_final_trues.extend(y_te_act)

            print(f"    [+] Fold {fold_id} 최종 통합 완료 -> 종합 Acc: {res_final['accuracy']*100:.2f}%, Macro-F1: {res_final['macro_f1']*100:.2f}%, FPS: {res_final['fps']:.1f}")

        # 전체 리포팅 출력
        self._print_all_detailed_reports(
            stage1_fold_reports,
            sub_dead_fold_reports,
            sub_lunge_fold_reports,
            final_fold_reports,
            np.array(all_macro_trues),
            np.array(all_macro_preds),
            np.array(all_final_trues),
            np.array(all_final_preds),
            time.time() - t_global_start
        )

        return {
            'stage1_reports': stage1_fold_reports,
            'sub_dead_reports': sub_dead_fold_reports,
            'sub_lunge_reports': sub_lunge_fold_reports,
            'final_reports': final_fold_reports
        }

    def _print_all_detailed_reports(
        self,
        s1_reports, sub_dead_reports, sub_lunge_reports, final_reports,
        y_macro_true, y_macro_pred, y_final_true, y_final_pred, total_elapsed
    ):
        """4단계 다차원 터미널 세부 성적표 출력"""
        num_macro = len(MACRO_GROUP_NAMES)
        print("\n" + "=" * 115)
        print("★ [리포트 1] Stage 1: 상위 6대 메가 그룹 분류 3-Fold 성능 성적표")
        print("=" * 115)
        print(f"{'상위 그룹 명칭':<38} | {'정밀도(Prec)':^12} | {'재현율(Rec)':^12} | {'F1-Score':^12}")
        print("-" * 115)

        s1_cm = confusion_matrix(y_macro_true, y_macro_pred, labels=list(range(num_macro)))
        s1_precs = precision_score(y_macro_true, y_macro_pred, average=None, labels=list(range(num_macro)), zero_division=0)
        s1_recs = recall_score(y_macro_true, y_macro_pred, average=None, labels=list(range(num_macro)), zero_division=0)
        s1_f1s = f1_score(y_macro_true, y_macro_pred, average=None, labels=list(range(num_macro)), zero_division=0)

        for i, g_name in enumerate(MACRO_GROUP_NAMES):
            print(f"{g_name:<38} | {s1_precs[i]*100:>11.2f}% | {s1_recs[i]*100:>11.2f}% | {s1_f1s[i]*100:>11.2f}%")
        print("-" * 115)
        s1_mean_acc = np.mean([r['accuracy'] for r in s1_reports]) * 100.0
        s1_mean_f1 = np.mean([r['macro_f1'] for r in s1_reports]) * 100.0
        print(f"★ Stage 1 3-Fold 평균: 정확도 {s1_mean_acc:.2f}% | Macro-F1 {s1_mean_f1:.2f}%")

        # 리포트 2: Stage 2-A
        print("\n" + "=" * 115)
        print("★ [리포트 2] Stage 2-A: 데드리프트 패밀리 전용 2진 Expert 튜닝 및 성능 성적표")
        print("=" * 115)
        avg_dead_f1 = np.mean([r['dead_f1'] for r in sub_dead_reports]) * 100.0
        avg_dead_rec = np.mean([r['dead_recall'] for r in sub_dead_reports]) * 100.0
        avg_dead_prec = np.mean([r['dead_precision'] for r in sub_dead_reports]) * 100.0

        avg_stiff_f1 = np.mean([r['stiff_f1'] for r in sub_dead_reports]) * 100.0
        avg_stiff_rec = np.mean([r['stiff_recall'] for r in sub_dead_reports]) * 100.0
        avg_stiff_prec = np.mean([r['stiff_precision'] for r in sub_dead_reports]) * 100.0
        avg_roc_auc_dead = np.mean([r['roc_auc'] for r in sub_dead_reports])

        print(f"• 평균 최적 결정 임계값 τ*_dead: {np.mean([r['best_threshold'] for r in sub_dead_reports]):.2f} | 평균 ROC-AUC: {avg_roc_auc_dead:.4f}")
        print("-" * 115)
        print(f"{'세부 운동 종목':<30} | {'정밀도 (Precision)':<22} | {'재현율 (Recall)':<22} | {'F1-Score':<15}")
        print("-" * 115)
        print(f"{'02. 바벨 데드리프트 (Class 1)':<26} | {avg_dead_prec:6.2f}%                 | {avg_dead_rec:6.2f}%                 | {avg_dead_f1:6.2f}%")
        print(f"{'03. 스티프 데드리프트 (Class 2)':<25} | {avg_stiff_prec:6.2f}%                 | {avg_stiff_rec:6.2f}%                 | {avg_stiff_f1:6.2f}%")
        print("=" * 115)

        # 리포트 3: Stage 2-B
        print("\n" + "=" * 115)
        print("★ [리포트 3] Stage 2-B: 무빙 런지 패밀리 전용 2진 Expert 튜닝 및 성능 성적표")
        print("=" * 115)
        avg_dyn_f1 = np.mean([r['dynamic_f1'] for r in sub_lunge_reports]) * 100.0
        avg_dyn_rec = np.mean([r['dynamic_recall'] for r in sub_lunge_reports]) * 100.0
        avg_dyn_prec = np.mean([r['dynamic_precision'] for r in sub_lunge_reports]) * 100.0

        avg_cross_f1 = np.mean([r['cross_f1'] for r in sub_lunge_reports]) * 100.0
        avg_cross_rec = np.mean([r['cross_recall'] for r in sub_lunge_reports]) * 100.0
        avg_cross_prec = np.mean([r['cross_precision'] for r in sub_lunge_reports]) * 100.0
        avg_roc_auc_lunge = np.mean([r['roc_auc'] for r in sub_lunge_reports])

        print(f"• 평균 최적 결정 임계값 τ*_cross: {np.mean([r['best_threshold'] for r in sub_lunge_reports]):.2f} | 평균 ROC-AUC: {avg_roc_auc_lunge:.4f}")
        print("-" * 115)
        print(f"{'세부 운동 종목':<30} | {'정밀도 (Precision)':<22} | {'재현율 (Recall)':<22} | {'F1-Score':<15}")
        print("-" * 115)
        print(f"{'05. 다이나믹 런지 (Class 4)':<26} | {avg_dyn_prec:6.2f}%                 | {avg_dyn_rec:6.2f}%                 | {avg_dyn_f1:6.2f}%")
        print(f"{'07. 크로스 런지 (Class 6)':<26} | {avg_cross_prec:6.2f}%                 | {avg_cross_rec:6.2f}%                 | {avg_cross_f1:6.2f}%")
        print("=" * 115)

        # 리포트 4: End-to-End 최종 8종 통합
        print("\n" + "=" * 115)
        print("★ [리포트 4] End-to-End: 최종 8종 운동 통합 성적표 (소프트 베이지안 결합)")
        print("=" * 115)
        print(f"{'운동 종목 명칭':<20} | {'샘플수':^7} | {'정밀도(Prec)':^12} | {'재현율(Rec)':^12} | {'F1-Score':^12}")
        print("-" * 115)

        total_res = Stage1MetricsHelper.evaluate_predictions(
            y_true=y_final_true,
            y_pred=y_final_pred,
            inference_time_ms=np.mean([r['inference_time_ms'] for r in final_reports])
        )

        for i, ex_name in enumerate(EXERCISE_NAMES_SHORT):
            cnt = int(np.sum(y_final_true == i))
            prec = total_res['class_precs'][i] * 100.0
            rec = total_res['class_recs'][i] * 100.0
            f1 = total_res['class_f1s'][i] * 100.0
            print(f"{ex_name:<20} | {cnt:>7d} | {prec:>11.2f}% | {rec:>11.2f}% | {f1:>11.2f}%")
        print("-" * 115)

        accs = [r['accuracy'] * 100.0 for r in final_reports]
        f1s = [r['macro_f1'] * 100.0 for r in final_reports]
        print(f"★ 최종 3-Fold 평균 정확도 (Accuracy): {np.mean(accs):.2f}% (±{np.std(accs):.2f}%)")
        print(f"★ 최종 3-Fold 평균 Macro-F1 Score  : {np.mean(f1s):.2f}% (±{np.std(f1s):.2f}%)")
        print(f"★ 전체 3-Fold 총 소요 시간         : {total_elapsed:.1f}초")
        print("=" * 115)

        # JSON 저장
        os.makedirs(os.path.dirname(OUTPUT_HIERARCHICAL_JSON), exist_ok=True)
        serializable_output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mean_accuracy': float(np.mean(accs)),
            'mean_macro_f1': float(np.mean(f1s)),
            'total_summary': _to_serializable(total_res)
        }
        with open(OUTPUT_HIERARCHICAL_JSON, 'w', encoding='utf-8') as f:
            json.dump(serializable_output, f, indent=2, ensure_ascii=False)
        print(f"\n[+] 계층형 3-Fold 결과 JSON 저장 완료: {OUTPUT_HIERARCHICAL_JSON}\n")

if __name__ == '__main__':
    trainer = HierarchicalKFoldPipelineTrainer()
    trainer.run_hierarchical_3fold_cv()

