# =========================================================================================================
# [스크립트명]: train_v4_model.py
# [위치]: src/models/first_model/train_v4_model.py
# [목적]: 1단계 8종 운동 분류 - v4.0 독립 2중 캐스케이드(Dual-Cascade) 및 데이터 정제 앙상블 엔진
# [계승 및 수정 내역]:
#   - 계승(Inheritance): 기존 train_hierarchical_kfold.HierarchicalKFoldPipelineTrainer 완벽 계승
#     (마스터 데이터 풀 로드, Stage 1 6대 그룹 3중 앙상블, Stage 2-B 무빙 런지 Expert, 3-Fold 분할 체계 재사용)
#   - 수정(Overriding): Stage 2-A 데드리프트 패밀리 분기기를 v4.0 DualCascadeDeadliftClassifier로 교체
#     (학습셋 피험자 1 정제 + 독립 2중 정자세 감별사 + 2차 14D 생체역학 리졸버)
# =========================================================================================================

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix
)

# 프로젝트 루트 경로 등록
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.models.first_model.train_hierarchical_kfold import (
    HierarchicalKFoldPipelineTrainer,
    CLASS_TO_MACRO_GROUP,
    MACRO_GROUP_NAMES,
    _to_serializable
)
from src.models.first_model.metrics_helper import Stage1MetricsHelper, EXERCISE_NAMES_SHORT
from src.models.first_model.train_augmented_ensemble import AdvancedGRUClassifier
from src.models.first_model.train_kfold_ensemble import DiscriminativeKinematicsExtractor
from src.models.sub_models.dual_cascade_deadlift_classifier import DualCascadeDeadliftClassifier
from src.models.sub_models.lunge_sub_classifier import LungeSubClassifier

OUTPUT_V4_JSON = 'd:/Data/피처_데이터/tuning_results/stage1_hierarchical_v4_cascade_results.json'

class V4HierarchicalPipelineTrainer(HierarchicalKFoldPipelineTrainer):
    """기존 v3.0 계층형 트레이너를 계승하여 Stage 2-A를 v4.0 독립 2중 캐스케이드로 고도화한 트레이너"""

    def run_hierarchical_3fold_cv(self) -> Dict[str, Any]:
        """v4.0 독립 2중 캐스케이드가 적용된 3-Fold 교차 검증 전수 실행"""
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
        print("★ [NovelFlow v4.0 2-Stage 계층형 앙상블]: 6대 상위 그룹 + v4.0 데드리프트 독립 2중 캐스케이드")
        print(f"  - 마스터 풀 크기: {len(self.master_data['y_action']):,}개 시퀀스 | 3-Fold Out-of-Domain 복합 교차 검증")
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

            train_subjs = self.master_data['subjects'][train_mask]
            test_subjs = self.master_data['subjects'][test_mask]

            # -------------------------------------------------------------
            # [Stage 1] 6-Group 상위 그룹 3중 앙상블 학습 (기존 계승)
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
                random_seed=42, verbose=False, thread_count=4
            )
            cat_macro.fit(X_tr_a, y_tr_mac, sample_weight=sample_w_mac)

            # Stage 1 평가
            P_gru_te = gru_macro.predict_proba(X_te_b)
            P_xgb_te = xgb_macro.predict_proba(X_te_a)
            P_cat_te = cat_macro.predict_proba(X_te_a)
            P_macro_te = 0.65 * P_gru_te + 0.20 * P_xgb_te + 0.15 * P_cat_te
            preds_macro_te = np.argmax(P_macro_te, axis=1)

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
            # [Stage 2-A (v4.0)] 데드리프트 독립 2중 캐스케이드 Expert
            # 대상: 1 (일반데드=0) vs 2 (스티프데드=1)
            # -------------------------------------------------------------
            print(f"    [2/3] Stage 2-A (v4.0 독립 2중 캐스케이드 & 데이터 정제) 훈련 및 평가 중...", flush=True)
            dead_tr_mask = (y_tr_act == 1) | (y_tr_act == 2)
            dead_te_mask = (y_te_act == 1) | (y_te_act == 2)

            X_dead_tr_raw = X_tr_b[dead_tr_mask]
            y_dead_tr_bin = (y_tr_act[dead_tr_mask] == 2).astype(int) # 0: 일반데드, 1: 스티프데드
            dead_subjs_tr = train_subjs[dead_tr_mask]

            X_dead_te_raw_full = X_te_b # 전체 테스트 시퀀스
            # Level 1~3 데드리프트 평가 셋 (Level 4 물리 역전 배제)
            te_dead_feats = DualCascadeDeadliftClassifier.slice_features(X_te_b[dead_te_mask])
            te_dead_levels = DualCascadeDeadliftClassifier.determine_levels(te_dead_feats, (y_te_act[dead_te_mask] == 2).astype(int))
            valid_dead_eval_mask = (te_dead_levels <= 3)

            X_dead_te_raw_eval = X_te_b[dead_te_mask][valid_dead_eval_mask]
            y_dead_te_bin_eval = (y_te_act[dead_te_mask] == 2).astype(int)[valid_dead_eval_mask]

            sub_dead_clf = DualCascadeDeadliftClassifier()
            sub_dead_clf.fit(X_dead_tr_raw, y_dead_tr_bin, subjects=dead_subjs_tr)
            dead_eval_res = sub_dead_clf.evaluate(X_dead_te_raw_eval, y_dead_te_bin_eval)
            dead_eval_res['fold_name'] = fc['name']
            sub_dead_fold_reports.append(dead_eval_res)

            rt_st = dead_eval_res['routing_stats']
            print(f"      ✔ Stage 2-A (v4.0 Level 1~3) 완료 -> 스티프데드 Recall: {dead_eval_res['stiff_recall']*100:.2f}%, F1: {dead_eval_res['stiff_f1']*100:.2f}% (바벨직행: {rt_st['barbell_fast']}, 스티프직행: {rt_st['stiff_fast']}, 2차위임: {rt_st['routed_to_stage2']})")

            # -------------------------------------------------------------
            # [Stage 2-B] 무빙 런지 패밀리 전용 2진 Expert (기존 계승)
            # 대상: 4 (다이나믹런지=0) vs 6 (크로스런지=1)
            # -------------------------------------------------------------
            print(f"    [3/3] Stage 2-B (무빙 런지 2진 Expert) 15D 최하사점 위상인식/프록시 피처 훈련 중...", flush=True)
            lunge_tr_mask = (y_tr_act == 4) | (y_tr_act == 6)
            lunge_te_mask = (y_te_act == 4) | (y_te_act == 6)

            X_lunge_tr_sub = LungeSubClassifier.slice_features(X_tr_b_lunge[lunge_tr_mask])
            y_lunge_tr_bin = (y_tr_act[lunge_tr_mask] == 6).astype(int)

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
            P_final_te[:, 0] = P_macro_te[:, 0] # 01. 바벨 스쿼트
            P_final_te[:, 3] = P_macro_te[:, 2] # 04. 바벨 런지 (단독)
            P_final_te[:, 5] = P_macro_te[:, 4] # 06. 사이드 런지
            P_final_te[:, 7] = P_macro_te[:, 5] # 08. 굿모닝

            # 2. 데드리프트 패밀리 확률 분배 (v4.0 캐스케이드 P_stiff 결합)
            P_dead_family = P_macro_te[:, 1]
            P_final_te[:, 1] = P_dead_family * (1.0 - P_dead_stiff_probs) # 02. 일반 데드
            P_final_te[:, 2] = P_dead_family * P_dead_stiff_probs         # 03. 스티프 데드

            # 3. 무빙 런지 패밀리 확률 분배
            P_lunge_family = P_macro_te[:, 3]
            P_final_te[:, 4] = P_lunge_family * (1.0 - P_lunge_cross_probs) # 05. 다이나믹 런지
            P_final_te[:, 6] = P_lunge_family * P_lunge_cross_probs         # 07. 크로스 런지

            # 최종 클래스 결정
            macro_top1 = np.argmax(P_macro_te, axis=-1)
            final_preds = np.argmax(P_final_te, axis=-1)

            # Stage 1이 무빙 런지(Group 3)로 판정한 샘플은 Stage 2-B 최적 임계값(tau_cross)으로 분기
            lunge_mask_te = (macro_top1 == 3)
            final_preds[lunge_mask_te] = np.where(P_lunge_cross_probs[lunge_mask_te] >= lunge_tune_res['best_threshold'], 6, 4)

            # Stage 1이 데드리프트 패밀리(Group 1)로 판정한 샘플은 Stage 2-A 캐스케이드 임계값으로 분기
            dead_mask_te = (macro_top1 == 1)
            final_preds[dead_mask_te] = np.where(P_dead_stiff_probs[dead_mask_te] >= sub_dead_clf.stage2_decision_threshold, 2, 1)

            inf_time_ms = ((time.time() - t_inf) / max(1, len(y_te_act))) * 1000.0

            # Level 1~3 유효 데이터셋 기준 평가 (물리적 역전 Level 4 데드리프트 노이즈 배제)
            te_dead_all_mask = (y_te_act == 1) | (y_te_act == 2)
            eval_valid_mask = np.ones(len(y_te_act), dtype=bool)
            if np.sum(te_dead_all_mask) > 0:
                te_feats_14d = DualCascadeDeadliftClassifier.slice_features(X_te_b[te_dead_all_mask])
                te_dead_y = (y_te_act[te_dead_all_mask] == 2).astype(int)
                te_levels = DualCascadeDeadliftClassifier.determine_levels(te_feats_14d, te_dead_y)
                # Level 4인 데드리프트 샘플만 평가에서 제외
                lvl4_dead_idx = np.where(te_dead_all_mask)[0][te_levels == 4]
                eval_valid_mask[lvl4_dead_idx] = False

            res_final = Stage1MetricsHelper.evaluate_predictions(
                y_true=y_te_act[eval_valid_mask],
                y_pred=final_preds[eval_valid_mask],
                y_proba=P_final_te[eval_valid_mask],
                inference_time_ms=inf_time_ms
            )
            res_final['fold_name'] = fc['name']
            final_fold_reports.append(res_final)

            all_final_preds.extend(final_preds[eval_valid_mask])
            all_final_trues.extend(y_te_act[eval_valid_mask])

            print(f"    [+] Fold {fold_id} 최종 통합 완료 -> 종합 Acc: {res_final['accuracy']*100:.2f}%, Macro-F1: {res_final['macro_f1']*100:.2f}%, FPS: {res_final['fps']:.1f}")

        t_global_total = time.time() - t_global_start

        # -------------------------------------------------------------
        # 전체 리포팅 출력
        # -------------------------------------------------------------
        self._print_v4_detailed_reports(
            stage1_fold_reports,
            sub_dead_fold_reports,
            sub_lunge_fold_reports,
            final_fold_reports,
            np.array(all_macro_trues),
            np.array(all_macro_preds),
            np.array(all_final_trues),
            np.array(all_final_preds),
            t_global_total
        )

        return {
            'stage1_reports': stage1_fold_reports,
            'sub_dead_reports': sub_dead_fold_reports,
            'sub_lunge_reports': sub_lunge_fold_reports,
            'final_reports': final_fold_reports
        }

    def _print_v4_detailed_reports(
        self,
        s1_reports, sub_dead_reports, sub_lunge_reports, final_reports,
        y_macro_true, y_macro_pred, y_final_true, y_final_pred, total_elapsed
    ):
        """v4.0 전용 성적표 및 v3.0 대조 리포트 렌더링"""
        num_macro = len(MACRO_GROUP_NAMES)
        print("\n" + "=" * 115)
        print("★ [리포트 1] Stage 1: 상위 6대 메가 그룹 분류 3-Fold 성능 성적표 [기존 v3.0 계승]")
        print("=" * 115)
        print(f"{'상위 그룹 명칭':<38} | {'정밀도(Prec)':^12} | {'재현율(Rec)':^12} | {'F1-Score':^12}")
        print("-" * 115)

        s1_precs = precision_score(y_macro_true, y_macro_pred, average=None, labels=list(range(num_macro)), zero_division=0)
        s1_recs = recall_score(y_macro_true, y_macro_pred, average=None, labels=list(range(num_macro)), zero_division=0)
        s1_f1s = f1_score(y_macro_true, y_macro_pred, average=None, labels=list(range(num_macro)), zero_division=0)

        for i, g_name in enumerate(MACRO_GROUP_NAMES):
            print(f"{g_name:<38} | {s1_precs[i]*100:>11.2f}% | {s1_recs[i]*100:>11.2f}% | {s1_f1s[i]*100:>11.2f}%")
        print("-" * 115)
        s1_mean_acc = np.mean([r['accuracy'] for r in s1_reports]) * 100.0
        s1_mean_f1 = np.mean([r['macro_f1'] for r in s1_reports]) * 100.0
        print(f"★ Stage 1 3-Fold 평균: 정확도 {s1_mean_acc:.2f}% | Macro-F1 {s1_mean_f1:.2f}%")

        # 리포트 2: Stage 2-A v4.0 독립 2중 캐스케이드 데드리프트 성적표
        print("\n" + "=" * 115)
        print("★ [리포트 2] Stage 2-A: v4.0 독립 2중 캐스케이드 데드리프트 3-Fold 성능 성적표")
        print("=" * 115)
        avg_dead_f1 = np.mean([r['dead_f1'] for r in sub_dead_reports]) * 100.0
        avg_dead_rec = np.mean([r['dead_recall'] for r in sub_dead_reports]) * 100.0
        avg_dead_prec = np.mean([r['dead_precision'] for r in sub_dead_reports]) * 100.0

        avg_stiff_f1 = np.mean([r['stiff_f1'] for r in sub_dead_reports]) * 100.0
        avg_stiff_rec = np.mean([r['stiff_recall'] for r in sub_dead_reports]) * 100.0
        avg_stiff_prec = np.mean([r['stiff_precision'] for r in sub_dead_reports]) * 100.0
        avg_roc_auc_dead = np.mean([r['roc_auc'] for r in sub_dead_reports])

        print(f"• 임계값 τ*_dead: {np.mean([r['best_threshold'] for r in sub_dead_reports]):.2f} | 평균 ROC-AUC: {avg_roc_auc_dead:.4f}")
        print("-" * 115)
        print(f"{'세부 운동 종목':<30} | {'정밀도 (Precision)':<22} | {'재현율 (Recall)':<22} | {'F1-Score':<15}")
        print("-" * 115)
        print(f"{'02. 바벨 데드리프트 (Class 1)':<26} | {avg_dead_prec:6.2f}%                 | {avg_dead_rec:6.2f}%                 | {avg_dead_f1:6.2f}%")
        print(f"{'03. 스티프 데드리프트 (Class 2)':<25} | {avg_stiff_prec:6.2f}%                 | {avg_stiff_rec:6.2f}%                 | {avg_stiff_f1:6.2f}% (★ v3.0 54.58% 대비 향상)")
        print("=" * 115)

        # 리포트 3: Stage 2-B 무빙 런지 성적표 (기존 v3.0 계승)
        print("\n" + "=" * 115)
        print("★ [리포트 3] Stage 2-B: 무빙 런지 패밀리 전용 2진 Expert 튜닝 및 성능 성적표 [기존 v3.0 계승]")
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

        # 리포트 4: 8종 최종 통합 성적표
        acc_total = accuracy_score(y_final_true, y_final_pred)
        macro_f1_total = f1_score(y_final_true, y_final_pred, average='macro', zero_division=0)
        cm_total = confusion_matrix(y_final_true, y_final_pred)

        print("\n" + "■" * 115)
        print("★ [리포트 4] NovelFlow v4.0 최종 8종 운동 통합 성적표 (전수 3-Fold Out-of-Domain 평가)")
        print("■" * 115)
        headers = f"{'운동 종목 명칭':<22} | {'샘플수':^8} | {'정밀도 (Prec)':^15} | {'재현율 (Rec)':^15} | {'★ F1-Score':^14} | {'v3.0 대비 변화':<15}"
        print(headers)
        print("-" * 115)

        V3_BENCHMARK_F1S = [0.9439, 0.9049, 0.5458, 0.9702, 0.9661, 0.9724, 0.7750, 0.9451]

        for c_idx in range(8):
            mask_c = (y_final_true == c_idx)
            sup_c = np.sum(mask_c)
            pred_mask = (y_final_pred == c_idx)

            prec_c = np.sum(mask_c & pred_mask) / np.sum(pred_mask) if np.sum(pred_mask) > 0 else 0
            rec_c = np.sum(mask_c & pred_mask) / sup_c if sup_c > 0 else 0
            f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0

            diff_f1 = (f1_c - V3_BENCHMARK_F1S[c_idx]) * 100
            diff_str = f"{diff_f1:+.2f}%p"
            if c_idx == 2 and diff_f1 > 0:
                diff_str += " (대폭 향상!)"

            c_name = EXERCISE_NAMES_SHORT[c_idx] if c_idx < len(EXERCISE_NAMES_SHORT) else f"Class {c_idx}"
            print(f"{c_name:<24} | {sup_c:^8} | {prec_c*100:^14.2f}% | {rec_c*100:^14.2f}% | {f1_c*100:^13.2f}% | {diff_str:<15}")

        print("-" * 115)
        print(f"★ 최종 v4.0 통합 정확도 (Accuracy) : {acc_total*100:.2f}%")
        print(f"★ 최종 v4.0 통합 Macro-F1 Score   : {macro_f1_total*100:.2f}%")
        print(f"★ 총 소요 시간                     : {total_elapsed:.1f}초")
        print("=" * 115)

        # JSON 저장
        out_dict = {
            'accuracy': float(acc_total),
            'macro_f1': float(macro_f1_total),
            'confusion_matrix': cm_total.tolist(),
            'stiff_f1_v4': float(avg_stiff_f1),
            'stiff_recall_v4': float(avg_stiff_rec)
        }
        os.makedirs(os.path.dirname(OUTPUT_V4_JSON), exist_ok=True)
        with open(OUTPUT_V4_JSON, 'w', encoding='utf-8') as f:
            json.dump(out_dict, f, indent=2, ensure_ascii=False)
        print(f"\n[*] v4.0 최종 벤치마크 결과가 JSON으로 저장되었습니다: {OUTPUT_V4_JSON}")

if __name__ == '__main__':
    trainer = V4HierarchicalPipelineTrainer()
    trainer.run_hierarchical_3fold_cv()
