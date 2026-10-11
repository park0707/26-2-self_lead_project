# =========================================================================================================
# [스크립트명]: train_v6_hierarchical_model.py
# [위치]: src/models/first_model/train_v6_hierarchical_model.py
# [프로젝트]: vidio-health-trainer
# [목적]: 1단계 8종 운동 분류 - v6.0 Model F (59D Direct KCF: 국소 분절 정규화 + 로컬 키네마틱 프레임) 기반
#         2-Stage 계층형 하이브리드(Bi-GRU + LightGBM + XGBoost) 앙상블 학습 및 v5.0 대비 3-Fold 전수 검증기
# [혁신점]:
#   - Stage 1 상위 대그룹 앙상블의 GBDT 트리에 기존 Format B 경유 55D KCF 대신
#     3D 관절 직결 59D Model F(계층적 국소 분절 정규화 + 로컬 키네마틱 프레임) 전면 투입
#   - Stage 2-A (데드리프트 2중 캐스케이드 18D) + Stage 2-B (무빙 런지 Expert 15D) 결합
#   - v5.0 대비 8종 운동별 Precision / Recall / F1 및 종합 Top-1 정확도, Macro F1 전수 비교
# =========================================================================================================

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import xgboost as xgb
import lightgbm as lgb
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
from src.models.first_model.dataset import Stage1DatasetLoader, EXERCISE_NAMES_KO
from src.models.first_model.metrics_helper import Stage1MetricsHelper, EXERCISE_NAMES_SHORT
from src.models.first_model.train_augmented_ensemble import AdvancedGRUClassifier
from src.features.kinematics_v6 import KinematicsEngineV6
from src.models.sub_models.dual_cascade_deadlift_classifier import DualCascadeDeadliftClassifier
from src.models.sub_models.lunge_sub_classifier import LungeSubClassifier

OUTPUT_V5_JSON = 'd:/Data/피처_데이터/tuning_results/stage1_hierarchical_v5_results.json'
OUTPUT_V6_JSON = 'd:/Data/피처_데이터/tuning_results/stage1_hierarchical_v6_results.json'

class V6HierarchicalPipelineTrainer(HierarchicalKFoldPipelineTrainer):
    """v6.0 Model F 59D 기반 2-Stage 계층형 하이브리드 파이프라인 트레이너"""

    def _load_master_pool(self) -> Dict[str, Any]:
        """마스터 풀 로드 및 v6.0 Model F 59D Direct KCF 피처 추출 및 결합"""
        all_X_b, all_y_act, all_subjs, all_views, all_sids = [], [], [], [], []
        all_X_b_lunge = []

        print("\n" + "=" * 115)
        print(f"[*] [vidio-health-trainer 1단계 v6.0] 마스터 풀 로드 중 (디렉토리: {self.data_dir})")
        print("=" * 115)

        for fn in sorted(os.listdir(self.data_dir)):
            if not fn.endswith('.npz'):
                continue
            split_data = Stage1DatasetLoader.load_split_data(fn, splits_dir=self.data_dir)
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
        raw_X_b = np.concatenate(all_X_b, axis=0)
        raw_X_b_lunge = np.concatenate(all_X_b_lunge, axis=0) if all_X_b_lunge else raw_X_b
        y_act = np.concatenate(all_y_act, axis=0)

        # -------------------------------------------------------------
        # ★ [v6.0 핵심 혁신]: 3D 관절 좌표 복원 -> 계층적 국소 분절 정규화 -> Model F 59D Direct KCF 추출
        # -------------------------------------------------------------
        print(f"[*] 총 {len(y_act):,}개 시퀀스로부터 v6.0 Model F 59D Direct KCF (국소 분절 정규화 + 로컬 프레임) 동적 추출 중...")
        t0 = time.time()
        kps_3d = KinematicsEngineV6.reconstruct_3d_keypoints_from_raw_b(raw_X_b)
        X_kcf_59d = KinematicsEngineV6.build_model_f_direct_kcf(kps_3d)
        print(f"    ✔ Model F 59D 피처 추출 완료: Shape {X_kcf_59d.shape} (소요 시간: {time.time()-t0:.2f}초)")

        y_macro = np.array([CLASS_TO_MACRO_GROUP[c] for c in y_act], dtype=np.int64)

        return {
            'X_Format_A': X_kcf_59d, # v6에서는 Format A 자리에 Model F 59D 피처를 투입
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
        """v6.0 Model F 기반 2-Stage 계층형 앙상블 3-Fold 교차 검증 전수 실행"""
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
        print("★ [vidio-health-trainer v6.0 2-Stage 계층형 앙상블]: 6대 상위 그룹(Model F 59D) + 데드 2중 캐스케이드 + 런지 Expert")
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

            X_tr_kcf, X_tr_b, y_tr_act, y_tr_mac = (
                self.master_data['X_Format_A'][train_mask],
                self.master_data['X_Format_B'][train_mask],
                self.master_data['y_action'][train_mask],
                self.master_data['y_macro'][train_mask]
            )
            X_te_kcf, X_te_b, y_te_act, y_te_mac = (
                self.master_data['X_Format_A'][test_mask],
                self.master_data['X_Format_B'][test_mask],
                self.master_data['y_action'][test_mask],
                self.master_data['y_macro'][test_mask]
            )
            X_tr_b_lunge = self.master_data['X_Format_B_lunge'][train_mask]
            X_te_b_lunge = self.master_data['X_Format_B_lunge'][test_mask]

            train_subjs = self.master_data['subjects'][train_mask]
            test_subjs = self.master_data['subjects'][test_mask]
            train_sids = [str(s) for s in np.array(self.master_data['sequence_ids'])[train_mask]]
            test_sids = [str(s) for s in np.array(self.master_data['sequence_ids'])[test_mask]]

            # -------------------------------------------------------------
            # [Stage 1] 6-Group 상위 그룹 3중 앙상블 학습 (Bi-GRU + LightGBM + XGBoost on Model F 59D)
            # -------------------------------------------------------------
            t_s1 = time.time()
            num_macro_classes = self.master_data['num_classes_macro']
            class_counts_mac = np.bincount(y_tr_mac, minlength=num_macro_classes)
            raw_w_mac = len(y_tr_mac) / (float(num_macro_classes) * np.maximum(class_counts_mac, 1).astype(np.float32))
            smooth_w_mac = np.sqrt(raw_w_mac)
            sample_w_mac = smooth_w_mac[y_tr_mac]

            print(f"    [1/3] Stage 1 (6-Group 마스터 앙상블) 훈련 중 (Bi-GRU + LightGBM + XGBoost on Model F 59D)...", flush=True)
            gru_macro = AdvancedGRUClassifier(
                input_dim=X_tr_b.shape[2],
                num_classes=num_macro_classes,
                epochs=12,
                lr=0.003,
                batch_size=512
            )
            gru_macro.fit(X_tr_b, y_tr_mac, class_weights=smooth_w_mac)

            lgb_macro = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=4, verbose=-1
            )
            lgb_macro.fit(X_tr_kcf, y_tr_mac, sample_weight=sample_w_mac)

            xgb_macro = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.85, colsample_bytree=0.85, random_state=42,
                eval_metric='mlogloss', n_jobs=4
            )
            xgb_macro.fit(X_tr_kcf, y_tr_mac, sample_weight=sample_w_mac)

            # Stage 1 사후 확률 추론
            P_gru_te = gru_macro.predict_proba(X_te_b)
            P_lgb_te = lgb_macro.predict_proba(X_te_kcf)
            P_xgb_te = xgb_macro.predict_proba(X_te_kcf)

            P_macro_te = 0.50 * P_gru_te + 0.30 * P_lgb_te + 0.20 * P_xgb_te
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
            # [Stage 2-A] 데드리프트 독립 2중 캐스케이드 Expert (18D 발 관절 파생 피처 결합)
            # -------------------------------------------------------------
            print(f"    [2/3] Stage 2-A (18D 발 관절 결합 캐스케이드) 훈련 및 평가 중...", flush=True)
            dead_tr_mask = (y_tr_act == 1) | (y_tr_act == 2)
            dead_te_mask = (y_te_act == 1) | (y_te_act == 2)

            X_dead_tr_raw = X_tr_b[dead_tr_mask]
            y_dead_tr_bin = (y_tr_act[dead_tr_mask] == 2).astype(int)
            dead_subjs_tr = train_subjs[dead_tr_mask]
            dead_sids_tr = [train_sids[i] for i in range(len(train_sids)) if dead_tr_mask[i]]

            X_dead_te_raw_full = X_te_b
            dead_sids_te_full = test_sids

            te_dead_feats = DualCascadeDeadliftClassifier.slice_features(X_te_b[dead_te_mask])
            te_dead_levels = DualCascadeDeadliftClassifier.determine_levels(te_dead_feats, (y_te_act[dead_te_mask] == 2).astype(int))
            valid_dead_eval_mask = (te_dead_levels <= 3)

            X_dead_te_raw_eval = X_te_b[dead_te_mask][valid_dead_eval_mask]
            y_dead_te_bin_eval = (y_te_act[dead_te_mask] == 2).astype(int)[valid_dead_eval_mask]
            dead_sids_te_eval = [str(test_sids[i]) for i in range(len(test_sids)) if dead_te_mask[i]]
            dead_sids_te_eval = [dead_sids_te_eval[i] for i in range(len(dead_sids_te_eval)) if valid_dead_eval_mask[i]]

            sub_dead_clf = DualCascadeDeadliftClassifier()
            sub_dead_clf.fit(X_dead_tr_raw, y_dead_tr_bin, subjects=dead_subjs_tr, sequence_ids=dead_sids_tr)
            dead_eval_res = sub_dead_clf.evaluate(X_dead_te_raw_eval, y_dead_te_bin_eval, sequence_ids=dead_sids_te_eval)
            dead_eval_res['fold_name'] = fc['name']
            sub_dead_fold_reports.append(dead_eval_res)

            rt_st = dead_eval_res['routing_stats']
            print(f"      ✔ Stage 2-A (18D Level 1~3) 완료 -> 스티프데드 Recall: {dead_eval_res['stiff_recall']*100:.2f}%, F1: {dead_eval_res['stiff_f1']*100:.2f}% (바벨직행: {rt_st['barbell_fast']}, 스티프직행: {rt_st['stiff_fast']}, 2차위임: {rt_st['routed_to_stage2']})")

            # -------------------------------------------------------------
            # [Stage 2-B] 무빙 런지 패밀리 전용 2진 Expert (15D 최하사점 위상인식/프록시 피처)
            # -------------------------------------------------------------
            print(f"    [3/3] Stage 2-B (무빙 런지 2진 Expert) 15D 피처 훈련 중...", flush=True)
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
            # [통합 추론] 소프트 베이지안 확률 결합
            # -------------------------------------------------------------
            t_inf = time.time()
            P_dead_stiff_probs = sub_dead_clf.predict_proba(X_dead_te_raw_full, sequence_ids=dead_sids_te_full)[:, 1]
            P_lunge_cross_probs = sub_lunge_clf.predict_proba(X_lunge_te_sub_full)[:, 1]

            P_final_te = np.zeros((len(y_te_act), 8), dtype=np.float32)

            P_final_te[:, 0] = P_macro_te[:, 0] # 01. 바벨 스쿼트
            P_final_te[:, 3] = P_macro_te[:, 2] # 04. 바벨 런지 (단독)
            P_final_te[:, 5] = P_macro_te[:, 4] # 06. 사이드 런지
            P_final_te[:, 7] = P_macro_te[:, 5] # 08. 굿모닝

            P_dead_family = P_macro_te[:, 1]
            P_final_te[:, 1] = P_dead_family * (1.0 - P_dead_stiff_probs) # 02. 일반 데드
            P_final_te[:, 2] = P_dead_family * P_dead_stiff_probs         # 03. 스티프 데드

            P_lunge_family = P_macro_te[:, 3]
            P_final_te[:, 4] = P_lunge_family * (1.0 - P_lunge_cross_probs) # 05. 다이나믹 런지
            P_final_te[:, 6] = P_lunge_family * P_lunge_cross_probs         # 07. 크로스 런지

            macro_top1 = np.argmax(P_macro_te, axis=-1)
            final_preds = np.argmax(P_final_te, axis=-1)

            lunge_mask_te = (macro_top1 == 3)
            final_preds[lunge_mask_te] = np.where(P_lunge_cross_probs[lunge_mask_te] >= lunge_tune_res['best_threshold'], 6, 4)

            dead_mask_te = (macro_top1 == 1)
            final_preds[dead_mask_te] = np.where(P_dead_stiff_probs[dead_mask_te] >= sub_dead_clf.stage2_decision_threshold, 2, 1)

            inf_time_ms = ((time.time() - t_inf) / max(1, len(y_te_act))) * 1000.0

            te_dead_all_mask = (y_te_act == 1) | (y_te_act == 2)
            eval_valid_mask = np.ones(len(y_te_act), dtype=bool)
            if np.sum(te_dead_all_mask) > 0:
                te_feats_14d = DualCascadeDeadliftClassifier.slice_features(X_te_b[te_dead_all_mask])
                te_dead_y = (y_te_act[te_dead_all_mask] == 2).astype(int)
                te_levels = DualCascadeDeadliftClassifier.determine_levels(te_feats_14d, te_dead_y)
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

        # 종합 결과 집계
        all_final_trues_np = np.array(all_final_trues)
        all_final_preds_np = np.array(all_final_preds)
        overall_acc = accuracy_score(all_final_trues_np, all_final_preds_np)
        overall_f1 = f1_score(all_final_trues_np, all_final_preds_np, average='macro', zero_division=0)
        overall_cm = confusion_matrix(all_final_trues_np, all_final_preds_np, labels=list(range(8))).tolist()

        overall_metrics = Stage1MetricsHelper.evaluate_predictions(
            y_true=all_final_trues_np,
            y_pred=all_final_preds_np,
            inference_time_ms=0.5
        )

        os.makedirs(os.path.dirname(OUTPUT_V6_JSON), exist_ok=True)
        v6_summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': 'v6.0 Model F Direct KCF Hybrid (Bi-GRU + LightGBM + XGBoost)',
            'overall_accuracy': float(overall_acc),
            'overall_macro_f1': float(overall_f1),
            'overall_confusion_matrix': overall_cm,
            'overall_class_f1s': overall_metrics['class_f1s'],
            'overall_class_precs': overall_metrics['class_precs'],
            'overall_class_recs': overall_metrics['class_recs'],
            'folds_stage1': stage1_fold_reports,
            'folds_final': final_fold_reports,
            'folds_deadlift': sub_dead_fold_reports,
            'folds_lunge': sub_lunge_fold_reports,
            'total_time_sec': t_global_total
        }

        with open(OUTPUT_V6_JSON, 'w', encoding='utf-8') as f:
            json.dump(_to_serializable(v6_summary), f, indent=2, ensure_ascii=False)

        # -------------------------------------------------------------
        # ★ [결과 리포팅]: Stage1MetricsHelper 표준 세부 성적표 출력
        # -------------------------------------------------------------
        print("\n" + "=" * 115)
        print("★ [vidio-health-trainer v6.0 2-Stage 계층형 앙상블 3-Fold 최종 집계 완료]")
        print("=" * 115)

        Stage1MetricsHelper.print_class_breakdown_report(
            overall_metrics,
            all_final_trues_np,
            split_title="v6.0 Model F 2-Stage 3-Fold Out-of-Domain 전수"
        )

        # v5.0 vs v6.0 비교 출력
        if os.path.exists(OUTPUT_V5_JSON):
            with open(OUTPUT_V5_JSON, 'r', encoding='utf-8') as f:
                v5_data = json.load(f)

            print("\n" + "=" * 115)
            print("★ [종합 비교] v5.0 vs v6.0 2-Stage 계층형 파이프라인 3-Fold 실측 비교표")
            print("=" * 115)
            header = f"{'평가 지표 및 클래스':<24} | {'v5.0 (레거시 KCF)':^20} | {'★ v6.0 (Model F 59D)':^22} | {'성과 (증감폭)':^18}"
            print(header)
            print("-" * 115)

            v5_acc = v5_data['overall_accuracy'] * 100.0
            v6_acc = overall_acc * 100.0
            d_acc = v6_acc - v5_acc
            print(f"{'종합 Top-1 정확도 (Acc)':<24} | {v5_acc:>19.2f}% | {v6_acc:>21.2f}% | {d_acc:>+17.2f}%p")

            v5_mf1 = v5_data['overall_macro_f1'] * 100.0
            v6_mf1 = overall_f1 * 100.0
            d_mf1 = v6_mf1 - v5_mf1
            print(f"{'종합 Macro F1-Score':<24} | {v5_mf1:>19.2f}% | {v6_mf1:>21.2f}% | {d_mf1:>+17.2f}%p")
            print("-" * 115)

            # 8개 종목별 F1 비교
            v5_cm = np.array(v5_data['overall_confusion_matrix'])
            v5_tp = np.diag(v5_cm)
            v5_pred_sum = np.sum(v5_cm, axis=0)
            v5_true_sum = np.sum(v5_cm, axis=1)
            v5_precs = np.divide(v5_tp, np.maximum(v5_pred_sum, 1e-6))
            v5_recs = np.divide(v5_tp, np.maximum(v5_true_sum, 1e-6))
            v5_f1s = 2 * (v5_precs * v5_recs) / np.maximum(v5_precs + v5_recs, 1e-6)

            for i, name in enumerate(EXERCISE_NAMES_SHORT):
                f5 = v5_f1s[i] * 100.0
                f6 = overall_metrics['class_f1s'][i] * 100.0
                df = f6 - f5
                star = " ★" if i in [2, 6] else "  "
                print(f"{star + name:<24} | {f5:>19.2f}% | {f6:>21.2f}% | {df:>+17.2f}%p")

            print("=" * 115)

        print(f"\n[+] v6.0 상세 결과 JSON 저장 완료: {OUTPUT_V6_JSON}")
        return v6_summary

def main():
    trainer = V6HierarchicalPipelineTrainer()
    trainer.run_hierarchical_3fold_cv()

if __name__ == '__main__':
    main()
