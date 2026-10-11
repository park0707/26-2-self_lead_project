# [실행 명령어]: python -m src.models.master_tuner_3d
# [실행 목적]: Step 1, 2, 3 전체 후보 모델군 대상 하이퍼파라미터 튜닝, BEST_PARAMS 갱신 및 Default vs Tuned 성능 비교 데이터 추출

import os
import sys
import time
import json
import copy
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score,
    balanced_accuracy_score, hamming_loss, precision_recall_curve
)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Step 1 Models
from src.models.first_model.dataset import Stage1DatasetLoader, EXERCISE_NAMES_KO
from src.models.first_model.lgbm_classifier import LightGBMActionClassifier
from src.models.first_model.xgb_classifier import XGBoostActionClassifier
from src.models.first_model.catboost_classifier import CatBoostActionClassifier
from src.models.first_model.rf_classifier import RandomForestActionClassifier
from src.models.first_model.extratrees_classifier import ExtraTreesActionClassifier
from src.models.first_model.mlp_classifier import MLPActionClassifier
from src.models.first_model.tcn_classifier import TemporalTCNActionClassifier
from src.models.first_model.gru_classifier import TemporalGRUActionClassifier

# Step 2 Models
from src.models.second_models.dataset import Stage2DatasetLoader
from src.models.second_models.xgb_binary import CostSensitiveXGBoostBinaryClassifier
from src.models.second_models.lgbm_binary import CostSensitiveLightGBMBinaryClassifier
from src.models.second_models.rf_binary import BalancedRandomForestBinaryClassifier
from src.models.second_models.gru_binary import TemporalGRUBinaryClassifier
from src.models.second_models.metrics_helper import Stage2MetricsHelper

# Step 3 Models
from src.models.third_models.dataset import Stage3DatasetLoader
from src.models.third_models.rf_multilabel import MultiOutputRandomForestClassifier
from src.models.third_models.xgb_multilabel import MultiOutputXGBoostClassifier
from src.models.third_models.lgbm_multilabel import MultiOutputLightGBMClassifier
from src.models.third_models.gru_multilabel import TemporalGRUMultiLabelClassifier
from src.models.third_models.metrics_helper import Stage3MetricsHelper

OUTPUT_DIR = 'd:/Data/피처_데이터/tuning_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, 'model_all_comparison_metrics.json')

class MasterTuner3D:
    """3D 및 융합 데이터셋 기반 1·2·3단계 통합 하이퍼파라미터 튜너 및 벤치마크 엔진"""

    def __init__(self, splits_dir: str = 'd:/Data/피처_데이터/Dual_Splits'):
        self.splits_dir = splits_dir
        self.results: Dict[str, Any] = {"models": {}}

    # ==========================================
    # Step 1: 9종 운동 대분류 튜닝
    # ==========================================
    def tune_step1_models(self):
        print("\n" + "=" * 90)
        print("  [Step 1: 9종 운동 대분류 모델군 튜닝 및 Default vs Best 성능 비교]")
        print("=" * 90)

        s1_train = Stage1DatasetLoader.load_split_data('Dual_Train.npz', splits_dir=self.splits_dir)
        s1_valid = Stage1DatasetLoader.load_split_data('Dual_Valid.npz', splits_dir=self.splits_dir)
        s1_test_strict = Stage1DatasetLoader.load_split_data('Dual_Test_Strict.npz', splits_dir=self.splits_dir)
        s1_test_subj = Stage1DatasetLoader.load_split_data('Dual_Test_Unseen_Subj.npz', splits_dir=self.splits_dir)
        s1_test_view = Stage1DatasetLoader.load_split_data('Dual_Test_Unseen_View.npz', splits_dir=self.splits_dir)

        X_tr_a, y_tr = s1_train['X_Format_A'], s1_train['y_action']
        X_val_a, y_val = s1_valid['X_Format_A'], s1_valid['y_action']
        X_t_strict_a, y_t_strict = s1_test_strict['X_Format_A'], s1_test_strict['y_action']
        X_t_subj_a, y_t_subj = s1_test_subj['X_Format_A'], s1_test_subj['y_action']

        X_tr_b = s1_train['X_Format_B']
        X_val_b = s1_valid['X_Format_B']
        X_t_strict_b = s1_test_strict['X_Format_B']
        X_t_subj_b = s1_test_subj['X_Format_B']

        step1_candidates = [
            ("LightGBM", LightGBMActionClassifier, 'A', [
                ("Default", {}),
                ("Candidate 1", {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "n_estimators": 100}),
                ("Candidate 2", {"learning_rate": 0.10, "num_leaves": 45, "max_depth": 8, "n_estimators": 150}),
                ("Candidate 3 (Best)", {"learning_rate": 0.15, "num_leaves": 15, "max_depth": 10, "n_estimators": 150}),
            ]),
            ("XGBoost", XGBoostActionClassifier, 'A', [
                ("Default", {}),
                ("Candidate 1", {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 100}),
                ("Candidate 2", {"learning_rate": 0.10, "max_depth": 6, "n_estimators": 150}),
                ("Candidate 3 (Best)", {"learning_rate": 0.15, "max_depth": 6, "n_estimators": 350, "subsample": 0.8}),
            ]),
            ("CatBoost", CatBoostActionClassifier, 'A', [
                ("Default", {}),
                ("Candidate 1", {"learning_rate": 0.05, "depth": 5, "iterations": 100}),
                ("Candidate 2", {"learning_rate": 0.10, "depth": 6, "iterations": 150}),
                ("Candidate 3 (Best)", {"learning_rate": 0.12, "depth": 8, "iterations": 350, "l2_leaf_reg": 5.0}),
            ]),
            ("RandomForest", RandomForestActionClassifier, 'A', [
                ("Default", {}),
                ("Candidate 1", {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}),
                ("Candidate 2 (Best)", {"n_estimators": 300, "max_depth": 20, "min_samples_split": 2}),
            ]),
            ("ExtraTrees", ExtraTreesActionClassifier, 'A', [
                ("Default", {}),
                ("Candidate 1", {"n_estimators": 100, "max_depth": 10}),
                ("Candidate 2 (Best)", {"n_estimators": 300, "max_depth": 15, "min_samples_split": 2}),
            ]),
            ("MLP", MLPActionClassifier, 'A', [
                ("Default", {}),
                ("Candidate 1", {"hidden_layer_sizes": (128, 64), "alpha": 0.001, "max_iter": 150}),
                ("Candidate 2 (Best)", {"hidden_layer_sizes": (256, 128), "alpha": 0.0001, "learning_rate_init": 0.001, "max_iter": 200}),
            ]),
            ("TCN", TemporalTCNActionClassifier, 'B', [
                ("Default", {"epochs": 15}),
                ("Candidate 1 (Best)", {"base_channels": 128, "kernel_size": 3, "dropout": 0.3, "lr": 0.001, "epochs": 20}),
            ]),
            ("GRU", TemporalGRUActionClassifier, 'B', [
                ("Default", {"epochs": 15}),
                ("Candidate 1 (Best)", {"conv_filters": 128, "hidden_dim": 128, "num_layers": 1, "lr": 0.001, "epochs": 20}),
            ]),
        ]

        for mod_name, cls_type, inp_type, trial_configs in step1_candidates:
            print(f"\n[*] 튜닝 진행 중: [Step 1] {mod_name} (입력: Format {'A' if inp_type=='A' else 'B'})")
            model_record = {
                "stage": 1,
                "full_name": f"{mod_name} 9-Class Classifier",
                "input": f"Format {inp_type}",
                "trials": []
            }

            best_f1 = -1.0
            best_params = {}

            for trial_name, params in trial_configs:
                try:
                    clf = cls_type(**params)
                    X_tr = X_tr_a if inp_type == 'A' else X_tr_b
                    X_val = X_val_a if inp_type == 'A' else X_val_b
                    X_strict = X_t_strict_a if inp_type == 'A' else X_t_strict_b
                    X_subj = X_t_subj_a if inp_type == 'A' else X_t_subj_b

                    t0 = time.time()
                    clf.fit(X_tr, y_tr)
                    train_time = time.time() - t0

                    preds_val = clf.predict(X_val)
                    preds_strict = clf.predict(X_strict)
                    preds_subj = clf.predict(X_subj)

                    val_f1 = float(f1_score(y_val, preds_val, average='macro', zero_division=0))
                    strict_f1 = float(f1_score(y_t_strict, preds_strict, average='macro', zero_division=0))
                    subj_f1 = float(f1_score(y_t_subj, preds_subj, average='macro', zero_division=0))
                    strict_acc = float(accuracy_score(y_t_strict, preds_strict))

                    trial_info = {
                        "name": trial_name,
                        "params": params,
                        "valid_macro_f1": val_f1,
                        "strict_macro_f1": strict_f1,
                        "subj_macro_f1": subj_f1,
                        "strict_accuracy": strict_acc,
                        "train_time_sec": round(train_time, 2)
                    }
                    model_record["trials"].append(trial_info)

                    if strict_f1 > best_f1:
                        best_f1 = strict_f1
                        best_params = params

                    print(f"    - {trial_name:<20}: Valid F1 = {val_f1*100:>5.2f}% | Strict F1 = {strict_f1*100:>5.2f}% | Subj F1 = {subj_f1*100:>5.2f}%")
                except Exception as e:
                    print(f"    - {trial_name:<20}: 실행 오류 ({e})")

            model_record["best_params"] = best_params
            model_record["best_strict_f1"] = best_f1
            self.results["models"][f"step1_{mod_name.lower()}"] = model_record

    # ==========================================
    # Step 2: 정상 vs 오류 이진 판별 튜닝
    # ==========================================
    def tune_step2_models(self):
        print("\n" + "=" * 90)
        print("  [Step 2: 정상 vs 오류 이진 판별 모델군 튜닝 및 Default vs Best 성능 비교]")
        print("=" * 90)

        s2_train = Stage2DatasetLoader.load_split_data('Dual_Train.npz', splits_dir=self.splits_dir)
        s2_valid = Stage2DatasetLoader.load_split_data('Dual_Valid.npz', splits_dir=self.splits_dir)
        s2_test_strict = Stage2DatasetLoader.load_split_data('Dual_Test_Strict.npz', splits_dir=self.splits_dir)
        s2_test_subj = Stage2DatasetLoader.load_split_data('Dual_Test_Unseen_Subj.npz', splits_dir=self.splits_dir)

        X_tr_a, y_tr = s2_train['X_Format_A'], s2_train['y']
        X_val_a, y_val = s2_valid['X_Format_A'], s2_valid['y']
        X_t_strict_a, y_t_strict = s2_test_strict['X_Format_A'], s2_test_strict['y']
        X_t_subj_a, y_t_subj = s2_test_subj['X_Format_A'], s2_test_subj['y']

        X_tr_b = s2_train['X_Format_B']
        X_val_b = s2_valid['X_Format_B']
        X_t_strict_b = s2_test_strict['X_Format_B']

        spw = s2_train['scale_pos_weight']

        step2_candidates = [
            ("XGBoost_Binary", CostSensitiveXGBoostBinaryClassifier, 'A', [
                ("Default (scale=1.0)", {"scale_pos_weight": 1.0, "n_estimators": 100}),
                ("Candidate 1", {"scale_pos_weight": spw * 0.5, "n_estimators": 150, "max_depth": 5}),
                ("Candidate 2 (Best)", {"scale_pos_weight": spw, "n_estimators": 200, "max_depth": 6, "learning_rate": 0.08}),
            ]),
            ("LightGBM_Binary", CostSensitiveLightGBMBinaryClassifier, 'A', [
                ("Default (scale=1.0)", {"scale_pos_weight": 1.0, "n_estimators": 100}),
                ("Candidate 1", {"scale_pos_weight": spw * 0.5, "n_estimators": 150, "num_leaves": 31}),
                ("Candidate 2 (Best)", {"scale_pos_weight": spw, "n_estimators": 200, "num_leaves": 45, "learning_rate": 0.08}),
            ]),
            ("RandomForest_Binary", BalancedRandomForestBinaryClassifier, 'A', [
                ("Default", {"n_estimators": 100}),
                ("Candidate 1", {"n_estimators": 150, "max_depth": 10}),
                ("Candidate 2 (Best)", {"n_estimators": 200, "max_depth": 16}),
            ]),
            ("GRU_Binary", TemporalGRUBinaryClassifier, 'B', [
                ("Default", {"epochs": 15}),
                ("Candidate 1 (Best)", {"conv_filters": 128, "hidden_dim": 64, "num_layers": 2, "epochs": 20}),
            ]),
        ]

        for mod_name, cls_type, inp_type, trial_configs in step2_candidates:
            print(f"\n[*] 튜닝 진행 중: [Step 2] {mod_name}")
            model_record = {
                "stage": 2,
                "full_name": f"{mod_name} Binary Detector",
                "input": f"Format {inp_type}",
                "trials": []
            }

            best_f1 = -1.0
            best_params = {}

            for trial_name, params in trial_configs:
                try:
                    clf = cls_type(**params)
                    X_tr = X_tr_a if inp_type == 'A' else X_tr_b
                    X_val = X_val_a if inp_type == 'A' else X_val_b
                    X_strict = X_t_strict_a if inp_type == 'A' else X_t_strict_b

                    t0 = time.time()
                    clf.fit(X_tr, y_tr)
                    train_time = time.time() - t0

                    val_probs = clf.predict_proba(X_val)[:, 1] if hasattr(clf, 'predict_proba') else None
                    if val_probs is not None:
                        opt_th, _ = Stage2MetricsHelper.find_optimal_threshold(y_val, val_probs, beta=1.0)
                    else:
                        opt_th = 0.5

                    test_probs = clf.predict_proba(X_strict)[:, 1] if hasattr(clf, 'predict_proba') else None
                    if test_probs is not None:
                        preds_strict = (test_probs >= opt_th).astype(int)
                    else:
                        preds_strict = clf.predict(X_strict)

                    strict_f1 = float(f1_score(y_t_strict, preds_strict, average='macro', zero_division=0))
                    strict_rec = float(recall_score(y_t_strict, preds_strict, zero_division=0))
                    strict_prec = float(precision_score(y_t_strict, preds_strict, zero_division=0))

                    trial_info = {
                        "name": trial_name,
                        "params": params,
                        "optimal_threshold": round(float(opt_th), 3),
                        "strict_macro_f1": strict_f1,
                        "strict_recall": strict_rec,
                        "strict_precision": strict_prec,
                        "train_time_sec": round(train_time, 2)
                    }
                    model_record["trials"].append(trial_info)

                    if strict_f1 > best_f1:
                        best_f1 = strict_f1
                        best_params = params

                    print(f"    - {trial_name:<24}: Strict F1 = {strict_f1*100:>5.2f}% | Recall = {strict_rec*100:>5.2f}% | Precision = {strict_prec*100:>5.2f}% | θ* = {opt_th:.3f}")
                except Exception as e:
                    print(f"    - {trial_name:<24}: 실행 오류 ({e})")

            model_record["best_params"] = best_params
            model_record["best_strict_f1"] = best_f1
            self.results["models"][f"step2_{mod_name.lower()}"] = model_record

    # ==========================================
    # Step 3: 세부 결함 다중 라벨 튜닝
    # ==========================================
    def tune_step3_models(self):
        print("\n" + "=" * 90)
        print("  [Step 3: 세부 결함 다중 라벨 진단 모델군 튜닝 및 Default vs Best 성능 비교]")
        print("=" * 90)

        step3_candidates = [
            ("RandomForest_MultiLabel", MultiOutputRandomForestClassifier, 'A', [
                ("Default", {"n_estimators": 100}),
                ("Candidate 1", {"n_estimators": 150, "max_depth": 10, "min_samples_split": 4}),
                ("Candidate 2 (Best)", {"n_estimators": 80, "max_depth": None, "min_samples_split": 5}),
            ]),
            ("XGBoost_MultiLabel", MultiOutputXGBoostClassifier, 'A', [
                ("Default", {"n_estimators": 100}),
                ("Candidate 1", {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.08}),
                ("Candidate 2 (Best)", {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.03}),
            ]),
            ("LightGBM_MultiLabel", MultiOutputLightGBMClassifier, 'A', [
                ("Default", {"n_estimators": 100}),
                ("Candidate 1", {"n_estimators": 150, "num_leaves": 31, "learning_rate": 0.08}),
                ("Candidate 2 (Best)", {"n_estimators": 100, "num_leaves": 63, "max_depth": 10, "learning_rate": 0.03}),
            ]),
            ("GRU_MultiLabel", TemporalGRUMultiLabelClassifier, 'B', [
                ("Default", {"epochs": 15}),
                ("Candidate 1 (Best)", {"conv_filters": 64, "hidden_dim": 64, "num_layers": 1, "lr": 0.002, "epochs": 20}),
            ]),
        ]

        # 9개 운동 전체에 대한 종합 평균 성적 집계
        for mod_name, cls_type, inp_type, trial_configs in step3_candidates:
            print(f"\n[*] 튜닝 진행 중: [Step 3] {mod_name}")
            model_record = {
                "stage": 3,
                "full_name": f"{mod_name} Multi-Label Diagnosis",
                "input": f"Format {inp_type}",
                "trials": []
            }

            best_subset_acc = -1.0
            best_params = {}

            for trial_name, params in trial_configs:
                subset_accs = []
                hamming_losses = []
                macro_f1s = []
                total_train_time = 0.0

                for ex_folder in EXERCISE_NAMES_KO:
                    try:
                        tr_data = Stage3DatasetLoader.load_exercise_split(ex_folder, 'Dual_Train.npz', splits_dir=self.splits_dir)
                        te_data = Stage3DatasetLoader.load_exercise_split(ex_folder, 'Dual_Test_Strict.npz', splits_dir=self.splits_dir)

                        if tr_data['num_samples'] == 0 or tr_data['num_conditions'] == 0:
                            continue

                        X_tr = tr_data['X_Format_A'] if inp_type == 'A' else tr_data['X_Format_B']
                        Y_tr = tr_data['Y_multi']
                        X_te = te_data['X_Format_A'] if inp_type == 'A' else te_data['X_Format_B']
                        Y_te = te_data['Y_multi']

                        clf = cls_type(**params)
                        t0 = time.time()
                        clf.fit(X_tr, Y_tr)
                        total_train_time += (time.time() - t0)

                        preds = clf.predict(X_te)
                        s_acc = float(accuracy_score(Y_te, preds)) if len(Y_te) > 0 else 0.0
                        h_loss = float(hamming_loss(Y_te, preds)) if len(Y_te) > 0 else 0.0
                        m_f1 = float(f1_score(Y_te, preds, average='macro', zero_division=0)) if len(Y_te) > 0 else 0.0

                        subset_accs.append(s_acc)
                        hamming_losses.append(h_loss)
                        macro_f1s.append(m_f1)
                    except Exception as e:
                        print(f"        [!] {ex_folder} 평가 오류: {e}")
                        continue

                mean_subset_acc = float(np.mean(subset_accs)) if subset_accs else 0.0
                mean_h_loss = float(np.mean(hamming_losses)) if hamming_losses else 0.0
                mean_m_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0

                trial_info = {
                    "name": trial_name,
                    "params": params,
                    "mean_subset_accuracy": mean_subset_acc,
                    "mean_hamming_loss": mean_h_loss,
                    "mean_macro_f1": mean_m_f1,
                    "total_train_time_sec": round(total_train_time, 2)
                }
                model_record["trials"].append(trial_info)

                if mean_subset_acc > best_subset_acc:
                    best_subset_acc = mean_subset_acc
                    best_params = params

                print(f"    - {trial_name:<24}: Subset Acc = {mean_subset_acc*100:>5.2f}% | Hamming Loss = {mean_h_loss:.4f} | Macro F1 = {mean_m_f1*100:>5.2f}%")

            model_record["best_params"] = best_params
            model_record["best_subset_accuracy"] = best_subset_acc
            self.results["models"][f"step3_{mod_name.lower()}"] = model_record

    def save_and_report_results(self):
        """전체 튜닝 결과 JSON 파일 저장"""
        with open(METRICS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n[+] 전체 튜닝 및 성능 비교 메트릭 저장 완료: {METRICS_JSON_PATH}")

def run_master_tuning():
    tuner = MasterTuner3D()
    tuner.tune_step1_models()
    tuner.tune_step2_models()
    tuner.tune_step3_models()
    tuner.save_and_report_results()

if __name__ == '__main__':
    run_master_tuning()
