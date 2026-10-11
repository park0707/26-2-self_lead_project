# =========================================================================================================
# [스크립트명]: general_test_v5.py
# [위치]: src/models/first_model/general_test_v5.py
# [목적]: AI Hub 전수(20,003개 시퀀스) 학습 기반 1단계 v5.0 운동 경향성(KCF) 하이브리드 모델 피팅 및
#         독립 외부 벤치마크(RepCount, UCF101, Kaggle Workout) 제로샷 일반화 성능 검증기
# =========================================================================================================

import os
import sys
import time
import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 한글 윈도우 콘솔 UTF-8 출력 보정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 프로젝트 루트 경로 등록
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.models.first_model.dataset import Stage1DatasetLoader, EXERCISE_NAMES_KO
from src.models.first_model.train_hierarchical_kfold import CLASS_TO_MACRO_GROUP, MACRO_GROUP_NAMES
from src.models.first_model.train_augmented_ensemble import AdvancedGRUClassifier
from src.features.kinematic_coordination import KinematicCoordinationExtractor
from src.models.sub_models.dual_cascade_deadlift_classifier import DualCascadeDeadliftClassifier
from src.models.sub_models.lunge_sub_classifier import LungeSubClassifier

# 데이터셋 경로
AIHUB_MASTER_DIR = 'd:/Data/피처_데이터/Condition4_NovelFlow_3D' if os.path.exists('d:/Data/피처_데이터/Condition4_NovelFlow_3D/Dual_Train.npz') else 'd:/Data/피처_데이터/Condition3_MediaPipe_3D'
BENCHMARK_DIR = 'd:/Data/피처_데이터/External_Benchmarks_3D'
OUTPUT_JSON_PATH = 'd:/Data/피처_데이터/tuning_results/generalization_v5_results.json'

EXERCISE_SHORT_NAMES = [
    '01.스쿼트', '02.데드리프트', '03.스티프데드', '04.바벨런지',
    '05.다이나믹런지', '06.사이드런지', '07.크로스런지', '08.굿모닝'
]

class GeneralizationV5Tester:
    """AI Hub 전수 학습 기반 v5.0 운동 경향성(KCF) 모델 및 외부 벤치마크 일반화 검증 엔진"""

    def __init__(self):
        self.aihub_data = None
        self.gru_macro = None
        self.lgb_macro = None
        self.xgb_macro = None
        self.sub_dead_clf = None
        self.sub_lunge_clf = None

    def load_aihub_master_pool(self) -> Dict[str, Any]:
        """AI Hub 전수 데이터 로드 및 55D 순수 KCF 피처 추출"""
        print("\n" + "=" * 105)
        print(f" [1단계] AI Hub 전수 학습 데이터 로드 및 55D KCF 추출 (경로: {AIHUB_MASTER_DIR})")
        print("=" * 105)

        all_X_b, all_y_act, all_subjs = [], [], []
        npz_files = sorted([f for f in os.listdir(AIHUB_MASTER_DIR) if f.endswith('.npz')])

        for fn in npz_files:
            split = Stage1DatasetLoader.load_split_data(fn, splits_dir=AIHUB_MASTER_DIR)
            all_X_b.append(split['X_Format_B'])
            all_y_act.append(split['y_action'])
            all_subjs.extend(split['subjects'])

        raw_X_b = np.concatenate(all_X_b, axis=0)
        y_act = np.concatenate(all_y_act, axis=0)

        def get_subj_num(s):
            s = str(s)
            for digit in ['1','2','3','4','5','6','7','8','9']:
                if f"피험자{digit}" in s or s == digit or f"_{digit}_" in s:
                    return int(digit)
            return 1

        subjs_arr = np.array([get_subj_num(s) for s in all_subjs])

        print(f"  [*] 마스터 시퀀스 총량: {len(y_act):,}개 로드 완료")
        print(f"  [*] Format B ({raw_X_b.shape[1]}F, {raw_X_b.shape[2]}D)로부터 55D 운동 경향성(KCF) 동적 추출 중...")
        t0 = time.time()
        X_kcf = KinematicCoordinationExtractor.extract_features(raw_X_b)
        print(f"  [OK] 55D KCF 피처 추출 완료: Shape {X_kcf.shape} (소요: {time.time()-t0:.2f}초)")

        y_macro = np.array([CLASS_TO_MACRO_GROUP[c] for c in y_act], dtype=np.int64)

        return {
            'X_kcf': X_kcf,
            'X_Format_B': raw_X_b,
            'y_action': y_act,
            'y_macro': y_macro,
            'subjects': subjs_arr
        }

    def train_full_v5_pipeline(self):
        """AI Hub 전수 데이터로 v5.0 파이프라인 학습"""
        print("\n" + "=" * 105)
        print(" [2단계] v5.0 KCF 하이브리드 파이프라인 전수 훈련 시작")
        print("=" * 105)
        t_tr_start = time.time()

        X_kcf = self.aihub_data['X_kcf']
        X_b = self.aihub_data['X_Format_B']
        y_act = self.aihub_data['y_action']
        y_mac = self.aihub_data['y_macro']
        subjs = self.aihub_data['subjects']

        num_classes_mac = len(MACRO_GROUP_NAMES)
        class_counts = np.bincount(y_mac, minlength=num_classes_mac)
        smooth_w = np.sqrt(len(y_mac) / (float(num_classes_mac) * np.maximum(class_counts, 1).astype(np.float32)))
        sample_w = smooth_w[y_mac]

        # 1. Bi-GRU
        print("    -> (1) Temporal Bi-GRU 분류기 학습 (12 Epochs)...", flush=True)
        self.gru_macro = AdvancedGRUClassifier(
            input_dim=X_b.shape[2],
            num_classes=num_classes_mac,
            epochs=12,
            lr=0.003,
            batch_size=512
        )
        self.gru_macro.fit(X_b, y_mac, class_weights=smooth_w)

        # 2. LightGBM on KCF
        print("    -> (2) LightGBM Classifier 학습 (55D KCF, n_estimators=100)...", flush=True)
        self.lgb_macro = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, n_jobs=4, verbose=-1
        )
        self.lgb_macro.fit(X_kcf, y_mac, sample_weight=sample_w)

        # 3. XGBoost on KCF
        print("    -> (3) XGBoost Classifier 학습 (55D KCF, n_estimators=100)...", flush=True)
        self.xgb_macro = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.85, colsample_bytree=0.85, random_state=42,
            eval_metric='mlogloss', n_jobs=4
        )
        self.xgb_macro.fit(X_kcf, y_mac, sample_weight=sample_w)

        # 4. Stage 2-A: 데드리프트 캐스케이드
        print("\n  [Stage 2/3] 데드리프트 패밀리 전용 v4.0 독립 2중 캐스케이드 훈련...")
        dead_mask = (y_act == 1) | (y_act == 2)
        X_dead_tr = X_b[dead_mask]
        y_dead_bin = (y_act[dead_mask] == 2).astype(int)
        dead_subjs = subjs[dead_mask]

        self.sub_dead_clf = DualCascadeDeadliftClassifier()
        self.sub_dead_clf.fit(X_dead_tr, y_dead_bin, subjects=dead_subjs)
        print("    -> ✔ v4.0 데드리프트 독립 감별사 및 14D 생체역학 리졸버 구축 완료")

        # 5. Stage 2-B: 무빙 런지 서브모델
        print("\n  [Stage 3/3] 무빙 런지 패밀리 전용 15D 최하사점 서브모델 훈련...")
        lunge_mask = (y_act == 4) | (y_act == 6)
        X_lunge_tr = LungeSubClassifier.slice_features(X_b[lunge_mask])
        y_lunge_bin = (y_act[lunge_mask] == 6).astype(int)

        self.sub_lunge_clf = LungeSubClassifier()
        lunge_tune = self.sub_lunge_clf.tune_and_fit(X_lunge_tr, y_lunge_bin)
        print(f"    -> ✔ 무빙 런지 분기기 학습 완료 (최적 임계값 τ* = {lunge_tune['best_threshold']:.2f})")

        t_tr_elapsed = time.time() - t_tr_start
        print("\n" + "=" * 105)
        print(f"  ★ [성공] 1단계 v5.0 전체 파이프라인 100% 학습 완료! (총 소요: {t_tr_elapsed:.1f}초)")
        print("=" * 105)

    def predict_probabilities(self, X_b: np.ndarray) -> np.ndarray:
        """3D 시계열 텐서(Format B)에 대해 55D KCF 추출 후 8대 운동 소프트맥스 사후확률 추론"""
        # 1. 55D KCF 피처 추출
        X_kcf = KinematicCoordinationExtractor.extract_features(X_b)

        # 2. Stage 1: 6대 상위 그룹 하이브리드 앙상블 (GRU 0.50 + LGB 0.30 + XGB 0.20)
        P_gru = self.gru_macro.predict_proba(X_b)
        P_lgb = self.lgb_macro.predict_proba(X_kcf)
        P_xgb = self.xgb_macro.predict_proba(X_kcf)
        P_macro = 0.50 * P_gru + 0.30 * P_lgb + 0.20 * P_xgb

        # 3. Stage 2-A: 데드리프트 캐스케이드 확률
        P_dead_stiff = self.sub_dead_clf.predict_proba(X_b)[:, 1]

        # 4. Stage 2-B: 무빙 런지 서브모델 확률
        X_b_lunge_sub = LungeSubClassifier.slice_features(X_b)
        P_lunge_cross = self.sub_lunge_clf.predict_proba(X_b_lunge_sub)[:, 1]

        # 5. 베이지안 확률 결합 (8개 클래스 복원)
        N = len(X_b)
        P_final = np.zeros((N, 8), dtype=np.float32)

        # 단독 운동
        P_final[:, 0] = P_macro[:, 0] # 01. 바벨 스쿼트
        P_final[:, 3] = P_macro[:, 2] # 04. 바벨 런지 (단독)
        P_final[:, 5] = P_macro[:, 4] # 06. 사이드 런지
        P_final[:, 7] = P_macro[:, 5] # 08. 굿모닝

        # 데드리프트 패밀리 분배
        P_dead_family = P_macro[:, 1]
        P_final[:, 1] = P_dead_family * (1.0 - P_dead_stiff) # 02. 일반 데드
        P_final[:, 2] = P_dead_family * P_dead_stiff         # 03. 스티프 데드

        # 무빙 런지 패밀리 분배
        P_lunge_family = P_macro[:, 3]
        P_final[:, 4] = P_lunge_family * (1.0 - P_lunge_cross) # 05. 다이나믹 런지
        P_final[:, 6] = P_lunge_family * P_lunge_cross         # 07. 크로스 런지

        return P_final

    def evaluate_benchmark(
        self,
        dataset_name: str,
        npz_filename: str,
        target_class_map: Dict[int, str]
    ) -> Dict[str, Any]:
        """외부 벤치마크 데이터셋에 대해 호환 운동 필터링 및 8-Class 제로샷 일반화 평가"""
        npz_path = os.path.join(BENCHMARK_DIR, npz_filename)
        if not os.path.exists(npz_path):
            print(f"[!] 파일을 찾을 수 없습니다: {npz_path}")
            return {}

        data = np.load(npz_path, allow_pickle=True)
        X_b = data['X_Format_B']
        y_raw = data['y_action']
        sids = data['sequence_ids'] if 'sequence_ids' in data else np.arange(len(y_raw))

        target_classes = sorted(list(target_class_map.keys()))
        mask = np.isin(y_raw, target_classes)
        
        # Kaggle RDL(루마니안 데드리프트) 클립 제외
        if 'Kaggle' in dataset_name:
            rdl_mask = np.array(['rdl' in str(s).lower() or 'romanian' in str(s).lower() for s in sids])
            mask = mask & (~rdl_mask)

        X_b_eval = X_b[mask]
        y_true = y_raw[mask]
        sids_eval = sids[mask]

        N_target = len(y_true)
        if N_target == 0:
            print(f"[!] 타깃 데이터가 없습니다: {dataset_name}")
            return {}

        t_inf_start = time.time()
        P_final = self.predict_probabilities(X_b_eval)
        t_inf_end = time.time()
        fps_inf = N_target / max(1e-5, (t_inf_end - t_inf_start))

        preds = np.argmax(P_final, axis=-1)
        top2_indices = np.argsort(P_final, axis=-1)[:, -2:]

        top1_hits = 0
        top2_hits_total = 0
        for i in range(N_target):
            if preds[i] == y_true[i]:
                top1_hits += 1
            if y_true[i] in top2_indices[i]:
                top2_hits_total += 1

        top1_acc = (top1_hits / N_target) * 100.0
        top2_acc = (top2_hits_total / N_target) * 100.0

        print("\n" + "=" * 105)
        print(f" [벤치마크 평가 결과]: {dataset_name} (총 {N_target}개 시퀀스)")
        print("=" * 105)
        print(f"{'클래스 명칭':<20} | {'샘플수':>7} | {'Top-1 Recall':>14} | {'Top-1 적중':>11} | {'Top-2 적중':>11} | {'주요 오분류'}")
        print("-" * 105)

        class_reports = []
        for c_idx in target_classes:
            c_name = target_class_map[c_idx]
            c_mask = (y_true == c_idx)
            n_c = int(np.sum(c_mask))
            if n_c == 0:
                continue

            c_preds = preds[c_mask]
            hits = int(np.sum(c_preds == c_idx))
            rec = (hits / n_c) * 100.0
            top2_hits = int(np.sum([c_idx in top2_indices[c_mask][i] for i in range(n_c)]))
            top2_rec = (top2_hits / n_c) * 100.0

            err_preds = c_preds[c_preds != c_idx]
            if len(err_preds) > 0:
                err_u, err_c = np.unique(err_preds, return_counts=True)
                top_err_idx = err_u[np.argmax(err_c)]
                err_desc = f"{EXERCISE_SHORT_NAMES[top_err_idx]} ({np.max(err_c)}건)"
            else:
                err_desc = "오분류 없음 (100% 정답)"

            print(f"{c_name:<20} | {n_c:>7d} | {rec:>13.2f}% | {hits:>7d}/{n_c:<3d} | {top2_hits:>7d}/{n_c:<3d} | {err_desc}")

            class_reports.append({
                'class_id': c_idx,
                'class_name': c_name,
                'count': n_c,
                'hits': hits,
                'recall': rec,
                'top2_recall': top2_rec
            })

        print("-" * 105)
        print(f"  ▶ 종합 Top-1 Accuracy: {top1_acc:.2f}%  |  Top-2 Accuracy: {top2_acc:.2f}%  |  추론 속도: {fps_inf:.1f} seq/s")

        # 8-Class 혼동 행렬
        print("\n  [혼동 행렬: 실제 타깃 운동 -> 모델의 8종 선택지 예측 분포]")
        header = f"{'실제 운동':<16} | " + " | ".join([f"{EXERCISE_SHORT_NAMES[i][:4]:>5}" for i in range(8)])
        print("  " + "-" * len(header))
        print("  " + header)
        print("  " + "-" * len(header))

        cm_rows = []
        for c_idx in target_classes:
            c_name = target_class_map[c_idx]
            c_mask = (y_true == c_idx)
            row_preds = preds[c_mask]
            row_counts = [int(np.sum(row_preds == i)) for i in range(8)]
            cm_rows.append(row_counts)
            row_str = f"  {c_name:<16} | " + " | ".join([f"{cnt:>5d}" for cnt in row_counts])
            print(row_str)
        print("  " + "-" * len(header))

        return {
            'dataset_name': dataset_name,
            'num_samples': N_target,
            'top1_accuracy': top1_acc,
            'top2_accuracy': top2_acc,
            'fps': fps_inf,
            'class_reports': class_reports,
            'confusion_matrix': cm_rows
        }

    def run_full_generalization_benchmark(self):
        """AI Hub 100% 학습 -> 3대 외부 벤치마크 검증 및 결과 집계"""
        print("\n" + "=" * 115)
        print("  ★ [NovelFlow] 1단계 8종 운동 분류 모델 (v5.0 KCF 하이브리드) 외부 일반화 벤치마크")
        print("=" * 115)

        self.aihub_data = self.load_aihub_master_pool()
        self.train_full_v5_pipeline()

        # 1. RepCount
        repcount_target_map = {0: '01.스쿼트 (RepCount)'}
        res_repcount = self.evaluate_benchmark(
            dataset_name="RepCount (YouTube RGB)",
            npz_filename="RepCount_3D_features.npz",
            target_class_map=repcount_target_map
        )

        # 2. UCF101
        ucf101_target_map = {0: '01.스쿼트 (BodyWeight)', 4: '05.다이나믹런지 (Lunges)'}
        res_ucf101 = self.evaluate_benchmark(
            dataset_name="UCF101 (In-the-wild RGB)",
            npz_filename="UCF101_3D_features.npz",
            target_class_map=ucf101_target_map
        )

        # 3. Kaggle Workout
        kaggle_target_map = {0: '01.바벨스쿼트', 1: '02.바벨데드리프트'}
        res_kaggle = self.evaluate_benchmark(
            dataset_name="Kaggle Workout (Barbell RGB)",
            npz_filename="Kaggle_Workout_3D_features.npz",
            target_class_map=kaggle_target_map
        )

        # 결과 저장
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
        final_summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': 'v5.0 KCF Hybrid',
            'train_dataset': 'AI Hub Full (Condition 4 NovelFlow 3D, 20,003 samples)',
            'benchmarks': {
                'RepCount': res_repcount,
                'UCF101': res_ucf101,
                'Kaggle_Workout': res_kaggle
            }
        }
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        print(f"\n[+] 종합 일반화 검증 결과 JSON 저장 완료: {OUTPUT_JSON_PATH}")

def main():
    tester = GeneralizationV5Tester()
    tester.run_full_generalization_benchmark()

if __name__ == '__main__':
    main()
