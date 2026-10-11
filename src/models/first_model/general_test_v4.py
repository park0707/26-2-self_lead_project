# =========================================================================================================
# [스크립트명]: general_test_v4.py
# [위치]: src/models/first_model/general_test_v4.py
# [목적]: AI Hub 전수(12,762개 시퀀스 100%) 학습 기반 1단계 v4.0 2-Stage 캐스케이드 모델 피팅 및
#         독립 외부 벤치마크(UI-PRMD Kinect 3D, REHAB24-6 OptiTrack 3D) 호환 운동 제로샷 일반화 성능 검증기
#
# [실행 방법]:
#   - 프로젝트 루트 디렉토리(d:\)에서:
#       python -m src.models.first_model.general_test_v4
#   - 또는 직접 실행:
#       python src/models/first_model/general_test_v4.py
# =========================================================================================================

import os
import sys
import time
import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
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
from src.models.first_model.train_kfold_ensemble import DiscriminativeKinematicsExtractor
from src.models.sub_models.dual_cascade_deadlift_classifier import DualCascadeDeadliftClassifier
from src.models.sub_models.lunge_sub_classifier import LungeSubClassifier

# 데이터셋 경로
AIHUB_MASTER_DIR = 'd:/Data/피처_데이터/Condition4_NovelFlow_3D' if os.path.exists('d:/Data/피처_데이터/Condition4_NovelFlow_3D/Dual_Train.npz') else 'd:/Data/피처_데이터/Condition3_MediaPipe_3D'
BENCHMARK_DIR = 'd:/Data/피처_데이터/External_Benchmarks_3D'
OUTPUT_JSON_PATH = 'd:/Data/피처_데이터/tuning_results/generalization_v4_results.json'

EXERCISE_SHORT_NAMES = [
    '01.스쿼트', '02.데드리프트', '03.스티프데드', '04.바벨런지',
    '05.다이나믹런지', '06.사이드런지', '07.크로스런지', '08.굿모닝'
]

class GeneralizationV4Tester:
    """AI Hub 100% 학습 기반 v4.0 2-Stage 모델 및 외부 벤치마크 일반화 검증 엔진"""

    def __init__(self):
        self.aihub_data = None
        self.uiprmd_data = None
        self.rehab_data = None

        # 학습된 모델 컴포넌트
        self.gru_macro = None
        self.xgb_macro = None
        self.cat_macro = None
        self.sub_dead_clf = None
        self.sub_lunge_clf = None

    def load_aihub_master_pool(self) -> Dict[str, Any]:
        """AI Hub 전수 데이터(12,762개 시퀀스 100%) 로드 및 판별 피처 융합"""
        print("\n" + "=" * 105)
        print(f" [1단계] AI Hub 전수 학습 데이터 로드 및 3D 판별 피처 결합 (경로: {AIHUB_MASTER_DIR})")
        print("=" * 105)

        all_X_a, all_X_b, all_y_act, all_subjs = [], [], [], []
        npz_files = sorted([f for f in os.listdir(AIHUB_MASTER_DIR) if f.endswith('.npz')])

        for fn in npz_files:
            split = Stage1DatasetLoader.load_split_data(fn, splits_dir=AIHUB_MASTER_DIR)
            all_X_a.append(split['X_Format_A'])
            all_X_b.append(split['X_Format_B'])
            all_y_act.append(split['y_action'])
            all_subjs.extend(split['subjects'])

        raw_X_a = np.concatenate(all_X_a, axis=0)
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
        print(f"  [*] Format B ({raw_X_b.shape[1]}F, {raw_X_b.shape[2]}D)로부터 3D 생체역학 판별 피처(49D) 동적 융합 중...")
        discrim_feats = DiscriminativeKinematicsExtractor.extract_features(raw_X_b)
        enhanced_X_a = np.concatenate([raw_X_a, discrim_feats], axis=-1)

        y_macro = np.array([CLASS_TO_MACRO_GROUP[c] for c in y_act], dtype=np.int64)

        print(f"  [OK] Enhanced Format A 융합 완료: Shape {enhanced_X_a.shape}")
        print(f"  [OK] 6대 상위 기능 그룹 분포:")
        for g_idx, g_name in enumerate(MACRO_GROUP_NAMES):
            cnt = int(np.sum(y_macro == g_idx))
            print(f"       - {g_name:<32}: {cnt:>5d}개 ({cnt/len(y_macro)*100:.1f}%)")

        return {
            'X_Format_A': enhanced_X_a,
            'X_Format_B': raw_X_b,
            'y_action': y_act,
            'y_macro': y_macro,
            'subjects': subjs_arr
        }

    def train_full_v4_pipeline(self):
        """AI Hub 12,762개 100% 데이터로 v4.0 2-Stage 모델 피팅"""
        print("\n" + "=" * 105)
        print(" [2단계] v4.0 2-Stage 캐스케이드 앙상블 시스템 100% 전수 훈련 시작")
        print("=" * 105)
        t_tr_start = time.time()

        X_a = self.aihub_data['X_Format_A']
        X_b = self.aihub_data['X_Format_B']
        y_act = self.aihub_data['y_action']
        y_mac = self.aihub_data['y_macro']
        subjs = self.aihub_data['subjects']

        # -------------------------------------------------------------
        # Stage 1: 6대 상위 그룹 3중 하이브리드 앙상블 (Bi-GRU + XGB + Cat)
        # -------------------------------------------------------------
        print("\n  [Stage 1/3] 6대 상위 기능 그룹 3중 하이브리드 앙상블 훈련 (GRU 0.65 + XGB 0.20 + Cat 0.15)...")
        num_classes_mac = 6
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

        # 2. XGBoost
        print("    -> (2) XGBoost Classifier 학습 (n_estimators=100)...", flush=True)
        self.xgb_macro = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.85, colsample_bytree=0.85, random_state=42,
            eval_metric='mlogloss', n_jobs=4
        )
        self.xgb_macro.fit(X_a, y_mac, sample_weight=sample_w)

        # 3. CatBoost
        print("    -> (3) CatBoost Classifier 학습 (iterations=100)...", flush=True)
        self.cat_macro = CatBoostClassifier(
            iterations=100, depth=5, learning_rate=0.1,
            random_seed=42, verbose=False, thread_count=4
        )
        self.cat_macro.fit(X_a, y_mac, sample_weight=sample_w)

        # -------------------------------------------------------------
        # Stage 2-A (v4.0): 데드리프트 독립 2중 캐스케이드 & 데이터 정제
        # -------------------------------------------------------------
        print("\n  [Stage 2/3] 데드리프트 패밀리(02 일반 vs 03 스티프) 전용 v4.0 독립 2중 캐스케이드 훈련...")
        dead_mask = (y_act == 1) | (y_act == 2)
        X_dead_tr = X_b[dead_mask]
        y_dead_bin = (y_act[dead_mask] == 2).astype(int) # 0: 일반, 1: 스티프
        dead_subjs = subjs[dead_mask]

        self.sub_dead_clf = DualCascadeDeadliftClassifier()
        self.sub_dead_clf.fit(X_dead_tr, y_dead_bin, subjects=dead_subjs)
        print("    -> ✔ v4.0 데드리프트 독립 감별사(1A/1B) 및 14D 생체역학 리졸버 구축 완료")

        # -------------------------------------------------------------
        # Stage 2-B: 무빙 런지 패밀리 2진 서브모델
        # -------------------------------------------------------------
        print("\n  [Stage 3/3] 무빙 런지 패밀리(05 다이나믹 vs 07 크로스) 전용 15D 최하사점 서브모델 훈련...")
        lunge_mask = (y_act == 4) | (y_act == 6)
        X_lunge_tr = LungeSubClassifier.slice_features(X_b[lunge_mask])
        y_lunge_bin = (y_act[lunge_mask] == 6).astype(int) # 0: 다이나믹, 1: 크로스

        self.sub_lunge_clf = LungeSubClassifier()
        lunge_tune = self.sub_lunge_clf.tune_and_fit(X_lunge_tr, y_lunge_bin)
        print(f"    -> ✔ 무빙 런지 분기기 학습 완료 (최적 임계값 τ* = {lunge_tune['best_threshold']:.2f})")

        t_tr_elapsed = time.time() - t_tr_start
        print("\n" + "=" * 105)
        print(f"  ★ [성공] 1단계 v4.0 전체 파이프라인 100% 학습 완료! (총 학습 소요 시간: {t_tr_elapsed:.1f}초)")
        print("=" * 105)

    def predict_probabilities(self, X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
        """새로운 3D 시퀀스(Format A, Format B)에 대해 8대 운동 소프트맥스 사후확률 추론"""
        # 1. 3D 생체역학 판별 피처 결합 (Enhanced Format A 구성)
        disc_feats = DiscriminativeKinematicsExtractor.extract_features(X_b)
        enhanced_X_a = np.concatenate([X_a, disc_feats], axis=-1)

        # 2. Stage 1: 6대 상위 그룹 앙상블 추론
        P_gru = self.gru_macro.predict_proba(X_b)
        P_xgb = self.xgb_macro.predict_proba(enhanced_X_a)
        P_cat = self.cat_macro.predict_proba(enhanced_X_a)
        P_macro = 0.65 * P_gru + 0.20 * P_xgb + 0.15 * P_cat # (N, 6)

        # 3. Stage 2-A: 데드리프트 캐스케이드 확률
        P_dead_stiff = self.sub_dead_clf.predict_proba(X_b)[:, 1] # (N,)

        # 4. Stage 2-B: 무빙 런지 서브모델 확률
        X_b_lunge_sub = LungeSubClassifier.slice_features(X_b)
        P_lunge_cross = self.sub_lunge_clf.predict_proba(X_b_lunge_sub)[:, 1] # (N,)

        # 5. 베이지안 확률 결합 (8개 클래스 복원)
        N = len(X_a)
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
            raise FileNotFoundError(f"벤치마크 파일을 찾을 수 없습니다: {npz_path}")

        d = np.load(npz_path, allow_pickle=True)
        X_a_all = d['X_Format_A']
        X_b_all = d['X_Format_B']
        # 호환 운동 서브셋 필터링 (target_class_map에 정의된 클래스만 엄밀히 포함)
        target_classes = sorted(list(target_class_map.keys()))
        class_filter = np.isin(y_act_all, target_classes)
        base_mask = (d['is_target'] == 1) if 'is_target' in d else np.ones(len(y_act_all), dtype=bool)
        target_mask = base_mask & class_filter

        X_a = X_a_all[target_mask]
        X_b = X_b_all[target_mask]
        y_true = y_act_all[target_mask]
        N_target = len(y_true)

        print("\n" + "=" * 105)
        print(f"  ★ [{dataset_name}] 외부 도메인 제로샷 일반화 검증")
        print(f"     전체 데이터: {len(y_act_all):,}개 중 호환 타깃 시퀀스: {N_target:,}개 (OOD: {len(y_act_all)-N_target:,}개)")
        print("=" * 105)

        t_inf_start = time.time()
        P_final = self.predict_probabilities(X_a, X_b)
        t_inf_elapsed = time.time() - t_inf_start
        fps_inf = N_target / max(t_inf_elapsed, 1e-4)

        preds = np.argmax(P_final, axis=-1)

        # Top-2 판별
        top2_indices = np.argsort(P_final, axis=-1)[:, -2:]
        top2_correct = np.array([y_true[i] in top2_indices[i] for i in range(N_target)])
        top2_acc = float(np.mean(top2_correct)) * 100.0
        top1_acc = float(accuracy_score(y_true, preds)) * 100.0

        # 타깃 클래스별 세부 성능 산출
        class_reports = []
        target_classes = sorted(list(target_class_map.keys()))

        print("\n" + "-" * 105)
        print(f"{'검증 대상 운동':<20} | {'샘플수':>7} | {'Recall(재현율)':>14} | {'Top-1 맞힘':>11} | {'Top-2 포함':>11} | {'주요 오분류'}")
        print("-" * 105)

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

            # 오분류 상위 2개 탐색
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

        # 8-Class 혼동 행렬 출력
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
        """AI Hub 100% 학습 -> UI-PRMD 검증 -> REHAB24-6 검증 -> 종합 도메인 시프트 대조표 출력"""
        print("\n" + "=" * 115)
        print("  ★ [NovelFlow] 1단계 8종 운동 분류 모델 (v4.0 2-Stage 캐스케이드) 외부 일반화 공인 벤치마크")
        print("  ★ [학습 도메인]: AI Hub 단안 RGB 비디오 3D 추정 전수 풀 (12,762개)")
        print("  ★ [외부 도메인 1]: RepCount YouTube 고화질 비디오 (미국/글로벌)")
        print("  ★ [외부 도메인 2]: UCF101 웹 수집 인더와일드 비디오 (센트럴 플로리다 대)")
        print("  ★ [외부 도메인 3]: Kaggle Workout 바벨 3대 운동 비디오 (글로벌 헬스장)")
        print("=" * 115)

        # 1. 학습 데이터 로드 및 모델 100% 피팅
        self.aihub_data = self.load_aihub_master_pool()
        self.train_full_v4_pipeline()

        # 2. RepCount 평가 (YouTube HD RGB Video)
        repcount_target_map = {
            0: '01.스쿼트 (RepCount)'
        }
        res_repcount = self.evaluate_benchmark(
            dataset_name="RepCount (YouTube RGB)",
            npz_filename="RepCount_3D_features.npz",
            target_class_map=repcount_target_map
        )

        # 3. UCF101 평가 (In-the-wild RGB Video)
        ucf101_target_map = {
            0: '01.스쿼트 (BodyWeight)',
            4: '05.다이나믹런지 (Lunges)'
        }
        res_ucf101 = self.evaluate_benchmark(
            dataset_name="UCF101 (In-the-wild RGB)",
            npz_filename="UCF101_3D_features.npz",
            target_class_map=ucf101_target_map
        )

        # 4. Kaggle Workout 평가 (Barbell RGB Video: 루마니안 데드리프트 제외)
        kaggle_target_map = {
            0: '01.바벨스쿼트',
            1: '02.바벨데드리프트'
        }
        res_kaggle = self.evaluate_benchmark(
            dataset_name="Kaggle Workout (Barbell RGB)",
            npz_filename="Kaggle_Workout_3D_features.npz",
            target_class_map=kaggle_target_map
        )

        # 5. 비디오 도메인 시프트 종합 비교표 출력
        print("\n" + "=" * 125)
        print("                       [외부 비디오 3대 벤치마크 (RepCount / UCF101 / Kaggle) 일반화 성능 종합 비교표]")
        print("=" * 125)
        print(f"{'검증 도메인 (영상 환경)':<32} | {'모달리티':<18} | {'대상 운동 구성':<22} | {'샘플수':>7} | {'Top-1 정확도':>12} | {'Top-2 정확도':>12}")
        print("-" * 125)
        print(f"{'AI Hub (In-Domain RGB 3D)':<32} | {'AI Hub 3D 추정':<18} | {'8대 전 종목 (3-Fold)':<22} | {12762:>7,d} | {'94.49%':>12} | {'98.80%':>12}")
        print(f"{res_repcount['dataset_name']:<32} | {'YouTube 720p/1080p':<18} | {'스쿼트 (고화질 긴호흡)':<22} | {res_repcount['num_samples']:>7d} | {res_repcount['top1_accuracy']:>11.2f}% | {res_repcount['top2_accuracy']:>11.2f}%")
        print(f"{res_ucf101['dataset_name']:<32} | {'In-the-wild 240p':<18} | {'스쿼트 / 런지 (단편클립)':<22} | {res_ucf101['num_samples']:>7d} | {res_ucf101['top1_accuracy']:>11.2f}% | {res_ucf101['top2_accuracy']:>11.2f}%")
        print(f"{res_kaggle['dataset_name']:<32} | {'Kaggle Barbell RGB':<18} | {'스쿼트/바벨데드 (RDL제외)':<22} | {res_kaggle['num_samples']:>7d} | {res_kaggle['top1_accuracy']:>11.2f}% | {res_kaggle['top2_accuracy']:>11.2f}%")
        print("=" * 125)

        # 6. 결과 JSON 저장
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
        final_summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'train_dataset': 'AI Hub Full (Condition 4 NovelFlow 3D, 12,762 samples)',
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
    tester = GeneralizationV4Tester()
    tester.run_full_generalization_benchmark()

if __name__ == '__main__':
    main()
