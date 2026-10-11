# [실행 유틸리티]: 1단계 9종 운동 분류 전용 초정밀 다차원 평가 모듈

import os
import sys
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, cohen_kappa_score, top_k_accuracy_score,
    classification_report, confusion_matrix
)

EXERCISE_NAMES_SHORT = [
    '01.스쿼트',
    '02.데드리프트',
    '03.스티프데드',
    '04.바벨런지',
    '05.다이나믹런지',
    '06.사이드런지',
    '07.크로스런지',
    '08.굿모닝'
]

TUNING_RESULTS_DIR = 'd:/Data/피처_데이터/tuning_results'

class Stage1MetricsHelper:
    """1단계 8종 운동 분류 모델을 위한 전문 학술 평가, 그리드서치 비교 및 터미널 리포팅 유틸리티"""

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        inference_time_ms: float = 0.0
    ) -> Dict[str, Any]:
        """기본 지표 + 전문 지표(Top-2 Acc, Balanced Acc, Cohen's Kappa) 전수 정밀 계산"""
        acc = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Top-2 Accuracy (차선책 정확도: 근접 오답 판별)
        top2_acc = 0.0
        if y_proba is not None:
            try:
                top2_acc = top_k_accuracy_score(y_true, y_proba, k=2, labels=list(range(len(EXERCISE_NAMES_SHORT))))
            except Exception:
                top2_acc = float(acc)
        else:
            top2_acc = float(acc)

        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        fps = 1000.0 / max(1e-4, inference_time_ms) if inference_time_ms > 0 else 0.0

        # 9개 클래스별 개별 지표
        class_f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
        class_precs = precision_score(y_true, y_pred, average=None, zero_division=0)
        class_recs = recall_score(y_true, y_pred, average=None, zero_division=0)

        # 9x9 혼동 행렬
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EXERCISE_NAMES_SHORT))))

        return {
            'accuracy': float(acc),
            'balanced_accuracy': float(balanced_acc),
            'cohen_kappa': float(kappa),
            'top2_accuracy': float(top2_acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'macro_precision': float(macro_prec),
            'macro_recall': float(macro_rec),
            'inference_time_ms': float(inference_time_ms),
            'fps': float(fps),
            'class_f1s': [float(v) for v in class_f1s],
            'class_precs': [float(v) for v in class_precs],
            'class_recs': [float(v) for v in class_recs],
            'confusion_matrix': cm
        }

    @staticmethod
    def calculate_advanced_metrics(
        res_train: Optional[Dict[str, Any]],
        res_valid: Dict[str, Any],
        res_strict: Dict[str, Any],
        res_subj: Dict[str, Any],
        res_view: Dict[str, Any]
    ) -> Dict[str, Any]:
        """스포츠 AI 도메인 특화 고급 지표 (PDR, SGG, VIS) 계산"""
        f1_train = res_train['macro_f1'] if res_train else 1.0
        f1_strict = res_strict['macro_f1']
        f1_subj = res_subj['macro_f1']
        f1_view = res_view['macro_f1']
        f1_valid = res_valid['macro_f1']

        # 1. 시점 저하율 (Perspective Degradation Ratio, PDR)
        pdr = max(0.0, (1.0 - (f1_strict / max(1e-4, f1_subj)))) * 100.0

        # 2. 피험자 일반화 갭 (Subject Generalization Gap, SGG)
        sgg = max(0.0, (f1_train - f1_subj)) * 100.0 if res_train else 0.0

        # 3. 시점 불변성 지수 (View Invariance Score, VIS)
        all_test_f1s = [f1_strict, f1_subj, f1_view, f1_valid]
        vis = (min(all_test_f1s) / max(1e-4, max(all_test_f1s))) * 100.0

        return {
            'PDR_percent': float(pdr),
            'SGG_percent': float(sgg),
            'VIS_percent': float(vis)
        }

    @staticmethod
    def print_model_header(model_name: str, hyperparams: Dict[str, Any], train_samples: int, feat_dim: str):
        """모델 정보 및 하이퍼파라미터 헤더 출력"""
        print("=" * 96)
        print(f"  [1단계 모델 평가 & 튜닝] {model_name}")
        print("=" * 96)
        print(f"[*] 입력 피처 규격   : {feat_dim}")
        print(f"[*] 학습 데이터 규모 : {train_samples:,}개 시퀀스 (피험자 1~6 x 시점 A, B, E)")
        print("[*] 핵심 파라미터    :")
        for k, v in hyperparams.items():
            print(f"    - {k:<22s}: {v}")
        print("-" * 96)

    @staticmethod
    def print_multi_split_summary(split_results: Dict[str, Dict[str, Any]], adv_metrics: Optional[Dict[str, Any]] = None):
        """4대 분할 세트별 정밀 성능(Acc, Balanced Acc, Top-2 Acc, Kappa, Macro-F1, FPS) 요약 출력"""
        print("\n[★ 6대 핵심 성능 평가 지표 가이드]")
        print("  - 정확도 (Accuracy)       : 전체 시퀀스 중 정답을 맞힌 비율로 모델의 전반적인 기본 분류 성공률을 보여준다.")
        print("  - Top-2 Acc (차선책 정확도): 모델이 예측한 상위 2개 후보 내에 실제 정답이 포함될 확률로 경계성 오분류 발생 시의 안전성을 보여준다.")
        print("  - 균형정확도 (Balanced Acc): 클래스별 데이터 불균형을 보정하여 소수 운동 클래스까지 왜곡 없이 공정하게 분류했는지를 보여준다.")
        print("  - Kappa (코헨 카파 계수)  : 우연히 맞힐 확률(Chance Agreement)을 완전히 배제한 순수 예측 일치도로 1.0에 가까울수록 신뢰도가 높음을 보여준다.")
        print("  - Macro-F1               : 9개 운동별 F1 점수의 산술 평균으로 다수/소수 클래스에 편향되지 않고 고르게 잘 맞히는지를 보여주는 핵심 지표이다.")
        print("  - FPS (초당 추론 속도)   : 1초 동안 실시간으로 처리할 수 있는 프레임 수로 모바일 환경(30 FPS 기준) 배포 가능 여부를 보여준다.")

        print("\n[1. 분할 세트별 초정밀 성능 지표 요약표]")
        print("-" * 96)
        header = f"{'평가 데이터 분할':<36} | {'정확도':^7} | {'Top-2 Acc':^9} | {'균형정확도':^8} | {'Kappa':^7} | {'Macro-F1':^8} | {'FPS':^8}"
        print(header)
        print("-" * 96)
        
        for split_name, res in split_results.items():
            acc = res['accuracy'] * 100.0
            raw_top2 = res.get('top2_accuracy', res['accuracy'])
            top2 = raw_top2 * 100.0 if raw_top2 <= 1.0 else raw_top2
            bacc = res.get('balanced_accuracy', res['accuracy']) * 100.0 if res.get('balanced_accuracy', 1.0) <= 1.0 else res.get('balanced_accuracy', 100.0)
            kap = res.get('cohen_kappa', 0.0)
            f1 = res['macro_f1'] * 100.0
            fps = res['fps']
            print(f"{split_name:<36} | {acc:>6.2f}% | {top2:>8.2f}% | {bacc:>7.2f}% | {kap:>6.3f} | {f1:>7.2f}% | {fps:>7.1f}")
        print("-" * 96)

        if adv_metrics:
            print(f"  ★ [도메인 지표] 시점 저하율(PDR): {adv_metrics['PDR_percent']:5.2f}% (낮을수록 측면 90도에 강건)")
            print(f"  ★ [도메인 지표] 인물 일반화 갭(SGG): {adv_metrics['SGG_percent']:5.2f}% (낮을수록 체형 과적합 없음)")
            print(f"  ★ [도메인 지표] 시점 불변성 지수(VIS): {adv_metrics['VIS_percent']:5.2f}% (높을수록 시점 독립적)")
            print("-" * 96)

    @staticmethod
    def print_class_breakdown_report(res: Dict[str, Any], y_true: np.ndarray, split_title: str = "Strict Test (복합 홀드아웃)"):
        """8개 운동별 개별 세부 성적표 출력"""
        print(f"\n[2. 8개 운동별 세부 성적표 - {split_title}]")
        print("-" * 96)
        header = f"{'운동 종목 명칭':<20} | {'샘플수':^7} | {'정밀도(Prec)':^12} | {'재현율(Rec)':^12} | {'F1-Score':^12}"
        print(header)
        print("-" * 96)

        for i, ex_name in enumerate(EXERCISE_NAMES_SHORT):
            cnt = int(np.sum(y_true == i))
            prec = res['class_precs'][i] * 100.0 if i < len(res['class_precs']) else 0.0
            rec = res['class_recs'][i] * 100.0 if i < len(res['class_recs']) else 0.0
            f1 = res['class_f1s'][i] * 100.0 if i < len(res['class_f1s']) else 0.0
            print(f"{ex_name:<20} | {cnt:>7d} | {prec:>11.2f}% | {rec:>11.2f}% | {f1:>11.2f}%")
        print("-" * 96)

    @staticmethod
    def print_top_confusion_analysis(res: Dict[str, Any], top_k: int = 3):
        """가장 많이 발생한 오분류(Confusion) 패턴 분석 출력"""
        cm = res.get('confusion_matrix')
        if cm is None:
            return

        print("\n[3. 주요 오분류(Confusion) 패턴 분석 (Top Errors)]")
        confusions = []
        for i in range(len(EXERCISE_NAMES_SHORT)):
            for j in range(len(EXERCISE_NAMES_SHORT)):
                if i != j and cm[i, j] > 0:
                    confusions.append((cm[i, j], i, j))
        
        confusions.sort(key=lambda x: x[0], reverse=True)
        if not confusions:
            print("  - 오분류 없음 (100% 정답)")
        else:
            for rank, (cnt, true_idx, pred_idx) in enumerate(confusions[:top_k], 1):
                t_name = EXERCISE_NAMES_SHORT[true_idx]
                p_name = EXERCISE_NAMES_SHORT[pred_idx]
                print(f"  {rank}위: 정답 [{t_name}] ──> 예측 [{p_name}] 오분류 ({cnt}건 발생)")
        print("-" * 96)

    @staticmethod
    def print_top_vs_random_comparison_table(
        best_trial: Dict[str, Any],
        other_trials: List[Dict[str, Any]],
        param_keys: List[str]
    ):
        """요구사항 5번: 최고 파라미터 조합 1개와 다른 후보 파라미터 조합 5개 비교 대조표 출력"""
        print("\n[4. 하이퍼파라미터 탐색 비교: 최고 파라미터 (Best) vs 후보 파라미터 5종 비교]")
        print("=" * 96)
        
        all_trials = [best_trial] + other_trials[:5]
        header = f"{'조합 구분 (Rank)':<18} | {'Valid F1':^10} | {'Strict F1':^10} | {'Subj F1':^10} | {'주요 파라미터 설정'}"
        print(header)
        print("-" * 96)

        for idx, t in enumerate(all_trials):
            label = "★ [1위 BEST]" if idx == 0 else f"후보 #{idx} (대조군)"
            v_f1 = t['valid_f1'] * 100.0
            st_f1 = t['strict_f1'] * 100.0
            sb_f1 = t['subj_f1'] * 100.0
            
            p_str = ", ".join([f"{k}={t['params'].get(k)}" for k in param_keys[:3]])
            print(f"{label:<18} | {v_f1:>8.2f}% | {st_f1:>8.2f}% | {sb_f1:>8.2f}% | {p_str}")

        print("=" * 96 + "\n")

    @staticmethod
    def evaluate_subject_breakdown(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        subjects: np.ndarray
    ) -> Dict[str, Any]:
        """피험자별 개별 성능 및 피험자 간 분산(Standard Deviation) 정밀 산출"""
        unique_subjs = np.unique(subjects)
        subj_metrics = {}
        f1_list = []

        for subj in unique_subjs:
            mask = (subjects == subj)
            if np.sum(mask) == 0:
                continue
            y_t_sub = y_true[mask]
            y_p_sub = y_pred[mask]
            acc_s = float(accuracy_score(y_t_sub, y_p_sub))
            f1_s = float(f1_score(y_t_sub, y_p_sub, average='macro', zero_division=0))
            subj_metrics[str(subj)] = {
                'samples': int(np.sum(mask)),
                'accuracy': acc_s,
                'macro_f1': f1_s
            }
            f1_list.append(f1_s)

        f1_std = float(np.std(f1_list)) if f1_list else 0.0
        f1_mean = float(np.mean(f1_list)) if f1_list else 0.0

        return {
            'subject_metrics': subj_metrics,
            'mean_subject_f1': f1_mean,
            'std_subject_f1': f1_std
        }

    @staticmethod
    def evaluate_group_confusion(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """4대 운동 대분류(스쿼트, 데드리프트류, 런지류, 굿모닝) 그룹화 정확도 분석"""
        # 그룹 매핑: 0: 스쿼트, 1,2: 데드리프트, 3,4,5,6,7: 런지, 8: 굿모닝
        group_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3}
        y_true_grp = np.array([group_map.get(y, 0) for y in y_true])
        y_pred_grp = np.array([group_map.get(y, 0) for y in y_pred])

        grp_acc = float(accuracy_score(y_true_grp, y_pred_grp))
        grp_f1 = float(f1_score(y_true_grp, y_pred_grp, average='macro', zero_division=0))

        return {
            'group_accuracy': grp_acc,
            'group_macro_f1': grp_f1,
            'group_names': ['스쿼트류', '데드리프트류', '런지류', '굿모닝류']
        }

    @staticmethod
    def evaluate_label_shuffling_test(
        clf_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int = 42
    ) -> Dict[str, Any]:
        """라벨 셔플링(Label Shuffling) 널 가설 대조 실험: 데이터 누수 및 피처 유효성 검증"""
        np.random.seed(seed)
        shuffled_y_train = np.random.permutation(y_train)
        
        # 모델 복제 및 학습
        shuffled_clf = clf_class()
        shuffled_clf.fit(X_train, shuffled_y_train)
        
        preds = shuffled_clf.predict(X_test)
        shuffled_acc = float(accuracy_score(y_test, preds))
        shuffled_f1 = float(f1_score(y_test, preds, average='macro', zero_division=0))

        # 9개 클래스 무작위 추측 확률: 1/9 = 11.11%
        expected_random = 1.0 / len(EXERCISE_NAMES_SHORT)
        has_leakage = (shuffled_acc > (expected_random + 0.15)) # 26% 초과 시 누수 의심

        return {
            'shuffled_accuracy': shuffled_acc,
            'shuffled_macro_f1': shuffled_f1,
            'expected_random_accuracy': expected_random,
            'leakage_detected': bool(has_leakage)
        }

    @staticmethod
    def save_best_params_json(model_name: str, best_params: Dict[str, Any], summary_metrics: Dict[str, Any]) -> str:
        """요구사항 4번: Grid Search 최고 파라미터 조합을 JSON으로 영구 저장"""
        os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)
        safe_name = model_name.lower().replace(' ', '_').replace('+', '').replace('-', '_').replace('(', '').replace(')', '')
        filepath = os.path.join(TUNING_RESULTS_DIR, f"best_stage1_{safe_name}.json")
        
        save_dict = {
            'model_name': model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'best_hyperparameters': best_params,
            'performance_metrics': summary_metrics
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        
        print(f"[+] 최고 파라미터 조합 JSON 저장 완료: {filepath}")
        return filepath
