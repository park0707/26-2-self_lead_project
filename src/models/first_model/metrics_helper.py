import time
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

EXERCISE_NAMES_SHORT = [
    '01.스쿼트',
    '02.데드리프트',
    '03.스티프데드',
    '04.바벨런지',
    '05.포워드런지',
    '06.백워드런지',
    '07.사이드런지',
    '08.크로스런지',
    '09.굿모닝'
]

class Stage1MetricsHelper:
    """1단계 9종 운동 분류 모델을 위한 전문 학술 평가 및 터미널 리포팅 유틸리티"""

    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, inference_time_ms: float = 0.0) -> Dict[str, Any]:
        """정량적 핵심 지표 계산"""
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        fps = 1000.0 / max(1e-4, inference_time_ms) if inference_time_ms > 0 else 0.0

        # 클래스별 개별 지표
        class_f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
        class_precs = precision_score(y_true, y_pred, average=None, zero_division=0)
        class_recs = recall_score(y_true, y_pred, average=None, zero_division=0)

        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EXERCISE_NAMES_SHORT))))

        return {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_prec,
            'macro_recall': macro_rec,
            'inference_time_ms': inference_time_ms,
            'fps': fps,
            'class_f1s': class_f1s,
            'class_precs': class_precs,
            'class_recs': class_recs,
            'confusion_matrix': cm
        }

    @staticmethod
    def print_model_header(model_name: str, hyperparams: Dict[str, Any], train_samples: int, feat_dim: str):
        """모델 정보 및 하이퍼파라미터 헤더 출력"""
        print("=" * 85)
        print(f"  [1단계 모델 평가] {model_name}")
        print("=" * 85)
        print(f"[*] 입력 피처 규격: {feat_dim}")
        print(f"[*] 학습 데이터 규모: {train_samples:,}개 시퀀스 (피험자 1~6 x 시점 A, B, E)")
        print("[*] 모델 하이퍼파라미터 설정:")
        for k, v in hyperparams.items():
            print(f"    - {k:<20s}: {v}")
        print("-" * 85)

    @staticmethod
    def print_multi_split_summary(split_results: Dict[str, Dict[str, Any]]):
        """4대 분할 세트별 성능 요약 테이블 출력"""
        print("\n[1. 분할 세트별 종합 성능 요약표]")
        print("-" * 85)
        header = f"{'평가 데이터 분할':<38} | {'정확도(Acc)':^11} | {'Macro-F1':^11} | {'처리속도(FPS)':^12}"
        print(header)
        print("-" * 85)
        
        for split_name, res in split_results.items():
            acc = res['accuracy'] * 100
            f1 = res['macro_f1'] * 100
            fps = res['fps']
            print(f"{split_name:<38} | {acc:>10.2f}% | {f1:>10.2f}% | {fps:>10.1f} FPS")
        print("-" * 85)

    @staticmethod
    def print_class_breakdown_report(res: Dict[str, Any], y_true: np.ndarray, split_title: str = "Strict Test (복합 홀드아웃)"):
        """9개 운동별 개별 세부 성적표 출력"""
        print(f"\n[2. 9개 운동별 세부 성적표 - {split_title}]")
        print("-" * 85)
        header = f"{'운동 종목 명칭':<20} | {'샘플수':^7} | {'정밀도(Prec)':^12} | {'재현율(Rec)':^12} | {'F1-Score':^12}"
        print(header)
        print("-" * 85)

        for i, ex_name in enumerate(EXERCISE_NAMES_SHORT):
            cnt = int(np.sum(y_true == i))
            prec = res['class_precs'][i] * 100 if i < len(res['class_precs']) else 0.0
            rec = res['class_recs'][i] * 100 if i < len(res['class_recs']) else 0.0
            f1 = res['class_f1s'][i] * 100 if i < len(res['class_f1s']) else 0.0
            print(f"{ex_name:<20} | {cnt:>7d} | {prec:>11.2f}% | {rec:>11.2f}% | {f1:>11.2f}%")
        print("-" * 85)

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
        print("=" * 85 + "\n")
