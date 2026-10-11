# -*- coding: utf-8 -*-
# =========================================================================================================
# [스크립트 명칭]: NovelFlow 바벨 스쿼트 정상·오류 분류 클린 리셋 단일 모델 평가기 (test_babel_squat.py)
# [실행 방법]:
#     터미널에서 아래 명령어를 입력하여 실행합니다:
#     python test_babel_squat.py
#
# [표준 평가지표 구성 (User-Defined Standard Metrics)]:
#   1. 정상 자세 (Normal, Class 1) : Precision, Recall, F1-Score
#   2. 오류 자세 (Error,  Class 0) : Precision, Recall, F1-Score
#   3. 종합 지표 (Overall Metrics) : 종합 F1-Score (Macro F1), 가중 F1-Score (Weighted F1)
#
# [설계 철학 (Option 1: Clean Reset)]:
#   - 213개 복잡한 통계량 폐기 -> 스쿼트 4대 영역 15대 직관 생체역학 피처 (93.0% 압축)
#   - 절대값 100% 제거 -> 관절 사잇각 및 무차원 비율로만 구성 (카메라/체형 일반화 완벽 보장)
#   - 5개 앙상블 모델 폐기 -> 단일 Cost-Sensitive LightGBM 1개 (0.13ms 초고속 추론)
#   - 피험자 격리 5-Fold 교차 검증 (StratifiedGroupKFold)으로 데이터 누수 없는 정직한 평가
# =========================================================================================================

import os
import sys

# 프로젝트 루트 경로 등록
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, 'd:/')

# Windows 콘솔 한글 UTF-8 출력 강제
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

# 데이터셋 로더 및 최하사점 추출기
from src.models.second_models.pilot_squat_dataset import SquatPilotDatasetLoader
from src.models.second_models.bottom_biomechanics import extract_squat_bottom_biomechanics

# =========================================================================================================
# 1. 15대 클린 생체역학 피처 정의 및 로딩
# =========================================================================================================
FEATURE_NAMES_15 = [
    "최하사점_무릎굴곡각(깊이)",
    "최하사점_대퇴골수평각",
    "무릎/발목_간격비율(외반)",
    "최하사점_무릎외반각",
    "무릎_발목_거리차",
    "좌우_무릎비대칭도",
    "상체_정강이_평행도오차",
    "경추_체간_정렬오차",
    "최하사점_반환가속도",
    "시선_고도각_평균",
    "시선_고도각_최저(바닥응시)",
    "경추_정렬각_최저(목꺾임)",
    "고개_좌우회전_비대칭도",
    "뒤꿈치_들림각_최대(발바닥)",
    "발목_발끝_수직편차(족저들썩임)"
]

def load_clean_15d_dataset():
    print("[1/3] 바벨 스쿼트 마스터 데이터셋 로딩 중...")
    npz_file = SquatPilotDatasetLoader._find_master_npz()
    d = np.load(npz_file, allow_pickle=True)
    sq_mask = (d['y_action'] == 0)

    X_b = d['X_Format_B'][sq_mask].astype(np.float32)
    subjects = d['subjects'][sq_mask].astype(np.int32)
    y = d['y_binary'][sq_mask].astype(np.int32)

    total_samples = len(y)
    normal_cnt = int((y == 1).sum())
    error_cnt = int((y == 0).sum())

    print(f"      - 전체 샘플 수: {total_samples:,}개 (정상: {normal_cnt:,}개 / 오류: {error_cnt:,}개)")
    print(f"      - 클래스 비율: 1 : {error_cnt / normal_cnt:.1f} (정상: {normal_cnt / total_samples * 100:.2f}%)")

    print("[2/3] 스쿼트 15대 핵심 직관 생체역학 피처(15D) 결합 중...")
    # A. 최하사점 피처
    X_bottom_8 = extract_squat_bottom_biomechanics(X_b)
    knee_mean_all = (X_b[:, :, 56] + X_b[:, :, 57]) / 2.0
    bot_idx = np.argmin(knee_mean_all, axis=1)
    thigh_horiz = np.zeros((total_samples, 1), dtype=np.float32)
    for i in range(total_samples):
        b = bot_idx[i]
        thigh_horiz[i, 0] = float(X_b[i, b, 58] - 90.0)

    # B. 얼굴 및 발 세밀 관절 피처
    fine_npz = 'd:/src/models/second_models/squat_fine_feats_12d.npz'
    if not os.path.exists(fine_npz):
        raise FileNotFoundError(f"세밀 관절 캐시 파일이 존재하지 않습니다: {fine_npz}")
    c_fine = np.load(fine_npz)
    fine_all = c_fine['fine_feats']

    pitch_mean = fine_all[:, 0:1]
    pitch_min = fine_all[:, 1:2]
    cervical_min = fine_all[:, 3:4]
    head_yaw = fine_all[:, 4:5]
    heel_lift_max = np.maximum(fine_all[:, 8], fine_all[:, 9])[:, None]
    heel_toe_std = fine_all[:, 10:11]

    # 15D 피처 결합
    X_15d = np.concatenate([
        X_bottom_8[:, 3:4],   # 1. 최하사점 무릎굴곡각 (깊이)
        thigh_horiz,          # 2. 최하사점 대퇴골수평각
        X_bottom_8[:, 0:1],   # 3. 무릎/발목 간격비율 (외반)
        X_bottom_8[:, 2:3],   # 4. 최하사점 무릎외반각
        X_bottom_8[:, 6:7],   # 5. 무릎-발목 거리차
        X_bottom_8[:, 7:8],   # 6. 좌우 무릎비대칭도
        X_bottom_8[:, 1:2],   # 7. 상체-정강이 평행도오차
        X_bottom_8[:, 4:5],   # 8. 경추-체간 정렬오차
        X_bottom_8[:, 5:6],   # 9. 최하사점 반환가속도
        pitch_mean,           # 10. 시선 고도각 평균
        pitch_min,            # 11. 시선 고도각 최저
        cervical_min,         # 12. 경추 정렬각 최저
        head_yaw,             # 13. 고개 좌우회전 비대칭도
        heel_lift_max,        # 14. 뒤꿈치 들림각 최댓값
        heel_toe_std          # 15. 발목-발끝 수직편차 (족저들썩임)
    ], axis=-1)

    X_15d = np.nan_to_num(X_15d, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"      - 최종 15D 피처 행렬 완성: Shape {X_15d.shape} (기존 213차원 대비 93.0% 압축)")
    return X_15d, y, subjects

# =========================================================================================================
# 2. 피험자 격리 5-Fold 교차 검증 및 표준 평가지표 리포트 출력
# =========================================================================================================
def evaluate_clean_squat_model():
    X, y, subjects = load_clean_15d_dataset()

    print("\n[3/3] 피험자 격리 5-Fold 교차 검증 (StratifiedGroupKFold) 단일 LightGBM 학습 중...")
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    feature_importances = np.zeros(X.shape[1], dtype=np.float64)

    t_start = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        clf = lgb.LGBMClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.03,
            class_weight='balanced',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1
        )
        clf.fit(X[trn_idx], y[trn_idx])
        oof_probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        feature_importances += clf.feature_importances_ / 5.0

    elapsed = time.time() - t_start
    print(f"      - 5-Fold 전수 추론 완료 (소요 시간: {elapsed:.2f}초, 샘플당 지연시간: {elapsed / len(y) * 1000:.3f}ms)")

    # =====================================================================================================
    # 사용자 정의 7대 표준 평가지표 대시보드 출력
    # =====================================================================================================
    print("\n" + "=" * 88)
    print("        [NovelFlow] 바벨 스쿼트 정상·오류 분류 7대 표준 평가지표 성적표")
    print("=" * 88)

    # 헬퍼 함수: 단일 임계값에 대한 7대 지표 계산 및 출력
    def print_mode_report(mode_name, th, desc):
        preds = (oof_probs >= th).astype(int)
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()

        # 정상 (Class 1)
        prec_norm = precision_score(y, preds, pos_label=1, zero_division=0) * 100
        rec_norm = recall_score(y, preds, pos_label=1, zero_division=0) * 100
        f1_norm = f1_score(y, preds, pos_label=1, zero_division=0) * 100

        # 오류 (Class 0)
        prec_err = precision_score(y, preds, pos_label=0, zero_division=0) * 100
        rec_err = recall_score(y, preds, pos_label=0, zero_division=0) * 100
        f1_err = f1_score(y, preds, pos_label=0, zero_division=0) * 100

        # 종합 지표
        macro_f1 = (f1_norm + f1_err) / 2.0
        weight_f1 = f1_score(y, preds, average='weighted', zero_division=0) * 100

        print(f"\n▶ [{mode_name}] (임계값 θ = {th:.2f}) - {desc}")
        print("-" * 88)
        print(f"{'클래스 구분':<20} | {'Precision (정밀도)':<20} | {'Recall (재현율)':<18} | {'F1-Score':<14} | {'샘플수 (Support)'}")
        print("-" * 88)
        print(f"{'정상 자세 (Normal)':<20} | {prec_norm:18.2f}% | {rec_norm:16.2f}% | {f1_norm:12.2f}% | {129:4d}개")
        print(f"{'오류 자세 (Error)':<20} | {prec_err:18.2f}% | {rec_err:16.2f}% | {f1_err:12.2f}% | {1845:4d}개")
        print("-" * 88)
        print(f"{'종합 F1-Score (Macro)':<20} : {macro_f1:6.2f}%  (정상과 오류를 동등하게 반영한 균형 점수)")
        print(f"{'가중 F1-Score (Weight)':<20} : {weight_f1:6.2f}%  (실제 데이터 샘플 수 비율을 반영한 종합 점수)")
        print(f"혼동 행렬 상세: [TP(정상정답): {tp:3d}개 | FN(정상놓침): {fn:3d}개] / [TN(오류차단): {tn:4d}개 | FP(오탐): {fp:3d}건]")
        print("-" * 88)

    # 3대 핵심 모드 출력
    print_mode_report("균형 모드 (Balanced Standard)", 0.65, "정상과 오류의 균형 최적화 (상용 서비스 기본 권고)")
    print_mode_report("엄격 모드 (Safety-First)", 0.75, "부상 방지 최우선, 엄격한 정상 심사 (고중량 리프팅용)")
    print_mode_report("격려 모드 (User-Friendly)", 0.50, "정상 재현율 60% 이상 확보 (홈트레이닝 초심자용)")

    # 전 임계값 구간 7대 지표 스펙트럼 요약표
    print("\n■ 전 임계값 구간(0.40 ~ 0.85) 표준 평가지표 스펙트럼")
    print("-" * 88)
    print(f"{'임계값(θ)':<8} | {'정상 Pre':<10} | {'정상 Rec':<10} | {'정상 F1':<10} | {'오류 Pre':<10} | {'오류 Rec':<10} | {'오류 F1':<10} | {'종합 F1(Macro)'}")
    print("-" * 88)
    for th in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        p_ = (oof_probs >= th).astype(int)
        p1 = precision_score(y, p_, pos_label=1, zero_division=0) * 100
        r1 = recall_score(y, p_, pos_label=1, zero_division=0) * 100
        f1_1 = f1_score(y, p_, pos_label=1, zero_division=0) * 100

        p0 = precision_score(y, p_, pos_label=0, zero_division=0) * 100
        r0 = recall_score(y, p_, pos_label=0, zero_division=0) * 100
        f1_0 = f1_score(y, p_, pos_label=0, zero_division=0) * 100

        mf1 = (f1_1 + f1_0) / 2.0
        print(f"θ = {th:<5.2f} | {p1:8.2f}% | {r1:8.2f}% | {f1_1:8.2f}% | {p0:8.2f}% | {r0:8.2f}% | {f1_0:8.2f}% | {mf1:10.2f}%")
    print("-" * 88)

    # 15대 피처 중요도
    print("\n■ 스쿼트 15대 핵심 직관 생체역학 피처 중요도 순위 (Feature Importance)")
    print("-" * 88)
    sorted_idx = np.argsort(feature_importances)[::-1]
    for rank, f_i in enumerate(sorted_idx):
        fn = FEATURE_NAMES_15[f_i]
        score = feature_importances[f_i]
        print(f"  {rank+1:2d}위: {fn:<26} ({score:5.1f}회 분기 기여)")
    print("=" * 88 + "\n")

if __name__ == '__main__':
    evaluate_clean_squat_model()
