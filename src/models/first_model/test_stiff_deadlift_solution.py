# =========================================================================================================
# [스크립트명]: test_stiff_deadlift_solution.py
# [위치]: src/models/first_model/test_stiff_deadlift_solution.py
# [목적]: 스티프 데드리프트 3대 생체역학 솔루션 적용 후 3-Fold 교차 검증 성능 측정 및 시점별 오분류율 개선 분석
# [실행 방법]:
#   - 루트 디렉토리(d:\)에서: python -m src.models.first_model.test_stiff_deadlift_solution
#   - 또는: python src/models/first_model/test_stiff_deadlift_solution.py
# =========================================================================================================

import os
import sys
import time
import json
import numpy as np

# 한글 출력 인코딩 보정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 프로젝트 루트 경로(d:/) 자동 등록 (실행 위치 무관하게 동작 보장)
current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.models.first_model.dataset import EXERCISE_NAMES_KO
from src.models.first_model.metrics_helper import EXERCISE_NAMES_SHORT
from src.models.first_model.train_kfold_ensemble import DualAxisKFoldEnsembleTrainer

def print_banner(title: str):
    print("\n" + "=" * 95)
    print(f"  ★ {title}")
    print("=" * 95)

def main():
    print_banner("스티프 데드리프트 3대 생체역학 솔루션 실측 검증 (Dual-Axis 3-Fold CV)")
    print("[1] 적용된 3대 생체역학 지표:")
    print("    - 솔루션 1: 정강이 지면 수직도 (Shank Verticality Angle & Delta, Knee-Shank Interaction)")
    print("    - 솔루션 2: 고관절-무릎 동조 각속도 상관계수 (Pearson r & RMS Velocity Ratio)")
    print("    - 솔루션 3: 시작 정지자세 기준 상대적 변화량 (Δθ, 피험자 개인차 편향 소거)")
    print("-" * 95)

    # 1. 데이터 로드 및 3대 솔루션 피처 융합
    trainer = DualAxisKFoldEnsembleTrainer()

    # 2. 3-Fold 복합 교차 검증 수행
    t_start = time.time()
    results = trainer.run_3fold_cross_validation()
    total_elapsed = time.time() - t_start

    fold_results = results['fold_results']
    total_summary = results['total_summary']

    # 3. 데드리프트 vs 스티프데드 집중 정밀 분석
    print_banner("데드리프트 (Class 1) vs 스티프 데드리프트 (Class 2) 세부 성능 분석")
    
    dead_f1s = []
    dead_recalls = []
    dead_precs = []
    stiff_f1s = []
    stiff_recalls = []
    stiff_precs = []

    for fr in fold_results:
        # Class 1: 데드리프트
        dead_f1s.append(fr['class_f1s'][1])
        dead_recalls.append(fr['class_recs'][1])
        dead_precs.append(fr['class_precs'][1])
        # Class 2: 스티프 데드리프트
        stiff_f1s.append(fr['class_f1s'][2])
        stiff_recalls.append(fr['class_recs'][2])
        stiff_precs.append(fr['class_precs'][2])

    print(f"{'구분':<20} | {'정밀도 (Precision)':<20} | {'재현율 (Recall)':<20} | {'F1-Score':<15}")
    print("-" * 85)
    print(f"{'02. 바벨 데드리프트':<16} | {np.mean(dead_precs)*100:6.2f}% (±{np.std(dead_precs)*100:.2f}%)   | {np.mean(dead_recalls)*100:6.2f}% (±{np.std(dead_recalls)*100:.2f}%)   | {np.mean(dead_f1s)*100:6.2f}%")
    print(f"{'03. 스티프 데드리프트':<15} | {np.mean(stiff_precs)*100:6.2f}% (±{np.std(stiff_precs)*100:.2f}%)   | {np.mean(stiff_recalls)*100:6.2f}% (±{np.std(stiff_recalls)*100:.2f}%)   | {np.mean(stiff_f1s)*100:6.2f}%")
    print("-" * 85)

    print("\n[Fold별 스티프 데드리프트 성능 추이]")
    for i, fr in enumerate(fold_results, start=1):
        print(f"  - Fold {i} ({fr['fold_name']}): Recall={fr['class_recs'][2]*100:.2f}%, Precision={fr['class_precs'][2]*100:.2f}%, F1={fr['class_f1s'][2]*100:.2f}% (테스트 N={fr['test_samples']}개 중)")

    acc_list = [r['accuracy'] for r in fold_results]
    f1_list = [r['macro_f1'] for r in fold_results]

    print(f"\n[전체 3-Fold 평균]")
    print(f"  * 전체 8종 운동 평균 정확도 (Accuracy): {np.mean(acc_list)*100:.2f}% (±{np.std(acc_list)*100:.2f}%)")
    print(f"  * Macro-F1 Score                     : {np.mean(f1_list)*100:.2f}% (±{np.std(f1_list)*100:.2f}%)")
    print(f"  * 총 소요 시간                       : {total_elapsed:.1f}초")
    print("=" * 95)
    print("✔ 스티프 데드리프트 3대 생체역학 솔루션 검증 완료!\n")

if __name__ == '__main__':
    main()
