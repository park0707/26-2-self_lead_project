# =========================================================================================================
# [스크립트명]: test_hierarchical_solution.py
# [위치]: src/models/first_model/test_hierarchical_solution.py
# [목적]: 1단계 8종 운동 분류 - 2-Stage 계층형(Hierarchical) 앙상블 시스템 원클릭 3-Fold 학습 및 평가
# [실행 방법]:
#   - 루트 디렉토리(d:\)에서: python -m src.models.first_model.test_hierarchical_solution
#   - 또는: python src/models/first_model/test_hierarchical_solution.py
# =========================================================================================================

import os
import sys

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

from src.models.first_model.train_hierarchical_kfold import HierarchicalKFoldPipelineTrainer

def main():
    print("\n" + "=" * 115)
    print("  ★ [NovelFlow] 2-Stage 계층형(Hierarchical) 3D 운동 분류 앙상블 시스템 (공인 1단계 확정 모델)")
    print("===================================================================================================================")
    print("  1. Stage 1: 6대 상위 기능 그룹 3중 하이브리드 앙상블 (Bi-GRU 0.65 + XGB 0.20 + Cat 0.15)")
    print("     - Group 0: 바벨 스쿼트, Group 1: 데드리프트 패밀리(02+03), Group 2: 바벨 런지(04, 단독)")
    print("     - Group 3: 무빙 런지 패밀리(05+07), Group 4: 사이드 런지(06), Group 5: 굿모닝(08)")
    print("  2. Stage 2-A: 데드리프트 전용 2중 하이브리드 앙상블 (하체 7-Joint 26D Bi-GRU + 11D 생체역학 LightGBM)")
    print("  3. Stage 2-B: 무빙 런지 패밀리 전용 2진 Expert (15D 3D가림/프록시 피처 + 지능형 증강 + OOF 임계값 튜닝)")
    print("  4. Decision: 소프트 베이지안 결합(Soft Bayesian Marginalization)을 통한 8종 최종 통합")
    print("-" * 115)

    trainer = HierarchicalKFoldPipelineTrainer()
    trainer.run_hierarchical_3fold_cv()

if __name__ == '__main__':
    main()
