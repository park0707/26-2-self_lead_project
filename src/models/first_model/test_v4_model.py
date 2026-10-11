# =========================================================================================================
# [스크립트명]: test_v4_model.py
# [위치]: src/models/first_model/test_v4_model.py
# [목적]: 1단계 8종 운동 분류 - v4.0 독립 2중 캐스케이드(Dual-Cascade) 및 데이터 정제 시스템 원클릭 검증기
# [실행 방법]:
#   - 루트 디렉토리(d:\)에서: python -m src.models.first_model.test_v4_model
#   - 또는 직접 실행: python src/models/first_model/test_v4_model.py
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

from src.models.first_model.train_v4_model import V4HierarchicalPipelineTrainer

def main():
    print("\n" + "=" * 115)
    print("  ★ [NovelFlow] v4.0 독립 2중 캐스케이드(Dual-Cascade) 3D 운동 분류 앙상블 시스템")
    print("  ★ (데이터셋 라벨 노이즈 정제 + 1차 독립 2중 정자세 감별 + 2차 14D 생체역학 리졸버)")
    print("===================================================================================================================")
    print("  1. Stage 1: 6대 상위 기능 그룹 3중 하이브리드 앙상블 (Bi-GRU 0.65 + XGB 0.20 + Cat 0.15) [기존 v3.0 계승]")
    print("     - Group 0: 바벨 스쿼트, Group 1: 데드리프트 패밀리(02+03), Group 2: 바벨 런지(04, 단독)")
    print("     - Group 3: 무빙 런지 패밀리(05+07), Group 4: 사이드 런지(06), Group 5: 굿모닝(08)")
    print("  2. Stage 2-A (v4.0 신규 탑재): 데드리프트 전용 독립 2중 캐스케이드 Expert")
    print("     - [데이터 정제]: 훈련셋에서 피험자 1(Subj_1)의 스티프 데드(0.15° 오차 노이즈) 및 Level 4 바벨 역전 데이터 자동 배제")
    print("     - [1차 감별사]: 모델 1A(바벨 정자세 P_B) & 모델 1B(스티프 정자세 P_S) 독립 확률 추론")
    print("     - [2차 리졸버]: 둘 다 낮아(Neither) 판정을 유보한 모호군(약 35%)을 위임받아 14D 생체역학 지표로 미세 분기")
    print("  3. Stage 2-B: 무빙 런지 패밀리 전용 2진 Expert (15D 비폐색 프록시 피처 + OOF 임계값 튜닝) [기존 v3.0 계승]")
    print("  4. Decision: 3-Fold Out-of-Domain(인물 3그룹 x 시점 3그룹) 전수 평가 및 v3.0 대비 성적 변화 대조")
    print("-" * 115)

    trainer = V4HierarchicalPipelineTrainer()
    trainer.run_hierarchical_3fold_cv()

if __name__ == '__main__':
    main()
