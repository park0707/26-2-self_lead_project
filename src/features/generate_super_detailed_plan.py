import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import os

def create_super_detailed_plan_v2(docx_path: str = "d:/자주프 문서/1주차/모델_파이프라인_초정밀_기획안.docx"):
    os.makedirs(os.path.dirname(docx_path), exist_ok=True)
    doc = docx.Document()

    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(18)
        run.font.bold = True
        run.font.color.rgb = RGBColor(24, 43, 73)
        p.paragraph_format.space_after = Pt(6)

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(100, 110, 125)
        p.paragraph_format.space_after = Pt(16)

    def add_h1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(13.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(16, 52, 96)
        p.paragraph_format.space_before = Pt(13)
        p.paragraph_format.space_after = Pt(4)

    def add_h2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 86, 160)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)

    def add_p(text, bold=False, italic=False, color=None):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(9.0)
        run.bold = bold
        run.italic = italic
        if color:
            run.font.color.rgb = color
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(3)
        return p

    def add_bullet(text, bold_prefix="", level=0):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_after = Pt(2)
        if bold_prefix:
            run_b = p.add_run(bold_prefix)
            run_b.font.name = 'Malgun Gothic'
            run_b.font.size = Pt(9.0)
            run_b.bold = True
            run_b.font.color.rgb = RGBColor(40, 50, 60)
        run_t = p.add_run(text)
        run_t.font.name = 'Malgun Gothic'
        run_t.font.size = Pt(9.0)
        return p

    def style_table(table, col_widths, headers, data, align_center_cols=[]):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr_row = table.rows[0]
        for idx, text in enumerate(headers):
            cell = hdr_row.cells[idx]
            cell.text = text
            shading = parse_xml(r'<w:shd {} w:fill="182B49"/>'.format(nsdecls('w')))
            cell._tc.get_or_add_tcPr().append(shading)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = 'Malgun Gothic'
                run.font.size = Pt(8.5)
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)

        for row_idx, row_data in enumerate(data):
            row = table.rows[row_idx + 1]
            fill_color = "F7F9FC" if row_idx % 2 == 1 else "FFFFFF"
            for col_idx, val in enumerate(row_data):
                cell = row.cells[col_idx]
                cell.text = str(val)
                shading = parse_xml(r'<w:shd {} w:fill="{}"/>'.format(nsdecls('w'), fill_color))
                cell._tc.get_or_add_tcPr().append(shading)
                p = cell.paragraphs[0]
                if col_idx in align_center_cols:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    run.font.name = 'Malgun Gothic'
                    run.font.size = Pt(8.0)
                    run.font.color.rgb = RGBColor(40, 40, 40)

        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = width

    # 제목 영역
    add_title("웨이트 트레이닝 3단계 계층형 모델링 및 초정밀 다차원 평가체계 기획안")
    add_subtitle("스포츠 생체역학·시점 불변성·불균형·다중 결함 특화 8대 전문 평가 지표군 및 전수 탐색 파이프라인")

    # 1. 기획 개요
    add_h1("1. 기획 개요 및 평가 체계 고도화 배경")
    add_p("본 문서는 단순 정확도(Accuracy)와 기본 Macro-F1의 한계를 극복하고, 웨이트 트레이닝 인체 관절 데이터의 고유한 도메인 특성(시점 회전 왜곡, 1:17 극심한 불균형, 복합 결함 동시 발생, 생체역학적 부하 연계, 파이프라인 누적 오차)을 엄격하고 다각도로 측정할 수 있는 [초정밀 다차원 평가 체계]를 정의합니다.")

    # 2. 프로젝트 특화 8대 초정밀 평가 지표 체계
    add_h1("2. 웨이트 트레이닝 특화 8대 초정밀 평가 지표 체계 (Evaluation Framework)")
    add_p("각 단계별 목적과 데이터 특성에 맞추어 총 8개 차원의 전문 평가 지표를 도입합니다.")

    eval_headers = ["평가 차원", "전문 지표 명칭 및 수식", "적용 단계", "임계 목표치", "공학적·생체역학적 분석 의미"]
    eval_data = [
        ["1. 시점 불변성\n(View Invariance)", "• 시점 저하율 (PDR):\n  PDR = 1 - (F1_Strict / F1_Train)\n• 시점 불변성 지수 (VIS):\n  VIS = min(F1_views) / max(F1_views)", "1단계, 2단계", "PDR < 25%\nVIS > 0.75", "측면 90도(시점C)에서 양발/양팔이 겹쳐 가려질 때(Self-occlusion) 모델이 얼마나 붕괴하지 않고 형태를 유지하는지 정량 측정"],
        ["2. 인물 일반화\n(Subject Invariance)", "• 피험자 일반화 갭 (SGG):\n  SGG = F1_Train - F1_Subj_Test\n• 피험자 간 성능 분산 (Var_Subj)", "1단계, 2단계", "SGG < 5.0%\nVar < 0.02", "키, 체중, 팔다리 길이 비율 등 체형 차이에 구애받지 않고 새로운 사용자에게 일관된 성능을 보장하는지 측정"],
        ["3. 극단적 불균형\n(1:17 Imbalance)", "• PR-AUC (Precision-Recall AUC)\n• Balanced Accuracy (BAcc)\n• Matthews Corr Coeff (MCC)\n• G-Mean = sqrt(Recall * Specificity)", "2단계\n(정상/오류)", "PR-AUC > 0.45\nBAcc > 85%\nMCC > 0.60", "다수 클래스(오류 94.5%)에 안주하지 않고, 5.5%의 희귀한 정상 자세를 위양성(False Positive) 없이 얼마나 정밀하게 가려내는지 측정"],
        ["4. 세부 결함 일치\n(Multi-Label Match)", "• Subset Accuracy (Exact Match)\n• Hamming Loss (비트 오차율)\n• Multi-Label Jaccard IoU\n• Ranking Loss", "3단계\n(세부 결함)", "Exact > 50%\nH-Loss < 0.15\nIoU > 0.70", "한 동작에 4개 감점 조건이 있을 때 '4개 조건을 전부 완벽히 맞춘 비율' 및 결함 간 교차 일치도 정밀 측정"],
        ["5. 생체역학 위험 연계\n(Kinematic Weight)", "• 주요 위험 구간 가중 오차 (KW-F1):\n  최대 굴곡(Bottom Position) 구간 가중\n• 요추/슬관절 모멘트 암 오차 상관계수", "2단계, 3단계", "KW-F1 > 90%\nCorr(r) > 0.85", "단순 서 있는 구간의 사소한 오차보다 부상 위험이 가장 높은 '바닥 최하점(반동 구간)'에서의 분류 성공률을 2배 가중 평가"],
        ["6. 파이프라인 누적\n(Cascade Propagation)", "• 누적 오차 전달률 (CETR):\n  CETR = e_1 + (1-e_1)*e_2 + ...\n• 종단 간 최종 정확도 (End-to-End F1)", "1->2->3\n파이프라인", "E2E F1 > 88%\nCETR < 12%", "1단계 운동 오분류가 2단계와 3단계로 전파되어 최종 교정 피드백이 엉뚱하게 나가는 연쇄 오류를 종합 제어"],
        ["7. 조기 차단 효율\n(Early-Exit Efficiency)", "• 3단계 스킵 연산 절감률:\n  FLOPs_Saved = Normal_Ratio * FLOPs_M3\n• 조기 종료 정확도 (Early-Exit Precision)", "파이프라인 서빙", "절감률 > 8%\n정밀도 > 95%", "자세가 '정상'인 사용자에게 3단계 연산을 조기 차단(Early Exit)하여 서버 연산 비용과 배터리 소모를 얼마나 아꼈는지 측정"],
        ["8. 실시간 서빙 품질\n(Serving SLA)", "• 추론 지연시간 95/99 백분위수 (P95/P99)\n• 초당 처리량 (Throughput FPS)\n• 모델 메모리 점유량 (RAM MB)", "전체 모델", "P95 < 20ms\nFPS > 60\nRAM < 150MB", "스마트폰, 웹캠, 셋톱박스 등 저사양 온디바이스 엣지 환경에서 프레임 드랍 없이 30 FPS 이상 부드럽게 코칭 가능한지 검증"]
    ]
    t_eval = doc.add_table(rows=len(eval_data)+1, cols=5)
    style_table(t_eval, [Inches(1.2), Inches(1.8), Inches(0.9), Inches(1.1), Inches(2.2)], eval_headers, eval_data, align_center_cols=[0, 2, 3])
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3. 1단계 모델군 및 탐색 공간
    add_h1("3. 1단계: 9종 운동 대분류 확장 모델군 및 탐색 공간")
    add_p("CatBoost, XGBoost, LightGBM, ExtraTrees, TabNet, 1D-CNN+Bi-GRU, TCN, CTR-GCN 등 8대 모델의 초정밀 파라미터 그리드입니다.")
    add_bullet("Tree-based: CatBoost (depth 4~10, l2_reg 1~10), XGBoost (depth 4~10, subsample 0.7~1.0), LightGBM (leaves 15~127)", "GBDT 패밀리: ")
    add_bullet("Deep Tabular: TabNet (n_steps 3~7, gamma 1.2~2.0, lambda 1e-4~1e-3), ResNet-Tabular", "딥 태뷸러: ")
    add_bullet("Deep Sequence: 1D-CNN+Bi-GRU (hidden 64~256, layers 1~3, dropout 0.1~0.4), Temporal TCN (dilations 1,2,4,8)", "시계열 신경망: ")
    add_bullet("Spatial Graph: ST-GCN, CTR-GCN (채널별 동적 위상 정밀화로 측면 90도 PDR < 15% 달성 목표)", "골격 그래프: ")

    # 4. 2단계 불균형 극복 탐색 공간
    add_h1("4. 2단계: 정상/오류 판별 (1:17.0 불균형 극복) 탐색 공간")
    add_bullet("Cost-Sensitive GBDT: scale_pos_weight [5, 10, 17, 25], max_delta_step [1, 3, 5, 7], threshold [0.3~0.6]", "비용 민감 부스팅: ")
    add_bullet("Focal Loss Temporal Model: gamma [1.0~3.0], alpha [0.50~0.85], pos_weight [10~20]", "Focal Loss 신경망: ")
    add_bullet("Supervised Contrastive: SupCon Projection Head (dim 64, 128), Temperature [0.05~0.1], KNN k [5, 11, 21]", "대조 학습 임베딩: ")

    # 5. 3단계 세부 결함 진단 탐색 공간
    add_h1("5. 3단계: 세부 결함 다중 라벨 진단 탐색 공간")
    add_bullet("Classifier Chains: 체인 순서 전략 [해부학적 인과관계(골반->상체), 빈도순, 무작위], max_depth [4, 6, 8]", "분류기 체인: ")
    add_bullet("Multi-Head Bi-GRU with ASL: Asymmetric Loss (gamma_pos 0~1, gamma_neg 2~4, clip 0.05~0.1)", "비대칭 손실 다중 헤드: ")
    add_bullet("Multi-Output ExtraTrees: 250 트리, entropy 불순도 측정, SHAP 원인 관절 기여도 분석", "다중 출력 트리: ")

    # 6. 최적화 실행 전략 및 3중 자동 기록
    add_h1("6. 하이퍼파라미터 최적화 실행 및 3중 자동 기록 체계")
    add_p("Optuna TPE 베이지안 최적화(100 Trial) -> 상위 영역 Focused Grid Search -> Valid Set 조기종료 -> Strict Test 1회 블라인드 공인 평가로 이어집니다.")
    add_bullet("1. Data/피처_데이터/tuning_results/ : Trial별 파라미터, 8대 지표, 혼동행렬 JSON 전수 영구 보관", "원천 데이터 DB: ")
    add_bullet("2. 자주프 문서/1주차/ : Word(.docx) 및 Markdown(.md) 정식 연구 보고서 자동 갱신", "공식 기술 문서: ")
    add_bullet("3. walkthrough.md : 챔피언 선발 결과 및 파레토 프론티어 분석 실시간 기록", "작업 기록: ")

    # 7. [TODO] 계획서 연계 과적합 및 누수 진단 실험
    add_h1("7. [TODO] 연구 계획서 1.8절 연계: 과적합·누수 진단 실험 로드맵")
    add_bullet("타깃 라벨 무작위 셔플 시 성능의 우연 수준(Random Guessing, F1 ~ 11.1%) 붕괴 여부를 확인하여 Shortcut Learning 및 데이터 누수 완전 차단 검증", "1. 라벨 셔플링 대조 실험: ")
    add_bullet("상위 기여 피처가 운동역학적으로 핵심 관절(knee_valgus, torso_inclination 등)인지 정량 해석하고 비정상 좌표 의존성 배제", "2. SHAP & Permutation 타당성: ")
    add_bullet("25% -> 50% -> 75% -> 100% 데이터 증분에 따른 Train/Valid/Strict 수렴 및 과적합 포화도 추적", "3. 학습 곡선(Learning Curve): ")
    add_bullet("피험자 그룹 간 5-Fold CV 수행으로 폴드 간 성능 분산(Var < 0.02) 및 모델 일반화 안정성 정밀 측정", "4. 피험자 5-Fold CV 분산: ")

    # 8. [TODO] 계획서 연계 AI Hub 및 MediaPipe 융합 실험
    add_h1("8. [TODO] 연구 계획서 1.5절·1.3절 연계: AI Hub 및 MediaPipe 융합 실험 로드맵")
    add_bullet("조건 (1) AI Hub 정제 라벨 단독(상한선) vs 조건 (2) AI Hub + MediaPipe 결합 피처(상호 보완성) vs 조건 (3) MediaPipe 추출 단독(실전 서빙 하한선) 3대 조건 정밀 비교", "1. 3대 키포인트 소스 조건 비교: ")
    add_bullet("MediaPipe world_landmarks(미터 단위 3D) vs landmarks(정규화 2D) 입력 차원에 따른 운동 분류 및 결함 판별 성능 격차 분석", "2. 3D 월드 vs 2D 정규화 좌표 대조: ")
    add_bullet("가시성(Visibility) 점수 미달, 화면 밖 이탈, 비정상 좌표 튐(Jitter) 감지 및 카메라 시점별 프레임 결측률 통계 체계화", "3. 품질 필터링 및 시점별 결측 집계: ")

    doc.save(docx_path)
    print(f"[+] 8대 초정밀 평가체계 및 TODO 로드맵 반영 Word 기획안 생성 완료: {docx_path}")

if __name__ == '__main__':
    create_super_detailed_plan_v2()
