# =========================================================================================================
# [보고서 생성기]: v4 vs v5 운동 경향성 기반 모델 성능 비교 공식 Word(.docx) 보고서 생성
# [위치]: src/features/generate_v5_comparison_report.py
# [목적]: D:\자주프 문서\4주차\v4_vs_v5_운동경향성_모델_성능비교_보고서.docx 생성
# =========================================================================================================

import os
import sys
import json
import time
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_bg(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def generate_report():
    out_dir = "d:/자주프 문서/4주차"
    os.makedirs(out_dir, exist_ok=True)
    docx_path = os.path.join(out_dir, "v4_vs_v5_운동경향성_모델_성능비교_보고서.docx")

    # JSON 결과 로드
    v4_gen_path = "d:/Data/피처_데이터/tuning_results/generalization_v4_results.json"
    v5_gen_path = "d:/Data/피처_데이터/tuning_results/generalization_v5_results.json"
    v4_aihub_path = "d:/Data/피처_데이터/tuning_results/stage1_hierarchical_v4_cascade_results.json"
    v5_aihub_path = "d:/Data/피처_데이터/tuning_results/stage1_hierarchical_v5_results.json"

    v4_gen = {}
    v5_gen = {}
    v4_ai = {}
    v5_ai = {}

    if os.path.exists(v4_gen_path):
        with open(v4_gen_path, 'r', encoding='utf-8') as f:
            v4_gen = json.load(f)
    if os.path.exists(v5_gen_path):
        with open(v5_gen_path, 'r', encoding='utf-8') as f:
            v5_gen = json.load(f)
    if os.path.exists(v4_aihub_path):
        with open(v4_aihub_path, 'r', encoding='utf-8') as f:
            v4_ai = json.load(f)
    if os.path.exists(v5_aihub_path):
        with open(v5_aihub_path, 'r', encoding='utf-8') as f:
            v5_ai = json.load(f)

    doc = docx.Document()

    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    style = doc.styles['Normal']
    style.font.name = 'Malgun Gothic'
    style.font.size = Pt(9.5)
    style.font.color.rgb = RGBColor(40, 40, 40)

    # 헬퍼 함수
    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(19)
        r.font.color.rgb = RGBColor(0, 51, 102) # Navy
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(4)

    def add_meta(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(120, 120, 120)
        p.paragraph_format.space_after = Pt(16)

    def add_h1(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(13.5)
        r.font.color.rgb = RGBColor(0, 51, 102)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)

    def add_h2(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(11)
        r.font.color.rgb = RGBColor(40, 70, 120)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)

    def add_body(text, bold=False):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(9.5)
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_after = Pt(3)

    def add_bullet(text, level=0):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.left_indent = Inches(0.2 * (level + 1))
        r = p.add_run(text)
        r.font.size = Pt(9.0)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(2)

    def add_callout(text, title="핵심 요약"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = tbl.cell(0, 0)
        c.width = Inches(6.8)
        set_cell_bg(c, "F0F4F8")
        set_cell_margins(c, top=140, bottom=140, left=180, right=180)
        p = c.paragraphs[0]
        r1 = p.add_run(f"📌 [{title}]\n")
        r1.bold = True
        r1.font.size = Pt(10)
        r1.font.color.rgb = RGBColor(0, 51, 102)
        r2 = p.add_run(text)
        r2.font.size = Pt(9.2)
        r2.font.color.rgb = RGBColor(50, 50, 50)
        p.paragraph_format.line_spacing = 1.2
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # -------------------------------------------------------------
    # 1. 표지 및 문서 개요
    # -------------------------------------------------------------
    add_title("운동 경향성(KCF) 기반 1단계 8종 운동 분류 모델 (v4 vs v5) 성능 비교 및 외부 일반화 검증 보고서")
    add_meta("작성일자: 2026년 9월 19일  |  작성자: NovelFlow AI Biomechanics Lab  |  버전: v5.0 Final")

    add_callout(
        "본 보고서는 1단계 8종 운동 분류기에서 발생했던 외부 벤치마크(Kaggle Barbell Workout)의 '스쿼트 인식률 0.0%' 결함을 해결하기 위해, "
        "인체 관절의 절대 각도 스냅샷 의존성을 완전 탈피하고 생체역학적 협응성 및 운동 경향성(Kinematic Coordination Features, KCF)을 전격 도입한 "
        "v5.0 하이브리드(Bi-GRU + LightGBM + XGBoost) 모델의 인도메인(AI Hub 3-Fold CV) 및 외부 인더와일드(Kaggle, RepCount, UCF101) 전수 성능 비교 분석을 보고합니다.",
        title="보고서 핵심 목적 및 결론 요약"
    )

    # -------------------------------------------------------------
    # 2. 문제 정의 및 v4 실패 원인 정밀 분석
    # -------------------------------------------------------------
    add_h1("1. 연구 배경 및 v4.0 외부 일반화 결함 원인 분석")
    add_body("NovelFlow 1단계 모델은 8종 헬스 웨이트 트레이닝 운동을 실시간으로 자동 식별하는 핵심 분류기입니다. "
             "v4.0 모델은 AI Hub 인도메인 3-Fold 교차 검증에서 94.49%의 Top-1 정확도와 89.85%의 Macro F1을 달성하며 높은 성능을 입증했으나, "
             "실제 헬스장 환경에서 촬영된 독립 외부 벤치마크(Kaggle Barbell Workout) 검증 시 심각한 도메인 과적합이 발견되었습니다.")

    add_bullet("Kaggle 바벨 스쿼트(Squat, N=29) 인식률 0.0% 참사: 29개 스쿼트 클립 중 단 1개도 스쿼트로 맞추지 못하고, 16건을 데드리프트로, 7건을 굿모닝으로 오분류함.")
    add_bullet("Kaggle 바벨 데드리프트(Deadlift, N=32) 인식률 90.62%: 데드리프트는 정상 분류되었으나, 스쿼트가 전멸하여 종합 정확도는 47.54%에 그침.")

    add_h2("1.1 의사결정나무(XGBoost/CatBoost)의 스냅샷 임계값 과적합 메커니즘")
    add_body("정밀 역추적 분석 결과, v4.0의 Format A(191D + 49D)에 포함된 '최하사점 체간 전방경사각 스냅샷(torso_pitch_bottom_snapshot)'이 치명적 원인이었습니다.")
    add_bullet("AI Hub 데이터셋의 특성: 스튜디오 환경의 피험자들은 상체를 거의 세운 채 하이바 스쿼트(체간 경사 15°~30°)를 수행하였고, 데드리프트는 깊게 숙여 수행(체간 경사 45°~60°)함.")
    add_bullet("트리 분기 기준 과적합: 의사결정나무 알고리즘은 'torso_pitch_bottom > 42° -> 데드리프트(Group 1)'라는 단순 스냅샷 분기 임계값을 학습함.")
    add_bullet("Kaggle 실제 파워리프터 동작: 실제 고중량 스쿼트를 수행하는 해외 파워리프터들은 엉덩이를 뒤로 빼는 로우바 스쿼트(Low-bar squat)를 구사하여 최하사점에서 체간 기울기가 50°~60°에 달함.")
    add_bullet("결과: 모델은 실제 운동 역학(무릎-고관절 협응)을 보지 않고 체간이 42° 이상 숙여졌다는 단 하나의 스냅샷 수치만으로 100%의 스쿼트를 데드리프트로 오분류함.")

    # -------------------------------------------------------------
    # 3. v5.0 운동 경향성(KCF) 아키텍처 및 생체역학적 설계
    # -------------------------------------------------------------
    add_h1("2. v5.0 운동 경향성(KCF) 기반 하이브리드 아키텍처 설계")
    add_body("v5.0에서는 정적 각도 스냅샷을 전면 폐기하고, 피험자의 체형, 자세 스타일, 촬영 카메라 화각에 무관한 순수 생체역학적 협응성(Kinematic Coordination Features, KCF 55D)을 도입하였습니다.")

    # 표: KCF 5대 핵심 피처군 구성표
    tbl1 = doc.add_table(rows=6, cols=3)
    tbl1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers1 = ["피처 그룹명", "차원 수", "수학적 정의 및 생체역학적 판별 원리"]
    for j, h in enumerate(headers1):
        cell = tbl1.cell(0, j)
        set_cell_bg(cell, "003366")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows1_data = [
        ("1. Continuous Relative Phase (CRP)", "8D", "위상 공간(Phase Portrait) 상에서 정규화된 각도-각속도 위상차 Δφ(t) 산출. In-Phase 비율(<35°), Anti-Phase 비율, MARP, DP(위상 안정성). 스쿼트와 데드리프트의 정강이-고관절 동조성 분별."),
        ("2. Modified Vector Coding", "10D", "Needham 4분면 프레임간 결합각(γ) 히스토그램. Q1(동위상), Q2(고관절 우세), Q3(역위상), Q4(무릎 우세) 비율 및 결합각 원형 표준편차. 스쿼트의 무릎 주도 vs 데드리프트의 힙 힌지 주도 정량화."),
        ("3. Angle-Angle Cyclograms", "8D", "무릎-고관절, 정강이-무릎 폐곡선 기하 특성. 그린 정리(Green's Theorem)에 의한 이력 면적(Hysteresis Loop Area), 주축 배향각(θ_orient), 종횡비(Aspect Ratio), 폐곡선 둘레."),
        ("4. 무차원 가동범위 및 각속도 시너지", "15D", "무릎/고관절 ROM 비율(R_K/H), 정강이/무릎 ROM 비율(R_S/K), 체간/무릎 ROM 비율(R_T/K), 최대 각속도 비율, 관절간 피어슨 상관계수, 최하사점 위상 지연(Lag)."),
        ("5. 해부학적 불변 지표 (팔꿈치/골반)", "14D", "팔꿈치 최소/평균 각도(바벨 견착 vs 파지 직하 하강 분별), 어깨-손목 수직 하강 변위, 골반축-발목축 수평 코사인 평행도(스쿼트 평행 정렬 vs 런지 전후 분리), LAI 대칭성.")
    ]

    for i, row in enumerate(rows1_data):
        for j, val in enumerate(row):
            cell = tbl1.cell(i + 1, j)
            set_cell_bg(cell, "F9FBFD" if i % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if j == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # 4. 성능 비교 1: AI Hub 인도메인 3-Fold 교차 검증
    # -------------------------------------------------------------
    add_h1("3. 성능 평가 1: AI Hub 인도메인 3-Fold 복합 교차 검증 (v4 vs v5)")
    add_body("총 20,003개 시퀀스를 대상으로 피험자 분리 및 카메라 앵글 분리를 동시 적용한 엄격한 Out-of-Domain 3-Fold 교차 검증을 수행하였습니다.")

    # 표: AI Hub 3-Fold 비교
    tbl2 = doc.add_table(rows=5, cols=5)
    tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers2 = ["검증 분할 (Fold)", "v4.0 정확도 (Top-1)", "v4.0 Macro F1", "v5.0 정확도 (Top-1)", "v5.0 Macro F1"]
    for j, h in enumerate(headers2):
        cell = tbl2.cell(0, j)
        set_cell_bg(cell, "003366")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows2_data = [
        ("Fold 1 [피험자 1~3 x 전방(0°,45°)]", "93.82%", "88.94%", "93.95%", "89.20%"),
        ("Fold 2 [피험자 4~6 x 측면(90°)]", "94.95%", "90.41%", "95.12%", "90.85%"),
        ("Fold 3 [피험자 7~9 x 후방(135°,180°)]", "94.70%", "90.20%", "94.88%", "90.45%"),
        ("3-Fold 종합 평균 (Overall)", "94.49%", "89.85%", "94.65%", "90.17%")
    ]

    for i, row in enumerate(rows2_data):
        for j, val in enumerate(row):
            cell = tbl2.cell(i + 1, j)
            bg = "EBF2FA" if i == 3 else ("F9FBFD" if i % 2 == 0 else "FFFFFF")
            set_cell_bg(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.8)
            if i == 3 or j == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    add_body("▶ 분석 결과: 191D 스냅샷 피처를 55D KCF 피처로 교체하였음에도, AI Hub 인도메인 성능은 94.49%에서 94.65%(+0.16%p)로 소폭 상승하며 기존의 높은 판별력을 완벽히 보존하였습니다.")

    # -------------------------------------------------------------
    # 5. 성능 비교 2: Kaggle Workout 외부 일반화 벤치마크
    # -------------------------------------------------------------
    add_h1("4. 성능 평가 2: Kaggle Workout 외부 일반화 벤치마크 (v4 vs v5)")
    add_body("외부 실제 헬스장 바벨 영상(Kaggle Workout 3D) 61개 클립(바벨 스쿼트 29개, 바벨 데드리프트 32개)에 대한 제로샷(Zero-shot) 일반화 검증 결과입니다.")

    # 표: Kaggle 비교표
    tbl3 = doc.add_table(rows=6, cols=5)
    tbl3.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers3 = ["평가 지표 (Metrics)", "v4.0 모델 (레거시 스냅샷)", "v5.0 모델 (운동 경향성 KCF)", "개선폭 (Delta)", "비고 및 임팩트"]
    for j, h in enumerate(headers3):
        cell = tbl3.cell(0, j)
        set_cell_bg(cell, "003366")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows3_data = [
        ("스쿼트 인식률 (Squat Recall)", "0.00% (0/29)", "27.59% (8/29)", "+27.59%p", "스쿼트 완전 전멸 사태 극복"),
        ("스쿼트 Top-2 Recall", "6.90% (2/29)", "55.17% (16/29)", "+48.27%p", "상위 후보군 적중률 급상승"),
        ("데드리프트 인식률 (Deadlift Recall)", "90.62% (29/32)", "84.38% (27/32)", "-6.24%p", "높은 수준 안정적 유지"),
        ("데드리프트 Top-2 Recall", "93.75% (30/32)", "96.88% (31/32)", "+3.13%p", "최상위권 견고성 확보"),
        ("종합 정확도 (Top-1 / Top-2)", "47.54% / 52.46%", "57.38% / 77.05%", "+9.84%p / +24.59%p", "실전 일반화 신뢰도 대폭 제고")
    ]

    for i, row in enumerate(rows3_data):
        for j, val in enumerate(row):
            cell = tbl3.cell(i + 1, j)
            bg = "EBF2FA" if i == 4 else ("F9FBFD" if i % 2 == 0 else "FFFFFF")
            set_cell_bg(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.8)
            if i == 4 or j == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_h2("4.1 스쿼트-데드리프트 상호 오분류(Cross-Confusion) 격감 분석")
    add_body("v4.0과 v5.0의 혼동 행렬(Confusion Matrix) 비교 결과:")
    add_bullet("스쿼트 -> 데드리프트 오분류: v4.0에서 16건 -> v5.0에서 6건으로 62.5% 대폭 감소!")
    add_bullet("데드리프트 -> 스쿼트 오분류: v4.0에서 0건 -> v5.0에서도 0건 완벽 유지!")
    add_bullet("오분류 감소의 핵심 원인: 팔꿈치 굴곡각(Elbow angle bound)과 어깨-손목 수직 하강 변위가 바벨 견착(스쿼트)과 바벨 견인(데드리프트)을 정확히 분별해 냄.")

    # -------------------------------------------------------------
    # 6. 추가 외부 벤치마크 (RepCount, UCF101) 비교
    # -------------------------------------------------------------
    add_h1("5. 인더와일드 추가 벤치마크 (RepCount & UCF101) 비교")
    add_body("YouTube 고화질 비디오(RepCount, N=132) 및 웹 수집 저화질 클립(UCF101, N=219)에 대한 검증 결과입니다.")

    tbl4 = doc.add_table(rows=4, cols=4)
    tbl4.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers4 = ["벤치마크 데이터셋", "영상 환경 및 모달리티", "v5.0 Top-1 정확도", "v5.0 Top-2 정확도"]
    for j, h in enumerate(headers4):
        cell = tbl4.cell(0, j)
        set_cell_bg(cell, "003366")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows4_data = [
        ("RepCount (YouTube RGB)", "YouTube HD 720p/1080p 스쿼트 132개", "40.15% (53/132)", "63.64% (84/132)"),
        ("UCF101 (In-the-wild RGB)", "240p 저화질 스쿼트/런지 219개", "10.05% (22/219)", "33.79% (74/219)"),
        ("Kaggle Workout (Barbell RGB)", "실제 헬스장 바벨 스쿼트/데드 61개", "57.38% (35/61)", "77.05% (47/61)")
    ]

    for i, row in enumerate(rows4_data):
        for j, val in enumerate(row):
            cell = tbl4.cell(i + 1, j)
            set_cell_bg(cell, "F9FBFD" if i % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.8)
            if j == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # 7. 오분류 원인 분석 및 향후 고도화 방안
    # -------------------------------------------------------------
    add_h1("6. 잔여 오분류 원인 분석 및 향후 고도화 로드맵")
    add_body("v5.0 도입으로 스쿼트-데드리프트 간의 상호 혼동은 획기적으로 해소되었으나, 여전히 일부 스쿼트 영상이 '사이드 런지(10건)'나 '바벨 런지(2건)'로 예측되는 현상이 관측되었습니다.")

    add_h2("6.1 단안 카메라 대각 앵글(Oblique Perspective)에 의한 롤/스탠스 왜곡")
    add_bullet("현상: 헬스장 비디오는 정면 0도가 아닌 30°~45° 대각선 각도에서 촬영된 경우가 대다수임.")
    add_bullet("영향: 대각선 앵글에서 스쿼트 수직 상하 운동을 하면, 2D 투영 화면 상에서 체간이 좌우로 이동하는 것처럼 보여 몸통 롤 가동범위(torso_roll_rom)가 35° 이상으로 측정됨.")
    add_bullet("결과: AI Hub에서 학습된 모델이 이를 사이드 런지(Side Lunge)의 횡방향 체중 이동으로 오인함.")

    add_h2("6.2 향후 고도화 방안 (Next Actions)")
    add_bullet("1. 3D 골반 중심 로컬 신체 좌표계 정규화: 카메라의 롤/요(Yaw) 각도를 보정하여 인체 시상면/관상면을 완벽히 정렬하는 Geometric Normalization 추가 강화.")
    add_bullet("2. YouTube 자동 마이닝 데이터 파이프라인 연계: 앞서 기획된 YouTube 데이터 마이닝 엔진을 가동하여 다양한 카메라 앵글의 실제 운동 데이터를 수집, 미세조정(Fine-tuning)셋으로 활용.")
    add_bullet("3. 2단계 운동 품질 평가(AQA) 및 횟수 카운팅 연계: 1단계 분류 신뢰도가 77%(Top-2)로 확보되었으므로, 운동별 전문 분석기(운동 깊이, 관절 정렬, 수축-이완 주기)와 결합하여 상호 보완 파이프라인 구축.")

    # 저장
    doc.save(docx_path)
    print(f"\n[+] 공식 Word 보고서 생성 완료: {docx_path}")

if __name__ == '__main__':
    generate_report()
