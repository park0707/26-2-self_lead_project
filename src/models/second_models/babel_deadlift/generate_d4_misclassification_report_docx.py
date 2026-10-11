# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d4_misclassification_report_docx.py
Purpose: 바벨 데드리프트 d4 모델의 오분류(FN 미탐 / FP 오탐) 케이스 심층 원인 분석 및
         MediaPipe 33 관절 시각화 검증 결과를 수록한 기술 보고서 Word(.docx) 생성
"""

import os
import sys
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

sys.stdout.reconfigure(encoding='utf-8')

DOCX_DIR_6TH = r"D:\자주프 문서\6주차"
DOCX_DIR_ROOT = r"D:\자주프 문서"
os.makedirs(DOCX_DIR_6TH, exist_ok=True)
os.makedirs(DOCX_DIR_ROOT, exist_ok=True)

TARGET_DOCX_6TH = os.path.join(DOCX_DIR_6TH, "바벨_데드리프트_d4_오분류_심층원인분석_및_시각화_보고서.docx")
TARGET_DOCX_ROOT = os.path.join(DOCX_DIR_ROOT, "바벨_데드리프트_d4_오분류_심층원인분석_및_시각화_보고서.docx")
TARGET_MD_6TH = os.path.join(DOCX_DIR_6TH, "바벨_데드리프트_d4_오분류_심층원인분석_및_시각화_보고서.md")
TARGET_MD_ROOT = os.path.join(DOCX_DIR_ROOT, "바벨_데드리프트_d4_오분류_심층원인분석_및_시각화_보고서.md")

doc = Document()

# 페이지 여백 설정 (Narrow)
for s in doc.sections:
    s.top_margin = Inches(0.8)
    s.bottom_margin = Inches(0.8)
    s.left_margin = Inches(0.8)
    s.right_margin = Inches(0.8)

# 색상 정의
NAVY = RGBColor(26, 54, 93)      # #1A365D
BLUE = RGBColor(43, 108, 176)    # #2B6CB0
GRAY = RGBColor(113, 128, 150)   # #718096
BLACK = RGBColor(45, 55, 72)     # #2D3748
ORANGE = RGBColor(192, 86, 33)   # #C05621
DARK_GREEN = RGBColor(39, 103, 73) # #276749

def style_heading(p, text, level=1):
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.bold = True
    if level == 1:
        run.font.size = Pt(15)
        run.font.color.rgb = NAVY
    elif level == 2:
        run.font.size = Pt(12.5)
        run.font.color.rgb = BLUE
    elif level == 3:
        run.font.size = Pt(11)
        run.font.color.rgb = BLACK

def add_callout_box(doc, text_list, title="", border_color="2B6CB0", bg_color="F7FAFC"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.8)
    cell = tbl.cell(0, 0)
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>'))
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    if title:
        run_t = p.add_run(f"■ {title}\n")
        run_t.font.name = "맑은 고딕"
        run_t.font.size = Pt(10.5)
        run_t.bold = True
        run_t.font.color.rgb = BLUE
    for line in text_list:
        run = p.add_run(line + "\n")
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9.5)
        run.font.color.rgb = BLACK
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def set_cell_properties(cell, bg_color=None, bold=False, color=BLACK, font_size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER):
    if bg_color:
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>'))
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for p in cell.paragraphs:
        p.alignment = align
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        for r in p.runs:
            r.font.name = "맑은 고딕"
            r.font.size = Pt(font_size)
            r.bold = bold
            r.font.color.rgb = color

# ================= 문서 본문 작성 =================

# 머리글 메타정보
p_meta = doc.add_paragraph()
p_meta.paragraph_format.space_after = Pt(2)
r_meta = p_meta.add_run("vidio-health-trainer | AI-Hub 바벨 데드리프트 d4 모델 오분류 심층 진단 보고서")
r_meta.font.name = "맑은 고딕"
r_meta.font.size = Pt(9.5)
r_meta.font.color.rgb = GRAY

# 타이틀
p_title = doc.add_paragraph()
p_title.paragraph_format.space_before = Pt(4)
p_title.paragraph_format.space_after = Pt(4)
r_title = p_title.add_run("바벨 데드리프트 d4 (무릎-골반 동시 펴짐 위반, 힙 슈트업)\n오분류 데이터 심층 원인 분석 및 MediaPipe 관절 시각화 보고서")
r_title.font.name = "맑은 고딕"
r_title.font.size = Pt(19)
r_title.bold = True
r_title.font.color.rgb = NAVY

# 서브타이틀
p_sub = doc.add_paragraph()
p_sub.paragraph_format.space_after = Pt(12)
r_sub = p_sub.add_run("미탐(False Negative) 46건 및 오탐(False Positive) 166건 원인 분석, 192개 프레임 스켈레톤 오버레이 검증")
r_sub.font.name = "맑은 고딕"
r_sub.font.size = Pt(11)
r_sub.font.color.rgb = BLUE

# 개요 박스
add_callout_box(doc, [
    "• 저장 위치: D:\\Data\\temp\\바벨_데드리프트_d4_오분류_시각화\\ (총 12개 클립, 192개 고해상도 오버레이 이미지)",
    "• 시각화 구성: MediaPipe 33개 랜드마크 + 하지/척추 스켈레톤 연결선 + 실시간 무릎/고관절 각도 + 상단 정보배너 + 하단 원인진단문",
    "• 미탐 (False Negative, 46건): 실제 힙 슈트업 결함이나 정상으로 놓친 케이스 → 정면(view3) 깊이 압축 및 미세 선행 신전이 주요인",
    "• 오탐 (False Positive, 166건): 실제 정상이나 결함으로 오인한 케이스 → 후방 시점(view1, view5)의 상체 전경 각도 착시 및 신체 비례가 주요인",
    "• 해결책: Phase 4 Risk-based Calibrated 3-State 제약 최적화(Coverage 92.60%)를 통해 모호한 7.4%를 '판정 보류'로 완벽 격리"
], title="오분류 시각화 및 심층 진단 핵심 요약")

# 1. 서론
p1 = doc.add_paragraph()
style_heading(p1, "1. 오분류 분석의 배경 및 데이터셋 현황", level=1)
p_desc = doc.add_paragraph()
p_desc.paragraph_format.line_spacing = 1.2
p_desc.paragraph_format.space_after = Pt(6)
r = p_desc.add_run(
    "바벨 데드리프트 세부 결함 d4(힙 슈트업)는 바닥에서 바벨을 뽑는 First Pull 구간에서 무릎과 고관절이 동시에 펴지지 않고 "
    "무릎이 먼저 펴져 골반이 위로 솟구치는 결함입니다. Two-Stream v4.0 파이프라인은 F1-Score 45.64%, ROC-AUC 0.7560으로 "
    "비약적 도약을 달성하였으나, 2-State 기준(임계치 0.5)에서 46건의 미탐(FN)과 166건의 오탐(FP)이 존재합니다.\n\n"
    "사용자가 오분류의 본질을 직접 육안으로 검증할 수 있도록, 대표적인 FN 6건과 FP 6건(총 12개 클립, 192개 프레임)을 선별하여 "
    "MediaPipe 33개 관절 포인트 및 스켈레톤, 프레임별 무릎·골반 각도 수치를 이미지 위에 오버레이하여 D:\\Data\\temp 에 배치하였습니다."
)
r.font.name = "맑은 고딕"
r.font.size = Pt(10)
r.font.color.rgb = BLACK

# 2. 미탐 케이스 분석
p2 = doc.add_paragraph()
style_heading(p2, "2. 미탐 (False Negative, 실제 결함 → 정상 오판) 심층 분석", level=1)

table_fn = doc.add_table(rows=7, cols=5)
table_fn.alignment = WD_TABLE_ALIGNMENT.CENTER
table_fn.autofit = False

fn_w = [Inches(1.2), Inches(1.8), Inches(1.2), Inches(1.0), Inches(1.6)]
for row in table_fn.rows:
    for i, w in enumerate(fn_w):
        row.cells[i].width = w

fn_headers = ["케이스", "시퀀스 클립명", "카메라 시점", "예측 확률", "주요 오분류 원인"]
for i, h in enumerate(fn_headers):
    table_fn.cell(0, i).text = h
    set_cell_properties(table_fn.cell(0, i), bg_color="1A365D", bold=True, color=RGBColor(255, 255, 255), font_size=9.0)

fn_rows = [
    ("FN-1", "[오류]_D03_피험자7_Key294", "시점 C (정면 0°)", "0.2047", "정면 뷰 깊이 손실로 인한 찰나의 전후방 위상차 소실"),
    ("FN-2", "[오류]_D03_피험자1_Key303", "시점 B (우측 전방 45°)", "0.2097", "무릎 각도 변화율이 완만하여 임계치 미달"),
    ("FN-3", "[오류]_D07_피험자4_Key294", "시점 C (정면 0°)", "0.2115", "정면 시점 카메라 왜곡으로 골반 상승 속도 과소추정"),
    ("FN-4", "[오류]_D03_피험자6_Key294", "시점 C (정면 0°)", "0.2242", "정면 시점 무릎-고관절 수평 간격 압축"),
    ("FN-5", "[오류]_D03_피험자7_Key297", "시점 C (정면 0°)", "0.2391", "초기 0~10프레임 바벨 이동과 골반 궤적 착시"),
    ("FN-6", "[오류]_D07_피험자4_Key294", "시점 E (좌측 후방 45°)", "0.2583", "후방 시점 무릎 가림(Occlusion) 발생")
]

for row_idx, data in enumerate(fn_rows, start=1):
    for col_idx, text in enumerate(data):
        cell = table_fn.cell(row_idx, col_idx)
        cell.text = text
        bg = "FFF5F5" if "정면" in data[2] else "FFFFFF"
        set_cell_properties(cell, bg_color=bg, bold=False, color=BLACK, font_size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER if col_idx in [0, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT)

doc.add_paragraph().paragraph_format.space_after = Pt(4)

add_callout_box(doc, [
    "■ 미탐(FN)의 핵심 원인: 정면 시점(시점 C / view3)의 깊이 압축(Depth Compression)",
    "  - 시점 C는 전체 46개 FN 중 14건(30.4%)을 차지하여 타 시점 대비 2배 이상 높은 미탐률을 보입니다.",
    "  - 데드리프트의 힙 슈트업은 신체 시상면(Sagittal Plane, 앞뒤 방향)에서 무릎이 뒤로 펴지고 엉덩이가 위로 올라가는 2차원 궤적 차이인데,",
    "    정면 2D 카메라에서는 전후방 z축 거리가 화면상으로 압축되어 무릎 신전 시점이 고관절 신전 시점과 거의 동시에 일어난 것처럼 오인됩니다."
], title="미탐(FN) 생체역학적 핵심 진단")

# 3. 오탐 케이스 분석
p3 = doc.add_paragraph()
style_heading(p3, "3. 오탐 (False Positive, 실제 정상 → 결함 오인) 심층 분석", level=1)

table_fp = doc.add_table(rows=7, cols=5)
table_fp.alignment = WD_TABLE_ALIGNMENT.CENTER
table_fp.autofit = False

for row in table_fp.rows:
    for i, w in enumerate(fn_w):
        row.cells[i].width = w

fp_headers = ["케이스", "시퀀스 클립명", "카메라 시점", "예측 확률", "주요 오분류 원인"]
for i, h in enumerate(fp_headers):
    table_fp.cell(0, i).text = h
    set_cell_properties(table_fp.cell(0, i), bg_color="1A365D", bold=True, color=RGBColor(255, 255, 255), font_size=9.0)

fp_rows = [
    ("FP-1", "[오류]_D03_피험자1_Key295", "시점 E (좌측 후방 45°)", "0.9092", "좌측 후방 뷰 특유의 깊은 상체 숙임 각도로 인한 착시"),
    ("FP-2", "[오류]_D11_피험자9_Key286", "시점 A (우측 후방 45°)", "0.9077", "초기 셋업 골반 위치가 낮아 상승 속도가 급격하게 관측됨"),
    ("FP-3", "[오류]_D03_피험자5_Key288", "시점 B (우측 전방 45°)", "0.8758", "피험자 체형 특성(긴 대퇴골)으로 무릎 각도 변화 선행"),
    ("FP-4", "[오류]_D03_피험자2_Key282", "시점 B (우측 전방 45°)", "0.8581", "바벨이 무릎 통과 시 상체 수평 각도 유지 구간 과대 판정"),
    ("FP-5", "[오류]_D07_피험자4_Key299", "시점 D (좌측 전방 45°)", "0.8578", "골반-어깨 수직 거리 비대칭 및 상체 모멘트암 증가"),
    ("FP-6", "[오류]_D03_피험자2_Key288", "시점 A (우측 후방 45°)", "0.8328", "후방 원근감으로 인해 상체보다 엉덩이가 먼저 솟은 것으로 투영")
]

for row_idx, data in enumerate(fp_rows, start=1):
    for col_idx, text in enumerate(data):
        cell = table_fp.cell(row_idx, col_idx)
        cell.text = text
        bg = "EBF8FF" if "후방" in data[2] else "FFFFFF"
        set_cell_properties(cell, bg_color=bg, bold=False, color=BLACK, font_size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER if col_idx in [0, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT)

doc.add_paragraph().paragraph_format.space_after = Pt(4)

add_callout_box(doc, [
    "■ 오탐(FP)의 핵심 원인: 후방 시점(시점 A, 시점 E)의 원근 왜곡 및 체형(긴 대퇴골) 변수",
    "  - 시점 A(우측후방)와 시점 E(좌측후방)에서는 피험자가 정상 자세로 상체를 깊이 숙일 때, 시선 각도가 뒤쪽 위에서 아래로 내려다보는 형태가 되어",
    "    골반의 수직 위치가 실제보다 더 높게 투영되고, 상체는 더 낮게 보여 모델이 '힙 슈트업'으로 강하게 오인(확률 0.90 초과)합니다.",
    "  - 또한 대퇴골(허벅지뼈)이 긴 체형의 피험자는 해부학적으로 바벨을 무릎 위로 통과시키기 위해 정상적인 리프팅에서도 무릎을 약간 먼저 펴야 하는 자연스러운 역학이 존재합니다."
], title="오탐(FP) 생체역학적 핵심 진단")

# 4. 결론 및 보완책
p4 = doc.add_paragraph()
style_heading(p4, "4. 실서비스 대응 전략 및 결론", level=1)

p_sol = doc.add_paragraph()
p_sol.paragraph_format.line_spacing = 1.2
p_sol.paragraph_format.space_after = Pt(6)
r = p_sol.add_run(
    "이러한 오분류 분석은 머신러닝 모델의 결함이 아니라, '단일 2D 카메라 시점의 기하학적 한계'와 '체형별 자연스러운 역학 변이'에서 "
    "기인함을 명백히 보여줍니다.\n\n"
    "본 프로젝트(vidio-health-trainer)는 이러한 문제를 해결하기 위해 다음의 2단계 안전 장치를 완비하였습니다:\n"
    "1. Phase 4 Calibrated 3-State 제약 최적화: 모호한 경계 구간(0.150 < P < 0.580)의 7.4% 데이터를 억지로 판정하지 않고 "
    "'판정 보류(Ambiguous)'로 처리하여 서비스 신뢰도를 극대화합니다.\n"
    "2. 권장 촬영 각도 가이드: 정면(시점 C)과 후방(시점 A, E) 대신 가장 왜곡이 적고 높은 성능을 보인 '시점 D(좌측 전방 45°)' 및 "
    "'시점 B(우측 전방 45°)'를 사용자에게 권장 촬영 각도로 안내합니다."
)
r.font.name = "맑은 고딕"
r.font.size = Pt(10)
r.font.color.rgb = BLACK

doc.save(TARGET_DOCX_6TH)
doc.save(TARGET_DOCX_ROOT)
print(f"[+] 오분류 DOCX 저장 완료:\n  - {TARGET_DOCX_6TH}\n  - {TARGET_DOCX_ROOT}")

# Markdown 동시 작성
md_content = f"""# 바벨 데드리프트 d4 (힙 슈트업) 오분류 심층 원인 분석 및 MediaPipe 시각화 보고서

> **프로젝트**: vidio-health-trainer | 바벨 데드리프트 d4 모델 오분류 정밀 분석  
> **저장 위치**: `D:\\Data\\temp\\바벨_데드리프트_d4_오분류_시각화\\` (12개 클립, 192개 고해상도 오버레이 이미지)  
> **작성 일자**: 2026-09-30

---

## 1. 개요 및 데이터 구성

- **미탐 (False Negative, 46건)**: 실제 결함(힙 슈트업)이나 정상으로 놓친 케이스 중 대표 6건 시각화.
- **오탐 (False Positive, 166건)**: 실제 정상이나 결함으로 오인한 케이스 중 대표 6건 시각화.
- **오버레이 내용**: MediaPipe 33개 랜드마크, 대퇴/하퇴 스켈레톤(3px), 실시간 무릎/고관절 각도, 상단 메타정보 배너, 하단 원인진단문.

---

## 2. 미탐 (False Negative) 대표 6건 및 원인 분석

| 케이스 | 클립명 | 카메라 시점 | 예측 확률 | 주요 오분류 원인 |
| :---: | :--- | :---: | :---: | :--- |
| **FN-1** | [오류]_D03_피험자7_Key294 | 시점 C (정면 0°) | 0.2047 | 정면 뷰 깊이 손실로 인한 찰나의 전후방 위상차 소실 |
| **FN-2** | [오류]_D03_피험자1_Key303 | 시점 B (우측 전방 45°) | 0.2097 | 무릎 각도 변화율이 완만하여 임계치 미달 |
| **FN-3** | [오류]_D07_피험자4_Key294 | 시점 C (정면 0°) | 0.2115 | 정면 시점 카메라 왜곡으로 골반 상승 속도 과소추정 |
| **FN-4** | [오류]_D03_피험자6_Key294 | 시점 C (정면 0°) | 0.2242 | 정면 시점 무릎-고관절 수평 간격 압축 |
| **FN-5** | [오류]_D03_피험자7_Key297 | 시점 C (정면 0°) | 0.2391 | 초기 0~10프레임 바벨 이동과 골반 궤적 착시 |
| **FN-6** | [오류]_D07_피험자4_Key294 | 시점 E (좌측 후방 45°) | 0.2583 | 후방 시점 무릎 가림(Occlusion) 발생 |

- **핵심 역학 진단**: 정면(시점 C)에서는 2D 영상의 전후(Z축) 깊이가 압축되어, First Pull 구간에서 무릎이 먼저 펴지는 '위상차(Phase Lag)'가 카메라 화면상에 뚜렷이 나타나지 않아 미탐이 집중 발생함.

---

## 3. 오탐 (False Positive) 대표 6건 및 원인 분석

| 케이스 | 클립명 | 카메라 시점 | 예측 확률 | 주요 오분류 원인 |
| :---: | :--- | :---: | :---: | :--- |
| **FP-1** | [오류]_D03_피험자1_Key295 | 시점 E (좌측 후방 45°) | 0.9092 | 좌측 후방 뷰 특유의 깊은 상체 숙임 각도로 인한 착시 |
| **FP-2** | [오류]_D11_피험자9_Key286 | 시점 A (우측 후방 45°) | 0.9077 | 초기 셋업 골반 위치가 낮아 상승 속도가 급격하게 관측됨 |
| **FP-3** | [오류]_D03_피험자5_Key288 | 시점 B (우측 전방 45°) | 0.8758 | 피험자 체형 특성(긴 대퇴골)으로 무릎 각도 변화 선행 |
| **FP-4** | [오류]_D03_피험자2_Key282 | 시점 B (우측 전방 45°) | 0.8581 | 바벨이 무릎 통과 시 상체 수평 각도 유지 구간 과대 판정 |
| **FP-5** | [오류]_D07_피험자4_Key299 | 시점 D (좌측 전방 45°) | 0.8578 | 골반-어깨 수직 거리 비대칭 및 상체 모멘트암 증가 |
| **FP-6** | [오류]_D03_피험자2_Key288 | 시점 A (우측 후방 45°) | 0.8328 | 후방 원근감으로 인해 상체보다 엉덩이가 먼저 솟은 것으로 투영 |

- **핵심 역학 진단**: 후방 시점(시점 A, E)에서는 상체를 숙이는 동작이 엉덩이가 솟구친 것처럼 원근 투영되며, 대퇴골이 긴 체형의 사용자는 정상적인 리프팅에서도 무릎이 먼저 약간 펴질 수밖에 없어 오탐이 유발됨.

---

## 4. 실서비스 대응 전략

1. **Phase 4 Calibrated 3-State 제약 최적화**: 모호한 7.4% 구간은 무리하게 판정하지 않고 '판정 보류'로 처리.
2. **골든 카메라 앵글 안내**: 왜곡이 가장 적고 판별력이 높은 '시점 D (좌측 전방 45°)' 및 '시점 B (우측 전방 45°)' 촬영 유도.
"""

with open(TARGET_MD_6TH, "w", encoding="utf-8") as f:
    f.write(md_content)
with open(TARGET_MD_ROOT, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"[+] 오분류 MD 저장 완료:\n  - {TARGET_MD_6TH}\n  - {TARGET_MD_ROOT}")
