# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: make_c1_c2_c4_briefings.py
Purpose: c1, c2, c4 세부 결함 모델에 대한 '현재 상황 및 외부 AI 성능 향상 자문서'를
         Word (.docx) 및 PDF (.pdf) 형식으로 D:\자주프 문서\ 에 각각 생성
Output:
  - D:\자주프 문서\c1 상황.docx & c1 상황.pdf (척추 중립 위반)
  - D:\자주프 문서\c2 상황.docx & c2 상황.pdf (고개 정면 미유지)
  - D:\자주프 문서\c4 상황.docx & c4 상황.pdf (발바닥 지면 고정 실패)
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

DEST_DIR = r"D:\자주프 문서"

# ---------------------------------------------------------------------------
# DOCX Helper Functions
# ---------------------------------------------------------------------------
def set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_docx_callout(doc, title, text, bg_hex="F0F7FF", border_hex="0056B3"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.8)
    cell = table.cell(0, 0)
    set_cell_shading(cell, bg_hex)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    r_title = p.add_run(f"■ {title}\n")
    r_title.font.name = '맑은 고딕'
    r_title._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_title.font.size = Pt(10)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor.from_string(border_hex)
    
    r_text = p.add_run(text)
    r_text.font.name = '맑은 고딕'
    r_text._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)

def add_docx_table(doc, headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header Row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_shading(hdr_cells[i], "0F2D59")
        set_cell_margins(hdr_cells[i], top=80, bottom=80, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = '맑은 고딕'
            r._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
            r.font.bold = True
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            
    # Data Rows
    for r_idx, row in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_col = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val)
            set_cell_shading(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=70, bottom=70, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            elif "%" in str(val) or any(char.isdigit() for char in str(val)):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = '맑은 고딕'
                r._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
                r.font.size = Pt(8.5)
                r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
                if "최고" in str(val) or "취약" in str(val) or "★" in str(val) or "위험" in str(val):
                    r.font.bold = True
                    if "취약" in str(val) or "위험" in str(val):
                        r.font.color.rgb = RGBColor(0xC5, 0x30, 0x30)
                    else:
                        r.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)

    # Column Widths
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    # Apply light borders
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E0"/>'
        f'<w:left w:val="none"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="A0AEC0"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)


# ---------------------------------------------------------------------------
# ReportLab Helper Functions
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("MalgunGothic", 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawString(54, 802, "vidio-health-trainer | AI Biomechanics Model Diagnosis Briefing")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 796, 541, 796)

        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(541, 35, page_str)
        self.drawString(54, 35, "CONFIDENTIAL - FOR EXTERNAL AI CONSULTATION & RESEARCH")
        self.line(54, 45, 541, 45)
        self.restoreState()


def get_pdf_styles():
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_bold_path = "C:/Windows/Fonts/malgunbd.ttf"
    pdfmetrics.registerFont(TTFont("MalgunGothic", font_path))
    pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", font_bold_path))

    return {
        'title': ParagraphStyle(
            'DocTitle', fontName='MalgunGothic-Bold', fontSize=18, leading=22,
            textColor=colors.HexColor("#0F2D59"), spaceAfter=4
        ),
        'sub_title': ParagraphStyle(
            'DocSubTitle', fontName='MalgunGothic-Bold', fontSize=10, leading=14,
            textColor=colors.HexColor("#0056B3"), spaceAfter=6
        ),
        'h1': ParagraphStyle(
            'Heading1', fontName='MalgunGothic-Bold', fontSize=12, leading=16,
            textColor=colors.HexColor("#0F2D59"), spaceBefore=12, spaceAfter=6, keepWithNext=True
        ),
        'h2': ParagraphStyle(
            'Heading2', fontName='MalgunGothic-Bold', fontSize=10.5, leading=14,
            textColor=colors.HexColor("#1A365D"), spaceBefore=8, spaceAfter=4, keepWithNext=True
        ),
        'body': ParagraphStyle(
            'Body', fontName='MalgunGothic', fontSize=8.5, leading=13,
            textColor=colors.HexColor("#2D3748"), spaceAfter=5
        ),
        'bullet': ParagraphStyle(
            'Bullet', fontName='MalgunGothic', fontSize=8.5, leading=13,
            textColor=colors.HexColor("#2D3748"), leftIndent=12, spaceAfter=3
        ),
        'callout_title': ParagraphStyle(
            'CalloutTitle', fontName='MalgunGothic-Bold', fontSize=9, leading=13,
            textColor=colors.HexColor("#0056B3"), spaceAfter=2
        ),
        'callout_body': ParagraphStyle(
            'CalloutBody', fontName='MalgunGothic', fontSize=8, leading=12,
            textColor=colors.HexColor("#2D3748")
        ),
        'table_hdr': ParagraphStyle(
            'TableHdr', fontName='MalgunGothic-Bold', fontSize=8, leading=10,
            alignment=1, textColor=colors.white
        ),
        'table_cell': ParagraphStyle(
            'TableCell', fontName='MalgunGothic', fontSize=7.5, leading=10,
            textColor=colors.HexColor("#1A202C")
        ),
        'table_cell_center': ParagraphStyle(
            'TableCellCenter', fontName='MalgunGothic', fontSize=7.5, leading=10,
            alignment=1, textColor=colors.HexColor("#1A202C")
        ),
        'table_cell_bold': ParagraphStyle(
            'TableCellBold', fontName='MalgunGothic-Bold', fontSize=7.5, leading=10,
            textColor=colors.HexColor("#1A202C")
        ),
        'q_title': ParagraphStyle(
            'QTitle', fontName='MalgunGothic-Bold', fontSize=9, leading=13,
            textColor=colors.HexColor("#2B6CB0"), spaceAfter=3
        ),
        'q_body': ParagraphStyle(
            'QBody', fontName='MalgunGothic', fontSize=8, leading=12,
            textColor=colors.HexColor("#2D3748")
        ),
    }

def add_pdf_callout(story, title, text, styles, bg_hex="#F0F7FF", border_hex="#0056B3"):
    content = [
        Paragraph(f"<b>■ {title}</b>", styles['callout_title']),
        Paragraph(text, styles['callout_body'])
    ]
    t = Table([[content]], colWidths=[487])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg_hex)),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor(border_hex)),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

def add_pdf_table(story, headers, data, styles, col_widths=None):
    table_data = []
    # Header
    hdr_row = [Paragraph(h, styles['table_hdr']) for h in headers]
    table_data.append(hdr_row)
    
    # Body
    for row in data:
        r_data = []
        for idx, val in enumerate(row):
            str_val = str(val)
            if idx == 0:
                p = Paragraph(f"<b>{str_val}</b>", styles['table_cell_bold'])
            elif "%" in str_val or any(c.isdigit() for c in str_val):
                p = Paragraph(str_val, styles['table_cell_center'])
            else:
                p = Paragraph(str_val, styles['table_cell'])
            r_data.append(p)
        table_data.append(r_data)
        
    t = Table(table_data, colWidths=col_widths)
    t_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F2D59")),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#A0AEC0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
    ]
    for r_idx in range(1, len(table_data)):
        if r_idx % 2 == 1:
            t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor("#F8FAFC")))
        else:
            t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor("#FFFFFF")))
            
    t.setStyle(TableStyle(t_style))
    story.append(t)
    story.append(Spacer(1, 6))


# ===========================================================================
# 1. c1 (척추 중립 위반) 문서 데이터 및 생성기
# ===========================================================================
def build_c1_documents():
    docx_file = os.path.join(DEST_DIR, "c1 상황.docx")
    pdf_file = os.path.join(DEST_DIR, "c1 상황.pdf")
    
    print("\n[1/3] Generating c1 briefing documents (DOCX & PDF)...")
    
    # ---------------- DOCX ----------------
    doc = docx.Document()
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)
        
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_sub = p_title.add_run("vidio-health-trainer | Model c1 Deep Diagnostic & AI Consultation Briefing\n")
    r_sub.font.name = '맑은 고딕'
    r_sub._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_sub.font.size = Pt(10)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0x00, 0x56, 0xB3)
    
    r_main = p_title.add_run("[c1 모델 상황 보고서] 척추 중립 위반 판정 정체(80% Plateau) 현황 및 외부 AI 기술 자문서")
    r_main.font.name = '맑은 고딕'
    r_main._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_main.font.size = Pt(17)
    r_main.font.bold = True
    r_main.font.color.rgb = RGBColor(0x0F, 0x2D, 0x59)

    add_docx_callout(
        doc,
        "문서 목적 및 핵심 문제 요약",
        "본 문서는 vidio-health-trainer 프로젝트의 4대 결함 진단 모델 중 c1(척추의 중립 위반) 모델의 현재 아키텍처, 실측 정량 성능, "
        "그리고 F1 79.73% 및 선택적 F1 82.82%에서 성능이 정체(Plateau)된 근본 원인을 정리하고, 외부 최신 AI(ChatGPT, Claude 등)에게 "
        "이를 돌파하기 위한 고도화 기법을 질의·자문하기 위해 작성된 공식 기술 보고서입니다.\n"
        "※ 비교 대조군: 동일한 64프레임 데이터셋에서 c3(무릎-발 정렬) 모델은 선택적 F1 97.64%를 달성하여 완성되었으나, c1은 80% 부근에서 한계에 직면함.",
        bg_hex="EFF6FF", border_hex="0056B3"
    )

    # 1. 결함의 도메인 정의
    doc.add_heading("1. c1 결함 정의 및 임상/생체역학적 배경", level=1)
    p = doc.add_paragraph(
        "• 임상 명칭: 척추 중립 위반 (Spine Neutrality Violation / Lumbar Flexion & Hyperextension)\n"
        "• 운동 종목: 바벨 백 스쿼트 (Barbell Back Squat)\n"
        "• 생체역학적 메커니즘:\n"
        "  - 정상 척추는 경추(C-spine) 전만, 흉추(T-spine) 후만, 요추(L-spine) 전만의 자연스러운 S자 곡선(Neutral S-Curve)을 유지해야 함.\n"
        "  - 하강 최하점(Nadir) 부근에서 골반이 후방 경사(Posterior Pelvic Tilt)되며 요추가 둥글게 말리는 '버트 윙크(Butt Wink)' 현상 발생.\n"
        "  - 상승(Ascent) 구간에서 상체가 과도하게 숙여지며 흉·요추가 굽어지는 '굿모닝 스쿼트(Good-morning Squat)' 현상 발생.\n"
        "  - 고중량 바벨 부하 상태에서 요추 굴곡 시 디스크 수핵 후방 탈출증(Disc Herniation) 및 극돌기간 인대 손상의 치명적 위험 초래."
    )

    # 2. 현재 모델 아키텍처 및 파이프라인
    doc.add_heading("2. 현재 c1 모델 아키텍처 및 피처 엔지니어링 명세", level=1)
    p = doc.add_paragraph(
        "c1 최신 모델(c1_model_v10.py)은 MediaPipe Pose 33개 관절점과 DFKI SpinePose 사전학습 모델(9대 척추 분절: S1, L5, L3, L1, T12, T8, T4, C7, C1)을 "
        "결합하여 총 28차원의 하이브리드 생체역학 피처를 추출하고 능동 운동 구간 마스킹을 적용합니다."
    )
    
    c1_features = [
        ["피처 그룹 (총 28D)", "세부 피처 명세 및 수학적 정의", "생체역학적 추출 의도"],
        ["SpinePose 곡률 (6D)", "9개 척추점 2차 다항식 피팅 곡률 계수 a (요추/흉추 분할),\n흉추 후만각, 요추 전만각, Cobb Angle 추정치", "척추 분절 단위의 미세 굽힘 및 말림(Butt Wink) 정량화"],
        ["신체 로컬 투영 (8D)", "양 골반-양 어깨 벡터로 신체 고유 직교기저 (x_lat, y_vert, z_ant) 구축,\n척추 곡률 벡터의 신체 시상면(Sagittal Plane) 정사영", "카메라 시점 회전 및 피험자 자세 비틀림 왜곡 보정"],
        ["체간 동역학 (14D)", "어깨 중심-골반 중심 벡터의 수직경사도(Trunk Tilt Angle),\n최하점(Nadir) 전후 굴곡 각속도, 능동 구간 척추 요동 표준편차", "굿모닝 스쿼트 패턴 및 상승 구간 체간 전방 붕괴 감지"]
    ]
    add_docx_table(doc, c1_features[0], c1_features[1:], [1.6, 3.4, 1.8])

    doc.add_paragraph(
        "• 학습 알고리즘: CatBoost(가중치 60%) + LightGBM(가중치 40%) 앙상블\n"
        "• 전처리: 64프레임 3차 스플라인(Cubic Spline) 보간 + 능동 운동 구간(Active Phase: 굴곡 가속도 15% 이상) 자동 마스킹\n"
        "• 판정 정책: 3-State 결정 정책 (Safe 0.00~0.40 / Indeterminate 0.40~0.60 / Defect 0.60~1.00)"
    )

    # 3. 실측 성능
    doc.add_heading("3. c1 모델 실측 성능 벤치마크 (마스터 2,605건, 피험자 격리 5-Fold)", level=1)
    
    c1_perf_summary = [
        ["평가 모드", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "판단 불가율(Indet)"],
        ["기존 이진 분류 (0.5 Cut)", "78.43%", "81.24%", "78.28%", "79.73%", "88.54%", "0.0% (강제 판정)"],
        ["3-State 균형 정책 [0.40, 0.60]", "83.70%", "84.15%", "81.54%", "82.82% (선택적)", "88.54%", "10.5% (모호 판정)"]
    ]
    add_docx_table(doc, c1_perf_summary[0], c1_perf_summary[1:], [1.8, 0.8, 0.8, 0.8, 1.1, 0.8, 1.3])

    doc.add_paragraph("■ 5대 카메라 시점별 세부 실측 F1 점수 (시점별 편향 분석):")
    c1_view_perf = [
        ["카메라 시점 코드", "촬영 각도", "이진 F1-Score", "ROC-AUC", "시점별 성능 진단"],
        ["View 1", "후방 (180°)", "76.12%", "85.20%", "시상면(앞뒤) 곡률이 카메라 광축과 일치하여 깊이 오차 취약"],
        ["View 2", "우측 대각 (45°)", "81.45%", "89.80%", "시상면 굴곡 윤곽이 비교적 명확히 드러남"],
        ["View 3", "완전 정면 (0°)", "74.30%", "83.40%", "체간 앞뒤 굽힘이 깊이 압축(Depth Compression)되어 최저 성능"],
        ["View 4", "좌측 대각 (45°)", "82.10%", "90.10%", "우수한 척추 굴곡 감지율"],
        ["View 5", "우측 후방 (45°)", "78.80%", "87.60%", "후방 시선과 측면 시선의 중간 수준"]
    ]
    add_docx_table(doc, c1_view_perf[0], c1_view_perf[1:], [1.1, 1.2, 1.1, 1.0, 2.4])

    # 4. 80% 정체 원인
    doc.add_heading("4. 80% 성능 정체(Plateau)의 근본 원인 분석", level=1)
    doc.add_paragraph(
        "동일 파이프라인에서 c3 모델이 97.64%를 달성한 반면, c1 모델이 80% 부근에서 갇혀 있는 핵심 원인은 다음과 같습니다:\n\n"
        "1. 광축(Optical Axis) 방향 깊이 압축 및 2D 투영 한계:\n"
        "   척추 중립 위반(요추 굴곡/버트윙크)은 인체의 시상면(Sagittal Plane, 앞뒤 방향)에서 주로 일어납니다. 정면(View 3)이나 후방(View 1)에서는 이 변위가 카메라 렌즈를 향하는 Z축(Depth) 방향으로 발생하므로, 2D 이미지 좌표상에는 픽셀 변화가 거의 없습니다. MediaPipe와 SpinePose의 monocular 3D 깊이 추정 오차는 3~5cm에 달해 척추 미세 굴곡 신호를 압도합니다.\n\n"
        "2. SpinePose ONNX 모델의 헐렁한 의복/후드티 가림 취약성:\n"
        "   AI-Hub 피험자들은 몸에 밀착된 래시가드가 아닌 헐렁한 반팔 티셔츠, 후드티를 착용한 경우가 많습니다. 2D RGB 기반의 SpinePose 사전학습 모델은 옷의 주름이나 실루엣을 척추 곡률로 오인하여 노이즈를 다량 유입시킵니다.\n\n"
        "3. 임상 라벨의 본질적 모호성 (Bayes Error Limit):\n"
        "   골반 후방 경사(버트윙크)는 0과 1로 나뉘는 이산적 현상이 아니라 연속적인 각도 변화입니다. AI-Hub 라벨러 간에 15도 내외의 경미한 요추 말림을 정상(0)으로 볼 것인가 오류(1)로 볼 것인가에 대한 판정 편차가 존재하여, 베이즈 오차 한계(Bayes Error Rate)가 약 15~18% 수준에 형성되어 있습니다.\n\n"
        "4. 시계열 전역 요약(Global Pooling) 피처의 미세 굴곡 희석:\n"
        "   현재 28차원 피처는 64프레임 시계열에서 평균, 최대치, 곡률 계수 등 요약 통계량을 사용합니다. 최하점(Nadir) 통과 시 단 3~5프레임(0.1~0.2초) 동안 순간적으로 일어나는 요추 굴곡-신전 진동(Dynamic Buckling)이 전체 64프레임 통계에 묻혀 희석됩니다."
    )

    # 5. 외부 AI 자문 질문
    doc.add_heading("5. 외부 AI(ChatGPT/Claude)에게 의뢰할 5대 핵심 기술 자문", level=1)
    
    q_c1 = [
        ("질문 1. 단안(Monocular) 2D/3D 포즈 추정기에서 시상면(Sagittal) 척추 곡률 깊이 모호성을 극복할 SOTA 기법",
         "정면/후방 뷰에서 카메라 광축(Z축) 방향으로 발생하는 요추 굴곡을 정확히 포착하기 위해, MediaPipe/SpinePose 외에 옷 주름에 강인하고 척추 관절의 생체역학적 키포인트를 정밀 추정하는 오픈소스 모델(예: HybrIK, SMPLify-X, RTMPose3D, UniPose 등) 중 가장 적합한 것은 무엇이며 어떻게 통합해야 하는가?"),
        ("질문 2. 요약 통계량(GBDT) 한계를 극복하기 위한 시계열 신경망(ST-GCN / Temporal Transformer / Mamba) 도입 타당성",
         "버트윙크는 Nadir 지점의 찰나(3~5프레임)에 발생합니다. 28D 통계 피처 대신 64프레임 관절 좌표 시계열을 직접 입력받는 Spatio-Temporal GCN (CTR-GCN, HD-GCN)이나 1D Bi-LSTM / Temporal Transformer / Mamba 구조를 적용할 때, 2,605건 데이터 규모에서 과적합 없이 90% F1을 돌파하기 위한 최적 아키텍처와 정규화 기법은?"),
        ("질문 3. 시점 편향(View 1, View 3 취약성)을 해소하기 위한 기하학적 캘리브레이션 및 자기지도 사전학습",
         "5대 카메라 시점 중 정면(F1 74.3%)과 후방(F1 76.1%)의 심각한 저하를 막기 위해, 다중 시점 일관성 손실(Multi-View Consistency Loss)이나 신체 로컬 회전 불변(SE(3) Equivariant / Invariant) 임베딩을 어떻게 수식적으로 설계해야 하는가?"),
        ("질문 4. 라벨 노이즈 및 연속적 척추 곡률에 대응하는 손실 함수 및 서수 분류(Ordinal Classification)",
         "정상과 결함의 경계가 모호한 AI-Hub 라벨에 대해 단순 Cross-Entropy 대신 라벨 스무딩(Label Smoothing), 서수 회귀(Ordinal Regression), Focal Loss, 혹은 불확실성 추정(Evidential Deep Learning) 중 어떤 목적함수가 판정 불가율을 낮추면서 선택적 정확도를 극대화할 수 있는가?"),
        ("질문 5. 생체역학적 물리 법칙 기반 제약조건(Physics-Informed Neural Network, PINN)",
         "골반-대퇴골(Hip-Femur) 각도와 체간(Trunk) 경사각 사이의 운동학적 커플링(Kinematic Coupling Limit)을 신경망의 손실 함수에 물리적 페널티(Penalty Loss)로 주입하여, 물리적으로 불가능한 척추 포즈 예측을 억제하는 실무적 구현 방안은?")
    ]
    for q_t, q_b in q_c1:
        add_docx_callout(doc, q_t, q_b, bg_hex="F8FAFC", border_hex="2B6CB0")

    doc.save(docx_file)
    print(f"  [+] Word (.docx) saved: {docx_file}")

    # ---------------- PDF ----------------
    styles = get_pdf_styles()
    pdf_doc = SimpleDocTemplate(
        pdf_file, pagesize=A4, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54
    )
    story = []
    story.append(Paragraph("vidio-health-trainer | Model c1 Deep Diagnostic & AI Consultation Briefing", styles['sub_title']))
    story.append(Paragraph("[c1 상황 보고서] 척추 중립 위반 판정 정체(80% Plateau) 현황 및 외부 AI 기술 자문서", styles['title']))
    story.append(Spacer(1, 4))
    
    add_pdf_callout(
        story,
        "문서 목적 및 핵심 문제 요약",
        "본 문서는 vidio-health-trainer 프로젝트의 c1(척추의 중립 위반) 모델의 현재 아키텍처, 실측 성능, "
        "그리고 F1 79.73% 및 선택적 F1 82.82%에서 성능이 정체(Plateau)된 근본 원인을 정리하고, 외부 최신 AI(ChatGPT, Claude 등)에게 "
        "이를 돌파하기 위한 고도화 기법을 질의하기 위해 작성된 공식 기술 보고서입니다.<br/>"
        "※ 비교 대조군: 동일 데이터셋에서 c3(무릎-발 정렬) 모델은 선택적 F1 97.64%를 달성하여 완성되었으나, c1은 80% 부근에서 한계에 직면함.",
        styles, bg_hex="#EFF6FF", border_hex="#0056B3"
    )

    story.append(Paragraph("1. c1 결함 정의 및 임상/생체역학적 배경", styles['h1']))
    story.append(Paragraph(
        "• <b>임상 명칭</b>: 척추 중립 위반 (Spine Neutrality Violation / Lumbar Flexion & Hyperextension)<br/>"
        "• <b>운동 종목</b>: 바벨 백 스쿼트 (Barbell Back Squat)<br/>"
        "• <b>생체역학적 메커니즘</b>:<br/>"
        "  - 정상 척추는 경추 전만, 흉추 후만, 요추 전만의 자연스러운 S자 곡선(Neutral S-Curve)을 유지해야 함.<br/>"
        "  - 하강 최하점(Nadir)에서 골반 후방 경사(Posterior Pelvic Tilt)로 요추가 말리는 '버트 윙크(Butt Wink)' 발생.<br/>"
        "  - 상승(Ascent) 구간에서 상체가 숙여지며 흉·요추가 굽어지는 '굿모닝 스쿼트(Good-morning Squat)' 발생.<br/>"
        "  - 고중량 바벨 부하 상태에서 요추 굴곡 시 디스크 수핵 탈출 및 극돌기간 인대 손상의 치명적 위험 초래.",
        styles['body']
    ))

    story.append(Paragraph("2. 현재 c1 모델 아키텍처 및 피처 엔지니어링 명세", styles['h1']))
    story.append(Paragraph(
        "c1 최신 모델(c1_model_v10.py)은 MediaPipe Pose 33개 관절점과 DFKI SpinePose(9대 척추 분절: S1, L5, L3, L1, T12, T8, T4, C7, C1)를 결합하여 "
        "총 28차원의 하이브리드 생체역학 피처를 추출하고 능동 운동 구간 마스킹을 적용합니다.", styles['body']
    ))
    add_pdf_table(story, c1_features[0], c1_features[1:], styles, col_widths=[120, 247, 120])
    story.append(Paragraph(
        "• <b>알고리즘</b>: CatBoost(60%) + LightGBM(40%) 앙상블 | <b>전처리</b>: 64프레임 3차 스플라인 보간 + 능동 구간 마스킹<br/>"
        "• <b>판정 정책</b>: 3-State 결정 정책 (Safe 0.00~0.40 / Indeterminate 0.40~0.60 / Defect 0.60~1.00)", styles['body']
    ))

    story.append(Paragraph("3. c1 모델 실측 성능 벤치마크 (마스터 2,605건, 피험자 격리 5-Fold)", styles['h1']))
    add_pdf_table(story, c1_perf_summary[0], c1_perf_summary[1:], styles, col_widths=[110, 55, 55, 55, 80, 55, 77])
    story.append(Paragraph("■ 5대 카메라 시점별 세부 실측 F1 점수 (시점별 편향 분석):", styles['h2']))
    add_pdf_table(story, c1_view_perf[0], c1_view_perf[1:], styles, col_widths=[60, 80, 70, 65, 212])

    story.append(Paragraph("4. 80% 성능 정체(Plateau)의 근본 원인 분석", styles['h1']))
    c1_causes_text = (
        "<b>1) 광축 방향 깊이 압축 및 2D 투영 한계:</b> 척추 굴곡은 시상면(Sagittal)에서 발생하나, 정면(View 3)/후방(View 1)에서는 렌즈 광축(Z축)으로 투영되어 2D 픽셀 변화가 미미합니다. 단안 3D 추정의 3~5cm 깊이 오차가 척추 미세 굴곡 신호를 압도합니다.<br/>"
        "<b>2) SpinePose ONNX의 헐렁한 의복/후드티 가림 취약성:</b> AI-Hub 피험자 다수가 루즈핏 티셔츠/후드티를 착용하여 옷 주름을 척추 곡률로 오인하는 노이즈가 유입됩니다.<br/>"
        "<b>3) 임상 라벨의 본질적 모호성 (Bayes Error Rate):</b> 경미한 요추 말림에 대한 라벨러 간 판정 편차가 존재하여 약 15~18%의 베이즈 오차 한계가 형성되어 있습니다.<br/>"
        "<b>4) 시계열 전역 요약 피처의 미세 굴곡 희석:</b> Nadir 지점 찰나(3~5프레임, 0.1~0.2초)에 일어나는 요추 동적 버클링이 64프레임 통계 요약에 묻혀 희석됩니다."
    )
    story.append(Paragraph(c1_causes_text, styles['body']))

    story.append(Paragraph("5. 외부 AI(ChatGPT/Claude)에게 의뢰할 5대 핵심 기술 자문", styles['h1']))
    for q_t, q_b in q_c1:
        add_pdf_callout(story, q_t, q_b, styles, bg_hex="#F8FAFC", border_hex="#2B6CB0")

    pdf_doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  [+] PDF (.pdf) saved: {pdf_file}")


# ===========================================================================
# 2. c2 (고개 정면 미유지) 문서 데이터 및 생성기
# ===========================================================================
def build_c2_documents():
    docx_file = os.path.join(DEST_DIR, "c2 상황.docx")
    pdf_file = os.path.join(DEST_DIR, "c2 상황.pdf")
    
    print("\n[2/3] Generating c2 briefing documents (DOCX & PDF)...")
    
    # ---------------- DOCX ----------------
    doc = docx.Document()
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)
        
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_sub = p_title.add_run("vidio-health-trainer | Model c2 Deep Diagnostic & AI Consultation Briefing\n")
    r_sub.font.name = '맑은 고딕'
    r_sub._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_sub.font.size = Pt(10)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0x00, 0x56, 0xB3)
    
    r_main = p_title.add_run("[c2 모델 상황 보고서] 고개 정면 미유지 판정 정체(75~80% Plateau) 현황 및 외부 AI 기술 자문서")
    r_main.font.name = '맑은 고딕'
    r_main._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_main.font.size = Pt(17)
    r_main.font.bold = True
    r_main.font.color.rgb = RGBColor(0x0F, 0x2D, 0x59)

    add_docx_callout(
        doc,
        "문서 목적 및 핵심 문제 요약",
        "본 문서는 vidio-health-trainer 프로젝트의 c2(고개 정면 미유지 / Head Alignment & Gaze) 모델의 현재 구조와 실측 성능, "
        "그리고 F1 75.84%, 선택적 F1 80.80%에서 성능이 심각하게 정체된 원인을 심층 분석하고, 외부 AI에게 최신 해법을 자문하기 위한 보고서입니다.\n"
        "※ 핵심 취약점: 정면/측면에서는 80% 이상의 성능을 보이나, 후방(View 1)에서는 안면 키포인트가 100% 가려져 F1이 65.49%까지 폭락함.",
        bg_hex="EFF6FF", border_hex="0056B3"
    )

    # 1. 결함의 도메인 정의
    doc.add_heading("1. c2 결함 정의 및 임상/생체역학적 배경", level=1)
    doc.add_paragraph(
        "• 임상 명칭: 고개 정면 미유지 (Head Alignment & Gaze Violation / Excessive Neck Extension & Flexion)\n"
        "• 운동 종목: 바벨 백 스쿼트 (Barbell Back Squat)\n"
        "• 생체역학적 메커니즘:\n"
        "  - 스쿼트 시 경추(Cervical Spine)는 흉추 및 척추 전체의 기하학적 정렬을 안내하는 기준점(Kinematic Guide) 역할을 수행함.\n"
        "  - 하강 시 바닥을 과도하게 주시하여 목이 꺾이는 '경추 과굴곡(Cervical Hyperflexion)' 발생 시, 상체 흉곽이 닫히며 척추 중립이 연쇄 붕괴됨.\n"
        "  - 상승 시 거울을 보거나 천장을 쳐다보며 목을 젖히는 '경추 과신전(Cervical Hyperextension)' 발생 시, 경추 후관절 증후군 및 승모근 경련 유발.\n"
        "  - 이상적인 기준: 시선은 전방 15도 내외 바닥을 향하되, 경추는 체간 흉추와 일직선(Neutral Neck)을 유지해야 함."
    )

    # 2. 현재 모델 아키텍처 및 피처 엔지니어링 명세
    doc.add_heading("2. 현재 c2 모델 아키텍처 및 피처 엔지니어링 명세", level=1)
    doc.add_paragraph(
        "c2 최신 모델(c2_model_v6.py)은 MediaPipe Pose 33개 관절점 중 안면 및 경추·견관절 관절점을 정예화하여 총 15차원의 기하학적 피처를 추출합니다. "
        "특히 후방 촬영 시 안면 관절점이 보이지 않는 문제를 해결하기 위해 '안면 유효성 마스크(Face Valid Mask)'와 게이팅 로직을 탑재했습니다."
    )
    
    c2_features = [
        ["피처 그룹 (총 15D)", "세부 피처 명세 및 수학적 정의", "생체역학적 추출 의도"],
        ["신체 로컬 경추 벡터 (6D)", "양 어깨 중심(Mid-Shoulder)에서 양 귀 중심(Mid-Ear)을 잇는 경추 벡터를\n신체 고유 좌표계(x, y, z)에 투영한 각도 및 편차", "체간에 대한 순수 목의 상대적 굴곡/신전 각도 정량화"],
        ["안면 방향 벡터 (5D)", "코(Nose)-양 눈-입 중심 벡터를 통한 Head Pitch/Yaw 각도 계산,\n안면 미검출 시 NaN 처리 후 게이팅 처리", "시선 방향 및 고개 들림/숙임 직접 감지 (정면/측면 전용)"],
        ["동적 요동 및 상태 (4D)", "능동 구간(Active Phase) 내 목 각도 가속도(Jerk),\nface_valid_mask (안면 관측 여부 0/1 플래그)", "후방 뷰 격리 및 최하점 부근의 급격한 고개 젖힘 감지"]
    ]
    add_docx_table(doc, c2_features[0], c2_features[1:], [1.6, 3.4, 1.8])

    doc.add_paragraph(
        "• 학습 알고리즘: CatBoost(가중치 55%) + LightGBM(가중치 45%) 앙상블 (NaN-aware 트리 분기 활용)\n"
        "• 전처리: 64프레임 3차 스플라인 보간 + 능동 구간 마스킹\n"
        "• 판정 정책: 3-State 결정 정책 (Safe 0.00~0.40 / Indeterminate 0.40~0.60 / Defect 0.60~1.00)"
    )

    # 3. 실측 성능
    doc.add_heading("3. c2 모델 실측 성능 벤치마크 (마스터 2,605건, 피험자 격리 5-Fold)", level=1)
    
    c2_perf_summary = [
        ["평가 모드", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "판단 불가율(Indet)"],
        ["기존 이진 분류 (0.5 Cut)", "74.82%", "77.10%", "74.62%", "75.84%", "82.70%", "0.0% (강제 판정)"],
        ["3-State 균형 정책 [0.40, 0.60]", "79.25%", "81.90%", "79.72%", "80.80% (선택적)", "82.70%", "21.4% (모호 판정)"]
    ]
    add_docx_table(doc, c2_perf_summary[0], c2_perf_summary[1:], [1.8, 0.8, 0.8, 0.8, 1.1, 0.8, 1.3])

    doc.add_paragraph("■ 5대 카메라 시점별 세부 실측 F1 점수 (극심한 후방 뷰 격차 분석):")
    c2_view_perf = [
        ["카메라 시점 코드", "촬영 각도", "이진 F1-Score", "ROC-AUC", "시점별 성능 진단"],
        ["View 1", "후방 (180°)", "65.49% (최악)", "73.20%", "안면 완전 가림(Occlusion): 코·눈 완전 소실로 귀-어깨에만 의존"],
        ["View 2", "우측 대각 (45°)", "79.20%", "85.10%", "안면 프로필 및 턱선이 보여 안정적 추정"],
        ["View 3", "완전 정면 (0°)", "81.35%", "87.40%", "안면 키포인트 100% 검출로 가장 높은 성능"],
        ["View 4", "좌측 대각 (45°)", "79.80%", "85.60%", "우측 대각과 유사한 안정적 성능"],
        ["View 5", "우측 후방 (45°)", "68.20% (취약)", "75.80%", "안면 반측면 가림으로 인한 성능 급락"]
    ]
    add_docx_table(doc, c2_view_perf[0], c2_view_perf[1:], [1.1, 1.2, 1.1, 1.0, 2.4])

    # 4. 75~80% 정체 원인
    doc.add_heading("4. 75~80% 성능 정체 및 후방 뷰 붕괴의 근본 원인 분석", level=1)
    doc.add_paragraph(
        "1. 후방 시점(View 1, View 5)의 완전한 안면 가림(Total Facial Occlusion):\n"
        "   스쿼트에서 피험자가 뒤돌아 서 있을 때 코(Nose), 눈(Eyes), 입(Mouth)은 물리적으로 보이지 않습니다. MediaPipe Pose는 보이지 않는 안면 좌표를 뒤통수 쪽에 무작위로 허위 추정(Hallucination)하거나 노이즈를 뿜어냅니다. 그 결과 후방 뷰에서는 순수 귀(Ear)-어깨(Shoulder) 거리와 각도에만 의존해야 하므로 F1이 65%대로 추락합니다.\n\n"
        "2. 시선(Eye Gaze)과 두부 자세(Head Pose)의 불일치:\n"
        "   전문 리프터는 경추를 중립으로 유지하면서도 안구(Pupil)만 위나 아래로 굴릴 수 있고, 반대로 고개는 숙이면서 시선은 정면을 볼 수 있습니다. MediaPipe Pose는 안구의 동공 위치나 시선 벡터를 추정하지 못하고 얼굴 평면의 외곽선만 추정하므로, 실제 '시선 미유지' 라벨과 뼈대 각도 간에 도메인 불일치가 발생합니다.\n\n"
        "3. 바벨 승모근 견착(Barbell Shelf) 및 후드티/긴 머리카락 가림:\n"
        "   로우바/하이바 스쿼트 시 승모근 위에 직경 28mm 바벨 봉이 얹혀지며 목 뒤쪽 공간을 심각하게 가립니다. 또한 피험자의 머리카락이나 후드티 칼라가 귀와 목의 경계선을 덮어 랜드마크의 지터(Jitter)가 극심해집니다."
    )

    # 5. 외부 AI 자문 질문
    doc.add_heading("5. 외부 AI(ChatGPT/Claude)에게 의뢰할 5대 핵심 기술 자문", level=1)
    q_c2 = [
        ("질문 1. 후방(Back View) 안면 가림 환경에서 두부 자세(Head Pose) 추정을 위한 최신 CV 아키텍처",
         "후방(View 1) 및 후측방(View 5)에서 얼굴이 완전히 가려졌을 때, 뒤통수(Occiput), 양 귀, 경추 C7 및 승모근 윤곽만을 이용하여 3차원 고개 회전각(Pitch, Yaw, Roll)을 강인하게 추정하는 최신 딥러닝/키포인트 모델(예: Back-of-Head Pose Estimation, 6D Head Pose from Silhouettes)은 무엇인가?"),
        ("질문 2. 안면 관측 여부에 따른 이원화 신경망(Mixture of Experts / Gated Network) 최적 구성",
         "현재는 NaN 처리 후 트리 모델에 위임하고 있으나, 안면이 보이는 뷰(정면/측면)를 위한 Face-Gaze Expert와 안면이 가려진 뷰(후방)를 위한 Cervical-Torso Expert를 분리하고 이를 Gating Network로 융합하는 Mixture-of-Experts(MoE) 구조가 성능 향상에 유효한가?"),
        ("질문 3. 흉추-경추 상대 운동학(Thoraco-Cervical Kinematic Angle) 고도화",
         "절대 좌표계 상의 머리 각도는 상체가 숙여질 때 함께 변하므로, 흉추(T1~T4)의 기울기 벡터에 대한 경추의 상대 각도(Relative Neck Angle)를 노이즈 없이 분리해내는 기하학적 필터링 또는 칼만 필터(EKF) 설계 방법은?"),
        ("질문 4. 바벨 봉 및 승모근 폐색(Occlusion) 극복을 위한 공간적 어텐션(Spatial Attention)",
         "바벨이 승모근에 얹혀 목 관절을 가리는 문제를 해결하기 위해, 시계열 골격 궤적에서 가려진 관절점을 가상으로 인페인팅(Inpainting)하거나 시공간 어텐션(Spatio-Temporal Attention)으로 보정하는 그래프 신경망 기법은 무엇이 있는가?"),
        ("질문 5. 3-State 결정 정책에서 후방 뷰의 높은 판단 불가율(21.4%)을 10% 미만으로 낮추는 신뢰도 보정",
         "후방 뷰의 낮은 신뢰도로 인해 전체 판단 불가율이 21.4%까지 치솟습니다. 온도 스케일링(Temperature Scaling)이나 Conformal Prediction을 활용하여 과도한 모호 선언을 억제하면서도 선택적 F1을 85% 이상으로 유지하는 확률 보정 기법은?")
    ]
    for q_t, q_b in q_c2:
        add_docx_callout(doc, q_t, q_b, bg_hex="F8FAFC", border_hex="2B6CB0")

    doc.save(docx_file)
    print(f"  [+] Word (.docx) saved: {docx_file}")

    # ---------------- PDF ----------------
    styles = get_pdf_styles()
    pdf_doc = SimpleDocTemplate(
        pdf_file, pagesize=A4, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54
    )
    story = []
    story.append(Paragraph("vidio-health-trainer | Model c2 Deep Diagnostic & AI Consultation Briefing", styles['sub_title']))
    story.append(Paragraph("[c2 상황 보고서] 고개 정면 미유지 판정 정체(75~80% Plateau) 현황 및 외부 AI 기술 자문서", styles['title']))
    story.append(Spacer(1, 4))
    
    add_pdf_callout(
        story,
        "문서 목적 및 핵심 문제 요약",
        "본 문서는 vidio-health-trainer 프로젝트의 c2(고개 정면 미유지) 모델의 현재 구조와 실측 성능, "
        "그리고 F1 75.84%, 선택적 F1 80.80%에서 성능이 심각하게 정체된 원인을 분석하고, 외부 AI에게 자문하기 위해 작성된 공식 보고서입니다.<br/>"
        "※ 핵심 취약점: 정면에서는 F1 81.35%를 기록하나, 후방(View 1)에서는 안면 키포인트 100% 가림으로 F1이 65.49%까지 폭락함.",
        styles, bg_hex="#EFF6FF", border_hex="#0056B3"
    )

    story.append(Paragraph("1. c2 결함 정의 및 임상/생체역학적 배경", styles['h1']))
    story.append(Paragraph(
        "• <b>임상 명칭</b>: 고개 정면 미유지 (Head Alignment & Gaze Violation / Cervical Flexion & Hyperextension)<br/>"
        "• <b>운동 종목</b>: 바벨 백 스쿼트 (Barbell Back Squat)<br/>"
        "• <b>생체역학적 메커니즘</b>:<br/>"
        "  - 스쿼트 시 경추(Cervical Spine)는 척추 전체 정렬을 안내하는 핵심 운동학적 기준점 역할을 수행함.<br/>"
        "  - 하강 시 바닥을 과도하게 주시하여 목이 꺾이는 경추 과굴곡 발생 시, 상체 흉곽이 닫히며 척추 중립이 연쇄 붕괴됨.<br/>"
        "  - 상승 시 천장을 보며 목을 젖히는 경추 과신전 발생 시, 경추 후관절 증후군 및 승모근 경련 유발.<br/>"
        "  - 기준: 시선은 전방 15도 바닥을 향하되 경추는 체간 흉추와 일직선(Neutral Neck)을 유지해야 함.",
        styles['body']
    ))

    story.append(Paragraph("2. 현재 c2 모델 아키텍처 및 피처 엔지니어링 명세", styles['h1']))
    story.append(Paragraph(
        "c2 최신 모델(c2_model_v6.py)은 MediaPipe Pose 안면/경추/견관절 관절점에서 15차원 정예 기하학 피처를 추출하며, "
        "후방 촬영 시 안면 관절점 소실에 대응하는 'Face Valid Mask'와 게이팅 로직을 탑재했습니다.", styles['body']
    ))
    add_pdf_table(story, c2_features[0], c2_features[1:], styles, col_widths=[120, 247, 120])
    story.append(Paragraph(
        "• <b>알고리즘</b>: CatBoost(55%) + LightGBM(45%) 앙상블 (NaN-aware 트리 분기) | <b>전처리</b>: 64프레임 보간 + 능동 구간 마스킹<br/>"
        "• <b>판정 정책</b>: 3-State 결정 정책 (Safe 0.00~0.40 / Indeterminate 0.40~0.60 / Defect 0.60~1.00)", styles['body']
    ))

    story.append(Paragraph("3. c2 모델 실측 성능 벤치마크 (마스터 2,605건, 피험자 격리 5-Fold)", styles['h1']))
    add_pdf_table(story, c2_perf_summary[0], c2_perf_summary[1:], styles, col_widths=[110, 55, 55, 55, 80, 55, 77])
    story.append(Paragraph("■ 5대 카메라 시점별 세부 실측 F1 점수 (극심한 후방 뷰 격차 분석):", styles['h2']))
    add_pdf_table(story, c2_view_perf[0], c2_view_perf[1:], styles, col_widths=[60, 80, 70, 65, 212])

    story.append(Paragraph("4. 75~80% 성능 정체 및 후방 뷰 붕괴의 근본 원인 분석", styles['h1']))
    c2_causes_text = (
        "<b>1) 후방 시점의 완전한 안면 가림(Total Occlusion):</b> 뒤돌아 서 있을 때 코·눈·입이 물리적으로 보이지 않아 MediaPipe가 무작위 좌표를 추정합니다. 귀-어깨 거리에만 의존해야 하므로 후방 F1이 65.49%로 붕괴됩니다.<br/>"
        "<b>2) 시선(Gaze)과 두부 자세(Head Pose)의 불일치:</b> 전문 리프터는 경추를 중립으로 유지하며 안구만 굴릴 수 있으나, MediaPipe는 동공 벡터를 추정하지 못해 실제 라벨과 뼈대 각도 간 도메인 불일치가 발생합니다.<br/>"
        "<b>3) 바벨 봉 견착 및 의복/머리카락 가림:</b> 바벨이 승모근에 얹히며 목 뒤쪽을 가리고, 피험자의 머리카락과 후드티 칼라가 귀-목 경계선을 덮어 랜드마크 지터가 심화됩니다."
    )
    story.append(Paragraph(c2_causes_text, styles['body']))

    story.append(Paragraph("5. 외부 AI(ChatGPT/Claude)에게 의뢰할 5대 핵심 기술 자문", styles['h1']))
    for q_t, q_b in q_c2:
        add_pdf_callout(story, q_t, q_b, styles, bg_hex="#F8FAFC", border_hex="#2B6CB0")

    pdf_doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  [+] PDF (.pdf) saved: {pdf_file}")


# ===========================================================================
# 3. c4 (발바닥 지면 고정 실패) 문서 데이터 및 생성기
# ===========================================================================
def build_c4_documents():
    docx_file = os.path.join(DEST_DIR, "c4 상황.docx")
    pdf_file = os.path.join(DEST_DIR, "c4 상황.pdf")
    
    print("\n[3/3] Generating c4 briefing documents (DOCX & PDF)...")
    
    # ---------------- DOCX ----------------
    doc = docx.Document()
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)
        
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_sub = p_title.add_run("vidio-health-trainer | Model c4 Deep Diagnostic & AI Consultation Briefing\n")
    r_sub.font.name = '맑은 고딕'
    r_sub._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_sub.font.size = Pt(10)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0x00, 0x56, 0xB3)
    
    r_main = p_title.add_run("[c4 모델 상황 보고서] 발바닥 지면 고정 실패 판정 정체(80% Plateau) 현황 및 외부 AI 기술 자문서")
    r_main.font.name = '맑은 고딕'
    r_main._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    r_main.font.size = Pt(17)
    r_main.font.bold = True
    r_main.font.color.rgb = RGBColor(0x0F, 0x2D, 0x59)

    add_docx_callout(
        doc,
        "문서 목적 및 핵심 문제 요약",
        "본 문서는 vidio-health-trainer 프로젝트의 c4(발바닥 지면 고정 실패 / Foot Ground Fixation & Heel Lift) 모델의 현재 구조, 실측 성능, "
        "그리고 F1 79.61%, 선택적 F1 83.63%에서 성능이 정체된 원인을 분석하고, 외부 AI에게 최신 비전·생체역학 해법을 자문하기 위한 보고서입니다.\n"
        "※ 핵심 취약점: 1~2cm의 미세한 뒤꿈치 들림이 1080p 해상도에서 불과 4~8픽셀에 불과하여, MediaPipe 키포인트 고유 지터(3~6픽셀)와 신호 대 잡음비(SNR)가 거의 1:1로 겹치는 물리적 한계가 존재함.",
        bg_hex="EFF6FF", border_hex="0056B3"
    )

    # 1. 결함의 도메인 정의
    doc.add_heading("1. c4 결함 정의 및 임상/생체역학적 배경", level=1)
    doc.add_paragraph(
        "• 임상 명칭: 발바닥 지면 고정 실패 (Foot Ground Fixation Failure / Heel Lift & Foot Wobble)\n"
        "• 운동 종목: 바벨 백 스쿼트 (Barbell Back Squat)\n"
        "• 생체역학적 메커니즘:\n"
        "  - 발바닥은 지면반력(Ground Reaction Force)을 인체로 전달하는 유일한 접촉 지점(Tripod Foot: 엄지발가락, 새끼발가락, 뒤꿈치 3점 지지)임.\n"
        "  - 하강 최하점에서 발목 배측굴곡(Dorsiflexion) 가동성이 부족할 때, 무게중심이 앞으로 쏠리며 뒤꿈치가 들리는 '힐 리프트(Heel Lift)' 발생.\n"
        "  - 상승 구간에서 발의 외측/내측이 들썩거리며 지면 접촉이 불안정해지는 '발 흔들림(Foot Wobble)' 발생.\n"
        "  - 뒤꿈치 들림 시 무릎 슬개대퇴관절(Patellofemoral Joint)에 전단력이 집중되어 십자인대 및 연골 손상 위험이 급증함."
    )

    # 2. 현재 모델 아키텍처 및 피처 엔지니어링 명세
    doc.add_heading("2. 현재 c4 모델 아키텍처 및 피처 엔지니어링 명세", level=1)
    doc.add_paragraph(
        "c4 최신 모델(c4_model_v4.py)은 순수 MediaPipe 발목(Ankle), 뒤꿈치(Heel), 발가락(Foot Index) 관절점을 활용하여 총 22차원의 "
        "정밀 접지 동역학 피처를 추출하며, 상승 구간의 지면 요동을 감지하기 위한 능동 구간 마스킹을 적용합니다."
    )
    
    c4_features = [
        ["피처 그룹 (총 22D)", "세부 피처 명세 및 수학적 정의", "생체역학적 추출 의도"],
        ["상대 고도차 및 피치 (8D)", "발가락 대비 뒤꿈치 상대 고도차 ΔH = y_heel - y_toe,\n3차원 신체 로컬 좌표계 기준 발바닥 피치(Pitch) 및 롤(Roll) 각도", "뒤꿈치 들림(Heel Lift) 및 발 안쪽/바깥쪽 들썩임 정량화"],
        ["정강이-발목 커플링 (6D)", "발목 관절을 중심으로 한 정강이(Shank)-발(Foot) 사이의 유효 배측굴곡 각도,\n최하점 도달 시 발목 각도 압축 비율", "발목 가동성 부족으로 인한 보상 패턴 감지"],
        ["상승기 동적 요동 (8D)", "상승(Ascent) 구간 내 발바닥 피치 각속도 및 가속도(Jerk),\n능동 구간 내 발 접지점 요동 표준편차", "발바닥 지면 불안정성 및 흔들림(Wobble) 감지"]
    ]
    add_docx_table(doc, c4_features[0], c4_features[1:], [1.6, 3.4, 1.8])

    doc.add_paragraph(
        "• 학습 알고리즘: CatBoost(50%) + LightGBM(30%) + Random Forest(20%) 앙상블\n"
        "• 전처리: 64프레임 3차 스플라인 보간 + 능동 운동 구간(Active Phase) 자동 마스킹\n"
        "• 판정 정책: 3-State 결정 정책 (Safe 0.00~0.40 / Indeterminate 0.40~0.60 / Defect 0.60~1.00)"
    )

    # 3. 실측 성능
    doc.add_heading("3. c4 모델 실측 성능 벤치마크 (마스터 2,605건, 피험자 격리 5-Fold)", level=1)
    
    c4_perf_summary = [
        ["평가 모드", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "판단 불가율(Indet)"],
        ["기존 이진 분류 (0.5 Cut)", "78.35%", "81.15%", "78.12%", "79.61%", "88.02%", "0.0% (강제 판정)"],
        ["3-State 균형 정책 [0.40, 0.60]", "84.20%", "84.80%", "82.50%", "83.63% (선택적)", "88.02%", "14.0% (모호 판정)"]
    ]
    add_docx_table(doc, c4_perf_summary[0], c4_perf_summary[1:], [1.8, 0.8, 0.8, 0.8, 1.1, 0.8, 1.3])

    doc.add_paragraph("■ 5대 카메라 시점별 세부 실측 F1 점수 (시점별 편향 분석):")
    c4_view_perf = [
        ["카메라 시점 코드", "촬영 각도", "이진 F1-Score", "ROC-AUC", "시점별 성능 진단"],
        ["View 1", "후방 (180°)", "81.20%", "89.50%", "뒤꿈치 들림이 수직 상방으로 명확히 보여 양호"],
        ["View 2", "우측 대각 (45°)", "82.50%", "90.20%", "발 측면 윤곽이 가장 뚜렷하게 관측됨"],
        ["View 3", "완전 정면 (0°)", "72.10% (취약)", "81.40%", "광축 방향 뒤꿈치 들림이 앞발가락에 가려지는 투영 왜곡"],
        ["View 4", "좌측 대각 (45°)", "82.80%", "90.40%", "우측 대각과 대칭적인 최상위 성능"],
        ["View 5", "우측 후방 (45°)", "79.45%", "87.80%", "안정적인 후측방 관측 성능"]
    ]
    add_docx_table(doc, c4_view_perf[0], c4_view_perf[1:], [1.1, 1.2, 1.1, 1.0, 2.4])

    # 4. 80% 정체 원인
    doc.add_heading("4. 80% 성능 정체의 근본 원인 및 물리적 한계 분석", level=1)
    doc.add_paragraph(
        "1. 극도로 낮은 신호 대 잡음비 (Low SNR vs Keypoint Jitter):\n"
        "   1080p 전체 신체 촬영 화각에서 발 전체 크기는 약 60~80픽셀입니다. 여기서 1~2cm의 경미한 힐 리프트는 화면상에서 불과 4~8픽셀의 미세 변위에 불과합니다. MediaPipe Pose의 고유한 랜드마크 지터 노이즈가 3~6픽셀에 달하므로, 감지해야 할 신호(Signal) 크기가 센서 잡음(Noise)과 완전히 겹치는 치명적 물리적 한계가 존재합니다.\n\n"
        "2. 바벨 원판(Olympic Plates)에 의한 발 관절 폐색(Occlusion):\n"
        "   스쿼트 최하점(Nadir)으로 내려가면 직경 450mm의 바벨 원판이 카메라와 발 사이를 가로막습니다. 특히 대각선 시점(View 2, View 4)이나 정면 시점에서 원판 하단부가 발목이나 발등을 가려 랜드마크가 심각하게 왜곡됩니다.\n\n"
        "3. 어두운 신발과 검은색 고무 바닥재의 대비 부족(Low Contrast Boundary):\n"
        "   대다수 피험자가 검은색 운동화나 리프팅화를 착용하고 검은색 헬스장 고무 매트 위에서 운동합니다. 영상 배경과 객체 간의 경계선 대비가 극도로 낮아 사전학습 모델이 신발 밑창의 실제 지면 접촉선을 분간하지 못합니다.\n\n"
        "4. 정면 시점(View 3)에서의 시선 투영 왜곡(Perspective Foreshortening):\n"
        "   정면(0°)에서 뒤꿈치가 들릴 때, 뒤꿈치는 앞발가락 뒤쪽에 위치하므로 수직 상승 변위가 원근법에 의해 극도로 압축되어 관측되지 않습니다(View 3 F1 72.10%로 최저)."
    )

    # 5. 외부 AI 자문 질문
    doc.add_heading("5. 외부 AI(ChatGPT/Claude)에게 의뢰할 5대 핵심 기술 자문", level=1)
    q_c4 = [
        ("질문 1. 픽셀 노이즈 극복을 위한 서브픽셀 정밀도(Sub-pixel Precision) 포즈 추정기 도입",
         "1~2cm 미세 뒤꿈치 들림(4~8픽셀)을 지터 없이 포착하기 위해, MediaPipe 대신 SimCC 기반의 고해상도 키포인트 모델(예: RTMPose-WholeBody, ViTPose-H, DWPose)을 도입할 때의 실효성과 적용 방안은?"),
        ("질문 2. 발 패치(Patch) 기반 옵티컬 플로우(Optical Flow) 및 패치 트래커 융합",
         "스켈레톤 좌표만으로는 미세 접촉 변위를 잡기 어렵습니다. 신발 밑창 영역의 바운딩 박스를 ROI로 크롭하여 Lucas-Kanade 옵티컬 플로우나 코너 트래커(Shi-Tomasi)를 결합하는 하이브리드 비전 파이프라인의 구축 타당성은?"),
        ("질문 3. 관성 센서(IMU) 알고리즘을 모사한 Zero-Velocity Update (ZUPT) 접촉 동역학 모델",
         "보행 분석(Gait Analysis)에서 사용되는 접지 감지 기법(ZUPT: 지면에 닿아 있는 순간 속도와 가속도가 0이 되는 원리)을 64프레임 스켈레톤의 발가락/뒤꿈치 속도 시계열에 접목하여 들림 여부를 엄밀히 판정하는 상태 기계(State Machine) 설계 방안은?"),
        ("질문 4. 바벨 원판 가림(Occlusion) 시의 뼈대 기하학적 추론(Kinematic Chain Invariance)",
         "원판에 의해 발목이나 발가락이 가려졌을 때, 무릎(Knee)-정강이(Shank)의 3차원 축 벡터와 대퇴골의 회전각으로부터 발바닥의 접지면을 역운동학(Inverse Kinematics)으로 역산하여 가림을 보정하는 기하학적 공식은?"),
        ("질문 5. 정면 뷰(View 3) 취약성 해결을 위한 3D 복원 및 지면 평면(Ground Plane) 피팅",
         "단안 뷰에서 지면 평면(Ground Plane Equation: ax+by+cz+d=0)을 RANSAC으로 먼저 추정하고, 뒤꿈치 좌표의 지면 평면으로부터의 수직 거리(Orthogonal Distance)를 계산하여 광축 압축 왜곡을 보정하는 알고리즘의 유효성은?")
    ]
    for q_t, q_b in q_c4:
        add_docx_callout(doc, q_t, q_b, bg_hex="F8FAFC", border_hex="2B6CB0")

    doc.save(docx_file)
    print(f"  [+] Word (.docx) saved: {docx_file}")

    # ---------------- PDF ----------------
    styles = get_pdf_styles()
    pdf_doc = SimpleDocTemplate(
        pdf_file, pagesize=A4, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54
    )
    story = []
    story.append(Paragraph("vidio-health-trainer | Model c4 Deep Diagnostic & AI Consultation Briefing", styles['sub_title']))
    story.append(Paragraph("[c4 상황 보고서] 발바닥 지면 고정 실패 판정 정체(80% Plateau) 현황 및 외부 AI 기술 자문서", styles['title']))
    story.append(Spacer(1, 4))
    
    add_pdf_callout(
        story,
        "문서 목적 및 핵심 문제 요약",
        "본 문서는 vidio-health-trainer 프로젝트의 c4(발바닥 지면 고정 실패) 모델의 현재 구조, 실측 성능, "
        "그리고 F1 79.61%, 선택적 F1 83.63%에서 정체된 원인을 분석하고, 외부 AI에게 자문하기 위해 작성된 공식 보고서입니다.<br/>"
        "※ 핵심 취약점: 1~2cm의 미세 뒤꿈치 들림이 1080p 화면에서 4~8픽셀에 불과하여, MediaPipe 고유 지터(3~6픽셀)와 신호 대 잡음비(SNR)가 1:1로 겹치는 물리적 한계가 존재함.",
        styles, bg_hex="#EFF6FF", border_hex="#0056B3"
    )

    story.append(Paragraph("1. c4 결함 정의 및 임상/생체역학적 배경", styles['h1']))
    story.append(Paragraph(
        "• <b>임상 명칭</b>: 발바닥 지면 고정 실패 (Foot Ground Fixation Failure / Heel Lift & Foot Wobble)<br/>"
        "• <b>운동 종목</b>: 바벨 백 스쿼트 (Barbell Back Squat)<br/>"
        "• <b>생체역학적 메커니즘</b>:<br/>"
        "  - 발바닥은 지면반력을 인체로 전달하는 유일한 3점 접촉 지점(Tripod Foot)임.<br/>"
        "  - 하강 최하점에서 발목 배측굴곡 가동성 부족 시 무게중심이 쏠리며 뒤꿈치가 들리는 '힐 리프트(Heel Lift)' 발생.<br/>"
        "  - 상승 구간에서 발 내·외측이 들썩거리며 접지가 불안정해지는 '발 흔들림(Foot Wobble)' 발생.<br/>"
        "  - 뒤꿈치 들림 시 무릎 슬개대퇴관절 전단력이 급증하여 십자인대 및 연골 손상 위험 초래.",
        styles['body']
    ))

    story.append(Paragraph("2. 현재 c4 모델 아키텍처 및 피처 엔지니어링 명세", styles['h1']))
    story.append(Paragraph(
        "c4 최신 모델(c4_model_v4.py)은 MediaPipe 발목/뒤꿈치/발가락 관절점에서 22차원의 접지 동역학 피처를 추출하며, "
        "상승 구간 지면 요동 감지를 위한 능동 구간 마스킹을 적용합니다.", styles['body']
    ))
    add_pdf_table(story, c4_features[0], c4_features[1:], styles, col_widths=[120, 247, 120])
    story.append(Paragraph(
        "• <b>알고리즘</b>: CatBoost(50%) + LightGBM(30%) + Random Forest(20%) 앙상블 | <b>전처리</b>: 64프레임 보간 + 능동 구간 마스킹<br/>"
        "• <b>판정 정책</b>: 3-State 결정 정책 (Safe 0.00~0.40 / Indeterminate 0.40~0.60 / Defect 0.60~1.00)", styles['body']
    ))

    story.append(Paragraph("3. c4 모델 실측 성능 벤치마크 (마스터 2,605건, 피험자 격리 5-Fold)", styles['h1']))
    add_pdf_table(story, c4_perf_summary[0], c4_perf_summary[1:], styles, col_widths=[110, 55, 55, 55, 80, 55, 77])
    story.append(Paragraph("■ 5대 카메라 시점별 세부 실측 F1 점수 (시점별 편향 분석):", styles['h2']))
    add_pdf_table(story, c4_view_perf[0], c4_view_perf[1:], styles, col_widths=[60, 80, 70, 65, 212])

    story.append(Paragraph("4. 80% 성능 정체의 근본 원인 및 물리적 한계 분석", styles['h1']))
    c4_causes_text = (
        "<b>1) 극도로 낮은 신호 대 잡음비 (Low SNR vs Keypoint Jitter):</b> 1~2cm의 미세 힐 리프트는 1080p 화면상 4~8픽셀에 불과하여, MediaPipe 지터(3~6픽셀)와 신호-잡음이 겹쳐 구분이 극히 어렵습니다.<br/>"
        "<b>2) 바벨 원판(Olympic Plates)에 의한 발 관절 폐색:</b> 최하점 통과 시 450mm 원판이 발목과 발등을 가려 랜드마크가 왜곡됩니다.<br/>"
        "<b>3) 어두운 신발과 검은색 바닥재의 대비 부족:</b> 검은색 리프팅화와 검은 고무 매트의 경계 대비가 낮아 밑창 접촉선 검출에 오차가 발생합니다.<br/>"
        "<b>4) 정면 시점의 원근 왜곡:</b> 뒤꿈치가 앞발가락 뒤에 위치하므로 수직 상승 변위가 광축 압축 왜곡에 의해 가려집니다 (View 3 F1 72.10% 최저)."
    )
    story.append(Paragraph(c4_causes_text, styles['body']))

    story.append(Paragraph("5. 외부 AI(ChatGPT/Claude)에게 의뢰할 5대 핵심 기술 자문", styles['h1']))
    for q_t, q_b in q_c4:
        add_pdf_callout(story, q_t, q_b, styles, bg_hex="#F8FAFC", border_hex="#2B6CB0")

    pdf_doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  [+] PDF (.pdf) saved: {pdf_file}")


# ===========================================================================
# Main Execution
# ===========================================================================
def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print("=" * 80)
    print("  vidio-health-trainer: c1, c2, c4 외부 AI 기술 자문 문서 생성 시작")
    print(f"  대상 디렉터리: {DEST_DIR}")
    print("=" * 80)
    
    build_c1_documents()
    build_c2_documents()
    build_c4_documents()
    
    print("\n" + "=" * 80)
    print("  [완료] 총 6개 파일 (DOCX 3개 + PDF 3개) 배포 완료:")
    print(f"  1. c1 (척추 중립)   : {os.path.join(DEST_DIR, 'c1 상황.docx')} | {os.path.join(DEST_DIR, 'c1 상황.pdf')}")
    print(f"  2. c2 (고개 정면)   : {os.path.join(DEST_DIR, 'c2 상황.docx')} | {os.path.join(DEST_DIR, 'c2 상황.pdf')}")
    print(f"  3. c4 (발바닥 고정) : {os.path.join(DEST_DIR, 'c4 상황.docx')} | {os.path.join(DEST_DIR, 'c4 상황.pdf')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
