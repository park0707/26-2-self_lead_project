# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d2_misclassification_analysis_report_docx.py
Purpose: 바벨 데드리프트 d2 v5.0 오분류 데이터(FP 46건, FN 28건, Decline 35건)
         전수 심층 원인 분석 및 해결책 제시 공식 Word 보고서(.docx) 생성
"""

import os
import sys
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

sys.stdout.reconfigure(encoding='utf-8')

DOCX_DIR = r"D:\자주프 문서\6주차"
os.makedirs(DOCX_DIR, exist_ok=True)
DOCX_PATH = os.path.join(DOCX_DIR, "바벨_데드리프트_d2_오분류_데이터_심층원인분석_보고서.docx")

doc = Document()

for s in doc.sections:
    s.top_margin = Inches(0.8)
    s.bottom_margin = Inches(0.8)
    s.left_margin = Inches(0.8)
    s.right_margin = Inches(0.8)

NAVY = RGBColor(26, 54, 93)     # #1A365D
BLUE = RGBColor(43, 108, 176)   # #2B6CB0
GRAY = RGBColor(113, 128, 150)  # #718096
BLACK = RGBColor(45, 55, 72)    # #2D3748
WHITE = RGBColor(255, 255, 255)
RED = RGBColor(155, 44, 44)     # #9B2C2C

def style_heading(p, text, level=1):
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.bold = True
    if level == 1:
        run.font.size = Pt(16)
        run.font.color.rgb = NAVY
    elif level == 2:
        run.font.size = Pt(13)
        run.font.color.rgb = BLUE
    elif level == 3:
        run.font.size = Pt(11)
        run.font.color.rgb = BLACK

def add_callout_box(doc, text_list, title="", border_color="9B2C2C", bg_color="FFF5F5"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
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
        run_t.font.color.rgb = RED
    for line in text_list:
        run = p.add_run(line + "\n")
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9.5)
        run.font.color.rgb = BLACK
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

# 타이틀
p_meta = doc.add_paragraph()
r_meta = p_meta.add_run("vidio-health-trainer | AI-Hub 바벨 데드리프트 d2 오분류 정밀 감사 보고서")
r_meta.font.name = "맑은 고딕"
r_meta.font.size = Pt(9.5)
r_meta.font.color.rgb = GRAY

p_title = doc.add_paragraph()
p_title.paragraph_format.space_before = Pt(4)
p_title.paragraph_format.space_after = Pt(6)
r_title = p_title.add_run("바벨 데드리프트 d2(동적 무릎 외반) 모델의\n오분류(FP/FN) 전수 추적 및 근원적 발생 원인 심층 분석 보고서")
r_title.font.name = "맑은 고딕"
r_title.font.size = Pt(18)
r_title.bold = True
r_title.font.color.rgb = NAVY

add_callout_box(doc, [
    "• 문서 목적: d2 v5.0 최종 모델의 5-Fold StratifiedGroupKFold 실측 데이터(총 385개) 중 오분류된 74건(FP 46건, FN 28건) 및 판정 보류 35건을 전수 감사하여 '왜 오분류가 발생하는지, 진짜 원인이 무엇인지' 과학적으로 규명함.",
    "• 핵심 결론 1 [단안 3D 시점 비대칭 왜곡]: 우측 전방(view2)의 FP 19건(41%)과 좌측 전방(view4)의 FN 14건(50%)은 MediaPipe의 45도 시선 축 단안 깊이 압축/팽창 편향(동일 프레임 무릎 간격 14.5cm vs 25.6cm, 11.1cm 괴리)에 기인함.",
    "• 핵심 결론 2 [체형 도메인 편차]: 피험자 6번이 FP 12건(26%)을 독점 발생시켰으며, 이는 골반(18.4cm)과 무릎(23.3cm)이 극도로 좁은 '내로우 스탠스 왜소 체형'으로 인해 정상 동작이 타인의 외반 결함 수치와 중첩되었기 때문임.",
    "• 핵심 결론 3 [복합 결함 연쇄 오염]: d1(척추 굽힘) 결함으로 상체가 무너지며 무릎이 보상 붕괴를 일으킨 실제 움직임을 AI가 감지했으나, 라벨러가 주 결함(d1)만 오류로 체크하고 d2는 정상(True)으로 표기한 라벨링 편향이 확인됨.",
    "• 핵심 결론 4 [정면(view3)의 무결성]: 시선 왜곡이 없는 정면 시점에서는 결함 18건 전수 100% 감지(FN 0건)를 달성하여 알고리즘 자체의 역학적 유효성은 완벽히 입증됨."
], title="오분류 심층 감사 요약 (Executive Summary)")

# 제1장
p1 = doc.add_paragraph()
style_heading(p1, "1. 오분류 및 판정 보류 데이터 전수 통계", level=1)

p1_desc = doc.add_paragraph()
p1_desc.add_run("AI-Hub 바벨 데드리프트 385개 전수 시퀀스에 대해 최적 3-State 윈도우 [0.320, 0.400]를 적용했을 때의 혼동 행렬 및 피험자/시점별 교차 분포는 다음과 같습니다:")

tbl_stat = doc.add_table(rows=6, cols=4)
tbl_stat.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_st = ["판정 상태 (Category)", "샘플 수 (건)", "전체 비율 (%)", "비고"]
for c_idx, h in enumerate(headers_st):
    cell = tbl_stat.cell(0, c_idx)
    cell.paragraphs[0].text = h
    cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
    cell.paragraphs[0].runs[0].font.size = Pt(9.5)
    cell.paragraphs[0].runs[0].font.bold = True
    cell.paragraphs[0].runs[0].font.color.rgb = WHITE
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="1A365D"/>'))

stat_data = [
    ("진양성 (TP, True Positive)", "52", "13.5%", "결함 확정 적발 (정면 시점 Recall 100%)"),
    ("진음성 (TN, True Negative)", "224", "58.2%", "정상 확정 완료"),
    ("위양성 (FP, False Positive)", "46", "11.9%", "오탐: 정상인데 결함으로 오판 (피험자 6 및 view2 집중)"),
    ("위음성 (FN, False Negative)", "28", "7.3%", "미탐: 결함인데 정상으로 오판 (view4에 50% 집중)"),
    ("판정 보류 (DECLINE, Retake)", "35", "9.1%", "모호 구간 재촬영 유도 (10% 이하 엄격 제약 준수)")
]

for r_idx, r in enumerate(stat_data, 1):
    for c_idx, val in enumerate(r):
        cell = tbl_stat.cell(r_idx, c_idx)
        cell.paragraphs[0].text = val
        cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        cell.paragraphs[0].runs[0].font.color.rgb = BLACK
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{"FFFFFF" if r_idx%2==1 else "F7FAFC"}"/>'))

doc.add_paragraph().paragraph_format.space_after = Pt(4)

# 제2장
p2 = doc.add_paragraph()
style_heading(p2, "2. 진짜 원인 1: 단안 3D 시점 비대칭 왜곡 (View2 vs View4의 11.1cm 괴리)", level=1)

p2_desc = doc.add_paragraph()
p2_desc.add_run("전체 5대 시점 중 우측 전방(view2)은 FP 19건(41%)이 집중되고, 대칭 위치인 좌측 전방(view4)은 FN 14건(50%)이 집중되는 극단적 대칭 붕괴가 관찰되었습니다. 동일 피험자의 동일 1회 반복 프레임 30을 실측 분석한 결과는 다음과 같습니다:")

tbl_vcomp = doc.add_table(rows=4, cols=4)
tbl_vcomp.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_vc = ["관측 항목 (동일 반복 #282 프레임 30)", "시점 B (우측 전방 view2)", "시점 D (좌측 전방 view4)", "실측 괴리도 (Delta)"]
for c_idx, h in enumerate(headers_vc):
    cell = tbl_vcomp.cell(0, c_idx)
    cell.paragraphs[0].text = h
    cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
    cell.paragraphs[0].runs[0].font.size = Pt(9.5)
    cell.paragraphs[0].runs[0].font.bold = True
    cell.paragraphs[0].runs[0].font.color.rgb = WHITE
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="2B6CB0"/>'))

vcomp_data = [
    ("무릎 간격 (Knee Width)", "14.5 cm", "25.6 cm", "+11.1 cm (1.76배 차이!)"),
    ("무릎/발목 비 (K/A Ratio)", "0.771 (외반 붕괴로 보임)", "1.225 (외반 없음으로 보임)", "+0.454 (정반대 판정 유발)"),
    ("모델 최종 결함 확률 P(Defect)", "0.699 -> 결함 오판 (FP)", "0.070 -> 정상 오판 (FN)", "Delta = 0.629 (동일 동작에서 확률 극단 분열)")
]

for r_idx, r in enumerate(vcomp_data, 1):
    for c_idx, val in enumerate(r):
        cell = tbl_vcomp.cell(r_idx, c_idx)
        cell.paragraphs[0].text = val
        cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        cell.paragraphs[0].runs[0].font.color.rgb = BLACK
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{"FFFFFF" if r_idx%2==1 else "F7FAFC"}"/>'))

doc.add_paragraph().paragraph_format.space_after = Pt(2)

p2_exp = doc.add_paragraph()
p2_exp.add_run(
    "■ 원인 메커니즘: 단일 2D RGB 카메라로 3차원 관절을 추정하는 MediaPipe Pose는 대각선 45도 시점에서 앞다리와 뒷다리가 시선 축 상에 겹칠 때 단안 깊이 모호성(Monocular Depth Ambiguity)을 겪습니다. "
    "우측 전방(view2)에서는 원근 투영에 의해 두 무릎 사이 간격을 14.5cm로 과도하게 압축하여 정상 동작을 결함(FP)으로 오판하게 만들며, "
    "좌측 전방(view4)에서는 반대로 25.6cm로 팽창 추정하여 실제 무릎이 모인 결함을 정상(FN)으로 오판하게 만듭니다. "
    "반면 정면(view3)은 카메라 광축이 관상면과 수직이므로 투영 왜곡이 없어 FN = 0건(전수 감지)을 기록했습니다."
)
p2_exp.paragraph_format.space_after = Pt(6)

# 제3장
p3 = doc.add_paragraph()
style_heading(p3, "3. 진짜 원인 2: 피험자 6번의 내로우 스탠스 왜소 체형 편차", level=1)

p3_desc = doc.add_paragraph()
p3_desc.add_run("피험자별 오분류 통계에서 피험자 6번은 정상 샘플 35개 중 무려 12개를 결함으로 오탐(전체 FP의 26% 독점)시켰습니다. 9명 피험자의 생체역학 실측 치수를 전수 비교한 결과는 다음과 같습니다:")

tbl_subj = doc.add_table(rows=10, cols=5)
tbl_subj.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_sb = ["피험자 번호", "골반 너비 (Hip)", "무릎 간격 (Knee)", "발목 간격 (Ankle)", "FP 발생 건수"]
for c_idx, h in enumerate(headers_sb):
    cell = tbl_subj.cell(0, c_idx)
    cell.paragraphs[0].text = h
    cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
    cell.paragraphs[0].runs[0].font.size = Pt(9.5)
    cell.paragraphs[0].runs[0].font.bold = True
    cell.paragraphs[0].runs[0].font.color.rgb = WHITE
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="1A365D"/>'))

subj_rows = [
    ("피험자 1", "18.4 cm", "26.2 cm", "28.4 cm", "6건"),
    ("피험자 2", "20.1 cm", "26.3 cm", "28.6 cm", "4건"),
    ("피험자 3", "20.5 cm", "26.7 cm", "29.1 cm", "5건"),
    ("피험자 4", "20.7 cm", "25.5 cm", "26.1 cm", "2건"),
    ("피험자 5", "21.3 cm", "28.9 cm", "29.4 cm", "1건"),
    ("피험자 6 (이상치)", "18.4 cm", "23.3 cm (최저)", "24.0 cm (최저)", "12건 (독점 폭증!)"),
    ("피험자 7", "19.2 cm", "25.6 cm", "28.6 cm", "5건"),
    ("피험자 8", "20.1 cm", "26.5 cm", "29.5 cm", "7건"),
    ("피험자 9", "20.2 cm", "26.8 cm", "28.0 cm", "4건")
]

for r_idx, r in enumerate(subj_rows, 1):
    for c_idx, val in enumerate(r):
        cell = tbl_subj.cell(r_idx, c_idx)
        cell.paragraphs[0].text = val
        cell.paragraphs[0].runs[0].font.name = "맑은 고딕"
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        if "피험자 6" in r[0]:
            cell.paragraphs[0].runs[0].font.bold = True
            cell.paragraphs[0].runs[0].font.color.rgb = RED
        else:
            cell.paragraphs[0].runs[0].font.color.rgb = BLACK
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{"FFF5F5" if "피험자 6" in r[0] else ("FFFFFF" if r_idx%2==1 else "F7FAFC")}"/>'))

doc.add_paragraph().paragraph_format.space_after = Pt(2)

p3_exp = doc.add_paragraph()
p3_exp.add_run(
    "■ 원인 메커니즘: 피험자 6번은 발목 간격이 24.0cm, 무릎 간격이 23.3cm로 타 피험자 평균(발목 28.5cm, 무릎 26.5cm)보다 무려 3~4.5cm 좁은 극단적 내로우 스탠스(Narrow Stance) 습관을 가지고 있습니다. "
    "이로 인해 피험자 6번의 정상 동작 시 무릎 간격(23.3cm)이 타 피험자가 실제로 무릎을 안으로 모은 결함 상태(23~24cm)와 수치상 완전히 겹치게 되었으며, "
    "절대 거리 피처(world_kawr_min 등)가 이를 결함으로 오탐지한 것입니다."
)
p3_exp.paragraph_format.space_after = Pt(6)

# 제4장
p4 = doc.add_paragraph()
style_heading(p4, "4. 진짜 원인 3: 복합 결함 연쇄 오염과 라벨링 목적 편향", level=1)

p4_desc = doc.add_paragraph()
p4_desc.add_run(
    "피험자 6번 및 8번의 오탐(FP) 사례 다수는 AI-Hub 촬영 프로토콜 상 'd1: 척추 중립 위반(Round Back)'을 유도한 시퀀스였습니다. "
    "피험자가 고중량 바벨을 들어올리며 허리가 심하게 굽어질 때, 상체가 전방으로 무너지며 무게중심을 지탱하기 위해 무릎이 안쪽으로 보상 붕괴(Compensatory Knee Collapse)를 일으킵니다. "
    "실제 비전 AI 센서는 무릎이 모인 물리적 현상을 정확하게 감지하여 결함 확률 0.60~0.70을 부여했으나, "
    "AI-Hub 라벨러는 촬영 과제 목적이 d1(척추)이었으므로 d2(발과 무릎 방향 일치)는 True(정상)로 단순 기계적 표기를 하였습니다. "
    "즉, 모델이 틀린 것이 아니라 라벨러의 단일 목적 라벨링 편향에 의해 정밀한 물리 감지가 FP로 오분류된 것입니다."
)

# 제5장
p5 = doc.add_paragraph()
style_heading(p5, "5. 향후 근원적 해결을 위한 3대 솔루션", level=1)

add_callout_box(doc, [
    "1. [시점별 캘리브레이션 및 3-State 재촬영 가이드 연동]:",
    "   - view2(우측 전방)와 view4(좌측 전방)의 11.1cm 왜곡을 해결하기 위해, 사용자에게 '가급적 정면(시점 C) 또는 우측 45도에서 수행'하도록 UX 안내를 고정하고, 45도 시점에서는 시점별 확률 보정 계수(view-specific temperature scaling)를 적용함.",
    "2. [골반-발목 너비 연동 무차원화 (Dimensionless Normalization)]: ",
    "   - 피험자 6번과 같은 내로우 스탠스 왜소 체형에 대응하기 위해, 절대 센티미터(cm) 피처 의존도를 낮추고 '개인별 셋업 정지 자세 기준 무릎 폭 변화율(Delta KAWR)'의 가중치를 추가 강화함.",
    "3. [다중 라벨 연쇄 감지 (Cascading Multi-label Filter)]: ",
    "   - d1(척추 결함)이 감지된 시퀀스에서는 상체 붕괴에 수반된 2차적 무릎 흔들림을 주 결함의 종속 증상으로 통합 보고하는 메타러너 룰을 적용하여 불필요한 FP를 억제함."
], title="최종 기술 권고사항", border_color="2B6CB0", bg_color="F7FAFC")

doc.save(DOCX_PATH)
print(f"[+] 오분류 심층 원인 분석 Word 보고서 저장 완료: {DOCX_PATH}")
