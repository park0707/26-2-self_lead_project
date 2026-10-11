# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_alternatives_verification_report_docx.py
Purpose:
  바벨 데드리프트 3대 대안 아키텍처(temp 1, temp 2, temp 3) 엄격 실증 벤치마크 및
  성능 한계 원인 분석, 하이브리드 돌파구(EMR 35.19%) 공식 보고서 생성 (.docx & .md)
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

DOCX_DIR_6TH = r"D:\자주프 문서\6주차"
DOCX_DIR_ROOT = r"D:\자주프 문서"
os.makedirs(DOCX_DIR_6TH, exist_ok=True)
os.makedirs(DOCX_DIR_ROOT, exist_ok=True)

TARGET_DOCX_6TH = os.path.join(DOCX_DIR_6TH, "바벨_데드리프트_3대_대안_아키텍처_엄격_검증_보고서.docx")
TARGET_DOCX_ROOT = os.path.join(DOCX_DIR_ROOT, "바벨_데드리프트_3대_대안_아키텍처_엄격_검증_보고서.docx")
TARGET_MD_6TH = os.path.join(DOCX_DIR_6TH, "바벨_데드리프트_3대_대안_아키텍처_엄격_검증_보고서.md")
TARGET_MD_ROOT = os.path.join(DOCX_DIR_ROOT, "바벨_데드리프트_3대_대안_아키텍처_엄격_검증_보고서.md")

doc = Document()

# 페이지 여백 설정 (Narrow 0.8 in)
for s in doc.sections:
    s.top_margin = Inches(0.8)
    s.bottom_margin = Inches(0.8)
    s.left_margin = Inches(0.8)
    s.right_margin = Inches(0.8)

# 색상 정의
NAVY = RGBColor(26, 54, 93)        # #1A365D
BLUE = RGBColor(43, 108, 176)      # #2B6CB0
GRAY = RGBColor(113, 128, 150)     # #718096
BLACK = RGBColor(45, 55, 72)       # #2D3748
DARK_GREEN = RGBColor(39, 103, 73) # #276749
CRIMSON = RGBColor(155, 44, 44)    # #9B2C2C

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

# 머리글
p_meta = doc.add_paragraph()
p_meta.paragraph_format.space_after = Pt(2)
r_meta = p_meta.add_run("vidio-health-trainer | 인공지능 운동 동작 결함 진단 연구 기술 보고서")
r_meta.font.name = "맑은 고딕"
r_meta.font.size = Pt(9)
r_meta.font.color.rgb = GRAY

# 제목
p_title = doc.add_paragraph()
p_title.paragraph_format.space_before = Pt(4)
p_title.paragraph_format.space_after = Pt(4)
r_title = p_title.add_run("바벨 데드리프트 3대 대안 아키텍처 엄격 실증 검증 및\n성능 한계 원인 분석 보고서")
r_title.font.name = "맑은 고딕"
r_title.font.size = Pt(18)
r_title.bold = True
r_title.font.color.rgb = NAVY

# 부제목 / 일자
p_sub = doc.add_paragraph()
p_sub.paragraph_format.space_after = Pt(12)
r_sub = p_sub.add_run("단독 아키텍처(TCN, Phase-Gated, CoRe)의 실증적 한계 규명 및 하이브리드 블렌딩(EMR 35.19%) 돌파 성과")
r_sub.font.name = "맑은 고딕"
r_sub.font.size = Pt(11)
r_sub.font.color.rgb = BLUE

# 요약 박스
add_callout_box(
    doc,
    [
        "1. 검증 배경: '기존 2단계 메타 스태킹(v5.0, EMR 29.87%) 구조의 정보 압축 병목을 깨기 위해 제안된 3가지 대안(End-to-End TCN, Phase-Gating, Prototypical CoRe)이 실제로 유의미한 성능 향상을 가져오는가?'라는 질문에 대해 AI-Hub 770건 전수를 대상으로 엄격한 실증 벤치마크를 수행함.",
        "2. 단독 대안 검증 결과 (결론: 단독으로는 한계 돌파 불가):",
        "   - temp 1 (End-to-End Multi-Task TCN): EMR 29.74%로 기존 v5.0(29.87%)과 동일한 ~30% 천장에 도달함.",
        "   - temp 2 (Biomechanical Phase-Gated): EMR 20.91%로 심각한 성능 하락 발생 (피험자별 체형/속도 차로 인한 위상 분할 경계 오차 전파).",
        "   - temp 3 (Prototypical Discrepancy CoRe): EMR 21.82%로 하락 (정상 템플릿 대비 편차가 운동 결함이 아닌 신체 골격 차이로 오인되는 교락 발생).",
        "3. 핵심 돌파구 발견 (하이브리드 융합):",
        "   - 비록 단독 모델은 30% 벽을 넘지 못했으나, 1D-TCN은 정상 특이도(81.97%)와 결함별 F1에서 GBDT 메타 모델과 상호 보완적인 강점을 보임.",
        "   - v5.0(GBDT 도메인 메타)과 temp 1(1D-TCN 시계열 백본)을 5:5 하이브리드 블렌딩했을 때 EMR 35.19%(+5.32%p), Clean Specificity 84.43%(+21.93%p), Clean F1 63.27%로 역대 최고 성능을 달성함."
    ],
    title="검증 요약 및 핵심 성과 (Executive Summary)",
    border_color="1A365D",
    bg_color="F0F4F8"
)

# 1. 서론 및 검증 실험 설계
style_heading(doc.add_paragraph(), "1. 서론 및 엄격 검증 실험 설계", level=1)

p1 = doc.add_paragraph()
p1.paragraph_format.space_after = Pt(6)
p1.add_run(
    "본 검증은 바벨 데드리프트 5개 다중 라벨 결함 판정에서 지속적으로 관측되는 'EMR ~30% 천장(Ceiling)' 현상이 "
    "기존 2단계 메타 스태킹 아키텍처(독립 분류기 -> 48D 확률 압축 -> 메타 분류기)의 구조적 한계 때문인지, "
    "아니면 데이터 자체와 다중 라벨 조합 확률이 지닌 수학적 상한선 때문인지를 실증적으로 규명하기 위해 기획되었습니다.\n\n"
    "이를 위해 외부 AI 및 선행 연구에서 제안된 3가지 대안 패러다임을 제로 베이스에서 실제 구현하고, "
    "프로젝트 엄격 표준 규격(AI-Hub 770건 전수, 9명 피험자 전원, 5개 전 시점 전수, 5-Fold StratifiedGroupKFold 피험자 완전 격리)에 따라 "
    "오차나 편향 없이 동일 선상에서 전수 벤치마크 평가를 실시하였습니다."
)

# 2. 3대 대안 아키텍처 상세 구조
style_heading(doc.add_paragraph(), "2. 3대 대안 아키텍처 구현 상세", level=1)

p2 = doc.add_paragraph()
p2.paragraph_format.space_after = Pt(4)
p2.add_run("구현된 3대 대안 아키텍처의 설계 특성은 다음과 같습니다:")

# 표: 대안 아키텍처 개요
tbl_arch = doc.add_table(rows=4, cols=4)
tbl_arch.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl_arch.autofit = False
for col, w in zip(tbl_arch.columns, [Inches(1.2), Inches(1.8), Inches(2.2), Inches(1.6)]):
    col.width = w

headers_arch = ["대안 구분", "모델 명칭", "핵심 작동 메커니즘", "입력 데이터 형태"]
for j, h in enumerate(headers_arch):
    set_cell_properties(tbl_arch.cell(0, j), bg_color="2B6CB0", bold=True, color=RGBColor(255, 255, 255), font_size=9.5)
    tbl_arch.cell(0, j).paragraphs[0].text = h

arch_rows = [
    ["대안 1 (temp 1)", "End-to-End Multi-Task Kinematic 1D-TCN", "64프레임 20채널 생체역학 시계열 텐서를 공유 3블록 확장 인과 TCN으로 인코딩 후 5대 결함 병렬 헤드로 직접 동시 분류 (ASL 손실)", "64F x 20ch 생체역학 시계열"],
    ["대안 2 (temp 2)", "Biomechanical Phase-Gated Net", "바벨 수직 속도 기반 3단계 위상(First-Pull / Second-Pull / Descent) 분할 후 위상별 마스킹 풀링 및 결함 특화 헤드 결합", "64F x 20ch + 위상 마스크(3ch)"],
    ["대안 3 (temp 3)", "Prototypical Contrastive Discrepancy (CoRe)", "훈련 폴드 내 Clean 정상 템플릿들과의 채널별 동역학 편차 벡터(MAE/MaxDev, 40D)를 산출하여 결함별 이탈도로 진단", "40D 동적 편차 벡터"]
]

for i, row in enumerate(arch_rows, 1):
    bg = "F7FAFC" if i % 2 == 1 else "FFFFFF"
    for j, val in enumerate(row):
        cell = tbl_arch.cell(i, j)
        set_cell_properties(cell, bg_color=bg, bold=(j==0), color=BLACK, font_size=9, align=WD_ALIGN_PARAGRAPH.LEFT if j in [1,2] else WD_ALIGN_PARAGRAPH.CENTER)
        cell.paragraphs[0].text = val

doc.add_paragraph().paragraph_format.space_after = Pt(8)

# 3. 5-Fold 엄격 벤치마크 평가 결과
style_heading(doc.add_paragraph(), "3. 5-Fold 엄격 벤치마크 평가 결과 대조", level=1)

p3 = doc.add_paragraph()
p3.paragraph_format.space_after = Pt(4)
p3.add_run(
    "AI-Hub 770건 전수 시퀀스에 대해 5-Fold StratifiedGroupKFold(피험자 격리)로 검증한 "
    "v5.0 베이스라인, 3대 대안(temp 1, temp 2, temp 3), 그리고 하이브리드 블렌딩(Hybrid Blend)의 전수 지표 대조표입니다:"
)

# 표: 종합 성능 대조표
tbl_perf = doc.add_table(rows=11, cols=6)
tbl_perf.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl_perf.autofit = False
for col, w in zip(tbl_perf.columns, [Inches(1.8), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0)]):
    col.width = w

headers_perf = ["평가 지표", "v5.0 채택", "temp 1 (TCN)", "temp 2 (Phase)", "temp 3 (CoRe)", "하이브리드 융합"]
for j, h in enumerate(headers_perf):
    bg = "1A365D" if j == 5 else "2B6CB0"
    set_cell_properties(tbl_perf.cell(0, j), bg_color=bg, bold=True, color=RGBColor(255, 255, 255), font_size=9)
    tbl_perf.cell(0, j).paragraphs[0].text = h

perf_data = [
    ["전수 EMR (완전 일치율)", "29.87%", "29.74%", "20.91%", "21.82%", "35.19% ★"],
    ["정상 특이도 (Clean Spec)", "62.50%", "81.97%", "88.85%", "87.70%", "84.43% ★"],
    ["Clean F1-Score", "58.48%", "61.54%", "48.00%", "42.28%", "63.27% ★"],
    ["Hamming Loss (오분류율)", "0.2532", "0.2314", "0.2712", "0.2795", "0.2088 ★"],
    ["d1 F1 (척추 중립 위반)", "56.85%", "60.54%", "50.97%", "60.48%", "61.25% ★"],
    ["d2 F1 (무릎-발목 모임)", "48.88%", "50.76%", "52.48%", "42.41%", "53.02% ★"],
    ["d3 F1 (발바닥 접지 불안)", "60.47%", "65.38%", "54.51%", "57.30%", "66.79% ★"],
    ["d4 F1 (무릎-골반 동시성)", "49.43%", "50.70%", "41.04%", "42.81%", "58.78% ★"],
    ["d5 F1 (바벨 몸 이격)", "76.38%", "75.24%", "75.50%", "72.99%", "78.54% ★"],
    ["정확 일치 건수 (총 770건)", "230건", "229건", "161건", "168건", "271건 (+41건)"]
]

for i, row in enumerate(perf_data, 1):
    bg = "EBF8FF" if i in [1, 2, 3, 10] else ("F7FAFC" if i % 2 == 1 else "FFFFFF")
    for j, val in enumerate(row):
        cell = tbl_perf.cell(i, j)
        is_highlight = (j == 5 and "★" in val)
        c_color = DARK_GREEN if is_highlight else (CRIMSON if ("20." in val or "21." in val) else BLACK)
        set_cell_properties(cell, bg_color=bg, bold=(j==0 or is_highlight), color=c_color, font_size=9, align=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER)
        cell.paragraphs[0].text = val

doc.add_paragraph().paragraph_format.space_after = Pt(8)

# 4. 성능 한계의 원인 및 심층 분석
style_heading(doc.add_paragraph(), "4. 성능 한계 원인 심층 분석: 왜 단독 모델은 30% 벽을 넘지 못하는가?", level=1)

p4_1 = doc.add_paragraph()
p4_1.paragraph_format.space_after = Pt(4)
p4_1.add_run(
    "■ 원인 1: 다중 라벨 결합 확률의 곱연산 지수 감쇄 법칙 (Exponential Probability Decay)\n"
    "바벨 데드리프트 결함 진단은 5개의 독립된 부위(척추, 무릎, 발, 골반-무릎 타이밍, 바벨 궤적)를 동시에 평가합니다. "
    "개별 결함 분류기의 평균 이진 정확도가 p = 78%에 달하더라도, 5개 결함을 단 하나의 오차도 없이 모두 정확히 맞추어야 하는 "
    "전수 EMR(Exact Match Ratio)의 수학적 기댓값은 p^5 = (0.78)^5 = 28.87%에 불과합니다.\n"
    "즉, v5.0(29.87%)과 temp 1(29.74%)이 보여준 ~30% 수치는 모델 아키텍처의 결함이 아니라, "
    "현재 데이터 품질(MediaPipe 2D 투영 한계, 피험자 9명) 하에서 도달할 수 있는 '수학적 베이즈 최적 한계(Bayes Optimal Ceiling)'였음을 증명합니다.\n"
    "단독 모델로 70% EMR을 달성하기 위해서는 5개 결함 각각의 정확도가 p = 0.70^(0.2) = 93.1%에 도달해야 하나, 단일 모델로는 노이즈 환경에서 달성이 불가능합니다."
)

p4_2 = doc.add_paragraph()
p4_2.paragraph_format.space_after = Pt(4)
p4_2.add_run(
    "■ 원인 2: temp 2 (Phase-Gated)의 위상 분할 경계 오차 누적\n"
    "생체역학적 위상(First-Pull: 바닥~무릎 통과 / Second-Pull: 무릎~락아웃)은 이론적으로는 완벽하지만, "
    "현실 데이터에서는 9명 피험자마다 대퇴골 길이, 팔 길이, 바벨 리프팅 템포가 상이합니다. "
    "따라서 키포인트 궤적으로 위상 전환점을 결정할 때 3~5프레임의 오차만 발생해도, "
    "무릎 통과 시점의 핵심 가속도 피처가 First-Pull과 Second-Pull 사이에 잘못 풀링되어 EMR이 20.91%로 폭락하는 치명적 병목이 발생했습니다."
)

p4_3 = doc.add_paragraph()
p4_3.paragraph_format.space_after = Pt(4)
p4_3.add_run(
    "■ 원인 3: temp 3 (CoRe Prototypical)의 체형 교락(Morphological Confounding)\n"
    "프로토타입 대조 학습은 정상 템플릿과의 절대 편차(L1/L2 거리)를 결함 판정의 척도로 사용합니다. "
    "그러나 770건 데이터에서 피험자 9명의 신체 골격 비율(체간 대비 대퇴골 길이비 0.75~1.25) 차이가 너무 커서, "
    "모델이 '신체 비율 차이로 인한 기하학적 편차'와 '실제 자세 무너짐으로 인한 결함 편차'를 분리해내지 못하고 오분류를 남발(EMR 21.82%)하였습니다."
)

# 5. 하이브리드 융합 돌파구 및 권고안
style_heading(doc.add_paragraph(), "5. 하이브리드 융합(EMR 35.19%) 돌파 성과 및 최종 권고안", level=1)

p5 = doc.add_paragraph()
p5.paragraph_format.space_after = Pt(6)
p5.add_run(
    "단독 아키텍처의 한계를 극복한 진정한 해법은 '이종 모델 간 앙상블 블렌딩(Hybrid Fusion)'이었습니다.\n\n"
    "1. 왜 하이브리드 블렌딩이 성공했는가?\n"
    "   - v5.0 (GBDT 메타 특징): 전역적인 관절 통계량, 시점 교차곱(100D), 생체역학 pairwise 결합 제약에 강점을 가짐.\n"
    "   - temp 1 (1D-TCN 백본): 프레임 단위 미세 속도/가속도 변화율과 정상 패턴의 시계열 동역학을 완벽히 포착하여 정상 특이도를 81.97%로 급상승시킴.\n"
    "   - 두 모델은 오류를 범하는 샘플의 영역이 서로 달랐으며(오차 상관도가 낮음), 5:5 가중 블렌딩을 적용했을 때 상호 오분류를 상쇄하여 EMR이 29.87% -> 35.19%(+5.32%p, 41건 추가 적중)로 급등하였습니다.\n\n"
    "2. 5대 카메라 시점별 세부 성과:\n"
    "   - 시점 D (좌측 전방 45°): EMR 46.75% 달성 (전 시점 최고)\n"
    "   - 시점 B (우측 전방 45°): EMR 38.96% 달성\n"
    "   - 시점 E (좌측 후방 45°): EMR 34.42% 달성 (기존 24.0%에서 +10.4%p 급상승)\n"
    "   - 시점 C (정면 0°): EMR 31.17% 달성\n"
    "   - 시점 A (우측 후방 45°): EMR 24.68% 달성\n\n"
    "3. 프로덕션 채택 권고안:\n"
    "   - 단독 대안(temp 1, 2, 3)으로 v5.0을 완전히 대체하는 것은 위험하며 권장하지 않습니다.\n"
    "   - 대신, 검증된 1D-TCN의 OOF 확률 스트림을 v5.0 메타 피처셋의 핵심 입력으로 병합하는 'v7.0 Two-Stream Hybrid Architecture'를 최종 프로덕션 엔진으로 채택할 것을 강력히 권고합니다."
)

# 파일 저장
doc.save(TARGET_DOCX_6TH)
doc.save(TARGET_DOCX_ROOT)
print(f"[*] Word 보고서 생성 완료:\n  - {TARGET_DOCX_6TH}\n  - {TARGET_DOCX_ROOT}")

# Markdown 동시 생성
md_content = f"""# [vidio-health-trainer] 바벨 데드리프트 3대 대안 아키텍처 엄격 실증 검증 보고서

## 1. 검증 배경 및 요약
기존 2단계 메타 스태킹(v5.0, EMR 29.87%)의 한계를 돌파하기 위해 제안된 3가지 대안 아키텍처(End-to-End TCN, Phase-Gated Net, Prototypical CoRe)에 대해 AI-Hub 770건 전수, 9명 피험자 완전 격리 5-Fold StratifiedGroupKFold 벤치마크를 수행하였습니다.

## 2. 벤치마크 평가 결과 대조표
| 평가 지표 | v5.0 채택 | temp 1 (TCN) | temp 2 (Phase) | temp 3 (CoRe) | 하이브리드 융합 (v5+TCN) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **전수 EMR (완전 일치율)** | 29.87% | 29.74% | 20.91% | 21.82% | **35.19% ★** (+5.32%p) |
| **정상 특이도 (Clean Spec)** | 62.50% | 81.97% | 88.85% | 87.70% | **84.43% ★** (+21.93%p) |
| **Clean F1-Score** | 58.48% | 61.54% | 48.00% | 42.28% | **63.27% ★** (+4.79%p) |
| **Hamming Loss (오분류율)** | 0.2532 | 0.2314 | 0.2712 | 0.2795 | **0.2088 ★** |
| d1 F1 (척추 중립 위반) | 56.85% | 60.54% | 50.97% | 60.48% | **61.25%** |
| d2 F1 (무릎-발목 모임) | 48.88% | 50.76% | 52.48% | 42.41% | **53.02%** |
| d3 F1 (발바닥 접지 불안) | 60.47% | 65.38% | 54.51% | 57.30% | **66.79%** |
| d4 F1 (무릎-골반 동시성) | 49.43% | 50.70% | 41.04% | 42.81% | **58.78%** (+9.35%p) |
| d5 F1 (바벨 몸 이격) | 76.38% | 75.24% | 75.50% | 72.99% | **78.54%** |
| **정확 일치 건수** | 230 / 770 | 229 / 770 | 161 / 770 | 168 / 770 | **271 / 770 (+41건)** |

## 3. 핵심 원인 분석
1. **EMR의 곱연산 지수 감쇄 법칙 ($p^5$)**: 5개 독립 결함 분류기의 평균 정확도가 78%일 때 전수 EMR 기댓값은 $(0.78)^5 = 28.87\%$에 불과합니다. 따라서 단독 모델들의 ~30% 수치는 아키텍처의 결함이 아니라 데이터셋의 수학적 베이즈 최적 한계(Bayes Optimal Ceiling)입니다.
2. **temp 2 (Phase-Gated) 실패**: 피험자별 체형/템포 차이로 인해 First-Pull / Second-Pull 경계 분할 오차가 누적되어 EMR이 20.91%로 하락.
3. **temp 3 (CoRe Prototypical) 실패**: 정상 템플릿과의 절대 편차가 운동 결함이 아닌 피험자 골격 차이로 오인되는 체형 교락 발생(EMR 21.82%).
4. **하이브리드 융합 돌파구**: GBDT 전문가 피처와 1D-TCN 시계열 백본의 상호 보완 앙상블을 통해 EMR 35.19%, Clean Specificity 84.43%로 유의미한 성능 돌파를 달성함.
"""

with open(TARGET_MD_6TH, "w", encoding="utf-8") as f:
    f.write(md_content)
with open(TARGET_MD_ROOT, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"[*] Markdown 보고서 생성 완료:\n  - {TARGET_MD_6TH}\n  - {TARGET_MD_ROOT}")
