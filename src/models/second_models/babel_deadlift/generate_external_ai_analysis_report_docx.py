# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_external_ai_analysis_report_docx.py
Purpose: 바벨 데드리프트 d2 모델 고도화를 위한 3대 외부 AI (Gemini, Perplexity, 아주LLM)
         자문 보고서 통합 심층 비교 분석 및 v5.0 개선 로드맵 Word(.docx) 보고서 생성
"""

import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

sys.stdout.reconfigure(encoding='utf-8')

DOCX_DIR = r"D:\자주프 문서\6주차"
os.makedirs(DOCX_DIR, exist_ok=True)
DOCX_PATH = os.path.join(DOCX_DIR, "바벨_데드리프트_d2_외부AI자문_통합분석_및_v5로드맵_보고서.docx")

doc = Document()

# 페이지 여백 설정 (Narrow)
for s in doc.sections:
    s.top_margin = Inches(0.8)
    s.bottom_margin = Inches(0.8)
    s.left_margin = Inches(0.8)
    s.right_margin = Inches(0.8)

# 색상 정의
NAVY = RGBColor(26, 54, 93)     # #1A365D
BLUE = RGBColor(43, 108, 176)   # #2B6CB0
GRAY = RGBColor(113, 128, 150)  # #718096
BLACK = RGBColor(45, 55, 72)    # #2D3748
WHITE = RGBColor(255, 255, 255)
CRIMSON = RGBColor(155, 44, 44) # #9B2C2C

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

def add_callout_box(doc, text_list, title="", border_color="2B6CB0", bg_color="F7FAFC"):
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
        run_t.font.color.rgb = BLUE
    for line in text_list:
        run = p.add_run(line + "\n")
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9.5)
        run.font.color.rgb = BLACK
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

# ----------------- 문서 본문 작성 -----------------

# 타이틀
p_meta = doc.add_paragraph()
p_meta.paragraph_format.space_after = Pt(2)
r_meta = p_meta.add_run("vidio-health-trainer | AI-Hub 바벨 데드리프트 d2 고도화 기술 분석 보고서")
r_meta.font.name = "맑은 고딕"
r_meta.font.size = Pt(9.5)
r_meta.font.color.rgb = GRAY

p_title = doc.add_paragraph()
p_title.paragraph_format.space_before = Pt(4)
p_title.paragraph_format.space_after = Pt(6)
r_title = p_title.add_run("바벨 데드리프트 d2(동적 무릎 외반) 모델 고도화를 위한\n외부 최상위 3대 AI 자문 통합 분석 및 v5.0 실천 로드맵")
r_title.font.name = "맑은 고딕"
r_title.font.size = Pt(19)
r_title.bold = True
r_title.font.color.rgb = NAVY

add_callout_box(doc, [
    "• 문서 목적: 자주프 문서/외부ai 응답 폴더 내 3대 최상위 AI(Gemini, Perplexity, 아주LLM)의 기술 자문 문서를 정밀 대조 분석하여 핵심 공통점, 비판적 결함 지적, 아키텍처 혁신안을 도출함",
    "• 적용 대상 모델: 바벨 데드리프트 결함 판정 모델 d2 ('발과 무릎의 방향 일치' / 무릎 외반 판별기)",
    "• 핵심 분석 결과 요약: 3대 AI 전원 '거대 모델 도입 배제 및 4k~5k 초경량 시계열 모델(Tiny TCN / Micro ST-GCN)의 GBDT Late-Fusion'에 합의하였으며, 특히 기존 3D FPPA의 굴곡-외반 혼동 결함 및 시점 D의 좌우 매핑 왜곡을 긴급 수정 과제로 선별함.",
    "• 목표 지표: 5-Fold StratifiedGroupKFold 기준 3-State F1-Score 65%~80% 돌파 및 보류율 10.0% 이하 엄격 달성"
], title="보고서 요약 (Executive Summary)")

# 제1장
p1 = doc.add_paragraph()
style_heading(p1, "1. 외부 3대 AI 자문 개요 및 핵심 입장 대조", level=1)

p1_desc = doc.add_paragraph()
p1_desc.add_run("바벨 데드리프트 d2 모델의 v4.0 실측 성적(ROC-AUC 0.7348, 3-State F1 55.10%, 보류율 9.9%)을 바탕으로 외부 최상위 AI에 요청한 5대 핵심 질문(Q1~Q5)에 대해 접수된 3개 자문 문서의 종합 분석 결과는 다음과 같습니다:")

tbl1 = doc.add_table(rows=6, cols=4)
tbl1.alignment = WD_TABLE_ALIGNMENT.CENTER
headers1 = ["자문 항목 (RFC)", "Gemini (gemini_for_d2)", "Perplexity (deadlift-d2-f1)", "아주LLM (새 채팅 _ 아주LLM)"]
for col_idx, h in enumerate(headers1):
    c = tbl1.cell(0, col_idx)
    c.paragraphs[0].text = h
    c.paragraphs[0].runs[0].font.name = "맑은 고딕"
    c.paragraphs[0].runs[0].font.size = Pt(9.5)
    c.paragraphs[0].runs[0].font.bold = True
    c.paragraphs[0].runs[0].font.color.rgb = WHITE
    tcPr = c._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="1A365D"/>'))

data1 = [
    ("Q1. 2~5cm 미세 변위 포착 피처", "3D 평면 법선 비틀림 각도(Twist) 및 FPPA 각가속도 2차 미분 피크", "부호 있는(Signed) 내측 잔차, 시간 정합 적분 결손량, 좌우 비대칭도", "관상면 투영 부호 외반각, 발 진행방향 대비 무릎 편위, 지속성 비"),
    ("Q2. 소규모(N=385) 최적 아키텍처", "하체 6관절 Micro ST-GCN (5k 미만) + GBDT OOF Late-Fusion", "15관절 Tiny ST-GCN (4k~20k) 또는 경량 TCN의 GBDT 융합", "4,162 파라미터 8채널 Tiny TCN + regularized GBDT 로지스틱 융합"),
    ("Q3. F1 직접 최적화 손실함수", "Asymmetric Soft-F1 Loss (BCE 탈피, 정밀도-재현율 동시 견인)", "Asymmetric Focal Loss + Soft-F1 결합 손실 (가중 분리 탐색)", "BCE 대비 AF + Soft-F1 점진 결합 (가중치 중복 적용 절대 경계)"),
    ("Q4. 후방/사각 시점 급락 대응", "Gradient Reversal Layer(GRL) 기반 View-Adversarial 학습", "공유-시점 잔차 분리 헤드(Shared/View-Residual) 및 품질 라우터", "동일 Rep 정면-사각 간 Feature Alignment 및 시점 D 좌우 부호 감사"),
    ("Q5. 물리적 무결성 데이터 증강", "Phase-Aware Valgus Injection (Hanning Window 2~4cm 무릎 주입)", "Bone-preserving Jitter, Temporal Warping, Stance/Yaw 회전", "동일 시간 재매핑, 상관 노이즈, 엄격한 좌우 대칭 반전 (인위주입 경계)")
]

for row_idx, r in enumerate(data1, 1):
    for col_idx, val in enumerate(r):
        c = tbl1.cell(row_idx, col_idx)
        c.paragraphs[0].text = val
        c.paragraphs[0].runs[0].font.name = "맑은 고딕"
        c.paragraphs[0].runs[0].font.size = Pt(8.5)
        c.paragraphs[0].runs[0].font.color.rgb = BLACK
        tcPr = c._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{"FFFFFF" if row_idx%2==1 else "F7FAFC"}"/>'))

doc.add_paragraph().paragraph_format.space_after = Pt(4)

# 제2장
p2 = doc.add_paragraph()
style_heading(p2, "2. 외부 AI가 밝혀낸 기존 모델의 치명적 결함과 원인 분석", level=1)

add_callout_box(doc, [
    "1. [기존 3D FPPA의 굴곡-외반 혼동 오류 (아주LLM & Perplexity 공통 지적)]",
    "   - 현상: v4.0에서 사용한 180° - ∠(Hip, Knee, Ankle)를 3차원 공간각으로 계산하면, 데드리프트 본연의 무릎 굴곡(Flexion)이 외반(Valgus) 각도로 크게 오인입됨.",
    "   - 해결: 반드시 직교화된 관상면(Coronal Plane)에 대퇴-하퇴 벡터를 투영한 뒤 '부호 있는 2D 투영 외반각'으로 분리 산출해야 함.",
    "",
    "2. [d2 정답 라벨의 본질 누락: '발과 무릎의 방향 일치' 미반영 (아주LLM 지적)]",
    "   - 현상: 현재 피처들은 무릎 사이 거리(KAWR)나 골반 대비 안쪽 이동만 측정함. 그러나 데드리프트 정답 라벨은 '발끝 진행 방향 대비 무릎의 정렬 상태'임.",
    "   - 해결: 발뒤꿈치(Heel)와 발끝(Foot Index)이 이루는 지면 투영 벡터와 무릎-발목 벡터 간의 상대 편위를 측정하는 피처 도입 필수.",
    "",
    "3. [시점 D (좌측 전방 45°)의 미스터리한 성능 붕괴 원인 (아주LLM & Perplexity 분석)]",
    "   - 현상: 우측 전방인 시점 B는 Recall 94.12%로 결함을 거의 완벽히 감지하나, 좌측 전방인 시점 D는 Recall 20.00% (F1 30.0%)로 심각하게 붕괴됨.",
    "   - 분석: 이는 시점 자체의 정보 결여라기보다는 '좌우 관절 반전 시 내측 부호(Sign)의 반전 누락' 또는 '카메라 시점 뷰 매핑 오류'일 가능성이 90% 이상임.",
    "",
    "4. [Cleanlab 과도한 프루닝(50개 = 13%)에 따른 검증 왜곡 경고]",
    "   - 현상: 385개 중 50개를 라벨 노이즈로 보고 학습에서 제외했으나, 사각 시점의 어려운 결함 표본이 노이즈로 오인되어 삭제되었을 위험 존재.",
    "   - 해결: 검증(Outer Fold)은 절대 무삭제 385개 전수로 평가하고, 프루닝은 학습 폴드 내부에서만 제한적으로 활용."
], title="핵심 기술 진단 4선", border_color="9B2C2C", bg_color="FFF5F5")

# 제3장
p3 = doc.add_paragraph()
style_heading(p3, "3. 3대 AI가 합의한 d2 v5.0 아키텍처 혁신 설계", level=1)

p3_desc = doc.add_paragraph()
p3_desc.add_run("3대 AI의 공통 의견을 종합하여, 가장 안전하고 실현 가능성이 높은 바벨 데드리프트 d2 v5.0의 최종 아키텍처를 다음과 같이 설계합니다:")

tbl2 = doc.add_table(rows=5, cols=3)
tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
headers2 = ["계층 (Tier)", "적용 혁신 기법", "기대 효과 및 세부 설계"]
for col_idx, h in enumerate(headers2):
    c = tbl2.cell(0, col_idx)
    c.paragraphs[0].text = h
    c.paragraphs[0].runs[0].font.name = "맑은 고딕"
    c.paragraphs[0].runs[0].font.size = Pt(9.5)
    c.paragraphs[0].runs[0].font.bold = True
    c.paragraphs[0].runs[0].font.color.rgb = WHITE
    tcPr = c._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="2B6CB0"/>'))

data2 = [
    ("Tier 1: 기하 및 피처 정밀화\n(Geometric Refinement)", "1. 직교화 관상면 투영 부호 외반각 (Signed Projected FPPA)\n2. 발 진행방향(Heel-Toe) 대비 무릎 내측 편위 피처\n3. 고관절-발목 축 기준 무릎 내측 잔차 및 적분 결손량", "무릎 굴곡과 외반의 완벽한 분리, d2 라벨('발-무릎 일치')과의 1:1 역학적 정합 달성, 2~5cm 미세 신호의 SNR 극대화"),
    ("Tier 2: 시점 D 감사 및 무결성\n(View D Integrity Audit)", "1. 좌우 관절(Left/Right) 및 시점 D 내측 부호(Sign) 전수 검사\n2. AI-Hub 5대 시점 매핑 및 랜드마크 스왑 여부 디버깅", "시점 D(좌측 전방)의 비정상적 Recall 20%를 시점 B(우측 전방 94%) 수준으로 정상화"),
    ("Tier 3: 초경량 시계열 융합\n(Tiny Temporal Model)", "1. 파라미터 4,162개 경량 8채널 Tiny TCN\n2. 5-Fold StratifiedGroupKFold OOF 임베딩 추출\n3. Regularized GBDT(LightGBM + CatBoost)와 Late-Fusion", "요약 통계량에서 유실된 64프레임 연속 동역학 패턴 학습, 과적합 제로(0%) 보장"),
    ("Tier 4: 불균형 손실 및 보정\n(Calibrated Decision)", "1. Asymmetric Focal Loss (쉬운 정상 음성 그래디언트 억제)\n2. Soft-F1 보조 손실 (0.1~0.2 완만한 결합)\n3. OOF 기반 3-State 이중 임계값 탐색 (보류율 ≤10% 제약)", "정밀도(Precision) 대폭 견인을 통해 허위 경보(FP) 통제 및 3-State F1 65%~80% 궤도 진입")
]

for row_idx, r in enumerate(data2, 1):
    for col_idx, val in enumerate(r):
        c = tbl2.cell(row_idx, col_idx)
        c.paragraphs[0].text = val
        c.paragraphs[0].runs[0].font.name = "맑은 고딕"
        c.paragraphs[0].runs[0].font.size = Pt(8.5)
        c.paragraphs[0].runs[0].font.color.rgb = BLACK
        tcPr = c._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{"FFFFFF" if row_idx%2==1 else "F7FAFC"}"/>'))

doc.add_paragraph().paragraph_format.space_after = Pt(4)

# 제4장
p4 = doc.add_paragraph()
style_heading(p4, "4. d2 v5.0 즉시 실행 액션 플랜 (Implementation Roadmap)", level=1)

add_callout_box(doc, [
    "• Step 1: 시점 D(좌측 전방 45°) 좌우 부호 및 기하 계산 전수 감사 스크립트 실행 (즉시 착수)",
    "• Step 2: 3D 관상면 투영 부호 외반각(Signed Projected FPPA) 및 발-무릎 일치 피처셋 v5.0 정밀 구축",
    "• Step 3: 파라미터 4,162개 경량 Tiny TCN 구축 및 5-Fold OOF 교차검증 파이프라인 통합",
    "• Step 4: Asymmetric Focal + Soft-F1 손실 기반 이중 앙상블 학습 및 3-State 보류 윈도우 최적화",
    "• Step 5: AI-Hub 385개 전수 무삭제 평가를 통한 F1-Score 대조 및 종합 보고서 배포"
], title="향후 5단계 실행 계획")

doc.save(DOCX_PATH)
print(f"[+] 외부 AI 자문 분석 및 v5.0 로드맵 Word 보고서 저장 완료: {DOCX_PATH}")
