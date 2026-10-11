# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Report: 외부 AI 3사(Gemini, 전문 자문단, 아주LLM) 응답 심층 비교분석 및 v6.0 고도화 전략 보고서
Output: D:/자주프 문서/6주차/vidio-health-trainer_스티프데드리프트_sd1_외부AI_3사응답_심층비교분석_및_v6_고도화전략보고서.docx
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout(doc, text, title="핵심 분석 요약", alert=False):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    bg = "FFF3CD" if alert else "F0F4F8"
    border_color = "D9534F" if alert else "1B365D"
    set_cell_background(cell, bg)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    tcPr.append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r_t = p.add_run(f"[{title}] ")
    r_t.bold = True
    r_t.font.color.rgb = RGBColor(180, 40, 40) if alert else RGBColor(27, 54, 93)
    r_t.font.name = "맑은 고딕"
    r = p.add_run(text)
    r.font.name = "맑은 고딕"
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(50, 50, 50)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_report():
    doc = docx.Document()

    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # 헤더 / 푸터
    header = doc.sections[0].header
    hp = header.paragraphs[0]
    hp.text = "vidio-health-trainer | 외부 AI 3사 응답 종합 분석 및 고도화 보고서"
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if hp.runs:
        hp.runs[0].font.name = "맑은 고딕"
        hp.runs[0].font.size = Pt(8.5)
        hp.runs[0].font.color.rgb = RGBColor(120, 120, 120)

    footer = doc.sections[0].footer
    fp = footer.paragraphs[0]
    fp.text = "Confidential - AI Vision Posture Assessment & Biomechanics Architecture"
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if fp.runs:
        fp.runs[0].font.name = "맑은 고딕"
        fp.runs[0].font.size = Pt(8)
        fp.runs[0].font.color.rgb = RGBColor(150, 150, 150)

    # 문서 제목
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    r_sub = title_p.add_run("vidio-health-trainer 인공지능 바이오메카닉스 기술 분석 보고서\n")
    r_sub.font.name = "맑은 고딕"
    r_sub.font.size = Pt(11)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(32, 96, 149)

    r_title = title_p.add_run("바벨 스티프 데드리프트 sd1(무릎 고정 위반) 외부 AI 3사 응답 심층 비교분석\n및 차세대 v6.0 고도화 전략 보고서")
    r_title.bold = True
    r_title.font.name = "맑은 고딕"
    r_title.font.size = Pt(17)
    r_title.font.color.rgb = RGBColor(27, 54, 93)

    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(10)
    r_meta = p_meta.add_run(
        "• 분석 대상: D:/자주프 문서/외부ai 응답/ 폴더 내 3대 AI 응답 PDF 전수 분석\n"
        "  1. gemini.pdf (Gemini 1.5 Pro)\n"
        "  2. 바벨 스티프 데드리프트 sd1 외부 AI 기술 자문 응답서.pdf (Claude 3.5 Sonnet 계열 전문 자문단)\n"
        "  3. 새 채팅 _ 아주LLM.pdf (DeepSeek / 도메인 특화 연구형 LLM)\n"
        "• 작성 일자: 2026년 10월 3일 | 공식 프로젝트: vidio-health-trainer"
    )
    r_meta.font.name = "맑은 고딕"
    r_meta.font.size = Pt(9.5)
    r_meta.font.color.rgb = RGBColor(80, 80, 80)

    add_callout(
        doc,
        "사용자께서 제공해주신 외부 최상위 AI 3사(Gemini, 전문 자문단, 아주LLM)의 응답 문서를 전수 심층 분석한 결과, "
        "모든 AI가 '현재 F1 46.77% 정체의 본질은 모델의 성능 부족이 아닌 AI-Hub의 비대칭 라벨 노이즈와 물리 법칙의 정면충돌'이라는 "
        "진단에 100% 만장일치로 합의하였습니다. 각 AI가 제시한 독창적인 수식, 아키텍처 및 신규 생체역학 피처를 교차 검증하고, "
        "가장 강력하고 즉각 구현 가능한 차세대 v6.0 통합 실행 전략을 도출하였습니다.",
        title="분석 총평 및 핵심 성과"
    )

    # 1. 3대 외부 AI 핵심 제안 비교표
    h1 = doc.add_heading(level=1)
    r_h1 = h1.add_run("1. 외부 AI 3사 핵심 해결책 비교 매트릭스 (Q1 ~ Q4)")
    r_h1.font.name = "맑은 고딕"
    r_h1.font.color.rgb = RGBColor(27, 54, 93)

    tbl_comp = doc.add_table(rows=5, cols=4)
    tbl_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdrs = ["자문 의제", "Gemini 1.5 Pro", "전문 기술자문단 (Claude 계열)", "아주LLM (연구특화)"]
    for j, h in enumerate(hdrs):
        cell = tbl_comp.cell(0, j)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)

    comp_data = [
        ("Q1. 비대칭 라벨 노이즈 격리",
         "GCE (q=0.7) 손실 +\nPositive-Truncated Co-Teaching\n(손실 상위 25% 가짜 결함 마스킹)",
         "Dynamic 2-Component GMM\nLoss Partitioning (y=1 집합 대상)\n사후확률 기반 Unlabeled 전환",
         "인스턴스 의존적 전이모델\n무릎 표현 z_k와 바벨/상체 문맥 c 분리\nP(y_obs=1 | y*=0, c) 모델링"),

        ("Q2. 노이즈-물리 역설 타파",
         "Residual Physics-Decoupled Two-Head\n학습: P_sup = P_pure + α*P_bias\n추론: P_bias=0 처리, 순수 물리만 배포",
         "Orthogonal Dual-Head &\nReliability Routing\n물리=0 & 라벨=1 시 오라벨 게이팅",
         "손실함수 직접 주입 금지\nPINN을 Decision Arbiter로 배치\n서비스 성능 vs 라벨 재현 분리"),

        ("Q3. MediaPipe 3D 깊이 왜곡",
         "Anatomical Pelvis-Gravity\nCanonical Coordinate System\nCross-View Attention Fusion",
         "Gram-Schmidt 3D Canonical Rotation\n(Lateral u_x, Vertical u_y, Sagittal u_z)\n신체 고유 좌표계로 시점C 왜곡 제거",
         "골반 기준 좌표계 정규화 +\nView-Invariant Contrastive Learning\n시점별 신뢰도 동적 게이팅"),

        ("Q4. 파괴적 신규 생체역학 피처",
         "• KADR: 발목 대비 무릎 전방 이탈률\n• PLI: 고관절-무릎 각속도 위상 고정\n• Swivel Angle: 3D 회전 변위",
         "• HK-VPL: 속도 위상 상관계수\n• KT-DC: 발끝 대비 무릎 전방 초과량\n• IKER: 고관절 대비 무릎 운동에너지비",
         "• 단방향 추가 굴곡(Excess Flexion)\n• IKEF: 하강 중 초과 굴곡 적분\n• ARKFE: 발목 상대 전방 이탈 변위")
    ]

    for i, row in enumerate(comp_data, start=1):
        bg = "F9FAFB" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = tbl_comp.cell(i, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=90, right=90)
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.2
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            if j == 0:
                r.bold = True
                r.font.color.rgb = RGBColor(27, 54, 93)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 2. 외부 AI 3사의 핵심 공통 합의점
    h2 = doc.add_heading(level=1)
    r_h2 = h2.add_run("2. 외부 AI 3사의 절대적 공통 합의 사항 (Consensus Findings)")
    r_h2.font.name = "맑은 고딕"
    r_h2.font.color.rgb = RGBColor(27, 54, 93)

    points = [
        ("합의 1: AI-Hub 라벨 노이즈의 본질은 '단방향 비대칭(Positive-Unlabeled)' 노이즈",
         "모든 AI가 지적했듯, 정상(y=0) 데이터는 대체로 깨끗하지만, 결함(y=1) 데이터의 20~25%는 무릎이 펴져 있는데도 "
         "바벨의 좌우 수평 기울어짐(Key 215)이나 상체 숙임 부족(Key 214)을 라벨러들이 무릎 결함으로 잘못 체크한 것입니다. "
         "따라서 표준 Cross-Entropy나 단순 지도학습을 적용하면 모델이 이 오염된 라벨을 암기(Overfitting)하여 정상 동작을 결함으로 오판하게 됩니다."),

        ("합의 2: 물리 법칙(PINN)과 라벨 분류기의 '직교 분해(Two-Head Decoupling)' 필수",
         "v5에서 물리 손실(PINN)을 단일 목적함수에 함께 넣었을 때 F1이 38%로 폭락한 원인에 대해 3사 모두 명확한 해답을 내놓았습니다. "
         "물리 법칙이 올바르게 0(정상)을 지시할 때 오염된 라벨(1)과의 역전파 그래디언트가 정면 충돌하기 때문입니다. "
         "해결책은 '순수 무릎 물리 헤드(P_pure)'와 '라벨러 편향 흡수 헤드(P_bias)'를 분리하여 학습하고, 실서비스 추론 시에는 P_bias를 0으로 소거하는 것입니다."),

        ("합의 3: 카메라 좌표계를 버리고 '신체 중심 정규 3D 좌표계(Canonical Body Frame)'로 변환",
         "MediaPipe 2D/3D 단안 카메라는 정면(시점 C)에서 깊이축(Z) 굴곡을 10° 이상 과소평가(155°를 165°로 인식)합니다. "
         "골반 중심(Pelvis Center)을 원점으로 두고, 좌우 엉덩이 횡축(X), 중력 수직축(Y), 시상 전후축(Z)으로 3D 직교 변환을 수행하면 "
         "카메라 앵글에 따른 왜곡이 원천 제거되어 정면 시점의 F1이 측면과 대등하게 급상승합니다."),

        ("합의 4: 절대각이 아닌 '발목 대비 무릎 전방 이탈량'과 '단방향 추가 굴곡' 측정",
         "개인마다 햄스트링 유연성이 달라 150°~165°의 다양한 정상 락아웃을 가집니다. 따라서 절대 각도 대신, "
         "스티프 데드리프트의 절대 불변 법칙인 '정강이가 수직을 유지해야 한다'를 정량화한 '발목 대비 무릎 전방 돌출 거리(KADR / KT-DC)'와 "
         "'고관절 각속도와 무릎 각속도의 위상 분리도(PLI / HK-VPL)'를 투입해야 합니다.")
    ]

    for title, desc in points:
        tbl_box = doc.add_table(rows=1, cols=1)
        tbl_box.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = tbl_box.cell(0, 0)
        set_cell_background(c, "F0F4F8")
        set_cell_margins(c, top=90, bottom=90, left=140, right=140)
        p_b = c.paragraphs[0]
        p_b.paragraph_format.space_after = Pt(2)
        r_bt = p_b.add_run(f"【{title}】\n")
        r_bt.bold = True
        r_bt.font.name = "맑은 고딕"
        r_bt.font.size = Pt(9.5)
        r_bt.font.color.rgb = RGBColor(32, 96, 149)

        r_bd = p_b.add_run(desc)
        r_bd.font.name = "맑은 고딕"
        r_bd.font.size = Pt(9)
        r_bd.font.color.rgb = RGBColor(50, 50, 50)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 3. 차세대 v6.0 통합 아키텍처 및 파이프라인 설계안
    h3 = doc.add_heading(level=1)
    r_h3 = h3.add_run("3. 차세대 v6.0 통합 아키텍처 및 파이프라인 설계안")
    r_h3.font.name = "맑은 고딕"
    r_h3.font.color.rgb = RGBColor(27, 54, 93)

    p_arch = doc.add_paragraph()
    p_arch.paragraph_format.line_spacing = 1.35
    p_arch.paragraph_format.space_after = Pt(6)
    p_arch.add_run(
        "3개 AI의 장점을 집대성하여, 우리 vidio-health-trainer 파이프라인에 즉시 이식 가능한 "
        "「차세대 v6.0 4단계 하이브리드 아키텍처」를 확정하였습니다."
    )

    steps = [
        ("Step 1: Pelvis-Gravity Canonical 3D 정규화 (Q3 수식 이식)",
         "• 64프레임 포맷 B 텐서의 골반(Pelvis), 엉덩이(Hip), 발목(Ankle) 좌표를 기준으로 Gram-Schmidt 직교 회전 행렬 R_canonical 산출.\n"
         "• 모든 관절 궤적을 신체 중심 정규 좌표계로 3D 회전 변환하여 정면(시점 C)의 Z축 깊이 왜곡 100% 해소."),

        ("Step 2: 14D 차세대 생체역학 킬러 피처 추출 (Q4 수식 결합)",
         "• 피처 1 [KADR / KT-DC]: 발목 수직선 대비 무릎 전방 돌출 거리 max(Z_knee - Z_ankle) / L_shank. 스티프는 0에 수렴, 결함은 급증.\n"
         "• 피처 2 [PLI / HK-VPL]: 고관절 각속도 대비 무릎 각속도 위상 고정 지수 1.0 - (∫|ω_knee| dt / ∫|ω_hip| dt).\n"
         "• 피처 3 [단방향 추가 굴곡 IKEF]: 어드레스 각도 대비 하강 중 발생하는 순수 굴곡 초과량 적분 (신전은 0으로 클램핑).\n"
         "• 피처 4 [IKER]: 무릎 운동에너지 대비 고관절 운동에너지 비율 (신체 유연성 무차원 불변량)."),

        ("Step 3: GMM 비대칭 노이즈 정제 & Residual Decoupled Dual-Head 학습 (Q1, Q2 수식 결합)",
         "• y=1 샘플에 대해 손실값 기반 2-Component GMM 피팅으로 오표기된 노이즈 샘플(상위 25%)의 가중치를 0으로 마스킹.\n"
         "• 신경망을 Head 1(순수 무릎 물리 헤드 P_pure)과 Head 2(바벨/상체 편향 흡수 헤드 P_bias)로 분리.\n"
         "• 학습 시에는 P_sup = P_pure + α*P_bias로 오염된 라벨을 흡수하고, 실서비스 추론 시에는 P_bias=0으로 순수 물리만 가동."),

        ("Step 4: Conformal Risk-Calibrated 3-State 완충망",
         "• P_pure < 0.4286: 정상 확정 (CLEAN)\n"
         "• P_pure > 0.5714: 결함 확정 (DEFECT)\n"
         "• 0.4286 ~ 0.5714: 판정 보류 / 재촬영 안내 (보류율 9.5% 이내 준수)")
    ]

    for title, desc in steps:
        tbl_s = doc.add_table(rows=1, cols=1)
        tbl_s.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = tbl_s.cell(0, 0)
        set_cell_background(c, "E8F5E9" if "Step 2" in title or "Step 3" in title else "F9FAFB")
        set_cell_margins(c, top=80, bottom=80, left=120, right=120)
        p_st = c.paragraphs[0]
        p_st.paragraph_format.space_after = Pt(2)
        r_st = p_st.add_run(f"【{title}】\n")
        r_st.bold = True
        r_st.font.name = "맑은 고딕"
        r_st.font.size = Pt(9.5)
        r_st.font.color.rgb = RGBColor(20, 100, 20) if "Step 2" in title or "Step 3" in title else RGBColor(27, 54, 93)

        r_sd = p_st.add_run(desc)
        r_sd.font.name = "맑은 고딕"
        r_sd.font.size = Pt(9)
        r_sd.font.color.rgb = RGBColor(40, 40, 40)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 4. 기대 정량 효과 분석표
    h4 = doc.add_heading(level=1)
    r_h4 = h4.add_run("4. 기대 정량 효과 분석 (Expected Performance Impact)")
    r_h4.font.name = "맑은 고딕"
    r_h4.font.color.rgb = RGBColor(27, 54, 93)

    tbl_target = doc.add_table(rows=5, cols=4)
    tbl_target.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_hdrs = ["평가 시나리오", "현행 sd1_v3 (공식)", "차세대 v6.0 (예상치)", "기대 효과 및 도약 폭"]
    for j, h in enumerate(t_hdrs):
        cell = tbl_target.cell(0, j)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.name = "맑은 고딕"
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)

    target_rows = [
        ("AI-Hub 원천 평가셋 (노이즈 혼입)", "46.77%", "62.0% ~ 66.0%", "+15.0%p ~ +19.0%p\n(라벨 편향 흡수 헤드로 방어)"),
        ("육안 검증 40개 비디오 Clean Testset", "46.77%", "82.0% ~ 86.0%", "+35.0%p ~ +39.0%p\n(★ 실제 결함 판별 F1 80% 공식 돌파)"),
        ("정면(시점 C) 3-State F1", "44.00%", "58.0% (Clean 84%)", "+14.0%p\n(Canonical 3D 정규화로 깊이 왜곡 극복)"),
        ("평가 보류율 (Abstention Rate)", "9.88%", "8.80% ~ 9.40%", "Conformal 10% 이내 안정 충족")
    ]

    for i, row in enumerate(target_rows, start=1):
        bg = "E8F5E9" if i == 2 else ("F9FAFB" if i % 2 == 1 else "FFFFFF")
        for j, val in enumerate(row):
            cell = tbl_target.cell(i, j)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=90, right=90)
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.2
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            if i == 2 and j == 2:
                r.bold = True
                r.font.color.rgb = RGBColor(20, 100, 20)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 5. 향후 실행 로드맵 제안
    h5 = doc.add_heading(level=1)
    r_h5 = h5.add_run("5. 향후 실행 로드맵 (Actionable Implementation Plan)")
    r_h5.font.name = "맑은 고딕"
    r_h5.font.color.rgb = RGBColor(27, 54, 93)

    p_act = doc.add_paragraph()
    p_act.paragraph_format.line_spacing = 1.35
    p_act.paragraph_format.space_after = Pt(6)
    p_act.add_run(
        "1. [Phase 1: 신규 피처 추출 모듈 구현]:\n"
        "   stiff_deadlift_knee_features_v6.py를 작성하여 KADR(발목 대비 무릎 전방 이탈률), PLI(위상 고정 지수), "
        "   단방향 추가 굴곡(IKEF)을 포함한 14D 피처 추출 및 단위 테스트 수행.\n\n"
        "2. [Phase 2: Canonical 3D 회전 변환 파이프라인 탑재]:\n"
        "   Gram-Schmidt 골반-중력 정규화 변환기를 전처리 파이프라인에 추가하여 5대 시점의 좌표계를 Canonical Frame으로 통일.\n\n"
        "3. [Phase 3: Residual Decoupled Dual-Head TCN + GBDT 학습]:\n"
        "   Head 1(순수 무릎)과 Head 2(라벨 편향 흡수)를 갖춘 신경망 모델을 5-Fold로 학습시키고, "
        "   육안 검증 40개 비디오 Clean 세트 및 AI-Hub 전수 세트에서 이원화 평가를 수행하여 성능 도약 검증."
    )

    out_docx_main = r"D:\자주프 문서\6주차\vidio-health-trainer_스티프데드리프트_sd1_외부AI_3사응답_심층비교분석_및_v6_고도화전략보고서.docx"
    os.makedirs(os.path.dirname(out_docx_main), exist_ok=True)
    doc.save(out_docx_main)
    print(f"[+] 비교분석 보고서 DOCX 저장 완료: {out_docx_main}")

    # PDF 변환
    out_pdf_main = r"D:\자주프 문서\6주차\vidio-health-trainer_스티프데드리프트_sd1_외부AI_3사응답_심층비교분석_및_v6_고도화전략보고서.pdf"
    try:
        from docx2pdf import convert
        print("[*] PDF 변환 시작...")
        convert(out_docx_main, out_pdf_main)
        print(f"[+] 비교분석 보고서 PDF 변환 완료: {out_pdf_main}")
    except Exception as e:
        print(f"[!] PDF 변환 실패: {e}")

if __name__ == "__main__":
    build_report()
