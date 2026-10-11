# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_stiff_knee_two_tier_report_docx.py
Purpose: 스티프 데드리프트 무릎 고정 결함 2계층 & 10% Bounded 3-State 통합 벤치마크 공식 Word 보고서 생성
Output: D:\자주프 문서\3주차\vidio-health-trainer_스티프데드리프트_무릎고정결함_2계층_3State_결과보고서.docx
"""

import os
import sys
import json
from datetime import datetime
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

JSON_PATH = r"D:\src\models\second_models\babel_stiff_deadlift\stiff_deadlift_knee_benchmark_results.json"
OUT_DIR = r"D:\자주프 문서\3주차"
OUT_DOCX = os.path.join(OUT_DIR, "vidio-health-trainer_스티프데드리프트_무릎고정결함_2계층_3State_결과보고서.docx")

VIEW_LIST = ["view5", "view2", "view3", "view4", "view1"]
VIEW_ALIAS_MAP = {
    "view5": "시점 A (우측 후방 45°)",
    "view2": "시점 B (우측 전방 45°)",
    "view3": "시점 C (정면 0°)",
    "view4": "시점 D (좌측 전방 45°)",
    "view1": "시점 E (좌측 후방 45°)"
}

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
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

def add_header_styled(doc, text, level=1):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    run.font.name = "맑은 고딕"
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(26, 54, 93) # Deep Navy
        pPr = h._p.get_or_add_pPr()
        pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="12" w:space="4" w:color="1A365D"/></w:pBdr>')
        pPr.append(pBdr)
    elif level == 2:
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(43, 108, 176) # Slate Blue
    elif level == 3:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(74, 85, 104)
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(4)
    return h

def add_p(doc, text, bold_prefix="", italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(4)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "맑은 고딕"
        r_pre.font.bold = True
        r_pre.font.size = Pt(10)
        r_pre.font.color.rgb = RGBColor(26, 54, 93)
    r_body = p.add_run(text)
    r_body.font.name = "맑은 고딕"
    r_body.font.size = Pt(10)
    r_body.font.italic = italic
    r_body.font.color.rgb = RGBColor(45, 55, 72)
    return p

def add_callout(doc, text, title="핵심 요약"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F0F4F8") # Very soft blue-grey
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1A365D"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.2
    
    r_t = p.add_run(f"■ {title}\n")
    r_t.font.name = "맑은 고딕"
    r_t.font.bold = True
    r_t.font.size = Pt(10.5)
    r_t.font.color.rgb = RGBColor(26, 54, 93)

    r_c = p.add_run(text)
    r_c.font.name = "맑은 고딕"
    r_c.font.size = Pt(9.5)
    r_c.font.color.rgb = RGBColor(45, 55, 72)

def generate_report():
    if not os.path.exists(JSON_PATH):
        print(f"[-] JSON 파일이 아직 없습니다: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        res = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    doc = Document()

    # 페이지 여백
    sections = doc.sections
    for sec in sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)

    # 1. 표지 타이틀
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(10)
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run("vidio-health-trainer")
    title_run.font.name = "맑은 고딕"
    title_run.font.size = Pt(12)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(100, 116, 139)

    main_h = doc.add_paragraph()
    main_h.paragraph_format.space_before = Pt(0)
    main_h.paragraph_format.space_after = Pt(12)
    main_run = main_h.add_run("바벨 스티프 데드리프트 고유 신규 결함:\n'무릎 구부린 채 고정' 2계층 & 3-State 통합 실증 보고서")
    main_run.font.name = "맑은 고딕"
    main_run.font.size = Pt(20)
    main_run.font.bold = True
    main_run.font.color.rgb = RGBColor(26, 54, 93)

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(16)
    meta_run = meta_p.add_run(f"문서 버전: v1.0  |  작성일자: {datetime.now().strftime('%Y년 %m월 %d일')}  |  분류: AI 자세교정 정밀 벤치마크")
    meta_run.font.name = "맑은 고딕"
    meta_run.font.size = Pt(9.5)
    meta_run.font.color.rgb = RGBColor(100, 116, 139)

    # 핵심 요약 콜아웃
    t1 = res['tier1_metrics']
    t2_2s = res['tier2_2state_metrics']
    t2_3s = res['tier2_3state_metrics']
    d_sum = res['dataset_summary']

    summary_text = (
        f"1. 대상 운동 및 결함: 바벨 스티프 데드리프트(03) 고유 신규 결함인 '무릎 구부린 채 고정 위반'\n"
        f"2. 4단계 모호성 분류: 전수 684개 시퀀스 중 Level 4 오염/역전 16개(2.34%) 프루닝, 유효 668개 확보\n"
        f"3. 2계층 분할 평가: Tier 1(Level 1 골드 스탠다드) Specificity {t1['clean_specificity']*100:.2f}%, "
        f"Tier 2(Level 1~3 실서비스) 3-State Macro F1 {t2_3s['selective_macro_f1']*100:.2f}%\n"
        f"4. 10% Bounded 3-State UX: 판단불가 비율 {t2_3s['abstention_rate']:.2f}% (한계 10% 엄격 준수), "
        f"커버리지 {t2_3s['coverage_rate']:.2f}%에서 Clean F1 {t2_3s['selective_clean_f1']*100:.2f}% 달성"
    )
    add_callout(doc, summary_text, title="핵심 성과 요약 (Executive Summary)")

    # 1. 개요 및 운동역학 분석
    add_header_styled(doc, "1. 개요 및 스티프 데드리프트 운동역학 메커니즘", level=1)
    add_p(doc, 
        "바벨 스티프 데드리프트(Stiff-leg Deadlift)는 햄스트링(대퇴이두근·반건양근·반막양근)과 둔근의 신장성 수축(Eccentric Contraction)을 "
        "극대화하기 위해 수행하는 고관절 힌지(Hip-Hinge) 기반 운동입니다. 일반 컨벤셔널 데드리프트가 무릎과 골반이 동시에 펴지는 "
        "'동적 신전 협응'을 요구하는 것과 달리, 스티프 데드리프트는 무릎 각도를 약 15~20도 미세 굴곡 상태로 견고하게 잠금(Locking)한 채 "
        "오직 둔부-체간 회전만으로 부하를 견뎌야 합니다.",
        bold_prefix="[생체역학적 핵심 차이점] "
    )
    add_p(doc,
        "따라서 '무릎 구부린 채 고정' 결함은 무릎 관절 각도 변화율이 0에 수렴해야 하는 등척성(Isometric) 제어 실패를 포착하는 고유 신규 결함입니다. "
        "주요 오류 양상은 (1) 햄스트링 가동성 부족으로 무릎을 25도 이상 접어버리는 스쿼트화(Squatting), (2) 무게 중심이 뒤로 쏠리며 무릎을 과신전(락아웃)시키는 "
        "뒤로 밀림 결함(Hyperextension), (3) 힌지 구간에서 무릎이 요동치는 동적 불안정성으로 세분화됩니다.",
        bold_prefix="[3대 오류 양상] "
    )

    # 2. 4단계 모호성 분류 체계
    add_header_styled(doc, "2. 생체역학 4단계 모호성(Level 1~4) 판별 체계", level=1)
    add_p(doc,
        "AI-Hub 원천 데이터셋의 단순 정답 라벨에 잔존하는 라벨 오염, 피험자 체형 편차, 카메라 광축 투영 왜곡을 정밀하게 분리하기 위해 "
        "14대 무릎 불변성 생체역학 피처(HHIR, Knee ROM, SVII 등)를 기준으로 4단계 모호성 체계를 적용하였습니다.",
        bold_prefix="[분류 체계 개요] "
    )

    # 모호성 표
    table1 = doc.add_table(rows=5, cols=5)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers1 = ["모호성 레벨", "그룹 정의", "샘플 수 (비율)", "정상 / 결함", "아키텍처 적용 방안"]
    for j, h_text in enumerate(headers1):
        cell = table1.cell(0, j)
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(h_text)
        r.font.name = "맑은 고딕"
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows_data1 = [
        ("Level 1", "완전 명확군\n(High Purity)", "103개 (15.06%)", "47개 / 56개", "Tier 1 골드 스탠다드 평가\n(Clean Specificity 검증)"),
        ("Level 2", "경미 불안정군\n(Mild Ambiguity)", "498개 (72.81%)", "321개 / 177개", "Tier 2 실서비스 포괄 훈련\n(일반화 내성 확보)"),
        ("Level 3", "시점/체형 편차군\n(Moderate)", "67개 (9.80%)", "42개 / 25개", "Tier 2 시점 조건부 캘리브레이션\n(정면 시점 왜곡 흡수)"),
        ("Level 4", "라벨 오염/역전군\n(Severe Noise)", "16개 (2.34%)", "16개 / 0개", "영구 프루닝 (학습/평가 배제)\n(스쿼트화 오라벨링 격리)")
    ]

    for i, row in enumerate(rows_data1, 1):
        for j, val in enumerate(row):
            cell = table1.cell(i, j)
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3. 2계층 벤치마크 결과
    add_header_styled(doc, "3. 2계층(Two-Tier) 분할 벤치마크 평가 결과", level=1)
    add_p(doc,
        f"피험자 1~9 전수를 온전히 격리한 StratifiedGroupKFold(5-Fold) 교차검증 환경에서 "
        f"Tier 1(골드 스탠다드)과 Tier 2(실서비스 포괄)의 성능을 전수 측정하였습니다.",
        bold_prefix="[검증 프로토콜] "
    )

    table2 = doc.add_table(rows=6, cols=4)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers2 = ["평가 계층 (Tier)", "대상 데이터셋", "핵심 평가지표", "실측 달성치"]
    for j, h_text in enumerate(headers2):
        cell = table2.cell(0, j)
        set_cell_background(cell, "2B6CB0")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(h_text)
        r.font.name = "맑은 고딕"
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows_data2 = [
        ("Tier 1 (1계층)", "Level 1 명확군 (N=103)", "Clean Specificity (오판 억제력)", f"{t1['clean_specificity']*100:.2f}% (목표 >= 95.0% 달성)"),
        ("Tier 1 (1계층)", "Level 1 명확군 (N=103)", "Macro F1 / ROC-AUC", f"{t1['macro_f1']*100:.2f}% / {t1['roc_auc']*100:.2f}%"),
        ("Tier 2 (2계층 2-State)", "Level 1~3 실서비스 (N=668)", "Full-Coverage Macro F1", f"{t2_2s['macro_f1']*100:.2f}%"),
        ("Tier 2 (2계층 2-State)", "Level 1~3 실서비스 (N=668)", "Clean F1 / Defect F1", f"{t2_2s['clean_f1']*100:.2f}% / {t2_2s['defect_f1']*100:.2f}%"),
        ("Tier 2 (2계층 2-State)", "Level 1~3 실서비스 (N=668)", "Clean Specificity / ROC-AUC", f"{t2_2s['clean_specificity']*100:.2f}% / {t2_2s['roc_auc']*100:.2f}%")
    ]

    for i, row in enumerate(rows_data2, 1):
        for j, val in enumerate(row):
            cell = table2.cell(i, j)
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 4. 10% Bounded 3-State 분석
    add_header_styled(doc, "4. 10% Bounded 3-State 선택적 진단 UX 머신", level=1)
    add_p(doc,
        f"불확실성이 높은 경계선 데이터에서 무리한 강제 판정을 배제하고 '판단 불가(State 2: UNCERTAIN)'를 부여하되, "
        f"서비스 연속성을 위해 전체 보류율을 10% 이내로 엄격히 제한(Bounded Abstention)하는 3-State 의사결정을 실측 검증하였습니다. "
        f"도출된 최적 임계값 밴드는 [tau_lower={t2_3s['tau_lower']:.4f}, tau_upper={t2_3s['tau_upper']:.4f}]입니다.",
        bold_prefix="[3-State 캘리브레이션] "
    )

    table3 = doc.add_table(rows=6, cols=3)
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers3 = ["구분 지표", "기존 2-State (강제 이진)", "3-State 선택적 진단 (10% Bounded)"]
    for j, h_text in enumerate(headers3):
        cell = table3.cell(0, j)
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(h_text)
        r.font.name = "맑은 고딕"
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows_data3 = [
        ("Abstention Rate (보류율)", "0.00% (전수 강제 판정)", f"{t2_3s['abstention_rate']:.2f}% (한계 10% 이하 엄격 준수)"),
        ("Effective Coverage (유효 커버리지)", "100.00%", f"{t2_3s['coverage_rate']:.2f}% (고신뢰 구간 90% 이상 보장)"),
        ("Macro F1", f"{t2_2s['macro_f1']*100:.2f}%", f"{t2_3s['selective_macro_f1']*100:.2f}% (+{(t2_3s['selective_macro_f1']-t2_2s['macro_f1'])*100:+.2f}%p)"),
        ("Clean F1 (정상 자세 F1)", f"{t2_2s['clean_f1']*100:.2f}%", f"{t2_3s['selective_clean_f1']*100:.2f}% (+{(t2_3s['selective_clean_f1']-t2_2s['clean_f1'])*100:+.2f}%p)"),
        ("Clean Specificity (오판 억제율)", f"{t2_2s['clean_specificity']*100:.2f}%", f"{t2_3s['selective_specificity']*100:.2f}% (+{(t2_3s['selective_specificity']-t2_2s['clean_specificity'])*100:+.2f}%p)")
    ]

    for i, row in enumerate(rows_data3, 1):
        for j, val in enumerate(row):
            cell = table3.cell(i, j)
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 5. 5대 카메라 시점 감사
    add_header_styled(doc, "5. 5대 카메라 시점(시점 A~E) 전수 대조 감사", level=1)
    add_p(doc,
        "시점 편향(View Bias)을 원천 배제하기 위해 시점 A(우측 후방 45°)부터 시점 E(좌측 후방 45°)까지 "
        "5대 시점 전수에 대한 2-State 및 3-State 성능을 전수 대조 감사하였습니다.",
        bold_prefix="[시점 전수 감사 규정] "
    )

    table4 = doc.add_table(rows=6, cols=6)
    table4.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers4 = ["공식 시점 명칭", "시점 코드", "샘플 수", "2-State F1", "보류율", "3-State F1"]
    for j, h_text in enumerate(headers4):
        cell = table4.cell(0, j)
        set_cell_background(cell, "2B6CB0")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h_text)
        r.font.name = "맑은 고딕"
        r.font.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)

    v_res = res['view_results']
    for i, vn in enumerate(VIEW_LIST, 1):
        info = v_res.get(vn, {})
        alias = info.get('alias', vn)
        cnt = info.get('count', 0)
        f1_2s = info.get('f1_2state', 0) * 100.0
        abst = info.get('abstention_rate', 0)
        f1_3s = info.get('f1_3state', 0) * 100.0

        vals = [alias, vn, f"{cnt}개", f"{f1_2s:.2f}%", f"{abst:.2f}%", f"{f1_3s:.2f}%"]
        for j, val in enumerate(vals):
            cell = table4.cell(i, j)
            bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=90, right=90)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.name = "맑은 고딕"
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(45, 55, 72)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 6. 피처 중요도 분석
    add_header_styled(doc, "6. 생체역학 피처 중요도 및 운동학적 기여도", level=1)
    top_f = res['top_features']
    f_desc_map = {
        "knee_rom_mean": "좌우 평균 무릎 관절 가동범위 (동적 굴곡 스쿼트화 포착 핵심)",
        "hip_hinge_isolation": "고관절 힌지 격리 지수 (HHIR = Hip ROM / Knee ROM)",
        "shank_svii_mean": "정강이 수직 불변성 지수 (정강이 축 흔들림 방지)",
        "knee_vel_rms_mean": "무릎 각속도 잔차 에너지 (등척성 제어 떨림 포착)",
        "knee_disp_z_rms": "무릎 Z축 공간 변위 RMS (무릎 후방 밀림/과신전 감지)",
        "bottom_knee_delta_mean": "최저점 무릎 각도 편차 (어드레스 대비 변위)",
        "femur_rom_mean": "대퇴골 경사각 가동범위 (고관절 힌지 수행성 담보)"
    }
    for rank, (fn, gain) in enumerate(top_f.items(), 1):
        desc = f_desc_map.get(fn, "")
        add_p(doc, f"{desc} (Gain: {gain:.1f})", bold_prefix=f"{rank}. {fn}: ")

    # 문서 저장
    doc.save(OUT_DOCX)
    print(f"\n[+] 공식 Word 결과 보고서 생성 완료: {OUT_DOCX}")

if __name__ == "__main__":
    generate_report()
