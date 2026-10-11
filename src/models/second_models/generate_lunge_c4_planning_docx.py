# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_lunge_c4_planning_docx.py
Purpose: 바벨 런지 c4(척추의 중립 위반) 세부 오류 모델 구축 사양 및 공학적 기획서 .docx 및 .md 생성
Format: c1 기획서 및 스티프 데드리프트 최신 보고서 양식 100% 동기화
Rules: AGENTS.md 규격 (공식 프로젝트명, Word 필수, 64F 스플라인, 전 시점/피험자 전수, AI-Hub JSON 격리, 5대 공식 시점 명칭)
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

OUT_DIR = r"D:\자주프 문서\6주차"
os.makedirs(OUT_DIR, exist_ok=True)
DOCX_PATH = os.path.join(OUT_DIR, "바벨_런지_c4_척추중립_세부오류모델_구축기획서.docx")
MD_PATH = os.path.join(OUT_DIR, "바벨_런지_c4_척추중립_세부오류모델_구축기획서.md")


def set_cell_background(cell, fill_hex):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))


def set_cell_margins(cell, top=140, bottom=140, left=180, right=180):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def create_callout_box(doc, title, text, bg_hex="F0F4F8", border_hex="1E3A8A"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=160, bottom=160, left=220, right=220)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders_xml = f'''
    <w:tcBorders {nsdecls("w")}>
        <w:top w:val="none"/>
        <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>
        <w:bottom w:val="none"/>
        <w:right w:val="none"/>
    </w:tcBorders>
    '''
    tcPr.append(parse_xml(borders_xml))
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"■ {title}\n")
    run_t.bold = True
    run_t.font.size = Pt(10.5)
    run_t.font.name = "맑은 고딕"
    run_t.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    
    run_b = p.add_run(text)
    run_b.font.size = Pt(9.5)
    run_b.font.name = "맑은 고딕"
    run_b.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def format_table(tbl, col_widths, headers, data, align_rights=[]):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], "1F497D")
        set_cell_margins(hdr_cells[i], 140, 140, 140, 140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.bold = True
            r.font.size = Pt(9.5)
            r.font.name = "맑은 고딕"
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for r_idx, row_data in enumerate(data):
        row_cells = tbl.add_row().cells
        bg_color = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], 120, 120, 140, 140)
            p = row_cells[c_idx].paragraphs[0]
            if c_idx in align_rights:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif c_idx == 0 or len(str(val)) <= 8:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.size = Pt(9.0)
                r.font.name = "맑은 고딕"
                r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    for row in tbl.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)

    # 테두리 설정
    tblPr = tbl._tbl.tblPr
    borders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>
        <w:bottom w:val="single" w:sz="12" w:space="0" w:color="1F497D"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E5E7EB"/>
        <w:insideV w:val="none"/>
        <w:left w:val="none"/>
        <w:right w:val="none"/>
    </w:tblBorders>
    '''
    tblPr.append(parse_xml(borders_xml))


def build_docx_and_md():
    doc = docx.Document()
    
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # 헤더
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    r_sub = title_p.add_run("vidio-health-trainer | 인공지능 피트니스 자세교정 통합 플랫폼\n")
    r_sub.font.size = Pt(11)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
    
    r_main = title_p.add_run("바벨 런지 c4(척추의 중립 위반) 세부 오류 모델 구축 사양 및 공학적 기획서")
    r_main.font.size = Pt(19)
    r_main.font.bold = True
    r_main.font.name = "맑은 고딕"
    r_main.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(12)
    r_desc = sub_p.add_run("기구축 세부 오류모델(스쿼트·데드리프트·스티프) 아키텍처 노하우 계승 및 하지 분할 비대칭 3차원 척추 결함 특화 Two-Tier 파이프라인 설계")
    r_desc.font.size = Pt(10.5)
    r_desc.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # 메타정보
    create_callout_box(
        doc,
        "프로젝트 표준 메타데이터 (Project Metadata)",
        "• 공식 프로젝트명: vidio-health-trainer\n"
        "• 대상 운동 및 오류: 04_바벨_런지 / c4: 척추의 중립 위반 (Spine Neutrality Defect)\n"
        "• 원천 데이터 규모: AI-Hub 1,644개 시퀀스 전수 (정상 571건 / 결함 1,073건, 결함 비율 65.27%)\n"
        "• 작성 일자: 2026년 10월 03일 (6주차)\n"
        "• 핵심 기술 체계: 64F 3차 스플라인 + 순수 비전 28D 특화 피처 + Level 1~4 Two-Tier Tri-Ensemble + 3-State UX\n"
        "• 규격 준수: AGENTS.md 표준 (피험자 9명 전수, 카메라 5대 공식 명칭 A~E 감사, AI-Hub JSON/라벨 엄격 격리)"
    )

    # 1. 기획 핵심 요약
    h1 = doc.add_heading("1. 기획 핵심 요약 (Executive Summary)", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.add_run("1. 연구 목표: ").bold = True
    p.add_run("바벨 런지의 제4 핵심 결함인 '척추의 중립 위반(Spine Neutrality Defect)'을 고정밀 판정하는 독립 세부 오류 모델 c4를 기획·설계한다.\n")
    p.add_run("2. 선행 아키텍처 계승: ").bold = True
    p.add_run("스쿼트(c1~c4), 데드리프트(d1~d5), 스티프 데드리프트(sd1~sd3 v4)에서 성공적으로 검증된 '64F 3차 스플라인 표준', '피험자 9명 전수 격리(StratifiedGroupKFold)', '5대 시점 전수 감사', 'Level 1~4 Two-Tier 모호성 분할', '3-State UX(Clean/Defect/Retake)' 구조를 100% 계승한다.\n")
    p.add_run("3. 런지 특화 척추 역학 혁신: ").bold = True
    p.add_run("스쿼트/데드리프트와 같은 양발 대칭 시상면 운동과 달리, 하지 분할(Split-Stance)로 인한 비대칭 기저면에서 발생하는 ① 전두면 체간 측굴(몸통 좌우 갸우뚱, Spine Roll), ② 경추 4방향 시선/목 이탈(Cervical 3-Axis: 좌/우/상/하), ③ 하지-골반-척추 운동사슬 보상 붕괴를 포착하는 28차원 특화 도메인 피처셋을 도입한다.\n")
    p.add_run("4. 조건 간 직교 분리: ").bold = True
    p.add_run("원천 데이터 정밀 감사 결과 밝혀진 조건 4(척추 중립: 경추 정렬 & 전두면 롤)와 조건 5(상체 과도한 숙임/젖힘: 순수 시상면 피치)의 라벨링 규칙을 명확히 분리하여 모델 간 피처 간섭을 원천 차단한다.")

    # 2. 도메인 및 생체역학
    h2 = doc.add_heading("2. 기획 개요 및 생체역학적 배경", level=1)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.add_run("바벨 런지(Barbell Lunge)는 양 발이 전후로 넓게 벌어지는 일측성 하지 분할(Split-Stance) 동작으로 인해, 신체의 기저면(Base of Support)이 좌우 방향으로 급격히 좁아집니다. 이로 인해 상체에 고중량 바벨을 얹고 하강할 때, 골반의 관상면 경사(Trendelenburg drop)와 이에 대응하는 체간의 반사적 측굴(Lateral Trunk Flexion)이 매우 빈번하게 발생합니다. 또한 밸런스를 잡기 위해 시선이 불안정해지면서 경추가 회전하거나 과도하게 숙여지고 젖혀지는 경추 정렬 붕괴가 연쇄적으로 동반됩니다.")

    create_callout_box(
        doc,
        "런지 척추 중립 파괴의 3대 임상적 위험성",
        "• 전두면 체간 측굴 (몸통 옆으로 갸우뚱): 요추부 척추간판(추간판)에 비대칭적 전단력(Asymmetric Shear Stress)을 집중시켜 급성 요추 염좌 및 디스크 탈출 유발.\n"
        "• 경추 4방향 시선 이탈 (하늘/바닥/좌/우 보고): 고중량 부하 하에서 경추 전만각(Cervical Lordosis)이 소실되거나 과신전되어 승모근 연축 및 경추 신경근 압박 초래.\n"
        "• 골반-요추 커플링 붕괴: 선행 다리 둔근 지지 실패로 골반이 주저앉으며 흉추-요추 연쇄 보상 굴곡이 발생하여 전신 코어 안정성 붕괴."
    )

    # 3. AI-Hub 원천 데이터 감사
    h3 = doc.add_heading("3. AI-Hub 원천 데이터 정밀 감사 (Data Audit & Ground Truth Profile)", level=1)
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.add_run("AI-Hub의 '04_바벨_런지' 데이터셋 1,644개 시퀀스(총 3,288개 JSON 파일)를 전수 감사한 결과, 4번째 조건인 '척추의 중립'은 정상 571건(34.73%), 결함 1,073건(65.27%)으로 결함이 우세한 분포를 형성하고 있습니다.")

    p_tbl = doc.add_paragraph()
    p_tbl.add_run("3.1 척추 중립 결함(1,073건) 세부 Key 전수 매핑 및 분류 체계\n").bold = True

    tbl1 = doc.add_table(rows=1, cols=4)
    format_table(
        tbl1,
        [1.2, 1.6, 1.0, 2.7],
        ["결함 유형군", "원천 Description 및 Key", "샘플 수", "운동역학적 결함 기전 및 판정 특징"],
        [
            ["체간 전두면 측굴\n(몸통 갸우뚱)", "Key 333, 337, 340, 342,\n344, 345, 346", "362건", "하지 지지 불안정에 따른 체간의 좌우 기울임(Spine Roll). 단독 결함(52건) 및 타 관절 결함과의 복합 발생(310건)"],
            ["경추 횡단면 회전\n(좌우 시선 편향)", "Key 347, 348, 351, 352,\n355, 356", "308건", "고개가 좌/우로 돌아가며 경추 Yaw 편차 발생. 어깨선 비대칭 비틀림과 동반되어 상체 중립 파괴"],
            ["경추 과신전\n(하늘보고)", "Key 349, 353,\n359, 360", "195건", "경추 시상면 과신전(Pitch Up). 턱이 들리고 목이 꺾여 경추-흉추 이행부 부하 집중"],
            ["경추 과굴곡\n(바닥보고)", "Key 350, 354,\n357, 358", "208건", "경추 시상면 과굴곡(Pitch Down). 고개를 떨구어 시선이 바닥으로 향하며 상부 흉추 후만 유발"],
            ["정상 기준군\n(Spine Neutral)", "Key 329, 330, 331, 332,\n334, 335, 336 등", "571건", "경추 3축 중립 및 전두면 척추 중심선 수직 유지. (Key 334 상체 뒤로 젖힘도 척추 중립은 정상 판정됨)"]
        ],
        align_rights=[2]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    create_callout_box(
        doc,
        "조건 4(척추 중립) vs 조건 5(상체 숙임/젖힘)의 상호 분리 메커니즘",
        "AI-Hub 라벨 분석 결과, Key 334('상체 뒤로 젖힘')는 조건 4(척추 중립)가 True(정상)이고 조건 5만 False(결함)입니다. 반면 Key 347~350('하늘/바닥/좌/우 보고 + 상체 뒤로 젖힘')은 조건 4와 조건 5가 동시에 False로 판정됩니다.\n"
        "이는 AI-Hub 평가 기준 상, 조건 5는 '체간의 전체적인 시상면 전후 경사(Torso Pitch)'만을 평가하며, 조건 4는 '경추 4방향 시선/목 정렬' 및 '체간의 전두면 좌우 갸우뚱(Roll)'을 평가함을 명백히 보여줍니다. 따라서 c4 피처셋에서는 순수 체간 피치를 배제하고 [경추 3축 벡터]와 [전두면 측굴각]에 100% 집중하여 조건 간 직교 분리(Orthogonal Decoupling)를 실현합니다."
    )

    # 4. 기존 모델 구성안 분석 및 계승
    h4 = doc.add_heading("4. 기존 세부 오류 모델 구성안 분석 및 핵심 계승 원칙", level=1)
    h4.paragraph_format.space_before = Pt(12)
    h4.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.add_run("본 프로젝트에서 기구축된 스쿼트, 데드리프트, 스티프 데드리프트 세부 오류 모델들의 핵심 공학적 설계와 성과를 전수 분석하고, 이를 바벨 런지 c4 파이프라인에 계승하는 체계를 확립하였습니다.")

    tbl2 = doc.add_table(rows=1, cols=4)
    format_table(
        tbl2,
        [1.3, 1.4, 1.2, 2.6],
        ["기존 세부 오류 모델", "세부 결함 정의 및 모델 구조", "실측 성능 달성치", "바벨 런지 c4 적용 및 계승 방식"],
        [
            ["바벨 스쿼트 c2\n(시선 전방 미주시)", "19D VOR 및 경추-체간\n협응 피처 + GBDT", "F1 91.20%\n정확도 89.5%", "• 코-양안-양귀 3D 기하 벡터 기반 경추 3축(Yaw, Pitch, Roll) 분해 기법 계승\n• 흉곽 정면 벡터 대비 두부 시선 벡터 편차각 피처 이식"],
            ["바벨 스쿼트 c4\n(척추 중립 위반)", "28D 골반-요추 커플링\n+ Dilated TCN", "F1 88.50%\nAUC 89.1%", "• C7(어깨중심)-천골(골반중심) 기준선 대비 척추 굴곡도 및 곡률 적분 계승\n• 최하사점 윈도우 위상 고립 기법 적용"],
            ["바벨 데드리프트 d1\n(척추 중립성 위반)", "14D 역학 피처 +\nTwo-Tier 3-State", "골든 F1 98.14%\n실무 F1 81.61%", "• 3-State 신뢰 보류 게이팅(Clean/Defect/Retake) 도입으로 실무 오탐 30% 급감 계승\n• 전방 45° 시점(시점 B, D) 최적 진단 앵글 활용"],
            ["스티프 데드리프트 sd2\n(척추 중립 위반 v3)", "3대 결함 특화 피처 +\nTwo-Tier Tri-Ensemble", "Tier 1 F1 97.01%\nTier 2 F1 91.12%", "• 3대 결함(롤, 락아웃 전만, 등 굽힘) 특화 피처 분할 전략 계승\n• Level 1~4 모호성 정량화 기반 2-Tier 계층 평가 체계 100% 이식\n• 노이즈 0% 물리적 골격 증강(Bone Scaling + DTW) 전면 적용"]
        ]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 5. 도메인 피처 엔지니어링 설계
    h5 = doc.add_heading("5. 바벨 런지 c4 특화 28차원 도메인 피처 엔지니어링 설계", level=1)
    h5.paragraph_format.space_before = Pt(12)
    h5.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.add_run("바벨 런지 척추 중립 결함의 3대 물리적 양상인 [경추 4방향 시선/목 이탈], [체간 전두면 좌우 갸우뚱], [하지 비대칭 사슬 붕괴]를 완벽히 포착하기 위해 총 28차원의 순수 비전 특화 피처셋(28D)을 설계하였습니다.")

    tbl3 = doc.add_table(rows=1, cols=4)
    format_table(
        tbl3,
        [1.3, 1.8, 0.7, 2.7],
        ["피처 패밀리", "피처명 (Feature Name)", "차원", "수학적/생체역학적 정의 및 결함 판정 역할"],
        [
            ["그룹 1: 경추 3축 정렬\n(Cervical 3-Axis)\n[8차원]",
             "cervical_yaw_max\ncervical_yaw_mean\ncervical_pitch_max\ncervical_pitch_min\ncervical_roll_max\near_shoulder_asym\nhead_trunk_yaw_diff\ncervical_bottom_pitch",
             "8D",
             "• 코(0), 양 눈(1~5), 양 귀(7, 8) 3D 기하 평면 기반 Yaw/Pitch/Roll 산출\n• 왼쪽/오른쪽보고(Yaw > 15°), 하늘보고(Pitch > 20°), 바닥보고(Pitch < -15°) 정밀 포착\n• 흉곽 정면 벡터 대비 두부 시선 벡터의 상대 편차각 감지\n• 최하사점(t_bottom) 순간의 경추 피치 각도 스냅샷"],
            ["그룹 2: 전두면 체간 측굴\n(Frontal Spine Roll)\n[8차원]",
             "trunk_roll_max\ntrunk_roll_mean\nshoulder_roll_max\npelvic_obliquity_roll\nspine_lateral_inflexion\ntorso_bottom_roll\nshoulder_pelvis_coupling\ntrunk_roll_integral",
             "8D",
             "• 양 어깨 중심-양 골반 중심을 잇는 척추 중심선의 전두면 롤 각도\n• '몸통 옆으로 갸우뚱(Key 333 계열)' 결함의 직접적 수치화\n• C7-천골 기준선 대비 흉추 횡방향 이탈 깊이(Inflexion Depth)\n• 어깨선과 골반선의 비대칭 기울기 커플링 지수 산출\n• 임계치(5°) 초과 체간 측굴 지속 시간 적분치"],
            ["그룹 3: 하지 비대칭 사슬\n(Kinetic Chain)\n[6차원]",
             "split_base_stability_ratio\npelvis_lateral_sway\nspine_knee_alignment\nasym_hip_drop_bottom\nkinetic_lateral_drift\ncounter_tilt_compensation",
             "6D",
             "• 앞발-뒷발 전후 스텝 간격 대비 좌우 기저면 너비비\n• 하지 분할 기저면 위에서 골반 중심의 좌우 요동폭(Lateral Sway)\n• 선행 무릎 수직선과 척추 축의 관상면 일치도\n• 최하사점에서 골반 기울임에 대한 상체의 반대방향 보상 꺾임(Counter-tilt) 계측"],
            ["그룹 4: 최하사점 위상\n(Bottom Dynamic Phase)\n[6차원]",
             "bottom_spine_stability_std\nbottom_head_jitter_std\ndescent_roll_vel_peak\nascent_recovery_sway\nbottom_active_roll_mean\nbottom_active_yaw_mean",
             "6D",
             "• 무릎 최하사점(t_bottom ± 6 프레임) Active Window 내 척추 안정성(표준편차)\n• 최대 부하 시점에서의 두부 시선 요동 표준편차\n• 하강 구간 최대 측굴 각속도 및 상승 복귀 구간 누적 요동치\n• Active Window 내 평균 체간 롤 및 경추 요 각도"]
        ],
        align_rights=[2]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 6. 증강 및 Two-Tier 평가
    h6 = doc.add_heading("6. 노이즈 0% 물리적 증강 및 Two-Tier 계층 평가 체계", level=1)
    h6.paragraph_format.space_before = Pt(12)
    h6.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.add_run("스티프 데드리프트 sd2 v3에서 입증된 SOTA 방법론을 100% 계승하여 인위적 노이즈를 배제한 물리적 골격 증강과 4단계 모호성 기반의 Two-Tier 평가 체계를 적용합니다.")

    create_callout_box(
        doc,
        "노이즈 0% 물리 기반 골격 증강 (Bone Scaling & DTW Temporal Warping)",
        "• 무작위 가우시안 지터 노이즈 영구 배제(0%): 인위적 난수는 척추 미세 각도의 물리적 정합성을 왜곡하므로 완전 금지함.\n"
        "• 뼈대 길이 가변(Bone Scaling): 대퇴골, 정강이, 체간 척추의 길이를 신체 비율 범위(±10%) 내에서 물리적으로 신축시켜 체형 다양성 확보.\n"
        "• 비선형 시간 왜곡(DTW Temporal Warping): 런지 하강-정점-상승의 위상 속도를 가변시켜 다양한 운동 템포에 대한 강건성 확보."
    )

    tbl4 = doc.add_table(rows=1, cols=4)
    format_table(
        tbl4,
        [1.3, 1.2, 1.6, 2.4],
        ["모호성 등급", "구분", "물리적 정량 판정 기준", "평가 계층 (Evaluation Tier)"],
        [
            ["Level 1", "명확군 (Clear)", "경추 편차 > 20° 또는 체간 측굴 > 10°\n정상 기준선 편차 < 3°", "★ Tier 1 (골든 벤치마크)\n목표 결함 F1: 97.0%+ (정밀도 98%+)"],
            ["Level 2", "안정군 (Moderate)", "경추 편차 14~20° 또는 체간 측굴 7~10°\n일관된 패턴의 전형적 결함", "★ Tier 2 (실무 전체 벤치마크)\n목표 결함 F1: 91.0%+ (정확도 85%+)"],
            ["Level 3", "경계군 (Borderline)", "경추 편차 8~14° 또는 체간 측굴 4~7°\n경미한 밸런스 흔들림", "Tier 2에 포함되어 실무 강건성 검증"],
            ["Level 4", "극단 모호군 (Ambiguous)", "카메라 원근 왜곡 또는 라벨 판정 경계\n인간 전문가도 육안 판정 애매", "신뢰 보류(3-State Retake) 대상으로 격리"]
        ]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 7. Tri-Ensemble 및 3-State
    h7 = doc.add_heading("7. 하이브리드 Tri-Ensemble 아키텍처 및 3-State 추론 엔진", level=1)
    h7.paragraph_format.space_before = Pt(12)
    h7.paragraph_format.space_after = Pt(6)

    create_callout_box(
        doc,
        "하이브리드 Tri-Ensemble & 3-State 추론 엔진 구성",
        "1. 시공간 시계열 심층망: 64프레임 Dilated TCN (경추-체간 위상 궤적 및 속도/가속도 패턴 학습)\n"
        "2. 도메인 트리 앙상블: LightGBM + XGBoost + CatBoost (28D 생체역학 피처 기반 초정밀 결정경계 학습)\n"
        "3. 3-State Selective Gating: 모델 추론 확률의 확신성(Margin = |P - 0.5|)이 임계치(tau_abstain) 미만인 애매한 구간(Level 4)에 대해 무리한 판정 대신 '재촬영 유도(신뢰 보류)'를 수행하여 실무 오탐율(False Positive)을 30% 이상 억제함."
    )

    # 8. 공식 5대 시점 감사 및 로드맵
    h8 = doc.add_heading("8. 공식 5대 시점 전수 감사 계획 및 단계별 로드맵", level=1)
    h8.paragraph_format.space_before = Pt(12)
    h8.paragraph_format.space_after = Pt(6)

    tbl5 = doc.add_table(rows=1, cols=4)
    format_table(
        tbl5,
        [1.2, 1.8, 1.2, 2.3],
        ["공식 시점 명칭", "카메라 각도 및 방향", "내부 시점 코드", "바벨 런지 c4 진단 특성 및 기대 효과"],
        [
            ["시점 A", "우측 후방 (45도)", "view5", "후방 견갑골 밸런스 및 척추 기립근 비대칭 관측"],
            ["시점 B", "우측 전방 (45도)", "view2", "경추 좌우 회전 및 전두면 체간 측굴이 동시에 시인되는 핵심 앵글"],
            ["시점 C", "완전 정면 (0도)", "view3", "★ 몸통 좌우 갸우뚱(체간 롤) 및 코-흉골 중심선 이탈 감지 최고 앵글"],
            ["시점 D", "좌측 전방 (45도)", "view4", "★ 경추 상하 피치(하늘/바닥) 및 시선 방향성 판별에 탁월한 권장 시점"],
            ["시점 E", "좌측 후방 (45도)", "view1", "후면 골반 경사 및 하지 분할 비대칭 보상 관측 보완"]
        ]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    p_step = doc.add_paragraph()
    p_step.add_run("8.1 5단계 실행 마일스톤 (Implementation Roadmap)\n").bold = True

    tbl6 = doc.add_table(rows=1, cols=4)
    format_table(
        tbl6,
        [1.2, 1.8, 1.2, 2.3],
        ["단계 (Phase)", "주요 작업 내용", "담당 산출물 모듈", "완료 기준 (KPI)"],
        [
            ["Step 1", "바벨 런지 MediaPipe 33 64F\n스플라인 전처리 데이터셋 구축", "build_lunge_full33_64f_dataset.py", "1,644개 시퀀스 100% 추출 완료\n(피험자 9명, 시점 A~E 전수)"],
            ["Step 2", "28차원 런지 척추 중립 특화\n피처 엔지니어링 모듈 구현", "lunge_c4_features.py\nlunge_c4_feature_extractor.py", "경추 8D + 측굴 8D + 사슬 6D + 위상 6D\n피처 규격 무결성 및 NaN 0% 검증"],
            ["Step 3", "Level 1~4 모호성 정량화\n및 캐시 데이터셋 생성", "build_lunge_c4_ambiguity_levels.py", "모호성 4단계 분류 캐시(.npz) 생성\nTier 1/Tier 2 분할 완료"],
            ["Step 4", "Two-Tier Tri-Ensemble\n모델 학습 및 하이퍼파라미터 튜닝", "train_and_eval_lunge_c4_twotier.py", "Tier 1 F1 97%+ / Tier 2 F1 91%+\n5-Fold 피험자 완전 격리 검증"],
            ["Step 5", "공식 5대 시점 전수 감사,\n배포 번들 패키징 및 최종 보고서", "lunge_c4_classifier.py\n바벨_런지_c4_척추중립_최종보고서.docx", "시점 A~E 전수 성능 감사표 작성\n단일 추론 인터페이스 단위테스트 100% 통과"]
        ]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 저장
    doc.save(DOCX_PATH)
    print(f"[+] Word 문서 생성 완료: {DOCX_PATH}")

    # Markdown 동시 생성
    md_content = f"""# [vidio-health-trainer] 바벨 런지 c4(척추의 중립 위반) 세부 오류 모델 구축 사양 및 공학적 기획서

> **프로젝트 공식 명칭**: `vidio-health-trainer`  
> **문서 버전 / 상태**: v1.0 (정식 기획 확정)  
> **대상 운동 및 오류**: `04_바벨_런지` / `c4`: 척추의 중립 위반 (Spine Neutrality Defect)  
> **원천 데이터 규모**: AI-Hub 1,644개 시퀀스 전수 (정상 571건 / 결함 1,073건, 결함 비율 65.27%)  
> **작성 일자**: 2026년 10월 03일 (6주차)  
> **핵심 기술 체계**: 64F 3차 스플라인 + 순수 비전 28D 특화 피처 + Level 1~4 Two-Tier Tri-Ensemble + 3-State UX  
> **규격 준수**: `AGENTS.md` 표준 (피험자 9명 전수, 카메라 5대 공식 명칭 A~E 감사, AI-Hub JSON/라벨 엄격 격리)  

---

## 1. 기획 핵심 요약 (Executive Summary)

1. **연구 목표**: 바벨 런지의 제4 핵심 결함인 '척추의 중립 위반(Spine Neutrality Defect)'을 고정밀 판정하는 독립 세부 오류 모델 c4를 기획·설계한다.
2. **선행 아키텍처 계승**: 스쿼트(c1~c4), 데드리프트(d1~d5), 스티프 데드리프트(sd1~sd3 v4)에서 성공적으로 검증된 '64F 3차 스플라인 표준', '피험자 9명 전수 격리(StratifiedGroupKFold)', '5대 시점 전수 감사', 'Level 1~4 Two-Tier 모호성 분할', '3-State UX(Clean/Defect/Retake)' 구조를 100% 계승한다.
3. **런지 특화 척추 역학 혁신**: 스쿼트/데드리프트와 같은 양발 대칭 시상면 운동과 달리, 하지 분할(Split-Stance)로 인한 비대칭 기저면에서 발생하는 ① 전두면 체간 측굴(몸통 좌우 갸우뚱, Spine Roll), ② 경추 4방향 시선/목 이탈(Cervical 3-Axis: 좌/우/상/하), ③ 하지-골반-척추 운동사슬 보상 붕괴를 포착하는 28차원 특화 도메인 피처셋을 도입한다.
4. **조건 간 직교 분리**: 원천 데이터 정밀 감사 결과 밝혀진 조건 4(척추 중립: 경추 정렬 & 전두면 롤)와 조건 5(상체 과도한 숙임/젖힘: 순수 시상면 피치)의 라벨링 규칙을 명확히 분리하여 모델 간 피처 간섭을 원천 차단한다.

---

## 2. 기획 개요 및 생체역학적 배경

바벨 런지(Barbell Lunge)는 양 발이 전후로 넓게 벌어지는 일측성 하지 분할(Split-Stance) 동작으로 인해, 신체의 기저면(Base of Support)이 좌우 방향으로 급격히 좁아집니다. 이로 인해 상체에 고중량 바벨을 얹고 하강할 때, 골반의 관상면 경사(Trendelenburg drop)와 이에 대응하는 체간의 반사적 측굴(Lateral Trunk Flexion)이 매우 빈번하게 발생합니다. 또한 밸런스를 잡기 위해 시선이 불안정해지면서 경추가 회전하거나 과도하게 숙여지고 젖혀지는 경추 정렬 붕괴가 연쇄적으로 동반됩니다.

### 런지 척추 중립 파괴의 3대 임상적 위험성
- **전두면 체간 측굴 (몸통 옆으로 갸우뚱)**: 요추부 척추간판(추간판)에 비대칭적 전단력(Asymmetric Shear Stress)을 집중시켜 급성 요추 염좌 및 디스크 탈출 유발.
- **경추 4방향 시선 이탈 (하늘/바닥/좌/우 보고)**: 고중량 부하 하에서 경추 전만각(Cervical Lordosis)이 소실되거나 과신전되어 승모근 연축 및 경추 신경근 압박 초래.
- **골반-요추 커플링 붕괴**: 선행 다리 둔근 지지 실패로 골반이 주저앉으며 흉추-요추 연쇄 보상 굴곡이 발생하여 전신 코어 안정성 붕괴.

---

## 3. AI-Hub 원천 데이터 정밀 감사 (Data Audit & Ground Truth Profile)

총 1,644개의 JSON 라벨 및 이미지 디렉토리를 전수 감사한 결과, 4번째 조건인 '척추의 중립'은 **정상 571건(34.73%) vs 결함 1,073건(65.27%)**의 분포를 나타냅니다.

### 3.1 척추 중립 결함(1,073건) 세부 Key 전수 매핑 및 분류 체계
| 결함 유형군 | 해당 Key 번호 | 샘플 수 | 운동역학적 결함 기전 및 판정 특징 |
| :--- | :--- | :---: | :--- |
| **체간 전두면 측굴 (몸통 갸우뚱)** | Key 333, 337, 340, 342, 344, 345, 346 | 362건 | 하지 지지 불안정에 따른 체간의 좌우 기울임(Spine Roll). 단독 결함(52건) 및 타 관절 결함과의 복합 발생(310건) |
| **경추 횡단면 회전 (좌우 시선 편향)** | Key 347, 348, 351, 352, 355, 356 | 308건 | 고개가 좌/우로 돌아가며 경추 Yaw 편차 발생. 어깨선 비대칭 비틀림과 동반되어 상체 중립 파괴 |
| **경추 과신전 (하늘보고)** | Key 349, 353, 359, 360 | 195건 | 경추 시상면 과신전(Pitch Up). 턱이 들리고 목이 꺾여 경추-흉추 이행부 부하 집중 |
| **경추 과굴곡 (바닥보고)** | Key 350, 354, 357, 358 | 208건 | 경추 시상면 과굴곡(Pitch Down). 고개를 떨구어 시선이 바닥으로 향하며 상부 흉추 후만 유발 |
| **정상 기준군 (Spine Neutral)** | Key 329, 330, 331, 332, 334, 335, 336 등 | 571건 | 경추 3축 중립 및 전두면 척추 중심선 수직 유지. (Key 334 상체 뒤로 젖힘도 척추 중립은 정상 판정됨) |

> **[조건 4(척추 중립) vs 조건 5(상체 숙임/젖힘)의 상호 분리 메커니즘]**  
> AI-Hub 라벨 분석 결과, Key 334('상체 뒤로 젖힘')는 조건 4(척추 중립)가 True(정상)이고 조건 5만 False(결함)입니다. 반면 Key 347~350('하늘/바닥/좌/우 보고 + 상체 뒤로 젖힘')은 조건 4와 조건 5가 동시에 False로 판정됩니다.  
> 이는 AI-Hub 평가 기준 상, 조건 5는 '체간의 전체적인 시상면 전후 경사(Torso Pitch)'만을 평가하며, 조건 4는 '경추 4방향 시선/목 정렬' 및 '체간의 전두면 좌우 갸우뚱(Roll)'을 평가함을 명백히 보여줍니다. 따라서 c4 피처셋에서는 순수 체간 피치를 배제하고 [경추 3축 벡터]와 [전두면 측굴각]에 100% 집중하여 조건 간 직교 분리(Orthogonal Decoupling)를 실현합니다.

---

## 4. 기존 세부 오류 모델 구성안 분석 및 핵심 계승 원칙

| 기존 세부 오류 모델 | 세부 결함 정의 및 모델 구조 | 실측 성능 달성치 | 바벨 런지 c4 적용 및 계승 방식 |
| :--- | :--- | :---: | :--- |
| **바벨 스쿼트 c2 (시선 전방 미주시)** | 19D VOR 및 경추-체간 협응 피처 + GBDT | F1 91.20% / Acc 89.5% | • 코-양안-양귀 3D 기하 벡터 기반 경추 3축(Yaw, Pitch, Roll) 분해 기법 계승<br>• 흉곽 정면 벡터 대비 두부 시선 벡터 편차각 피처 이식 |
| **바벨 스쿼트 c4 (척추 중립 위반)** | 28D 골반-요추 커플링 + Dilated TCN | F1 88.50% / AUC 89.1% | • C7(어깨중심)-천골(골반중심) 기준선 대비 척추 굴곡도 및 곡률 적분 계승<br>• 최하사점 윈도우 위상 고립 기법 적용 |
| **바벨 데드리프트 d1 (척추 중립성 위반)** | 14D 역학 피처 + Two-Tier 3-State | Golden F1 98.14% / 실무 81.61% | • 3-State 신뢰 보류 게이팅(Clean/Defect/Retake) 도입으로 실무 오탐 30% 급감 계승<br>• 전방 45° 시점(시점 B, D) 최적 진단 앵글 활용 |
| **스티프 데드리프트 sd2 (척추 중립 위반 v3)** | 3대 결함 특화 피처 + Two-Tier Tri-Ensemble | Tier 1 F1 97.01% / Tier 2 F1 91.12% | • 3대 결함(롤, 락아웃 전만, 등 굽힘) 특화 피처 분할 전략 계승<br>• Level 1~4 모호성 정량화 기반 2-Tier 계층 평가 체계 100% 이식<br>• 노이즈 0% 물리적 골격 증강(Bone Scaling + DTW) 전면 적용 |

---

## 5. 바벨 런지 c4 특화 28차원 도메인 피처 엔지니어링 설계

```mermaid
flowchart TD
    A["입력: 64프레임 런지 시퀀스 (MediaPipe 33관절 3D)"] --> B["[계통 1] 경추 3축(Yaw/Pitch/Roll) 기하 벡터 추출 (8D)"]
    A --> C["[계통 2] 체간 전두면 측굴 & 척추 롤 분석 (8D)"]
    A --> D["[계통 3] 하지 분할-골반-척추 운동사슬 협응 (6D)"]
    A --> E["[계통 4] 최하사점(Bottom Phase) 동적 안정성 (6D)"]
    B --> F["28차원 바벨 런지 c4 특화 피처 벡터 산출 (28D)"]
    C --> F
    D --> F
    E --> F
    F --> G["Level 1~4 모호성 정량화 & Two-Tier Tri-Ensemble 분류기"]
```

### 28차원 전용 도메인 피처셋 명세
1. **그룹 1: 경추 3축 정렬 피처 (8D)**
   - `cervical_yaw_max`, `cervical_yaw_mean`: 코-양안-양귀 3D 평면의 횡단면 회전각 (좌우 시선 이탈 감지)
   - `cervical_pitch_max`, `cervical_pitch_min`: 코-귀 벡터와 수평면 간의 피치각 (하늘/바닥 시선 감지)
   - `cervical_roll_max`: 양 눈/양 귀 선의 전두면 기울기
   - `ear_shoulder_asymmetry`: 좌/우 귀와 좌/우 어깨 간의 상대적 수직 거리 비대칭도
   - `head_trunk_yaw_diff`: 흉곽 정면 벡터 대비 두부 시선 벡터 간의 상대 편차각
   - `cervical_bottom_pitch`: 최하사점($t_{{bottom}}$) 순간의 경추 피치 각도 스냅샷
2. **그룹 2: 전두면 체간 측굴 피처 (8D)**
   - `trunk_roll_max`, `trunk_roll_mean`: 양 어깨 중심-양 골반 중심을 잇는 척추 중심선의 전두면 롤 각도 (몸통 갸우뚱 감지)
   - `shoulder_roll_max`: 양 어깨선의 수평선 대비 롤 각도 최대치
   - `pelvic_obliquity_roll`: 양 골반선의 수평선 대비 기울기
   - `spine_lateral_inflexion`: C7-천골 기준선 대비 흉추 횡방향 이탈 깊이(Inflexion Depth)
   - `torso_bottom_roll`: 최하사점 순간의 체간 롤 각도
   - `shoulder_pelvis_coupling`: 어깨선과 골반선의 비대칭 기울기 커플링 지수
   - `trunk_roll_integral`: 임계치(5°) 초과 체간 측굴 지속 시간 적분치
3. **그룹 3: 하지 비대칭 사슬 피처 (6D)**
   - `split_base_stability_ratio`: 앞발-뒷발 전후 스텝 간격 대비 좌우 기저면 너비비
   - `pelvis_lateral_sway`: 하지 분할 기저면 위에서 골반 중심의 좌우 요동폭(Lateral Sway)
   - `spine_knee_alignment`: 선행 무릎 수직선과 척추 축의 관상면 일치도
   - `asym_hip_drop_bottom`: 최하사점에서의 비대칭 골반 하강률
   - `kinetic_lateral_drift`: 발목-골반-어깨-두부로 이어지는 횡방향 중심선 누적 편차
   - `counter_tilt_compensation`: 골반 기울임에 대해 상체가 반대로 꺾이는 보상 작용(Counter-tilt) 계측
4. **그룹 4: 최하사점 위상 및 동역학 피처 (6D)**
   - `bottom_spine_stability_std`: 무릎 최하사점($t_{{bottom}} \pm 6$ 프레임) Active Window 내 척추 안정성(표준편차)
   - `bottom_head_jitter_std`: 최대 부하 시점에서의 두부 시선 요동 표준편차
   - `descent_roll_vel_peak`: 하강 구간 최대 측굴 각속도 피크
   - `ascent_recovery_sway`: 상승 복귀 구간 누적 요동치
   - `bottom_active_roll_mean`: Active Window 내 평균 체간 롤 각도
   - `bottom_active_yaw_mean`: Active Window 내 평균 경추 요 각도

---

## 6. 노이즈 0% 물리적 증강 및 Two-Tier 계층 평가 체계

1. **노이즈 0% 물리 기반 골격 증강 (Bone Scaling & DTW)**:
   - 무작위 가우시안 지터 노이즈 영구 배제(0%): 인위적 난수는 척추 미세 각도의 물리적 정합성을 왜곡하므로 완전 금지.
   - 뼈대 길이 가변(Bone Scaling): 대퇴골, 정강이, 체간 척추의 길이를 신체 비율 범위(±10%) 내에서 물리적으로 신축시켜 체형 다양성 확보.
   - 비선형 시간 왜곡(DTW Temporal Warping): 런지 하강-정점-상승의 위상 속도를 가변시켜 다양한 운동 템포에 대한 강건성 확보.
2. **4단계 물리적 모호성(Level 1~4) 정량화**:
   - **Level 1 (명확군)**: 경추 편차 > 20° 또는 체간 측굴 > 10°, 정상 기준선 편차 < 3° $\rightarrow$ **★ Tier 1 (골든 벤치마크, 목표 F1 97%+)**
   - **Level 2 (안정군)**: 경추 편차 14~20° 또는 체간 측굴 7~10° $\rightarrow$ **★ Tier 2 (실무 벤치마크, 목표 F1 91%+)**
   - **Level 3 (경계군)**: 경추 편차 8~14° 또는 체간 측굴 4~7° $\rightarrow$ Tier 2에 포함되어 실무 강건성 검증
   - **Level 4 (극단 모호군)**: 카메라 원근 왜곡 또는 라벨 판정 경계군 $\rightarrow$ 신뢰 보류(3-State Retake) 대상으로 격리

---

## 7. 하이브리드 Tri-Ensemble 아키텍처 및 3-State 추론 엔진

1. **시공간 시계열 심층망**: 64프레임 Dilated TCN (경추-체간 위상 궤적 및 속도/가속도 패턴 학습)
2. **도메인 트리 앙상블**: LightGBM + XGBoost + CatBoost (28D 생체역학 피처 기반 초정밀 결정경계 학습)
3. **3-State Selective Gating**: 모델 추론 확률의 확신성(Margin = $|P - 0.5|$)이 임계치($\tau_{{abstain}}$) 미만인 애매한 구간(Level 4)에 대해 무리한 판정 대신 '재촬영 유도(신뢰 보류)'를 수행하여 실무 오탐율(False Positive)을 30% 이상 억제함.

---

## 8. 공식 5대 시점 전수 감사 계획 및 단계별 로드맵

| 공식 시점 명칭 (`AGENTS.md`) | 각도 및 방향 | 내부 시점 코드 | 바벨 런지 c4 진단 특성 및 기대 효과 |
| :---: | :---: | :---: | :--- |
| **시점 A** | 우측 후방 (45도) | `view5` | 후방 견갑골 밸런스 및 척추 기립근 비대칭 관측 |
| **시점 B** | 우측 전방 (45도) | `view2` | 경추 좌우 회전 및 전두면 체간 측굴이 동시에 시인되는 핵심 앵글 |
| **시점 C** | 완전 정면 (0도) | `view3` | **★ 몸통 좌우 갸우뚱(체간 롤) 및 코-흉골 중심선 이탈 감지 최고 앵글** |
| **시점 D** | 좌측 전방 (45도) | `view4` | **★ 경추 상하 피치(하늘/바닥) 및 시선 방향성 판별에 탁월한 권장 시점** |
| **시점 E** | 좌측 후방 (45도) | `view1` | 후면 골반 경사 및 하지 분할 비대칭 보상 관측 보완 |

### 8.1 5단계 실행 마일스톤 (Implementation Roadmap)
| 단계 (Phase) | 주요 작업 내용 | 담당 산출물 모듈 | 완료 기준 (KPI) |
| :---: | :--- | :--- | :--- |
| **Step 1** | 바벨 런지 MediaPipe 33 64F 전처리 데이터셋 구축 | `build_lunge_full33_64f_dataset.py` | 1,644개 시퀀스 100% 추출 (피험자 1~9, 시점 A~E 전수) |
| **Step 2** | 28차원 런지 척추 중립 특화 피처 모듈 구현 | `lunge_c4_features.py`<br>`lunge_c4_feature_extractor.py` | 경추 8D + 측굴 8D + 사슬 6D + 위상 6D 무결성 검증 (NaN 0%) |
| **Step 3** | Level 1~4 모호성 정량화 및 캐시 생성 | `build_lunge_c4_ambiguity_levels.py` | 모호성 4단계 분류 캐시(.npz) 생성 및 Tier 1/2 분할 |
| **Step 4** | Two-Tier Tri-Ensemble 학습 및 튜닝 | `train_and_eval_lunge_c4_twotier.py` | Tier 1 F1 97%+ / Tier 2 F1 91%+ 달성 (피험자 격리 5-Fold) |
| **Step 5** | 공식 5대 시점 전수 감사 및 배포 번들링 | `lunge_c4_classifier.py`<br>`바벨_런지_c4_척추중립_최종보고서.docx` | 시점 A~E 전수 감사표 작성 및 단일 추론 단위테스트 100% 통과 |
"""

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[+] Markdown 문서 생성 완료: {MD_PATH}")

if __name__ == '__main__':
    build_docx_and_md()
