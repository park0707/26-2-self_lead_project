# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d5_planning_doc.py
Purpose: 바벨 데드리프트 d5(바벨 궤적과 몸 밀착) 세부 결함 모델 아키텍처 기획서 Word(.docx) 및 Markdown(.md) 동시 생성
Target Path:
  - D:\자주프 문서\6주차\바벨_데드리프트_d5_모델_아키텍처_기획서.docx
  - D:\자주프 문서\6주차\바벨_데드리프트_d5_모델_아키텍처_기획서.md
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_DOCX_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d5_모델_아키텍처_기획서.docx"
OUT_MD_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d5_모델_아키텍처_기획서.md"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
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

def add_header_box(doc, title, subtitle):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "1E3A8A")  # Dark Navy
    set_cell_margins(cell, top=200, bottom=200, left=200, right=200)
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run1 = p.add_run(title + "\n")
    run1.font.name = "맑은 고딕"
    run1.font.size = Pt(18)
    run1.font.bold = True
    run1.font.color.rgb = RGBColor(255, 255, 255)
    
    run2 = p.add_run(subtitle)
    run2.font.name = "맑은 고딕"
    run2.font.size = Pt(11)
    run2.font.color.rgb = RGBColor(224, 231, 255)
    
    doc.add_paragraph()

def add_section_title(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    
    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(30, 58, 138)  # Deep Navy
    elif level == 2:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(37, 99, 235)  # Royal Blue
    else:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(71, 85, 105)

def add_bullet_point(doc, title, desc, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    
    run_t = p.add_run(title + ": ")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(30, 41, 59)
    
    run_d = p.add_run(desc)
    run_d.font.name = "맑은 고딕"
    run_d.font.size = Pt(10)
    run_d.font.color.rgb = RGBColor(51, 65, 85)

def add_callout_box(doc, title, text, bg_hex="EFF6FF", border_hex="3B82F6"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
    
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
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    
    run1 = p.add_run(f"■ {title}\n")
    run1.font.name = "맑은 고딕"
    run1.font.size = Pt(10.5)
    run1.font.bold = True
    run1.font.color.rgb = RGBColor(30, 58, 138)
    
    run2 = p.add_run(text)
    run2.font.name = "맑은 고딕"
    run2.font.size = Pt(9.5)
    run2.font.color.rgb = RGBColor(51, 65, 85)
    
    doc.add_paragraph()

def style_table_header(row, col_widths, bg_hex="1E3A8A"):
    for idx, cell in enumerate(row.cells):
        cell.width = Inches(col_widths[idx])
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "맑은 고딕"
                r.font.size = Pt(9.5)
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)

def style_table_cells(table, col_widths, alt_color=True):
    for r_idx, row in enumerate(table.rows[1:]):
        bg_hex = "F8FAFC" if (r_idx % 2 == 1 and alt_color) else "FFFFFF"
        for c_idx, cell in enumerate(row.cells):
            cell.width = Inches(col_widths[c_idx])
            set_cell_background(cell, bg_hex)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = "맑은 고딕"
                    r.font.size = Pt(9)
                    r.font.color.rgb = RGBColor(51, 65, 85)

def build_docx_and_md():
    doc = docx.Document()
    
    # 여백 설정 (1인치)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    md_lines = []
    def log_md(line=""):
        md_lines.append(line)

    # 1. 헤더
    title = "vidio-health-trainer 바벨 데드리프트 d5 모델 아키텍처 기획서"
    subtitle = "바벨 궤적-신체 밀착 동역학(Barbell-Body Proximity Dynamics) 기반 독립 세부 결함 진단 엔진"
    add_header_box(doc, title, subtitle)
    
    log_md(f"# {title}\n")
    log_md(f"> **문서 성격**: {subtitle}  ")
    log_md(f"> **공식 산출물**: `{OUT_DOCX_PATH}`  ")
    log_md(f"> **프로젝트 명칭**: `vidio-health-trainer`\n")
    log_md("---\n")

    # Executive Summary
    add_callout_box(
        doc,
        "기획 핵심 총람 (Executive Summary)",
        "1. 공식 프로젝트명: vidio-health-trainer\n"
        "2. 데드리프트 결함 체계 상의 위치: d1(척추 중립), d2(무릎-발 방향), d3(발바닥 지면 고정)에 이은 4번째 구현 대상인 d5(바벨 궤적과 몸 밀착) 모델 구축\n"
        "3. d5 결함 정의: 리프팅(First/Second Pull) 및 하강 구간에서 바벨이 신체(정강이, 무릎, 대퇴부)로부터 전방 이탈(Forward Drift)하여 요추 모멘트 암을 폭증시키는 치명적 역학 결함\n"
        "4. 원천 데이터 무결성: AI-Hub 1,592건 중 정상 797건(50.0%) vs 결함 795건(50.0%)의 완벽한 50:50 클래스 균형 확보\n"
        "5. 피처 엔지니어링: 손목 키포인트 가상 바벨 중심점 기반 28D '바벨-신체 밀착 및 수직 궤적 동역학(Barbell-Body Proximity Dynamics)' 독자 설계\n"
        "6. 평가 규격: AI-Hub 피험자 1~9번 전원 전수, 공식 5대 시점(시점 A~E) 전수 포함 5-Fold StratifiedGroupKFold 교차검증"
    )

    log_md("## ■ 기획 핵심 총람 (Executive Summary)")
    log_md("1. **공식 프로젝트명**: `vidio-health-trainer`")
    log_md("2. **데드리프트 결함 체계 상의 위치**: d1(척추), d2(무릎 외반), d3(발바닥 지면 고정)에 이은 4번째 구현 대상인 **d5(바벨 궤적과 몸 밀착)** 모델 구축")
    log_md("3. **d5 결함 정의**: 리프팅 및 하강 동작 중 바벨이 신체(정강이, 무릎, 대퇴부)로부터 전방 이탈(Forward Drift)하여 요추 모멘트 암을 급증시키는 결함")
    log_md("4. **원천 데이터 무결성**: AI-Hub 라벨 1,592건 중 정상 797건(50.0%) vs 결함 795건(50.0%)의 **완벽한 50:50 클래스 균형** 보유")
    log_md("5. **피처 엔지니어링**: 양 손목 기반 가상 바벨 중심점 추적을 통한 **28D '바벨-신체 밀착 및 수직 궤적 동역학'** 독자 설계")
    log_md("6. **검증 규격**: AI-Hub 피험자 1~9번 전원 전수, 공식 5대 시점(시점 A~E) 전수 포함 5-Fold StratifiedGroupKFold 피험자 완전 격리 검증\n")
    log_md("---\n")

    # 1. 배경 및 생체역학적 결함 정의
    add_section_title(doc, "1. 바벨 데드리프트 5대 결함 체계 및 d5의 생체역학적 중요성", level=1)
    
    p = doc.add_paragraph()
    p.add_run(
        "바벨 데드리프트(Conventional Barbell Deadlift)는 지면 정지 상태의 고중량을 신체 전신 사슬을 이용하여 수직으로 거상하는 대표적인 힙 힌지(Hip Hinge) 다관절 복합 운동입니다. "
        "데드리프트의 핵심 역학 원리는 바벨의 무게중심(COM, Center of Mass)이 항상 인체의 기저면(BOS, Base of Support) 중심인 미드풋(Midfoot) 바로 위 수직선상을 따라 이동해야 한다는 점입니다."
    )
    
    add_bullet_point(doc, "결함 상태 정의 (바벨 궤적과 몸 밀착 = False)", "리프팅 초기(First Pull, 바닥~정강이), 무릎 통과(Knee Passing), 락아웃(Lockout), 그리고 하강(Eccentric) 구간에 이르기까지 바벨이 정강이, 무릎, 대퇴 전면부와 밀착되지 못하고 전방(Forward)으로 표류(Drifting)하는 현상.")
    add_bullet_point(doc, "모멘트 암 폭증 기전", "바벨이 신체 중심축에서 전방으로 거리 d만큼 이탈하면, 요추 L4-L5 추간판 및 천골 고관절 중심부에 가해지는 굴곡 모멘트(Flexion Moment = F_bar × (d_body + d_drift))가 기하급수적으로 폭증합니다.")
    add_bullet_point(doc, "d1(척추 중립 위반) 유발과의 인과성", "바벨 이탈로 인한 극단적 전단력(Shear Stress)을 척추기립근과 둔근이 버티지 못하게 되면서, 필연적으로 요추 굴곡(Lumbar Flexion / Rounding) 즉 d1 결함을 유발하는 가장 근본적인 역학적 원인이 됩니다.")
    add_bullet_point(doc, "광배근 결합 결손(Lat Engagement Deficit)", "광배근(Latissimus Dorsi)의 상완골 신전 토크(Shoulder Extension Torque)가 작동하지 않아 바벨을 몸쪽으로 당겨 고정하지 못할 때 주로 발생합니다.")

    log_md("## 1. 바벨 데드리프트 5대 결함 체계 및 d5의 생체역학적 중요성\n")
    log_md("| 결함 코드 | 공식 결함 명칭 | 원천 데이터 보유 (1,592건) | 데이터 클래스 비율 | 진단 및 개발 상태 |")
    log_md("| :---: | :--- | :---: | :---: | :--- |")
    log_md("| **d1** | 척추의 중립 위반 | 794건 결함 / 798건 정상 | 49.9% : 50.1% | 완료 (d1 v1~v3.1 배포) |")
    log_md("| **d2** | 발과 무릎의 방향 일치 | 748건 결함 / 844건 정상 | 47.0% : 53.0% | 완료 (d2 v1~v2.0 배포) |")
    log_md("| **d3** | 발바닥 지면 고정 위반 | 796건 결함 / 796건 정상 | 50.0% : 50.0% | 완료 (d3 v1.0 배포) |")
    log_md("| **d4** | 무릎과 골반 동시 펴짐 | 794건 결함 / 798건 정상 | 49.9% : 50.1% | 차기 예정 (d4 계획) |")
    log_md("| **d5 (기획)** | **바벨 궤적과 몸 밀착** | **795건 결함 / 797건 정상** | **49.9% : 50.1%** | **★ 본 기획 착수 대상** |\n")

    # 표 1: 5대 결함 체계
    t1 = doc.add_table(rows=6, cols=5)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False
    
    widths1 = [1.0, 1.8, 1.4, 1.1, 1.2]
    headers1 = ["결함 코드", "공식 결함 명칭", "원천 데이터 현황", "클래스 비율", "개발 상태"]
    for i, h in enumerate(headers1):
        t1.cell(0, i).paragraphs[0].text = h
    style_table_header(t1.rows[0], widths1)
    
    rows_data1 = [
        ["d1", "척추의 중립 위반", "794 결함 / 798 정상", "49.9% : 50.1%", "완료 (d1 v3.1)"],
        ["d2", "발과 무릎의 방향 일치", "748 결함 / 844 정상", "47.0% : 53.0%", "완료 (d2 v2.0)"],
        ["d3", "발바닥 지면 고정 위반", "796 결함 / 796 정상", "50.0% : 50.0%", "완료 (d3 v1.0)"],
        ["d4", "무릎과 골반 동시 펴짐", "794 결함 / 798 정상", "49.9% : 50.1%", "차기 예정"],
        ["d5 (기획)", "바벨 궤적과 몸 밀착", "795 결함 / 797 정상", "49.9% : 50.1%", "★ 신규 기획"]
    ]
    for r_idx, r_data in enumerate(rows_data1):
        for c_idx, val in enumerate(r_data):
            cell = t1.cell(r_idx + 1, c_idx)
            cell.paragraphs[0].text = val
            if r_idx == 4:  # Highlight d5
                for r in cell.paragraphs[0].runs:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(30, 58, 138)
    style_table_cells(t1, widths1)
    doc.add_paragraph()

    # 2. 데이터셋 현황 및 d5 전용 캐시 구축 전략
    add_section_title(doc, "2. 데이터셋 파이프라인 및 d5 전용 캐시 구축 전략", level=1)
    
    p = doc.add_paragraph()
    p.add_run(
        "기존 데드리프트 테스트 캐시(385개 시퀀스, 77개 클립)는 스쿼트 3대 결함 모델(c1, c3, c4)의 제로샷 전이 평가를 위해 "
        "정상 클립과 척추(d1), 무릎(d2), 발바닥(d3) 결함 클립 위주로 샘플링되어 바벨 밀착 결함(d5)이 미포함 상태였습니다. "
        "이에 따라 AI-Hub 원천 데이터(1,592건)로부터 d5 전용 고해상도 64프레임 전처리 데이터셋을 신규 구축합니다."
    )
    
    add_bullet_point(doc, "샘플링 대상 규격", "피험자 1번~9번 전원 전수(9명)를 모두 포함하며, 각 피험자별 d5 결함 클립(3~4개) 및 정상 클립(3~4개)을 균형 샘플링하여 총 60~70개 클립(300~350개 멀티뷰 시퀀스) 또는 전수 확장 파이프라인 가동.")
    add_bullet_point(doc, "64프레임 3차 스플라인 전처리 표준", "원천 시퀀스를 프레임 누락 없이 64프레임으로 3차 스플라인(Cubic Spline) 보간하여 일관된 시간축 동역학 보존.")
    add_bullet_point(doc, "공식 5대 시점 전수 처리", "시점 A(우측 후방 45°/view5), 시점 B(우측 전방 45°/view2), 시점 C(정면 0°/view3), 시점 D(좌측 전방 45°/view4), 시점 E(좌측 후방 45°/view1)의 5개 전 시점 독립 데이터셋 구축.")

    log_md("## 2. 데이터셋 파이프라인 및 d5 전용 캐시 구축 전략\n")
    log_md("- **캐시 디렉터리**: `D:\\Data\\전처리_데이터\\MediaPipe_Full33_64F\\02_바벨_데드리프트_d5`")
    log_md("- **샘플링 원칙**: 9명 전 피험자(피험자 1~9), 공식 5대 시점(시점 A~E) 전수 반영")
    log_md("- **데이터셋 규모**: 총 70개 클립 × 5개 시점 = **350개 64F 시퀀스** (결함 175건 vs 정상 175건, 50:50 균형)")
    log_md("- **전처리 규격**: 순수 비전 MediaPipe Full 33 관절 추출 + 64프레임 3차 스플라인 보간\n")
    log_md("---\n")

    # 3. 28D 생체역학 피처 엔지니어링 설계
    add_section_title(doc, "3. d5 모델 생체역학 피처 엔지니어링 설계 (28D Barbell Proximity Dynamics)", level=1)
    
    p = doc.add_paragraph()
    p.add_run(
        "MediaPipe Pose 33개 관절 랜드마크 중 양 손목(좌: 15, 우: 16)의 중심점을 바벨의 실시간 3D 위치(Virtual Barbell Center: P_bar)로 정의합니다. "
        "정강이(Shank), 무릎(Knee), 대퇴골(Femur), 골반(Hip)의 공간 좌표와 바벨 좌표 간의 시공간 상호작용을 정량화한 28차원 피처셋을 설계합니다."
    )

    t2 = doc.add_table(rows=6, cols=4)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False
    
    widths2 = [1.5, 0.8, 1.8, 2.4]
    headers2 = ["피처 그룹", "차원", "핵심 피처 명칭", "생체역학적 기능 및 판별 원리"]
    for i, h in enumerate(headers2):
        t2.cell(0, i).paragraphs[0].text = h
    style_table_header(t2.rows[0], widths2)
    
    rows_data2 = [
        [
            "Group 1: 바벨-하지 수평 이격 거리\n(Horizontal Clearance)",
            "8D",
            "d5_bar_shank_dist_max\nd5_bar_knee_dist_pass\nd5_bar_thigh_dist_mean\nd5_bar_midfoot_init_offset",
            "리프팅 중 바벨과 정강이/무릎/허벅지 간의 전방 수평 이격 거리 정량화 (체형 정규화 포함). 이격이 클수록 모멘트 암 급증."
        ],
        [
            "Group 2: 바벨 수직 궤적 선형성\n(Trajectory Linearity)",
            "6D",
            "d5_bar_drift_horizontal_range\nd5_bar_drift_std\nd5_bar_path_curvature\nd5_bar_forward_overshoot",
            "이상적인 수직 직선 궤적(Vertical Line) 대비 바벨의 전후(X/Z) 표류 폭 및 궤적 곡률 측정. 바벨이 앞으로 둥글게 돌아나오는 궤적 감지."
        ],
        [
            "Group 3: 상체 모멘트 암 & 광배 결합\n(Torso & Lat Coupling)",
            "5D",
            "d5_shoulder_wrist_angle_max\nd5_shoulder_bar_moment_arm\nd5_lat_extension_deficit",
            "어깨 수직선 대비 팔의 전방 경사각 및 광배근 신전 결손 측정. 팔이 수직보다 앞으로 뻗어 있으면 광배근 결합 붕괴."
        ],
        [
            "Group 4: 구간별 위상 동역학\n(Phase-Specific Dynamics)",
            "5D",
            "d5_p1_shank_clearance_rate\nd5_p2_thigh_drag_contact\nd5_eccentric_knee_bypass\nd5_peak_drift_timing",
            "Phase 1(이륙~무릎) 정강이 이탈률, Phase 2(무릎~락아웃) 허벅지 밀착 지속도, 하강 시 무릎 둘러내림 여부, 최대 이탈 발생 시점 추적."
        ],
        [
            "Group 5: 멀티뷰 투영 & 왜곡 보정\n(Perspective Invariance)",
            "4D",
            "d5_sagittal_proj_drift\nd5_coronal_overlap_deficit\nd5_view_vis_confidence",
            "공식 5대 시점(A~E)별 카메라 각도에 따른 회전 보정. 사각(45°) 및 정면(0°) 투영 왜곡을 정규화하여 시점 편향 원천 차단."
        ]
    ]
    for r_idx, r_data in enumerate(rows_data2):
        for c_idx, val in enumerate(r_data):
            cell = t2.cell(r_idx + 1, c_idx)
            cell.paragraphs[0].text = val
    style_table_cells(t2, widths2)
    doc.add_paragraph()

    log_md("## 3. d5 모델 생체역학 피처 엔지니어링 설계 (28D Barbell Proximity Dynamics)\n")
    log_md("| 피처 그룹 | 차원 | 대표 핵심 피처 | 생체역학적 기능 및 판별 원리 |")
    log_md("| :--- | :---: | :--- | :--- |")
    log_md("| **Group 1: 바벨-하지 수평 이격 거리** | 8D (F01~F08) | `d5_bar_shank_dist_max`<br/>`d5_bar_knee_dist_pass`<br/>`d5_bar_thigh_dist_mean`<br/>`d5_bar_midfoot_init_offset` | 리프팅 중 바벨과 정강이/무릎/허벅지 간의 전방 수평 이격 거리 정량화 (체형 정규화 포함). 이격 클수록 모멘트암 폭증 |")
    log_md("| **Group 2: 바벨 수직 궤적 선형성** | 6D (F09~F14) | `d5_bar_drift_horizontal_range`<br/>`d5_bar_drift_std`<br/>`d5_bar_path_curvature`<br/>`d5_bar_forward_overshoot` | 수직 직선 궤적 대비 바벨의 전후(X/Z) 표류 폭 및 궤적 곡률 측정. 바벨이 앞으로 호를 그리며 돌아나오는 궤적 감지 |")
    log_md("| **Group 3: 상체 모멘트 암 & 광배 결합** | 5D (F15~F19) | `d5_shoulder_wrist_angle_max`<br/>`d5_shoulder_bar_moment_arm`<br/>`d5_lat_extension_deficit` | 어깨 수직선 대비 팔의 전방 경사각 및 광배근 신전 결손 측정. 팔이 수직보다 앞으로 뻗어 있으면 결함 |")
    log_md("| **Group 4: 구간별 위상 동역학** | 5D (F20~F24) | `d5_p1_shank_clearance_rate`<br/>`d5_p2_thigh_drag_contact`<br/>`d5_eccentric_knee_bypass`<br/>`d5_peak_drift_timing` | Phase 1(이륙~무릎) 정강이 이탈률, Phase 2(무릎~락아웃) 허벅지 밀착도, 하강 시 무릎 둘러내림 여부 추적 |")
    log_md("| **Group 5: 멀티뷰 투영 & 왜곡 보정** | 4D (F25~F28) | `d5_sagittal_proj_drift`<br/>`d5_coronal_overlap_deficit`<br/>`d5_view_vis_confidence` | 공식 5대 시점(A~E)별 카메라 각도에 따른 회전 보정. 사각 및 정면 투영 왜곡을 정규화하여 시점 편향 차단 |\n")
    log_md("---\n")

    # 4. 모델 아키텍처 및 검증 체계
    add_section_title(doc, "4. 모델링 및 학습·검증 체계", level=1)
    
    add_bullet_point(doc, "3-Way Tri-Ensemble 아키텍처", "스쿼트 c4 및 데드리프트 d3에서 입증된 고신뢰 앙상블 구조(LightGBM 50% + CatBoost 30% + Random Forest 20% 소프트 보팅)를 계승하여 단일 모델의 과적합을 방지하고 일반화 성능 극대화.")
    add_bullet_point(doc, "5-Fold 피험자 완전 격리 교차검증", "AI-Hub 피험자 1번~9번 전원(9명)을 StratifiedGroupKFold로 분할하여, 학습 데이터에 포함된 피험자가 검증 세트에 절대 유출되지 않도록 엄격한 일반화 능력 검증.")
    add_bullet_point(doc, "공식 5대 카메라 시점 전수 감사", "시점 A(우측 후방 45°), 시점 B(우측 전방 45°), 시점 C(정면 0°), 시점 D(좌측 전방 45°), 시점 E(좌측 후방 45°)의 5대 시점별 F1, Accuracy, ROC-AUC를 전수 산출하여 시점 간 성능 편차 5%p 이내 방어.")
    add_bullet_point(doc, "3-State 선택적 판정 Gate", "판정 경계 영역(tau_norm ~ tau_def)의 불확실한 시퀀스를 안전하게 RETAKE(재측정)로 보류하여, 확정 판정(NORMAL / DEFECT)에 대한 판별 신뢰도 F1 85%+ 달성.")

    log_md("## 4. 모델링 및 학습·검증 체계\n")
    log_md("1. **앙상블 아키텍처**: LightGBM(50%) + CatBoost(30%) + Random Forest(20%)의 Tri-Ensemble 소프트 보팅")
    log_md("2. **피험자 완전 격리 검증**: AI-Hub 피험자 1~9번 전원 대상 5-Fold StratifiedGroupKFold 교차검증")
    log_md("3. **공식 5대 시점 전수 감사 (Official Multi-View Taxonomy)**:")
    log_md("   - 시점 A: 우측 후방 45° (`view5`)")
    log_md("   - 시점 B: 우측 전방 45° (`view2`)")
    log_md("   - 시점 C: 정면 0° (`view3`)")
    log_md("   - 시점 D: 좌측 전방 45° (`view4`)")
    log_md("   - 시점 E: 좌측 후방 45° (`view1`)")
    log_md("4. **3-State 선택적 판정 Gate**: 판정 모호 구간을 RETAKE로 격리하여 안전성 및 F1 극대화\n")
    log_md("---\n")

    # 5. 단계별 구현 및 실험 로드맵
    add_section_title(doc, "5. 단계별 구현 및 실험 로드맵", level=1)
    
    t3 = doc.add_table(rows=5, cols=4)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3.autofit = False
    
    widths3 = [1.0, 2.2, 1.8, 1.5]
    headers3 = ["단계", "주요 작업 내용", "산출물 파일", "목표 일정 및 완료 조건"]
    for i, h in enumerate(headers3):
        t3.cell(0, i).paragraphs[0].text = h
    style_table_header(t3.rows[0], widths3)
    
    rows_data3 = [
        [
            "Phase 1",
            "d5 전용 캐시 구축 파이프라인 가동\n(피험자 1~9번, 5대 전 시점, 정상/결함 균형)",
            "build_d5_cache.py",
            "350개 시퀀스 npz 생성 및 무결성 검증 완료"
        ],
        [
            "Phase 2",
            "28D 바벨 밀착 피처 추출기 및\nd5 베이스라인 모델 클래스 구현",
            "d5_model_v1.py",
            "28D 피처 추출 파이프라인 단위 테스트 통과"
        ],
        [
            "Phase 3",
            "5-Fold 피험자 격리 학습 및\n공식 5대 시점 전수 감사 평가",
            "train_and_eval_d5_model.py",
            "OOF F1 및 5대 시점별 성능 지표 도출"
        ],
        [
            "Phase 4",
            "3-State 게이트 튜닝 및\n공식 Word(.docx) 결과 보고서 배포",
            "find_optimal_3state_d5.py\n공식 Word 보고서",
            "Selective F1 85%+ 달성 및 문서 배포"
        ]
    ]
    for r_idx, r_data in enumerate(rows_data3):
        for c_idx, val in enumerate(r_data):
            cell = t3.cell(r_idx + 1, c_idx)
            cell.paragraphs[0].text = val
    style_table_cells(t3, widths3)
    doc.add_paragraph()

    log_md("## 5. 단계별 구현 및 실험 로드맵\n")
    log_md("| 단계 | 주요 작업 내용 | 산출물 파일 | 목표 일정 및 완료 조건 |")
    log_md("| :---: | :--- | :--- | :--- |")
    log_md("| **Phase 1** | d5 전용 캐시 구축 파이프라인 가동 (피험자 1~9, 5대 시점 균형) | `build_d5_cache.py` | 350개 64F 시퀀스 npz 생성 완료 |")
    log_md("| **Phase 2** | 28D 바벨 밀착 피처 추출기 및 d5 베이스라인 모델 구현 | `d5_model_v1.py` | 28D 피처 추출 단위 테스트 무결성 검증 |")
    log_md("| **Phase 3** | 5-Fold 피험자 격리 학습 및 공식 5대 시점 전수 감사 | `train_and_eval_d5_model.py` | OOF F1 및 시점별 메트릭 도출 |")
    log_md("| **Phase 4** | 3-State 선택적 게이트 튜닝 및 공식 보고서 배포 | `find_optimal_3state_d5.py`<br/>공식 Word 보고서 | Selective F1 85%+ 및 `.docx` 배포 완료 |\n")

    # 저장
    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc.save(OUT_DOCX_PATH)
    print(f"[+] DOCX 저장 완료: {OUT_DOCX_PATH}")

    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[+] Markdown 저장 완료: {OUT_MD_PATH}")

if __name__ == "__main__":
    build_docx_and_md()
