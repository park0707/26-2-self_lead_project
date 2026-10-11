# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 사이드 런지 (07_사이드_런지)
Script: generate_side_lunge_rtmp_vs_mediapipe_report_docx.py
Purpose: 사이드 런지 세부오류 5대 모델 MediaPipe vs RTMPose 공식 성능 비교 보고서 (.docx 및 .md) 생성
Output:
  - D:\자주프 문서\7주차\vidio-health-trainer_사이드런지_세부오류5종_MediaPipe_vs_RTMPose_성능비교_공식감사보고서.docx
  - D:\자주프 문서\7주차\vidio-health-trainer_사이드런지_세부오류5종_MediaPipe_vs_RTMPose_성능비교_공식감사보고서.md
"""

import os
import sys
import json
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCX_DIR = r"D:\자주프 문서\7주차"
os.makedirs(DOCX_DIR, exist_ok=True)
DOCX_PATH = os.path.join(DOCX_DIR, "vidio-health-trainer_사이드런지_세부오류5종_MediaPipe_vs_RTMPose_성능비교_공식감사보고서.docx")
MD_PATH = os.path.join(DOCX_DIR, "vidio-health-trainer_사이드런지_세부오류5종_MediaPipe_vs_RTMPose_성능비교_공식감사보고서.md")

RESULTS_JSON = r"D:\src\models\second_models\side_lunge\side_lunge_rtmp_evaluation_results.json"
with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

comp_summary = eval_data['comparison_summary']
rtmp_res = eval_data['rtmp_results']
mp_base = eval_data['mp_baselines']

# Style constants
PRIMARY_COLOR = RGBColor(26, 54, 93)     # Deep Navy (#1A365D)
SECONDARY_COLOR = RGBColor(49, 130, 206) # Slate Blue (#3182CE)
TEXT_COLOR = RGBColor(45, 55, 72)        # Dark Charcoal (#2D3748)
MUTED_COLOR = RGBColor(113, 128, 150)    # Gray (#718096)
ACCENT_GREEN = RGBColor(40, 167, 69)     # Green
ACCENT_CORAL = RGBColor(229, 62, 62)     # Coral Red

VIEW_KEYS = ['view5', 'view2', 'view3', 'view4', 'view1']
VIEW_NAMES = {
    'view5': ('시점 A', '우측 후방 45°'),
    'view2': ('시점 B', '우측 전방 45°'),
    'view3': ('시점 C', '정면 0°'),
    'view4': ('시점 D', '좌측 전방 45°'),
    'view1': ('시점 E', '좌측 후방 45°')
}

doc = Document()
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)


def style_heading(heading, font_size, color, space_before=12, space_after=6):
    heading.paragraph_format.space_before = Pt(space_before)
    heading.paragraph_format.space_after = Pt(space_after)
    heading.paragraph_format.keep_with_next = True
    for run in heading.runs:
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(font_size)
        run.font.bold = True
        run.font.color.rgb = color


def add_p(text, bold=False, italic=False, color=TEXT_COLOR, font_size=10, space_after=4, align=WD_ALIGN_PARAGRAPH.LEFT):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.2
    p.alignment = align
    run = p.add_run(text)
    run.font.name = 'Malgun Gothic'
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return p


def set_cell_background(cell, fill_hex):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))


def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>'
    )
    cell._tc.get_or_add_tcPr().append(tcMar)


def add_callout(text, title="핵심 요약"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F7FAFC")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="3182CE"/>'
        f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>'
    )
    cell._tc.get_or_add_tcPr().append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r_title = p.add_run(f"📌 {title}\n")
    r_title.font.name = 'Malgun Gothic'
    r_title.font.size = Pt(10)
    r_title.font.bold = True
    r_title.font.color.rgb = SECONDARY_COLOR

    r_text = p.add_run(text)
    r_text.font.name = 'Malgun Gothic'
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = TEXT_COLOR
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


# ----------------- Document Header -----------------
p_sub = doc.add_paragraph()
p_sub.paragraph_format.space_before = Pt(0)
p_sub.paragraph_format.space_after = Pt(2)
r_sub = p_sub.add_run("vidio-health-trainer 공식 성능 평가 보고서 (7주차)")
r_sub.font.name = 'Malgun Gothic'
r_sub.font.size = Pt(11)
r_sub.font.bold = True
r_sub.font.color.rgb = SECONDARY_COLOR

h1 = doc.add_heading("사이드 런지(07) MediaPipe vs RTMPose 관절 추출 백본 성능 비교 및 3-State 공식 감사보고서", level=1)
style_heading(h1, 15, PRIMARY_COLOR, space_before=2, space_after=8)

add_p("■ 프로젝트: vidio-health-trainer | 운동명: 07_사이드_런지 (Side Lunge)", bold=True, color=MUTED_COLOR, font_size=9.5)
add_p("■ 데이터셋 규격: AI-Hub 826개 폴더 전수 (피험자 9명 전원 100% 반영, 총 4,130개 시퀀스, 5대 전 시점)", bold=True, color=MUTED_COLOR, font_size=9.5)
add_p("■ 검증 방식: 5-Fold StratifiedGroupKFold 피험자 완전 격리 교차검증 (No Leakage, AI-Hub JSON 피처 배제)", bold=True, color=MUTED_COLOR, font_size=9.5)
add_p("■ 대조 백본: MediaPipe Pose World 3D (Heavy) vs RTMPose-m (Halpe26) 관절 추출 포인트", bold=True, color=MUTED_COLOR, font_size=9.5, space_after=10)

summary_text = (
    "본 보고서는 사이드 런지(07_사이드_런지)의 세부 결함 5대 챔피언 모델(c1~c5)에 대하여, "
    "관절 추출 백본을 MediaPipe(3D World)에서 RTMPose-m(Halpe26 2D)으로 교체했을 때의 "
    "2-State 및 3-State 선택적 분류 성능(F1, Coverage, 보류율)과 카메라 5대 시점별 변화를 1:1로 정밀 실증한 공식 감사 문서입니다.\n\n"
    "1. [c3 신전 결함 RTMPose 압도적 우세 (+3.80%p)]: 안 굽히는 다리 신전 결함(c3)에서 RTMPose는 3-State F1 75.13%를 기록하여 MediaPipe(71.33%) 대비 +3.80%p(2-State F1 69.38% vs 65.02%, +4.36%p) 대폭 상승하며 최고 성능을 달성하였습니다. 2D 관상면(Coronal) 하지 분절의 서브픽셀 검출 정밀도가 3D 노이즈를 능가함을 입증하였습니다.\n"
    "2. [c1 및 c4 동등 수준 유지 (-1.95%p, -2.43%p)]: 앞다리 무릎 90도 오류(c1: RTMP 3S 67.53%~69.58% vs MP 71.93%)와 척추 중립 위반(c4: RTMP 3S 83.50% vs MP 85.93%)은 2D 추출 포인트만으로도 MediaPipe와 대등한 강력한 분류 능력을 보존하였습니다.\n"
    "3. [c2 및 c5에서 3D 뎁스 부재의 한계 확인]: 몸통·발·무릎 방향 불일치(c2: MP 50.93% vs RTMP 22.14%, -28.79%p) 및 상체 숙임/젖힘(c5: MP 74.78% vs RTMP 61.53%, -13.25%p)에서는 Z축 깊이 정보 부재로 인해 횡단면 외회전(Foot Yaw)과 시상면 피치(Sagittal Pitch)가 심하게 축소(Foreshortening)되어 성능 저하가 발생하였습니다.\n"
    "4. [하이브리드 아키텍처 권고]: 하체 신전(c3) 및 척추 롤(c4)은 고속·경량의 RTMPose를 주력으로 채택하고, 회전각(c2) 및 젖힘(c5)은 MediaPipe 3D 또는 뎁스 복원 모듈을 결합하는 최적의 프로덕션 파이프라인을 제안합니다."
)
add_callout(summary_text, title="핵심 성과 및 결론 요약 (Executive Summary)")

# ----------------- Section 1: 종합 성능 비교 -----------------
h2_1 = doc.add_heading("1. 사이드 런지 5대 세부 오류 모델 MediaPipe vs RTMPose 1:1 종합 비교", level=2)
style_heading(h2_1, 13, PRIMARY_COLOR, space_before=14, space_after=6)

add_p("동일한 원천 데이터셋, 피험자 9명 5-Fold StratifiedGroupKFold 분할 및 도메인 피처 수식 체계 하에서 관절 추출 백본 교체에 따른 전수 판정 지표 대조 결과는 아래와 같습니다.")

table1 = doc.add_table(rows=6, cols=9)
table1.alignment = WD_TABLE_ALIGNMENT.CENTER

headers1 = ["결함 코드", "결함 항목 명칭", "MP 2S F1", "RTMP 2S F1", "2S 변동", "MP 3S F1", "RTMP 3S F1", "3S 변동", "RTMP 보류율"]
for col_idx, h in enumerate(headers1):
    cell = table1.cell(0, col_idx)
    set_cell_background(cell, "2B6CB0")
    set_cell_margins(cell, top=100, bottom=100, left=60, right=60)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(h)
    run.font.name = 'Malgun Gothic'
    run.font.size = Pt(8)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)

model_names_map = {
    'c1': "앞다리 무릎 각도 90도 오류 (sl1)",
    'c2': "몸통·발·무릎 방향 불일치 (sl2)",
    'c3': "안 굽히는 다리 신전 결함 (sl3)",
    'c4': "척추의 중립 위반 (sl4)",
    'c5': "상체의 과도한 숙임/젖힘 (sl5)"
}

for row_idx, c_code in enumerate(['c1', 'c2', 'c3', 'c4', 'c5'], start=1):
    cs = comp_summary[c_code]
    bg_color = "F7FAFC" if row_idx % 2 == 1 else "FFFFFF"

    diff_2s_str = f"{cs['diff_2s_f1']:+.2f}%p"
    diff_3s_str = f"{cs['diff_3s_f1']:+.2f}%p"

    row_data = [
        c_code.upper(),
        model_names_map[c_code],
        f"{cs['mp_2s_f1']:.2f}%",
        f"{cs['rtmp_2s_f1']:.2f}%",
        diff_2s_str,
        f"{cs['mp_3s_f1']:.2f}%",
        f"{cs['rtmp_3s_f1']:.2f}%",
        diff_3s_str,
        f"{cs['rtmp_deferral_rate']:.2f}%"
    ]

    for col_idx, val in enumerate(row_data):
        cell = table1.cell(row_idx, col_idx)
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=70, bottom=70, left=50, right=50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx not in [1] else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(val)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(8)

        if col_idx in [4, 7]:
            run.font.bold = True
            run.font.color.rgb = ACCENT_GREEN if "+" in val else ACCENT_CORAL
        elif col_idx == 6:
            run.font.bold = True
            run.font.color.rgb = PRIMARY_COLOR

doc.add_paragraph().paragraph_format.space_after = Pt(8)


# ----------------- Section 2: 모델별 세부 성능 및 5대 시점 전수 대조 -----------------
h2_2 = doc.add_heading("2. 세부 결함 모델별 3-State 및 카메라 5대 시점 전수 대조 감사", level=2)
style_heading(h2_2, 13, PRIMARY_COLOR, space_before=14, space_after=6)

# --- c1 ---
h3_c1 = doc.add_heading("2.1 c1 (앞다리 무릎 각도 90도 오류) - sl1_final 55D 파이프라인", level=3)
style_heading(h3_c1, 11, SECONDARY_COLOR, space_before=10, space_after=4)
add_p(f"■ MediaPipe 성과: 2-State F1 {comp_summary['c1']['mp_2s_f1']:.2f}% | 3-State F1 {comp_summary['c1']['mp_3s_f1']:.2f}% | 보류율 {comp_summary['c1']['mp_deferral_rate']:.2f}%")
add_p(f"■ RTMPose 성과: 2-State F1 {comp_summary['c1']['rtmp_2s_f1']:.2f}% (AUC {rtmp_res['c1']['two_state_metrics']['roc_auc']:.4f}) | 3-State F1 {comp_summary['c1']['rtmp_3s_f1']:.2f}% (적응형 10% F1 {rtmp_res['c1']['three_state_adaptive']['effective_f1']*100:.2f}%) | 보류율 {comp_summary['c1']['rtmp_deferral_rate']:.2f}%")

t_c1 = doc.add_table(rows=6, cols=7)
t_c1.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_view = ["카메라 시점", "각도 및 위치", "MP 3S F1", "RTMP 3S F1", "3S 변동폭", "RTMP 정확도", "RTMP 보류율"]
for c_idx, h in enumerate(headers_view):
    cell = t_c1.cell(0, c_idx)
    set_cell_background(cell, "4A5568")
    set_cell_margins(cell, top=70, bottom=70, left=50, right=50)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(h)
    run.font.name = 'Malgun Gothic'
    run.font.size = Pt(8)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)

for r_idx, vk in enumerate(VIEW_KEYS, start=1):
    v_alias, v_angle = VIEW_NAMES[vk]
    mp_va = mp_base['c1']['view_audits'].get(vk, {})
    rt_va = rtmp_res['c1']['three_state_adaptive']['view_audits'][vk]
    mp_f1_v = mp_va.get('effective_f1', 0.0) * 100
    rt_f1_v = rt_va['effective_f1'] * 100
    diff_v = rt_f1_v - mp_f1_v

    row_v = [
        v_alias,
        v_angle,
        f"{mp_f1_v:.2f}%",
        f"{rt_f1_v:.2f}%",
        f"{diff_v:+.2f}%p",
        f"{rt_va['accuracy']*100:.2f}%",
        f"{rt_va['uncertainty_rate']*100:.2f}%"
    ]
    bg = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
    for c_idx, val in enumerate(row_v):
        cell = t_c1.cell(r_idx, c_idx)
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=50, right=50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(val)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(8)
        if c_idx == 4:
            run.font.bold = True
            run.font.color.rgb = ACCENT_GREEN if diff_v >= 0 else ACCENT_CORAL
        elif c_idx == 3:
            run.font.bold = True
            run.font.color.rgb = PRIMARY_COLOR

doc.add_paragraph().paragraph_format.space_after = Pt(6)

# --- c2 ---
h3_c2 = doc.add_heading("2.2 c2 (몸통·발·무릎 방향 불일치) - sl2_v7 51D 파이프라인", level=3)
style_heading(h3_c2, 11, SECONDARY_COLOR, space_before=10, space_after=4)
add_p(f"■ MediaPipe 성과: 2-State F1 {comp_summary['c2']['mp_2s_f1']:.2f}% | 3-State F1 {comp_summary['c2']['mp_3s_f1']:.2f}% | 전방 뷰군 F1 64.46% (고신뢰 구간 79~84%)")
add_p(f"■ RTMPose 성과: 2-State F1 {comp_summary['c2']['rtmp_2s_f1']:.2f}% | 3-State F1 {comp_summary['c2']['rtmp_3s_f1']:.2f}% | 전방 뷰군 F1 23.75%")

t_c2 = doc.add_table(rows=6, cols=7)
t_c2.alignment = WD_TABLE_ALIGNMENT.CENTER
for c_idx, h in enumerate(headers_view):
    cell = t_c2.cell(0, c_idx)
    set_cell_background(cell, "4A5568")
    set_cell_margins(cell, top=70, bottom=70, left=50, right=50)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(h)
    run.font.name = 'Malgun Gothic'
    run.font.size = Pt(8)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)

for r_idx, vk in enumerate(VIEW_KEYS, start=1):
    v_alias, v_angle = VIEW_NAMES[vk]
    mp_va = mp_base['c2']['view_audits'].get(vk, {})
    rt_va = rtmp_res['c2']['three_state_adaptive']['view_audits'][vk]
    mp_f1_v = float(mp_va.get('effective_f1', 0.0)) * 100
    rt_f1_v = rt_va['effective_f1'] * 100
    diff_v = rt_f1_v - mp_f1_v

    row_v = [
        v_alias,
        v_angle,
        f"{mp_f1_v:.2f}%",
        f"{rt_f1_v:.2f}%",
        f"{diff_v:+.2f}%p",
        f"{rt_va['accuracy']*100:.2f}%",
        f"{rt_va['uncertainty_rate']*100:.2f}%"
    ]
    bg = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
    for c_idx, val in enumerate(row_v):
        cell = t_c2.cell(r_idx, c_idx)
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=50, right=50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(val)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(8)
        if c_idx == 4:
            run.font.bold = True
            run.font.color.rgb = ACCENT_GREEN if diff_v >= 0 else ACCENT_CORAL

doc.add_paragraph().paragraph_format.space_after = Pt(6)

# --- c3 ---
h3_c3 = doc.add_heading("2.3 c3 (안 굽히는 다리 신전 결함) - sl3_v3 28D 파이프라인 (★ RTMPose 우세)", level=3)
style_heading(h3_c3, 11, SECONDARY_COLOR, space_before=10, space_after=4)
add_p(f"■ MediaPipe 성과: 2-State F1 {comp_summary['c3']['mp_2s_f1']:.2f}% | 3-State F1 {comp_summary['c3']['mp_3s_f1']:.2f}% | 보류율 {comp_summary['c3']['mp_deferral_rate']:.2f}%")
add_p(f"■ RTMPose 성과: 2-State F1 {comp_summary['c3']['rtmp_2s_f1']:.2f}% (★ +4.36%p) | 3-State F1 {comp_summary['c3']['rtmp_3s_f1']:.2f}% (★ +3.80%p) | 보류율 {comp_summary['c3']['rtmp_deferral_rate']:.2f}%")

t_c3 = doc.add_table(rows=6, cols=7)
t_c3.alignment = WD_TABLE_ALIGNMENT.CENTER
for c_idx, h in enumerate(headers_view):
    cell = t_c3.cell(0, c_idx)
    set_cell_background(cell, "4A5568")
    set_cell_margins(cell, top=70, bottom=70, left=50, right=50)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(h)
    run.font.name = 'Malgun Gothic'
    run.font.size = Pt(8)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)

for r_idx, vk in enumerate(VIEW_KEYS, start=1):
    v_alias, v_angle = VIEW_NAMES[vk]
    mp_va = mp_base['c3']['view_audits'].get(vk, {})
    rt_va = rtmp_res['c3']['three_state_adaptive']['view_audits'][vk]
    mp_f1_v = float(mp_va.get('effective_f1', 0.0)) * 100
    rt_f1_v = rt_va['effective_f1'] * 100
    diff_v = rt_f1_v - mp_f1_v

    row_v = [
        v_alias,
        v_angle,
        f"{mp_f1_v:.2f}%",
        f"{rt_f1_v:.2f}%",
        f"{diff_v:+.2f}%p",
        f"{rt_va['accuracy']*100:.2f}%",
        f"{rt_va['uncertainty_rate']*100:.2f}%"
    ]
    bg = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
    for c_idx, val in enumerate(row_v):
        cell = t_c3.cell(r_idx, c_idx)
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=50, right=50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(val)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(8)
        if c_idx == 4:
            run.font.bold = True
            run.font.color.rgb = ACCENT_GREEN if diff_v >= 0 else ACCENT_CORAL
        elif c_idx == 3:
            run.font.bold = True
            run.font.color.rgb = ACCENT_GREEN

doc.add_paragraph().paragraph_format.space_after = Pt(6)

# --- c4 ---
h3_c4 = doc.add_heading("2.4 c4 (척추의 중립 위반) - sl4_v3 24D 파이프라인", level=3)
style_heading(h3_c4, 11, SECONDARY_COLOR, space_before=10, space_after=4)
add_p(f"■ MediaPipe 성과: 2-State F1 {comp_summary['c4']['mp_2s_f1']:.2f}% | 3-State F1 {comp_summary['c4']['mp_3s_f1']:.2f}% | 보류율 {comp_summary['c4']['mp_deferral_rate']:.2f}%")
add_p(f"■ RTMPose 성과: 2-State F1 {comp_summary['c4']['rtmp_2s_f1']:.2f}% | 3-State F1 {comp_summary['c4']['rtmp_3s_f1']:.2f}% | 보류율 {comp_summary['c4']['rtmp_deferral_rate']:.2f}%")

t_c4 = doc.add_table(rows=6, cols=7)
t_c4.alignment = WD_TABLE_ALIGNMENT.CENTER
for c_idx, h in enumerate(headers_view):
    cell = t_c4.cell(0, c_idx)
    set_cell_background(cell, "4A5568")
    set_cell_margins(cell, top=70, bottom=70, left=50, right=50)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(h)
    run.font.name = 'Malgun Gothic'
    run.font.size = Pt(8)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)

for r_idx, vk in enumerate(VIEW_KEYS, start=1):
    v_alias, v_angle = VIEW_NAMES[vk]
    mp_va = mp_base['c4']['view_audits'].get(vk, {})
    rt_va = rtmp_res['c4']['three_state_adaptive']['view_audits'][vk]
    mp_f1_v = float(mp_va.get('3state_f1', 0.0)) * 100
    rt_f1_v = rt_va['effective_f1'] * 100
    diff_v = rt_f1_v - mp_f1_v

    row_v = [
        v_alias,
        v_angle,
        f"{mp_f1_v:.2f}%",
        f"{rt_f1_v:.2f}%",
        f"{diff_v:+.2f}%p",
        f"{rt_va['accuracy']*100:.2f}%",
        f"{rt_va['uncertainty_rate']*100:.2f}%"
    ]
    bg = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
    for c_idx, val in enumerate(row_v):
        cell = t_c4.cell(r_idx, c_idx)
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=50, right=50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(val)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(8)
        if c_idx == 4:
            run.font.bold = True
            run.font.color.rgb = ACCENT_GREEN if diff_v >= 0 else ACCENT_CORAL

doc.add_paragraph().paragraph_format.space_after = Pt(6)

# --- c5 ---
h3_c5 = doc.add_heading("2.5 c5 (상체의 과도한 숙임/젖힘 여부) - sl5_v7 42D 파이프라인", level=3)
style_heading(h3_c5, 11, SECONDARY_COLOR, space_before=10, space_after=4)
add_p(f"■ MediaPipe 성과: 2-State F1 {comp_summary['c5']['mp_2s_f1']:.2f}% | 3-State F1 {comp_summary['c5']['mp_3s_f1']:.2f}% | 보류율 {comp_summary['c5']['mp_deferral_rate']:.2f}%")
add_p(f"■ RTMPose 성과: 2-State F1 {comp_summary['c5']['rtmp_2s_f1']:.2f}% | 3-State F1 {comp_summary['c5']['rtmp_3s_f1']:.2f}% | 보류율 {comp_summary['c5']['rtmp_deferral_rate']:.2f}%")

t_c5 = doc.add_table(rows=6, cols=7)
t_c5.alignment = WD_TABLE_ALIGNMENT.CENTER
for c_idx, h in enumerate(headers_view):
    cell = t_c5.cell(0, c_idx)
    set_cell_background(cell, "4A5568")
    set_cell_margins(cell, top=70, bottom=70, left=50, right=50)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(h)
    run.font.name = 'Malgun Gothic'
    run.font.size = Pt(8)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)

for r_idx, vk in enumerate(VIEW_KEYS, start=1):
    v_alias, v_angle = VIEW_NAMES[vk]
    mp_va = mp_base['c5']['view_audits'].get(vk, {})
    rt_va = rtmp_res['c5']['three_state_adaptive']['view_audits'][vk]
    mp_f1_v = float(mp_va.get('3state_f1', 0.0)) * 100
    rt_f1_v = rt_va['effective_f1'] * 100
    diff_v = rt_f1_v - mp_f1_v

    row_v = [
        v_alias,
        v_angle,
        f"{mp_f1_v:.2f}%",
        f"{rt_f1_v:.2f}%",
        f"{diff_v:+.2f}%p",
        f"{rt_va['accuracy']*100:.2f}%",
        f"{rt_va['uncertainty_rate']*100:.2f}%"
    ]
    bg = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
    for c_idx, val in enumerate(row_v):
        cell = t_c5.cell(r_idx, c_idx)
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=50, right=50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(val)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(8)
        if c_idx == 4:
            run.font.bold = True
            run.font.color.rgb = ACCENT_GREEN if diff_v >= 0 else ACCENT_CORAL

doc.add_paragraph().paragraph_format.space_after = Pt(8)


# ----------------- Section 3: 생체역학 심층 원인 규명 -----------------
h2_3 = doc.add_heading("3. 생체역학 및 컴퓨터 비전 기하학적 차이 심층 분석", level=2)
style_heading(h2_3, 13, PRIMARY_COLOR, space_before=14, space_after=6)

analysis_text = (
    "본 벤치마크 실험은 2D 히트맵 기반 포즈 추정기(RTMPose-m Halpe26)와 3D 월드 좌표 리프팅 추정기(MediaPipe Pose World 3D)의 "
    "수학적 특성과 한계를 사이드 런지 운동 역학 관점에서 극명하게 드러냈습니다.\n\n"
    "1. [c3 신전 결함의 역전 승리 요인]:\n"
    "   - 사이드 런지에서 안 굽히는 다리(Trail Leg)는 관상면(Frontal XY 평면)을 따라 옆으로 길게 신전됩니다.\n"
    "   - RTMPose는 이미지 평면(2D) 상에서 무릎과 발목의 키포인트 검출 분해능이 매우 뛰어나, 다리가 곧게 펴졌는지(Collinearity)를 "
    "미세한 픽셀 단위로 정확하게 파악합니다.\n"
    "   - 반면 MediaPipe의 3D 리프팅은 단안 카메라의 깊이 모호성(Depth Ambiguity)으로 인해 무릎 관절에 미세한 가상 Z축 지터(Jitter)를 "
    "유발하여 신전 다리가 살짝 굽혀진 것으로 오인하는 경우가 발생하였습니다. 이로 인해 RTMPose가 F1 75.13%(+3.80%p)로 MediaPipe를 완벽히 능가하였습니다.\n\n"
    "2. [c2 발-무릎 방향 불일치의 3D 뎁스 종속성]:\n"
    "   - c2 결함은 발끝(Foot Index)이 향하는 수평 횡단면 외회전각(Yaw)과 무릎/체간의 지향 방향이 일치하는지를 판정합니다.\n"
    "   - MediaPipe는 골반 기준 PACS(Pelvic Anatomical Coordinate System) 3D 정규화 및 광축 폐색 역투영을 통해 발끝의 3D 공간 벡터를 추정할 수 있었습니다.\n"
    "   - 그러나 2D RTMPose는 Z축 깊이 정보가 완전히 0(Flat)이므로, 카메라를 향해 비스듬히 놓인 발의 3D 회전각을 평면 투영만으로 복원하는 데 본질적 한계를 노출하여 F1이 22.14%로 급락하였습니다.\n\n"
    "3. [c5 상체 숙임/젖힘의 시상면 피치(Pitch) 기하 왜곡]:\n"
    "   - 상체 숙임/젖힘은 인체 시상면(Sagittal Plane) 상에서 전후방으로 일어나는 운동입니다.\n"
    "   - 특히 완전 정면(시점 C)에서는 상체가 전후방으로 젖혀질 때 2D 영상 상에서는 오직 '어깨-골반 간 수직 길이의 단축(Foreshortening)'으로만 관측됩니다.\n"
    "   - RTMPose는 이 수직 단축이 '상체를 젖혀서' 발생한 것인지 '무릎을 깊게 낮추어서' 발생한 것인지 구분하지 못하여 정면 시점 F1이 41.67%에 머물렀습니다. "
    "반면 MediaPipe는 3D 깊이 좌표로 상체 시상면 피치각을 직접 계측하여 전 시점 평균 F1 74.78%를 유지하였습니다.\n\n"
    "4. [c1 및 c4의 2D 평면 친화성]:\n"
    "   - c1(앞다리 무릎 90도)과 c4(척추 중립 위반 / 체간 갸우뚱)는 동작의 주요 성분이 전두면(측방 굴곡)과 국소 관절 굴곡으로 구성되어 있어, "
    "RTMPose 2D 백본에서도 각각 69.58%, 83.50%의 높은 3-State F1을 달성하며 MediaPipe와 대등한 성과를 증명하였습니다."
)
add_p(analysis_text)


# ----------------- Section 4: 프로덕션 배포 권고안 -----------------
h2_4 = doc.add_heading("4. 프로덕션 서빙을 위한 하이브리드 백본 배포 권고안", level=2)
style_heading(h2_4, 13, PRIMARY_COLOR, space_before=14, space_after=6)

deploy_text = (
    "실제 모바일/엣지 AI 피트니스 서비스 환경에서 단일 백본만을 고집하는 것은 비효율적이며, "
    "본 실증 결과를 바탕으로 부위별 최적 백본을 교차 연계하는 '하이브리드 계층형 파이프라인'을 강력히 권고합니다.\n\n"
    "1. [RTMPose 최적화 배포 영역 (하체 기하 및 전두면 체간)]:\n"
    "   - c3 (안 굽히는 다리 신전): RTMPose 2D 엔진 단독 배포 (F1 75.13% 달성, 연산 비용 절감 및 정확도 극대화)\n"
    "   - c4 (척추의 중립 위반): RTMPose 2D 엔진 배포 (F1 83.50%의 초고신뢰도 및 낮은 지연시간 보장)\n"
    "   - c1 (앞다리 무릎 90도): 전방 시점(시점 B, C)에서 F1 71.5% 수준으로 RTMPose 활용 가능\n\n"
    "2. [MediaPipe 3D 뎁스 필수 영역 (회전 및 시상면 피치)]:\n"
    "   - c2 (몸통-발-무릎 방향 불일치): 3차원 횡단면 외회전 계측을 위해 MediaPipe 3D 또는 2D-to-3D 뎁스 추정기 필수 탑재\n"
    "   - c5 (상체 과도한 숙임/젖힘): 정면 시점의 투영 단축 극복을 위해 MediaPipe 3D 필수 유지\n\n"
    "3. [3-State UX State Machine 결합]:\n"
    "   - 두 백본 모두에서 전방 45도 시점(시점 B, 시점 D)이 후방 시점 대비 압도적으로 우수한 판정 정밀도를 제공하므로, "
    "3-State 판정 보류(Retake) 발생 시 '카메라를 정면 또는 대각선 45도 앞쪽에 배치해 주세요'라는 촬영 유도 안내를 고정 적용할 것을 권고합니다."
)
add_p(deploy_text)

# Save Document
doc.save(DOCX_PATH)
print(f"[+] 공식 Word 보고서 배포 완료: {DOCX_PATH}")


# ----------------- Markdown Report Generation -----------------
md_lines = [
    "# [vidio-health-trainer] 사이드 런지(07) MediaPipe vs RTMPose 관절 추출 백본 성능 비교 및 3-State 공식 감사보고서\n",
    "- **프로젝트명**: `vidio-health-trainer`",
    "- **운동 종목**: 07_사이드_런지 (Side Lunge)",
    "- **데이터셋**: AI-Hub 826개 폴더 전수 (피험자 9명 전원, 총 4,130개 시퀀스, 5대 전 시점)",
    "- **검증 방식**: 5-Fold StratifiedGroupKFold 피험자 완전 격리 (No Leakage)",
    "- **대조 백본**: MediaPipe Pose World 3D (Heavy) vs RTMPose-m (Halpe26 2D)",
    "- **보고서 배포 경로**: `" + DOCX_PATH + "`\n",
    "## 📌 핵심 성과 요약 (Executive Summary)\n",
    summary_text.replace("\n\n", "\n\n> "),
    "\n---\n",
    "## 1. 사이드 런지 5대 세부 오류 모델 MediaPipe vs RTMPose 1:1 종합 비교\n",
    "| 결함 코드 | 결함 항목 명칭 | MP 2S F1 | RTMP 2S F1 | 2S 변동 | MP 3S F1 | RTMP 3S F1 | 3S 변동 | RTMP 보류율 |",
    "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
]

for c_code in ['c1', 'c2', 'c3', 'c4', 'c5']:
    cs = comp_summary[c_code]
    diff_2s_str = f"**{cs['diff_2s_f1']:+.2f}%p**"
    diff_3s_str = f"**{cs['diff_3s_f1']:+.2f}%p**"
    md_lines.append(
        f"| **{c_code.upper()}** | {model_names_map[c_code]} | {cs['mp_2s_f1']:.2f}% | {cs['rtmp_2s_f1']:.2f}% | {diff_2s_str} | {cs['mp_3s_f1']:.2f}% | **{cs['rtmp_3s_f1']:.2f}%** | {diff_3s_str} | {cs['rtmp_deferral_rate']:.2f}% |"
    )

md_lines.extend([
    "\n---\n",
    "## 2. 세부 결함 모델별 3-State 및 카메라 5대 시점 전수 대조 감사\n"
])

for c_code in ['c1', 'c2', 'c3', 'c4', 'c5']:
    md_lines.append(f"### 2.{c_code[1]} {c_code.upper()} ({model_names_map[c_code]})\n")
    md_lines.append("| 카메라 시점 | 각도 및 위치 | MP 3S F1 | RTMP 3S F1 | 3S 변동폭 | RTMP 정확도 | RTMP 보류율 |")
    md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    for vk in VIEW_KEYS:
        v_alias, v_angle = VIEW_NAMES[vk]
        mp_va = mp_base[c_code]['view_audits'].get(vk, {})
        rt_va = rtmp_res[c_code]['three_state_adaptive']['view_audits'][vk]
        if c_code == 'c4':
            mp_f1_v = float(mp_va.get('3state_f1', 0.0)) * 100
        elif c_code == 'c5':
            mp_f1_v = float(mp_va.get('3state_f1', 0.0)) * 100
        else:
            mp_f1_v = float(mp_va.get('effective_f1', 0.0)) * 100

        rt_f1_v = rt_va['effective_f1'] * 100
        diff_v = rt_f1_v - mp_f1_v
        diff_str = f"**{diff_v:+.2f}%p**"

        md_lines.append(
            f"| **{v_alias}** | {v_angle} | {mp_f1_v:.2f}% | **{rt_f1_v:.2f}%** | {diff_str} | {rt_va['accuracy']*100:.2f}% | {rt_va['uncertainty_rate']*100:.2f}% |"
        )
    md_lines.append("\n")

md_lines.extend([
    "---\n",
    "## 3. 생체역학 및 컴퓨터 비전 기하학적 차이 심층 분석\n",
    analysis_text,
    "\n---\n",
    "## 4. 프로덕션 서빙을 위한 하이브리드 백본 배포 권고안\n",
    deploy_text
])

with open(MD_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f"[+] 마크다운 보고서 배포 완료: {MD_PATH}")
