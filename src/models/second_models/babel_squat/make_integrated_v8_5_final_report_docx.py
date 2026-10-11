# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: make_integrated_v8_5_final_report_docx.py
Purpose: 바벨 스쿼트 통합 진단 엔진 v8.5 (해결책 5: 운동사슬 39D 융합 & 품질게이트 계승) 구축 및 최종 평가 결과 보고서 생성
Target DOCX: D:\자주프 문서\5주차\vidio-health-trainer_바벨스쿼트_통합진단엔진_v8_5_해결책5_운동사슬39D_구축_및_최종평가_보고서.docx
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

OUT_DIR = r"D:\자주프 문서\5주차"
DOCX_NAME = "vidio-health-trainer_바벨스쿼트_통합진단엔진_v8_5_해결책5_운동사슬39D_구축_및_최종평가_보고서.docx"
OUT_PATH = os.path.join(OUT_DIR, DOCX_NAME)

def set_cell_background(cell, hex_color):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_report():
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = Document()

    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)

    style_normal = doc.styles['Normal']
    style_normal.font.name = '맑은 고딕'
    style_normal.font.size = Pt(10)
    style_normal.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("vidio-health-trainer\n바벨 스쿼트 마스터 통합 진단 엔진 v8.5 개발 보고서")
    r_title.bold = True
    r_title.font.size = Pt(20)
    r_title.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("외부 AI 자문 [해결책 5] 실증 적용: 39D 운동사슬(Kinetic Chain) 연계 및 4-Defect EMR 63.42% 대기록 달성")
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_meta = p_meta.add_run("작성일자: 2026. 09. 27 | 시스템 버전: v8.5 Master | 대상 표본: AI-Hub 골든셋 전수 2,605건")
    r_meta.font.size = Pt(9)
    r_meta.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

    doc.add_paragraph()

    # I. 핵심 성과 요약
    doc.add_heading("1. 핵심 성과 요약 (Executive Summary)", level=1)
    p_box = doc.add_paragraph()
    p_box.paragraph_format.left_indent = Inches(0.2)
    p_box.paragraph_format.right_indent = Inches(0.2)
    r_box = p_box.add_run(
        "■ 프로젝트 공식 명칭: vidio-health-trainer\n"
        "■ 공식 5대 카메라 시점 체계 (Official 5-View Taxonomy):\n"
        "  - 시점 A: 우측 후방 (45도) [Right Back 45° / view5]\n"
        "  - 시점 B: 우측 전방 (45도) [Right Front 45° / view2]\n"
        "  - 시점 C: 정면 (0도) [Front View 0° / view3]\n"
        "  - 시점 D: 좌측 전방 (45도) [Left Front 45° / view4]\n"
        "  - 시점 E: 좌측 후방 (45도) [Left Back 45° / view1]\n"
        "■ 외부 AI 자문 [해결책 5: 운동사슬(Kinetic Chain) 연계 분석 및 3위상 분할] 완결 성과:\n"
        "  - [대망의 4-Defect EMR 60% 벽 공식 돌파]: 100% 강제 판정 63.42% (1,652건 전수 완전 일치!), 선택적 판정 66.08% 달성!\n"
        "  - [4대 세부 결함(c1~c4) 전원 80%~94% 대도약]: c1 94.29%, c2 80.87%, c3 92.52%, c4 85.74% F1 동시 석권!\n"
        "  - [정상 진단 신기록 경신]: Clean F1 66.43% 달성 및 정상 재현율(Recall) 73.23% (v6.0 34.81% 대비 2배 이상 도약)\n"
        "  - [사선 45도 및 정면 시점의 경이적 적중률]:\n"
        "    * 시점 C (정면 0도)       : Clean F1 80.65% (정상 재현율 92.59% -> 27명 중 25명 완벽 적중!)\n"
        "    * 시점 B (우측 전방 45도) : Clean F1 72.73% (Recall 74.07%)\n"
        "    * 시점 D (좌측 전방 45도) : Clean F1 74.51% (Recall 70.37%)\n"
        "  - [v8.4 품질 게이트 선택적 판정 100% 온전 계승]: 뼈대 지터 표본 재촬영 권고 분리 로직 완전 유지"
    )
    r_box.font.name = "맑은 고딕"
    r_box.font.size = Pt(9.5)

    doc.add_paragraph()

    # II. 운동사슬 아키텍처
    doc.add_heading("2. 해결책 5 인체 운동사슬(Kinetic Chain) 아키텍처 및 9대 피처", level=1)
    doc.add_paragraph(
        "스쿼트는 인체 관절이 독립적으로 작동하는 것이 아니라, 발바닥(c4)의 접지 상태가 무릎(c3)의 내외반에 영향을 미치고, "
        "골반과 척추(c1)의 숙임 및 경추(c2)의 시선 보상으로 이어지는 연쇄적인 '운동사슬(Kinetic Chain)' 운동입니다.\n\n"
        "v8.5에서는 신체를 4대 핵심 부위(발, 무릎, 골반/척추, 목/머리)로 정의하고, 최저점(Nadir)을 기준으로 동작을 "
        "3단계(하강구간 Descent, 최저점구간 Nadir, 상승구간 Ascent)로 정밀 분할하여 상호 커플링 계수를 도출했습니다."
    )

    t_feat = doc.add_table(rows=1, cols=4)
    t_feat.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["피처 그룹", "피처명", "물리적 정의 및 상관 수식", "도메인 역할 및 효과"]
    for j, h in enumerate(headers):
        cell = t_feat.rows[0].cells[j]
        cell.paragraphs[0].text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "2B6CB0")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.paragraphs[0].runs[0].font.size = Pt(9)

    k_rows = [
        ("하체-중체 연쇄", "corr_foot_knee", "Corr(발뒤꿈치 들림량, 무릎 외반 비율)", "발목 흔들림과 무릎 모임의 연쇄 작용 정량화"),
        ("슬관절-체간 연쇄", "corr_knee_trunk", "Corr(무릎 외반 비율, 체간 전경각)", "무릎 모임과 상체 숙임의 보상 작용 분리"),
        ("상체-경추 연쇄", "corr_trunk_neck", "Corr(체간 전경각, 경추-안면 피치)", "상체 숙임 시 시선 정면 유지를 위한 경추 보상 인지"),
        ("원격 운동사슬", "corr_foot_trunk", "Corr(발뒤꿈치 들림량, 체간 전경각)", "발들림으로 인한 무게중심 전방 쏠림 및 상체 과숙임 인과관계 인지"),
        ("하강 위상 동역학", "descent_valgus_delta", "하강 시작점 대비 최저점 직전 무릎 변위", "내려가는 동안 무릎이 급격히 무너지는 현상 감지"),
        ("최저점 최대 부하", "nadir_trunk_tilt_max", "최저점 구간 최대 체간 숙임각 (rad)", "스쿼트 바닥(Hole)에서의 버트윙크 및 허리 꺾임 정밀 감지"),
        ("최저점 최대 외반", "nadir_valgus_min", "최저점 구간 최소 무릎 너비비", "가장 깊은 위치에서의 무릎 내측 붕괴(Valgus Collapse) 감지"),
        ("상승 위상 동역학", "ascent_heel_lift_max", "일어설 때 발뒤꿈치 최대 들림 폭", "상승 시 하체 반동으로 인한 앞꿈치 중심 쏠림 감지"),
        ("키네마틱 위상차", "phase_lag", "고관절 최저점과 슬관절 최저점 프레임차", "상체와 하체의 협응 타이밍 불일치 감지")
    ]
    for grp, fn, formula, role in k_rows:
        row = t_feat.add_row()
        for j, val in enumerate([grp, fn, formula, role]):
            cell = row.cells[j]
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_margins(cell, 80, 80, 100, 100)

    doc.add_paragraph()

    # III. 공식 토너먼트 종합 대조표
    doc.add_heading("3. 통합 진단 엔진 공식 토너먼트 전수 대조 평가 결과", level=1)
    t_tour = doc.add_table(rows=1, cols=7)
    t_tour.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_headers = ["엔진 버전", "Clean F1", "Precision", "Recall", "4-Defect EMR", "Coverage", "정상 적중 / 골든수"]
    for j, h in enumerate(t_headers):
        cell = t_tour.rows[0].cells[j]
        cell.paragraphs[0].text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "1A365D")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.paragraphs[0].runs[0].font.size = Pt(9)

    tour_rows = [
        ("v6.0 프로덕션 베이스라인", "34.43%", "34.06%", "34.81%", "15.90%", "100.0%", "47 / 135건"),
        ("v8.0 마스터 챔피언", "55.40%", "53.85%", "57.04%", "48.02%", "100.0%", "77 / 135건"),
        ("v8.2 해결책 2 (계층적온도보정)", "60.50%", "58.22%", "62.96%", "53.36%", "100.0%", "85 / 135건"),
        ("v8.3 해결책 3 (선택적판정/데드밴드)", "62.88%", "62.61%", "63.16%", "53.51%", "98.9%", "72 / 133건"),
        ("v8.4 해결책 4 (품질게이트 선택적판정)", "65.13%", "60.28%", "70.83%", "59.94%", "86.3%", "85 / 120건"),
        ("v8.5 해결책 5 (전수 100% 강제판정)", "64.38%", "59.87%", "69.63%", "63.42%", "100.0%", "94 / 135건"),
        ("v8.5 해결책 5 (운동사슬 선택적판정)", "66.43%", "60.78%", "73.23%", "66.08%", "87.6%", "93 / 127건")
    ]
    for r_vals in tour_rows:
        row = t_tour.add_row()
        for j, val in enumerate(r_vals):
            cell = row.cells[j]
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            if "v8.5" in r_vals[0]:
                cell.paragraphs[0].runs[0].bold = True
                set_cell_background(cell, "EDF2F7")
            set_cell_margins(cell, 80, 80, 100, 100)

    doc.add_paragraph()

    # IV. 4대 세부 결함 및 5대 시점별 심층 성능
    doc.add_heading("4. 4대 세부 결함(c1~c4) 및 공식 5대 시점별 최종 성능", level=1)
    
    doc.add_heading("가. 4대 세부 결함별 진단 성능 (전수 100% 강제 판정 N=2,605)", level=2)
    t_sub = doc.add_table(rows=1, cols=5)
    t_sub.alignment = WD_TABLE_ALIGNMENT.CENTER
    s_headers = ["세부 결함 항목", "F1-Score", "Precision", "Recall", "Accuracy"]
    for j, h in enumerate(s_headers):
        cell = t_sub.rows[0].cells[j]
        cell.paragraphs[0].text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "2C5282")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.paragraphs[0].runs[0].font.size = Pt(9)

    sub_rows = [
        ("c1 (척추의 중립 위반)", "94.29%", "93.77%", "94.82%", "94.47%"),
        ("c2 (고개 정면/시선 위반)", "80.87%", "82.68%", "79.12%", "80.31%"),
        ("c3 (발과 무릎 방향 일치)", "92.52%", "91.84%", "93.20%", "92.59%"),
        ("c4 (발바닥 지면 고정 위반)", "85.74%", "86.65%", "84.85%", "85.91%")
    ]
    for r_vals in sub_rows:
        row = t_sub.add_row()
        for j, val in enumerate(r_vals):
            cell = row.cells[j]
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_margins(cell, 80, 80, 100, 100)

    doc.add_paragraph()

    doc.add_heading("나. 공식 5대 시점별 Clean F1 및 정상 재현율 (v8.5 기준)", level=2)
    t_view = doc.add_table(rows=1, cols=5)
    t_view.alignment = WD_TABLE_ALIGNMENT.CENTER
    v_headers = ["공식 시점 명칭", "시점 코드", "Clean F1", "정상 재현율 (Recall)", "골든 정상 적중 건수"]
    for j, h in enumerate(v_headers):
        cell = t_view.rows[0].cells[j]
        cell.paragraphs[0].text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "2C5282")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.paragraphs[0].runs[0].font.size = Pt(9)

    v_data = [
        ("시점 A: 우측 후방 (45도)", "view5", "51.43%", "66.67%", "18 / 27건 적중"),
        ("시점 B: 우측 전방 (45도)", "view2", "72.73%", "74.07%", "20 / 27건 적중 🚀"),
        ("시점 C: 정면 (0도)", "view3", "80.65%", "92.59%", "25 / 27건 적중 (역대 최고!) 🚀"),
        ("시점 D: 좌측 전방 (45도)", "view4", "74.51%", "70.37%", "19 / 27건 적중 🚀"),
        ("시점 E: 좌측 후방 (45도)", "view1", "44.44%", "44.44%", "12 / 27건 적중")
    ]
    for r_vals in v_data:
        row = t_view.add_row()
        for j, val in enumerate(r_vals):
            cell = row.cells[j]
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_margins(cell, 80, 80, 100, 100)

    doc.add_paragraph()

    # V. 결론 및 향후 계획
    doc.add_heading("5. 결론 및 다음 단계 (해결책 6 예고)", level=1)
    doc.add_paragraph(
        "1. 해결책 5의 역사적 성과:\n"
        "   - 운동사슬(Kinetic Chain) 연계를 도입함으로써, 100% 전수 강제 판정에서 4-Defect 완전 일치율(EMR)이 63.42%로 폭등하여 "
        "     마침내 프로젝트 사상 처음으로 60% 벽을 여유 있게 돌파했습니다.\n"
        "   - c3(무릎) F1이 92.52%로 90%를 돌파하고, c2(고개) 역시 80.87%로 80% 벽을 넘어섰습니다.\n"
        "   - 특히 정면(시점 C)에서는 27명의 정상 표본 중 25명을 정확히 적중시켜 재현율 92.59%, F1 80.65%라는 경이적인 성과를 기록했습니다.\n\n"
        "2. 다음 단계 권고 (해결책 6: View-Conditioned Mixture-of-Experts):\n"
        "   - 정면(시점 C)과 전방 사선(시점 B, D)이 72~80%대로 완성된 반면, 여전히 후방(시점 A, E)은 안면 비가시성으로 인해 44~51%대에 머물러 있습니다.\n"
        "   - 해결책 6(MoE 동적 라우팅)을 도입하여, 시점 조건부 게이팅 네트워크를 통해 후방 시점 전용 전문가를 분리 라우팅함으로써 "
        "     5대 시점 전 구간 70%+ 평준화를 완성할 것을 권고합니다."
    )

    doc.save(OUT_PATH)
    print(f"  [Word 보고서 생성 완료] -> {OUT_PATH}")

if __name__ == '__main__':
    create_report()
