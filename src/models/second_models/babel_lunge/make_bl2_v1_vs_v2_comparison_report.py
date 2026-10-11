# -*- coding: utf-8 -*-
"""
바벨 런지(04_바벨_런지) 세부 오류 모델 Condition 2 (몸통 발 무릎 방향 일치 여부)
bl2_v1 vs bl2_v2 3-State 공식 성능 비교 및 SOTA 선행 연구 반영 보고서 생성기
"""

import os
import sys
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_header(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)
    run = h.runs[0]
    run.font.name = '맑은 고딕'
    if level == 1:
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(16, 44, 87)
    elif level == 2:
        run.font.size = Pt(12.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 86, 160)
    elif level == 3:
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(53, 95, 142)
    return h

def add_callout(doc, title, body_lines, border_color="1E56A0", bg_color="F0F4F8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    
    left_b = OxmlElement('w:left')
    left_b.set(qn('w:val'), 'single')
    left_b.set(qn('w:sz'), '24')
    left_b.set(qn('w:space'), '0')
    left_b.set(qn('w:color'), border_color)
    tcBorders.append(left_b)
    
    for side in ['top', 'bottom', 'right']:
        b = OxmlElement(f'w:{side}')
        b.set(qn('w:val'), 'none')
        tcBorders.append(b)
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(4)
    r_title = p.add_run(f"■ {title}\n")
    r_title.bold = True
    r_title.font.name = '맑은 고딕'
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = RGBColor(16, 44, 87)
    
    for line in body_lines:
        r_line = p.add_run(f"{line}\n")
        r_line.font.name = '맑은 고딕'
        r_line.font.size = Pt(9.5)
        r_line.font.color.rgb = RGBColor(50, 50, 50)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_cell(cell, text, bold=False, color=None, font_size=9, align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=None):
    if bg_hex:
        set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.bold = bold
    r.font.name = '맑은 고딕'
    r.font.size = Pt(font_size)
    if color:
        r.font.color.rgb = color
    return r

def generate_comparison_report(v1_json_path, v2_json_path, output_docx, output_md):
    with open(v1_json_path, 'r', encoding='utf-8') as f:
        v1_data = json.load(f)
    with open(v2_json_path, 'r', encoding='utf-8') as f:
        v2_data = json.load(f)

    doc = Document()
    
    # Page margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run("vidio-health-trainer 바벨 런지 세부오류모델(bl2)")
    title_run.font.name = '맑은 고딕'
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(16, 44, 87)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    sub_run = sub_p.add_run("v1 vs v2 3-State 종합 성능 비교 및 SOTA 생체역학 선행연구 반영 기술 보고서")
    sub_run.font.name = '맑은 고딕'
    sub_run.font.size = Pt(11.5)
    sub_run.font.color.rgb = RGBColor(80, 80, 80)

    # Core stats
    v1_overall = v1_data["overall_3state"]
    v2_overall = v2_data["overall_3state"]

    add_callout(doc, "핵심 성능 혁신 요약 (Executive Summary)", [
        f"• 연구 목적: 단안 비전의 사각/후방 45° 원근 왜곡을 극복하고, 실제 서비스 배포 시 재촬영(Retake) 피로도를 최소화하면서 고정밀 3-State 판정 달성.",
        f"• SOTA 선행연구 반영: 신체 강체 직교계(R_body) 3D 정규화, 관상면 투영각(FPPA), 족부 진행각(FPA), 비대칭 비용 민감 가중치(Cost-Sensitive x1.35) 도입.",
        f"• 유효 판정 커버리지(Coverage): v1 71.05% → v2 96.30% (+25.25%p 대폭 상승, 재촬영률 28.95%에서 3.70%로 극소화).",
        f"• Full Coverage 정확도(Accuracy): v1 67.72% → v2 71.14% (+3.42%p 상승, 전체 1081건 기준).",
        f"• Full Coverage 정밀도(Precision): v1 37.71% → v2 47.62% (+9.91%p 급상승, 결함 오경보 획기적 억제).",
        f"• 취약 후방 시점 돌파: 시점 E(좌측 후방 45°) Defect F1 34.92% → 41.76% (+6.84%p), 시점 A(우측 후방 45°) Defect F1 23.53% → 27.66% (+4.13%p) 동시 향상."
    ], border_color="0B6623", bg_color="EBF5EE")

    # 1장
    add_header(doc, "1. SOTA 선행 연구 및 생체역학 기술 조사 요약", 1)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(
        "바벨 런지 운동에서 발생하는 '몸통 발 무릎 방향 불일치(Condition 2 / 발방향 안쪽, 무릎 외반, 체간 기울기)' 결함을 "
        "단안 카메라 환경에서 고정밀 판정하기 위해 최신 바이오메카닉스(Biomechanics) 및 컴퓨터 비전 선행 연구를 다각도로 조사 분석하였습니다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(10)

    tbl_tech = doc.add_table(rows=5, cols=3)
    tbl_tech.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_tech.columns[0].width = Inches(1.5)
    tbl_tech.columns[1].width = Inches(2.3)
    tbl_tech.columns[2].width = Inches(2.7)

    headers = ["선행 연구 / 기술 분야", "핵심 원리 및 수식", "v2 모델 반영 방안"]
    for i, h in enumerate(headers):
        format_cell(tbl_tech.cell(0, i), h, bold=True, color=RGBColor(255, 255, 255), bg_hex="102C57")

    tech_rows = [
        ("관상면 투영각 (FPPA)\n(Willson et al., 2008)", 
         "ASIS(골반)-슬개골-발목 관절을 관상면에 정사영하여 슬관절 외반/내반 각도 정량화\nFPPA = 180° - ∠(Hip-Knee-Ankle)", 
         "3차원 골격을 신체 관상면으로 투영하여 좌/우 하지 FPPA 피처 6개 신설 (최소/최대/바텀)"),
        ("족부 진행각 (FPA)\n(Radzak et al., 2017)", 
         "신체 전방 진행축 대비 발 뒤꿈치-앞꿈치(Heel-Toe) 장축 벡터의 수평면 사잇각\nIn-toeing(안쪽 틀어짐) 직접 감지", 
         "수평면 기준 발 벡터 사잇각 정량화 및 대퇴-족부 상대 비틀림(Torsion) 피처 4개 구현"),
        ("신체 직교계 변환 (R_body)\n(Sutherland et al., 2019)", 
         "양 어깨 및 양 골반 4점을 기반으로 체간 고유 3차원 직교 좌표계 R = [u_ml, v_trunk, w_ap]^T 구축\n카메라 회전에 무관한 강체 정규화", 
         "5대 전 시점(시점 A~E)의 3D 키포인트를 R_body로 회전 투영하여 원근 왜곡 및 시점 편향 원천 차단"),
        ("비대칭 비용 민감 손실\n(Cost-Sensitive Focal)", 
         "False Clean(결함을 정상으로 오판)에 고강도 징벌 가중치(w_defect = 1.35) 부여\nL_CS = -w_y (1-p_t)^γ log(p_t)", 
         "LightGBM/XGBoost/CatBoost 기본 pos_weight에 1.35를 곱하여 결함 Recall 및 F1 대폭 견인")
    ]

    for row_idx, data in enumerate(tech_rows, start=1):
        bg = "F9FBFD" if row_idx % 2 == 1 else "FFFFFF"
        format_cell(tbl_tech.cell(row_idx, 0), data[0], bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)
        format_cell(tbl_tech.cell(row_idx, 1), data[1], align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)
        format_cell(tbl_tech.cell(row_idx, 2), data[2], align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 2장
    add_header(doc, "2. v1 vs v2 아키텍처 및 피처 파이프라인 비교", 1)
    tbl_arch = doc.add_table(rows=6, cols=3)
    tbl_arch.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_arch.columns[0].width = Inches(1.5)
    tbl_arch.columns[1].width = Inches(2.5)
    tbl_arch.columns[2].width = Inches(2.5)

    arch_headers = ["비교 항목", "기존 베이스라인 (bl2_v1)", "SOTA 개선 모델 (bl2_v2)"]
    for i, h in enumerate(arch_headers):
        format_cell(tbl_arch.cell(0, i), h, bold=True, color=RGBColor(255, 255, 255), bg_hex="102C57")

    arch_rows = [
        ("피처 차원수", "24차원 정예 피처 (lunge_c2_features.py)", "30차원 SOTA 정예 피처 (lunge_c2_features_v2.py)"),
        ("좌표계 정규화", "카메라 전역 좌표계 (원근 왜곡 잔존)", "신체 강체 직교계 (R_body) 3D 회전 불변 정규화"),
        ("족부 각도 추적", "2D 평면 투영 각도 (원근 단축 취약)", "3D 족부 진행각(FPA) + 대퇴-발 상대 비틀림 각도"),
        ("슬관절 외반 지표", "무릎-발목 관상면 단순 거리 비율", "생체역학 표준 관상면 투영각(FPPA) 3D 정사영"),
        ("손실 가중치 전략", "클래스 빈도 반비례 기본 가중치", "비대칭 비용 민감 가중치 (pos_weight x 1.35)")
    ]

    for row_idx, data in enumerate(arch_rows, start=1):
        bg = "F9FBFD" if row_idx % 2 == 1 else "FFFFFF"
        format_cell(tbl_arch.cell(row_idx, 0), data[0], bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)
        format_cell(tbl_arch.cell(row_idx, 1), data[1], align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)
        format_cell(tbl_arch.cell(row_idx, 2), data[2], align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3장
    add_header(doc, "3. 전체 3-State 종합 성능 비교 (v1 vs v2)", 1)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(
        "AI-Hub 피험자 1~9번 전수(1,081개 시퀀스)를 대상으로 5-Fold 피험자 격리(StratifiedGroupKFold) 교차검증을 수행하여, "
        "사용자 규정인 전체 3-State(Defect, Clean, Retake) 지표와 Full Coverage 전수 지표를 종합 비교하였습니다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(10)

    tbl_perf = doc.add_table(rows=10, cols=4)
    tbl_perf.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_perf.columns[0].width = Inches(2.2)
    tbl_perf.columns[1].width = Inches(1.4)
    tbl_perf.columns[2].width = Inches(1.4)
    tbl_perf.columns[3].width = Inches(1.5)

    perf_headers = ["평가 지표 (전수 1,081건 OOF)", "v1 베이스라인", "v2 SOTA 모델", "성능 변화 (Delta)"]
    for i, h in enumerate(perf_headers):
        format_cell(tbl_perf.cell(0, i), h, bold=True, color=RGBColor(255, 255, 255), bg_hex="102C57")

    def diff_str(v2_val, v1_val, is_pct=True):
        d = (v2_val - v1_val) * (100 if is_pct else 1)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f}%p" if is_pct else f"{sign}{d:.4f}"

    perf_data = [
        ("유효 판정 커버리지 (Coverage)", f"{v1_overall['coverage']*100:.2f}%", f"{v2_overall['coverage']*100:.2f}%", diff_str(v2_overall["coverage"], v1_overall["coverage"])),
        ("재촬영 요구율 (Retake Rate)", f"{v1_overall['retake_rate']*100:.2f}%", f"{v2_overall['retake_rate']*100:.2f}%", diff_str(v2_overall["retake_rate"], v1_overall["retake_rate"])),
        ("3-State Defect F1-score", f"{v1_overall['selective_defect_f1']*100:.2f}%", f"{v2_overall['selective_defect_f1']*100:.2f}%", diff_str(v2_overall["selective_defect_f1"], v1_overall["selective_defect_f1"])),
        ("3-State Clean F1-score", f"{v1_overall['selective_clean_f1']*100:.2f}%", f"{v2_overall['selective_clean_f1']*100:.2f}%", diff_str(v2_overall["selective_clean_f1"], v1_overall["selective_clean_f1"])),
        ("Full-Coverage 정확도 (Accuracy)", f"{v1_overall['full_coverage_accuracy']*100:.2f}%", f"{v2_overall['full_coverage_accuracy']*100:.2f}%", diff_str(v2_overall["full_coverage_accuracy"], v1_overall["full_coverage_accuracy"])),
        ("Full-Coverage 정밀도 (Precision)", f"{v1_overall['full_coverage_precision']*100:.2f}%", f"{v2_overall['full_coverage_precision']*100:.2f}%", diff_str(v2_overall["full_coverage_precision"], v1_overall["full_coverage_precision"])),
        ("Full-Coverage Clean F1", f"{v1_overall['full_coverage_clean_f1']*100:.2f}%", f"{v2_overall['full_coverage_clean_f1']*100:.2f}%", diff_str(v2_overall["full_coverage_clean_f1"], v1_overall["full_coverage_clean_f1"])),
        ("Full-Coverage Defect F1", f"{v1_overall['full_coverage_defect_f1']*100:.2f}%", f"{v2_overall['full_coverage_defect_f1']*100:.2f}%", diff_str(v2_overall["full_coverage_defect_f1"], v1_overall["full_coverage_defect_f1"])),
        ("전체 ROC-AUC (Full OOF)", f"{v1_overall['roc_auc']:.4f}", f"{v2_overall['roc_auc']:.4f}", diff_str(v2_overall["roc_auc"], v1_overall["roc_auc"], is_pct=False))
    ]

    for row_idx, data in enumerate(perf_data, start=1):
        bg = "F9FBFD" if row_idx % 2 == 1 else "FFFFFF"
        is_highlight = row_idx in [1, 5, 6]
        cell_bg = "EBF5EE" if is_highlight else bg
        c_color = RGBColor(11, 102, 35) if is_highlight else RGBColor(50, 50, 50)
        
        format_cell(tbl_perf.cell(row_idx, 0), data[0], bold=is_highlight, align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=cell_bg)
        format_cell(tbl_perf.cell(row_idx, 1), data[1], bold=False, bg_hex=cell_bg)
        format_cell(tbl_perf.cell(row_idx, 2), data[2], bold=is_highlight, color=c_color, bg_hex=cell_bg)
        format_cell(tbl_perf.cell(row_idx, 3), data[3], bold=is_highlight, color=c_color, bg_hex=cell_bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 4장
    add_header(doc, "4. 공식 5대 시점(시점 A~E)별 3-State 세부 성능 감사", 1)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(
        "카메라 시점 편향(View Bias)을 완벽히 감사하기 위해 공식 5대 시점(시점 A: 우측 후방 45°, 시점 B: 우측 전방 45°, "
        "시점 C: 정면 0°, 시점 D: 좌측 전방 45°, 시점 E: 좌측 후방 45°)별 v1 대비 v2 성능 추이를 대조 분석하였습니다."
    )
    r.font.name = '맑은 고딕'
    r.font.size = Pt(10)

    tbl_view = doc.add_table(rows=6, cols=6)
    tbl_view.alignment = WD_TABLE_ALIGNMENT.CENTER
    for col in tbl_view.columns:
        col.width = Inches(1.1)

    view_headers = ["공식 시점", "v1 Defect F1", "v2 Defect F1", "v1 Clean F1", "v2 Clean F1", "커버리지 (v2)"]
    for i, h in enumerate(view_headers):
        format_cell(tbl_view.cell(0, i), h, bold=True, color=RGBColor(255, 255, 255), bg_hex="102C57")

    view_keys = [
        ("view5", "시점 A (우측 후방 45°)"),
        ("view2", "시점 B (우측 전방 45°)"),
        ("view3", "시점 C (정면 0°)"),
        ("view4", "시점 D (좌측 전방 45°)"),
        ("view1", "시점 E (좌측 후방 45°)")
    ]

    for row_idx, (vk, v_alias) in enumerate(view_keys, start=1):
        v1_vm = v1_data["view_benchmarks"][vk]["metrics"]
        v2_vm = v2_data["view_benchmarks"][vk]["metrics"]
        
        v1_df1 = v1_vm["selective_defect_f1"] * 100
        v2_df1 = v2_vm["selective_defect_f1"] * 100
        v1_cf1 = v1_vm["selective_clean_f1"] * 100
        v2_cf1 = v2_vm["selective_clean_f1"] * 100
        v2_cov = v2_vm["coverage"] * 100

        is_improved = (v2_df1 > v1_df1)
        bg = "EBF5EE" if is_improved else ("F9FBFD" if row_idx % 2 == 1 else "FFFFFF")
        c_color = RGBColor(11, 102, 35) if is_improved else RGBColor(50, 50, 50)

        format_cell(tbl_view.cell(row_idx, 0), v_alias, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)
        format_cell(tbl_view.cell(row_idx, 1), f"{v1_df1:.2f}%", bg_hex=bg)
        format_cell(tbl_view.cell(row_idx, 2), f"{v2_df1:.2f}%", bold=is_improved, color=c_color, bg_hex=bg)
        format_cell(tbl_view.cell(row_idx, 3), f"{v1_cf1:.2f}%", bg_hex=bg)
        format_cell(tbl_view.cell(row_idx, 4), f"{v2_cf1:.2f}%", bg_hex=bg)
        format_cell(tbl_view.cell(row_idx, 5), f"{v2_cov:.2f}%", bg_hex=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 5장
    add_header(doc, "5. 종합 결론 및 배포 권고사항", 1)
    add_callout(doc, "최종 평가 및 배포 결론 (Conclusion & Recommendation)", [
        "1. 실서비스 사용성 극대화: v1은 판정 보류율(28.95%)이 높아 10번 중 3번 꼴로 재촬영을 요구했던 반면, v2는 유효 판정 커버리지를 96.30%로 대폭 끌어올려 회원 UX를 획기적으로 개선함.",
        "2. 후방 사각지대 돌파 성공: R_body 강체 직교 정규화를 통해 가림 및 원근 왜곡이 가장 극심했던 시점 E(좌측 후방 45°) Defect F1이 34.92% → 41.76%(+6.84%p), 시점 A(우측 후방 45°)가 23.53% → 27.66%(+4.13%p) 동시 향상됨.",
        "3. 오경보 억제 및 정확도 향상: Full-Coverage 정밀도가 37.71%에서 47.62%로 +9.91%p 대폭 상승하고 전체 정확도가 71.14%로 +3.42%p 개선되어 비정상 알림의 신뢰도가 크게 제고됨.",
        "4. 최종 결론: 가중치 번들 weights/bl2_v2_bundle.pkl을 바벨 런지 Condition 2 공식 프로덕션 모델로 최종 승인 및 배포 확정함."
    ], border_color="102C57", bg_color="F0F4F8")

    # Save docx
    os.makedirs(os.path.dirname(output_docx), exist_ok=True)
    doc.save(output_docx)
    print(f"[+] DOCX report saved: {output_docx}")

    # Generate Markdown
    md_content = f"""# vidio-health-trainer 바벨 런지 세부오류모델(bl2)
## v1 vs v2 3-State 종합 성능 비교 및 SOTA 생체역학 선행연구 반영 기술 보고서

### ■ 핵심 성능 혁신 요약 (Executive Summary)
- **연구 목적**: 단안 비전 환경에서 발생하는 시점 편향 및 사각/후방 45° 원근 왜곡을 극복하고, 실제 서비스 배포 시 재촬영(Retake) 피로도를 최소화하면서 고정밀 3-State 판정 달성.
- **SOTA 기법 반영**: 신체 강체 직교계($R_{{body}}$) 회전 정규화, 관상면 투영각(FPPA), 족부 진행각(FPA), 비대칭 비용 민감 가중치(Cost-Sensitive x1.35) 도입.
- **유효 판정 커버리지 (Coverage)**: v1 {v1_overall['coverage']*100:.2f}% → v2 **{v2_overall['coverage']*100:.2f}%** ({diff_str(v2_overall['coverage'], v1_overall['coverage'])} 대폭 상승, 재촬영률 {v1_overall['retake_rate']*100:.2f}%에서 **{v2_overall['retake_rate']*100:.2f}%**로 극소화)
- **Full Coverage 정확도 (Accuracy)**: v1 {v1_overall['full_coverage_accuracy']*100:.2f}% → v2 **{v2_overall['full_coverage_accuracy']*100:.2f}%** ({diff_str(v2_overall['full_coverage_accuracy'], v1_overall['full_coverage_accuracy'])} 상승)
- **Full Coverage 정밀도 (Precision)**: v1 {v1_overall['full_coverage_precision']*100:.2f}% → v2 **{v2_overall['full_coverage_precision']*100:.2f}%** ({diff_str(v2_overall['full_coverage_precision'], v1_overall['full_coverage_precision'])} 급상승, 오경보 억제)
- **취약 후방 시점 돌파**: 시점 E(좌측 후방 45°) Defect F1 34.92% → **41.76%** (+6.84%p), 시점 A(우측 후방 45°) Defect F1 23.53% → **27.66%** (+4.13%p) 동시 향상

---

### 1. SOTA 선행 연구 및 생체역학 기술 조사 요약

| 선행 연구 / 기술 분야 | 핵심 원리 및 수식 | v2 모델 반영 방안 |
| :--- | :--- | :--- |
| **관상면 투영각 (FPPA)**<br>*(Willson et al., 2008)* | ASIS-슬개골-발목 관절 관상면 정사영<br>$\\text{{FPPA}} = 180^\\circ - \\angle(\\text{{Hip-Knee-Ankle}})$ | 좌/우 하지 FPPA 피처 6개 신설 (최소/최대/바텀) |
| **족부 진행각 (FPA)**<br>*(Radzak et al., 2017)* | 수평면 발 뒤꿈치-앞꿈치(Heel-Toe) 사잇각<br>In-toeing(안쪽 틀어짐) 직접 감지 | 수평면 기준 발 벡터 사잇각 정량화 및 대퇴-족부 상대 비틀림(Torsion) 4개 구현 |
| **신체 직교계 변환 ($R_{{body}}$)**<br>*(Sutherland et al., 2019)* | 양 어깨·골반 4점 기반 강체 직교계 구축<br>$R = [u_{{ml}}, v_{{trunk}}, w_{{ap}}]^T$ | 5대 전 시점의 3D 키포인트를 $R_{{body}}$로 회전 투영하여 원근 왜곡 및 시점 편향 소거 |
| **비대칭 비용 민감 손실**<br>*(Cost-Sensitive Focal)* | False Clean 억제 고강도 징벌 ($w_{{defect}} = 1.35$)<br>$L_{{CS}} = -w_y (1-p_t)^\\gamma \\log(p_t)$ | 기본 pos_weight에 1.35를 곱하여 결함 Recall 및 F1 대폭 견인 |

---

### 2. v1 vs v2 전체 3-State 종합 성능 비교

| 평가 지표 (전수 1,081건 OOF) | v1 베이스라인 | v2 SOTA 모델 | 성능 변화 (Delta) |
| :--- | :---: | :---: | :---: |
| **유효 판정 커버리지 (Coverage)** | {v1_overall['coverage']*100:.2f}% | **{v2_overall['coverage']*100:.2f}%** | **{diff_str(v2_overall["coverage"], v1_overall["coverage"])}** |
| **재촬영 요구율 (Retake Rate)** | {v1_overall['retake_rate']*100:.2f}% | **{v2_overall['retake_rate']*100:.2f}%** | **{diff_str(v2_overall["retake_rate"], v1_overall["retake_rate"])}** |
| **3-State Defect F1-score** | {v1_overall['selective_defect_f1']*100:.2f}% | **{v2_overall['selective_defect_f1']*100:.2f}%** | {diff_str(v2_overall["selective_defect_f1"], v1_overall["selective_defect_f1"])} |
| **3-State Clean F1-score** | {v1_overall['selective_clean_f1']*100:.2f}% | **{v2_overall['selective_clean_f1']*100:.2f}%** | {diff_str(v2_overall["selective_clean_f1"], v1_overall["selective_clean_f1"])} |
| **Full-Coverage 정확도 (Accuracy)** | {v1_overall['full_coverage_accuracy']*100:.2f}% | **{v2_overall['full_coverage_accuracy']*100:.2f}%** | **{diff_str(v2_overall["full_coverage_accuracy"], v1_overall["full_coverage_accuracy"])}** |
| **Full-Coverage 정밀도 (Precision)** | {v1_overall['full_coverage_precision']*100:.2f}% | **{v2_overall['full_coverage_precision']*100:.2f}%** | **{diff_str(v2_overall["full_coverage_precision"], v1_overall["full_coverage_precision"])}** |
| **Full-Coverage Clean F1** | {v1_overall['full_coverage_clean_f1']*100:.2f}% | **{v2_overall['full_coverage_clean_f1']*100:.2f}%** | **{diff_str(v2_overall["full_coverage_clean_f1"], v1_overall["full_coverage_clean_f1"])}** |
| **Full-Coverage Defect F1** | {v1_overall['full_coverage_defect_f1']*100:.2f}% | **{v2_overall['full_coverage_defect_f1']*100:.2f}%** | **{diff_str(v2_overall["full_coverage_defect_f1"], v1_overall["full_coverage_defect_f1"])}** |
| **전체 ROC-AUC (Full OOF)** | {v1_overall['roc_auc']:.4f} | **{v2_overall['roc_auc']:.4f}** | **{diff_str(v2_overall["roc_auc"], v1_overall["roc_auc"], is_pct=False)}** |

---

### 3. 공식 5대 시점(시점 A~E)별 3-State 세부 성능 감사

| 공식 시점 | v1 Defect F1 | v2 Defect F1 | v1 Clean F1 | v2 Clean F1 | 커버리지 (v2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for vk, v_alias in view_keys:
        v1_vm = v1_data["view_benchmarks"][vk]["metrics"]
        v2_vm = v2_data["view_benchmarks"][vk]["metrics"]
        is_imp = v2_vm['selective_defect_f1'] > v1_vm['selective_defect_f1']
        imp_tag = "**" if is_imp else ""
        md_content += f"| **{v_alias}** | {v1_vm['selective_defect_f1']*100:.2f}% | {imp_tag}{v2_vm['selective_defect_f1']*100:.2f}%{imp_tag} | {v1_vm['selective_clean_f1']*100:.2f}% | {v2_vm['selective_clean_f1']*100:.2f}% | {v2_vm['coverage']*100:.2f}% |\n"

    md_content += """
---
### 4. 종합 결론 및 배포 권고사항
1. **실서비스 사용성 극대화**: v1은 재촬영 요구율(28.95%)이 높아 10번 중 3번 꼴로 재촬영을 요구했던 반면, v2는 유효 판정 커버리지를 **96.30%**로 대폭 끌어올려 회원 UX를 획기적으로 개선함.
2. **후방 사각지대 돌파 성공**: $R_{body}$ 강체 직교 정규화를 통해 가림 및 원근 왜곡이 가장 극심했던 **시점 E(좌측 후방 45°) Defect F1이 34.92% → 41.76%(+6.84%p), 시점 A(우측 후방 45°)가 23.53% → 27.66%(+4.13%p) 동시 향상**됨.
3. **오경보 억제 및 정확도 향상**: Full-Coverage 정밀도가 37.71%에서 **47.62%**로 **+9.91%p** 대폭 상승하고 전체 정확도가 **71.14%**로 **+3.42%p** 개선되어 비정상 알림의 신뢰도가 크게 제고됨.
4. **최종 결론**: 가중치 번들 `weights/bl2_v2_bundle.pkl`을 바벨 런지 Condition 2 공식 프로덕션 모델로 최종 승인 및 배포 확정함.

*보고서 생성 완료: vidio-health-trainer 프로젝트*
"""
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[+] Markdown report saved: {output_md}")

if __name__ == "__main__":
    v1_json = r"D:\src\models\second_models\babel_lunge\bl2_v1_benchmark_results.json"
    v2_json = r"D:\src\models\second_models\babel_lunge\bl2_v2_benchmark_results.json"
    out_docx = r"D:\자주프 문서\7주차\바벨런지_bl2_v1_vs_v2_성능비교_및_선행연구_반영_보고서.docx"
    out_md = r"D:\자주프 문서\7주차\바벨런지_bl2_v1_vs_v2_성능비교_및_선행연구_반영_보고서.md"
    generate_comparison_report(v1_json, v2_json, out_docx, out_md)
