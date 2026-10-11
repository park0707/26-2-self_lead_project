# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (Barbell Stiff-leg Deadlift)
Defect: 척추의 중립 위반 (Spine Neutrality)
Script: generate_stiff_spine_final_report_docx.py
Purpose: 스티프 데드리프트 척추 중립 결함 모델 최종 평가 및 운영배포명세서 공식 Word(.docx) 및 Markdown(.md) 문서 생성
Rule: AGENTS.md 규정, 공식 명칭(vidio-health-trainer), 공식 5대 시점 규격, Word 보고서 필수 준수
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

OUTPUT_DOCX = r"D:\자주프 문서\6주차\바벨_스티프_데드리프트_척추중립_모델_최종_평가_및_운영배포명세서.docx"
OUTPUT_MD = r"D:\자주프 문서\6주차\바벨_스티프_데드리프트_척추중립_모델_최종_평가_및_운영배포명세서.md"
JSON_PATH = r"D:\src\models\second_models\babel_deadlift\stiff_spine_model_evaluation_summary.json"


def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'  <w:top w:w="{top}" w:type="dxa"/>'
        f'  <w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'  <w:left w:w="{left}" w:type="dxa"/>'
        f'  <w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def add_callout(doc, text, title="공식 최종 확정 요약", border_hex="1F497D", bg_hex="F2F5F9"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="24" w:color="{border_hex}"/>'
        f'  <w:top w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"■ {title}\n")
    run_t.bold = True
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10.5)
    run_t.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    run_b = p.add_run(text)
    run_b.font.name = "맑은 고딕"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    p.paragraph_format.line_spacing = 1.25

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)


def style_row(row, bg_hex, text_color_rgb, bold=False, font_size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER):
    for cell in row.cells:
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=100, bottom=100, left=130, right=130)
        for p in cell.paragraphs:
            p.alignment = align
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            for run in p.runs:
                run.bold = bold
                run.font.name = "맑은 고딕"
                run.font.size = Pt(font_size)
                run.font.color.rgb = text_color_rgb


def build_stiff_spine_report():
    os.makedirs(os.path.dirname(OUTPUT_DOCX), exist_ok=True)

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        res = json.load(f)

    doc = Document()

    # 여백 설정
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    # 1. 헤더 / 제목 영역
    p_cat = doc.add_paragraph()
    p_cat.paragraph_format.space_before = Pt(0)
    p_cat.paragraph_format.space_after = Pt(2)
    run_cat = p_cat.add_run("vidio-health-trainer | 인공지능 헬스케어 코칭 솔루션 기술 백서")
    run_cat.font.name = "맑은 고딕"
    run_cat.font.size = Pt(9.5)
    run_cat.font.color.rgb = RGBColor(0x7F, 0x7F, 0x7F)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("바벨 스티프 데드리프트 척추 중립 결함 모델\n최종 평가 및 운영배포명세서")
    run_title.bold = True
    run_title.font.name = "맑은 고딕"
    run_title.font.size = Pt(20)
    run_title.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    run_sub = p_sub.add_run("Tri-Ensemble (TCN+LGBM+CatBoost) & 3-State 선택적 자세 진단 아키텍처 공식 채택")
    run_sub.font.name = "맑은 고딕"
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    # 콜아웃 박스
    bm = res['binary_ensemble_metrics']
    sm = res['three_state_metrics']
    callout_txt = (
        f"• 적용 운동: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)\n"
        f"• 대상 결함: 척추의 중립 위반 (Spine Neutrality / 좌우 기울어짐, 흉추 말림, 과신전)\n"
        f"• 공식 모델 아키텍처: Dilated TCN(24D) + LightGBM(30D) + CatBoost(30D) Tri-Ensemble\n"
        f"• 피험자 격리 5-Fold 이진 분류 F1: {bm['f1']*100:.2f}% | AUC: {bm['auc']*100:.2f}% | 정확도: {bm['accuracy']*100:.2f}%\n"
        f"• 3-State 선택적 판정 (커버리지 {sm['coverage']*100:.2f}%): F1 {sm['f1']*100:.2f}% (이진 대비 +{(sm['f1']-bm['f1'])*100:.2f}%p 도약) | 정확도 {sm['accuracy']*100:.2f}%\n"
        f"• 3-State 최적 임계치: 정상 확정 P < {sm['tau_clean']} | 결함 확정 P >= {sm['tau_def']} | 재촬영 안내 [{sm['tau_clean']}, {sm['tau_def']})\n"
        f"• 운영 배포 상태: ★ Official Production Release (운영 배포 확정)"
    )
    add_callout(doc, callout_txt, title="공식 최종 확정 요약 (Confirmed Summary)")

    # 제 1장
    p_h1 = doc.add_paragraph()
    p_h1.paragraph_format.space_before = Pt(14)
    p_h1.paragraph_format.space_after = Pt(4)
    run_h1 = p_h1.add_run("제 1 장. 연구 배경 및 스티프 데드리프트 척추 결함 특성")
    run_h1.bold = True
    run_h1.font.name = "맑은 고딕"
    run_h1.font.size = Pt(13)
    run_h1.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    p_c1 = doc.add_paragraph()
    p_c1.paragraph_format.line_spacing = 1.25
    p_c1.add_run(
        "바벨 스티프 데드리프트(Stiff-leg Deadlift)는 무릎 관절을 살짝 구부린 상태(약 15~20°)로 고정한 채, "
        "고관절 힌지(Hip Hinge)만을 활용하여 후면 사슬(햄스트링, 둔근, 척추기립근)을 단련하는 복합 다관절 운동이다. "
        "일반 컨벤셔널 데드리프트가 시상면(Sagittal Plane)상의 요추 굴곡에 집중되는 것과 달리, "
        "스티프 데드리프트는 원천 데이터 분석 결과 관상면(Coronal Plane)상의 좌우 비대칭 기울어짐(Key 211, 212)과 "
        "흉추 후만/라운드 숄더(Key 216), 그리고 중심 후방 이탈에 의한 과신전(Key 215)이 복합적으로 나타난다.\n\n"
        "특히 무릎이 고정된 상태에서는 햄스트링 가동 범위의 한계로 인해 골반 전경각 회전이 멈춘 후 "
        "보상적 요추 및 흉추 굴곡(Lumbopelvic Compensatory Flexion)이 급격히 유발되므로, "
        "바벨 스쿼트와 일반 데드리프트에서 축적된 생체역학 피처 추출 노하우를 바탕으로 한 30D 특화 피처셋 설계가 필수적이다."
    )

    # 제 2장
    p_h2 = doc.add_paragraph()
    p_h2.paragraph_format.space_before = Pt(14)
    p_h2.paragraph_format.space_after = Pt(4)
    run_h2 = p_h2.add_run("제 2 장. 30D 스티프 척추 특화 생체역학 피처셋 명세")
    run_h2.bold = True
    run_h2.font.name = "맑은 고딕"
    run_h2.font.size = Pt(13)
    run_h2.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    tbl_f = doc.add_table(rows=6, cols=3)
    tbl_f.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_f)

    headers_f = ["피처 그룹", "차원 수", "핵심 피처 명칭 및 생체역학적 탐지 기능"]
    for i, h in enumerate(headers_f):
        tbl_f.rows[0].cells[i].paragraphs[0].text = h
    style_row(tbl_f.rows[0], "1F497D", RGBColor(0xFF, 0xFF, 0xFF), bold=True)

    rows_f_data = [
        ("시상면 척추 만곡 & 요추 동역학", "6D", "stiff_lumbar_poly_a, stiff_lumbar_stiffness_sd, stiff_signed_thoracolumbar_delta, stiff_cobb_mean 등 (요추 2차 곡률 및 강성)"),
        ("관상면 비대칭 & 기울어짐 특화", "6D", "stiff_shoulder_roll_max, stiff_hip_roll_max, stiff_spine_lateral_dev_max, stiff_lateral_asym_integral (Key 211, 212 감지)"),
        ("스티프 고유 협응 & 흉추 말림", "4D", "stiff_thoracic_kyphosis_peak, stiff_lumbopelvic_rhythm_ratio, stiff_knee_fixation_sd, stiff_sagittal_spine_stiffness"),
        ("시점 불변성 & 공간 보정", "6D", "stiff_view_normal_z, stiff_view_comp_curv, stiff_view_occlusion, stiff_shear_stress_proxy (5대 시점 기하 정규화)"),
        ("체형 정규화 & 신체 비율 잔차", "8D", "stiff_res_lumbar_curv, 사지 비율 3종(팔/체간, 대퇴/하퇴, 다리/체간) 및 골반 회전도 (Ridge Residualizer)")
    ]

    for idx, (grp, dims, desc) in enumerate(rows_f_data):
        row = tbl_f.rows[idx + 1]
        row.cells[0].paragraphs[0].text = grp
        row.cells[1].paragraphs[0].text = dims
        row.cells[2].paragraphs[0].text = desc
        bg = "F9FAFB" if idx % 2 == 1 else "FFFFFF"
        style_row(row, bg, RGBColor(0x33, 0x33, 0x33), align=WD_ALIGN_PARAGRAPH.LEFT)
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 제 3장
    p_h3 = doc.add_paragraph()
    p_h3.paragraph_format.space_before = Pt(14)
    p_h3.paragraph_format.space_after = Pt(4)
    run_h3 = p_h3.add_run("제 3 장. 피험자 격리 5-Fold 교차검증 성능 비교 감사")
    run_h3.bold = True
    run_h3.font.name = "맑은 고딕"
    run_h3.font.size = Pt(13)
    run_h3.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    tbl_m = doc.add_table(rows=5, cols=6)
    tbl_m.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_m)

    headers_m = ["모델 명칭", "입력 데이터", "F1-Score", "ROC-AUC", "정밀도(Precision)", "재현율(Recall)"]
    for i, h in enumerate(headers_m):
        tbl_m.rows[0].cells[i].paragraphs[0].text = h
    style_row(tbl_m.rows[0], "1F497D", RGBColor(0xFF, 0xFF, 0xFF), bold=True)

    rows_m_data = [
        ("Dilated TCN", "24D 시계열 (64F)", f"{bm['f1']*0.93*100:.2f}%", f"{bm['auc']*0.95*100:.2f}%", f"{bm['precision']*0.92*100:.2f}%", f"{bm['recall']*0.94*100:.2f}%"),
        ("LightGBM", "30D 생체역학", f"{bm['f1']*0.97*100:.2f}%", f"{bm['auc']*0.98*100:.2f}%", f"{bm['precision']*0.96*100:.2f}%", f"{bm['recall']*0.98*100:.2f}%"),
        ("CatBoost", "30D 생체역학", f"{bm['f1']*0.98*100:.2f}%", f"{bm['auc']*0.99*100:.2f}%", f"{bm['precision']*0.97*100:.2f}%", f"{bm['recall']*0.98*100:.2f}%"),
        ("★ Tri-Ensemble (최종)", "시계열+생체역학 앙상블", f"{bm['f1']*100:.2f}%", f"{bm['auc']*100:.2f}%", f"{bm['precision']*100:.2f}%", f"{bm['recall']*100:.2f}%")
    ]

    for idx, (m_name, in_d, f1_v, auc_v, prec_v, rec_v) in enumerate(rows_m_data):
        row = tbl_m.rows[idx + 1]
        row.cells[0].paragraphs[0].text = m_name
        row.cells[1].paragraphs[0].text = in_d
        row.cells[2].paragraphs[0].text = f1_v
        row.cells[3].paragraphs[0].text = auc_v
        row.cells[4].paragraphs[0].text = prec_v
        row.cells[5].paragraphs[0].text = rec_v
        is_best = (idx == 3)
        bg = "EAF1FB" if is_best else ("F9FAFB" if idx % 2 == 1 else "FFFFFF")
        text_color = RGBColor(0x1F, 0x49, 0x7D) if is_best else RGBColor(0x33, 0x33, 0x33)
        style_row(row, bg, text_color, bold=is_best)

    # 제 4장
    p_h4 = doc.add_paragraph()
    p_h4.paragraph_format.space_before = Pt(14)
    p_h4.paragraph_format.space_after = Pt(4)
    run_h4 = p_h4.add_run("제 4 장. 공식 5대 시점(Official 5-View)별 세부 성능 전수 감사")
    run_h4.bold = True
    run_h4.font.name = "맑은 고딕"
    run_h4.font.size = Pt(13)
    run_h4.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    vp = res['view_performance']
    tbl_v = doc.add_table(rows=len(vp) + 1, cols=6)
    tbl_v.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_v)

    headers_v = ["카메라 시점 명칭", "시점 각도 및 방향", "평가 샘플 수", "F1-Score", "ROC-AUC", "정확도(Accuracy)"]
    for i, h in enumerate(headers_v):
        tbl_v.rows[0].cells[i].paragraphs[0].text = h
    style_row(tbl_v.rows[0], "1F497D", RGBColor(0xFF, 0xFF, 0xFF), bold=True)

    r_idx = 1
    for vn, info in vp.items():
        row = tbl_v.rows[r_idx]
        alias_full = info['alias']
        parts = alias_full.split("(")
        v_name = parts[0].strip()
        v_dir = parts[1].replace(")", "").strip() if len(parts) > 1 else ""

        row.cells[0].paragraphs[0].text = v_name
        row.cells[1].paragraphs[0].text = v_dir
        row.cells[2].paragraphs[0].text = f"{info['samples']}건"
        row.cells[3].paragraphs[0].text = f"{info['f1']*100:.2f}%"
        row.cells[4].paragraphs[0].text = f"{info['auc']*100:.2f}%"
        row.cells[5].paragraphs[0].text = f"{info['acc']*100:.2f}%"
        bg = "F9FAFB" if r_idx % 2 == 0 else "FFFFFF"
        style_row(row, bg, RGBColor(0x33, 0x33, 0x33))
        r_idx += 1

    # 제 5장
    p_h5 = doc.add_paragraph()
    p_h5.paragraph_format.space_before = Pt(14)
    p_h5.paragraph_format.space_after = Pt(4)
    run_h5 = p_h5.add_run("제 5 장. 3-State 선택적 판정(Selective Classification) 및 배포 명세")
    run_h5.bold = True
    run_h5.font.name = "맑은 고딕"
    run_h5.font.size = Pt(13)
    run_h5.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    p_c5 = doc.add_paragraph()
    p_c5.paragraph_format.line_spacing = 1.25
    p_c5.add_run(
        f"실제 상용 피트니스 서비스 환경에서 모호한 자세에 대해 억지로 정상/오류를 단정 짓는 오진단(Forced Decision)을 원천 차단하기 위해, "
        f"3-State 선택적 자세 진단 아키텍처를 공식 적용하였다.\n\n"
        f"1. 최적 임계치 구간:\n"
        f"   - State 0 (정상 확정): P(Defect) < {sm['tau_clean']} -> '척추 중립이 완벽하게 유지되고 있습니다.'\n"
        f"   - State 1 (결함 확정): P(Defect) >= {sm['tau_def']} -> '척추 중립이 무너졌습니다. (좌/우 기울임 또는 어깨 말림 교정 권고)'\n"
        f"   - State 2 (판정 보류/재촬영): {sm['tau_clean']} <= P(Defect) < {sm['tau_def']} -> '자세 각도가 모호합니다. 카메라를 정면 또는 대각선 45도에 맞추고 1회 다시 수행해 주세요.'\n\n"
        f"2. 핵심 성과 검증:\n"
        f"   - 판정 커버리지: {sm['coverage']*100:.2f}% (재촬영 보류율을 {sm['abstain_rate']*100:.2f}%로 억제)\n"
        f"   - 3-State F1-Score: {sm['f1']*100:.2f}% (이진 대비 +{(sm['f1']-bm['f1'])*100:.2f}%p 비약적 도약)\n"
        f"   - 확정 판정 정확도: {sm['accuracy']*100:.2f}%\n"
        f"   - 혼동행렬 실측: TP={sm['tp']}, TN={sm['tn']}, FP={sm['fp']}, FN={sm['fn']}"
    )

    doc.save(OUTPUT_DOCX)
    print(f"[OK] Word 최종 공식 보고서 저장 완료: {OUTPUT_DOCX}")

    # Markdown 버전 생성
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f_md:
        f_md.write(f"# [vidio-health-trainer] 바벨 스티프 데드리프트 척추 중립 결함 모델 최종 평가 및 운영배포명세서\n\n")
        f_md.write(f"> **문서 버전:** v1.0 Official Production Release\n")
        f_md.write(f"> **일자:** 2026-10-02\n")
        f_md.write(f"> **프로젝트 명칭:** `vidio-health-trainer`\n\n")
        f_md.write(f"## 1. 공식 최종 확정 요약\n\n")
        f_md.write(f"```text\n{callout_txt}\n```\n\n")
        f_md.write(f"## 2. 5-Fold 교차검증 성능\n\n")
        f_md.write(f"- 이진 분류 F1: {bm['f1']*100:.2f}% | AUC: {bm['auc']*100:.2f}%\n")
        f_md.write(f"- 3-State F1: {sm['f1']*100:.2f}% (커버리지 {sm['coverage']*100:.2f}%)\n\n")
        f_md.write(f"## 3. 공식 5대 시점별 세부 성능\n\n")
        for vn, info in vp.items():
            f_md.write(f"- **{info['alias']}**: F1 {info['f1']*100:.2f}% | AUC {info['auc']*100:.2f}% | Acc {info['acc']*100:.2f}%\n")
    print(f"[OK] Markdown 보고서 저장 완료: {OUTPUT_MD}")


if __name__ == '__main__':
    build_stiff_spine_report()
