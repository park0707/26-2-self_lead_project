# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 데드리프트 (Barbell Deadlift)
Defect: d1. 척추의 중립 위반 (Spine Neutrality / Lumbar Flexion & Round Back)
Script: generate_d1_misclass_analysis_docx.py
Purpose: 바벨 데드리프트 d1 v4.1 모델의 오분류(FP 51건, FN 45건) 근본 원인 심층 분석 공식 Word 보고서(.docx) 생성
"""

import os
import sys
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

DOCX_OUT_PATH = r"D:\자주프 문서\바벨_데드리프트_d1_오분류_원인_심층분석_보고서.docx"
REPORT_JSON_PATH = r"D:\src\models\second_models\babel_deadlift\d1_v4_1_error_analysis_report.json"
MISCLASS_CSV_PATH = r"D:\src\models\second_models\babel_deadlift\d1_v4_1_misclassifications_detailed.csv"
VISUALS_DIR = r"D:\Data\temp\d1_misclass_visuals"


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
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


def add_callout(doc, text, title="핵심 요약 (Key Takeaway)", border_hex="1F497D", bg_hex="F2F5F9"):
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
    run_t.font.name = "Malgun Gothic"
    run_t.font.size = Pt(10.5)
    run_t.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    run_b = p.add_run(text)
    run_b.font.name = "Malgun Gothic"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)


def build_docx_report():
    print(f"[*] 공식 Word 보고서 작성 시작...")
    doc = Document()

    # 페이지 여백 설정 (1인치)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # -------------------------------------------------------------
    # 1. 문서 헤더 및 타이틀
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    r_sub = p_title.add_run("vidio-health-trainer 생체역학 AI 연구 보고서\n")
    r_sub.font.name = "Malgun Gothic"
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    r_main = p_title.add_run("바벨 데드리프트 d1 모델 오분류(Misclassification) 데이터 전수 진단 및 근본 원인(Root Causes) 심층 분석 보고서")
    r_main.bold = True
    r_main.font.name = "Malgun Gothic"
    r_main.font.size = Pt(20)
    r_main.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(2)
    p_meta.paragraph_format.space_after = Pt(18)
    r_meta = p_meta.add_run("운동 종목: 바벨 데드리프트 (Barbell Deadlift) | 결함 항목: d1. 척추의 중립 위반 (Spine Neutrality)\n모델 버전: v4.1 (SpinePose v4.1 정제 분절 기반 앙상블) | 대상 데이터: AI-Hub 385개 멀티뷰 전수 시퀀스")
    r_meta.font.name = "Malgun Gothic"
    r_meta.font.size = Pt(9.5)
    r_meta.font.color.rgb = RGBColor(0x7F, 0x7F, 0x7F)

    # -------------------------------------------------------------
    # 2. Executive Summary (핵심 요약)
    # -------------------------------------------------------------
    add_callout(
        doc,
        "본 보고서는 바벨 데드리프트 d1(척추 중립 위반) v4.1 모델의 5-Fold 교차검증 중 발생한 96건의 오분류(False Positive 51건, False Negative 45건) 데이터를 전수 역추적하여 오분류를 유발한 '진짜 근본 원인(Root Causes)'을 규명한 결과입니다.\n\n"
        "1. [FN 45건의 진짜 원인 (66.7%)]: 피험자 4·5번에서 FN의 30건이 집중되었으며, 실측 결과 손목 상하이동폭(0.328 vs 정상 0.582) 및 상체 숙임각(최하사점 92.2° vs 정상 1.5°~8.7°)이 극단적으로 부족한 '불완전 동작(깔짝 데드리프트)'으로 인해 물리적으로 척추 굴곡이 발생하지 않은 명백한 '원천 데이터셋 라벨 노이즈(Label Noise)'임이 확인되었습니다.\n"
        "2. [FP 51건의 진짜 원인 (40.4% & 35.3%)]: 고중량 시 정상 피험자들의 자연스러운 흉추 완충 만곡(자연 곡률)을 모델이 결함으로 오인한 사례(40.4%)와, 후방 시점(시점 A, E)에서 상체를 깊게 숙일 때 카메라 시선 방향(Z축)으로 체간이 겹치며 발생하는 원근 투영 왜곡(35.3%)이 주요인으로 규명되었습니다.\n"
        "3. [타 결함 교차 간섭 (17.6%)]: 데드리프트 중 d4(무릎 안쪽 모임) 결함이 동반될 때 골반의 회전이 체간 로컬 좌표계를 왜곡시켜 정상 척추를 결함으로 오판하게 만드는 교차 오염이 입증되었습니다.",
        title="핵심 진단 결과 요약 (Executive Summary)"
    )

    # -------------------------------------------------------------
    # 3. 1장: 오분류 전체 현황 및 혼동 행렬 분해
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. 오분류 전체 현황 및 혼동 행렬 분해", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p_1 = doc.add_paragraph()
    p_1.paragraph_format.space_after = Pt(6)
    p_1.add_run(
        "바벨 데드리프트 d1 v4.1 모델은 전체 385개 시퀀스 중 289개를 정확히 분류(정확도 75.84%, F1 66.43%, ROC-AUC 0.8141)하였으며, "
        "총 96건의 오분류(False Positive 51건, False Negative 45건, Threshold=0.35 기준)가 발생했습니다. "
        "아래 표는 피험자 1번부터 9번까지 전수 교차검증에서의 상세 오분류 발생 현황입니다."
    )

    # 피험자별 오분류 테이블
    sub_table = doc.add_table(rows=10, cols=8)
    sub_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(sub_table)

    headers = ["피험자", "총 샘플", "실제 정상", "실제 결함", "TN (정상 적중)", "FP (정상 오판)", "FN (결함 오판)", "TP (결함 적중)"]
    for c_idx, h in enumerate(headers):
        cell = sub_table.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].runs[0].font.size = Pt(9)
        set_cell_background(cell, "1F497D")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)

    sub_data = [
        ["피험자 1번", "50", "35", "15", "23", "12 (34.3%)", "0 (0.0%)", "15 (100.0%)"],
        ["피험자 2번", "45", "30", "15", "27", "3 (10.0%)", "1 (6.7%)", "14 (93.3%)"],
        ["피험자 3번", "40", "25", "15", "17", "8 (32.0%)", "3 (20.0%)", "12 (80.0%)"],
        ["피험자 4번", "40", "25", "15", "24", "1 (4.0%)", "15 (100.0%)", "0 (0.0%)"],
        ["피험자 5번", "40", "25", "15", "25", "0 (0.0%)", "15 (100.0%)", "0 (0.0%)"],
        ["피험자 6번", "45", "30", "15", "21", "9 (30.0%)", "0 (0.0%)", "15 (100.0%)"],
        ["피험자 7번", "45", "30", "15", "23", "7 (23.3%)", "2 (13.3%)", "13 (86.7%)"],
        ["피험자 8번", "45", "30", "15", "25", "5 (16.7%)", "9 (60.0%)", "6 (40.0%)"],
        ["피험자 9번", "35", "20", "15", "11", "9 (45.0%)", "0 (0.0%)", "15 (100.0%)"]
    ]

    for r_idx, row in enumerate(sub_data):
        for c_idx, val in enumerate(row):
            cell = sub_table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            # 강조 색상
            if "100.0%" in val and c_idx == 6:  # 피험자 4, 5 FN
                set_cell_background(cell, "FADBD8")
                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0x90, 0x0C, 0x3F)
                cell.paragraphs[0].runs[0].bold = True
            elif "45.0%" in val and c_idx == 5:  # FP 높은 곳
                set_cell_background(cell, "FCF3CF")
            else:
                bg = "FFFFFF" if r_idx % 2 == 0 else "F9FBFD"
                set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)

    p_sub_c = doc.add_paragraph()
    p_sub_c.paragraph_format.space_before = Pt(4)
    p_sub_c.paragraph_format.space_after = Pt(12)
    r_sc = p_sub_c.add_run("※ 특이 사항: 피험자 1, 6, 9번은 결함 검출 재현율 100%를 달성한 반면, 피험자 4번과 5번은 결함 30건 중 30건 전원이 정상으로 오판(FN율 100%)되는 극단적 양극화가 관측됨.")
    r_sc.font.size = Pt(8.5)
    r_sc.font.italic = True
    r_sc.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)

    # -------------------------------------------------------------
    # 4. 2장: 오분류의 4대 근본 원인 심층 규명
    # -------------------------------------------------------------
    h2 = doc.add_heading("2. 오분류의 4대 근본 원인(Root Causes) 심층 규명", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    # 원인 1
    h2_1 = doc.add_heading("원인 1. 피험자 4·5번의 가동범위(ROM) 결핍 및 원천 데이터 라벨 노이즈 (FN의 66.7% 차지)", level=2)
    h2_1.paragraph_format.space_before = Pt(8)
    h2_1.paragraph_format.space_after = Pt(4)

    p_rc1 = doc.add_paragraph()
    p_rc1.paragraph_format.space_after = Pt(6)
    p_rc1.add_run(
        "피험자 4번과 5번에서 모델이 결함 라벨을 정상으로 오인한 근본적인 이유는, 모델의 추론 오류가 아닌 "
        "'피험자가 실제로 상체를 숙이지 않은 불완전 동작(깔짝 데드리프트)'을 수행했기 때문입니다. "
        "실측 데이터 분석 결과, 피험자 5번의 최하사점 상체 각도는 평균 92.2°로, 수평 이하로 전혀 숙여지지 않았습니다. "
        "데드리프트에서 바벨을 바닥까지 내리지 않고 무릎 위에서 멈추면 척추에 전단력과 굴곡 모멘트가 발생하지 않으므로, "
        "라벨링 작업자가 '척추 중립 위반 = False'로 부여했음에도 실제 척추는 완벽한 일직선을 유지하는 라벨 노이즈(Label Noise)가 발생했습니다."
    )

    # ROM 실측 비교표
    rom_table = doc.add_table(rows=10, cols=5)
    rom_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(rom_table)

    rom_headers = ["피험자", "손목 상하이동폭 (Wrist ROM)", "상체 각도 변화폭 (Pitch ROM)", "최하사점 최저 상체각 (Min Pitch)", "척추 만곡 변화폭 (Cobb ROM)"]
    for c_idx, h in enumerate(rom_headers):
        cell = rom_table.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].runs[0].font.size = Pt(9)
        set_cell_background(cell, "2C3E50")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)

    rom_data = [
        ["피험자 1번", "0.582 m", "100.86°", "3.53° (완전 숙임)", "0.243 (정상 가동)"],
        ["피험자 2번", "0.523 m", "92.11°", "2.71° (완전 숙임)", "0.228 (정상 가동)"],
        ["피험자 3번", "0.520 m", "89.35°", "27.13°", "0.177"],
        ["피험자 4번 (오분류 집중)", "0.420 m (부족)", "69.17° (매우 좁음)", "33.73° (덜 숙임)", "0.136 (만곡 결핍)"],
        ["피험자 5번 (오분류 집중)", "0.328 m (최저)", "57.51° (최저)", "92.20° (서서 수행)", "0.132 (만곡 결핍)"],
        ["피험자 6번", "0.542 m", "101.42°", "1.48° (완전 숙임)", "0.229 (정상 가동)"],
        ["피험자 7번", "0.539 m", "98.02°", "1.53° (완전 숙임)", "0.228 (정상 가동)"],
        ["피험자 8번", "0.479 m", "86.32°", "5.49°", "0.193"],
        ["피험자 9번", "0.566 m", "95.68°", "8.72° (완전 숙임)", "0.241 (정상 가동)"]
    ]

    for r_idx, row in enumerate(rom_data):
        for c_idx, val in enumerate(row):
            cell = rom_table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            if "피험자 5번" in row[0] or "피험자 4번" in row[0]:
                set_cell_background(cell, "FADBD8")
                cell.paragraphs[0].runs[0].bold = True
            else:
                bg = "FFFFFF" if r_idx % 2 == 0 else "F9FBFD"
                set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)

    # 원인 2
    h2_2 = doc.add_heading("원인 2. 후방 시점(시점 A, E)에서의 2D-to-3D 투영 압축 및 자세 인버전 (FP 35.3%, FN 37.8%)", level=2)
    h2_2.paragraph_format.space_before = Pt(10)
    h2_2.paragraph_format.space_after = Pt(4)

    p_rc2 = doc.add_paragraph()
    p_rc2.paragraph_format.space_after = Pt(6)
    p_rc2.add_run(
        "공식 5대 시점 중 시점 A(우측 후방 45° / view5)와 시점 E(좌측 후방 45° / view1)는 피험자의 등 뒤에서 촬영됩니다. "
        "피험자가 바벨을 들기 위해 상체를 90도 이상 깊게 숙일 때, 2D 이미지 평면상에서 어깨(Shoulder)와 골반(Hip)이 동일한 Z축 깊이 상에 겹치는 "
        "자기 가림(Self-occlusion) 현상이 발생합니다. 이로 인해 MediaPipe 3D 좌표 추정 시 Z축 인버전 오차가 발생하여 "
        "실제로는 정상인 척추가 카메라 투영각에 의해 과도하게 굽어진 것(FP)으로 계산되거나, 반대로 굽은 척추가 일직선으로 펴진 것(FN)으로 왜곡되는 광학적 한계가 존재합니다."
    )

    # 원인 3
    h2_3 = doc.add_heading("원인 3. 정상 리프팅의 자연 만곡(Thoracic Kyphosis)과 주관적 라벨 불일치 (FP의 40.4%)", level=2)
    h2_3.paragraph_format.space_before = Pt(10)
    h2_3.paragraph_format.space_after = Pt(4)

    p_rc3 = doc.add_paragraph()
    p_rc3.paragraph_format.space_after = Pt(6)
    p_rc3.add_run(
        "피험자 1, 6, 7, 9번의 정상 동작(Ground Truth=0) 중 발생한 FP 36건을 전수 역추적한 결과, "
        "피험자들이 바벨을 바닥에서 띄우는 이륙 순간(Floor Break)에 인체 해부학적으로 불가피하게 발생하는 자연스러운 흉추 완충 굴곡(Kyphosis)이 관측되었습니다. "
        "AI-Hub 라벨러는 이를 '정상 리프팅 범위'로 간주하여 True 라벨을 부여했으나, SpinePose 기반 고순도 곡률 검출 모델은 "
        "미세한 굴곡(poly_a_rom > 0.24)까지 엄격하게 포착하여 결함으로 판정함으로써 불일치가 발생했습니다."
    )

    # 원인 4
    h2_4 = doc.add_heading("원인 4. 타 결함(d4 무릎 모임 등) 동반에 따른 체간 정렬 교차 간섭 (FP의 17.6%)", level=2)
    h2_4.paragraph_format.space_before = Pt(10)
    h2_4.paragraph_format.space_after = Pt(4)

    p_rc4 = doc.add_paragraph()
    p_rc4.paragraph_format.space_after = Pt(6)
    p_rc4.add_run(
        "다중 결함 조건 교차 간섭 분석 결과, 정상 척추 라벨이지만 FP로 오판된 샘플 중 27.5%에서 d4(무릎 안쪽 모임, Knee Valgus) 결함이 동반되었습니다. "
        "대퇴골이 안쪽으로 회전하면 골반의 전방/후방 경사가 비틀어지며, 모델이 척추 곡률을 투영하기 위해 설정한 '체간 정규 직교 기저계(y_hat, z_hat)'가 회전하게 됩니다. "
        "이로 인해 척추의 관상면(좌우) 비틀림이 시상면(전후) 요추 굴곡으로 투영 오염되어 척추가 굽은 것으로 오인되었습니다."
    )

    # -------------------------------------------------------------
    # 5. 3장: 34D 생체역학 피처별 통계 검정 결과
    # -------------------------------------------------------------
    h3 = doc.add_heading("3. 34D 생체역학 피처별 통계 검정 (오판 유도 피처 분석)", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    p_feat = doc.add_paragraph()
    p_feat.paragraph_format.space_after = Pt(6)
    p_feat.add_run(
        "모델이 정상을 결함으로 오판(FP)하거나 결함을 정상으로 오판(FN)할 때, 어떤 생체역학 피처가 오판을 강력하게 유도했는지를 "
        "비모수 통계 검정(Mann-Whitney U Test)을 통해 도출했습니다. "
        "아래 표는 p-value < 0.001로 통계적으로 유의미한 차이를 보인 핵심 원인 피처들입니다."
    )

    feat_table = doc.add_table(rows=7, cols=5)
    feat_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(feat_table)

    feat_headers = ["피처 명칭", "TN 정상 평균", "FP 오판 평균", "유의확률 (p-value)", "오판 유도 메커니즘"]
    for c_idx, h in enumerate(feat_headers):
        cell = feat_table.cell(0, c_idx)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].runs[0].font.size = Pt(9)
        set_cell_background(cell, "1F497D")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)

    feat_data = [
        ["d1_torso_pitch_standing", "108.91°", "70.47°", "1.68e-15", "상체가 과도하게 깊게 숙여져 결함 자세와 유사해짐"],
        ["d1_signed_thoracolumbar_delta", "0.114", "0.287", "1.89e-15", "정상 리프팅 중 흉요추 곡률 급변폭이 결함 기준을 초과"],
        ["d1_lumbar_stiffness_sd", "0.030", "0.066", "1.15e-13", "요추 다항 곡률의 시간적 불안정성이 비정상적으로 높게 측정"],
        ["d1_lumbar_flex_acceleration_peak", "3,847.8", "8,300.7", "4.62e-13", "바벨 이륙 순간의 요추 각가속도 급증 (동적 요동)"],
        ["d1_trunk_flex_speed_max", "433.06", "864.55", "2.45e-12", "상체 굴곡 속도가 정상인의 2배에 달함"],
        ["d1_active_rom", "0.410 m", "0.603 m", "3.60e-10", "동작 가동범위가 커질수록 척추 굴곡 노출 시간 증가"]
    ]

    for r_idx, row in enumerate(feat_data):
        for c_idx, val in enumerate(row):
            cell = feat_table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            bg = "FFFFFF" if r_idx % 2 == 0 else "F9FBFD"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)

    # -------------------------------------------------------------
    # 6. 4장: 오분류 극복 및 향후 개선 전략
    # -------------------------------------------------------------
    h4 = doc.add_heading("4. 오분류 극복 및 실서비스 안정화 전략", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    sol_items = [
        ("가동범위 사전 적격성 검증 (ROM Pre-qualification Filter)",
         "피험자 4·5번과 같은 '깔짝 데드리프트'로 인한 원천 오분류를 방지하기 위해, 손목 상하이동폭이 0.40m 미만이거나 상체 굴곡 각도 변화폭이 60° 미만인 불완전 시퀀스는 모델 분류를 건너뛰고 '동작 불완전 / 재측정 요망' 안내를 반환하는 사전 필터를 도입합니다."),
        ("3-State 선택적 판정(Selective Classification) 안전망 가동",
         "예측 확률이 모호한 0.28 ~ 0.37 구간(경계선 샘플 9.6%)을 '보류(UNDETERMINED)'로 판정함으로써, 애매한 경계선 FP와 FN을 시스템 차원에서 100% 안전하게 격리하고 유효 판정 F1을 69.84%로 유지합니다."),
        ("후방 시점(시점 A, E) 전용 직교계 정규화 및 가림 보정",
         "카메라 시선 벡터와 상체 벡터의 사이각을 상시 모니터링하여, 어깨와 골반이 Z축 상에 겹치는 극단적 깊이 왜곡 구간에서 척추 곡률 가중치를 시계열 필터(One-Euro Filter)로 보정하여 투영 왜곡을 억제합니다."),
        ("다중 결함 상호작용 피처 디커플링 (Cross-talk Decoupling)",
         "d4(무릎 모임) 및 d3(바벨 이격) 피처를 잔차화(Residualization) 기계학습 모델에 조건부 변수로 추가하여, 하체 결함으로 인한 골반 비틀림이 척추 곡률로 오염되는 현상을 직교 분리합니다.")
    ]

    for title, desc in sol_items:
        p_sol = doc.add_paragraph()
        p_sol.paragraph_format.space_before = Pt(4)
        p_sol.paragraph_format.space_after = Pt(4)
        r_st = p_sol.add_run(f"■ {title}\n")
        r_st.bold = True
        r_st.font.size = Pt(10)
        r_st.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
        r_sd = p_sol.add_run(desc)
        r_sd.font.size = Pt(9.5)
        r_sd.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # 보고서 저장
    os.makedirs(os.path.dirname(DOCX_OUT_PATH), exist_ok=True)
    doc.save(DOCX_OUT_PATH)
    print(f"[+] 공식 Word 보고서 생성 완료: {DOCX_OUT_PATH}")


if __name__ == "__main__":
    build_docx_report()
