# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: build_c4_v2_report_docx.py
Purpose: 바벨 스쿼트 c4(발바닥 지면 고정) v1 vs v2 3D 발 관절 지면 피처 도입 및 성능 비교 공식 Word/Markdown 보고서 생성기
- 규정 준수:
  1) 공식 프로젝트명 vidio-health-trainer 적용
  2) Word 문서(.docx) 기본/필수 규격 배포
  3) AI-Hub 9명 전수 5-Fold StratifiedGroupKFold 완전 격리 실측 수치 반영
"""

import os
import sys
import shutil
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

OUTPUT_DIR_5TH = r"D:\자주프 문서\5주차"
OUTPUT_DIR_SUB = r"D:\자주프 문서\5주차\바벨 스쿼트 오류 분류 모델"
DOCX_NAME = "바벨스쿼트_c4_발바닥지면고정_v1_vs_v2_발관절피처_성능비교_보고서.docx"
MD_NAME = "바벨스쿼트_c4_발바닥지면고정_v1_vs_v2_발관절피처_성능비교_보고서.md"

def set_cell_shading(cell, color_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def style_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    for run in h.runs:
        run.font.name = '맑은 고딕'
        if level == 1:
            run.font.color.rgb = RGBColor(26, 54, 93)
            run.font.size = Pt(14)
            run.bold = True
        elif level == 2:
            run.font.color.rgb = RGBColor(43, 84, 126)
            run.font.size = Pt(11.5)
            run.bold = True
        elif level == 3:
            run.font.color.rgb = RGBColor(74, 85, 104)
            run.font.size = Pt(10)
            run.bold = True
    return h

def build_docx(target_path):
    doc = docx.Document()

    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    doc.styles['Normal'].font.name = '맑은 고딕'
    doc.styles['Normal'].font.size = Pt(9.5)
    doc.styles['Normal'].font.color.rgb = RGBColor(51, 51, 51)

    # 헤더
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("vidio-health-trainer 5주차 핵심 생체역학 기술 보고서")
    title_run.font.size = Pt(10)
    title_run.font.color.rgb = RGBColor(100, 116, 139)
    title_run.bold = True

    h0 = doc.add_paragraph()
    h0.paragraph_format.space_before = Pt(4)
    h0.paragraph_format.space_after = Pt(12)
    r0 = h0.add_run("바벨 스쿼트 c4 (발바닥 지면 고정) v1 vs v2 성능 비교 및\n3D 발 관절 지면 피처 도입 결과 보고서")
    r0.font.size = Pt(17)
    r0.font.bold = True
    r0.font.color.rgb = RGBColor(15, 23, 42)

    # 메타 정보 박스
    meta_table = doc.add_table(rows=3, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        [("프로젝트", "vidio-health-trainer"), ("대상 운동/결함", "바벨 스쿼트 c4 (발바닥 지면 고정)")],
        [("최종 후보 모델", "v2-A (26D 결합) / v2-B (10D 단독)"), ("실측 성능 도약", "★ F1 61.03% → 90.18% (+29.15%p)")],
        [("작성 일자", "2026-09-23"), ("검증 데이터/프로토콜", "Level 1~3 실무군 1,824건 / 5-Fold StratifiedGroupKFold")]
    ]
    for r_idx, row in enumerate(meta_data):
        for c_idx, (k, v) in enumerate(row):
            cell_k = meta_table.cell(r_idx, c_idx * 2)
            cell_v = meta_table.cell(r_idx, c_idx * 2 + 1)
            cell_k.text = k
            cell_v.text = v
            set_cell_shading(cell_k, "F1F5F9")
            set_cell_shading(cell_v, "FFFFFF")
            set_cell_margins(cell_k, 40, 40, 60, 60)
            set_cell_margins(cell_v, 40, 40, 60, 60)
            cell_k.paragraphs[0].runs[0].font.bold = True
            cell_k.paragraphs[0].runs[0].font.size = Pt(8.5)
            cell_v.paragraphs[0].runs[0].font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 요약 하이라이트 박스
    box_table = doc.add_table(rows=1, cols=1)
    box_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_box = box_table.cell(0, 0)
    set_cell_shading(cell_box, "EFF6FF") # Soft Blue
    set_cell_margins(cell_box, 100, 100, 140, 140)
    bp = cell_box.paragraphs[0]
    br1 = bp.add_run("■ Executive Summary: 15관절 물리적 한계 돌파 및 실측 F1 90.18% 달성\n")
    br1.bold = True
    br1.font.size = Pt(10)
    br1.font.color.rgb = RGBColor(30, 64, 175)

    br2 = bp.add_run(
        "1. 기존 v1.0 모델의 구조적 한계: 레거시 15개 관절 체계에서는 발목(Ankle) 1점만 존재하여, 정상적인 발목 굴곡(Dorsiflexion)과 실제 결함인 발뒤꿈치 들림(Heel Lift)을 물리적으로 구별할 수 없어 F1 61.03%, ROC-AUC 71.62%에 정체되었습니다.\n"
        "2. 추가 관절 도입: AI-Hub 3D JSON 24관절 체계의 Left Foot, Right Foot(발끝/중족골 3D 좌표) 관절 포인트를 식별하고, Ankle과의 기하학적 결합을 통해 발바닥 경사각(Pitch), 수직 높이차(Y-Delta), 뒤꿈치 최대 상승고(Heel Lift) 등 10D 지면 접지 피처를 신규 개발했습니다.\n"
        "3. 상관계수 폭증: 기존 16D 발목 피처의 라벨 상관계수(r < 0.13) 대비, 신규 피처인 Foot Pitch Drop(r = 0.621), Heel Lift Max(r = 0.582), Foot Pitch Nadir(r = 0.564)는 5~10배 이상 높은 통계적 변별력을 입증했습니다.\n"
        "4. 실측 성능 도약: 9명 피험자 전수 5-Fold StratifiedGroupKFold 완전 격리 실측 결과, Cascade F1이 61.03%에서 90.18%로 +29.15%p 폭등하였으며, ROC-AUC는 71.62%에서 96.69%(+25.07%p)로 수직 상승했습니다. 오분류 건수는 650건에서 171건으로 73.7% 대폭 감소했습니다."
    )
    br2.font.size = Pt(9)
    br2.font.color.rgb = RGBColor(30, 41, 59)

    # 1. 배경 및 문제 정의
    style_heading(doc, "1. 배경 및 v1.0 모델의 물리적 한계 분석", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "바벨 스쿼트에서 c4 결함(발바닥 지면 고정 / Foot Ground Contact)은 스쿼트 수행 도중 체중 중심이 전방으로 쏠리며 발뒤꿈치가 지면에서 떨어지거나(Heel Raise), "
        "앞꿈치가 들리는 현상(Toe Lift), 또는 발바닥이 과도하게 안/바깥쪽으로 젖혀지며 접지 안정성이 붕괴되는 치명적인 운동 오류입니다.\n\n"
        "그러나 기존 15개 관절 체계(MediaPipe 레거시) 기반 v1.0 모델은 하체 관절 중 고관절(Hip), 무릎(Knee), 발목(Ankle)만을 추적했습니다. "
        "발목 1개 포인트만으로는 스쿼트 하강 시 무릎이 앞으로 나가며 발생하는 정상적인 발목 굴곡(Ankle Dorsiflexion)과 실제 발뒤꿈치가 바닥에서 들리는 병리적 현상을 전혀 구분할 수 없었습니다. "
        "실제로 기존 v1 16D 피처와 c4 결함 라벨 간의 피어슨 상관계수를 측정한 결과, 대부분 r = -0.04 ~ +0.09 수준으로 상관성이 전무했습니다."
    )

    # 2. 추가 관절 포인트 및 피처 엔지니어링
    style_heading(doc, "2. AI-Hub 3D 발 관절(Foot) 식별 및 생체역학 지면 피처 설계", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "본 연구에서는 15관절 체계의 한계를 극복하기 위해 AI-Hub 원천 3D JSON 데이터의 24개 관절 체계로부터 'Left Foot'과 'Right Foot' 포인트를 직접 추출했습니다. "
        "해당 포인트는 중족골(Metatarsal) 및 발가락 끝(Toe) 부근의 3D 공간 좌표(x, y, z cm)를 나타냅니다."
    )

    # 피처 표
    feat_table = doc.add_table(rows=11, cols=4)
    feat_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["피처명", "차원/단위", "생체역학적 정의 및 공학적 공식", "라벨 상관계수(r)"]
    for i, h in enumerate(headers):
        cell = feat_table.cell(0, i)
        cell.text = h
        set_cell_shading(cell, "1E293B")
        set_cell_margins(cell, 60, 60, 80, 80)
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)

    feat_rows = [
        ("c4_true_foot_pitch_drop", "deg", "기립 대비 최하사점 발바닥 경사각 증가폭 (뒤꿈치 들림 핵심 지표)", "r = 0.621 (최고)"),
        ("c4_true_heel_lift_max", "cm", "기립 기준면 대비 추정 뒤꿈치(Heel) 최대 수직 상승 변위", "r = 0.582"),
        ("c4_true_foot_pitch_nadir", "deg", "최하사점 좌우 발바닥(Ankle-Foot) 지면 절대 경사각", "r = 0.564"),
        ("c4_true_heel_toe_ydelta_drop", "cm", "기립 대비 최하사점 발목-발끝 수직 높이차(Δy) 증가 변위", "r = 0.545"),
        ("c4_true_foot_pitch_max", "deg", "스쿼트 전 구간 중 발바닥이 지면과 이룬 최대 경사각", "r = 0.529"),
        ("c4_true_heel_toe_ydelta_nadir", "cm", "최하사점에서 발목과 발끝 간의 상대 수직 높이차", "r = 0.510"),
        ("c4_true_toe_lift_max", "cm", "기립 기준면 대비 발끝(앞꿈치)의 최대 수직 들림량", "r = 0.485"),
        ("c4_true_foot_ground_wobble", "cm", "발목-발끝 수직 높이차의 전 구간 시계열 표준편차 (지면 들썩임)", "r = 0.381"),
        ("c4_true_foot_roll_inversion", "deg", "최하사점 발목-발끝 좌우 기울기 롤각 (안쪽/바깥쪽 날 들림)", "r = -0.318"),
        ("c4_true_bilateral_foot_pitch_asym", "deg", "좌우 발바닥 경사각의 최대 절대 차이 (편측 뒤꿈치 들림)", "r = 0.154")
    ]

    for r_idx, (f_name, f_unit, f_desc, f_corr) in enumerate(feat_rows):
        row = feat_table.rows[r_idx + 1]
        for c_idx, val in enumerate([f_name, f_unit, f_desc, f_corr]):
            cell = row.cells[c_idx]
            cell.text = val
            bg = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
            if c_idx == 3 and "0.621" in val:
                bg = "FEF3C7"
            set_cell_shading(cell, bg)
            set_cell_margins(cell, 40, 40, 60, 60)
            p_c = cell.paragraphs[0]
            if len(p_c.runs) > 0:
                p_c.runs[0].font.size = Pt(8)
                if c_idx == 0 or c_idx == 3:
                    p_c.runs[0].font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 3. v1 vs v2 다차원 토너먼트 실측 결과
    style_heading(doc, "3. v1 vs v2 다차원 실측 토너먼트 결과 (5-Fold StratifiedGroupKFold)", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "평가 프로토콜: AI-Hub 9명 피험자 전수를 5-Fold StratifiedGroupKFold로 분할하여, 학습 데이터와 평가 데이터가 피험자 단위로 완전히 격리된 환경에서 실측을 수행했습니다. "
        "모호성 Level 1~3 실무군 1,824건 및 원천 전체 1,906건 전수에 대해 5대 후보 피처셋의 성능을 평가했습니다."
    )

    # 실측 표 (1824건 실무군)
    style_heading(doc, "[표 1] Level 1~3 실무군 (1,824건) 기준 실측 성능 비교표", level=2)
    t_perf = doc.add_table(rows=6, cols=8)
    t_perf.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_headers = ["모델 후보군", "피처수", "CatBoost F1", "LightGBM F1", "Cascade F1", "Opt F1", "ROC-AUC", "F1 개선폭"]
    for i, h in enumerate(t_headers):
        cell = t_perf.cell(0, i)
        cell.text = h
        set_cell_shading(cell, "1E293B")
        set_cell_margins(cell, 60, 60, 60, 60)
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(8)

    perf_rows = [
        ("v1.0 (레거시 16D)", "16", "60.70%", "60.85%", "61.03%", "69.33%", "71.62%", "기준선 (Baseline)"),
        ("v2-A (풀 결합 26D)", "26", "90.17%", "89.96%", "90.18%", "90.58%", "96.69%", "★ +29.15%p (최고 성능)"),
        ("v2-B (신규 발 단독 10D)", "10", "89.88%", "89.98%", "89.52%", "89.93%", "96.09%", "+28.49%p (초고효율)"),
        ("v2-C (핵심 정예 18D)", "18", "88.95%", "88.94%", "89.07%", "89.40%", "96.41%", "+28.04%p"),
        ("v2-D (발 핵심 단독 8D)", "8", "87.98%", "88.16%", "88.00%", "88.46%", "95.56%", "+26.97%p")
    ]

    for r_idx, row_vals in enumerate(perf_rows):
        row = t_perf.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.text = val
            bg = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
            if "★" in val or "v2-A" in row_vals[0]:
                bg = "EFF6FF"
            set_cell_shading(cell, bg)
            set_cell_margins(cell, 40, 40, 50, 50)
            p_c = cell.paragraphs[0]
            if len(p_c.runs) > 0:
                p_c.runs[0].font.size = Pt(8)
                if c_idx == 0 or c_idx == 4 or c_idx == 6:
                    p_c.runs[0].font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 원천 1906건 표
    style_heading(doc, "[표 2] 원천 1,906건 전체 기준 실측 성능 비교표", level=2)
    t_raw = doc.add_table(rows=6, cols=6)
    t_raw.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_raw_headers = ["모델 후보군", "피처 구성", "CatBoost F1", "Cascade F1", "Opt F1", "ROC-AUC"]
    for i, h in enumerate(t_raw_headers):
        cell = t_raw.cell(0, i)
        cell.text = h
        set_cell_shading(cell, "334155")
        set_cell_margins(cell, 50, 50, 60, 60)
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(8)

    raw_rows = [
        ("v1.0 (레거시 16D)", "레거시 발목 16D", "57.13%", "56.01%", "65.97%", "63.95%"),
        ("v2-A (풀 결합 26D)", "레거시 16D + 3D 발 10D", "89.74%", "89.34%", "90.33%", "96.48%"),
        ("v2-B (신규 발 단독 10D)", "3D 발 지면 접지 10D", "89.18%", "89.13%", "89.36%", "95.68%"),
        ("v2-C (핵심 정예 18D)", "신규 8D + 레거시 10D", "88.90%", "89.21%", "89.40%", "96.20%"),
        ("v2-D (발 핵심 단독 8D)", "3D 발 핵심 8D", "88.07%", "88.49%", "88.68%", "95.31%")
    ]

    for r_idx, row_vals in enumerate(raw_rows):
        row = t_raw.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.text = val
            bg = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
            set_cell_shading(cell, bg)
            set_cell_margins(cell, 35, 35, 50, 50)
            p_c = cell.paragraphs[0]
            if len(p_c.runs) > 0:
                p_c.runs[0].font.size = Pt(8)
                if c_idx == 0 or c_idx == 2 or c_idx == 5:
                    p_c.runs[0].font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 4. 오분류 감소 분석
    style_heading(doc, "4. 혼동 행렬(Confusion Matrix) 및 오분류 73.7% 격멸 분석", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "Level 1~3 실무군(1,824건: 정상 941건, 결함 883건) 환경에서 v1.0 기준선과 v2-A 모델의 혼동 행렬을 비교 분석했습니다.\n"
    )

    t_cm = doc.add_table(rows=3, cols=6)
    t_cm.alignment = WD_TABLE_ALIGNMENT.CENTER
    cm_headers = ["모델", "TN (정상 정답)", "FP (정상 오탐)", "FN (결함 미탐)", "TP (결함 검출)", "총 오분류 건수"]
    for i, h in enumerate(cm_headers):
        cell = t_cm.cell(0, i)
        cell.text = h
        set_cell_shading(cell, "1E293B")
        set_cell_margins(cell, 50, 50, 60, 60)
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)

    cm_data = [
        ("v1.0 (레거시 16D)", "665건", "276건 (오탐)", "374건 (미탐)", "509건", "650건 (오분류율 35.6%)"),
        ("v2-A (신규 26D)", "869건", "72건 (오탐)", "99건 (미탐)", "784건", "171건 (오분류율 9.4%)")
    ]
    for r_idx, row_vals in enumerate(cm_data):
        row = t_cm.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.text = val
            bg = "FEE2E2" if r_idx == 0 else "DCFCE7" # Red vs Green tint
            set_cell_shading(cell, bg)
            set_cell_margins(cell, 40, 40, 50, 50)
            p_c = cell.paragraphs[0]
            if len(p_c.runs) > 0:
                p_c.runs[0].font.size = Pt(8.5)
                p_c.runs[0].font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    p_ana = doc.add_paragraph()
    p_ana.add_run(
        "■ 정밀 분석 인사이트:\n"
        "1. 정상 오탐(FP) 격감: 정상적으로 발목이 굴곡되는 스쿼트 동작을 뒤꿈치가 들린 것으로 오인하던 FP가 276건에서 72건으로 73.9%(204건) 감소했습니다.\n"
        "2. 결함 미탐(FN) 격감: 미세하게 뒤꿈치나 앞꿈치가 들린 결함을 놓치던 FN이 374건에서 99건으로 73.5%(275건) 감소했습니다.\n"
        "3. 전체 오분류 479건 제거: 총 오분류가 650건에서 171건으로 무려 73.7% 격멸되었으며, 결함 탐지율(Recall)은 57.64%에서 88.79%로 +31.15%p 대폭 상승했습니다."
    )

    # 5. 최종 결론 및 제안
    style_heading(doc, "5. 최종 모델 선정 및 향후 계획", level=1)
    p_end = doc.add_paragraph()
    p_end.add_run(
        "1. 최종 v2 모델 제안: 최고 F1(90.18%) 및 ROC-AUC(96.69%)를 달성한 'v2-A (26D 풀 피처 결합 모델)'을 공식 v2.0 모델로 확정 추천합니다. "
        "동시에 임베디드나 엣지 디바이스 등 연산량이 극히 제한된 환경에서는 단 10개 피처만으로 F1 89.88%를 기록한 'v2-B (10D 단독 모델)' 역시 훌륭한 경량화 대안입니다.\n\n"
        "2. 프로덕션 모듈 구축 완료: 본 실측 결과를 바탕으로 추론 클래스 BabelSquatC4ModelV2(c4_model_v2.py) 및 학습 가중치 파일(c4_model_v2_weights.pkl)을 제작하여 저장 완료했습니다.\n\n"
        "3. 차기 고도화 계획 (v3 단계): c2, c3 모델과 마찬가지로 26D 피처 중 모델의 일반화 성능을 저해하거나 중복되는 노이즈 피처를 규명하기 위해 LOFO(Leave-One-Feature-Out) 중요도 분석을 수행하고, "
        "방해 피처를 정밀 소거하여 F1 92%+ 달성을 목표로 하는 공식 v3.0 모델 고도화를 제안합니다."
    )

    doc.save(target_path)
    print(f"[*] Word 보고서 생성 완료: {target_path} (크기: {os.path.getsize(target_path)/1024:.1f} KB)")

def main():
    os.makedirs(OUTPUT_DIR_5TH, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SUB, exist_ok=True)

    target_docx_5th = os.path.join(OUTPUT_DIR_5TH, DOCX_NAME)
    target_docx_sub = os.path.join(OUTPUT_DIR_SUB, DOCX_NAME)

    build_docx(target_docx_5th)
    shutil.copy2(target_docx_5th, target_docx_sub)
    print(f"[*] 서브폴더 복사 완료: {target_docx_sub}")

if __name__ == '__main__':
    main()
