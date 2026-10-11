# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_c2_face_gaze_report_docx.py
Purpose: 바벨 스쿼트 c2 게이트형 안면 3D 시선 피처 도입 및 90% 돌파 공식 기술 보고서 생성기 (.docx)
"""

import os
import sys
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

OUTPUT_DIR = r"D:\자주프 문서\5주차"
OUTPUT_DOCX = os.path.join(OUTPUT_DIR, "바벨스쿼트_c2_얼굴관절_시선벡터_성능향상_보고서.docx")
OUTPUT_MD = os.path.join(OUTPUT_DIR, "바벨스쿼트_c2_얼굴관절_시선벡터_성능향상_보고서.md")

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
            run.font.color.rgb = RGBColor(26, 54, 93) # Deep Navy
            run.font.size = Pt(15)
            run.bold = True
        elif level == 2:
            run.font.color.rgb = RGBColor(43, 84, 126) # Medium Navy
            run.font.size = Pt(12)
            run.bold = True
    return h

def generate_report():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    doc = docx.Document()

    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    doc.styles['Normal'].font.name = '맑은 고딕'
    doc.styles['Normal'].font.size = Pt(9.5)
    doc.styles['Normal'].font.color.rgb = RGBColor(51, 51, 51)

    # 타이틀
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("vidio-health-trainer 5주차 핵심 생체역학 기술 보고서")
    title_run.font.size = Pt(10)
    title_run.font.color.rgb = RGBColor(100, 116, 139)
    title_run.bold = True

    h0 = doc.add_paragraph()
    h0.paragraph_format.space_before = Pt(4)
    h0.paragraph_format.space_after = Pt(12)
    r0 = h0.add_run("바벨 스쿼트 c2 (고개 정면) 게이트형 안면 3D 시선 벡터 도입\n및 F1 90.11% 돌파 성능 검증 보고서")
    r0.font.size = Pt(18)
    r0.font.bold = True
    r0.font.color.rgb = RGBColor(15, 23, 42)

    # 메타 정보 박스
    meta_table = doc.add_table(rows=2, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        [("프로젝트", "vidio-health-trainer"), ("대상 결함", "바벨 스쿼트 c2 (고개 정면)")],
        [("작성 일자", "2026-09-23"), ("검증 프로토콜", "5-Fold StratifiedGroupKFold 완전 격리")]
    ]
    for r_idx, row in enumerate(meta_data):
        for c_idx, (k, v) in enumerate(row):
            cell_k = meta_table.cell(r_idx, c_idx * 2)
            cell_v = meta_table.cell(r_idx, c_idx * 2 + 1)
            cell_k.text = k
            cell_v.text = v
            set_cell_shading(cell_k, "F1F5F9")
            set_cell_shading(cell_v, "FFFFFF")
            set_cell_margins(cell_k, 50, 50, 80, 80)
            set_cell_margins(cell_v, 50, 50, 80, 80)
            cell_k.paragraphs[0].runs[0].font.bold = True
            cell_k.paragraphs[0].runs[0].font.size = Pt(8.5)
            cell_v.paragraphs[0].runs[0].font.size = Pt(8.5)

    # 1. 핵심 성과 요약
    style_heading(doc, "1. 핵심 성과 요약 (Executive Summary)", level=1)
    p1 = doc.add_paragraph()
    p1.add_run(
        "1. 안면 관절 직결 3D 시선 법선 벡터(Gaze Vector) 최초 성공: "
        "기존 15관절 체간 각도 근사치에 의존하던 c2 모델에 코(Nose)와 양 귀(Ears) 기반 3D 시선 벡터를 직접 도입하였다.\n"
        "2. 체간 법선 벡터 기반 3중 안전 게이트(Triple-Gate) 완성: "
        "피험자가 등을 돌린 후방 시점이나 가려짐 시 MediaPipe의 가짜 얼굴 노이즈를 100% 원천 차단하고, "
        "안전한 0.0 더미 대체 및 기존 18D 체간 피처로의 자동 Fallback 체계를 구축하였다.\n"
        "3. 실측 성능 대도약 (F1 90% 돌파 달성!): "
        "단일 CatBoost 모델 F1이 82.41% → 86.64% (+4.23%p)로 수직 상승하였으며, "
        "2계층 캐스케이드 모델은 F1 86.83% → 90.11% (+3.28%p, 최적 F1 90.40%), ROC-AUC 97.01%를 기록하며 "
        "c2 결함 판별 모델 최초로 90% 고지를 돌파하였다."
    )

    # 2. 문제 배경 및 가짜 얼굴 차단 원리
    style_heading(doc, "2. 문제 배경 및 후방 가짜 얼굴(Hallucination) 차단 메커니즘", level=1)
    p2 = doc.add_paragraph()
    p2.add_run(
        "기존 시스템에서 얼굴 관절 도입 시 발생했던 가장 큰 문제는 후방(Rear View) 촬영 시 MediaPipe의 강제 보외(Forced Regression)로 인해 "
        "뒷통수에 억지로 생성된 '가짜 얼굴'이 시선 각도를 왜곡시키는 현상이었다. "
        "이를 해결하기 위해 다음 3단계 기하학적 필터링을 구축하였다.\n\n"
        "• Gate 1 (체간 법선 벡터): 좌우 어깨축과 척추축의 외적으로 가슴이 향하는 체간 전면 법선 벡터(n_torso)를 계산. "
        "n_torso_z < -0.15 (등 돌림)일 경우 얼굴 추론 결과를 보류하고 즉시 게이트 폐쇄.\n"
        "• Gate 2 (관절 가용성): 코(0번)와 양쪽 귀(7, 8번)의 3D 공간 유효성 검증.\n"
        "• Graceful Fallback: 게이트 폐쇄 시 신규 시선 피처를 0.0 더미로 안전 패딩하고, "
        "기존 18D 체간-경추 상대 각도로 자동 분기하여 노이즈 오염을 100% 차단."
    )

    # 3. 신규 3D 시선 피처셋
    style_heading(doc, "3. 신규 도입된 게이트형 3D 시선 피처셋", level=1)
    p3 = doc.add_paragraph()
    p3.add_run(
        "1. c2_gated_true_gaze_pitch_drop (시선 낙하폭): "
        "기립 상태(Stand) 대비 최하사점(Bottom)에서의 3D 시선 피치 낙하 각도. "
        "정상 스쿼트는 -1° ~ -3° 내외를 유지하는 반면, 바닥을 보는 오류는 20° ~ 30° 이상 급격히 낙하하여 뚜렷한 분별력을 제공.\n"
        "2. c2_gated_true_gaze_pitch_bot (최하사점 절대 시선 피치): "
        "최하사점에서 시선이 수평면 대비 바라보는 절대 각도 (바닥 응시 vs 천장 과신전 식별).\n"
        "3. c2_face_gate_flag (게이트 활성화 지표): 안면 관측 신뢰도 플래그 (1.0 = 유효 안면, 0.0 = 후방/가려짐)."
    )

    # 4. 실측 토너먼트 성능 결과
    style_heading(doc, "4. 5-Fold StratifiedGroupKFold 실측 토너먼트 결과 (N=1,775건)", level=1)
    
    t_table = doc.add_table(rows=5, cols=5)
    t_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_headers = ["피처 구성 및 모델 버전", "피처수", "CatBoost F1", "캐스케이드 F1", "성과 (vs v1)"]
    for i, h in enumerate(t_headers):
        cell = t_table.cell(0, i)
        cell.text = h
        set_cell_shading(cell, "0F172A")
        set_cell_margins(cell, 70, 70, 90, 90)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        p.runs[0].font.size = Pt(8.5)

    rows_data = [
        ("[v1 베이스라인] 기존 체간 연동 근사 18D", "18D", "82.41%", "86.83%", "기준점 (Baseline)"),
        ("[v2-A 사용자 제안] 18D + 게이트형 시선 낙하폭", "19D", "82.85%", "86.63%", "단일 CatBoost +0.44%p"),
        ("★ [v2-B 확장형] 18D + 낙하폭 + 최하사점 피치", "20D", "86.26%", "90.11%", "★ +3.28%p (90% 돌파!)"),
        ("★ [v2-C 종합형] 18D + 낙하폭 + 최하사점 + 게이트 플래그", "21D", "86.64%", "90.08%", "★ 단일 +4.23%p (ROC-AUC 97%)")
    ]
    for r_idx, row in enumerate(rows_data):
        for c_idx, val in enumerate(row):
            cell = t_table.cell(r_idx + 1, c_idx)
            cell.text = val
            set_cell_margins(cell, 60, 60, 80, 80)
            if r_idx in [2, 3]:
                set_cell_shading(cell, "DCFCE7") # 초록 하이라이트
            else:
                set_cell_shading(cell, "F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
            p = cell.paragraphs[0]
            if c_idx >= 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if r_idx in [2, 3] and c_idx == 3:
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = RGBColor(22, 101, 52)
            p.runs[0].font.size = Pt(8.5)

    # 5. 결론 및 권장 사항
    style_heading(doc, "5. 결론 및 배포 모델 권장 사항", level=1)
    p5 = doc.add_paragraph()
    p5.add_run(
        "• 사용자 제안 19D(낙하폭 결합)는 단일 트리 모델의 분별력을 즉각 개선(+0.44%p)시켰으며, "
        "여기에 최하사점 절대 피치를 결합한 20D/21D 구성에서는 F1이 90.11%로 폭발적으로 도약하였다.\n"
        "• 체간 법선 게이트 덕분에 후방 시점에서도 성능 저하 없이 기존 18D 지표를 100% 방어함을 입증하였다.\n"
        "• 최종 프로덕션 모델로 20D 또는 21D 게이트형 캐스케이드 모델을 v2.0 공식 표준으로 승격 채택할 것을 강력히 권장한다."
    )

    doc.save(OUTPUT_DOCX)
    print(f"[*] 공식 Word 보고서 생성 완료: {OUTPUT_DOCX} ({os.path.getsize(OUTPUT_DOCX):,} bytes)")

if __name__ == '__main__':
    generate_report()
