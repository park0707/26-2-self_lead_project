# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: make_c1_to_c4_comprehensive_report_docx.py
Purpose: 바벨 스쿼트 4대 세부 결함 모델(c1~c4) 공식 프로덕션 버전 전체 성능 종합 Word 보고서(.docx) 생성
Target: D:\자주프 문서\5주차\vidio-health-trainer_바벨스쿼트_4대세부결함모델_c1_c4_전체성능_종합보고서.docx
"""

import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOC_DIR = r"D:\자주프 문서\5주차"
DOC_PATH = os.path.join(DOC_DIR, "vidio-health-trainer_바벨스쿼트_4대세부결함모델_c1_c4_전체성능_종합보고서.docx")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="1A365D"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def add_header_cell(row, idx, text, width, align=WD_ALIGN_PARAGRAPH.CENTER):
    c = row.cells[idx]
    c.width = width
    set_cell_background(c, "1A365D")
    set_cell_margins(c, top=140, bottom=140)
    p = c.paragraphs[0]
    p.alignment = align
    r = p.add_run(text)
    r.font.name = "맑은 고딕"
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(255, 255, 255)

def add_data_cell(row, idx, text, width, align=WD_ALIGN_PARAGRAPH.CENTER, bold=False, color_rgb=None, bg_hex=None):
    c = row.cells[idx]
    c.width = width
    if bg_hex:
        set_cell_background(c, bg_hex)
    set_cell_margins(c, top=100, bottom=100)
    p = c.paragraphs[0]
    p.alignment = align
    r = p.add_run(text)
    r.font.name = "맑은 고딕"
    r.font.size = Pt(9)
    r.font.bold = bold
    if color_rgb:
        r.font.color.rgb = color_rgb
    else:
        r.font.color.rgb = RGBColor(45, 55, 72)

def main():
    os.makedirs(DOC_DIR, exist_ok=True)
    doc = Document()

    # 페이지 여백 설정 (상하좌우 0.8인치)
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # 1. 문서 헤더 및 타이틀
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    tr = title_p.add_run("vidio-health-trainer 바벨 스쿼트 4대 세부 결함 모델(c1~c4) 전체 성능 종합 보고서")
    tr.font.name = "맑은 고딕"
    tr.font.size = Pt(17)
    tr.font.bold = True
    tr.font.color.rgb = RGBColor(26, 54, 93)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(16)
    sr = sub_p.add_run("순수 비전 100% MediaPipe 33관절 64프레임 기반 공식 프로덕션 모델(c1 v11, c2 v7, c3 v6, c4 v5) 전수(N=2,605) 5-Fold 실측 검증 및 3-State 선택적 판정 종합 감사")
    sr.font.name = "맑은 고딕"
    sr.font.size = Pt(10)
    sr.font.color.rgb = RGBColor(113, 128, 150)

    # 메타 정보 테이블
    info_tbl = doc.add_table(rows=5, cols=2)
    info_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(info_tbl, color="CBD5E0")
    meta = [
        ("문서 분류", "기술 감사 및 공식 프로덕션 배포 종합 보고서 (AI/생체역학 파이프라인)"),
        ("프로젝트 공식 명칭", "vidio-health-trainer (AI 기반 실시간 운동 자세 교정 및 결함 분류 시스템)"),
        ("대상 운동 및 결함", "바벨 백 스쿼트 (c1 척추 중립, c2 고개 정면, c3 발무릎 일치, c4 발바닥 지면 고정)"),
        ("검증 데이터셋 규모", "AI-Hub 바벨 스쿼트 2,605개 시퀀스 전수 (피험자 9명 x 5대 카메라 시점)"),
        ("검증 프로토콜", "피험자 격리 5-Fold StratifiedGroupKFold 교차 검증 (엄밀한 제로-리크 평가)")
    ]
    widths_info = [Inches(1.8), Inches(4.8)]
    for i, (k, v) in enumerate(meta):
        row = info_tbl.rows[i]
        add_data_cell(row, 0, k, widths_info[0], align=WD_ALIGN_PARAGRAPH.LEFT, bold=True, bg_hex="F7FAFC")
        add_data_cell(row, 1, v, widths_info[1], align=WD_ALIGN_PARAGRAPH.LEFT)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 1. 종합 요약
    h1 = doc.add_heading("1. 4대 세부 결함 모델 공식 확정 요약 (Executive Summary)", level=1)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    p_exec = doc.add_paragraph()
    p_exec.add_run(
        "본 보고서는 vidio-health-trainer 프로젝트의 2단계 핵심 파이프라인인 바벨 스쿼트 4대 세부 결함 모델(c1~c4)의 "
        "공식 프로덕션 버전을 최종 확정하고, 2,605개 마스터 데이터셋 전수(5대 시점 x 9명 피험자)에 대해 실측된 이진 판정 및 "
        "3-State 선택적 판정(Selective Classification) 성능을 종합 정리한 결과입니다.\n\n"
        "외부 AI 기술 자문과 도메인 생체역학 원리를 100% 수용하여 모델 아키텍처를 점진적 계승(Copy-and-Refine) 방식으로 고도화하였으며, "
        "단안 영상의 물리적 한계(노이즈 및 가림)를 3-State 선택적 판정으로 완벽히 통제함으로써 "
        "전체 커버리지 80.0%에서 평균 93.70% F1, 평균 정확도 93.61%, 위험 오류율 6.39%의 초고정밀 판정 체계를 구축하였습니다."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 2. 모델별 핵심 사양 및 기술 혁신
    h2 = doc.add_heading("2. 4대 세부 결함 모델 공식 확정 명세 및 핵심 기술 혁신", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    spec_tbl = doc.add_table(rows=5, cols=5)
    spec_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(spec_tbl)
    widths_spec = [Inches(0.6), Inches(1.5), Inches(0.8), Inches(1.5), Inches(2.2)]

    s_headers = ["코드", "결함 공식 명칭", "버전", "아키텍처 규격", "핵심 기술 혁신 및 도메인 원리"]
    for j, h in enumerate(s_headers):
        add_header_cell(spec_tbl.rows[0], j, h, widths_spec[j])

    spec_data = [
        ("c1", "척추 중립 위반", "v11.0", "3-Way Tri-Ensemble\n(GBDT+TCN)", "• Nadir ±8F 고해상도 국소 피처\n• Dilated TCN 시계열 텐서(64F x 24D) 결합\n• 3계층 불확실성 필터"),
        ("c2", "고개 정면 미유지", "v7.0", "19D 생체역학 앙상블\n(CatBoost+LGBM)", "• 단일 가시 귀(Visible Ear) 적응형 헤드 피치\n• 후방(코 미관측) 신뢰도 게이팅 이원화\n• 5대 전 시점 91%+ F1 달성"),
        ("c3", "발무릎 방향 불일치", "v6.0", "20D 관상면 앙상블\n(CatBoost+LGBM)", "• 관상면 무릎-발목 투영 비율(Valgus Ratio)\n• 정강이-대퇴골 회전 각도 결합\n• 스쿼트 가동범위 능동 마스킹"),
        ("c4", "발바닥 지면 고정 실패", "v5.0", "25D 하이브리드\n(CB+LGB+RF)", "• F0 접촉 상태 머신(Contact State Machine)\n• 발-카메라 Z축 가시성 계수 결합\n• 상승기 들썩임 전이 빈도 정량화")
    ]

    for i, s_row in enumerate(spec_data, 1):
        row = spec_tbl.rows[i]
        bg = "F7FAFC" if i % 2 == 1 else "FFFFFF"
        add_data_cell(row, 0, s_row[0], widths_spec[0], bold=True, bg_hex=bg)
        add_data_cell(row, 1, s_row[1], widths_spec[1], align=WD_ALIGN_PARAGRAPH.LEFT, bold=True, bg_hex=bg)
        add_data_cell(row, 2, s_row[2], widths_spec[2], bg_hex=bg)
        add_data_cell(row, 3, s_row[3], widths_spec[3], bg_hex=bg)
        add_data_cell(row, 4, s_row[4], widths_spec[4], align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 3. 이진 판정 성적표
    h3 = doc.add_heading("3. 이진 판정 (Full Decision / No Abstention) 종합 성적표", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    p_bin = doc.add_paragraph()
    p_bin.add_run("보류 없이 모든 표본에 대해 정상/결함을 강제로 판정할 때의 베이스라인 성적표입니다 (2,605개 전수 5-Fold 교차 검증).")

    bin_tbl = doc.add_table(rows=6, cols=6)
    bin_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(bin_tbl)
    widths_bin = [Inches(1.8), Inches(0.8), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0)]

    b_headers = ["모델 / 결함 명칭", "버전", "피처 차원", "이진 F1 Score", "이진 Accuracy", "ROC-AUC"]
    for j, h in enumerate(b_headers):
        add_header_cell(bin_tbl.rows[0], j, h, widths_bin[j])

    bin_data = [
        ("c1. 척추 중립 위반", "v11.0", "3-Way", "84.47%", "85.22%", "91.58%"),
        ("c2. 고개 정면 미유지", "v7.0", "19D", "83.96%", "82.69%", "91.94%"),
        ("c3. 발무릎 방향 불일치", "v6.0", "20D", "84.74%", "84.88%", "92.69%"),
        ("c4. 발바닥 지면 고정 실패", "v5.0", "25D", "93.29%", "93.36%", "98.37%"),
        ("4대 모델 평균 (Macro Mean)", "-", "-", "86.61%", "86.54%", "93.65%")
    ]

    for i, b_row in enumerate(bin_data, 1):
        row = bin_tbl.rows[i]
        is_mean = (i == 5)
        bg = "EBF8FF" if is_mean else ("F7FAFC" if i % 2 == 1 else "FFFFFF")
        c_rgb = RGBColor(43, 108, 176) if is_mean else None

        add_data_cell(row, 0, b_row[0], widths_bin[0], align=WD_ALIGN_PARAGRAPH.LEFT, bold=True, bg_hex=bg)
        add_data_cell(row, 1, b_row[1], widths_bin[1], bold=is_mean, bg_hex=bg)
        add_data_cell(row, 2, b_row[2], widths_bin[2], bold=is_mean, bg_hex=bg)
        add_data_cell(row, 3, b_row[3], widths_bin[3], bold=True, color_rgb=c_rgb, bg_hex=bg)
        add_data_cell(row, 4, b_row[4], widths_bin[4], bold=True, color_rgb=c_rgb, bg_hex=bg)
        add_data_cell(row, 5, b_row[5], widths_bin[5], bold=True, color_rgb=c_rgb, bg_hex=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 4. 3-State 선택적 판정 공식 확정 성적표
    h4 = doc.add_heading("4. 3-State 선택적 판정 (Selective Classification) 공식 확정 성적표", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    p_sel = doc.add_paragraph()
    p_sel.add_run(
        "실서비스 환경에서 애매한 자세나 랜드마크 지터 노이즈로 인해 사용자에게 잘못된 피드백을 주는 위험을 원천 차단하기 위해, "
        "공식 확정된 임계치(Double Threshold)를 적용하여 '정상 / 결함 / 판단 불가'로 판정한 결과입니다."
    )

    sel_tbl = doc.add_table(rows=6, cols=7)
    sel_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(sel_tbl)
    widths_sel = [Inches(1.6), Inches(1.4), Inches(0.7), Inches(0.7), Inches(0.8), Inches(0.8), Inches(0.6)]

    sl_headers = ["모델 / 결함 명칭", "공식 확정 임계치", "Coverage", "Abstention", "Selective F1", "Selective Acc", "오류율"]
    for j, h in enumerate(sl_headers):
        add_header_cell(sel_tbl.rows[0], j, h, widths_sel[j])

    sel_data = [
        ("c1. 척추 중립 위반", "[0.40, 0.60] 균형", "75.1%", "24.9%", "97.09%", "96.93%", "3.07%"),
        ("c2. 고개 정면 미유지", "전0.4/0.6, 후0.3/0.7", "64.2%", "35.8%", "93.20%", "92.82%", "7.18%"),
        ("c3. 발무릎 방향 불일치", "[0.40, 0.60] 균형", "92.2%", "7.8%", "87.42%", "87.59%", "12.41%"),
        ("c4. 발바닥 지면 고정", "[0.40, 0.60] 균형", "88.4%", "11.6%", "97.09%", "97.09%", "2.91%"),
        ("4대 모델 평균 (Macro)", "-", "80.0%", "20.0%", "93.70%", "93.61%", "6.39%")
    ]

    for i, sl_row in enumerate(sel_data, 1):
        row = sel_tbl.rows[i]
        is_mean = (i == 5)
        bg = "EBF8FF" if is_mean else ("F7FAFC" if i % 2 == 1 else "FFFFFF")
        c_rgb = RGBColor(43, 108, 176) if is_mean else None

        add_data_cell(row, 0, sl_row[0], widths_sel[0], align=WD_ALIGN_PARAGRAPH.LEFT, bold=True, bg_hex=bg)
        add_data_cell(row, 1, sl_row[1], widths_sel[1], bold=is_mean, bg_hex=bg)
        add_data_cell(row, 2, sl_row[2], widths_sel[2], bold=is_mean, bg_hex=bg)
        add_data_cell(row, 3, sl_row[3], widths_sel[3], bold=is_mean, bg_hex=bg)
        add_data_cell(row, 4, sl_row[4], widths_sel[4], bold=True, color_rgb=c_rgb, bg_hex=bg)
        add_data_cell(row, 5, sl_row[5], widths_sel[5], bold=True, color_rgb=c_rgb, bg_hex=bg)
        add_data_cell(row, 6, sl_row[6], widths_sel[6], bold=is_mean, bg_hex=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 5. 5대 시점별 세부 성적 감사
    h5 = doc.add_heading("5. 5대 카메라 시점별 선택적 F1 Score 전수 감사 성적표", level=1)
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(6)

    p_view = doc.add_paragraph()
    p_view.add_run(
        "특정 카메라 각도에서 성능이 급락하는 시점 편향(View Bias)을 감사하기 위해 5대 전 시점(각 시점별 521건)에 대한 "
        "세부 선택적 F1 성적을 대조한 결과입니다."
    )

    view_tbl = doc.add_table(rows=7, cols=6)
    view_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(view_tbl)
    widths_view = [Inches(1.8), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.9), Inches(1.2)]

    v_headers = ["카메라 시점 (View)", "c1 v11", "c2 v7", "c3 v6", "c4 v5", "시점별 평균 F1"]
    for j, h in enumerate(v_headers):
        add_header_cell(view_tbl.rows[0], j, h, widths_view[j])

    view_data = [
        ("view1 (완전 후방 180°)", "97.23%", "95.56%", "86.34%", "98.48%", "94.40%"),
        ("view2 (우측 45° 전측면)", "96.18%", "93.03%", "84.70%", "95.24%", "92.29%"),
        ("view3 (완전 정면 0°)", "96.57%", "91.86%", "94.72%", "97.03%", "95.04%"),
        ("view4 (좌측 45° 전측면)", "97.58%", "94.40%", "88.30%", "96.31%", "94.15%"),
        ("view5 (우후방 45° 후측면)", "97.89%", "91.03%", "82.53%", "98.58%", "92.51%"),
        ("전 시점 평균 (Macro Mean)", "97.09%", "93.20%", "87.42%", "97.09%", "93.70%")
    ]

    for i, v_row in enumerate(view_data, 1):
        row = view_tbl.rows[i]
        is_mean = (i == 6)
        bg = "EBF8FF" if is_mean else ("F7FAFC" if i % 2 == 1 else "FFFFFF")
        c_rgb = RGBColor(43, 108, 176) if is_mean else None

        add_data_cell(row, 0, v_row[0], widths_view[0], align=WD_ALIGN_PARAGRAPH.LEFT, bold=True, bg_hex=bg)
        add_data_cell(row, 1, v_row[1], widths_view[1], bold=is_mean, bg_hex=bg)
        add_data_cell(row, 2, v_row[2], widths_view[2], bold=is_mean, bg_hex=bg)
        add_data_cell(row, 3, v_row[3], widths_view[3], bold=is_mean, bg_hex=bg)
        add_data_cell(row, 4, v_row[4], widths_view[4], bold=is_mean, bg_hex=bg)
        add_data_cell(row, 5, v_row[5], widths_view[5], bold=True, color_rgb=c_rgb, bg_hex=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 6. 배포 아티팩트 및 향후 계획
    h6 = doc.add_heading("6. 공식 아티팩트 배포 현황 및 향후 계획", level=1)
    h6.paragraph_format.space_before = Pt(14)
    h6.paragraph_format.space_after = Pt(6)

    art_tbl = doc.add_table(rows=5, cols=4)
    art_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(art_tbl)
    widths_art = [Inches(0.6), Inches(2.0), Inches(2.0), Inches(2.0)]

    a_headers = ["결함", "프로덕션 소스 코드", "공식 가중치 아티팩트", "피처 캐시 아티팩트"]
    for j, h in enumerate(a_headers):
        add_header_cell(art_tbl.rows[0], j, h, widths_art[j])

    art_data = [
        ("c1", "c1_model_v11.py", "c1_model_v11_weights.pkl", "squat_c1_v11_a_diet_27d_feats.npz"),
        ("c2", "c2_model_v7.py", "c2_model_v7_weights.pkl", "squat_c2_v7_19d_feats.npz"),
        ("c3", "c3_model_v6.py", "c3_model_v6_weights.pkl", "squat_c3_foot_feats.npz"),
        ("c4", "c4_model_v5.py", "c4_model_v5_weights.pkl", "squat_c4_v5_25d_feats.npz")
    ]

    for i, a_row in enumerate(art_data, 1):
        row = art_tbl.rows[i]
        bg = "F7FAFC" if i % 2 == 1 else "FFFFFF"
        add_data_cell(row, 0, a_row[0], widths_art[0], bold=True, bg_hex=bg)
        add_data_cell(row, 1, a_row[1], widths_art[1], align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)
        add_data_cell(row, 2, a_row[2], widths_art[2], align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)
        add_data_cell(row, 3, a_row[3], widths_art[3], align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    p_future = doc.add_paragraph()
    p_future.add_run(
        "■ 향후 통합 파이프라인 계획:\n"
        "1. 다중 결함 동시 추론 통합 엔진 구축: c1~c4 모델을 단일 인터페이스(pipeline.predict(seq_kps))로 묶어 종합 자세 점수 및 결함 우선순위(Priority Ordering: c1 > c3 > c4 > c2) 출력.\n"
        "2. 실시간 웹/모바일 서비스 연동: 3-State 피드백 메시지(정상/결함/재촬영 유도)를 표준 JSON 포맷으로 패키징하여 프론트엔드로 전달."
    )

    doc.save(DOC_PATH)
    print(f"[*] 공식 Word 종합 보고서 생성 완료: {DOC_PATH}")

if __name__ == '__main__':
    main()
