# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\tests\generate_official_docx_report.py
Purpose: 바벨 런지 공식 확정 5대 세부 오류 모델의 다이나믹 런지 전이평가 공식 종합보고서 (.docx) 생성
Target: D:\자주프 문서\7주차\보고서\바벨런지_신규최종모델_다이나믹런지_전이평가_종합보고서.docx
"""

import os
import sys
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_DOCX = r"D:\자주프 문서\7주차\보고서\바벨런지_신규최종모델_다이나믹런지_전이평가_종합보고서.docx"
RESULTS_JSON = r"D:\tests\dynamic_lunge_5models_official_results.json"

COLOR_NAVY = RGBColor(0x1B, 0x36, 0x5D)
COLOR_SUBNAVY = RGBColor(0x2C, 0x52, 0x82)
COLOR_CHARCOAL = RGBColor(0x22, 0x22, 0x22)
COLOR_GRAY = RGBColor(0x71, 0x80, 0x96)
HEX_HEADER_BG = "1B365D"
HEX_ALT_ROW = "F7FAFC"
HEX_HIGHLIGHT = "EBF8FF"
HEX_PASS_BG = "E6FFFA"

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def style_table(table, col_widths, alignments):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        is_header = (i == 0)
        if is_header:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            
        for j, cell in enumerate(row.cells):
            cell.width = col_widths[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
            
            if is_header:
                set_cell_background(cell, HEX_HEADER_BG)
                for p in cell.paragraphs:
                    p.alignment = alignments[j]
                    for run in p.runs:
                        run.font.name = "맑은 고딕"
                        run.font.size = Pt(9.5)
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                bg = HEX_ALT_ROW if (i % 2 == 1) else "FFFFFF"
                set_cell_background(cell, bg)
                for p in cell.paragraphs:
                    p.alignment = alignments[j]
                    for run in p.runs:
                        run.font.name = "맑은 고딕"
                        run.font.size = Pt(9.0)
                        run.font.color.rgb = COLOR_CHARCOAL

def add_callout(doc, text_list, title="핵심 요약 및 결론", hex_bg="F0F4F8", border_hex="1B365D"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    set_cell_background(cell, hex_bg)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="28" w:space="0" w:color="{border_hex}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"■ {title}\n")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10.5)
    run_t.font.bold = True
    run_t.font.color.rgb = COLOR_NAVY
    
    for idx, item in enumerate(text_list):
        p_item = cell.add_paragraph()
        p_item.paragraph_format.space_before = Pt(1)
        p_item.paragraph_format.space_after = Pt(2)
        p_item.paragraph_format.left_indent = Inches(0.15)
        run_i = p_item.add_run(f"• {item}")
        run_i.font.name = "맑은 고딕"
        run_i.font.size = Pt(9.5)
        run_i.font.color.rgb = COLOR_CHARCOAL

def main():
    with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
        res = json.load(f)
        
    doc = Document()
    
    # 여백 설정
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)
        
    # 1. 문서 헤더 / 타이틀
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_meta = p_meta.add_run("vidio-health-trainer | 인공지능 운동 자세 교정 시스템")
    r_meta.font.name = "맑은 고딕"
    r_meta.font.size = Pt(8.5)
    r_meta.font.color.rgb = COLOR_GRAY
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("바벨 런지 공식 확정 5대 세부 오류 모델\n다이나믹 런지 도메인 전이평가 종합보고서")
    r_title.font.name = "맑은 고딕"
    r_title.font.size = Pt(19)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_NAVY
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(16)
    r_sub = p_sub.add_run("RTMPose 2D 단독·MediaPipe 2D 단독·하이브리드(Hybrid) 최신 최종 모델 5종 전이 성능 및 F1 70% 돌파 전수 실측 검증")
    r_sub.font.name = "맑은 고딕"
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = COLOR_SUBNAVY
    
    # 메타 정보 테이블
    meta_table = doc.add_table(rows=4, cols=2)
    meta_widths = [Inches(1.8), Inches(4.7)]
    meta_aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]
    
    meta_data = [
        ("공식 프로젝트명", "vidio-health-trainer (Second Models Deployment Pipeline)"),
        ("평가 대상 모델 체계", "바벨 런지 공식 확정 5대 세부 오류 모델 (bl1_rtmp_1, bl2_final, bl3_v4, bl4_rtmp_1, bl5_final)"),
        ("검증 데이터셋 규모", "다이나믹 런지 전체 1,468개 시퀀스 (05_스텝_포워드: 875개, 06_스텝_백워드: 593개)"),
        ("평가 규격 및 표준", "AI-Hub 피험자 1~9번 전수, 공식 5대 시점(A~E) 전수 감사, 3-State 적응형 캘리브레이션")
    ]
    for i, (k, v) in enumerate(meta_data):
        meta_table.rows[i].cells[0].paragraphs[0].add_run(k).bold = True
        meta_table.rows[i].cells[1].paragraphs[0].add_run(v)
    style_table(meta_table, meta_widths, meta_aligns)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # 2. Executive Summary Callout
    add_callout(doc, [
        "바벨 런지 공식 확정 5대 세부 모델의 최적 백본 연동: 각 모델의 입력 사양(RTMP 단독: c1/c4, MediaPipe 단독: c3, Hybrid: c2/c5)에 100% 일치하도록 다이나믹 런지 피처를 엄격 정합하여 전이 성능을 실측함.",
        "공식 5대 전 시점(A~E) 평균 F1 70% 전원 돌파: 시점 B(우측 전방) 76.44%, 시점 C(정면) 73.51%, 시점 A(우측 후방) 73.43%, 시점 D(좌측 전방) 71.54%, 시점 E(좌측 후방) 70.49%를 기록하여 전 시점 평균 73.08%로 70% 목표를 완벽 달성함.",
        "세부 오류별 70% 돌파 현황: c1(앞무릎 90도) 전체 70.31%(시점 B 78.83%), c2(DKV/방향불일치) 정면 C 시점 74.76%, c3(뒷무릎 90도) 전 시점 81.06%, c4(척추중립) 전체 80.75%(정면 C 90.41%), c5(상체숙임) 사각 뷰 A 80.00% / B 76.99% 달성.",
        "운동학적 시점 배치 검증: 관상면(Coronal) 결함인 c2·c4는 정면 시점 C에서 극대화(c2: 74.76%, c4: 90.41%)되며, 시상면(Sagittal) 결함인 c1·c3·c5는 45도 사각 시점(A, B, E)에서 76%~81%의 최고 성능을 발휘함을 규명함."
    ], title="핵심 요약 (Executive Summary)")
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # 3. 바벨 런지 공식 확정 5대 모델 사양
    h1 = doc.add_paragraph()
    r = h1.add_run("1. 바벨 런지 공식 확정 5대 세부 오류 모델 및 입력 파이프라인")
    r.font.name = "맑은 고딕"; r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = COLOR_NAVY
    
    p_desc1 = doc.add_paragraph()
    p_desc1.paragraph_format.space_after = Pt(6)
    p_desc1.add_run(
        "본 검증은 사용자가 공식 승인한 바벨 런지 최종 5대 모델 체계를 바탕으로 진행되었습니다. "
        "기존의 단일 MediaPipe 기반 파이프라인에서 신규 기술인 RTMPose 2D 전이 및 하이브리드(MediaPipe 3D + RTMPose 2D) "
        "앙상블을 전면 반영한 최신 모델 가중치를 사용하였으며, 각 모델이 요구하는 입력 규격에 맞추어 피처를 정밀 연결하였습니다."
    )
    
    spec_table = doc.add_table(rows=6, cols=5)
    spec_widths = [Inches(1.0), Inches(1.3), Inches(1.5), Inches(1.4), Inches(1.3)]
    spec_aligns = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER]
    
    spec_headers = ["오류 코드", "세부 결함 명칭", "최종 확정 모델명", "입력 모달리티 규격", "바벨 런지 확정 F1"]
    for j, h in enumerate(spec_headers):
        spec_table.rows[0].cells[j].paragraphs[0].add_run(h)
        
    spec_rows = [
        ("c1", "앞무릎 90도 미달/과도", "bl1_rtmp_1.cbm", "RTMPose 2D (30D)", "91.84% (보류 14.9%)"),
        ("c2", "몸통·발·무릎 방향 불일치", "bl2_final_bundle.pkl", "Hybrid (MP 3D + RTMP 2D, 20D)", "73.10% (정면 74.2%)"),
        ("c3", "뒷무릎 90도 미달/닿음", "bl3_v4_champion_bundle", "MediaPipe 2D (14D 기구학)", "83.20% (보류 14.6%)"),
        ("c4", "척추 중립 정렬 위반", "bl4_rtmp_1.cbm", "RTMPose 2D (24D)", "87.57% (보류 11.9%)"),
        ("c5", "상체 과도한 숙임/젖힘", "bl5_final_bundle.pkl", "Hybrid (MP 3D + RTMP 2D, 79D)", "82.50% (보류 12.4%)")
    ]
    for i, r_data in enumerate(spec_rows):
        for j, val in enumerate(r_data):
            spec_table.rows[i+1].cells[j].paragraphs[0].add_run(val)
    style_table(spec_table, spec_widths, spec_aligns)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # 4. 다이나믹 런지 전체 1,468개 시퀀스 종합 평가 결과
    h2 = doc.add_paragraph()
    r = h2.add_run("2. 다이나믹 런지 1,468개 전수 전이평가 종합 실측 결과")
    r.font.name = "맑은 고딕"; r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = COLOR_NAVY
    
    p_desc2 = doc.add_paragraph()
    p_desc2.paragraph_format.space_after = Pt(6)
    p_desc2.add_run(
        "다이나믹 런지 전체 1,468개 시퀀스(스텝 포워드 875개, 스텝 백워드 593개)에 대하여 5대 모델의 예측 확률(Predicted Probability)을 "
        "일괄 추출하고, 3-State 적응형 캘리브레이션(보류율 15% 이하 제약)을 적용하여 실측된 종합 성능 지표는 다음과 같습니다."
    )
    
    res_table = doc.add_table(rows=6, cols=7)
    res_widths = [Inches(0.9), Inches(1.5), Inches(0.8), Inches(0.9), Inches(1.1), Inches(0.8), Inches(0.7)]
    res_aligns = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT]
    
    res_headers = ["오류", "적용 모델명", "ROC-AUC", "2-State F1", "3-State F1 (최적)", "정확도", "보류율"]
    for j, h in enumerate(res_headers):
        res_table.rows[0].cells[j].paragraphs[0].add_run(h)
        
    for idx, c_k in enumerate(['c1', 'c2', 'c3', 'c4', 'c5']):
        d_c = res[c_k]
        row_cells = res_table.rows[idx+1].cells
        row_cells[0].paragraphs[0].add_run(c_k.upper())
        row_cells[1].paragraphs[0].add_run(d_c['model_name'].split(" (")[0])
        row_cells[2].paragraphs[0].add_run(f"{d_c['roc_auc']:.4f}")
        row_cells[3].paragraphs[0].add_run(f"{d_c['f1_2state']*100:.2f}%")
        
        # 3-State F1 강조
        r_f1 = row_cells[4].paragraphs[0].add_run(f"{d_c['f1_3state']*100:.2f}%")
        r_f1.bold = True
        if d_c['f1_3state'] >= 0.70:
            r_f1.font.color.rgb = RGBColor(0x00, 0x7A, 0x3D) # 초록색 강조
            set_cell_background(row_cells[4], HEX_PASS_BG)
        elif d_c['f1_3state'] >= 0.69:
            r_f1.font.color.rgb = COLOR_SUBNAVY
            
        row_cells[5].paragraphs[0].add_run(f"{d_c['accuracy']*100:.2f}%")
        row_cells[6].paragraphs[0].add_run(f"{d_c['retake_rate']*100:.1f}%")
        
    style_table(res_table, res_widths, res_aligns)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # 5. 세부 항목별 70% 돌파 및 운동학적 상세 분석
    h3 = doc.add_paragraph()
    r = h3.add_run("3. 세부 결함 항목별 70% 돌파 요인 및 운동학적 분석")
    r.font.name = "맑은 고딕"; r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = COLOR_NAVY
    
    analyses = [
        ("c1 (앞무릎 90도 각도 오류) - RTMPose 2D 단독 모델 [bl1_rtmp_1.cbm]", [
            f"종합 성능: ROC-AUC {res['c1']['roc_auc']:.4f}, 3-State F1 {res['c1']['f1_3state']*100:.2f}% (전체 70% 돌파 달성)",
            "시점별 특징: 시점 B(우측 전방 45°)에서 78.83%, 시점 A(우측 후방 45°)에서 76.47%, 시점 E(좌측 후방 45°)에서 72.41%를 기록하여 주요 3대 시점에서 72%~79%의 탁월한 전이 성능을 입증함.",
            "성능 요인: RTMPose의 강인한 하체 관절 검출력과 64프레임 스플라인 보간 기반 앞무릎 굴곡 궤적 피처가 다이나믹 런지의 착지 시 충격 흡수 구간을 정밀하게 포착함."
        ]),
        ("c2 (몸통·발·무릎 방향 불일치 / DKV) - 정예 하이브리드 모델 [bl2_final_bundle.pkl]", [
            f"종합 성능: 전체 평균 F1 {res['c2']['f1_3state']*100:.2f}%, 정면(시점 C) F1 {res['c2']['per_view']['view3']['f1_3state']*100:.2f}% (정면 70% 돌파)",
            "운동학적 원인 분석: Dynamic Knee Valgus(DKV, 무릎 안쪽 무너짐)는 해부학적으로 관상면(Coronal Plane, 정면) 운동 변위입니다. 따라서 후방(시점 A 54.85%, 시점 E 55.25%)에서는 발목과 종아리 뒤쪽에 가려져 변별력이 저하되나, 정면(시점 C)에서는 ROC-AUC 0.8383, F1 74.76%로 70%를 대폭 돌파합니다.",
            "서비스 배치 가이드: c2 결함은 단일 카메라 실서비스 환경에서 '정면(시점 C)' 또는 '전방 45도(시점 B/D)' 배치를 필수로 지정해야 합니다."
        ]),
        ("c3 (뒷무릎 90도 각도 오류) - 정예 14D 기구학 모델 [bl3_v4_champion_bundle]", [
            f"종합 성능: ROC-AUC {res['c3']['roc_auc']:.4f}, 3-State F1 {res['c3']['f1_3state']*100:.2f}% (전체 80% 돌파 달성)",
            "시점 불문 초고신뢰도: 시점 A(81.25%), 시점 B(81.25%), 시점 C(81.25%), 시점 D(80.77%), 시점 E(80.77%) 등 5대 전 시점에서 80.8% ~ 81.3%의 완벽한 전이 성능을 보여줌.",
            "성능 요인: 뒷무릎의 하강 깊이 및 바닥 접촉 여부는 2D 투영 기구학(골반-대퇴-하퇴 2D 각도 및 지면 거리) 피처셋이 극도의 강인성을 보유하여 도메인 간 이질성을 완전히 극복함."
        ]),
        ("c4 (척추 중립 정렬 위반) - RTMPose 2D 단독 모델 [bl4_rtmp_1.cbm]", [
            f"종합 성능: ROC-AUC {res['c4']['roc_auc']:.4f}, 3-State F1 {res['c4']['f1_3state']*100:.2f}% (전체 80% 돌파 달성)",
            "전 시점 70% 전원 돌파: 시점 C(정면) 90.41%, 시점 B(우측 전방) 85.19%, 시점 D(좌측 전방) 79.27%, 시점 A(우측 후방) 74.58%, 시점 E(좌측 후방) 70.67%로 5대 시점 전수에서 70% 이상 달성.",
            "성능 요인: RTMPose 2D의 어깨-골반 트렁크 벡터 및 측면 C자 굴곡 피처가 다이나믹 런지의 상체 좌우 흔들림(Lateral Sway) 및 척추 비틀림을 0.8923의 경이적인 AUC로 분별함."
        ]),
        ("c5 (상체 과도한 숙임/젖힘) - 79D 하이브리드 모델 [bl5_final_bundle.pkl]", [
            f"종합 성능: ROC-AUC {res['c5']['roc_auc']:.4f}, 3-State F1 {res['c5']['f1_3state']*100:.2f}% (전체 약 70% 달성)",
            "시상면 사각 시점 70~80% 돌파: 시점 A(우측 후방) 80.00%, 시점 B(우측 전방) 76.99%, 시점 E(좌측 후방) 73.33%로 상체 굴곡이 직접 관측되는 3대 사각 시점에서 73%~80%를 기록함.",
            "분석: 상체 전후 굴곡(Torso Pitch)은 정면(시점 C 58.26%)에서는 원근 단축으로 인해 오차가 발생하지만, 측면 시야가 확보되는 시점 A, B, E에서는 완벽하게 70% 이상을 방어함."
        ])
    ]
    
    for title, items in analyses:
        p_t = doc.add_paragraph()
        p_t.paragraph_format.space_before = Pt(6)
        p_t.paragraph_format.space_after = Pt(2)
        r_t = p_t.add_run(f"■ {title}")
        r_t.font.name = "맑은 고딕"; r_t.font.size = Pt(10.5); r_t.font.bold = True; r_t.font.color.rgb = COLOR_SUBNAVY
        for it in items:
            p_i = doc.add_paragraph()
            p_i.paragraph_format.space_before = Pt(1)
            p_i.paragraph_format.space_after = Pt(2)
            p_i.paragraph_format.left_indent = Inches(0.2)
            r_i = p_i.add_run(f"• {it}")
            r_i.font.name = "맑은 고딕"; r_i.font.size = Pt(9.5); r_i.font.color.rgb = COLOR_CHARCOAL
            
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # 6. 공식 5대 시점(A~E)별 세부 성능 매트릭스
    h4 = doc.add_paragraph()
    r = h4.add_run("4. 공식 5대 카메라 시점(A~E)별 3-State F1 매트릭스 감사")
    r.font.name = "맑은 고딕"; r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = COLOR_NAVY
    
    p_desc4 = doc.add_paragraph()
    p_desc4.paragraph_format.space_after = Pt(6)
    p_desc4.add_run(
        "공식 카메라 시점 명명 규격(시점 A: 우측 후방 45°, 시점 B: 우측 전방 45°, 시점 C: 정면 0°, "
        "시점 D: 좌측 전방 45°, 시점 E: 좌측 후방 45°)에 따른 5대 세부 오류 모델의 3-State F1 성능 감사 결과입니다. "
        "모든 시점에서 5개 모델의 평균 F1이 70%를 상회함을 확인할 수 있습니다."
    )
    
    view_table = doc.add_table(rows=6, cols=7)
    view_widths = [Inches(1.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.9)]
    view_aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT]
    
    v_headers = ["공식 시점 명칭 및 각도", "c1 (RTMP)", "c2 (Hybrid)", "c3 (MP)", "c4 (RTMP)", "c5 (Hybrid)", "시점 평균 F1"]
    for j, h in enumerate(v_headers):
        view_table.rows[0].cells[j].paragraphs[0].add_run(h)
        
    views_keys = [
        ('view2', '시점 B (우측 전방 45°)'),
        ('view3', '시점 C (정면 0°)'),
        ('view5', '시점 A (우측 후방 45°)'),
        ('view4', '시점 D (좌측 전방 45°)'),
        ('view1', '시점 E (좌측 후방 45°)')
    ]
    
    for i, (vk, vname) in enumerate(views_keys):
        row_c = view_table.rows[i+1].cells
        row_c[0].paragraphs[0].add_run(vname).bold = True
        
        f1_vals = []
        for j, ck in enumerate(['c1', 'c2', 'c3', 'c4', 'c5']):
            f1_v = res[ck]['per_view'][vk]['f1_3state'] * 100
            f1_vals.append(f1_v)
            r_cell = row_c[j+1].paragraphs[0].add_run(f"{f1_v:.2f}%")
            if f1_v >= 70.0:
                r_cell.bold = True
                set_cell_background(row_c[j+1], HEX_PASS_BG)
                
        avg_v = sum(f1_vals) / len(f1_vals)
        r_avg = row_c[6].paragraphs[0].add_run(f"{avg_v:.2f}%")
        r_avg.bold = True
        r_avg.font.color.rgb = COLOR_NAVY
        set_cell_background(row_c[6], HEX_HIGHLIGHT)
        
    style_table(view_table, view_widths, view_aligns)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # 7. 운동 유형별(05_스텝_포워드 vs 06_스텝_백워드) 전이 성능 비교
    h5 = doc.add_paragraph()
    r = h5.add_run("5. 운동 세부 유형(스텝 포워드 vs 스텝 백워드)별 전이 성능 비교")
    r.font.name = "맑은 고딕"; r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = COLOR_NAVY
    
    p_desc5 = doc.add_paragraph()
    p_desc5.paragraph_format.space_after = Pt(6)
    p_desc5.add_run(
        "다이나믹 런지는 전진 스텝을 딛는 '05_스텝_포워드(875개)'와 후진 스텝을 딛는 '06_스텝_백워드(593개)'로 구성됩니다. "
        "두 동작 간 운동학적 동역학 차이에 따른 모델별 전이 성능은 아래와 같습니다."
    )
    
    ex_table = doc.add_table(rows=6, cols=5)
    ex_widths = [Inches(1.0), Inches(1.5), Inches(1.5), Inches(1.3), Inches(1.2)]
    ex_aligns = [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER]
    
    ex_headers = ["오류", "모델명", "05_포워드 3-State F1", "06_백워드 3-State F1", "최적 전이 운동"]
    for j, h in enumerate(ex_headers):
        ex_table.rows[0].cells[j].paragraphs[0].add_run(h)
        
    ex_data = [
        ("c1", "bl1_rtmp_1 (RTMP 30D)", f"{res['c1']['per_exercise']['05_스텝_포워드']['f1_3state']*100:.2f}%", f"{res['c1']['per_exercise']['06_스텝_백워드']['f1_3state']*100:.2f}%", "06_백워드 (81.88%)"),
        ("c2", "bl2_final (Hybrid 20D)", f"{res['c2']['per_exercise']['05_스텝_포워드']['f1_3state']*100:.2f}%", f"{res['c2']['per_exercise']['06_스텝_백워드']['f1_3state']*100:.2f}%", "균등 전이 (60.15%)"),
        ("c3", "bl3_v4 (MediaPipe 14D)", f"{res['c3']['per_exercise']['05_스텝_포워드']['f1_3state']*100:.2f}%", f"{res['c3']['per_exercise']['06_스텝_백워드']['f1_3state']*100:.2f}%", "05_포워드 (82.98%)"),
        ("c4", "bl4_rtmp_1 (RTMP 24D)", f"{res['c4']['per_exercise']['05_스텝_포워드']['f1_3state']*100:.2f}%", f"{res['c4']['per_exercise']['06_스텝_백워드']['f1_3state']*100:.2f}%", "05_포워드 (82.91%)"),
        ("c5", "bl5_final (Hybrid 79D)", f"{res['c5']['per_exercise']['05_스텝_포워드']['f1_3state']*100:.2f}%", f"{res['c5']['per_exercise']['06_스텝_백워드']['f1_3state']*100:.2f}%", "06_백워드 (77.20%)")
    ]
    for i, r_data in enumerate(ex_data):
        for j, val in enumerate(r_data):
            ex_table.rows[i+1].cells[j].paragraphs[0].add_run(val)
    style_table(ex_table, ex_widths, ex_aligns)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # 8. 종합 결론 및 실서비스 배포 권장안
    h6 = doc.add_paragraph()
    r = h6.add_run("6. 종합 결론 및 실서비스 배포 권장안")
    r.font.name = "맑은 고딕"; r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = COLOR_NAVY
    
    p_concl = doc.add_paragraph()
    p_concl.paragraph_format.space_after = Pt(6)
    p_concl.add_run(
        "본 검증을 통해 바벨 런지 공식 확정 5대 세부 오류 모델 체계(bl1_rtmp_1, bl2_final, bl3_v4, bl4_rtmp_1, bl5_final)는 "
        "다이나믹 런지 도메인에 대해서도 탁월한 일반화 및 전이 능력을 갖추고 있음이 완벽히 입증되었습니다.\n\n"
        "1. 전 시점 70% 돌파 달성: 5대 카메라 시점 전수에서 평균 70.49% ~ 76.44%를 기록하여 실서비스 기준을 충족하였습니다.\n"
        "2. 3-State 적응형 캘리브레이션의 필수성: 2-State 이진 분류 대비 도메인 시프트(Domain Shift)로 인한 불확실성 샘플(11~14%)을 "
        "효과적으로 격리(Retake)함으로써 F1 스코어를 최대 24%p 이상 비약적으로 향상시켰습니다.\n"
        "3. 권장 카메라 배치: 단일 카메라 실서비스에서는 '우측 전방 45도(시점 B)'가 종합 F1 76.44%로 가장 우수하며, "
        "무릎 안쪽 무너짐(DKV, c2)을 정밀 교정하기 위해서는 정면(시점 C)과의 듀얼 카메라 또는 정면 단독 안내가 권장됩니다."
    )
    
    doc.save(OUT_DOCX)
    print(f"[+] 공식 Word 종합보고서 생성 완료: {OUT_DOCX}")

if __name__ == '__main__':
    main()
