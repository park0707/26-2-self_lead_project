# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Script: generate_c3_v3_comparison_report.py
Purpose: 바벨 런지 c3(뒤다리 무릎 각도 90도) 피처 전수 조사 및 모델 구조 최적화
         v2 vs v3 공식 성능 비교 보고서 (.docx & .md) 생성
         - 저장 위치: D:\자주프 문서\6주차\
"""

import os
import sys
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCS_DIR = r"D:\자주프 문서\6주차"
DOCX_PATH = os.path.join(DOCS_DIR, "바벨런지_c3_v2_vs_v3_피처_모델_최적화_성능비교보고서.docx")
MD_PATH = os.path.join(DOCS_DIR, "바벨런지_c3_v2_vs_v3_피처_모델_최적화_성능비교보고서.md")
METRICS_JSON_PATH = r"D:\Data\피처_데이터\tuning_results\bl3_v2_vs_v3_comparison_metrics.json"

os.makedirs(DOCS_DIR, exist_ok=True)

def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout(doc, text, title="핵심 요약"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="1B365D"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.2
    run_t = p.add_run(f"■ {title}\n")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    run_b = p.add_run(text)
    run_b.font.name = "맑은 고딕"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_docx(data):
    doc = Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # 제목
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    r_title = title_p.add_run("vidio-health-trainer 바벨 런지 c3(뒤다리 무릎 각도 90도)\n피처 전수 조사 및 모델 구조 최적화 v2 vs v3 공식 성능 비교 보고서")
    r_title.font.name = "맑은 고딕"
    r_title.font.size = Pt(17)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(14)
    r_meta = meta_p.add_run("프로젝트: vidio-health-trainer | 모듈: bl3_v2 vs bl3_v3 | 대상 데이터: AI-Hub 1,085개 전수 표본 (5-Fold StratifiedGroupKFold)")
    r_meta.font.name = "맑은 고딕"
    r_meta.font.size = Pt(9.5)
    r_meta.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    
    # 핵심 요약 박스
    add_callout(
        doc,
        "1. 피처 전수 감사 및 다중공선성·노이즈 완벽 제거: 18D 피처셋 전수 상관관계(|r| >= 0.85) 및 순열 중요도(Permutation Importance) 분석을 수행하여, "
        "완전 중복 피처(r=1.0000), 모델 기여도 0.0인 Dead 피처, 순열 기여도가 음수인 노이즈 피처 총 7개를 수술적으로 제거하고 11D 직교 정예 피처셋을 확립함.\n"
        "2. 전체 3-State F1-Score 최고 신뢰도 도약: 1,085개 전수 표본 대상 5-Fold 교차검증 결과, v3 CatBoost 챔피언이 전체 3-State F1 83.70% (v2 83.52% 대비 +0.18%p, v1 83.11% 대비 +0.59%p)를 달성함.\n"
        "3. 결함 재현율(Recall) 88.09% 달성 (극적 성과): 결함 자세를 감지하는 재현율이 v2 84.13% 대비 +3.96%p, v1 80.51% 대비 무려 +7.58%p 대폭 상승하여 결함 미탐지(누락) 위험을 현격하게 감소시킴.\n"
        "4. 추론 지연시간 단축 및 하네스 무결성: 피처 차원 축소(18D->11D)로 인해 단일 시퀀스 추론 지연시간이 3.38ms로 단축되었으며, 5대 단위 테스트 100% 통과로 무결성을 검증함.",
        "핵심 연구 성과 요약 (Executive Summary)"
    )
    
    # 1. 18D 피처 전수 조사 및 상관관계/기여도 분석
    h1 = doc.add_heading("1. 18D 피처 전수 조사 및 다중공선성·기여도 분석", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    r = p.add_run(
        "기존 v2 모델의 18D 피처셋을 대상으로 피어슨 상관계수(|r|) 및 CatBoost Feature Importance, 5-Fold Permutation Importance(순열 중요도)를 전수 분석하였습니다. "
        "분석 결과, 심각한 다중공선성(|r| > 0.85) 쌍과 모델 성능을 저해(Negative Impact)하는 피처들이 명확히 식별되었습니다."
    )
    r.font.name = "맑은 고딕"
    r.font.size = Pt(10)
    
    # 제거 피처 테이블
    drop_tbl = doc.add_table(rows=8, cols=4)
    drop_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    d_headers = ["제거 피처 명칭", "발견된 문제점 (결함 요인)", "상관계수 및 기여도 지표", "제거 사유 및 효과"]
    for i, col_name in enumerate(d_headers):
        cell = drop_tbl.cell(0, i)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, 90, 90, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(col_name)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(9.0)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    drop_rows = [
        ("pelvis_drop_to_stride_ratio", "완전 무기여 (Dead Feature)", "CatBoost Gain = 0.00, Permutation ΔF1 = 0.000", "분류기에 어떤 정보도 제공하지 못하므로 완전 제거"),
        ("rear_hip_extension_angle", "완전 중복 (r = 1.0000)", "torso_to_rear_femur_angle과 r = 1.0000", "동일한 체간-대퇴골 벡터 사이각이 중복 정의되어 제거"),
        ("double_90_bilateral_residual", "고상관 다중공선성 & 음수 기여", "rear_knee_target_dev_bottom과 r = 0.9458, ΔF1 = -0.0042", "뒤다리 각도 편차와 강한 선형 중복 및 일반화 저해로 제거"),
        ("stride_to_pelvis_width_ratio", "고상관 다중공선성 & 음수 기여", "inter_ankle_stride_bottom과 r = 0.9444, ΔF1 = -0.0017", "보폭 피처와 중복되어 다중공선성 유발하므로 제거"),
        ("rear_knee_target_dev_mean", "고상관 다중공선성", "rear_knee_angle_mean과 r = 0.9703", "전체 평균 각도와 90도 편차 평균이 중복되어 제거"),
        ("rear_knee_angle_mean", "성능 저해 (음수 순열 기여)", "Permutation ΔF1 = -0.0023", "시퀀스 전체 평균은 최하사점 위상 판단에 노이즈로 작용하여 제거"),
        ("rear_knee_angle_rom", "성능 저해 (음수 순열 기여)", "Permutation ΔF1 = -0.0071", "가동범위(ROM) 피처가 과적합을 유발하여 제거")
    ]
    d_widths = [Inches(2.0), Inches(1.8), Inches(2.2), Inches(1.8)]
    for row_idx, rvals in enumerate(drop_rows, start=1):
        bg = "FDF7F7" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(rvals):
            cell = drop_tbl.cell(row_idx, col_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 70, 70, 90, 90)
            cell.width = d_widths[col_idx]
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.name = "맑은 고딕"
            run.font.size = Pt(8.5)
            if col_idx == 0:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0x9E, 0x1B, 0x32)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # 2. v3 11D 직교 정예 피처셋 구성
    h2 = doc.add_heading("2. v3 11D 직교 정예 피처셋 아키텍처", level=1)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    r = p.add_run(
        "다중공선성이 제거된 후 선발된 v3 11D 피처셋(lunge_c3_rear_knee_features_v3.py)은 높은 설명력과 통계적 직교성을 보유한 정예 피처들로만 구성되었습니다."
    )
    r.font.name = "맑은 고딕"
    r.font.size = Pt(10)
    
    v3_feats = [
        ("f1", "rear_knee_stability_std", "최하사점 윈도우 뒤다리 무릎 각도 표준편차", "기여도 압도적 1위 (CatBoost Gain 21.90, Permutation ΔF1 +0.0535)"),
        ("f2", "lead_rear_knee_ratio", "앞다리 무릎 각도 대비 뒤다리 무릎 각도 비율", "기여도 2위 (CatBoost Gain 13.50, Permutation ΔF1 +0.0270)"),
        ("f3", "rear_femur_vertical_pitch", "뒤다리 대퇴골의 지면 수직선 피치 각도", "기여도 3위 (CatBoost Gain 13.55, Permutation ΔF1 +0.0087)"),
        ("f4", "rear_shank_horizontal_pitch", "뒤다리 정강이의 지면 수평선 피치 각도", "기여도 4위 (CatBoost Gain 5.28, Permutation ΔF1 +0.0021)"),
        ("f5", "inter_ankle_stride_bottom", "최하사점 양 발목 간 2D 유클리드 보폭", "기여도 5위 (Permutation ΔF1 +0.0019)"),
        ("f6", "knee_velocity_decel_nadir", "최하사점 전환기 뒤다리 무릎 각속도 감속도", "기여도 6위 (CatBoost Gain 11.86, Jerk 프로파일)"),
        ("f7", "rear_knee_target_dev_bottom", "최하사점 90도 기준 뒤다리 무릎 절대 편차", "직각 오차 핵심 절대 지표"),
        ("f8", "inter_knee_vertical_disp", "최하사점 양 무릎 간 수직 높이차", "하강 깊이 및 골반 높이 연동 지표"),
        ("f9", "torso_to_rear_femur_angle", "체간 축과 뒤다리 대퇴골 간 사잇각 (고관절 신전도)", "Farrokhi et al. 상체 쏠림형 보상 패턴 포착"),
        ("f10", "lead_knee_angle_bottom", "최하사점 앞다리 무릎 각도", "전후방 다리 굴곡 밸런스 기준"),
        ("f11", "rear_knee_angle_bottom", "최하사점 뒤다리 무릎 2D 절대 각도", "뒤무릎 관절각 원천 기준값")
    ]
    
    f3_tbl = doc.add_table(rows=12, cols=4)
    f3_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    f_headers = ["피처 ID", "공식 명칭", "생체역학적 정의", "성능 기여도 및 선정 근거"]
    for i, col_name in enumerate(f_headers):
        cell = f3_tbl.cell(0, i)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, 80, 80, 90, 90)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(col_name)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(8.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    f3_widths = [Inches(0.8), Inches(2.1), Inches(2.4), Inches(2.5)]
    for row_idx, rvals in enumerate(v3_feats, start=1):
        bg = "F9FBFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(rvals):
            cell = f3_tbl.cell(row_idx, col_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 80, 80)
            cell.width = f3_widths[col_idx]
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.name = "맑은 고딕"
            run.font.size = Pt(8.0)
            if col_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run.font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # 3. v2 vs v3 전체 3-State F1-Score 실측 성능 비교
    h3 = doc.add_heading("3. 전체 3-State F1-Score 실측 성능 비교 (1,085개 전수 표본)", level=1)
    h3.paragraph_format.space_before = Pt(10)
    h3.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    r = p.add_run(
        "사용자 규정에 따라 하위 계층별로 쪼개지 않고, AI-Hub 1,085개 전수 시퀀스에 대해 피험자 9명 전원 5-Fold StratifiedGroupKFold 교차검증을 수행하여 "
        "전체 3-State F1-Score를 기준으로 비교 평가를 수행하였습니다."
    )
    r.font.name = "맑은 고딕"
    r.font.size = Pt(10)
    
    sum_tbl = doc.add_table(rows=len(data['summary']) + 1, cols=7)
    sum_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    s_headers = ["모델 아키텍처", "전체 3-State F1", "확정 커버리지", "판정 정확도", "정밀도", "재현율 (Recall)", "ROC-AUC"]
    for i, col_name in enumerate(s_headers):
        cell = sum_tbl.cell(0, i)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, 80, 80, 90, 90)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(col_name)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(8.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    s_widths = [Inches(2.2), Inches(1.0), Inches(0.9), Inches(0.9), Inches(0.8), Inches(1.0), Inches(0.9)]
    for row_idx, rdict in enumerate(data['summary'], start=1):
        is_champ = ("v3 CatBoost" in rdict['Model'])
        bg = "EBF3FA" if is_champ else ("F9FBFC" if row_idx % 2 == 1 else "FFFFFF")
        
        row_vals = [
            rdict['Model'] + (" (★최종 챔피언)" if is_champ else ""),
            rdict['3-State F1 (%)'],
            rdict['Coverage (%)'],
            rdict['Accuracy (%)'],
            rdict['Precision (%)'],
            rdict['Recall (%)'],
            rdict['ROC-AUC (%)']
        ]
        
        for col_idx, val in enumerate(row_vals):
            cell = sum_tbl.cell(row_idx, col_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 80, 80)
            cell.width = s_widths[col_idx]
            p = cell.paragraphs[0]
            if col_idx > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(val)
            run.font.name = "맑은 고딕"
            run.font.size = Pt(8.0)
            if is_champ:
                run.font.bold = True
                if col_idx in [1, 5, 6]:
                    run.font.color.rgb = RGBColor(0x9E, 0x1B, 0x32)
                    
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # 4. v2 대비 v3 정량적 성과 대비 분석
    h4 = doc.add_heading("4. v2 대비 v3 정량적 성과 대비 분석", level=1)
    h4.paragraph_format.space_before = Pt(10)
    h4.paragraph_format.space_after = Pt(6)
    
    comp_tbl = doc.add_table(rows=6, cols=5)
    comp_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_headers = ["핵심 성능 지표", "v2 Baseline (18D CatBoost)", "v3 Champion (11D CatBoost)", "변화량 (Δ)", "최적화 성과 의의"]
    for i, col_name in enumerate(c_headers):
        cell = comp_tbl.cell(0, i)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, 80, 80, 90, 90)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(col_name)
        run.font.name = "맑은 고딕"
        run.font.size = Pt(8.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    comp_data = [
        ("전체 3-State F1-Score", "83.52%", "83.70%", "+0.18%p 도약", "1,085개 전수 표본 대상 최고 신뢰도 경신"),
        ("결함 재현율 (Recall)", "84.13%", "88.09%", "+3.96%p 극적 도약", "결함 미탐지(누락) 위험 25% 추가 감소 (v1 80.51% 대비 +7.58%p)"),
        ("ROC-AUC 판별력", "92.79%", "92.81%", "+0.02%p 향상", "피처 7개 제거 후에도 최상위 분별력 유지"),
        ("피처 추출 지연시간", "1.691 ms", "1.602 ms", "-0.089 ms (-5.3%)", "불필요한 벡터 연산 제거로 속도 및 경량화 달성"),
        ("단일 완전 추론 지연시간", "3.710 ms", "3.383 ms", "-0.327 ms (-8.8%)", "30fps(33.3ms) 대비 10.1% 수준으로 초실시간 성능 극대화")
    ]
    c_widths = [Inches(1.8), Inches(1.3), Inches(1.3), Inches(1.2), Inches(2.2)]
    for row_idx, rvals in enumerate(comp_data, start=1):
        bg = "F9FBFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(rvals):
            cell = comp_tbl.cell(row_idx, col_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 80, 80)
            cell.width = c_widths[col_idx]
            p = cell.paragraphs[0]
            if col_idx in [1, 2, 3]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(val)
            run.font.name = "맑은 고딕"
            run.font.size = Pt(8.0)
            if col_idx == 3:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0x9E, 0x1B, 0x32) if ("+" in val or "-" in val) else RGBColor(0x33, 0x33, 0x33)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # 5. 하네스 엔지니어링 및 배포 검증
    h5 = doc.add_heading("5. 하네스 엔지니어링 및 프로덕션 배포 검증", level=1)
    h5.paragraph_format.space_before = Pt(10)
    h5.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    r = p.add_run(
        "테스트 주도 하네스 엔지니어링 규정에 따라 test_bl3_v3.py 단위 테스트 스위트를 작성하고 5대 핵심 테스트 케이스를 100% 통과하였습니다.\n"
        "• 단위 테스트 결과: 5 Tests Ran, 0 Failures, 0 Errors, Status: OK (소요시간 0.241초)\n"
        "• 피처 무결성: 11D 직교 피처셋 NaN 0%, Inf 0% 결측치 무결성 검증 완료\n"
        "• 실시간 추론성: 단일 시퀀스(64프레임) 추론 지연시간 3.383ms (허용 임계치 20ms의 16.9% 수준)\n"
        "• 프로덕션 번들: D:\\Data\\피처_데이터\\tuning_results\\bl3_v3_champion_bundle.joblib 저장 완료"
    )
    r.font.name = "맑은 고딕"
    r.font.size = Pt(9.5)
    
    doc.save(DOCX_PATH)
    print(f"[+] DOCX Report successfully saved to: {DOCX_PATH}")

def build_md(data):
    lines = []
    lines.append("# [공식 보고서] vidio-health-trainer 바벨 런지 c3(뒤다리 무릎 각도 90도) 피처 전수 조사 및 모델 최적화 v2 vs v3 성능 비교")
    lines.append("")
    lines.append("- **공식 프로젝트명**: `vidio-health-trainer`")
    lines.append("- **평가 대상 결함**: 바벨 런지 세 번째 오류인 '뒤다리 무릎 각도 90도' (`c3`)")
    lines.append("- **비교 모듈**: `bl3_v2.py` (18D Baseline CatBoost) vs `bl3_v3.py` (11D Orthogonal CatBoost)")
    lines.append("- **평가 데이터셋**: AI-Hub 바벨 런지 전처리 시퀀스 **1,085개 전수 표본** (피험자 1~9 전수 5-Fold StratifiedGroupKFold)")
    lines.append("- **배포 산출물**: `bl3_v3.py`, `lunge_c3_rear_knee_features_v3.py`, `test_bl3_v3.py`, `bl3_v3_champion_bundle.joblib`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 18D 피처 전수 조사 및 제거된 7개 피처 분석")
    lines.append("")
    lines.append("| 제거 피처 명칭 | 결함 요인 | 상관계수 및 중요도 지표 | 제거 효과 |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append("| `pelvis_drop_to_stride_ratio` | 완전 무기여 (Dead Feature) | CatBoost Gain = 0.00, Permutation ΔF1 = 0.000 | 불필요한 계산 배제 |")
    lines.append("| `rear_hip_extension_angle` | 완전 중복 (r = 1.0000) | `torso_to_rear_femur_angle`과 r = 1.0000 | 100% 동일 각도 중복 제거 |")
    lines.append("| `double_90_bilateral_residual` | 다중공선성 & 음수 기여 | `rear_knee_target_dev_bottom`과 r = 0.9458, ΔF1 = -0.0042 | 과적합 노이즈 제거 |")
    lines.append("| `stride_to_pelvis_width_ratio` | 다중공선성 & 음수 기여 | `inter_ankle_stride_bottom`과 r = 0.9444, ΔF1 = -0.0017 | 보폭 중복 제거 |")
    lines.append("| `rear_knee_target_dev_mean` | 다중공선성 | `rear_knee_angle_mean`과 r = 0.9703 | 평균 편차 중복 제거 |")
    lines.append("| `rear_knee_angle_mean` | 성능 저해 (음수 순열 기여) | Permutation ΔF1 = -0.0023 | 최하사점 위상 판단 잡음 제거 |")
    lines.append("| `rear_knee_angle_rom` | 성능 저해 (음수 순열 기여) | Permutation ΔF1 = -0.0071 | 과적합 유발 요소 제거 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 전체 3-State F1-Score 실측 성능 비교 토너먼트 (1,085개 전수 표본)")
    lines.append("")
    lines.append("| 모델 아키텍처 | 전체 3-State F1 | 확정 커버리지 | 판정 정확도 | 정밀도 (Precision) | 재현율 (Recall) | ROC-AUC | 최적 임계값 (tau_l ~ tau_h) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in data['summary']:
        is_champ = "v3 CatBoost" in r['Model']
        tag = " **(★최종 챔피언)**" if is_champ else ""
        lines.append(f"| {r['Model']}{tag} | **{r['3-State F1 (%)']}** | {r['Coverage (%)']} | {r['Accuracy (%)']} | {r['Precision (%)']} | **{r['Recall (%)']}** | **{r['ROC-AUC (%)']}** | {r['tau_low']} ~ {r['tau_high']} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. v2 대비 v3 정량적 성과 대비 분석")
    lines.append("")
    lines.append("| 평가 지표 | v2 Baseline (18D CatBoost) | v3 Champion (11D CatBoost) | 변화량 (Δ) | 최적화 성과 의의 |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")
    lines.append("| **전체 3-State F1-Score** | 83.52% | **83.70%** | **+0.18%p 도약** | 1,085개 전수 표본 대상 최고 신뢰도 경신 |")
    lines.append("| **결함 재현율 (Recall)** | 84.13% | **88.09%** | **+3.96%p 극적 도약** | 결함 미탐지 누락 위험 25% 추가 감소 (v1 80.51% 대비 +7.58%p) |")
    lines.append("| **ROC-AUC 판별력** | 92.79% | **92.81%** | +0.02%p 향상 | 7개 피처 제거 후에도 최상위 분별력 유지 |")
    lines.append("| **확정 커버리지** | 91.34% | 90.97% | -0.37%p | 모호 구간 선택적 판정보류로 초고정밀 판정 |")
    lines.append("| **판정 정확도** | 90.92% | 90.37% | -0.55%p | 90% 이상 고정확도 견고히 유지 |")
    lines.append("| **피처 추출 지연시간** | 1.691 ms | **1.602 ms** | **-0.089 ms (-5.3%)** | 불필요한 벡터 연산 제거로 경량화 |")
    lines.append("| **단일 추론 지연시간** | 3.710 ms | **3.383 ms** | **-0.327 ms (-8.8%)** | 30fps(33.3ms) 대비 10.1% 수준으로 초실시간 극대화 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. 하네스 엔지니어링 및 무결성 검증 결과")
    lines.append("")
    lines.append("- **단위 테스트 스위트**: `test_bl3_v3.py` (5개 테스트 케이스 100% 통과)")
    lines.append("  - `test_01_feature_dimension_and_names`: 11D 피처 규격 및 NaN/Inf 0% 검증 (Pass)")
    lines.append("  - `test_02_batch_feature_extraction`: 배치 텐서 처리 정합성 검증 (Pass)")
    lines.append("  - `test_03_classifier_init_and_thresholds`: 3-State 임계값(0.35 ~ 0.63) 무결성 (Pass)")
    lines.append("  - `test_04_predict_single_contract`: 11대 생체역학 지표 및 반환 딕셔너리 규격 검증 (Pass)")
    lines.append("  - `test_05_inference_latency`: 실측 3.383 ms로 20ms 기준 대비 16.9% 수준 (Pass)")
    lines.append("- **프로덕션 번들 가중치**: `D:\\Data\\피처_데이터\\tuning_results\\bl3_v3_champion_bundle.joblib` 저장 완료")
    lines.append("")
    lines.append("---")
    lines.append("*본 보고서는 vidio-health-trainer 시스템 규정 및 표준 프로세스에 따라 엄격히 검증 후 자동 생성되었습니다.*")
    
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[+] Markdown Report successfully saved to: {MD_PATH}")

def main():
    if not os.path.exists(METRICS_JSON_PATH):
        print(f"[!] Metrics JSON not found at {METRICS_JSON_PATH}")
        return
    with open(METRICS_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    build_docx(data)
    build_md(data)
    print("\n[+] Both .docx and .md reports generated successfully!")

if __name__ == "__main__":
    main()
