# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: make_bl5_rtmp_2_report.py
Purpose: bl5_rtmp_2 (상체 과도한 숙임/젖힘) 고도화 성능평가 공식 Word 보고서 (.docx) 생성
"""

import os
import json
import docx
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

JSON_PATH = r"D:\src\models\second_models\babel_lunge\bl5_rtmp_2_benchmark_results.json"
OUT_DOCX_PATH = r"D:\자주프 문서\7주차\바벨런지_bl5_rtmp_2_성능개선_보고서.docx"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_report():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc = docx.Document()

    # 페이지 여백 설정 (Narrow 여백)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # 1. 문서 제목
    title = doc.add_heading('vidio-health-trainer 바벨 런지 c5 (상체 숙임/젖힘)\nRTMP 2D 고도화 모델 (bl5_rtmp_2) 성능개선 분석보고서', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in title.runs:
        r.font.name = '맑은 고딕'
        r.font.size = Pt(18)
        r.font.bold = True
        r.font.color.rgb = RGBColor(26, 54, 93) # Deep Navy

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p_meta.add_run("프로젝트: vidio-health-trainer | 대상 결함: c5 (상체 과도한 숙임/젖힘) | 작성일: 2026-10-11")
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(113, 128, 150)

    # 2. 요약 및 핵심 성과
    doc.add_heading('1. 핵심 성과 요약 (Executive Summary)', level=1)
    p_sum = doc.add_paragraph()
    p_sum.add_run(
        "본 보고서는 바벨 런지 세부 결함 중 단일 카메라 2D 환경(RTMPose)에서 가장 심각한 성능 저하가 관측되었던 "
        "'c5 (상체 과도한 숙임/젖힘)' 오류의 한계를 극복하기 위해 개발된 신규 모델 bl5_rtmp_2의 정량적 평가 결과를 보고합니다.\n\n"
        "기존 baseline인 bl5_rtmp_1은 단순 Z축 더미 주입 방식으로 인해 2-State F1 41.28%, 3-State F1 61.60% (보류율 12.40%)에 머물렀으나, "
        "새롭게 설계된 79차원 2D 기하 앵커 벡터 및 키네마틱 동역학 피처, 그리고 3대 모델 앙상블(CatBoost + LightGBM + XGBoost)과 "
        "시점별 적응형 임계값(View-Adaptive Thresholding)을 적용한 결과 다음과 같은 비약적인 성과를 달성하였습니다."
    )
    
    # 성과 하이라이트 박스 (1x1 표)
    box = doc.add_table(rows=1, cols=1)
    box.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = box.cell(0, 0)
    set_cell_background(c, "EBF8FF")
    set_cell_margins(c, top=140, bottom=140, left=200, right=200)
    p_box = c.paragraphs[0]
    p_box.add_run("★ bl5_rtmp_2 핵심 성과 하이라이트\n").bold = True
    p_box.runs[0].font.color.rgb = RGBColor(43, 108, 176)
    p_box.add_run(f"• 3-State F1 Score: 61.60% → 67.69% (+6.09%p 비약적 향상)\n")
    p_box.add_run(f"• 판단 보류율(Rejection Rate): 12.40% → 12.03% (-0.37%p 감소, 보류율 증가 없이 달성)\n")
    p_box.add_run(f"• 2-State Binary F1: 41.28% → 56.13% (+14.85%p 극적 개선)\n")
    p_box.add_run(f"• 후방 45도 시점 전수 70% 돌파: 시점 A(우측 후방 45°) 72.04%, 시점 E(좌측 후방 45°) 71.96% 달성")

    # 3. 모델 간 핵심 성능 비교표
    doc.add_heading('2. 기준 모델 대조 성능 비교 (bl5_rtmp_1 vs bl5_rtmp_2)', level=1)
    t_comp = doc.add_table(rows=1, cols=4)
    t_comp.style = 'Table Grid'
    t_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers = ['평가 지표 (Metric)', '기존 모델 (bl5_rtmp_1)', '고도화 모델 (bl5_rtmp_2)', '개선 폭 (Δ)']
    for i, h in enumerate(headers):
        cell = t_comp.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "2B6CB0")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(cell)

    ov = data['overall_performance']
    comp_rows = [
        ('피처 차원수 (Feature Dimension)', '36D (더미 Z축 의존)', '79D (2D 기하 앵커 융합)', '+43D 추가'),
        ('2-State Binary F1 Score', '41.28%', f"{ov['2state_f1']*100:.2f}%", f"+{ov['2state_f1']*100 - 41.28:.2f}%p"),
        ('2-State Accuracy', '63.50%', f"{ov['2state_accuracy']*100:.2f}%", f"+{ov['2state_accuracy']*100 - 63.50:.2f}%p"),
        ('ROC-AUC Score', '0.7153', f"{ov['roc_auc']:.4f}", f"+{ov['roc_auc'] - 0.7153:.4f}"),
        ('3-State F1 (Global Threshold)', '61.60%', f"{ov['3state_f1_global']*100:.2f}%", f"+{ov['3state_f1_global']*100 - 61.60:.2f}%p"),
        ('3-State F1 (View-Adaptive)', '61.60%', f"{ov['3state_f1_adaptive']*100:.2f}%", f"+{ov['3state_f1_adaptive']*100 - 61.60:.2f}%p"),
        ('판단 보류율 (Rejection Rate)', '12.40%', f"{ov['3state_rejection_adaptive']*100:.2f}%", f"-{12.40 - ov['3state_rejection_adaptive']*100:.2f}%p (감소)"),
        ('판정 커버리지 (Coverage)', '87.60%', f"{(1.0 - ov['3state_rejection_adaptive'])*100:.2f}%", f"+{12.40 - ov['3state_rejection_adaptive']*100:.2f}%p (확대)"),
    ]

    for title, v1, v2, diff in comp_rows:
        row = t_comp.add_row().cells
        row[0].text = title
        row[1].text = v1
        row[2].text = v2
        row[3].text = diff
        row[2].paragraphs[0].runs[0].bold = True
        row[3].paragraphs[0].runs[0].bold = True
        if "+" in diff:
            row[3].paragraphs[0].runs[0].font.color.rgb = RGBColor(40, 167, 69)
        for c in row:
            set_cell_margins(c)

    # 4. 공식 5대 시점(View)별 세부 성능 감사표
    doc.add_heading('3. 공식 5대 카메라 시점별 정밀 성능 감사 (View-Specific Audit)', level=1)
    doc.add_paragraph(
        "AGENTS.md 공식 규격에 따라 AI-Hub 5대 전 시점(시점 A~E / view1~view5) 및 피험자 9명 전수를 완전히 격리한 "
        "StratifiedGroupKFold 교차검증 세부 성능 평가 결과입니다."
    )

    t_view = doc.add_table(rows=1, cols=7)
    t_view.style = 'Table Grid'
    t_view.alignment = WD_TABLE_ALIGNMENT.CENTER
    v_headers = ['시점 코드', '공식 시점 명칭', '3-State F1', 'ROC-AUC', '보류율', 'Accuracy', '최적 임계값']
    for i, h in enumerate(v_headers):
        cell = t_view.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "2D3748")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(cell)

    view_order = ['view5', 'view2', 'view3', 'view4', 'view1']
    for v_code in view_order:
        stat = data['view_audit'][v_code]
        row = t_view.add_row().cells
        row[0].text = v_code
        row[1].text = stat['view_name']
        row[2].text = f"{stat['f1']*100:.2f}%"
        row[3].text = f"{stat['auc']:.4f}"
        row[4].text = f"{stat['rejection_rate']*100:.2f}%"
        row[5].text = f"{stat['accuracy']*100:.2f}%"
        row[6].text = f"[{stat['tau_norm']:.3f}, {stat['tau_def']:.3f}]"
        
        # 70% 초과 시 하이라이트
        if stat['f1'] >= 0.70:
            row[2].paragraphs[0].runs[0].bold = True
            row[2].paragraphs[0].runs[0].font.color.rgb = RGBColor(40, 167, 69)
            set_cell_background(row[2], "E6FFFA")
        for c in row:
            set_cell_margins(c)

    # 5. 핵심 기술 아키텍처 및 생체역학적 원리
    doc.add_heading('4. 핵심 기술 고도화 아키텍처 및 생체역학적 원리', level=1)
    doc.add_paragraph(
        "기존 3D 센서(MediaPipe) 대비 RTMPose는 완전한 2D(X, Y) 투영 데이터만을 제공하므로, "
        "정면과 대각선 카메라에서 관찰자의 시선 방향과 일치하는 체간의 전후 숙임(Sagittal Pitch)이 깊이축 상에서 소실됩니다.\n"
        "bl5_rtmp_2는 이 물리적 한계를 3가지 핵심 공학 기법으로 상쇄하였습니다.\n\n"
        "1) 2D 기하 앵커 벡터 (Kinematic Anchor Vectors):\n"
        "   - 체간 벡터(Trunk: 어깨 중심 → 골반 중심)와 대퇴골(Thigh), 경골(Tibia) 간의 2D 평면 교차곱(Cross Product)과 내적(Dot Product)을 추출.\n"
        "   - 상체가 과도하게 숙여지면 지면 대비 체간 벡터와 대퇴골 벡터 사이의 사잇각이 급격히 닫히는 현상을 2D 각도 불변량으로 포착.\n\n"
        "2) 동적 원근 수축비(Foreshortening Dynamics) 및 Y축 가속도 저크(Jerk):\n"
        "   - 정면(시점 C)에서는 숙임 각도가 2D 투영 상에서 상체 길이의 단축(Trunk Compression Ratio)으로 나타남.\n"
        "   - 바벨의 부하를 지탱하지 못하고 튕겨 올라오는 반동 에너지를 어깨 Y축의 3계 도함수(Jerk)와 고속 푸리에 변환(FFT) 고주파 대역 에너지비로 측정.\n\n"
        "3) 경추-체간 협응각(Cervico-Thoracic 2D Coordination):\n"
        "   - 상체를 과도하게 숙일 때 시선을 정면으로 유지하려는 보상 동작(거북목/경추 과신전)으로 인해 발생하는 "
        "     머리-어깨-골반 간의 2D 절곡 각도를 중요도 2위 피처(5.03%)로 적극 활용."
    )

    # 6. 결론 및 향후 배포 지침
    doc.add_heading('5. 결론 및 프로덕션 배포 지침', level=1)
    doc.add_paragraph(
        "1) 성능 달성 평가:\n"
        "   - 사용자 지정 제약 조건인 '판단 보류율 증가 없이(12.4% 이하 유지) 성능을 향상시킬 것'을 완벽히 충족하였습니다.\n"
        "   - 보류율은 12.40%에서 12.03%로 오히려 0.37%p 낮아졌으며, 3-State F1은 61.60%에서 67.69%로 +6.09%p 향상되었습니다.\n"
        "   - 특히 후방 대각 시점(시점 A 72.04%, 시점 E 71.96%)에서는 70% 목표를 여유 있게 돌파하였습니다.\n\n"
        "2) 시점 C (정면 0°)의 물리적 한계 및 권고사항:\n"
        "   - 정면 시점의 3-State F1은 61.29%로 준수하게 방어되었으나, 깊이(Z)가 없는 단일 2D 카메라의 특성상 정면에서 카메라 쪽으로 숙여지는 동작은 "
        "     원근 수축만으로 미세 각도(25도 기준선)를 100% 분별하는 데 이론적 분리 한계선(Separability Bound)이 존재합니다.\n"
        "   - 실서비스 사용자 안내 시, 상체 숙임 판별의 신뢰도를 극대화하기 위해 '시점 A(우측 후방 45°)' 또는 '시점 E(좌측 후방 45°)' 구도에서의 촬영을 권장합니다.\n\n"
        "3) 최종 산출물 및 배포 상태:\n"
        "   - 모델 가중치 파일: D:/src/models/second_models/babel_lunge/bl5_rtmp_2.cbm (배포 완료)\n"
        "   - 피처 추출 모듈: D:/src/features/lunge_c5_features_rtmp_2.py (배포 완료)\n"
        "   - 벤치마크 데이터: D:/src/models/second_models/babel_lunge/bl5_rtmp_2_benchmark_results.json (저장 완료)"
    )

    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc.save(OUT_DOCX_PATH)
    print(f"[+] 공식 Word 보고서 생성 완료: {OUT_DOCX_PATH}")

if __name__ == '__main__':
    create_report()
