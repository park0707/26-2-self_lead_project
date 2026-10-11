# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 런지 (Barbell Lunge)
Module: make_bl5_rtmp_3_report.py
Purpose: bl5_rtmp_3 (상체 과도한 숙임/젖힘) 초최적화 모델 공식 Word 보고서 (.docx) 생성
         - rtmp_1 vs rtmp_2 vs rtmp_3 3단계 계보 종합 대조
         - 다중공선성 및 저해 피처 정제 결과 상세 보고
         - 공식 5대 시점(시점 A~E) 세부 감사표 수록
"""

import os
import json
import docx
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

JSON_PATH = r"D:\src\models\second_models\babel_lunge\bl5_rtmp_3_benchmark_results.json"
OUT_DOCX_PATH = r"D:\자주프 문서\7주차\바벨런지_bl5_rtmp_3_성능비교_분석보고서.docx"

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
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON not found at {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc = docx.Document()

    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # 1. 문서 제목
    title = doc.add_heading('vidio-health-trainer 바벨 런지 c5 (상체 숙임/젖힘)\nRTMP 3단계 초최적화 (bl5_rtmp_3) 종합 성능비교 분석보고서', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in title.runs:
        r.font.name = '맑은 고딕'
        r.font.size = Pt(17)
        r.font.bold = True
        r.font.color.rgb = RGBColor(26, 54, 93)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p_meta.add_run("프로젝트: vidio-health-trainer | 대상 결함: c5 (상체 과도한 숙임/젖힘) | 작성일: 2026-10-11")
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(113, 128, 150)

    # 2. 요약 및 핵심 성과
    doc.add_heading('1. 핵심 성과 요약 (Executive Summary)', level=1)
    p_sum = doc.add_paragraph()
    p_sum.add_run(
        "본 보고서는 바벨 런지의 5대 세부 오류 중 2D 비전 단일 카메라 환경에서 추정 난도가 가장 높은 "
        "'c5 (상체 과도한 숙임/젖힘)' 판별을 위해, 피처 저해 요인 및 다중공선성(Multicollinearity)을 전수 정제하고 "
        "하이퍼파라미터 최적화(HPO)를 단행한 최종 고도화 모델 bl5_rtmp_3의 성능 평가 결과를 보고합니다.\n\n"
        "이전 버전인 bl5_rtmp_2(79D)에서 관측되었던 Z축 소실 무효 피처(분산 0.0) 4개와 상관계수 r=1.0의 완전 중복 피처 2개를 제거하여 "
        "총 73차원의 엄선된 피처셋을 구축하였으며, CatBoost(depth=7, L2=6.0, Scale_pos_weight=1.3)를 필두로 한 "
        "3대 트리 앙상블과 시점별 적응형 임계값 최적화를 결합하여 역대 최고 수준의 성적을 달성하였습니다."
    )

    # 성과 하이라이트 박스
    box = doc.add_table(rows=1, cols=1)
    box.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = box.cell(0, 0)
    set_cell_background(c, "EBF8FF")
    set_cell_margins(c, top=140, bottom=140, left=200, right=200)
    p_box = c.paragraphs[0]
    p_box.add_run("★ bl5_rtmp_3 핵심 성과 지표 요약\n").bold = True
    p_box.runs[0].font.color.rgb = RGBColor(43, 108, 176)
    p_box.add_run("• 3-State F1 Score: 61.60%(rtmp_1) → 67.69%(rtmp_2) → 67.86%(rtmp_3) (역대 최고치 경신)\n")
    p_box.add_run("• 판단 보류율(Rejection Rate): 12.40%(rtmp_1) → 12.03%(rtmp_2) → 11.71%(rtmp_3) (지속적 감소 성공)\n")
    p_box.add_run("• 2-State Binary F1: 41.28%(rtmp_1) → 56.13%(rtmp_2) → 59.58%(rtmp_3) (+18.30%p 대도약)\n")
    p_box.add_run("• 후방 45도 시점 안정적 72% 돌파: 시점 E(좌측 후방 45°) 72.85%, 시점 A(우측 후방 45°) 72.83% 달성")

    # 3. 3단계 모델 계보 비교표
    doc.add_heading('2. RTMP 3단계 모델 계보 대조 비교 (rtmp_1 vs rtmp_2 vs rtmp_3)', level=1)
    t_comp = doc.add_table(rows=1, cols=5)
    t_comp.style = 'Table Grid'
    t_comp.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ['평가 지표 (Metric)', '초기 모델 (rtmp_1)', '2차 고도화 (rtmp_2)', '최종 초최적화 (rtmp_3)', '누적 개선폭 (Δ)']
    for i, h in enumerate(headers):
        cell = t_comp.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "2B6CB0")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(cell)

    b = data['benchmark_comparison']
    m1, m2, m3 = b['bl5_rtmp_1'], b['bl5_rtmp_2'], b['bl5_rtmp_3']

    comp_rows = [
        ('피처 엔지니어링', '36D (더미 Z축)', '79D (기하앵커+상호작용)', '73D (노이즈/중복 6개 정제)', '최적 정제 피처셋'),
        ('모델 아키텍처', '단일 CatBoost (기본값)', '3대 모델 앙상블', 'HPO 튜닝 3대 앙상블', 'CatBoost d7/L2_6 튜닝'),
        ('2-State Binary F1', f"{m1['2state_f1']*100:.2f}%", f"{m2['2state_f1']*100:.2f}%", f"{m3['2state_f1']*100:.2f}%", f"+{m3['2state_f1']*100 - m1['2state_f1']*100:.2f}%p"),
        ('ROC-AUC Score', f"{m1['auc']:.4f}", f"{m2['auc']:.4f}", f"{m3['auc']:.4f}", f"+{m3['auc'] - m1['auc']:.4f}"),
        ('3-State F1 (최적 임계값)', f"{m1['3state_f1']*100:.2f}%", f"{m2['3state_f1']*100:.2f}%", f"{m3['3state_f1']*100:.2f}%", f"+{m3['3state_f1']*100 - m1['3state_f1']*100:.2f}%p"),
        ('판단 보류율 (Rejection Rate)', f"{m1['rejection']*100:.2f}%", f"{m2['rejection']*100:.2f}%", f"{m3['rejection']*100:.2f}%", f"-{m1['rejection']*100 - m3['rejection']*100:.2f}%p (감소)"),
        ('판정 커버리지 (Coverage)', f"{(1-m1['rejection'])*100:.2f}%", f"{(1-m2['rejection'])*100:.2f}%", f"{(1-m3['rejection'])*100:.2f}%", f"+{m1['rejection']*100 - m3['rejection']*100:.2f}%p (확대)"),
    ]

    for title, v1, v2, v3, diff in comp_rows:
        row = t_comp.add_row().cells
        row[0].text = title
        row[1].text = v1
        row[2].text = v2
        row[3].text = v3
        row[4].text = diff
        row[3].paragraphs[0].runs[0].bold = True
        row[4].paragraphs[0].runs[0].bold = True
        if "+" in diff:
            row[4].paragraphs[0].runs[0].font.color.rgb = RGBColor(40, 167, 69)
        for c in row:
            set_cell_margins(c)

    # 4. 공식 5대 시점별 세부 성적 감사표
    doc.add_heading('3. 공식 5대 카메라 시점별 최종 성적표 (View-Specific Audit)', level=1)
    doc.add_paragraph(
        "AGENTS.md 공식 5대 시점 명명 규격(시점 A~E / view1~5) 및 피험자 9명 전수를 완전히 격리한 "
        "5-Fold StratifiedGroupKFold 교차검증 정밀 측정 결과입니다."
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

        if stat['f1'] >= 0.70:
            row[2].paragraphs[0].runs[0].bold = True
            row[2].paragraphs[0].runs[0].font.color.rgb = RGBColor(40, 167, 69)
            set_cell_background(row[2], "E6FFFA")
        for c in row:
            set_cell_margins(c)

    # 5. 피처 정제 및 다중공선성 분석
    doc.add_heading('4. 저해 피처 정제 및 다중공선성 전수 감사 분석', level=1)
    doc.add_paragraph(
        "본 고도화 과정에서는 피처 간 상호 상관행렬(Correlation Matrix) 및 분산 감사를 진행하여 다음과 같이 6개의 불량 피처를 적출 및 제거하였습니다.\n\n"
        "1) 3D 깊이축 소실로 인한 무효 피처 (Zero Variance: 4개 적출)\n"
        "   - lunge_pelvic_tilt_sagittal: Z축 좌표가 0이 되면서 골반 시상면 경사가 상수화됨.\n"
        "   - lunge_trunk_knee_moment_arm: 3D 모멘트암 계산이 불가능하여 분산 0.0 기록.\n"
        "   - lunge_virtual_sagittal_pitch_3d: 3D 가상 시상면 투영 피처로 2D 환경에서 동작 불능.\n"
        "   - lunge_head_torso_hyperext_sync: 머리-체간 3D 과신전 동기화 피처 소실.\n"
        "   → 위 4개 피처는 트리 분기 시 순수 노이즈로 작용하여 과적합을 유발하므로 완전 제거함.\n\n"
        "2) 완전 중복 피처 (Exact Collinearity r=1.0000: 2개 적출)\n"
        "   - lunge_pelvis_spine_decoupling_angle: lunge_tibia_torso_parallelism과 상관계수 r=1.0000으로 완전 중복.\n"
        "   - v_neck_pelvis_y: v_trunk_y와 Y축 벡터 방향이 100% 동일하여 r=1.0000 기록.\n"
        "   → 중복 분기를 방지하고 단일 결정 트리의 정보 이득(Information Gain)을 집중시키기 위해 2개 피처를 제거함."
    )

    # 6. 하이퍼파라미터 최적화 (HPO) 결과
    doc.add_heading('5. 하이퍼파라미터 최적화(HPO) 명세', level=1)
    doc.add_paragraph(
        "CatBoost, LightGBM, XGBoost 3개 모델에 대해 5-Fold StratifiedGroupKFold 교차검증 상에서 체계적인 그리드 서치를 수행하여 "
        "다음과 같이 최적 하이퍼파라미터를 도출 및 고정하였습니다.\n\n"
        "• CatBoost (앙상블 가중치 50%):\n"
        "   - Depth: 6 → 7 (복잡한 2D 비선형 기하 관계 학습을 위해 1단계 상향)\n"
        "   - L2 Leaf Reg: 4.0 → 6.0 (깊어진 트리의 일반화 성능 방어를 위해 규제 강화)\n"
        "   - Scale Pos Weight: 1.2 → 1.3 (결함 클래스 재현율 보강)\n"
        "   - Iterations: 450, Learning Rate: 0.03\n\n"
        "• LightGBM (앙상블 가중치 30%):\n"
        "   - Num Leaves: 35, Max Depth: 6, Subsample: 0.8, Colsample By Tree: 0.8, Scale Pos Weight: 1.3\n\n"
        "• XGBoost (앙상블 가중치 20%):\n"
        "   - Max Depth: 5, Subsample: 0.8, Colsample By Tree: 0.8, Scale Pos Weight: 1.3"
    )

    # 7. 결론 및 배포 요약
    doc.add_heading('6. 결론 및 최종 배포 명세', level=1)
    doc.add_paragraph(
        "1) 목표 달성 검증:\n"
        "   - 사용자 지시 사항인 '파라미터 최적화', '성능 저해/상관관계 높은 피처 최적화', '보류율 증가 없이 성능 향상'을 100% 완수하였습니다.\n"
        "   - 3-State F1은 67.86%로 역대 최고 기록을 경신하였으며, 보류율은 11.71%로 오히려 기존(12.40%) 대비 0.69%p 감소하여 판정 신뢰 영역이 확장되었습니다.\n\n"
        "2) 프로덕션 배포 파일:\n"
        "   - 최종 프로덕션 모델: D:/src/models/second_models/babel_lunge/bl5_rtmp_3.cbm\n"
        "   - 73D 피처 추출 모듈: D:/src/features/lunge_c5_features_rtmp_3.py\n"
        "   - 벤치마크 결과 JSON: D:/src/models/second_models/babel_lunge/bl5_rtmp_3_benchmark_results.json\n"
        "   - 공식 Word 성능보고서: D:/자주프 문서/7주차/바벨런지_bl5_rtmp_3_성능비교_분석보고서.docx"
    )

    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc.save(OUT_DOCX_PATH)
    print(f"[+] 최종 공식 Word 보고서 생성 완료: {OUT_DOCX_PATH}")

if __name__ == '__main__':
    create_report()
