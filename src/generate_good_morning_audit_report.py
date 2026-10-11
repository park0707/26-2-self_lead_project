# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_good_morning_audit_report.py
Purpose: 굿모닝 세부 오류 최종 모델 3-State F1 성능 및 데이터 누수 전수 검사 공식 Word 보고서 생성
"""

import os
import sys
import docx
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = '맑은 고딕'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        if level == 1:
            run.font.size = Pt(15)
            run.font.color.rgb = RGBColor(24, 76, 120)
            run.bold = True
        elif level == 2:
            run.font.size = Pt(13)
            run.font.color.rgb = RGBColor(41, 128, 185)
            run.bold = True
    return h

def add_p(doc, text, bold=False, color=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    run = p.add_run(text)
    run.font.name = '맑은 고딕'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    run.font.size = Pt(10.5)
    run.bold = bold
    if color:
        run.font.color.rgb = color
    return p

def main():
    doc = docx.Document()

    # Margin
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_after = Pt(4)
    title_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = title_p.add_run('vidio-health-trainer 공식 기술 감사 보고서\n')
    run_sub.font.name = '맑은 고딕'
    run_sub._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = RGBColor(120, 120, 120)
    run_sub.bold = True

    run_main = title_p.add_run('굿모닝(Good Morning) 세부 오류 최종 모델\n원천 데이터 3-State F1 성능 및 데이터 누수 전수 검사 보고서')
    run_main.font.name = '맑은 고딕'
    run_main._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    run_main.font.size = Pt(17)
    run_main.bold = True
    run_main.font.color.rgb = RGBColor(20, 50, 90)

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(18)
    meta_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    m_run = meta_p.add_run('작성일자: 2026년 10월 8일 | 프로젝트: vidio-health-trainer | 대상 운동: 09_굿모닝 (c1, c2, c3 전수)')
    m_run.font.name = '맑은 고딕'
    m_run._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
    m_run.font.size = Pt(9.5)
    m_run.font.color.rgb = RGBColor(100, 100, 100)

    # 1. 성능 현황
    add_styled_heading(doc, '1. 굿모닝 세부 오류 최종 모델 원천 데이터 3-State F1 종합 성능', level=1)
    add_p(doc, '바벨 굿모닝(09_굿모닝)의 3대 세부 오류(무릎 고정 위반 gm1, 시선 유지 오류 gm2, 척추 중립 위반 gm3) 최종 챔피언 모델 전수에 대한 원천 데이터 3-State 선택적 분류(Selective Classification) F1 성능 검사 결과입니다.')

    table1 = doc.add_table(rows=1, cols=7)
    table1.style = 'Table Grid'
    headers = ['세부 오류 코드', '결함 세부 명칭', '최종 모델명', '원천 데이터수', '보류율(Retake)', '커버리지', '3-State F1']
    hdr_row = table1.rows[0]
    for idx, h in enumerate(headers):
        cell = hdr_row.cells[idx]
        set_cell_background(cell, '1F497D')
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = '맑은 고딕'
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.bold = True

    rows_data1 = [
        ['c1 (gm1)', '무릎 구부린 채 고정 위반 (D02)', 'gm1_v3 (CatBoost)', '678 seq', '9.59%', '90.41%', '79.80% (달성)'],
        ['c2 (gm2)', '시선 방향 유지 오류 (D10)', 'gm2_v1 (GBDT Ensemble)', '1,020 seq', '9.71%', '90.29%', '77.33% (달성)'],
        ['c3 (gm3)', '척추의 중립 오류 / 등 굽힘 (D14)', 'gm3_v3 (CatBoost+SpinePose)', '1,020 seq', '9.22%', '90.78%', '88.41% (달성)']
    ]

    for row_vals in rows_data1:
        row = table1.add_row()
        for idx, v in enumerate(row_vals):
            cell = row.cells[idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(v)
            r.font.name = '맑은 고딕'
            r._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
            r.font.size = Pt(9.5)
            if idx == 6:
                r.bold = True
                r.font.color.rgb = RGBColor(0, 102, 204)
            if idx == 0:
                r.bold = True

    add_p(doc, '')

    # 5대 시점별 세부 성적 테이블
    add_styled_heading(doc, '공식 5대 시점별 3-State F1 성능 상세', level=2)

    table_views = doc.add_table(rows=1, cols=6)
    table_views.style = 'Table Grid'
    v_headers = ['카메라 시점 명칭', '내부 코드', 'c1 (gm1_v3)', 'c2 (gm2_v1)', 'c3 (gm3_v3)', '시점별 평균 F1']
    hdr_v = table_views.rows[0]
    for idx, h in enumerate(v_headers):
        cell = hdr_v.cells[idx]
        set_cell_background(cell, '2E75B6')
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = '맑은 고딕'
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.bold = True

    v_data = [
        ['시점 A (우측 후방 45°)', 'view5', '79.41%', '71.43%', '81.91%', '77.58%'],
        ['시점 B (우측 전방 45°)', 'view2', '79.41%', '82.35%', '94.90%', '85.55%'],
        ['시점 C (정면 0°)', 'view3', '79.41%', '85.87%', '94.30%', '86.53%'],
        ['시점 D (좌측 전방 45°)', 'view4', '84.44%', '79.78%', '92.23%', '85.48%'],
        ['시점 E (좌측 후방 45°)', 'view1', '79.41%', '68.27%', '78.57%', '75.42%']
    ]

    for row_vals in v_data:
        row = table_views.add_row()
        for idx, v in enumerate(row_vals):
            cell = row.cells[idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(v)
            r.font.name = '맑은 고딕'
            r._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
            r.font.size = Pt(9.5)
            if idx in (0, 5):
                r.bold = True

    add_p(doc, '')

    # 2. AI-Hub 정답 라벨 사용 검사
    add_styled_heading(doc, '2. AI-Hub 정답 라벨의 피처/학습 사용 전수 검사 결과', level=1)
    add_p(doc, 'AGENTS.md의 "AI-Hub JSON 및 라벨 사용 엄격 격리 원칙(Strict Data Isolation & No Leakage)"을 바탕으로 소스코드 전수를 정밀 역추적 감사한 결과입니다.')

    table2 = doc.add_table(rows=1, cols=4)
    table2.style = 'Table Grid'
    hdr2 = table2.rows[0]
    for idx, h in enumerate(['검사 대상 모델', '피처 입력(X) 라벨 유입 여부', '학습(Training) 라벨 사용 방식', '규정 준수 여부']):
        cell = hdr2.cells[idx]
        set_cell_background(cell, '1F497D')
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = '맑은 고딕'
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.bold = True

    rows_data2 = [
        ['gm1_v3 (무릎 고정 위반)', '0% 유입 (완전 배제)\n- 13D 기하학 무릎 각도만 사용\n- context_sd3_prob 인덱스 영구 제외', '지도학습 정답 타깃(y)으로만 사용\n(conditions[0] D02 정상0/결함1)', '100% 준수 (PASS)'],
        ['gm2_v1 (시선 유지 오류)', '0% 유입 (완전 배제)\n- 15D 귀/어깨/골반 간접 생체역학\n- 순수 MediaPipe 랜드마크 추출', '지도학습 정답 타깃(y)으로만 사용\n(시선 방향 조건 정상0/결함1)', '100% 준수 (PASS)'],
        ['gm3_v3 (척추 중립 위반)', '0% 유입 (완전 배제)\n- 45D MediaPipe+SpinePose 비전\n- AI-Hub 3D JSON 좌표 일절 미참조', '지도학습 정답 타깃(y)으로만 사용\n(척추의 중립 조건 정상0/결함1)', '100% 준수 (PASS)']
    ]

    for row_vals in rows_data2:
        row = table2.add_row()
        for idx, v in enumerate(row_vals):
            cell = row.cells[idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in (0, 3) else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(v)
            r.font.name = '맑은 고딕'
            r._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
            r.font.size = Pt(9.0)
            if idx == 3:
                r.bold = True
                r.font.color.rgb = RGBColor(0, 128, 0)

    add_p(doc, '')

    # 3. 데이터 누수 7대 영역 전수 검사
    add_styled_heading(doc, '3. 데이터 누수(Data Leakage) 7대 영역 전수 감사 결과', level=1)
    add_p(doc, '머신러닝 파이프라인 전반에서 발생할 수 있는 잠재적 데이터 누수 위험 7대 항목에 대한 전수 검증 결과입니다.')

    leak_items = [
        ('1. 피험자 격리 누수 (Subject Leakage)', 'PASS (완전 무결)', '전 모델(gm1, gm2, gm3)이 StratifiedGroupKFold(5-Fold)를 사용하여 subject_id (피험자 1~9번) 단위로 완벽히 격리 분할함. Train 세트에 등장한 피험자가 Val/Test 세트에 교차 유입되는 피험자 오염이 0건임.'),
        ('2. 카메라 멀티뷰 시점 누수 (Multi-View Leakage)', 'PASS (완전 무결)', '동일 비디오의 5개 시점(view1~view5)이 피험자 단위(Group)로 함께 묶여 한 번에 동일 폴드에 배정되므로, 동일 시퀀스의 다른 시점이 Train과 Val에 쪼개져 들어가는 시점 누수가 구조적으로 원천 차단됨.'),
        ('3. AI-Hub 3D 정답 좌표 누수 (GT 3D Leakage)', 'PASS (완전 무결)', 'AI-Hub 3D JSON의 레이블러 키포인트 좌표를 입력 피처나 관절 보정에 일절 참조하지 않고, 100% 순수 비전(MediaPipe Full33 및 SpinePose 3D Lifting)으로만 관절을 추정하여 사용함.'),
        ('4. 전처리 및 시계열 보간 누수 (Temporal Leakage)', 'PASS (완전 무결)', '64프레임 3차 스플라인 보간, Savitzky-Golay 스무딩 필터, One-Euro 필터 등 모든 전처리가 개별 시퀀스 내부에서 독립적으로 실행되며, 데이터셋 전체 시계열 통계를 공유하거나 참조하지 않음.'),
        ('5. 전역 스케일러 누수 (Global Scaler Leakage)', 'PASS (완전 무결)', '전체 데이터셋의 Mean/Std를 미리 산출하여 일괄 표준화하는 Global Scaler를 일체 사용하지 않으며, 각 트리(CatBoost/LGBM/XGB)가 독립 피처의 고유 분기점을 학습하도록 설계됨.'),
        ('6. 3-State 임계값(Threshold) 결정 누수', 'PASS (완전 무결)', '3-State 보류 밴드(tau_clean, tau_defect) 결정 시 단일 Train 폴드에 피팅하지 않고, 5-Fold 전체의 Out-of-Fold (OOF, 학습 시 미참여 검증 데이터) 예측 확률만을 모아서 10% Bounded Conformal 탐색을 수행함.'),
        ('7. 데이터 증강(Augmentation) 누수', 'PASS (완전 무결)', '원천 데이터 검증셋에는 합성/증강 데이터가 전혀 포함되지 않으며, 원천 100% 실측 비전 시퀀스만을 대상으로 엄격히 교차검증을 수행함.')
    ]

    table3 = doc.add_table(rows=1, cols=3)
    table3.style = 'Table Grid'
    hdr3 = table3.rows[0]
    for idx, h in enumerate(['누수 검사 항목', '판정 결과', '세부 검증 내용 및 감사 근거']):
        cell = hdr3.cells[idx]
        set_cell_background(cell, '1F497D')
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = '맑은 고딕'
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.bold = True

    for title, res, desc in leak_items:
        row = table3.add_row()
        cell0 = row.cells[0]
        p0 = cell0.paragraphs[0]
        r0 = p0.add_run(title)
        r0.font.name = '맑은 고딕'
        r0._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        r0.font.size = Pt(9.0)
        r0.bold = True
        
        cell1 = row.cells[1]
        p1 = cell1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p1.add_run(res)
        r1.font.name = '맑은 고딕'
        r1._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        r1.font.size = Pt(9.0)
        r1.bold = True
        r1.font.color.rgb = RGBColor(0, 128, 0)
        
        cell2 = row.cells[2]
        p2 = cell2.paragraphs[0]
        r2 = p2.add_run(desc)
        r2.font.name = '맑은 고딕'
        r2._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')
        r2.font.size = Pt(9.0)

    add_p(doc, '')

    # 4. 결론 및 권고
    add_styled_heading(doc, '4. 종합 요약 및 향후 고도화 권고', level=1)
    add_p(doc, '1. [성능 무결성]: 굿모닝 세부 오류 3종 전 모델(gm1, gm2, gm3)이 원천 데이터 3-State F1 70% 목표를 전원 큰 폭으로 초과 달성(77.33% ~ 88.41%)하였으며, 5대 시점 전반에 걸쳐 고른 성능을 입증하였습니다.')
    add_p(doc, '2. [라벨 격리 무결성]: AI-Hub 정답 라벨은 지도학습의 목적변수(y)로만 정규 사용되었으며, 모델 입력 피처(X)나 전처리 파이프라인에 직간접적으로 유입된 사례는 0건으로 완벽히 격리되었습니다.')
    add_p(doc, '3. [데이터 누수 무결성]: 피험자 격리(StratifiedGroupKFold), 멀티뷰 격리, OOF 기반 임계값 튜닝 등 7대 핵심 데이터 누수 항목을 전수 감사한 결과, 어떠한 데이터 누수도 감지되지 않았습니다.')
    add_p(doc, '4. [권고사항]: gm1의 경우 초기 Stage 1 듀얼 분할셋(678 시퀀스)을 기반으로 캐싱되어 있으므로, gm2 및 gm3와 동일하게 MediaPipe_Full33_64F 원천 1,020 시퀀스(시점별 204개) 전수 데이터셋으로 동일 인터페이스 파이프라인을 일원화할 것을 권장합니다.')

    out_dir = r'D:\자주프 문서\8주차'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '굿모닝_세부오류_최종모델_3State_성능_및_데이터누수_전수검사보고서.docx')
    doc.save(out_path)
    print(f'[+] 보고서 저장 성공: {out_path}')

    out_path2 = r'D:\자주프 문서\굿모닝_세부오류_최종모델_3State_성능_및_데이터누수_전수검사보고서.docx'
    doc.save(out_path2)
    print(f'[+] 보고서 2차 저장 성공: {out_path2}')

if __name__ == '__main__':
    main()
