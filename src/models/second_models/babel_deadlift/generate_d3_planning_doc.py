# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d3_planning_doc.py
Purpose: 바벨 데드리프트 d3 세부 결함 모델 아키텍처 기획서 Word(.docx) 생성
Target Path: D:\자주프 문서\6주차\바벨_데드리프트_d3_모델_아키텍처_기획서.docx
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_DOCX_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d3_모델_아키텍처_기획서.docx"
OUT_MD_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d3_모델_아키텍처_기획서.md"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_header_box(doc, title, subtitle):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "1E3A8A")  # Dark Blue
    set_cell_margins(cell, top=200, bottom=200, left=200, right=200)
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run1 = p.add_run(title + "\n")
    run1.font.name = "맑은 고딕"
    run1.font.size = Pt(18)
    run1.font.bold = True
    run1.font.color.rgb = RGBColor(255, 255, 255)
    
    run2 = p.add_run(subtitle)
    run2.font.name = "맑은 고딕"
    run2.font.size = Pt(11)
    run2.font.color.rgb = RGBColor(224, 231, 255)
    
    doc.add_paragraph()

def add_section_title(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    
    run = p.add_run(text)
    run.font.name = "맑은 고딕"
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(30, 58, 138)  # Deep Navy
    elif level == 2:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(37, 99, 235)  # Royal Blue
    else:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(71, 85, 105)

def add_callout(doc, text, title="핵심 요약"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F1F5F9")  # Slate Light
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"■ {title}\n")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(10.5)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(15, 23, 42)
    
    run_b = p.add_run(text)
    run_b.font.name = "맑은 고딕"
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(51, 65, 85)
    doc.add_paragraph()

def style_table(table, col_widths, headers, data):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        hdr_cells[i].width = Inches(col_widths[i])
        set_cell_background(hdr_cells[i], "2563EB")  # Blue
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "맑은 고딕"
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            
    for row_idx, row_data in enumerate(data):
        row_cells = table.add_row().cells
        bg_color = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            row_cells[c_idx].width = Inches(col_widths[c_idx])
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 2, 3] else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "맑은 고딕"
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(30, 41, 59)

def main():
    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc = docx.Document()
    
    # 페이지 여백 설정 (1인치)
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)
        
    add_header_box(
        doc,
        "vidio-health-trainer 바벨 데드리프트 d3 모델 아키텍처 기획서",
        "운동 종목: 바벨 데드리프트 (Barbell Deadlift) | 부제: d3 결함 타깃 확정 및 25D 물리 접지 아키텍처 설계"
    )
    
    add_callout(
        doc,
        "1. 프로젝트 공식 명칭: vidio-health-trainer\n"
        "2. 데드리프트 결함 체계 현황: d1(척추 중립성 위반) 및 d2(발과 무릎 방향 일치)에 이은 세 번째 독립 세부 결함 모델(d3) 구축\n"
        "3. d3 후보 타깃 분석: AI-Hub 5대 결함 중 '발바닥 지면 고정'을 d3 최우선 추천 타깃으로 확정(스쿼트 c4 v5.1 25D 생체역학 완전 계승 가능, 현재 멀티뷰 385개 시퀀스에 정답 라벨 확보 완료)\n"
        "4. 평가 및 검증 원칙: AI-Hub 피험자 1~9번 전원 전수, 공식 5대 시점(시점 A~E, view1~view5) 전수 포함 5-Fold StratifiedGroupKFold 교차검증",
        title="기획 핵심 총람 (Executive Summary)"
    )
    
    add_section_title(doc, "1. 바벨 데드리프트 5대 결함 체계 및 d3 타깃 선정 검토", level=1)
    
    p = doc.add_paragraph()
    p.add_run(
        "AI-Hub 웨이트 트레이닝 원천 데이터셋(총 1,592건) 및 국가 공인 운동 가이드라인에 명시된 "
        "바벨 데드리프트의 공식 결함 조건은 총 5가지로 구성되어 있습니다. "
        "현재 vidio-health-trainer 프로젝트는 d1(척추)과 d2(무릎 외반)의 모델링을 완료한 상태이며, "
        "세 번째 세부 결함 모델인 d3의 타깃을 확정해야 합니다."
    )
    
    # 표 1: 데드리프트 5대 결함 현황 및 매핑
    table1 = doc.add_table(rows=1, cols=5)
    headers1 = ["결함 코드", "공식 결함 명칭", "원천 데이터 보유", "현재 Test Cache 현황", "기획 배정 권장안"]
    widths1 = [1.0, 1.6, 1.2, 1.3, 1.4]
    data1 = [
        ["d1", "척추의 중립 위반", "794건 (49.9%)", "27건 결함 / 50건 정상", "완료 (d1 v1~v3.1)"],
        ["d2", "발과 무릎의 방향 일치", "748건 (47.0%)", "18건 결함 / 59건 정상", "완료 (d2 v1~v2)"],
        ["d3 (추천)", "발바닥 지면 고정 위반", "796건 (50.0%)", "18건 결함 / 59건 정상", "★ d3 최우선 확정 대상"],
        ["d4 (예정)", "무릎과 골반 동시 펴짐", "794건 (49.9%)", "0건 (추가 전처리 필요)", "차기 d4 모델 배정"],
        ["d5 (예정)", "바벨 궤적과 몸 밀착", "795건 (49.9%)", "0건 (추가 전처리 필요)", "차기 d5 모델 배정"]
    ]
    style_table(table1, widths1, headers1, data1)
    doc.add_paragraph()
    
    add_section_title(doc, "1.1 d3 = '발바닥 지면 고정' 확정 사유 (Why Foot Grounding?)", level=2)
    p = doc.add_paragraph()
    p.add_run(
        "본 기획에서 d3 모델을 '발바닥 지면 고정(Foot Ground Fixation / Heel-Toe Elevation)'으로 확정하는 이유는 "
        "다음과 같은 3대 공학적/임상적 근거에 기반합니다:\n"
        "1. [스쿼트 c4 v5.1 아키텍처 100% 완전 계승]: 스쿼트에서 검증된 25D 물리 접지 동역학(발바닥 피치각, 뒤꿈치/앞꿈치 수직차, "
        "정강이-발목 커플링비, 수직 저크, 지면 싱귤래리티 보정)을 복사-정밀화(Copy-and-Refine) 방식으로 즉시 전이 가능.\n"
        "2. [데이터 파이프라인 즉각 활용성]: 현재 구축된 385개 64프레임 멀티뷰 시퀀스(피험자 9명 전원)에 결함 18클립(90개 시퀀스) vs "
        "정상 59클립(295개 시퀀스)으로 완벽히 레이블링되어 있어, 즉시 5-Fold 피험자 격리 학습 및 검증 착수가 가능함.\n"
        "3. [생체역학적 치명도]: 데드리프트는 바닥 정지 상태에서 고중량을 수직 견인하는 운동으로, 뒤꿈치 들림(앞쏠림)이나 앞꿈치 들림(뒤쏠림)은 "
        "지면 반력(GRF) 전달 효율을 급락시키고 요추 모멘트암을 왜곡하여 d1(척추 손상)을 직접 유발하는 근원적 결함임."
    )
    
    add_section_title(doc, "2. d3 모델 생체역학 피처 엔지니어링 설계 (25D Grounded Dynamics)", level=1)
    
    p = doc.add_paragraph()
    p.add_run(
        "스쿼트 c4 v5.1에서 입증된 25D 지면 접지 피처셋을 데드리프트 리프팅 역학에 최적화하여 4개 서브그룹으로 구조화합니다. "
        "데드리프트는 하강기 중심인 스쿼트와 달리 퍼스트 풀(바닥 이륙) 및 락아웃(기립 직전) 구간의 지면 지지가 핵심이므로, "
        "기립 슬라이스 기준선(Stand Baseline)과 퍼스트 풀 구간의 물리량을 집중 반영합니다."
    )
    
    table2 = doc.add_table(rows=1, cols=4)
    headers2 = ["그룹 구분", "피처 차원", "대표 핵심 피처", "생체역학적 기능 및 판별 원리"]
    widths2 = [1.3, 0.9, 1.8, 2.5]
    data2 = [
        ["Group 1\n발바닥 각도 & 고도", "7D\n(F01~F07)", "d3_foot_pitch_max_deg\nd3_toe_lift_max_norm\nd3_heel_toe_ydelta_nadir", "발바닥 경사각(Pitch) 및 앞꿈치/뒤꿈치 상대 높이차 추적 (앞꿈치 들림 및 뒤꿈치 붕괴 감지)"],
        ["Group 2\n정강이-발목 커플링", "5D\n(F08~F12)", "d3_coupling_shank_pitch\nd3_bilateral_pitch_asym\nd3_ank_min_angle", "정강이 전경각 대비 발목 족배굴곡 가동성 일치도 및 좌우 발바닥 지지 비대칭도 측정"],
        ["Group 3\n발목 롤 & 지면 저크", "6D\n(F13~F18)", "d3_foot_roll_inversion\nd3_delta_jerk_active\nd3_ascent_ankle_destab", "발바닥 안쪽/바깥날 롤링(외번/내번)과 바닥 이륙 시 발목 수직 가속도 급변(Jerk) 정량화"],
        ["Group 4\n체형 정규화 & 시점", "7D\n(F19~F25)", "d3_norm_delta_nadir\nd3_heel_view_visibility\nd3_contact_transition_rate", "발 길이 대비 정규화 고도차, 카메라 5대 시점별 발목 가시성 가중치, 접촉 상태 전이 빈도"]
    ]
    style_table(table2, widths2, headers2, data2)
    doc.add_paragraph()
    
    add_section_title(doc, "3. 모델링 및 학습·검증 체계 (Modeling & Evaluation Protocol)", level=1)
    
    p = doc.add_paragraph()
    p.add_run(
        "1. [앙상블 아키텍처]: LightGBM(50%) + CatBoost(30%) + Random Forest(20%)의 3-Way Tri-Ensemble 하이브리드 구조를 적용하여 "
        "트리 간 오분류 상호 보완 및 과적합을 억제합니다.\n"
        "2. [지면 싱귤래리티 방어 가드레일 (Singularity Guardrail)]:\n"
        "   - 기립 구간(Stand Slice: 0~T//8) 지면 기준선 추정 시 평균 대신 Median을 적용하여 초기 발 미세 움직임 노이즈 차단\n"
        "   - 발바닥 롤각(Roll) 아크탄젠트 점프 클리핑 (±35도 생체역학 한계 제한)\n"
        "   - 2D 정사영 시 발생하는 가상 앞꿈치 들림(toe_lift_norm <= 0.010) 필터링 보정\n"
        "3. [3-State 선택적 판정 (Selective Classification Gate)]:\n"
        "   - 정상 확정 임계치 tau_normal과 결함 확정 임계치 tau_defect를 그리드 탐색하여, 판정 모호 구간을 RETAKE(안전 보류)로 격리\n"
        "   - 커버리지 85~90% 이상을 유지하면서 판정 정확도 및 F1을 90% 이상으로 극대화\n"
        "4. [피험자 격리 5-Fold 및 공식 5대 시점 전수 감사]:\n"
        "   - AI-Hub 피험자 1번~9번 전원을 StratifiedGroupKFold로 분할하여 피험자 간 데이터 누수 완벽 차단\n"
        "   - 시점 A(우측 후방 45°), 시점 B(우측 전방 45°), 시점 C(정면 0°), 시점 D(좌측 전방 45°), 시점 E(좌측 후방 45°) 전 시점 개별 F1/AUC 감사"
    )
    
    add_section_title(doc, "4. 단계별 구현 및 실험 로드맵", level=1)
    
    table3 = doc.add_table(rows=1, cols=4)
    headers3 = ["단계", "주요 작업 내용", "산출물 파일", "목표 일정 및 완료 조건"]
    widths3 = [0.8, 2.5, 1.8, 1.4]
    data3 = [
        ["Phase 1", "d3 피처 추출기 및 베이스라인 모델 구축\n(스쿼트 c4 v5.1 계승)", "d3_model_v1.py", "25D 피처 추출 무결성 검증"],
        ["Phase 2", "5-Fold 피험자 격리 학습 및 전수 시점 감사 스크립트 작성", "train_and_eval_d3_model.py", "Baseline F1 및 Confusion Matrix 도출"],
        ["Phase 3", "3-State 선택적 판정 게이트 튜닝 및 최적 임계치 도출", "find_optimal_3state_d3.py", "Selective F1 85%+ 달성"],
        ["Phase 4", "결함 진단 보고서 및 종합 결과 문서화 (.docx 필수)", "d3_model_v1_evaluation_results.json\n공식 기획/결과 보고서", "Word 보고서 배포 완료"]
    ]
    style_table(table3, widths3, headers3, data3)
    doc.add_paragraph()
    
    doc.save(OUT_DOCX_PATH)
    print(f"[+] DOCX 기획서 생성 완료: {OUT_DOCX_PATH}")
    
    # MD 버전도 동기화 저장
    md_content = f"""# vidio-health-trainer 바벨 데드리프트 d3 모델 아키텍처 기획서

> **운동 종목**: 바벨 데드리프트 (Barbell Deadlift)  
> **공식 문서 위치**: `{OUT_DOCX_PATH}`  
> **프로젝트**: `vidio-health-trainer`

---

## ■ 기획 핵심 총람 (Executive Summary)
1. **프로젝트 공식 명칭**: `vidio-health-trainer`
2. **데드리프트 결함 체계 현황**: d1(척추 중립성 위반) 및 d2(발과 무릎 방향 일치)에 이은 세 번째 독립 세부 결함 모델(d3) 구축
3. **d3 후보 타깃 분석**: AI-Hub 5대 결함 중 **'발바닥 지면 고정'**을 d3 최우선 확정 타깃으로 선정 (스쿼트 c4 v5.1 25D 생체역학 완전 계승 가능, 현재 멀티뷰 385개 시퀀스에 정답 라벨 확보 완료)
4. **평가 및 검증 원칙**: AI-Hub 피험자 1~9번 전원 전수, 공식 5대 시점(시점 A~E, view1~view5) 전수 포함 5-Fold StratifiedGroupKFold 교차검증

---

## 1. 바벨 데드리프트 5대 결함 체계 및 d3 타깃 선정 검토

| 결함 코드 | 공식 결함 명칭 | 원천 데이터 보유 (1,592건) | 현재 Test Cache 현황 (385건) | 기획 배정 권장안 |
| :---: | :--- | :---: | :---: | :--- |
| **d1** | 척추의 중립 위반 | 794건 (49.9%) | 27건 결함 / 50건 정상 | 완료 (d1 v1~v3.1) |
| **d2** | 발과 무릎의 방향 일치 | 748건 (47.0%) | 18건 결함 / 59건 정상 | 완료 (d2 v1~v2) |
| **d3 (추천)** | **발바닥 지면 고정 위반** | **796건 (50.0%)** | **18건 결함 / 59건 정상** | **★ d3 최우선 확정 대상** |
| **d4 (예정)** | 무릎과 골반 동시 펴짐 | 794건 (49.9%) | 0건 (추가 전처리 필요) | 차기 d4 모델 배정 |
| **d5 (예정)** | 바벨 궤적과 몸 밀착 | 795건 (49.9%) | 0건 (추가 전처리 필요) | 차기 d5 모델 배정 |

### 1.1 d3 = '발바닥 지면 고정' 확정 사유
1. **스쿼트 c4 v5.1 아키텍처 100% 완전 계승**: 스쿼트에서 검증된 25D 물리 접지 동역학(발바닥 피치각, 뒤꿈치/앞꿈치 수직차, 정강이-발목 커플링비, 수직 저크, 지면 싱귤래리티 보정)을 복사-정밀화(Copy-and-Refine) 방식으로 즉시 전이 가능.
2. **데이터 파이프라인 즉각 활용성**: 현재 구축된 385개 64프레임 멀티뷰 시퀀스(피험자 9명 전원)에 결함 18클립(90개 시퀀스) vs 정상 59클립(295개 시퀀스)으로 완벽히 레이블링되어 있어, 즉시 5-Fold 피험자 격리 학습 및 검증 착수가 가능함.
3. **생체역학적 치명도**: 데드리프트는 바닥 정지 상태에서 고중량을 수직 견인하는 운동으로, 뒤꿈치 들림(앞쏠림)이나 앞꿈치 들림(뒤쏠림)은 지면 반력(GRF) 전달 효율을 급락시키고 요추 모멘트암을 왜곡하여 d1(척추 손상)을 직접 유발하는 근원적 결함임.

---

## 2. d3 모델 생체역학 피처 엔지니어링 설계 (25D Grounded Dynamics)

| 그룹 구분 | 피처 차원 | 대표 핵심 피처 | 생체역학적 기능 및 판별 원리 |
| :--- | :---: | :--- | :--- |
| **Group 1: 발바닥 각도 & 고도** | 7D (F01~F07) | `d3_foot_pitch_max_deg`<br/>`d3_toe_lift_max_norm`<br/>`d3_heel_toe_ydelta_nadir` | 발바닥 경사각(Pitch) 및 앞꿈치/뒤꿈치 상대 높이차 추적 (앞꿈치 들림 및 뒤꿈치 붕괴 감지) |
| **Group 2: 정강이-발목 커플링** | 5D (F08~F12) | `d3_coupling_shank_pitch`<br/>`d3_bilateral_pitch_asym`<br/>`d3_ank_min_angle` | 정강이 전경각 대비 발목 족배굴곡 가동성 일치도 및 좌우 발바닥 지지 비대칭도 측정 |
| **Group 3: 발목 롤 & 지면 저크** | 6D (F13~F18) | `d3_foot_roll_inversion`<br/>`d3_delta_jerk_active`<br/>`d3_ascent_ankle_destab` | 발바닥 안쪽/바깥날 롤링(외번/내번)과 바닥 이륙 시 발목 수직 가속도 급변(Jerk) 정량화 |
| **Group 4: 체형 정규화 & 시점** | 7D (F19~F25) | `d3_norm_delta_nadir`<br/>`d3_heel_view_visibility`<br/>`d3_contact_transition_rate` | 발 길이 대비 정규화 고도차, 카메라 5대 시점별 발목 가시성 가중치, 접촉 상태 전이 빈도 |

---

## 3. 모델링 및 학습·검증 체계

1. **앙상블 아키텍처**: LightGBM(50%) + CatBoost(30%) + Random Forest(20%)의 3-Way Tri-Ensemble 하이브리드 구조 적용
2. **지면 싱귤래리티 방어 가드레일 (Singularity Guardrail)**:
   - 기립 구간(Stand Slice: 0~T//8) 지면 기준선 추정 시 Median 적용
   - 발바닥 롤각(Roll) 아크탄젠트 점프 클리핑 (±35도 생체역학 한계)
   - 2D 정사영 시 발생하는 가상 앞꿈치 들림 필터링 보정
3. **3-State 선택적 판정 Gate**: 판정 모호 구간을 RETAKE(안전 보류)로 격리하여 F1 85%+ 극대화
4. **피험자 격리 5-Fold 및 공식 5대 시점 전수 감사**:
   - 시점 A: 우측 후방 45° (`view5`)
   - 시점 B: 우측 전방 45° (`view2`)
   - 시점 C: 정면 0° (`view3`)
   - 시점 D: 좌측 전방 45° (`view4`)
   - 시점 E: 좌측 후방 45° (`view1`)

---

## 4. 단계별 구현 및 실험 로드맵

| 단계 | 주요 작업 내용 | 산출물 파일 | 목표 일정 및 완료 조건 |
| :---: | :--- | :--- | :--- |
| **Phase 1** | d3 피처 추출기 및 베이스라인 모델 구축 | `d3_model_v1.py` | 25D 피처 추출 무결성 검증 |
| **Phase 2** | 5-Fold 피험자 격리 학습 및 전수 시점 감사 | `train_and_eval_d3_model.py` | Baseline F1 및 Confusion Matrix 도출 |
| **Phase 3** | 3-State 선택적 판정 게이트 튜닝 | `find_optimal_3state_d3.py` | Selective F1 85%+ 달성 |
| **Phase 4** | 결과 분석 및 최종 보고서 작성 | 공식 Word 보고서 | `.docx` 배포 완료 |
"""
    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] MD 기획서 생성 완료: {OUT_MD_PATH}")

if __name__ == "__main__":
    main()
