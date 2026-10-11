# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_v1_architecture_doc.py
Description: 바벨 데드리프트 통합 진단 엔진 v1.0 아키텍처 설계 보고서 생성 (.docx 및 .md)
Target Dir: D:\자주프 문서\6주차
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

TARGET_DIR = r"D:\자주프 문서\6주차"
os.makedirs(TARGET_DIR, exist_ok=True)
DOCX_PATH = os.path.join(TARGET_DIR, "바벨_데드리프트_통합진단엔진_v1.0_아키텍처_설계보고서.docx")
MD_PATH = os.path.join(TARGET_DIR, "바벨_데드리프트_통합진단엔진_v1.0_아키텍처_설계보고서.md")


def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def set_table_borders(table, color="CCCCCC"):
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement('w:tblBorders')
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), color)
        tblBorders.append(border)
    tblPr.append(tblBorders)


def create_docx():
    doc = docx.Document()
    
    # 여백 설정
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # 제목
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("vidio-health-trainer 6주차 통합 시스템 설계서")
    title_run.font.name = "맑은 고딕"
    title_run.font.size = Pt(11)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    h1_p = doc.add_paragraph()
    h1_run = h1_p.add_run("바벨 데드리프트 통합 진단 엔진 v1.0 아키텍처 설계 보고서")
    h1_run.font.name = "맑은 고딕"
    h1_run.font.size = Pt(18)
    h1_run.font.bold = True
    h1_run.font.color.rgb = RGBColor(0x11, 0x18, 0x27)

    sub_p = doc.add_paragraph()
    sub_run = sub_p.add_run("스쿼트 최신 v12.3 SOTA 아키텍처(FiLM + 32-Way Factorized Pairwise Energy Head + 3-State Master) 계승 기반")
    sub_run.font.name = "맑은 고딕"
    sub_run.font.size = Pt(11)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)

    # 메타 정보 박스
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(meta_table, "D1D5DB")
    meta_data = [
        ("공식 프로젝트명", "vidio-health-trainer"),
        ("대상 운동 종목", "02_바벨_데드리프트 (Barbell Deadlift)"),
        ("핵심 전이 아키텍처", "스쿼트 v12.3 공식 마스터 챔피언 엔진 (FiLM + Pairwise Energy Head + 3-State)"),
        ("작성 일시 및 환경", "2026-10-01 / AI-Hub 770건 5-Fold StratifiedGroupKFold 5대 시점 전수")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        cell_k, cell_v = row.cells[0], row.cells[1]
        cell_k.text = k
        cell_v.text = v
        set_cell_background(cell_k, "F3F4F6")
        set_cell_background(cell_v, "FFFFFF")
        cell_k.paragraphs[0].runs[0].font.bold = True
        cell_k.paragraphs[0].runs[0].font.size = Pt(9.5)
        cell_v.paragraphs[0].runs[0].font.size = Pt(9.5)
        cell_k.width = Inches(1.8)
        cell_v.width = Inches(5.0)

    doc.add_paragraph()

    # 섹션 1
    doc.add_heading("1. 배경 및 설계 철학 (Executive Summary)", level=1)
    p = doc.add_paragraph(
        "바벨 데드리프트는 5대 세부 결함(d1: 척추 중립 위반, d2: 발과 무릎 방향 일치, d3: 발바닥 지면 고정 위반, "
        "d4: 무릎과 골반 동시 펴짐 위반, d5: 바벨 궤적과 몸 밀착)으로 구성되며, 각 모델의 독립 F1-Score가 한계점에 도달함에 따라 "
        "전체 동작을 종합 판정하는 '통합 진단 엔진' 구축이 요구됩니다.\n\n"
        "기존의 단순 병렬 결합 방식은 각 모델의 오탐이 누적되어 정상 자세로 수행한 사용자의 40% 이상을 비정상으로 오진하는 "
        "False Positive Stacking 병목을 유발합니다. 이에 본 프로젝트의 검증된 SOTA인 '바벨 스쿼트 통합 엔진 v12.3'의 "
        "핵심 기술(32-Way Factorized Pairwise Energy Head, FiLM Gated Residual 중간 융합, Soft-EMR/Clean F1 복합 손실)을 "
        "공식 계승하여, 높은 전신 완전 일치율(Exact Match Ratio)과 안정적인 정상 보존력을 동시에 달성합니다."
    )
    p.runs[0].font.size = Pt(10)

    # 섹션 2
    doc.add_heading("2. 4계층(4-Layer) 통합 시스템 아키텍처", level=1)
    
    arch_table = doc.add_table(rows=5, cols=3)
    arch_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(arch_table, "D1D5DB")
    headers = ["계층 (Layer)", "주요 모듈 및 기술 사양", "수행 기능 및 출력"]
    for i, h in enumerate(headers):
        cell = arch_table.rows[0].cells[i]
        cell.text = h
        set_cell_background(cell, "1E3A8A")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(10)

    layer_data = [
        ("Layer 1: 서브전문가 레이어", "5대 세부 결함 독립 모델 (d1 v4.3, d2 v4, d3 v5.3-Opt, d4 v5.1, d5 v5.1)", "결함 확률(5D), 마진(5D), 도메인 피처(25D) 출력"),
        ("Layer 2: 메타 신경망 레이어", "JointMeta32WayMoENet (FiLM Gated Residual + 32-Way Energy Head)", "48D 텐서 수신 -> 32-Way 결합 상태 로짓 및 5-Way 보조 로짓 산출"),
        ("Layer 3: 캘리브레이션 레이어", "5대 시점(시점 A~E) 맞춤형 32D 벡터 스케일러 및 온도 보정", "시점별 센서 및 각도 왜곡 보정, Expected Calibration Error 최소화"),
        ("Layer 4: 선택적 마스터 레이어", "시계열 물리 필터링 + 신뢰도 데드밴드([0.18, 0.42]) 3-State 결정기", "State 0(CLEAN), State 1(DEFECT), State 2(RETAKE 보류) 최종 출력")
    ]
    for r_idx, r_data in enumerate(layer_data, 1):
        for c_idx, val in enumerate(r_data):
            cell = arch_table.rows[r_idx].cells[c_idx]
            cell.text = val
            cell.paragraphs[0].runs[0].font.size = Pt(9.5)
            if c_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                set_cell_background(cell, "F9FAFB")
            else:
                set_cell_background(cell, "FFFFFF")

    doc.add_paragraph()

    # 섹션 3
    doc.add_heading("3. 32개 결합 상태 및 쌍별 상호작용 에너지 모델링", level=1)
    p3 = doc.add_paragraph(
        "5대 결함의 이진 상태 조합은 2^5 = 32개 상태 공간으로 구성되며, 데드리프트의 물리적 동역학을 반영하기 위해 "
        "Unary 로짓(5D)과 10대 쌍별 상호작용 에너지(10D MLP), 완전 정상(s=00000) 전용 부스트(1D)로 에너지를 분해합니다.\n\n"
        "• State 0 (Clean, 00000): 5대 항목 무결점 완전 정상\n"
        "• State 1~31: 특정 결함 조합 (State 1: d5 단독, State 2: d4 단독, State 16: d1 단독 등)\n"
        "• 10대 Pairwise 결합: (d1, d5) 바벨 이탈-요추 굴곡 결합, (d3, d4) 발들림-굿모닝 결합, (d2, d3) 발-무릎 정렬 결합 등"
    )
    p3.runs[0].font.size = Pt(10)

    # 섹션 4
    doc.add_heading("4. 3-State 선택적 분류 및 실서비스 UX 정책", level=1)
    p4 = doc.add_paragraph(
        "사용자 경험(UX)과 부상 방지 안전성을 동시에 확보하기 위해 다음과 같은 3-State 전이 정책을 엄격히 적용합니다:\n\n"
        "1) State 0 (CLEAN 확정): P(Clean) >= 0.42 -> '완벽한 자세의 데드리프트입니다!'\n"
        "2) State 1 (DEFECT 확정): P(Clean) < 0.18 -> d4(굿모닝) > d1(척추) > d3(접지) > d2(무릎) > d5(바벨) 부상 위험도 우선순위 피드백\n"
        "3) State 2 (RETAKE 판정 보류): 0.18 <= P(Clean) < 0.42 -> 전체 보류율 8~10% 이내 통제, 최소 마진 부위 역추적 맞춤 가이드"
    )
    p4.runs[0].font.size = Pt(10)

    doc.save(DOCX_PATH)
    print(f"[+] DOCX 저장 완료: {DOCX_PATH}")


def create_md():
    content = """# vidio-health-trainer 6주차 통합 시스템 설계서
# 바벨 데드리프트 통합 진단 엔진 v1.0 아키텍처 설계 보고서
> **스쿼트 최신 v12.3 SOTA 아키텍처(FiLM + 32-Way Factorized Pairwise Energy Head + 3-State Master) 계승 기반**

---

| 항목 | 내용 |
|:---|:---|
| **공식 프로젝트명** | `vidio-health-trainer` |
| **대상 운동 종목** | 02_바벨_데드리프트 (Barbell Deadlift) |
| **핵심 전이 아키텍처** | 스쿼트 v12.3 공식 마스터 챔피언 엔진 (`babel_squat_integrated_v12_3.py`) |
| **작성 일시 및 환경** | 2026-10-01 / AI-Hub 770건 5-Fold StratifiedGroupKFold 5대 시점 전수 |

---

## 1. 배경 및 설계 철학 (Executive Summary)

바벨 데드리프트는 5대 세부 결함(d1: 척추 중립 위반, d2: 발과 무릎 방향 일치, d3: 발바닥 지면 고정 위반, d4: 무릎과 골반 동시 펴짐 위반, d5: 바벨 궤적과 몸 밀착)으로 구성되며, 각 모델의 독립 F1-Score가 한계점에 도달함에 따라 전체 동작을 종합 판정하는 '통합 진단 엔진' 구축이 요구됩니다.

기존의 단순 병렬 결합 방식은 각 모델의 오탐이 누적되어 정상 자세로 수행한 사용자의 40% 이상을 비정상으로 오진하는 False Positive Stacking 병목을 유발합니다. 이에 본 프로젝트의 검증된 SOTA인 '바벨 스쿼트 통합 엔진 v12.3'의 핵심 기술(32-Way Factorized Pairwise Energy Head, FiLM Gated Residual 중간 융합, Soft-EMR/Clean F1 복합 손실)을 공식 계승하여, 높은 전신 완전 일치율(Exact Match Ratio)과 안정적인 정상 보존력을 동시에 달성합니다.

---

## 2. 4계층(4-Layer) 통합 시스템 아키텍처

| 계층 (Layer) | 주요 모듈 및 기술 사양 | 수행 기능 및 출력 |
|:---|:---|:---|
| **Layer 1: 서브전문가 레이어** | 5대 세부 결함 독립 모델 (d1 v4.3, d2 v4, d3 v5.3-Opt, d4 v5.1, d5 v5.1) | 결함 확률(5D), 마진(5D), 도메인 피처(25D) 출력 |
| **Layer 2: 메타 신경망 레이어** | `JointMeta32WayMoENet` (FiLM Gated Residual + 32-Way Energy Head) | 48D 텐서 수신 -> 32-Way 결합 상태 로짓 및 5-Way 보조 로짓 산출 |
| **Layer 3: 캘리브레이션 레이어** | 5대 시점(시점 A~E) 맞춤형 32D 벡터 스케일러 및 온도 보정 | 시점별 센서 및 각도 왜곡 보정, Expected Calibration Error 최소화 |
| **Layer 4: 선택적 마스터 레이어** | 시계열 물리 필터링 + 신뢰도 데드밴드([0.18, 0.42]) 3-State 결정기 | State 0(CLEAN), State 1(DEFECT), State 2(RETAKE 보류) 최종 출력 |

---

## 3. 32개 결합 상태 및 쌍별 상호작용 에너지 모델링

5대 결함의 이진 상태 조합은 $2^5 = 32$개 상태 공간으로 구성되며, 데드리프트의 물리적 동역학을 반영하기 위해 Unary 로짓(5D)과 10대 쌍별 상호작용 에너지(10D MLP), 완전 정상($s=00000$) 전용 부스트(1D)로 에너지를 분해합니다:

$$E(s) = \sum_{k=1}^{5} u_k \cdot M_{s,k} + \sum_{(i,j) \in \text{Pairs}_{10}} w_{i,j} \cdot (M_{s,i} \cdot M_{s,j}) + c_0 \cdot \mathbb{I}(s = 0)$$

* **State 0 (Clean, `00000`)**: 5대 항목 무결점 완전 정상
* **State 1~31**: 특정 결함 조합 (State 1: d5 단독, State 2: d4 단독, State 16: d1 단독 등)
* **10대 Pairwise 결합**: (d1, d5) 바벨 이탈-요추 굴곡 결합, (d3, d4) 발들림-굿모닝 결합, (d2, d3) 발-무릎 정렬 결합 등

---

## 4. 3-State 선택적 분류 및 실서비스 UX 정책

1. **State 0 (CLEAN 확정)**: $P(\text{Clean}) \ge 0.42$ -> "완벽한 자세의 데드리프트입니다!"
2. **State 1 (DEFECT 확정)**: $P(\text{Clean}) < 0.18$ -> $d_4$(굿모닝) > $d_1$(척추) > $d_3$(접지) > $d_2$(무릎) > $d_5$(바벨) 부상 위험도 우선순위 피드백
3. **State 2 (RETAKE 판정 보류)**: $0.18 \le P(\text{Clean}) < 0.42$ -> 전체 보류율 8~10% 이내 통제, 최소 마진 부위 역추적 맞춤 가이드
"""
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] MD 저장 완료: {MD_PATH}")


if __name__ == "__main__":
    create_docx()
    create_md()
