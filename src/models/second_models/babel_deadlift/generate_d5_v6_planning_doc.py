# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: generate_d5_v6_planning_doc.py
Purpose: 바벨 데드리프트 d5(바벨 궤적 밀착) 성능 극대화(F1 88%+ 달성)를 위한
         v6.0 차세대 아키텍처 및 고도화 기획서 Word(.docx) 및 Markdown(.md) 동시 생성
Target Paths:
  - D:\자주프 문서\6주차\바벨_데드리프트_d5_성능향상_v6.0_고도화_기획서.docx
  - D:\자주프 문서\6주차\바벨_데드리프트_d5_성능향상_v6.0_고도화_기획서.md
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUT_DOCX_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d5_성능향상_v6.0_고도화_기획서.docx"
OUT_MD_PATH = r"D:\자주프 문서\6주차\바벨_데드리프트_d5_성능향상_v6.0_고도화_기획서.md"


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
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
    set_cell_background(cell, "1E3A8A")  # Navy
    set_cell_margins(cell, top=220, bottom=220, left=200, right=200)

    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run1 = p.add_run(title + "\n")
    run1.font.name = "맑은 고딕"
    run1.font.size = Pt(17)
    run1.font.bold = True
    run1.font.color.rgb = RGBColor(255, 255, 255)

    run2 = p.add_run(subtitle)
    run2.font.name = "맑은 고딕"
    run2.font.size = Pt(10.5)
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
        run.font.size = Pt(13.5)
        run.font.color.rgb = RGBColor(30, 58, 138)
    elif level == 2:
        run.font.size = Pt(11.5)
        run.font.color.rgb = RGBColor(37, 99, 235)
    else:
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(71, 85, 105)


def add_bullet_point(doc, title, desc, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))

    run_t = p.add_run(title + ": ")
    run_t.font.name = "맑은 고딕"
    run_t.font.size = Pt(9.5)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(30, 41, 59)

    run_d = p.add_run(desc)
    run_d.font.name = "맑은 고딕"
    run_d.font.size = Pt(9.5)
    run_d.font.color.rgb = RGBColor(71, 85, 105)


def add_styled_table(doc, headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1E3A8A")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "맑은 고딕"
            run.font.size = Pt(9.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=140, right=140)
            p = row_cells[c_idx].paragraphs[0]
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = "맑은 고딕"
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(30, 41, 59)

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    doc.add_paragraph()


def add_callout_box(doc, text, title="핵심 안내"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "EFF6FF")  # Blue-50
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)

    r_title = p.add_run(f"[{title}] ")
    r_title.font.name = "맑은 고딕"
    r_title.font.size = Pt(9.5)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(30, 58, 138)

    r_body = p.add_run(text)
    r_body.font.name = "맑은 고딕"
    r_body.font.size = Pt(9.5)
    r_body.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph()


def generate_documents():
    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc = docx.Document()

    # 페이지 여백 설정 (1인치)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # 헤더
    add_header_box(
        doc,
        "vidio-health-trainer | 차세대 모델 고도화 기획서",
        "바벨 데드리프트 d5(바벨 궤적 밀착) 성능 극대화(F1 88%+ 달성)를 위한 v6.0 기획서\n"
        "작성일: 2026-10-02 | 대상 결함: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity)"
    )

    # 1. 기획 개요 및 배경
    add_section_title(doc, "1. 기획 개요 및 추진 배경")
    p = doc.add_paragraph()
    p.add_run(
        "바벨 데드리프트 d5(바벨 궤적과 몸 밀착) 모델은 현재 v5.1 Pruned(32D 최적 피처셋) 버전으로 전수 5-Fold StratifiedGroupKFold "
        "검증 결과 기본 F1 81.60%, 3-State F1 82.57%(커버리지 91.43%)를 기록하며 프로덕션 베이스라인을 성공적으로 확보하였습니다. "
        "그러나 오분류 데이터 163건(21.17%)에 대한 심층 원인 분석 결과, d5 모델은 2D 투영의 물리적 불가항력에 부딪힌 d2(무릎 외반, F1 51.7%)와 달리 "
        "‘체형 차이에 의한 정적 모멘트 왜곡’, ‘시점 C(정면)에서의 전역 단일 임계치 강제’, ‘확신 구간 내 체계적 편향(73건)’이라는 명확하고 가시적인 "
        "알고리즘적 병목을 안고 있습니다. 본 기획서는 이러한 3대 병목을 정밀 해소하여 d5 모델의 F1 스코어를 88.0%~91.0% 수준으로 끌어올리기 위한 "
        "v6.0 차세대 아키텍처 및 세부 실행 계획을 수립합니다."
    )

    add_callout_box(
        doc,
        "d5는 시점 D(좌측 전방 45°)에서 F1 96.00%를 달성하여 문제 자체의 분별력이 이미 입증되었습니다. "
        "정면 시점(F1 55.88%)과 특정 피험자(피험자 4: 64.71%, 피험자 9: 71.43%)에 집중된 오분류를 교정하는 '체형 정규화'와 "
        "'시점 적응형 의사결정 레이어'를 도입함으로써 확실한 상향 돌파가 가능합니다.",
        title="핵심 기획 철학"
    )

    # 2. 현존 3대 핵심 병목과 기술적 진단
    add_section_title(doc, "2. 현존 3대 핵심 병목과 기술적 진단")

    add_bullet_point(
        doc,
        "병목 1: 해부학적 체형 차이(Morphological Variance) 미보정",
        "팔이 긴 피험자(피험자 3)는 정상 수행 시에도 셋업 위치에서 바벨이 무릎 앞쪽에 형성되어 FP(과잉 결함)가 16건 발생(에러의 80%)하며, "
        "상체가 긴 피험자(피험자 4, 9)는 결함 상태에서도 몸통 투영 범위에 가려져 FN(미탐)이 33건 발생합니다. "
        "동작 시작 전 정지 상태에서의 개인별 기준선(Zero-Baseline) 차감이 결여된 것이 주원인입니다."
    )
    add_bullet_point(
        doc,
        "병목 2: 시점 C(정면 0°)의 원근 압축과 단일 임계치 강제",
        "정면 뷰에서는 바벨이 몸통 앞을 가로막아 수평 변위가 깊이축으로 압축됩니다. "
        "동일한 판정 임계치(0.50)를 일괄 적용함으로써 시점 C에서만 전체 에러의 36.8%(60건)가 집중되고 FN이 39건 쏟아졌습니다."
    )
    add_bullet_point(
        doc,
        "병목 3: 확신 구간(Confident Band) 내 고확신 오판 73건",
        "불확실 구간 [0.40, 0.60] 외곽에서 모델이 0.8 이상 또는 0.2 이하로 확신하며 틀린 샘플이 73건에 달합니다. "
        "이는 피처 공간의 단순 선형 분리가 특정 시점/체형 조합의 비선형 왜곡을 잡아내지 못했음을 의미합니다."
    )

    # 3. v6.0 차세대 핵심 고도화 가설 및 해결 전략
    add_section_title(doc, "3. v6.0 핵심 고도화 4대 전략 및 메커니즘")

    headers_strat = ["고도화 전략", "기술적 해결 메커니즘", "목표 해결 병목", "예상 기대 효과"]
    data_strat = [
        [
            "전략 1: 어드레스 동적 영점화\n(Setup-Frame Zeroing)",
            "리프팅 시작 전 정지 구간(Phase 1 어드레스, 0~10F)의\nbar-shank 간격 및 shoulder-wrist 각도를\n개인별 영점(Baseline)으로 측정 후 전 구간 상대 차감(Δ)",
            "피험자 3의 긴 팔로 인한 FP(16건)\n피험자 4/9의 상체 길이로 인한 FN(33건)",
            "피험자 편차 오류 60% 이상 제거\n전체 F1 +2.5~3.0%p 상승"
        ],
        [
            "전략 2: 시점 C 적응형 의사결정\n(View-Adaptive Thresholding)",
            "시점 C(정면)에 대해 감도 우선 임계값(τ_front = 0.38)을\n동적 분기하고 관상면 팔꿈치-손목 벌어짐 피처 가중화",
            "시점 C의 압도적 미탐(FN 39건) 집중 해소",
            "시점 C F1 55.9% -> 78.0%+ 돌파\n전체 F1 +3.0~4.0%p 상승"
        ],
        [
            "전략 3: 후방 시점 가림 보정\n(Back-View Occlusion Filter)",
            "시점 A(우측 후방), 시점 E(좌측 후방)에서 등/둔근에\n의한 손목 가림 발생 시 z축 지터 필터링 및\n가상 시상각(Virtual Sagittal) 위주 동적 피처 스위칭",
            "후방 2개 시점(A, E)의 오판 76건(46.6%) 방어",
            "후방 시점 F1 75% -> 82%+ 상승\n전체 F1 +1.5~2.0%p 상승"
        ],
        [
            "전략 4: 2단계 잔차 메타 분류기\n(2-Stage Residual Booster)",
            "1단계 Tri-Ensemble의 OOF 확률이 확신 구간에 있더라도\n시점 C 및 신체 왜곡 패턴에 대해 2단계 경량 GBDT 잔차 보정",
            "확신 구간 내 체계적 편향 73건 오판 구제",
            "확신 구간 오류 73건 -> 30건 이하 억제\n3-State F1 90% 돌파"
        ]
    ]
    add_styled_table(doc, headers_strat, data_strat, [1.5, 2.2, 1.6, 1.2])

    # 4. v6.0 파이프라인 아키텍처 및 입력 규격
    add_section_title(doc, "4. v6.0 파이프라인 아키텍처 및 입력 규격")

    p = doc.add_paragraph()
    p.add_run(
        "v6.0 파이프라인은 기존 v5.1의 32D 최적 피처셋을 100% 계승(Baseline Clone)하되, "
        "어드레스 정적 영점화 및 체형 정규화 파라미터를 추가하여 총 36D로 정밀 확장합니다."
    )

    headers_spec = ["구분", "기존 v5.1 규격", "차세대 v6.0 기획 규격", "진화 내용"]
    data_spec = [
        ["피처 차원", "32D Optimized Set", "36D Zero-Calibrated Set", "동적 영점 차감 피처 4종 추가 (Δbar_dist, Δarm_ang 등)"],
        ["전처리 규격", "64F 3차 스플라인 보간", "64F 3차 스플라인 + Phase 1 자동 탐색", "어드레스(0~10F) 안정 구간 평균 기준선 산출 레이어 연동"],
        ["시점별 임계값", "단순 고정 임계값 (A:0.32, C:0.26 등)", "시점 적응형 2-Tier Soft Gating", "시점 메타데이터 조건부 확률 보정 레이어 탑재"],
        ["앙상블 구조", "Tri-Ensemble (LGB 50%, CB 30%, RF 20%)", "2-Stage Residual Tri-Ensemble", "기존 앙상블 상단에 체형/시점 편향 잔차 보정기 탑재"],
        ["3-State 마진", "전역 ±0.06 마진 (0.44~0.56)", "시점별 비대칭 마진 (정면: [0.35, 0.45])", "시점별 결함 발생 분포에 맞춤화된 비대칭 신뢰도 밴드"]
    ]
    add_styled_table(doc, headers_spec, data_spec, [1.2, 1.7, 1.9, 1.7])

    # 5. 정량적 목표 지표 (Target KPI)
    add_section_title(doc, "5. 정량적 목표 지표 (Target KPI)")

    headers_kpi = ["평가 지표", "현재 v5.1 실측치", "v6.0 기획 목표치", "달성 기준 및 비고"]
    data_kpi = [
        ["전수 OOF F1-Score", "81.60%", "★ 86.50% ~ 88.50%", "+5.0%p ~ +7.0%p 향상 (전 시점/전 피험자)"],
        ["전수 ROC-AUC", "0.8841", "★ 0.9200 이상", "분류 경계선 분별력 대폭 신장"],
        ["3-State 최종 F1", "82.57%", "★ 89.00% ~ 91.00%", "커버리지 90.0% 이상 유지 전제"],
        ["시점 C (정면 0°) F1", "55.88% (기본) / 70.0% (보정)", "★ 78.00% 이상", "정면 원근 소실 극복 및 미탐(FN) 최소화"],
        ["시점 D (좌측 전방) F1", "96.00%", "96.00% 이상 유지", "기존 최우수 성능 시점 무결성 보존"],
        ["피험자 4 & 9 FN 건수", "33건 (18건 / 15건)", "15건 이하 (-54% 감소)", "상체 긴 체형에 대한 미탐 극적 감소"],
        ["피험자 3 FP 건수", "16건", "6건 이하 (-62% 감소)", "긴 팔 체형에 대한 과잉 결함 오판 제거"]
    ]
    add_styled_table(doc, headers_kpi, data_kpi, [1.5, 1.6, 1.7, 1.7])

    # 6. 단계별 실행 로드맵 및 검증 계획
    add_section_title(doc, "6. 단계별 실행 로드맵 및 검증 계획")

    add_bullet_point(
        doc,
        "Phase 1: 어드레스 동적 영점화 피처 모듈 개발 및 무결성 검증 (1일차)",
        "기존 d5_model_v5_1.py 및 v4_gated.py를 Baseline Clone하여 64프레임 중 0~10프레임 어드레스 기준선을 추출하고, "
        "상대 변위 차감 36D 피처셋(d5_model_v6_features.py) 구축. 결측치(NaN/Inf) 및 인덱스 정합성 전수 감사."
    )
    add_bullet_point(
        doc,
        "Phase 2: 5-Fold 피험자 격리 학습 및 시점별 잔차 부스터 훈련 (2일차)",
        "피험자 1~9번 전수 대상 StratifiedGroupKFold 교차검증 수행. 1단계 Tri-Ensemble 확률에 시점/체형 보정 잔차기 결합. "
        "F1/AUC 변화 및 피험자 3, 4, 9의 오분류 추이 실측."
    )
    add_bullet_point(
        doc,
        "Phase 3: 시점별 비대칭 3-State 신뢰도 튜닝 및 다차원 감사 (3일차)",
        "시점 C, A, E 전용 불확실성 윈도우 탐색. 커버리지 90% 이상 조건에서 F1 89%+ 도달 여부 검증 및 회귀(Regression) 전수 감사."
    )
    add_bullet_point(
        doc,
        "Phase 4: 프로덕션 배포 패키징 및 최종 보고서 작성 (4일차)",
        "추론 함수(infer_d5_v6_sequence) 표준화 인터페이스 구현, D:자주프 문서/6주차 내 최종 공식 Word(.docx) 보고서 발행."
    )

    doc.save(OUT_DOCX_PATH)
    print(f"[+] DOCX Created: {OUT_DOCX_PATH}")

    # 마크다운 파일 동시 생성
    md_content = f"""# vidio-health-trainer | 차세대 모델 고도화 기획서
# 바벨 데드리프트 d5(바벨 궤적 밀착) 성능 극대화(F1 88%+ 달성)를 위한 v6.0 기획서

> **운동 종목**: 바벨 데드리프트 (Barbell Deadlift)  
> **대상 결함**: d5. 바벨 궤적과 몸 밀착 (Barbell Path Proximity to Body)  
> **공식 문서 위치**: `{OUT_DOCX_PATH}`  
> **작성일**: 2026-10-02  
> **프로젝트 명칭**: `vidio-health-trainer`

---

## 1. 기획 개요 및 추진 배경

바벨 데드리프트 d5(바벨 궤적과 몸 밀착) 모델은 현재 v5.1 Pruned(32D 최적 피처셋) 버전으로 5-Fold 피험자 완전 격리(StratifiedGroupKFold) 검증 결과 **기본 F1 81.60%**, **3-State F1 82.57%**(커버리지 91.43%)를 기록하며 프로덕션 베이스라인을 성공적으로 확보하였습니다.

그러나 오분류 데이터 163건(21.17%)에 대한 심층 원인 분석 결과, d5 모델은 2D 투영의 물리적 불가항력에 부딪힌 d2(무릎 외반, F1 51.7%)와 달리:
1. **체형 차이에 의한 정적 모멘트 왜곡** (팔 긴 피험자 FP 80%, 상체 긴 피험자 FN 33건)
2. **시점 C(정면 0°)에서의 전역 단일 임계치 강제** (정면 에러 60건 점유, 36.8%)
3. **확신 구간(Confident Band) 내 고확신 오판 73건** (체계적 편향 잔존)

이라는 명확하고 가시적인 **알고리즘적 병목**을 안고 있습니다. 본 기획서는 이러한 3대 병목을 정밀 해소하여 d5 모델의 F1 스코어를 **88.0%~91.0% 수준으로 돌파**하기 위한 v6.0 차세대 아키텍처 및 세부 실행 계획을 수립합니다.

---

## 2. 현존 3대 핵심 병목과 기술적 진단

1. **병목 1: 해부학적 체형 차이(Morphological Variance) 미보정**
   - 팔이 긴 피험자(피험자 3)는 정상 수행 시에도 셋업 위치에서 바벨이 무릎 앞쪽에 형성되어 FP(과잉 결함)가 16건 발생(피험자 3 에러의 80%)합니다.
   - 상체가 긴 피험자(피험자 4, 9)는 결함 상태에서도 몸통 투영 범위에 가려져 FN(미탐)이 33건 집중 발생합니다.
   - 동작 시작 전 정지 상태에서의 개인별 기준선(Zero-Baseline) 차감이 결여된 것이 주원인입니다.

2. **병목 2: 시점 C(정면 0°)의 원근 압축과 단일 임계치 강제**
   - 정면 뷰에서는 바벨이 몸통 앞을 가로막아 수평 변위가 깊이축으로 압축됩니다.
   - 5대 시점에 동일한 판정 임계치(0.50)를 일괄 적용함으로써 시점 C에서만 전체 에러의 36.8%(60건)가 집중되고 FN이 39건 쏟아졌습니다.

3. **병목 3: 확신 구간(Confident Band) 내 고확신 오판 73건**
   - 불확실 구간 [0.40, 0.60] 외곽에서 모델이 0.8 이상 또는 0.2 이하로 확신하며 틀린 샘플이 73건에 달합니다.
   - 이는 피처 공간의 단순 선형 분리가 특정 시점/체형 조합의 비선형 왜곡을 잡아내지 못했음을 의미합니다.

---

## 3. v6.0 핵심 고도화 4대 전략 및 메커니즘

| 고도화 전략 | 기술적 해결 메커니즘 | 목표 해결 병목 | 예상 기대 효과 |
| :--- | :--- | :--- | :--- |
| **전략 1: 어드레스 동적 영점화<br>(Setup-Frame Zeroing)** | 리프팅 시작 전 정지 구간(Phase 1 어드레스, 0~10F)의 bar-shank 간격 및 shoulder-wrist 각도를 개인별 영점으로 측정 후 전 구간 상대 차감(Δ) | 피험자 3의 긴 팔 FP(16건)<br>피험자 4/9의 긴 상체 FN(33건) | 피험자 편차 오류 60% 이상 제거<br>전체 F1 +2.5~3.0%p 상승 |
| **전략 2: 시점 C 적응형 의사결정<br>(View-Adaptive Thresholding)** | 시점 C(정면)에 대해 감도 우선 임계값(τ_front = 0.38)을 동적 분기하고 관상면 팔꿈치-손목 벌어짐 피처 가중화 | 시점 C의 압도적 미탐(FN 39건) 집중 해소 | 시점 C F1 55.9% -> 78.0%+ 돌파<br>전체 F1 +3.0~4.0%p 상승 |
| **전략 3: 후방 시점 가림 보정<br>(Back-View Occlusion Filter)** | 시점 A(우측 후방), 시점 E(좌측 후방)에서 등/둔근에 의한 손목 가림 발생 시 z축 지터 필터링 및 가상 시상각(Virtual Sagittal) 위주 동적 피처 스위칭 | 후방 2개 시점(A, E)의 오판 76건(46.6%) 방어 | 후방 시점 F1 75% -> 82%+ 상승<br>전체 F1 +1.5~2.0%p 상승 |
| **전략 4: 2단계 잔차 메타 분류기<br>(2-Stage Residual Booster)** | 1단계 Tri-Ensemble의 OOF 확률이 확신 구간에 있더라도 시점 C 및 신체 왜곡 패턴에 대해 2단계 경량 GBDT 잔차 보정 | 확신 구간 내 체계적 편향 73건 오판 구제 | 확신 구간 오류 73건 -> 30건 이하 억제<br>3-State F1 90% 돌파 |

---

## 4. v6.0 파이프라인 아키텍처 및 입력 규격

| 구분 | 기존 v5.1 규격 | 차세대 v6.0 기획 규격 | 진화 내용 |
| :--- | :---: | :---: | :--- |
| **피처 차원** | 32D Optimized Set | **36D Zero-Calibrated Set** | 동적 영점 차감 피처 4종 추가 (Δbar_dist, Δarm_ang 등) |
| **전처리 규격** | 64F 3차 스플라인 보간 | **64F 3차 스플라인 + Phase 1 자동 탐색** | 어드레스(0~10F) 안정 구간 평균 기준선 산출 레이어 연동 |
| **시점별 임계값** | 단순 고정 임계값 (A:0.32, C:0.26 등) | **시점 적응형 2-Tier Soft Gating** | 시점 메타데이터 조건부 확률 보정 레이어 탑재 |
| **앙상블 구조** | Tri-Ensemble (LGB 50%, CB 30%, RF 20%) | **2-Stage Residual Tri-Ensemble** | 기존 앙상블 상단에 체형/시점 편향 잔차 보정기 탑재 |
| **3-State 마진** | 전역 ±0.06 마진 (0.44~0.56) | **시점별 비대칭 마진 (정면: [0.35, 0.45])** | 시점별 결함 발생 분포에 맞춤화된 비대칭 신뢰도 밴드 |

---

## 5. 정량적 목표 지표 (Target KPI)

| 평가 지표 | 현재 v5.1 실측치 | v6.0 기획 목표치 | 달성 기준 및 비고 |
| :--- | :---: | :---: | :--- |
| **전수 OOF F1-Score** | 81.60% | **★ 86.50% ~ 88.50%** | +5.0%p ~ +7.0%p 향상 (전 시점/전 피험자) |
| **전수 ROC-AUC** | 0.8841 | **★ 0.9200 이상** | 분류 경계선 분별력 대폭 신장 |
| **3-State 최종 F1** | 82.57% | **★ 89.00% ~ 91.00%** | 커버리지 90.0% 이상 유지 전제 |
| **시점 C (정면 0°) F1** | 55.88% (기본) / 70.0% (보정) | **★ 78.00% 이상** | 정면 원근 소실 극복 및 미탐(FN) 최소화 |
| **시점 D (좌측 전방) F1** | 96.00% | **96.00% 이상 유지** | 기존 최우수 성능 시점 무결성 보존 |
| **피험자 4 & 9 FN 건수** | 33건 (18건 / 15건) | **15건 이하 (-54% 감소)** | 상체 긴 체형에 대한 미탐 극적 감소 |
| **피험자 3 FP 건수** | 16건 | **6건 이하 (-62% 감소)** | 긴 팔 체형에 대한 과잉 결함 오판 제거 |

---

## 6. 단계별 실행 로드맵 및 검증 계획

- **Phase 1: 어드레스 동적 영점화 피처 모듈 개발 및 무결성 검증 (1일차)**
  - 기존 `d5_model_v5_1.py` 및 `v4_gated.py`를 베이스라인 계승(Copy-and-Refine)하여 64프레임 중 0~10프레임 어드레스 기준선을 추출하고, 상대 변위 차감 36D 피처셋(`d5_model_v6_features.py`) 구축.
  - 결측치(NaN/Inf) 및 인덱스 정합성 전수 감사.
- **Phase 2: 5-Fold 피험자 격리 학습 및 시점별 잔차 부스터 훈련 (2일차)**
  - 피험자 1~9번 전수 대상 `StratifiedGroupKFold` 교차검증 수행.
  - 1단계 Tri-Ensemble 확률에 시점/체형 보정 잔차기 결합 및 F1/AUC 변화 측정.
- **Phase 3: 시점별 비대칭 3-State 신뢰도 튜닝 및 다차원 감사 (3일차)**
  - 시점 C, A, E 전용 불확실성 윈도우 탐색. 커버리지 90% 이상 조건에서 F1 89%+ 도달 여부 검증 및 회귀(Regression) 전수 감사.
- **Phase 4: 프로덕션 배포 패키징 및 최종 보고서 작성 (4일차)**
  - 추론 함수(`infer_d5_v6_sequence`) 표준화 인터페이스 구현 및 Word 보고서 최종 배포.
"""

    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] MD Created: {OUT_MD_PATH}")


if __name__ == "__main__":
    generate_documents()
