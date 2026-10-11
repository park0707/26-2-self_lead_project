# =========================================================================================================
# [보고서 생성기]: 런지 운동 통합 및 1단계 8종 분류 체계 개편 Word(.docx) 및 Markdown(.md) 공식 보고서 생성
# [실행 명령어]: python -m src.features.generate_lunge_merge_report
# =========================================================================================================

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_lunge_merge_report():
    out_dir = "d:/자주프 문서/3주차"
    os.makedirs(out_dir, exist_ok=True)
    docx_path = os.path.join(out_dir, "런지 운동 통합.docx")
    md_path = os.path.join(out_dir, "런지 운동 통합.md")

    # 1. Markdown 보고서 작성
    md_content = """# 런지 운동 통합 및 1단계 8종 분류 체계 개편 보고서

**작성일자**: 2026-09-09  
**프로젝트**: NovelFlow - 3D 인체 포즈 기반 지능형 운동 코칭 파이프라인  
**소속**: 2026-2 자기주도프로젝트 (3주차)  

---

## Executive Summary (요약)

본 보고서는 AI Hub 운동 데이터셋 기반의 1단계 운동 대분류 파이프라인에서 **`스텝 포워드 다이나믹 런지`**와 **`스텝 백워드 다이나믹 런지`**를 **`다이나믹 런지`**로 통합하여 기존 **9종 분류 체계를 8종 분류 체계로 전격 개편**한 기술적 배경, 생체역학적 타당성, 시스템 구현 내역 및 기대 효과를 기술합니다.

* **핵심 결론**: 포워드 런지와 백워드 런지는 **3단계 세부 결함(5대 평가 조건) 기준이 100% 동일**하며, 단안 2D 카메라 기반 3D 깊이 추정의 불확실성을 배제하고 학습 데이터를 2배(4,224 프레임)로 확충함으로써 1단계 모델의 오분류를 원천 차단하고 전체 파이프라인의 실용성을 극대화합니다.

---

## 1. 개편 추진 배경 (Why - 통합의 필연적 원인)

### 1.1 1단계 모델의 최대 취약점: 런지 전후 방향 오분류
기존 9종 분류 모델에서 가장 큰 성능 저하를 유발한 원인은 포워드 런지와 백워드 런지 간의 **상호 혼선(Confusion)**이었습니다.
* **Strict Test(복합 홀드아웃) 실측 결과**: 포워드 런지를 백워드 런지로 오분류하는 사례가 **30건 발생**하며, 포워드 런지 F1-Score가 **65.31%**로 9개 운동 중 최하위를 기록했습니다.
* **결과적 악영향**: 전체 모델의 Strict Test Macro-F1이 87.52%로 정체되는 주원인으로 작용했습니다.

### 1.2 단안 2D 기반 MediaPipe 3D의 기하학적 한계
* **정면 시점(View C)의 깊이($Z$) 모호성**: 사용자가 카메라를 정면으로 마주보고 운동할 때, 발을 앞으로 내딛거나 뒤로 뻗는 동작은 2D 카메라 센서 상에서 단순한 상하($Y$) 이동 및 미세한 원근 크기 변화로만 투영됩니다.
* **골반 중심화(Root-Centering) 착시**: 매 프레임 골반을 원점으로 정규화할 때, 앞으로 나아간 앞발보다 지면에 가만히 남겨진 뒷발이 골반 기준으로 뒤로 더 많이 밀려난 것으로 계산되어 전진/후진 부호가 뒤집히는 물리적 한계가 존재했습니다.

---

## 2. 선택의 공학적·생체역학적 타당성 검증 (Validity & Justification)

### 2.1 3단계 세부 결함(Conditions) 평가 기준의 100% 완전 일치
AI Hub 라벨 원천 데이터(`labels_json`)의 결함 평가 항목을 전수 조사한 결과, 포워드 런지와 백워드 런지는 **하위 코칭 및 결함 진단 기준이 토씨 하나 다르지 않고 100% 동일**합니다.

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ 05. 스텝 포워드 다이나믹 런지 (5대 조건) │ 06. 스텝 백워드 다이나믹 런지 (5대 조건) │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ 1. 앞다리 무릎 각도 90도 (Flexion)    │ 1. 앞다리 무릎 각도 90도 (Flexion)    │
│ 2. 몸통-발-무릎 방향 일치여부 (Valgus) │ 2. 몸통-발-무릎 방향 일치여부 (Valgus) │
│ 3. 뒤다리 무릎 각도 90도 (Flexion)    │ 3. 뒤다리 무릎 각도 90도 (Flexion)    │
│ 4. 척추의 중립 (Spine Neutral)        │ 4. 척추의 중립 (Spine Neutral)        │
│ 5. 상체의 과도한 숙임/젖힘 여부 (Torso)│ 5. 상체의 과도한 숙임/젖힘 여부 (Torso)│
└───────────────────────────────────────┴───────────────────────────────────────┘
```
*(※ 04번 바벨 런지도 위 5개 조건과 100% 동일합니다.)*

따라서 **2단계(자세 정상/오류 판별) 및 3단계(5대 세부 결함 진단) 모델 입장에서는 두 운동을 구분할 실익이 전혀 없으며**, 동일한 평가 파이프라인으로 100% 수렴됩니다.

### 2.2 데드리프트 vs 스티프 데드리프트와의 비교 차별점
* **데드리프트 계열 (분리 필수)**:
  * **바벨 데드리프트**: 무릎과 골반이 **동시에 펴지는 것**이 정상 조건.
  * **스티프 데드리프트**: 햄스트링 고립 운동이므로 **무릎을 굽힌 채 고정**해야 정상이며, 무릎을 굽히면 결함(Defect).
  * $\rightarrow$ 결함 기준이 상반되므로 **반드시 별도 클래스로 분리 유지**.
* **런지 계열 (통합 정당성 완벽)**:
  * 포워드와 백워드는 평가 기준이 완벽히 동일하므로 통합 시 **정보 손실(Information Loss)이 0%**입니다.

### 2.3 데이터 증대(Data Doubling) 효과
* 클래스 4번(`다이나믹 런지`)에 포워드 및 백워드 런지 데이터가 모두 결합되어 학습 데이터가 **총 4,224 프레임(212개 시퀀스)**으로 2배 확충되었습니다.
* 이에 따라 런지 동작의 공통 본질인 **[하지 분할 스텝 $\rightarrow$ 체간 수직 유지 $\rightarrow$ 양 무릎 90도 굴곡]** 패턴을 더욱 강건하게 학습할 수 있게 되었습니다.

---

## 3. 개편된 8종 클래스 체계 및 아키텍처 연계

### 3.1 8종 통합 운동 인덱스 매핑표

| 클래스 ID | 8종 통합 운동명 (`EXERCISE_NAMES_KO`) | 원본 데이터 폴더 (`RAW_LABELS_BASE`) | 비고 |
| :---: | :--- | :--- | :--- |
| **0** | `01_바벨_스쿼트` | `01_바벨_스쿼트` | 유지 |
| **1** | `02_바벨_데드리프트` | `02_바벨_데드리프트` | 유지 |
| **2** | `03_바벨_스티프_데드리프트` | `03_바벨_스티프_데드리프트` | 유지 |
| **3** | `04_바벨_런지` | `04_바벨_런지` | 고정형 런지 유지 |
| **4** | **`05_다이나믹_런지`** | `05_스텝_포워드_다이나믹_런지`<br>`06_스텝_백워드_다이나믹_런지` | **통합 (N:1 결합)** |
| **5** | **`06_사이드_런지`** | `07_사이드_런지` | 인덱스 6 $\rightarrow$ 5 |
| **6** | **`07_크로스_런지`** | `08_크로스_런지` | 인덱스 7 $\rightarrow$ 6 |
| **7** | **`08_굿모닝`** | `09_굿모닝` | 인덱스 8 $\rightarrow$ 7 |

### 3.2 시스템 아키텍처 무결성 보장 내역
1. **[dataset.py](file:///d:/src/models/first_model/dataset.py)**: `FOLDER_TO_EX_IDX`를 통해 9개 원천 폴더를 8개 정수 라벨($0 \sim 7$)로 동적 매핑.
2. **[metrics_helper.py](file:///d:/src/models/first_model/metrics_helper.py)**: $8 \times 8$ 혼동 행렬 및 8종 개별 세부 성적표 리포팅 체계 갱신.
3. **[train_augmented_ensemble.py](file:///d:/src/models/first_model/train_augmented_ensemble.py)**: Bi-GRU, XGBoost, CatBoost 분류기가 동적으로 `num_classes=8`에 맞추어 출력 레이어 구성.
4. **[third_models/dataset.py](file:///d:/src/models/third_models/dataset.py)**: 3단계 모델 로더에 `FOLDER_MATCHES` 별칭 매퍼를 추가하여 `05_다이나믹_런지`로 런지 결함 데이터 일괄 로드 지원.
5. **[pipeline_evaluator.py](file:///d:/src/models/pipeline_evaluator.py)**: 1단계 8종 예측 결과가 2/3단계 런지 평가 모듈로 무손실 라우팅되도록 정렬.

---

## 4. 기대 효과 및 향후 계획

1. **치명적 오분류 원천 제거**: 포워드 $\leftrightarrow$ 백워드 런지 간 혼선(30건)이 원천 소멸하여 1단계 **Strict Test Macro-F1이 95% ~ 98% 이상으로 수직 상승**할 것으로 기대됩니다.
2. **카메라 시점 불변성 극대화**: 정면 단안 카메라의 $Z$축 왜곡에 의존하지 않고 신체 기하 각도와 자세 정렬 피처 중심으로 분류하므로, 카메라 설치 각도 변화에 극도로 강건한 상용 모델이 완성됩니다.
3. **캐시 재생성 없는 Zero-Cost 적용**: `.npz` 피쳐 데이터는 그대로 유지한 채 동적 매퍼를 통해 10초 만에 8종 체계 학습 및 실측 평가를 즉시 수행할 수 있습니다.
"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[+] Markdown 보고서 저장 완료: {md_path}")

    # 2. Word (.docx) 보고서 작성
    doc = docx.Document()
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    def set_cell_bg(cell, hex_color):
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
        cell._tc.get_or_add_tcPr().append(shd)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(20)
        r.font.color.rgb = RGBColor(0, 51, 102)
        p.paragraph_format.space_after = Pt(4)

    def add_meta(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(100, 100, 100)
        p.paragraph_format.space_after = Pt(18)

    def add_h1(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(14)
        r.font.color.rgb = RGBColor(0, 51, 102)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)

    def add_h2(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(11.5)
        r.font.color.rgb = RGBColor(50, 50, 50)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)

    def add_p(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(40, 40, 40)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15

    def add_bullet(text, bold_prefix=''):
        p = doc.add_paragraph(style='List Bullet')
        if bold_prefix:
            r_b = p.add_run(bold_prefix)
            r_b.bold = True
            r_b.font.size = Pt(10)
            r_b.font.color.rgb = RGBColor(0, 51, 102)
        r = p.add_run(text)
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(40, 40, 40)
        p.paragraph_format.space_after = Pt(3)

    add_title("런지 운동 통합 및 1단계 8종 분류 체계 개편 보고서")
    add_meta("2026-2 자기주도프로젝트 | 3주차 기술 보고서 | 작성일: 2026-09-09 | 작성자: NovelFlow 팀")

    add_h1("Executive Summary (요약)")
    add_p("본 보고서는 AI Hub 운동 데이터셋 기반의 1단계 운동 대분류 파이프라인에서 「스텝 포워드 다이나믹 런지」와 「스텝 백워드 다이나믹 런지」를 「다이나믹 런지」로 통합하여, 기존 9종 분류 체계를 8종 분류 체계로 전격 개편한 기술적 배경, 생체역학적 타당성, 시스템 아키텍처 구현 내역 및 기대 효과를 기술합니다.")
    add_bullet("포워드 런지와 백워드 런지는 3단계 세부 결함(5대 평가 조건) 기준이 100% 완벽히 일치하여 단일 결함 평가 엔진으로 완벽히 수렴됩니다.", "핵심 결론: ")
    add_bullet("단안 2D 카메라 기반 3D 깊이(Z) 추정의 물리적 모호성을 배제하고 학습 데이터를 2배(4,224 프레임)로 확충하여 1단계 오분류를 원천 차단합니다.", "공학적 의의: ")

    add_h1("1. 개편 추진 배경 (Why - 통합의 필연적 원인)")
    add_h2("1.1 1단계 모델의 최대 취약점: 런지 전후 방향 오분류")
    add_p("기존 9종 분류 모델에서 가장 큰 성능 저하를 유발한 원인은 포워드 런지와 백워드 런지 간의 상호 혼선(Confusion)이었습니다.")
    add_bullet("복합 홀드아웃 평가에서 포워드 런지를 백워드 런지로 오분류하는 사례가 30건 발생하였으며, 포워드 런지 F1-Score가 65.31%로 9개 운동 중 최하위를 기록함.", "실측 지표: ")
    add_bullet("이로 인해 전체 모델의 Strict Test Macro-F1이 87.52%에 머무르는 최대 병목 지점이 됨.", "병목 현상: ")

    add_h2("1.2 단안 2D 기반 MediaPipe 3D의 기하학적 한계")
    add_bullet("사용자가 카메라를 정면으로 마주보고 운동할 때, 발을 앞으로 뻗거나 뒤로 뻗는 동작은 2D 센서상에서 단순한 상하(Y) 이동 및 미세한 원근 크기 변화로만 투영되어 깊이(Z) 분별 신호대잡음비(SNR)가 극도로 저하됨.", "정면 시점(View C) 깊이 왜곡: ")
    add_bullet("매 프레임 골반(Pelvis)을 원점으로 정규화(Root-Centering)할 때, 전진한 앞발보다 지면에 고정된 뒷발이 골반 기준으로 뒤로 더 많이 밀려난 것으로 계산되어 알고리즘이 뒷발을 스텝 발로 오인하여 부호가 뒤집힘.", "골반 중심화 착시 현상: ")

    add_h1("2. 선택의 공학적·생체역학적 타당성 검증 (Validity & Justification)")
    add_h2("2.1 3단계 세부 결함(Conditions) 평가 기준의 100% 완전 일치")
    add_p("AI Hub 라벨 원천 데이터의 결함 평가 항목을 전수 조사한 결과, 포워드 런지와 백워드 런지는 하위 코칭 및 결함 진단 기준이 100% 동일함을 확인하였습니다.")

    table1 = doc.add_table(rows=6, cols=3)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers1 = ["결함 진단 조건 (Conditions)", "스텝 포워드 다이나믹 런지", "스텝 백워드 다이나믹 런지"]
    rows_data1 = [
        ["1. 앞다리 무릎 각도 90도 (Flexion)", "O (동일 적용)", "O (동일 적용)"],
        ["2. 몸통 발 무릎 방향 일치여부 (Valgus)", "O (동일 적용)", "O (동일 적용)"],
        ["3. 뒤다리 무릎 각도 90도 (Flexion)", "O (동일 적용)", "O (동일 적용)"],
        ["4. 척추의 중립 (Spine Neutral)", "O (동일 적용)", "O (동일 적용)"],
        ["5. 상체의 과도한 숙임/젖힘 여부 (Torso)", "O (동일 적용)", "O (동일 적용)"]
    ]
    for col_idx, h_text in enumerate(headers1):
        cell = table1.cell(0, col_idx)
        cell.text = h_text
        set_cell_bg(cell, "003366")
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, row_vals in enumerate(rows_data1, 1):
        for col_idx, val in enumerate(row_vals):
            cell = table1.cell(row_idx, col_idx)
            cell.text = val
            set_cell_bg(cell, "F2F4F7" if row_idx % 2 == 1 else "FFFFFF")
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.size = Pt(9)
                    r.font.color.rgb = RGBColor(40, 40, 40)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_h2("2.2 데드리프트 vs 스티프 데드리프트와의 비교 차별점")
    add_bullet("바벨 데드리프트는 「무릎과 골반이 동시에 펴짐」이 정상인 반면, 스티프 데드리프트는 「무릎을 구부린 채 고정」해야 정상이고 무릎을 굽히면 결함이 되므로 기준이 상반되어 분리가 필수적임.", "데드리프트 계열 (분리 필수): ")
    add_bullet("포워드 런지와 백워드 런지는 평가 항목이 완벽히 동일하므로 통합 시 정보 손실(Information Loss)이 0%이며, 모델링 효율이 극대화됨.", "런지 계열 (통합 정당성): ")

    add_h2("2.3 데이터 증대(Data Doubling) 효과")
    add_bullet("다이나믹 런지 클래스(인덱스 4)에 총 4,224 프레임(212개 시퀀스)의 대규모 데이터가 집중되어 [하지 분할 스텝 -> 상체 직립 유지 -> 양 무릎 90도 굴곡]의 핵심 패턴 학습력이 대폭 향상됨.", "학습 데이터 2배 확충: ")

    add_h1("3. 개편된 8종 클래스 체계 및 아키텍처 연계")
    
    table2 = doc.add_table(rows=9, cols=4)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers2 = ["클래스 ID", "8종 통합 운동명", "원본 데이터 폴더", "개편 및 매핑 내용"]
    rows_data2 = [
        ["0", "01_바벨_스쿼트", "01_바벨_스쿼트", "단독 클래스 유지"],
        ["1", "02_바벨_데드리프트", "02_바벨_데드리프트", "단독 클래스 유지 (동조성 평가)"],
        ["2", "03_바벨_스티프_데드리프트", "03_바벨_스티프_데드리프트", "단독 클래스 유지 (무릎 고정 평가)"],
        ["3", "04_바벨_런지", "04_바벨_런지", "고정형 바벨 런지 유지"],
        ["4", "05_다이나믹_런지", "05_스텝_포워드 / 06_스텝_백워드", "★ 2개 런지 N:1 전격 통합 (데이터 2배)"],
        ["5", "06_사이드_런지", "07_사이드_런지", "인덱스 재정렬 (6 -> 5)"],
        ["6", "07_크로스_런지", "08_크로스_런지", "인덱스 재정렬 (7 -> 6)"],
        ["7", "08_굿모닝", "09_굿모닝", "인덱스 재정렬 (8 -> 7)"]
    ]
    for col_idx, h_text in enumerate(headers2):
        cell = table2.cell(0, col_idx)
        cell.text = h_text
        set_cell_bg(cell, "003366")
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, row_vals in enumerate(rows_data2, 1):
        for col_idx, val in enumerate(row_vals):
            cell = table2.cell(row_idx, col_idx)
            cell.text = val
            bg_col = "E6F0FA" if row_idx == 5 else ("F2F4F7" if row_idx % 2 == 1 else "FFFFFF")
            set_cell_bg(cell, bg_col)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [0, 1] else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    if row_idx == 5:
                        r.bold = True
                    r.font.size = Pt(9)
                    r.font.color.rgb = RGBColor(0, 51, 102) if row_idx == 5 else RGBColor(40, 40, 40)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_h2("3.2 시스템 아키텍처 수정 및 무결성 보장 내역")
    add_bullet("dataset.py: FOLDER_TO_EX_IDX 딕셔너리를 통해 9개 원천 폴더를 8개 정수 라벨(0~7)로 동적 매핑.", "1단계 데이터 로더: ")
    add_bullet("metrics_helper.py: 8x8 혼동 행렬 및 8종 개별 세부 성적표 리포팅 체계 갱신.", "평가 유틸리티: ")
    add_bullet("train_augmented_ensemble.py: Bi-GRU, XGBoost, CatBoost 모델의 최종 출력 레이어를 8개 클래스로 자동 동적 구성.", "앙상블 분류기: ")
    add_bullet("third_models/dataset.py: FOLDER_MATCHES 별칭 매퍼를 추가하여 05_다이나믹_런지로 런지 결함 데이터를 무손실 일괄 로드.", "3단계 결함 로더: ")
    add_bullet("pipeline_evaluator.py: 1단계 8종 예측 결과가 2/3단계 런지 평가 모듈로 무손실 라우팅되도록 종단간 연계 완료.", "종단간 파이프라인: ")

    add_h1("4. 기대 효과 및 결론")
    add_bullet("포워드-백워드 런지 간 상호 오분류(30건)가 원천 소멸하여 Strict Test Macro-F1이 95% ~ 98% 이상으로 수직 상승 기대.", "1. 치명적 오분류 원천 제거: ")
    add_bullet("정면 단안 카메라의 Z축 깊이 왜곡에 취약하지 않고 불변 기하 각도 피처 중심으로 학습하여 카메라 설치 각도 변화에 극도로 강건해짐.", "2. 시점 불변성 극대화: ")
    add_bullet(".npz 캐시 재생성 없이 즉시 8종 앙상블 학습 및 실측 평가 가능.", "3. Zero-Cost 적용: ")

    doc.save(docx_path)
    print(f"[+] Word 보고서 저장 완료: {docx_path}")

if __name__ == '__main__':
    create_lunge_merge_report()
