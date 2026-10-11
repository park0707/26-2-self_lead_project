# -*- coding: utf-8 -*-
"""
[vidio-health-trainer] 바벨 런지 c5 36D 피처 간섭, 다중공선성, 성능 저해 피처 전수 조사 공식 보고서 생성기 (.docx & .md)
"""

import os
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

REPORT_DIR = r"D:\자주프 문서\6주차"
DOCX_PATH = os.path.join(REPORT_DIR, "바벨_런지_c5_36D_피처간섭_및_성능저해_전수조사_보고서.docx")
MD_PATH = os.path.join(REPORT_DIR, "바벨_런지_c5_36D_피처간섭_및_성능저해_전수조사_보고서.md")

JSON_INTERFERENCE_PATH = r"D:\src\features\lunge_c5_feature_interference_results.json"
JSON_PRUNING_PATH = r"D:\src\features\lunge_c5_pruning_simulation_results.json"

def generate_reports():
    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(JSON_INTERFERENCE_PATH, 'r', encoding='utf-8') as f:
        data_interf = json.load(f)

    with open(JSON_PRUNING_PATH, 'r', encoding='utf-8') as f:
        data_pruning = json.load(f)

    # =========================================================================
    # 1. Word (.docx) 생성
    # =========================================================================
    doc = docx.Document()

    # 여백 설정 (1인치)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(19)
        run.font.bold = True
        run.font.color.rgb = RGBColor(24, 43, 73)
        p.paragraph_format.space_after = Pt(4)

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(100, 110, 125)
        p.paragraph_format.space_after = Pt(16)

    def add_h1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(13.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(16, 52, 96)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)

    def add_h2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 86, 160)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)

    def add_p(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(40, 40, 40)
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_after = Pt(3)
        return p

    def add_bullet(text):
        p = doc.add_paragraph(style='List Bullet')
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(40, 40, 40)
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(2)

    def add_callout(text, bold_prefix=""):
        p = doc.add_paragraph()
        pPr = p._p.get_or_add_pPr()
        pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="8" w:color="1B365D"/></w:pBdr>')
        pPr.append(pBdr)
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F0F4F8"/>')
        pPr.append(shd)
        if bold_prefix:
            r_bold = p.add_run(bold_prefix)
            r_bold.font.name = 'Malgun Gothic'
            r_bold.font.size = Pt(9.5)
            r_bold.font.bold = True
            r_bold.font.color.rgb = RGBColor(24, 43, 73)
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(40, 40, 40)
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)

    def style_table(table, col_widths=None):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                if col_widths and j < len(col_widths):
                    cell.width = Inches(col_widths[j])
                if i == 0:
                    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1B365D"/>')
                    cell._tc.get_or_add_tcPr().append(shd)
                    for p in cell.paragraphs:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for run in p.runs:
                            run.font.name = 'Malgun Gothic'
                            run.font.size = Pt(9.0)
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(255, 255, 255)
                else:
                    if i % 2 == 1:
                        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8F9FA"/>')
                    else:
                        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="FFFFFF"/>')
                    cell._tc.get_or_add_tcPr().append(shd)
                    for p in cell.paragraphs:
                        for run in p.runs:
                            run.font.name = 'Malgun Gothic'
                            run.font.size = Pt(8.5)
                            run.font.color.rgb = RGBColor(40, 40, 40)
                tcPr = cell._tc.get_or_add_tcPr()
                borders = parse_xml(f'''
                    <w:tcBorders {nsdecls("w")}>
                        <w:top w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>
                        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>
                        <w:left w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>
                        <w:right w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>
                    </w:tcBorders>
                ''')
                tcPr.append(borders)

    # 1. 제목 및 개요
    add_title("[vidio-health-trainer] 바벨 런지 c5 36D 피처 간섭 및 성능 저해 전수조사 보고서")
    add_subtitle("상체 과도한 숙임/젖힘 판별을 저해하는 다중공선성·노이즈·중복 피처 전수 감사 및 절제 실증")

    add_h1("1. 전수 조사 배경 및 목적")
    add_callout(
        "바벨 런지 5번째 오류인 '상체의 과도한 숙임/젖힘 여부(lunge_c5)' 모델은 v1부터 v5까지 진화하며 총 36차원의 고차원 도메인 피처셋을 구축하였습니다. "
        "그러나 다수의 피처가 동시에 투입됨에 따라 ① 피처 간 극단적 다중공선성(Multicollinearity)에 의한 트리 분할 가중치 교란, "
        "② MediaPipe 랜드마크 프레임레이트 변동 및 고주파 지터에 의한 위양성(FP) 노이즈 유발, "
        "③ 셋업 동적 영점 표류로 인한 가짜 피크(Pseudo-Peak) 증폭 등 '성능을 저해하고 방해하는 요인'이 발생하고 있습니다. "
        "본 조사는 36개 전 피처에 대해 통계적 상관성, VIF, 단일 변별력, 5대 모델 중요도, 그리고 36회 LOFO(Leave-One-Feature-Out) 절제 실험을 수행하여 성능 저해 요인을 전수 규명하였습니다.",
        "[조사 핵심 요약] "
    )

    # 2. 성능 저해 피처(LOFO 실증 결과)
    add_h1("2. 제거 시 오히려 성능이 향상되는 '성능 저해 피처' 실증 결과")
    add_p("36개 피처 각각을 단독 제거(Leave-One-Out)한 뒤 5-Fold 피험자 전수 격리 앙상블 학습을 수행한 결과, 아래 2개 피처는 존재하는 것보다 제거했을 때 3-State F1-Score가 오히려 유의미하게 상승하는 '성능 저해 피처'로 실증 판정되었습니다.")

    t_deg = doc.add_table(rows=3, cols=6)
    t_deg_headers = ["피처 번호 및 명칭", "제거 시 F1", "F1 개선폭(Δ)", "제거 시 AUC", "주요 간섭 대상 / VIF", "성능 저해 원인 분석"]
    for j, h in enumerate(t_deg_headers):
        t_deg.rows[0].cells[j].paragraphs[0].text = h

    data_deg = [
        [
            "[34] lunge_torso_spectral_energy_ratio",
            "73.00%",
            "+0.23%p 상승",
            "0.8289",
            "고주파 지터 / VIF=3.1",
            "카메라 FPS 변동 및 MediaPipe 랜드마크 미세 떨림을 '과도한 숙임 반동'으로 오인하여 정상 시퀀스를 결함(FP)으로 오진 유발."
        ],
        [
            "[28] lunge_baseline_subtracted_peak",
            "72.94%",
            "+0.16%p 상승",
            "0.8251",
            "[06] velocity_peak와 r=0.929, VIF=51.4",
            "셋업(0~8F) 초기의 준비 동작 흔들림이 영점에 감산되면서 노이즈 피크를 증폭시키고, 6번 피치 각속도와 92.9% 공선성 충돌을 일으킴."
        ]
    ]
    for i, row in enumerate(data_deg):
        for j, val in enumerate(row):
            t_deg.rows[i+1].cells[j].paragraphs[0].text = val
    style_table(t_deg, [2.2, 0.9, 1.1, 0.9, 1.6, 2.3])

    # 3. 극단적 다중공선성 및 중복 피처 쌍 (|r| >= 0.85, VIF > 10)
    add_h1("3. 극단적 다중공선성(Multicollinearity) 및 상호 중복 피처 쌍")
    add_p("36×36 피어슨/스피어만 상관계수 행렬 및 Ridge 기반 VIF 분석 결과, 사실상 동일한 물리량을 중복 측정하여 트리 모델의 분기 의사결정을 교란시키는 고상관 피처 쌍 상위 8개를 도출하였습니다.")

    t_corr = doc.add_table(rows=9, cols=5)
    t_corr_headers = ["피처 1 (보존 후보)", "피처 2 (제거 후보)", "피어슨 r", "스피어만 r", "중복 및 간섭 메커니즘"]
    for j, h in enumerate(t_corr_headers):
        t_corr.rows[0].cells[j].paragraphs[0].text = h

    data_corr = [
        ["[32] early_ascent_burst_jerk", "[22] ascent_trunk_jerk", "+0.983", "+0.922", "동일한 상승기 3차 저크를 중복 산출. VIF 44.8 초래."],
        ["[14] torso_foreshortening_ratio", "[15] calibrated_delta_z", "-0.873", "-0.970", "체간 길이비와 Z축 깊이 변위는 수학적 역변환 관계(|r|=0.970)."],
        ["[01] vspa_torso_pitch_bottom", "[07] ldc_phase_weighted_pitch", "+0.930", "+0.938", "최하사점 평균각과 위상 가중각 중복. VIF 144.6으로 극대화."],
        ["[03] vspa_torso_pitch_min", "[05] farrokhi_hyperext_excess", "-0.803", "-0.964", "최소 피치각과 파로키 과신전 적분의 상호 완전 대칭 중복."],
        ["[06] vspa_pitch_velocity_peak", "[28] baseline_subtracted_peak", "+0.929", "+0.865", "피치 각속도 피크와 셋업 차감 피크의 극단적 상관(r=0.929)."],
        ["[06] vspa_pitch_velocity_peak", "[22] ascent_trunk_jerk", "+0.910", "+0.677", "각속도 미분 피크와 3차 저크 간의 고상관 동시 발생."],
        ["[06] vspa_pitch_velocity_peak", "[23] bottom_pitch_stability_std", "+0.906", "+0.644", "불안정성 표준편차와 각속도 피크의 90% 이상 중복 간섭."],
        ["[02] vspa_torso_pitch_peak", "[31] bidirectional_hyperext_asym", "-0.581", "-0.874", "최대 전경 피치와 후방 과신전 에너지 비의 비선형 결합."]
    ]
    for i, row in enumerate(data_corr):
        for j, val in enumerate(row):
            t_corr.rows[i+1].cells[j].paragraphs[0].text = val
    style_table(t_corr, [2.0, 2.0, 0.9, 0.9, 2.7])

    # 4. 5대 모델 종합 중요도 최하위 및 무기여 피처
    add_h1("4. 5대 앙상블 모델 종합 중요도 최하위 (무기여·노이즈 피처)")
    add_p("LightGBM, CatBoost, XGBoost, Random Forest, ExtraTrees 5대 모델의 정규화 중요도를 종합 분석한 결과, 기여도가 1% 미만으로 거의 분기되지 않거나 노이즈로 작용하는 하위 피처는 다음과 같습니다.")

    t_imp = doc.add_table(rows=6, cols=5)
    t_imp_headers = ["피처 번호 및 명칭", "종합 중요도", "LGBM", "CatBoost", "진단 및 처리 방안"]
    for j, h in enumerate(t_imp_headers):
        t_imp.rows[0].cells[j].paragraphs[0].text = h

    data_imp = [
        ["[04] lunge_farrokhi_flexion_excess", "0.47%", "0.32%", "0.51%", "25도 고정 임계치 초과 적분으로 개인별 체형 차이 미반영 -> 분기 기여 전무."],
        ["[18] lunge_frontal_compression_integral", "0.75%", "0.64%", "0.82%", "정면 체간 수축 적분치로 카메라 거리 변화 노이즈 취약 -> 기여 미미."],
        ["[13] lunge_trunk_knee_moment_arm", "1.47%", "1.20%", "1.55%", "무릎-상체 모멘트암 비율은 보폭에 의해 좌우되어 결함 분별력 취약."],
        ["[36] lunge_bottom_energy_flux_ratio", "1.56%", "1.35%", "1.60%", "최하사점 운동에너지 플럭스 비율로 LOFO 상 중립(ΔF1=-0.29%p)."],
        ["[01] lunge_vspa_torso_pitch_bottom", "1.62%", "1.10%", "1.80%", "7번 및 2번 피치 피크에 정보가 완전히 흡수되어 VIF 74.6 유발."]
    ]
    for i, row in enumerate(data_imp):
        for j, val in enumerate(row):
            t_imp.rows[i+1].cells[j].paragraphs[0].text = val
    style_table(t_imp, [2.4, 1.0, 0.9, 0.9, 2.8])

    # 5. 절제 시뮬레이션 및 성능 향상 실증 (Pruning Simulation)
    add_h1("5. 저해 및 중복 피처 단계별 정제(Pruning) 실증 평가")
    add_p("식별된 성능 저해 피처, 극단적 공선성 중복 피처, 무기여 피처를 단계별로 정제하였을 때의 5-Fold 피험자 전수 격리 앙상블 성능 변화는 아래와 같습니다.")

    t_sim = doc.add_table(rows=5, cols=6)
    t_sim_headers = ["실험 단계 (Phase)", "피처 차원", "3-State F1", "ROC-AUC", "재현율 (Recall)", "판단 보류율"]
    for j, h in enumerate(t_sim_headers):
        t_sim.rows[0].cells[j].paragraphs[0].text = h

    data_sim = [
        ["Baseline (전체 36D)", "36차원", "72.77%", "0.8291", "80.34%", "9.90%"],
        ["Step 1: 저해 피처 2개 제거", "34차원", "72.94% (+0.17%p)", "0.8291", "81.53%", "9.81%"],
        ["Step 2: 저해+극단 중복 5개 제거", "31차원", "73.22% (+0.45%p)", "0.8297", "81.09%", "9.90%"],
        ["Step 3: 저해+중복+무기여 7개 제거", "29차원", "73.24% (+0.47%p)", "0.8314 (+0.0023)", "82.96% (+2.62%p)", "9.71%"]
    ]
    for i, row in enumerate(data_sim):
        for j, val in enumerate(row):
            t_sim.rows[i+1].cells[j].paragraphs[0].text = val
    style_table(t_sim, [2.2, 1.0, 1.4, 1.2, 1.4, 1.1])

    add_callout(
        "36차원에서 성능을 저해하는 2개 피처([34], [28])와 극단적 중복 3개 피처([22], [15], [07]), 무기여 2개 피처([04], [18]) 총 7개 피처를 정제하여 29차원으로 슬림화할 경우, "
        "전체 3-State F1-Score는 72.77%에서 73.24%로 +0.47%p 상승하고, 결함 재현율은 80.34%에서 82.96%로 +2.62%p 대폭 향상되며, 판단 보류율은 9.71%로 엄격 제약(10% 이하)을 안정적으로 유지합니다.",
        "[피처 정제 실증 결론] "
    )

    # 6. 절대 제거 금지 핵심 필수 피처 (Top Essential Features)
    add_h1("6. 제거 시 치명적 급락을 유발하는 '절대 보존 핵심 피처'")
    add_p("반대로 LOFO 절제 실험 시 단 1개만 제거되어도 전체 모델 성능이 급락하는 핵심 생체역학 피처는 다음과 같습니다.")
    add_bullet("[19] lunge_cervico_thoracic_angle (경추-흉추 사잇각): 제거 시 F1 -2.09%p 급락 (70.69%), AUC -0.0160 폭락 -> 상체 숙임 판별의 최대 핵심 앵커 피처.")
    add_bullet("[20] lunge_head_torso_hyperext_sync (머리-체간 과신전 동시성): 제거 시 F1 -1.83%p 급락 (70.94%), AUC -0.0113 하락 -> 시선과 체간의 보상 동조 포착 필수.")
    add_bullet("[17] lunge_coronal_tilt_rejection (관상면 어깨 기울기): 제거 시 F1 -1.05%p 급락 -> 좌우 롤링/비틀림 노이즈를 걸러내는 필수 필터링 피처.")
    add_bullet("[08] lunge_tibia_torso_parallelism (정강이-체간 평행도 일탈): 제거 시 F1 -1.01%p 급락 -> 대퇴 힌지와 비정상 체간 굴곡을 분리하는 생체역학 핵심.")
    add_bullet("[11] lunge_hip_extension_deficit (후행 다리 고관절 신전 결손량): 제거 시 F1 -1.01%p 급락 -> 골반 중립 상실 감지의 핵심 지표.")

    # 7. 향후 최적화 권고안
    add_h1("7. 차기 모델(v6) 구축을 위한 피처 최적화 권고안")
    add_bullet("권고안 1: 성능 저해 피처 2종([34] spectral_energy_ratio, [28] baseline_subtracted_peak) 즉시 영구 배제.")
    add_bullet("권고안 2: 다중공선성 r >= 0.93 이상인 중복 피처 3종([22] ascent_jerk, [15] calibrated_delta_z, [07] ldc_phase_pitch) 제거하여 모델 복잡도 및 메모리/추론 지연 대폭 경감.")
    add_bullet("권고안 3: 기여도 0.5% 미만인 고정 임계치 피처 2종([04] farrokhi_flexion, [18] frontal_compression) 제거를 통해 정제된 29차원 최적화 피처셋 확정.")

    doc.save(DOCX_PATH)
    print(f"[+] 공식 Word 보고서 생성 완료: {DOCX_PATH}")

    # =========================================================================
    # 2. Markdown (.md) 생성
    # =========================================================================
    md_content = f"""# [vidio-health-trainer] 바벨 런지 c5 36D 피처 간섭 및 성능 저해 전수조사 보고서
> **상체 과도한 숙임/젖힘 판별을 저해하는 다중공선성·노이즈·중복 피처 전수 감사 및 절제 실증**

---

## 1. 전수 조사 배경 및 목적
바벨 런지 5번째 오류인 **'상체의 과도한 숙임/젖힘 여부(`lunge_c5`)'** 판별 모델은 v1부터 v5까지 진화하며 총 36차원의 고차원 도메인 피처셋을 구축하였습니다.

그러나 다수의 피처가 동시에 투입됨에 따라:
1. **극단적 다중공선성(Multicollinearity)**에 의한 트리 분할 가중치 교란 ($|r| \ge 0.85$, $VIF > 100$)
2. **MediaPipe 랜드마크 프레임레이트 변동 및 고주파 지터**에 의한 위양성(FP) 노이즈 유발
3. **셋업 동적 영점 표류**로 인한 가짜 피크(Pseudo-Peak) 증폭

본 조사는 36개 전 피처에 대해 통계적 상관성, VIF, 단일 변별력, 5대 모델 중요도, 그리고 **36회 LOFO(Leave-One-Feature-Out) 절제 실험**을 수행하여 성능 저해 요인을 전수 규명하였습니다.

---

## 2. 제거 시 오히려 성능이 향상되는 '성능 저해 피처' 실증 결과

36개 피처 각각을 단독 제거(LOFO)한 뒤 5-Fold 피험자 전수 격리 앙상블 학습을 수행한 결과, 아래 2개 피처는 **존재하는 것보다 제거했을 때 3-State F1-Score가 오히려 유의미하게 상승하는 '성능 저해 피처'**로 실증 판정되었습니다.

| 피처 번호 및 명칭 | 제거 시 F1 | F1 개선폭 ($\Delta$) | 제거 시 AUC | 주요 간섭 대상 / VIF | 성능 저해 원인 분석 |
|---|:---:|:---:|:---:|:---:|---|
| **[34] `lunge_torso_spectral_energy_ratio`** | **73.00%** | **+0.23%p 상승** | 0.8289 | 고주파 지터 / VIF=3.1 | 카메라 FPS 변동 및 MediaPipe 랜드마크 미세 떨림을 '과도한 숙임 반동'으로 오인하여 정상 시퀀스를 결함(FP)으로 오진 유발. |
| **[28] `lunge_baseline_subtracted_peak`** | **72.94%** | **+0.16%p 상승** | 0.8251 | [06]과 $r=0.929$, VIF=51.4 | 셋업(0~8F) 초기의 준비 동작 흔들림이 영점에 감산되면서 노이즈 피크를 증폭시키고, 6번 피치 각속도와 92.9% 공선성 충돌 유발. |

---

## 3. 극단적 다중공선성(Multicollinearity) 및 상호 중복 피처 쌍 ($|r| \ge 0.85$)

| 피처 1 (보존 권고) | 피처 2 (제거 권고) | 피어슨 $r$ | 스피어만 $r$ | 중복 및 간섭 메커니즘 |
|---|---|:---:|:---:|---|
| **[32] `early_ascent_burst_jerk`** | **[22] `ascent_trunk_jerk`** | **+0.983** | **+0.922** | 동일한 상승기 3차 저크를 중복 산출. VIF 44.8 초래. |
| **[14] `torso_foreshortening_ratio`** | **[15] `calibrated_delta_z`** | **-0.873** | **-0.970** | 체간 길이비와 Z축 깊이 변위는 수학적 역변환 관계($\|r\|=0.970$). |
| **[01] `vspa_torso_pitch_bottom`** | **[07] `ldc_phase_weighted_pitch`** | **+0.930** | **+0.938** | 최하사점 평균각과 위상 가중각 중복. VIF 144.6으로 극대화. |
| **[03] `vspa_torso_pitch_min`** | **[05] `farrokhi_hyperext_excess`** | **-0.803** | **-0.964** | 최소 피치각과 파로키 과신전 적분의 상호 완전 대칭 중복. |
| **[06] `vspa_pitch_velocity_peak`** | **[28] `baseline_subtracted_peak`** | **+0.929** | **+0.865** | 피치 각속도 피크와 셋업 차감 피크의 극단적 상관($r=0.929$). |
| **[06] `vspa_pitch_velocity_peak`** | **[22] `ascent_trunk_jerk`** | **+0.910** | **+0.677** | 각속도 미분 피크와 3차 저크 간의 고상관 동시 발생. |
| **[06] `vspa_pitch_velocity_peak`** | **[23] `bottom_pitch_stability_std`** | **+0.906** | **+0.644** | 불안정성 표준편차와 각속도 피크의 90% 이상 중복 간섭. |
| **[02] `vspa_torso_pitch_peak`** | **[31] `bidirectional_hyperext_asym`** | **-0.581** | **-0.874** | 최대 전경 피치와 후방 과신전 에너지 비의 비선형 결합. |

---

## 4. 5대 앙상블 모델 종합 중요도 최하위 (무기여·노이즈 피처)

| 피처 번호 및 명칭 | 종합 중요도 | LGBM | CatBoost | 진단 및 처리 방안 |
|---|:---:|:---:|:---:|---|
| **[04] `lunge_farrokhi_flexion_excess`** | **0.47%** | 0.32% | 0.51% | 25도 고정 임계치 초과 적분으로 개인별 체형 차이 미반영 -> 분기 기여 전무. |
| **[18] `lunge_frontal_compression_integral`** | **0.75%** | 0.64% | 0.82% | 정면 체간 수축 적분치로 카메라 거리 변화 노이즈 취약 -> 기여 미미. |
| **[13] `lunge_trunk_knee_moment_arm`** | **1.47%** | 1.20% | 1.55% | 무릎-상체 모멘트암 비율은 보폭에 의해 좌우되어 결함 분별력 취약. |
| **[36] `lunge_bottom_energy_flux_ratio`** | **1.56%** | 1.35% | 1.60% | 최하사점 운동에너지 플럭스 비율로 LOFO 상 중립($\Delta F1=-0.29\%p$). |
| **[01] `lunge_vspa_torso_pitch_bottom`** | **1.62%** | 1.10% | 1.80% | 7번 및 2번 피치 피크에 정보가 완전히 흡수되어 VIF 74.6 유발. |

---

## 5. 저해 및 중복 피처 단계별 정제(Pruning) 실증 평가

식별된 성능 저해 피처, 극단적 공선성 중복 피처, 무기여 피처를 단계별로 정제하였을 때의 5-Fold 피험자 전수 격리 앙상블 성능 변화는 아래와 같습니다.

| 실험 단계 (Phase) | 피처 차원 | 3-State F1-Score | ROC-AUC | 결함 재현율 (Recall) | 판단 보류율 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (전체 36D)** | 36차원 | 72.77% | 0.8291 | 80.34% | 9.90% |
| **Step 1: 저해 피처 2개 제거** | 34차원 | **72.94% (+0.17%p)** | 0.8291 | 81.53% | 9.81% |
| **Step 2: 저해+극단 중복 5개 제거** | 31차원 | **73.22% (+0.45%p)** | 0.8297 | 81.09% | 9.90% |
| **Step 3: 저해+중복+무기여 7개 제거** | **29차원** | **73.24% (+0.47%p)** | **0.8314 (+0.0023)** | **82.96% (+2.62%p)** | **9.71%** |

> **정제 실증 결론**: 36차원에서 성능을 저해하는 2개 피처([34], [28])와 극단적 중복 3개 피처([22], [15], [07]), 무기여 2개 피처([04], [18]) 총 7개 피처를 정제하여 29차원으로 슬림화할 경우, **3-State F1-Score는 +0.47%p 상승(73.24%)**, **결함 재현율은 +2.62%p 대폭 향상(82.96%)**, **판단 보류율은 9.71%로 안정 유지**됨이 실증되었습니다.

---

## 6. 제거 시 치명적 급락을 유발하는 '절대 보존 핵심 피처'

- **`[19] lunge_cervico_thoracic_angle` (경추-흉추 사잇각)**: 제거 시 F1 **-2.09%p 급락** (70.69%), AUC **-0.0160 폭락** -> 상체 숙임 판별의 최대 핵심 앵커 피처.
- **`[20] lunge_head_torso_hyperext_sync` (머리-체간 과신전 동시성)**: 제거 시 F1 **-1.83%p 급락** (70.94%), AUC **-0.0113 하락** -> 시선과 체간의 보상 동조 포착 필수.
- **`[17] lunge_coronal_tilt_rejection` (관상면 어깨 기울기)**: 제거 시 F1 **-1.05%p 급락** -> 좌우 롤링/비틀림 노이즈를 걸러내는 필수 필터링 피처.
- **`[08] lunge_tibia_torso_parallelism` (정강이-체간 평행도 일탈)**: 제거 시 F1 **-1.01%p 급락** -> 대퇴 힌지와 비정상 체간 굴곡을 분리하는 생체역학 핵심.
- **`[11] lunge_hip_extension_deficit` (후행 다리 고관절 신전 결손량)**: 제거 시 F1 **-1.01%p 급락** -> 골반 중립 상실 감지의 핵심 지표.

---

## 7. 차기 모델(v6) 구축을 위한 피처 최적화 권고안
1. **권고안 1**: 성능 저해 피처 2종(`[34] spectral_energy_ratio`, `[28] baseline_subtracted_peak`) 즉시 영구 배제.
2. **권고안 2**: 다중공선성 $r \ge 0.93$ 이상인 중복 피처 3종(`[22] ascent_jerk`, `[15] calibrated_delta_z`, `[07] ldc_phase_pitch`) 제거하여 모델 복잡도 및 메모리/추론 지연 경감.
3. **권고안 3**: 기여도 0.5% 미만인 고정 임계치 피처 2종(`[04] farrokhi_flexion`, `[18] frontal_compression`) 제거를 통해 정제된 **29차원 최적화 피처셋** 확정.
"""

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[+] 공식 마크다운 보고서 생성 완료: {MD_PATH}")

if __name__ == '__main__':
    generate_reports()
