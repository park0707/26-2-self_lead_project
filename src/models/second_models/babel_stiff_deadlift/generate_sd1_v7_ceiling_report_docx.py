# -*- coding: utf-8 -*-
"""vidio-health-trainer / 스티프 데드리프트 sd1: v7 최종 확정 및 성능 천장 전수조사 보고서 (docx + pdf)"""
import os
import docx
from docx.shared import Pt
import generate_sd1_v7_ablation_report_docx as G
from generate_sd1_v7_ablation_report_docx import heading, para, callout, table

OUT_DIR = r"D:\자주프 문서\6주차"
BASE = "vidio-health-trainer_스티프데드리프트_sd1_v7_성능천장_전수조사보고서"
DOCX_PATH, PDF_PATH = os.path.join(OUT_DIR, BASE + ".docx"), os.path.join(OUT_DIR, BASE + ".pdf")


def build():
    doc = docx.Document()
    t = doc.add_heading(level=0); G.run(t, "vidio-health-trainer  sd1 v7 최종 확정 및 성능 천장 전수조사", bold=True, size=18, color=G.NAVY)
    para(doc, "대상: 바벨 스티프 데드리프트 sd1 (무릎 구부린 채 고정 위반, 이진 분류) | 입력: MediaPipe 3D 64프레임 (64,98)", size=9)

    heading(doc, "1. 결론 요약")
    callout(doc, "결정", "v7(12D, CatBoost, Conformal 3-State)을 공식 프로덕션 모델로 확정. "
            "sd1_classifier.py → SD1V7Classifier, weights/sd1_bundle.pkl = v7 번들. v6 파일은 보존.")
    callout(doc, "핵심 결론", "현재 데이터·라벨 조건에서 '무릎 판정' 성능의 추가 향상은 매우 어렵다. 원인은 모델/피처가 아니라 "
            "라벨(복합 오류 키에 의한 자동 부여)과 표본(고유 텐서 약 203개, 피험자 9명)이다. "
            "원시 통계에서 관측된 AUC 상승(67.9%)은 무릎이 아닌 상체·바벨 채널의 라벨 지름길로 판단되어 제품에 사용하지 않는다.", alert=True)

    heading(doc, "2. 천장 진단 6항목 (LOSO, 피험자 격리)")
    table(doc, ["항목", "결과", "해석"], [
        ["E1 라벨-물리 불일치", "결함 중 무릎이 실제 고정 52.6% (N=78) / 정상 중 52.8% (N=125)",
         "라벨이 무릎 물리와 무관. 무릎 물리만 보는 완벽 검출기의 AUC 상한 약 73.7% 이하"],
        ["E2 무릎 vs 비무릎", "무릎/하지 10D 58.0, 비무릎 3D 46.5, sd3 단독 49.5, v7 12D 57.7, 19D 54.9",
         "신호는 무릎 피처에서만 나옴. 비무릎 피처 추가는 우연 수준 이하"],
        ["E3 Key별 판별(정제학습 OOF)", "Key213 58.0 / Key214 59.2 / Key215 55.6 (vs 정상 AUC)",
         "결함 키 전반에서 약한 판별. 키 간 차이 작음"],
        ["E4 학습곡선", "피험자 2/4/6/8명: 원본 AUC 49.0 / 51.0 / 55.6 / 57.8",
         "8명 시점에서도 상승 중이나 완만. 피험자 추가가 가장 유효한 레버"],
        ["E5 모델·피처 탐색", "v7 12D: CatBoost 57.7 / LightGBM 56.6 / RF 54.9 / ET 54.4 / LogReg 54.3 / kNN 57.2",
         "다른 모델이 CatBoost를 뚜렷이 넘지 못함. 19D는 더 낮음(과적합)"],
        ["E6 통계적 검정력", "v7 AUC 57.6 CI [50.7, 63.6] / v6 55.9 / Δ +1.69 CI [+0.22, +3.56]",
         "CI 폭 약 13%p. 1~2%p 개선은 구분 불가, 검출 가능 최소 개선폭 약 6%p"],
    ], widths=[1.4, 3.0, 2.6])

    heading(doc, "3. 원시 통계 490D 발견과 귀인 분석 (Stage 5)")
    para(doc, "98채널 × [mean, std, min, max, range] 원시 통계 490D를 원본 라벨로 학습하면 LightGBM 원본 AUC가 67.9%로 상승했다. "
         "이 신호의 출처를 채널 그룹별로 분리했다 (원본 라벨 학습, LOSO).")
    table(doc, ["채널 그룹", "차원", "LightGBM 풀링 AUC", "피험자내 AUC", "특이도"], [
        ["v7 12D(수작업 무릎)", 12, "60.2", "63.2", "69.3"],
        ["원시통계 전체", 490, "67.9", "66.8", "84.6"],
        ["하체/무릎 채널", 255, "57.8", "58.7", "74.8"],
        ["무릎·정강이 채널만", 130, "52.0", "51.5", "61.5"],
        ["상체/바벨 채널만 (무릎 정보 없음)", 205, "62.0", "61.5", "78.3"],
    ], widths=[2.4, 0.7, 1.4, 1.2, 0.9], hl_rows=(4,))
    para(doc, "· 무릎 정보가 없는 상체/바벨 채널(62.0)이 무릎·정강이 채널(52.0)보다 높다. sd1 라벨이 바벨 기울기·이격·뒤로 젖힘 키(Key213~215)에 "
         "자동 부여되기 때문에 모델이 다른 결함의 흔적을 학습한 것으로 해석된다.")
    para(doc, "· 피험자별 결함 비율은 37.5~42.9%로 거의 동일하며 피험자내 AUC도 풀링 AUC와 비슷하다. 피험자 간 라벨 비율 차이에 의한 착시는 아니다.")
    para(doc, "· gain 상위 20 피처에 상체(Rsh-Relb.dx, Relb-Rwr.* 등)가 다수. 그룹별 gain 합: 상체/바벨 46.4%, 하체/무릎 53.6%.")
    callout(doc, "제품 판단", "이 상승분을 쓰면 무릎이 펴져 있어도 '무릎 결함'을 안내하는 허위 피드백 위험이 있다. 무릎 중심 피처인 v7을 유지한다. "
            "또한 490D 탐색은 6모델×4피처셋 다중 비교로 낙관 편향이 있고, 피험자 9명 부트스트랩 CI도 넓다.", alert=True)

    heading(doc, "4. 문헌 조사 (직접 검증된 것만)")
    table(doc, ["출처", "시사점"], [
        ["Parmar et al., Fitness-AQA, ECCV 2022 (arXiv:2202.14019)", "운동은 BackSquat/BarbellRow/OverheadPress. 세밀한 단일 결함 비교 수치는 우리 조건과 직접 비교 불가"],
        ["Northcutt et al., Pervasive Label Errors in Test Sets (arXiv:2103.14749)", "테스트셋 라벨 오류 평균 3.3% 이상이 모델 순위를 뒤집을 수 있음. 노이즈 라벨 평가의 한계 근거"],
        ["Varoquaux, Cross-validation failure (arXiv:1706.07581)", "100샘플 수준에서 CV 오차막대 ±10%. 본 연구 E6의 CI 폭 13%p와 일치"],
        ["QEVD-FIT-COACH (NeurIPS)", "VLM 코칭 피드백 벤치마크로 단일 결함 F1 직접 비교 불가"],
        ["미확인 항목", "세밀한 무릎 결함 SOTA 수치, MotionBERT/ST-GCN 전이·증강·semi-supervised·multi-task의 소규모 노이즈 데이터 이득 사례: 출처 확인 못함. "
                      "서브에이전트 보고의 '배틀로프' 등 일부 주장은 검증 결과 오류여서 제외"],
    ], widths=[2.8, 4.2])

    heading(doc, "5. 전처리 점검: 시점 중복 원인과 Condition3 재평가 (읽기 전용 분석)")
    para(doc, "· 원본 이미지는 시점 A~E가 각각 고유(해시 상이). 그러나 학습 데이터 Condition4(폴더명은 구명칭, 이후 'Condition4'로 표기)는 "
         "view1/2/3/5의 텐서가 완전 동일(차이 0.0), 2,063개 → 635개 고유.")
    para(doc, "· 원인(src/features/build_pipeline_splits.py): 시점을 제거한 base_sid로 전처리 파일을 조회. 전처리 파일(스티프 207개)은 "
         "시퀀스당 1개, 메타데이터상 전부 시점 A, 2D 좌표만 보유(keypoints_3d 없음) → 빌더가 z를 0으로 채움. "
         "즉 Condition4의 시점 1/2/3/5는 시점 A 복사본이며 깊이 정보가 없음.")
    para(doc, "· 규정(5시점 전수·3D 입력) 미충족 상태이므로, 시점별 성능표는 시점 차이를 보여주지 못하며 같은 샘플이 4회 평가되는 효과가 있음.")
    table(doc, ["평가 (v7 12D, LOSO, CatBoost)", "Condition4 (현재)", "Condition3 (시점별 고유 3D)"], [
        ["고유 텐서 수", "203", "668"],
        ["원본라벨 AUC (정제학습, 시점별)", "57.0±0.2", "52.9±0.6"],
        ["원본라벨 AUC (원본학습, 시점별)", "58.3±1.0", "54.0±0.8"],
        ["시점 평균 앙상블 AUC (원본학습)", "58.3±1.0", "57.1±2.6"],
        ["19D 전체 AUC (원본학습, 시점별)", "-", "52.6±0.8"],
        ["결함 F1 / 특이도 (정제학습)", "34.2 / 81.5", "44.1 / 58.0 (특이도 하락으로 F1 부풀림)"],
        ["무릎각 단독 AUC", "54.3", "49.7"],
        ["결함 중 직선무릎 비율", "62.0% (정상 74.9%)", "16.7% (정상 15.9%)"],
        ["정제로 바뀐 라벨 수", "103", "11 (임계값이 Condition4 분포 기준이라 비교 불가)"],
    ], widths=[2.8, 2.0, 2.2])
    para(doc, "· Condition3은 시점별 원본 MediaPipe 3D로 노이즈가 커서(무릎각 평균 140° vs 162°) 오히려 성능이 같거나 낮음. 시점별 AUC(원본): "
         "시점E 54.4 / 시점B 54.4 / 시점C 47.0 / 시점D 48.0 / 시점A 54.5 (N=155, 시점D 48). 표본이 작아 해석은 보수적으로.")
    callout(doc, "판정", "시점 중복·2D 입력은 성능 한계의 원인이 아니다. 진짜 시점별 3D로 교체해도 향상이 없으므로 로더·전처리 변경은 하지 않는다 "
            "(Condition4 유지, v7 유지).")

    heading(doc, "6. 최종 판정: 성능 한계 (향상 경로 불가 확인)")
    callout(doc, "결론", "현재 데이터·라벨 조건에서 sd1(무릎 구부린 채 고정 위반)은 원본 라벨 AUC 약 57~58%가 실질적 한계이다. "
            "이를 넘기 위해 필요한 두 가지 경로가 모두 이번 프로젝트에서 수행 불가로 확정되었다.", alert=True)
    table(doc, ["경로", "상태", "의미"], [
        ["정답 무릎 굴곡 라벨 검증셋 직접 구축 (100~200건)", "수행 불가 (사용자 판단)", "라벨-물리 불일치와 순환 평가를 해소할 유일한 방법이 막힘"],
        ["피험자 확대 (현재 9명)", "수행 불가 (데이터 한계)", "학습곡선이 8명 시점에서도 상승 중이나 추가 확보 불가"],
        ["피처·모델 추가, 원시통계 490D, 앙상블", "비권고 (실험 기각)", "E2·E5·Stage 5: 이득 없음 또는 상체/바벨 지름길 학습"],
        ["시점별 3D 데이터(Condition3) 교체", "기각 (실험)", "AUC 동급 이하, 특이도 하락"],
    ], widths=[2.8, 1.6, 2.6])
    para(doc, "운영 권고: v7을 최종 모델로 유지한다. 3-State(평가 보류) 출력과 보류율 ≤10%를 유지하고, 성능 보고 시 원본 라벨 AUC, "
         "LOSO 시드 평균±표준편차, 피험자 단위 부트스트랩 CI를 함께 표기한다. 시점별 성능표는 데이터 구조상 시점 A 단일 입력임을 명시한다.")

    doc.save(DOCX_PATH); print("[+] DOCX:", DOCX_PATH)
    try:
        from docx2pdf import convert
        convert(DOCX_PATH, PDF_PATH); print("[+] PDF:", PDF_PATH)
    except Exception as e:
        print("[-] PDF 변환 실패:", e)


if __name__ == "__main__":
    build()
