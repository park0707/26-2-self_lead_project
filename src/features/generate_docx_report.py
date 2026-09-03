import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_report():
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
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = RGBColor(24, 43, 73)
        p.paragraph_format.space_after = Pt(6)

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(100, 110, 125)
        p.paragraph_format.space_after = Pt(20)

    def add_h1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(16, 52, 96)
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)

    def add_h2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 86, 160)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)

    def add_p(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(40, 40, 40)
        p.paragraph_format.line_spacing = 1.35
        p.paragraph_format.space_after = Pt(4)
        return p

    def add_bullet(text):
        p = doc.add_paragraph(style='List Bullet')
        run = p.add_run(text)
        run.font.name = 'Malgun Gothic'
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(40, 40, 40)
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_after = Pt(3)

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
                            run.font.size = Pt(9.5)
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(255, 255, 255)
                else:
                    if i % 2 == 1:
                        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F7FB"/>')
                        cell._tc.get_or_add_tcPr().append(shd)
                    for p in cell.paragraphs:
                        for run in p.runs:
                            run.font.name = 'Malgun Gothic'
                            run.font.size = Pt(9)
                            run.font.color.rgb = RGBColor(40, 40, 40)

    # 1. 문서 제목
    add_title('피트니스 및 재활 자세 분석을 위한\n2D 15-Joint 데이터 전처리 및 97차원 생체역학 피처 생성 보고서')
    add_subtitle('연구 과제: 자기주도프로젝트 1단계 | 연구자: 박현준 | 최종 업데이트: 2026.09.02')

    # 1장
    add_h1('1. 연구 배경 및 데이터 전처리·피처 생성 목적')
    add_p('본 연구는 단일 스마트폰 또는 웹캠의 2D RGB 영상 입력 환경에서 운동자의 자세 결함(무릎 모임, 허리 과도 굽힘, 엉덩이 선들림, 시선 이탈 등)을 실시간으로 정밀하게 자동 판별하고 교정 피드백을 제공하는 AI 시스템 구축을 목표로 합니다.')
    add_p('원시 센서 및 영상 데이터는 다음과 같은 본질적 한계를 가지고 있어 사전 전처리와 고차 피처 추출이 필수적입니다:')
    add_bullet('센서 및 좌표계 이종성: AI Hub(카메라 추정 24관절), UI-PRMD(Kinect 22관절), REHAB24-6(OptiTrack 26관절)의 상이한 관절 토폴로지.')
    add_bullet('카메라 거치 및 체격 오차: 피험자마다 다른 신장(키), 카메라와의 거리, 거치대의 미세한 기울기(Roll)로 인한 픽셀 좌표 왜곡.')
    add_bullet('단순 (x, y) 좌표의 한계: 단순 픽셀 좌표만으로는 관절의 실제 굴곡 각도, 뼈대의 물리적 방향, 움직임의 템포(속도/가속도)를 모델이 직접적으로 학습하기 어려움.')
    add_p('따라서 본 단계에서는 (1) 15개 공통 표준 스켈레톤으로 데이터를 일원화하고, (2) 5단계 기하 정규화를 통해 외적 노이즈를 제거하며, (3) AI Hub 9종 운동 12대 오류 조건을 100% 식별할 수 있는 97차원 생체역학 피처를 추출하여 로컬 디스크에 영구 저장했습니다.')

    # 2장
    add_h1('2. 15-Joint 2D 공통 스켈레톤 토폴로지 및 뼈대 엣지 정의')
    add_p('인체 주요 관절 15개를 표준 인덱스로 정의하고, 신체 물리적 연결을 표현하는 14개의 Bone Edge를 구축했습니다 (src/core/skeleton2d.py).')

    t1 = doc.add_table(rows=10, cols=3)
    t1_headers = ['Index', '관절 명칭 (Joint Name)', '해부학적 역할 및 정규화 기준']
    for j, h in enumerate(t1_headers):
        t1.rows[0].cells[j].paragraphs[0].add_run(h)

    joints_data = [
        ('0', 'PELVIS (골반 중심)', '신체 무게중심 (Root) -> 모든 프레임의 원점 (0, 0) 기준'),
        ('1', 'TORSO (체간/척추)', '척추 중립 및 상체 기울기 측정 기준선 (길이 L = 1.0)'),
        ('2, 3', 'L / R SHOULDER (좌/우 어깨)', '상체 기울기 및 2D In-plane 수평 회전 정렬 기준축'),
        ('4, 5', 'L / R ELBOW (좌/우 팔꿈치)', '팔꿈치 굴곡/신전 각도 계산'),
        ('6, 7', 'L / R WRIST (좌/우 손목)', '상지 궤적 및 바벨-몸 밀착 거리 측정'),
        ('8, 9', 'L / R HIP (좌/우 고관절)', '고관절 굴곡 각도(Hip Flexion) 및 골반 수평도'),
        ('10, 11', 'L / R KNEE (좌/우 무릎)', '스쿼트/런지 굴곡 깊이 및 Knee Valgus(무릎 모임) 검출'),
        ('12, 13', 'L / R ANKLE (좌/우 발목)', '스탠스 폭 및 지면 접지 각도'),
        ('14', 'NECK (목/경추)', '상부 체간 정렬 및 시선/고개 방향 오류 검출')
    ]
    for row_idx, r in enumerate(joints_data, start=1):
        t1.rows[row_idx].cells[0].paragraphs[0].add_run(r[0])
        t1.rows[row_idx].cells[1].paragraphs[0].add_run(r[1])
        t1.rows[row_idx].cells[2].paragraphs[0].add_run(r[2])
    style_table(t1, [0.8, 2.2, 3.5])

    # 3장
    add_h1('3. 2D 5단계 기하 정규화 및 시간축 처리 엔진')
    add_p('모든 이종 데이터셋을 동일한 정규화 공간(Canonical Space)으로 통일하기 위해 5단계 기하 정규화를 수행했습니다 (src/preprocessing/normalizer2d.py).')
    add_bullet('Stage 1 (결측치 보간): (0, 0) 동시 0 또는 NaN 프레임을 탐지하여 3차 큐빅 스플라인(Cubic Spline)으로 부드럽게 시간축 보간.')
    add_bullet('Stage 2 (Root Centering): 매 프레임 Pelvis(골반 중심) 관절을 원점 (0, 0)으로 평행 이동하여 화면 내 위치 편차 제거.')
    add_bullet('Stage 3 (2D In-plane 회전 정렬): 초기 선 자세의 어깨선 벡터 기울기 각도(theta)를 산출하고 2D 회전 행렬 R(-theta)를 전체 시퀀스에 일괄 적용하여 카메라 롤 오차 보정.')
    add_bullet('Stage 4 (신체 스케일 정규화): 체간(Torso-Pelvis) 거리를 1.0으로 스케일링하여 피험자 신장 및 카메라 거리 차이 제거.')
    add_bullet('Stage 5 (편측 운동 좌우 대칭 통일): 런지 계열 운동 시 좌측 다리 전방 샘플을 X축 반전 및 좌우 관절 Swap을 통해 항상 우측 다리 전방으로 미러링.')
    add_bullet('Stage 6 (시간축 리샘플링): 3차 스플라인을 통해 모든 시퀀스를 고정 길이 T=64 프레임 (64, 15, 2) 텐서로 통일.')

    # 4장
    add_h1('4. AI Hub 9종 운동 12대 오류 조건 100% 식별 피처 체계')
    add_p('AI Hub 9종 운동에 존재하는 모든 세부 평가 규칙을 전수 분석하여 19대 핵심 지표(97차원 피처)로 완벽히 매핑했습니다 (src/features/kinematics_2d.py).')

    t2 = doc.add_table(rows=7, cols=4)
    t2_headers = ['스트림 구분', '세부 피처 항목', '판별하는 운동 결함 및 수식', '차원 수']
    for j, h in enumerate(t2_headers):
        t2.rows[0].cells[j].paragraphs[0].add_run(h)

    stream_data = [
        ('Stream 1: Bone Vector', '14개 신체 뼈대 분절', 'b_ij = (dx, dy), L_ij = ||b_ij||', '42차원 (Format B)'),
        ('Stream 2: 관절 굴곡각', '무릎/고관절/발목/팔꿈치 (좌·우)', '스쿼트 깊이(Depth), 런지 90도 각도 도달 여부', '8개 지표 (40차원)'),
        ('Stream 3: 자세 결함 지표', '체간 경사각, Knee Valgus, 스탠스폭', '허리 과도 숙임/젖힘, 무릎 내측 모임 결함', '6개 지표 (30차원)'),
        ('Stream 4: 좌우 비대칭성', '무릎 & 고관절 비대칭률 (%)', '짝다리 짚기, 편측 쏠림 불균형 검출', '2개 지표 (10차원)'),
        ('Stream 5: 모션 다이내믹스', '8대 관절 각속도 (1차 시간 미분)', '통제력 상실 급하강 및 반동 튕기기 검출', '8차원 (Format B)'),
        ('Stream 6: AIHub 특화 지표', '시선 정렬, 바벨 밀착, 무릎-골반 동조성', '고개 정면 이탈, 바벨 이격, 엉덩이 선들림 검출', '3개 지표 (15차원)')
    ]
    for row_idx, r in enumerate(stream_data, start=1):
        for c_idx, val in enumerate(r):
            t2.rows[row_idx].cells[c_idx].paragraphs[0].add_run(val)
    style_table(t2, [1.4, 1.8, 2.3, 1.0])

    add_h2('2계층 피처 캐시 규격')
    add_bullet('Format A (1차 머신러닝용): 19개 핵심 지표 x 5대 통계량 (Min, Max, Mean, Std, ROM) + 템포 지표 2개 = 총 97차원 1D 벡터.')
    add_bullet('Format B (2차 시계열 심층 모델용): 64프레임 전 구간 69차원 시계열 피처 텐서 = (64, 69) 2D 행렬.')

    # 5장
    add_h1('5. 로컬 디스크 생성 결과물 및 정량적 무결성 검증')
    add_p('12 코어 멀티프로세싱을 통해 전체 3대 데이터셋(총 15,925개)에 대한 97차원 피처 캐시를 완벽히 완료하여 로컬 디스크에 영구 저장했습니다.')

    t3 = doc.add_table(rows=5, cols=5)
    t3_headers = ['데이터셋 구분', '원본 데이터 소스', '전처리 시퀀스 (.npz)', 'Format A 캐시 (Shape)', 'Format B 캐시 (Shape)']
    for j, h in enumerate(t3_headers):
        t3.rows[0].cells[j].paragraphs[0].add_run(h)

    build_data = [
        ('AI Hub 피트니스', '2D JSON 라벨', '9,637개', '(9637, 97)', '(9637, 64, 69)'),
        ('UI-PRMD 재활', 'Kinect 66D Positions', '2,000개', '(2000, 97)', '(2000, 64, 69)'),
        ('REHAB24-6 모션캡처', '세그멘테이션 반복 NPY', '4,288개', '(4288, 97)', '(4288, 64, 69)'),
        ('총합 (Total)', '3대 데이터셋 통합', '15,925개', '97차원 Flat 테이블', '(64, 69) 시계열 텐서')
    ]
    for row_idx, r in enumerate(build_data, start=1):
        for c_idx, val in enumerate(r):
            t3.rows[row_idx].cells[c_idx].paragraphs[0].add_run(val)
    style_table(t3, [1.4, 1.4, 1.2, 1.2, 1.3])

    add_h2('피험자 단위 누수 방지 분할 (dataset_splits.json) 검증 결과')
    add_bullet('Train Set: 6,745개 (70.0%) — 정상 301개 (4.46%) vs 오류 6,444개 (95.54%)')
    add_bullet('Validation Set: 1,445개 (15.0%) — 정상 74개 (5.12%) vs 오류 1,371개 (94.88%)')
    add_bullet('Test Set: 1,447개 (15.0%) — 정상 54개 (3.73%) vs 오류 1,393개 (96.27%)')
    add_bullet('피험자 중복 및 데이터 누수: 0건 (완벽한 독립 보장)')
    add_bullet('외부 벤치마크 (UI-PRMD + REHAB): 6,288개 100% Frozen Zero-shot 테스트용 보존')

    out_path = 'd:/자주프 문서/전처리 및 피쳐 생성 문서.docx'
    doc.save(out_path)
    print(f'Report successfully saved at: {out_path}')

if __name__ == '__main__':
    create_report()
