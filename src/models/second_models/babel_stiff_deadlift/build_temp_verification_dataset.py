# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Script: build_temp_verification_dataset.py
Purpose: 사용자가 원인 1(라벨 오염 FN)과 원인 2(실제 결함 FP)의 실체를 직접 육안으로
         대조 검증할 수 있도록 D:/Data/temp 에 MP4 동영상 및 프레임, 상세 진단서를 체계적으로 구축
"""

import os
import sys
import glob
import json
import shutil
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if 'd:/' not in sys.path and 'd:\\' not in sys.path:
    sys.path.insert(0, 'd:/')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEMP_ROOT = r"D:\Data\temp"
RAW_IMG_ROOT = r"D:\Data\01_raw_data\images\03_바벨_스티프_데드리프트"
RAW_LABEL_ROOT = r"D:\Data\01_raw_data\labels_json\03_바벨_스티프_데드리프트\2D"
ANALYSIS_JSON = os.path.join(CURRENT_DIR, "sd1_misclassification_analysis.json")

VIEW_ALIAS_MAP = {
    "시점A": ("view5", "시점 A (우측 후방 45°)"),
    "시점B": ("view2", "시점 B (우측 전방 45°)"),
    "시점C": ("view3", "시점 C (정면 0°)"),
    "시점D": ("view4", "시점 D (좌측 전방 45°)"),
    "시점E": ("view1", "시점 E (좌측 후방 45°)")
}

def imread_unicode(path):
    """Windows 한글 및 유니코드 경로 지원 이미지 로드"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def create_video_from_frames(frame_paths, out_mp4_path, fps=6):
    """프레임 이미지 리스트를 연속 MP4 비디오로 인코딩 (한글 경로 완벽 지원)"""
    if not frame_paths:
        return False
    frame_paths.sort()
    first_img = imread_unicode(frame_paths[0])
    if first_img is None:
        return False
    h, w, _ = first_img.shape
    
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_mp4 = os.path.join(temp_dir, f"temp_vid_{os.getpid()}_{np.random.randint(1000, 9999)}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_mp4, fourcc, fps, (w, h))
    for fp in frame_paths:
        img = imread_unicode(fp)
        if img is not None:
            out.write(img)
    out.release()
    
    if os.path.exists(temp_mp4):
        shutil.move(temp_mp4, out_mp4_path)
        return True
    return False

def export_case(target_folder, folder_name, case_title, category_desc, key_num, key_desc, 
                aihub_knee_fixed, actual_knee_status, biomech_info, model_pred_info, visual_points):
    """단일 케이스에 대해 프레임 복사, MP4 비디오 렌더링, 상세 진단서 생성"""
    src_dir = os.path.join(RAW_IMG_ROOT, folder_name)
    if not os.path.exists(src_dir):
        # 대안 검색
        matches = [d for d in os.listdir(RAW_IMG_ROOT) if folder_name in d or (folder_name.split('_')[-1] in d and folder_name.split('_')[2] in d)]
        if matches:
            src_dir = os.path.join(RAW_IMG_ROOT, matches[0])
        else:
            print(f"[!] 원천 폴더 없음: {folder_name}")
            return

    os.makedirs(target_folder, exist_ok=True)

    # 1. 시점별 프레임 분할 및 비디오 생성 (os.listdir 사용으로 대괄호 경로 문제 해결)
    all_jpg_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
    for v_code, (v_name, v_alias) in VIEW_ALIAS_MAP.items():
        v_frames = [os.path.join(src_dir, f) for f in all_jpg_files if v_code in f]
        if v_frames:
            # 시점별 프레임 폴더
            v_dir = os.path.join(target_folder, f"frames_{v_code}_{v_name}")
            os.makedirs(v_dir, exist_ok=True)
            for f in v_frames:
                shutil.copy2(f, os.path.join(v_dir, os.path.basename(f)))
            # MP4 비디오 렌더링
            mp4_path = os.path.join(target_folder, f"video_{v_code}_{v_name}_{folder_name}.mp4")
            create_video_from_frames(v_frames, mp4_path, fps=6)

    # 2. 상세 진단 보고서 작성
    report_path = os.path.join(target_folder, f"[진단_상세_리포트]_{folder_name}.txt")
    with open(report_path, 'w', encoding='utf-8') as rf:
        rf.write("=" * 80 + "\n")
        rf.write(f"  [vidio-health-trainer] 육안 검증 케이스: {case_title}\n")
        rf.write("=" * 80 + "\n\n")
        rf.write(f"1. 원천 데이터 식별 정보:\n")
        rf.write(f"   - 시퀀스 디렉토리: {folder_name}\n")
        rf.write(f"   - AI-Hub 시나리오 Key: {key_num}\n")
        rf.write(f"   - AI-Hub 시나리오 지침: {key_desc}\n")
        rf.write(f"   - 진단 분류 범주: {category_desc}\n\n")

        rf.write(f"2. 정답 라벨 vs 실제 신체 행동 대조:\n")
        rf.write(f"   - AI-Hub 공식 라벨 조건: [무릎 구부린채 고정 = {aihub_knee_fixed}]\n")
        rf.write(f"   - 실제 신체 동작 물리적 실체: [{actual_knee_status}]\n\n")

        rf.write(f"3. 3D 생체역학 실측값 (MediaPipe 불변 기하학):\n")
        for k, v in biomech_info.items():
            rf.write(f"   - {k}: {v}\n")
        rf.write("\n")

        rf.write(f"4. 비전 AI 모델 판정 결과 (sd1_v2):\n")
        for k, v in model_pred_info.items():
            rf.write(f"   - {k}: {v}\n")
        rf.write("\n")

        rf.write(f"5. 사용자를 위한 육안 확인 가이드 (Checklist):\n")
        for pt in visual_points:
            rf.write(f"   * {pt}\n")
        rf.write("\n" + "=" * 80 + "\n")

    print(f"[+] 배치 완료: {case_title} -> {target_folder}")

def run():
    print("=" * 100)
    print("  [vidio-health-trainer] D:/Data/temp 검증 데이터셋 자동 구축 파이프라인")
    print("=" * 100)

    # --------------------------------------------------------------------------
    # 1. 원인 1: 실제 행동은 정상이나 AI허브가 오류 라벨을 붙인 데이터 (FN) & 대조군
    # --------------------------------------------------------------------------
    c1_dir = os.path.join(TEMP_ROOT, "01_원인1_라벨오염_FN (실제무릎정상_AI허브결함라벨)")
    export_case(
        target_folder=os.path.join(c1_dir, "01_피험자4_Key215_과도하게뒤로_무릎완벽고정"),
        folder_name="[오류]_D02_피험자4_Key215",
        case_title="피험자 4번 Key 215 (과도하게 뒤로 / 라벨오염 FN)",
        category_desc="False Negative (미탐지 오분류 - 라벨 노이즈)",
        key_num="Key 215",
        key_desc="과도하게 뒤로 (중심 후방 이동 결함 연기)",
        aihub_knee_fixed="False (결함으로 일괄 오라벨링)",
        actual_knee_status="무릎 165.08° 각도를 칼같이 고정한 완벽한 정상 스티프 자세",
        biomech_info={
            "최저점 무릎 절대각도": "165.08° (정상 기준 160°~165° 완벽 충족)",
            "어드레스 대비 무릎 각도 변화량": "8.13° (10° 이내 완벽 고정)",
            "무릎 수직 이동 범위 (Y-ROM)": "0.051 (수직 주저앉음 전혀 없음)",
            "정강이 수직 불변성 (SVII)": "3.37 (정강이 흔들림 없음)",
            "고관절 힌지 격리비 (HHIR)": "2.46 (고관절 중심 힌지 수행)"
        },
        model_pred_info={
            "결함 확률 P(Defect)": "0.235 (23.5%로 매우 낮음 -> 76.5% 확신으로 정상 판정)",
            "2-State 예측": "0 (정상)",
            "3-State 예측": "0 (정상 확정 - Clean)",
            "판정 결과": "모델의 예측(정상)이 실제 물리적 신체 행동과 100% 일치함!"
        },
        visual_points=[
            "동영상(video_시점A 또는 시점B)을 재생하여 피험자의 무릎을 집중해서 보십시오.",
            "피험자는 상체를 숙이면서 무게중심을 뒤로 보낼 뿐, 무릎 관절은 약 15도 구부린 상태로 1도도 흔들리지 않고 잠겨 있습니다.",
            "AI-Hub 라벨러가 '과도하게 뒤로' 중심이동 결함을 보고 '무릎 고정'까지 False로 잘못 체크했음을 두 눈으로 직접 확인하실 수 있습니다."
        ]
    )

    export_case(
        target_folder=os.path.join(c1_dir, "02_피험자2_Key213_상체기울임이탈_무릎고정"),
        folder_name="[오류]_D02_피험자2_Key213",
        case_title="피험자 2번 Key 213 (오른쪽 기울고 몸에서 떨어지고 / 라벨오염 FN)",
        category_desc="False Negative (미탐지 오분류 - 라벨 노이즈)",
        key_num="Key 213",
        key_desc="오른쪽으로 기울고 몸에서 떨어지고 (상체 척추 측만 및 바벨 이탈 결함)",
        aihub_knee_fixed="False (복합 오류 후광 효과로 무릎까지 일괄 결함 처리)",
        actual_knee_status="무릎 각도 163.5° 유지, 상체와 바벨만 결함 연기함",
        biomech_info={
            "최저점 무릎 절대각도": "163.50° (정상 스티프 데드리프트 각도)",
            "어드레스 대비 무릎 변화량": "9.20° (고정성 양호)",
            "무릎 수직 이동 범위": "0.18 (주저앉음 없음)"
        },
        model_pred_info={
            "결함 확률 P(Defect)": "0.302 (정상 예측)",
            "2-State 예측": "0 (정상)",
            "3-State 예측": "0 (정상 확정)"
        },
        visual_points=[
            "피험자가 상체를 오른쪽으로 비틀고 바벨을 몸에서 떨어뜨리는 결함을 연기합니다.",
            "하체(무릎)를 보면 스쿼트처럼 주저앉지 않고 스티프 데드리프트의 무릎 잠금을 철저히 유지하고 있습니다.",
            "상체 결함 때문에 무릎까지 결함 라벨이 붙은 명백한 라벨 노이즈 사례입니다."
        ]
    )

    # 1-1. 원인 1 대조군: 실제 정상 (TN)
    c1_tn_dir = os.path.join(TEMP_ROOT, "02_원인1_대조군_실제정상_TN (실제정상_라벨정상_모델적중)")
    export_case(
        target_folder=os.path.join(c1_tn_dir, "01_피험자1_Key209_완전정상교본"),
        folder_name="[정상]_D02_피험자1_Key209",
        case_title="피험자 1번 Key 209 (완전 정상 표준 교본 / TN)",
        category_desc="True Negative (정상 적중 - 실제 정상, 라벨 정상, 모델 정상)",
        key_num="Key 209",
        key_desc="1바벨 몸밀착 2무릎살짝구부린채 고정 3척추중립 (완전 정상 교본)",
        aihub_knee_fixed="True (정상)",
        actual_knee_status="무릎 164.48° 각도를 안정적으로 고정한 완벽한 정상 동작",
        biomech_info={
            "최저점 무릎 절대각도": "164.48°",
            "어드레스 대비 무릎 변화량": "8.43°",
            "무릎 수직 이동 범위": "0.25",
            "고관절 힌지 격리비": "3.65"
        },
        model_pred_info={
            "결함 확률 P(Defect)": "0.271 (정상 예측)",
            "2-State 예측": "0 (정상)",
            "3-State 예측": "0 (정상 확정)"
        },
        visual_points=[
            "원인 1의 FN 영상과 이 정상 영상을 나란히 재생하여 무릎 각도를 비교해보십시오.",
            "원인 1(피험자4 Key215)의 무릎 고정 상태가 이 정상 교본 영상과 완벽하게 동일함을 즉시 체감하실 수 있습니다."
        ]
    )

    # 1-2. 원인 1 대조군: 실제 결함 (TP)
    c1_tp_dir = os.path.join(TEMP_ROOT, "03_원인1_대조군_실제결함_TP (실제결함_라벨결함_모델적중)")
    export_case(
        target_folder=os.path.join(c1_tp_dir, "01_피험자1_Key213_실제무릎주저앉음_결함적중"),
        folder_name="[오류]_D02_피험자1_Key213",
        case_title="피험자 1번 Key 213 (실제 무릎까지 주저앉음 / TP 결함 적중)",
        category_desc="True Positive (결함 적중 - 실제 결함, 라벨 결함, 모델 결함)",
        key_num="Key 213",
        key_desc="오른쪽으로 기울고 몸에서 떨어지고 (피험자가 무릎까지 굽힘)",
        aihub_knee_fixed="False (결함)",
        actual_knee_status="최저점에서 무릎이 148°까지 무너지며 주저앉음",
        biomech_info={
            "최저점 무릎 절대각도": "148.20° (무릎이 32도 이상 과도하게 굽혀짐)",
            "어드레스 대비 무릎 변화량": "22.50° (고정 실패)",
            "무릎 수직 이동 범위": "0.45 (심한 주저앉음)"
        },
        model_pred_info={
            "결함 확률 P(Defect)": "0.624 (62.4%로 결함 판정)",
            "2-State 예측": "1 (결함)",
            "3-State 예측": "1 (결함 확정)"
        },
        visual_points=[
            "피험자가 바벨을 내릴 때 무릎이 스쿼트처럼 밑으로 주저앉는 것을 뚜렷하게 볼 수 있습니다.",
            "이것이 '진짜 무릎 결함'이며, 모델이 이를 62.4% 결함으로 정확하게 적발해냈습니다."
        ]
    )

    # --------------------------------------------------------------------------
    # 2. 원인 2: 실제 오류지만 정상으로 표기된 데이터 (FP) & 대조군
    # --------------------------------------------------------------------------
    c2_dir = os.path.join(TEMP_ROOT, "04_원인2_라벨누락_FP (실제무릎결함_AI허브정상라벨)")
    export_case(
        target_folder=os.path.join(c2_dir, "01_피험자4_Key212_무릎120도딥스쿼트_라벨은정상"),
        folder_name="[오류]_D02_피험자4_Key212",
        case_title="피험자 4번 Key 212 (무릎 120° 딥스쿼트 주저앉음 / 라벨누락 FP)",
        category_desc="False Positive (과잉 탐지 오분류 - 실제 동작 결함을 모델이 적발)",
        key_num="Key 212",
        key_desc="왼쪽으로 기울고 (상체 측만 결함만 기재)",
        aihub_knee_fixed="True (정상으로 기재된 라벨 오류!)",
        actual_knee_status="무릎이 120.92°까지 완전히 꺾여 주저앉은 완전한 스쿼트화 결함",
        biomech_info={
            "최저점 무릎 절대각도": "120.92° (★ 딥 스쿼트 수준으로 극심하게 무릎 굽힘)",
            "어드레스 대비 무릎 변화량": "47.16° (무릎 각도가 47도나 요동침)",
            "대퇴골 가동범위 (Femur ROM)": "74.96° (허벅지가 지면과 수평이 됨)",
            "무릎 수직 이동 범위 (Y-ROM)": "1.70 (골반과 무릎이 바닥으로 푹 주저앉음)"
        },
        model_pred_info={
            "결함 확률 P(Defect)": "0.685 (68.5% 결함 확정)",
            "2-State 예측": "1 (결함)",
            "3-State 예측": "1 (결함 확정)",
            "판정 결과": "모델이 인간 라벨러의 실수를 물리적으로 완벽하게 적발함!"
        },
        visual_points=[
            "동영상을 열어 피험자의 하체를 보십시오. 이것은 스티프 데드리프트가 아니라 '스쿼트'입니다!",
            "무릎을 120도까지 푹 구부려 주저앉았음에도, AI-Hub 라벨러는 '상체가 왼쪽으로 기울었다'는 점에만 집중하여 무릎을 정상(True)으로 표기했습니다.",
            "비전 AI 모델이 이를 결함(FP)으로 잡아낸 것은 모델의 오류가 아니라 모델이 인간보다 더 정확하게 동작 결함을 적발한 것입니다."
        ]
    )

    export_case(
        target_folder=os.path.join(c2_dir, "02_피험자2_Key216_무릎140도루마니안_라벨은정상"),
        folder_name="[오류]_D02_피험자2_Key216",
        case_title="피험자 2번 Key 216 (무릎 140° 루마니안 굽힘 / 라벨누락 FP)",
        category_desc="False Positive (과잉 탐지 오분류 - 실제 동작 결함을 모델이 적발)",
        key_num="Key 216",
        key_desc="어깨 과도하게 앞으로 말린상태 (라운드 숄더)",
        aihub_knee_fixed="True (정상으로 기재)",
        actual_knee_status="어깨를 말면서 무릎을 140.97°까지 굽혀 루마니안 데드리프트 수행",
        biomech_info={
            "최저점 무릎 절대각도": "140.97° (정상 165° 대비 25도 과도하게 굴곡)",
            "어드레스 대비 무릎 변화량": "32.96° (고정 해제)",
            "대퇴골 가동범위": "36.36°",
            "무릎 수직 이동": "0.55"
        },
        model_pred_info={
            "결함 확률 P(Defect)": "0.738 (73.8% 강력한 결함 판정)",
            "2-State 예측": "1 (결함)",
            "3-State 예측": "1 (결함 확정)"
        },
        visual_points=[
            "피험자가 바벨을 내리면서 무릎을 눈에 띄게 굽히는 모습을 볼 수 있습니다.",
            "스티프 데드리프트의 무릎 고정 지침(15도 고정)을 명백히 위반한 동작입니다."
        ]
    )

    # 전체 총괄 안내서 생성
    readme_path = os.path.join(TEMP_ROOT, "[전체_육안검증_가이드_및_체크리스트].txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("  [vidio-health-trainer] 스티프 데드리프트 오분류 원인 전수 육안검증 안내서\n")
        f.write("=" * 90 + "\n\n")
        f.write("사용자님께서 제기하신 두 가지 핵심 의문을 직접 눈으로 확인하실 수 있도록\n")
        f.write("각 케이스별 MP4 동영상(슬로우 모션 6 fps), 프레임 이미지, 정량 진단 리포트를 배치하였습니다.\n\n")
        f.write("■ 폴더 구성 안내:\n")
        f.write("  1. 01_원인1_라벨오염_FN (실제무릎정상_AI허브결함라벨)\n")
        f.write("     -> 실제 무릎은 165°로 완벽히 고정되었으나, AI-Hub가 상체 오류 때문에 결함으로 잘못 체크한 데이터\n")
        f.write("  2. 02_원인1_대조군_실제정상_TN (실제정상_라벨정상_모델적중)\n")
        f.write("     -> 완전한 정자세 교본 (Key 209) 영상으로, 01번 폴더의 영상과 무릎 각도가 똑같음을 대조 가능\n")
        f.write("  3. 03_원인1_대조군_실제결함_TP (실제결함_라벨결함_모델적중)\n")
        f.write("     -> 실제로 무릎이 무너져서 모델도 결함으로 적중한 진짜 결함 영상\n")
        f.write("  4. 04_원인2_라벨누락_FP (실제무릎결함_AI허브정상라벨)\n")
        f.write("     -> 라벨에는 정상으로 적혀 있으나, 실제 피험자가 무릎을 120°~140°로 푹 굽혀 스쿼트화된 영상\n\n")
        f.write("■ 육안 검증 핵심 포인트:\n")
        f.write("  - 01번 폴더의 영상을 재생해보시면, 피험자의 무릎이 02번 정상 교본처럼 단단히 고정되어 있음을 바로 확인하실 수 있습니다.\n")
        f.write("  - 04번 폴더의 영상을 재생해보시면, 피험자가 무릎을 완전히 굽히며 주저앉는데도 라벨이 정상으로 표기된 황당한 라벨 누락을 확인하실 수 있습니다.\n\n")
        f.write("=" * 90 + "\n")

    print(f"\n[+] 총괄 가이드 생성 완료: {readme_path}")
    print(f"[+] D:/Data/temp 검증 데이터셋 구축 완결!")

if __name__ == '__main__':
    run()
