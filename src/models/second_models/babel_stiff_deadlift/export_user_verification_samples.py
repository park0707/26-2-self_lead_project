# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Exercise: 바벨 스티프 데드리프트 (03_바벨_스티프_데드리프트)
Script: export_user_verification_samples.py
Purpose: 사용자가 직접 육안으로 비교 분석할 수 있도록 D:/Data/temp 에
         1. 랜덤 정상 데이터 (2개)
         2. 랜덤 오류 데이터 (2개)
         3. 라벨 정상 & 모델 예측 정상 (2개)
         4. 라벨 오류 & 모델 예측 오류 (2개)
         의 MP4 비디오, 프레임, 상세 진단서를 렌더링하여 배포
"""

import os
import sys
import glob
import json
import shutil
import tempfile
import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sd1_classifier import SD1Classifier
from stiff_deadlift_knee_features_v3 import extract_knee_invariance_features_10d
from src.models.first_model.dataset import Stage1DatasetLoader

TEMP_ROOT = r"D:\Data\temp"
RAW_IMG_ROOT = r"D:\Data\01_raw_data\images\03_바벨_스티프_데드리프트"

VIEW_MAP = {
    "시점A": ("view5", "시점 A (우측 후방 45°)"),
    "시점B": ("view2", "시점 B (우측 전방 45°)"),
    "시점C": ("view3", "시점 C (정면 0°)"),
    "시점D": ("view4", "시점 D (좌측 전방 45°)"),
    "시점E": ("view1", "시점 E (좌측 후방 45°)")
}

KEY_DESCRIPTIONS = {
    "Key209": "스티프 데드리프트 정상 동작",
    "Key210": "오류: 바벨을 몸에 밀착하지 않고 앞으로 멀리 떨어뜨림",
    "Key211": "오류: 척추 중립을 무너뜨리고 등을 굽히거나 과신전",
    "Key212": "오류: 시선 처리 및 턱 들림 오류",
    "Key213": "오류: 무릎 구부린 채 고정 위반 (무릎이 스쿼트/루마니안처럼 굽혀짐)",
    "Key214": "오류: 골반 힌지가 미흡하고 허리로만 숙임",
    "Key215": "오류: 바벨을 수평으로 유지하지 못하고 좌우로 기울임",
    "Key216": "오류: 일어설 때 엉덩이가 먼저 과도하게 들리는 굿모닝 패턴"
}

def imread_unicode(path):
    """Windows 한글 경로 지원 이미지 읽기"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def create_mp4(frame_paths, out_mp4_path, fps=6):
    """프레임 리스트를 연속 MP4 동영상으로 인코딩"""
    if not frame_paths:
        return False
    frame_paths.sort()
    first_img = imread_unicode(frame_paths[0])
    if first_img is None:
        return False
    h, w, _ = first_img.shape
    
    temp_dir = tempfile.gettempdir()
    temp_mp4 = os.path.join(temp_dir, f"temp_{np.random.randint(10000, 99999)}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_mp4, fourcc, fps, (w, h))
    for fp in frame_paths:
        img = imread_unicode(fp)
        if img is not None:
            writer.write(img)
    writer.release()
    
    if os.path.exists(temp_mp4):
        shutil.move(temp_mp4, out_mp4_path)
        return True
    return False

def find_raw_folder(sid):
    """시퀀스 ID에서 실제 이미지 폴더명 검색"""
    # sid 예: [오류]_D02_피험자1_Key213_view2
    parts = sid.split('_')
    # view 부분 제외
    folder_prefix = '_'.join([p for p in parts if not p.startswith('view')])
    
    if os.path.exists(os.path.join(RAW_IMG_ROOT, folder_prefix)):
        return os.path.join(RAW_IMG_ROOT, folder_prefix), folder_prefix
    
    # 유사 폴더명 검색
    for name in os.listdir(RAW_IMG_ROOT):
        if folder_prefix in name or (parts[0] in name and parts[2] in name and parts[3] in name):
            return os.path.join(RAW_IMG_ROOT, name), name
            
    return None, None

def export_single_case(target_category_dir, case_idx, sid, y_label, y_pred_prob, diag_state, diag_label, x_seq):
    """단일 케이스에 대해 폴더 생성, MP4 비디오 렌더링, 프레임 복사 및 상세 진단서 작성"""
    folder_path, folder_name = find_raw_folder(sid)
    if folder_path is None:
        print(f"[!] 원천 폴더 미발견: {sid}")
        return False

    case_folder_name = f"Case_{case_idx}_{folder_name}"
    case_dir = os.path.join(target_category_dir, case_folder_name)
    os.makedirs(case_dir, exist_ok=True)

    # 1. 시점별 프레임 분할 및 MP4 렌더링
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    # 5개 시점별 처리
    for v_hangul, (v_code, v_alias_full) in VIEW_MAP.items():
        v_frames = [os.path.join(folder_path, f) for f in all_files if (v_hangul in f or v_code in f)]
        if v_frames:
            # 프레임 폴더
            frames_dir = os.path.join(case_dir, f"frames_{v_hangul}_{v_code}")
            os.makedirs(frames_dir, exist_ok=True)
            for f in v_frames:
                shutil.copy2(f, os.path.join(frames_dir, os.path.basename(f)))
            
            # MP4 비디오 생성
            mp4_path = os.path.join(case_dir, f"비디오_{v_hangul}_{v_code}.mp4")
            create_mp4(v_frames, mp4_path, fps=6)

    # 2. 생체역학 지표 산출
    feats_10 = extract_knee_invariance_features_10d(x_seq)
    bot_ang = feats_10[0]
    bot_delta = feats_10[1]
    femur_rom = feats_10[2]
    disp_y = feats_10[3]
    shank_svii = feats_10[4]
    hhir = feats_10[5]
    rel_delta = feats_10[6]
    norm_disp_y = feats_10[7]
    addr_ang = feats_10[8]

    # Key 정보 파싱
    key_code = [p for p in sid.split('_') if p.startswith('Key')][0] if any(p.startswith('Key') for p in sid.split('_')) else "알수없음"
    key_desc = KEY_DESCRIPTIONS.get(key_code, "정보 없음")

    # 3. 상세 진단 텍스트 작성
    report_text = f"""================================================================================
  [vidio-health-trainer] 스티프 데드리프트 sd1 육안 검증 케이스 상세 진단서
================================================================================

1. 시퀀스 기본 정보
  • 시퀀스 ID : {sid}
  • 원천 폴더 : {folder_name}
  • AI-Hub 수행 동작 코드 : {key_code} ({key_desc})

2. 정답 라벨 vs 모델 예측 대조
  • AI-Hub 공식 라벨 (y) : {'[1] 결함 (Knee Bend Violation)' if y_label == 1 else '[0] 정상 (Normal)'}
  • 모델 결함 확신도 P(Defect) : {y_pred_prob*100:.2f}%
  • 3-State 최종 진단 상태 : {diag_label} (State: {diag_state})
  • 판정 기준 임계값 : tau_lower=0.4286 (정상 한계), tau_upper=0.5714 (결함 한계)

3. 실측 생체역학 지표 (Biomechanics Kinematics)
  • 어드레스 초기 무릎 각도 : {addr_ang:.1f}° (개인별 햄스트링 가동성 기준선)
  • 최저점(Bottom) 무릎 각도 : {bot_ang:.1f}°
  • 무릎 각도 변화량 (Δθ) : {bot_delta:.1f}°
  • 상대 굴곡 변화율 (Δθ/θ_0) : {rel_delta*100:.2f}%
  • 대퇴골 지면 경사각 ROM : {femur_rom:.1f}°
  • 무릎 수직 이동 가동범위 : {disp_y:.3f}
  • 정강이 수직 불변성 지수(SVII) : {shank_svii:.1f}° (지면 수직 유지도)
  • 고관절 힌지 격리비(HHIR) : {hhir:.2f}

4. 사용자 육안 관찰 포인트 (Checkpoints for Human Verification)
  [무릎 관절 움직임 관찰]
  - 최하단으로 상체가 내려갈 때 무릎이 스쿼트처럼 앞으로 튀어나오며 접히는지,
    아니면 약 15° 정도 살짝 굽힌 상태(Soft Knee Lock)를 그대로 유지하며 엉덩이만 뒤로 빠지는지 확인하세요.
  
  [바벨 및 상체 움직임 관찰]
  - 바벨이 지면과 평행을 유지하고 있는지, 좌우로 기울어지지는 않았는지 확인하세요.
  - 등이 둥글게 말리지 않고 척추 중립을 유지하는지 확인하세요.
================================================================================
"""
    txt_path = os.path.join(case_dir, f"[상세진단]_{folder_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"  [+] 케이스 내보내기 완료: {case_folder_name}")
    return True

def main():
    print("=" * 80)
    print("  [vidio-health-trainer] 육안 검증용 4대 카테고리 샘플 데이터 구축 시작")
    print("=" * 80)

    # 1. 모델 로드
    clf = SD1Classifier()

    # 2. 데이터셋 로드
    cache_path = os.path.join(CURRENT_DIR, "stiff_deadlift_knee_ambiguity_levels.npz")
    cache_data = np.load(cache_path, allow_pickle=True)
    orig_labels = cache_data['labels']
    sequence_ids = cache_data['sequence_ids']
    l4_mask = cache_data['l4_mask']

    splits = Stage1DatasetLoader.load_all_dual_splits()
    seen_sids = {}
    for sname, sdata in splits.items():
        stiff_mask = (sdata['y_action'] == 2)
        s_sids = sdata['sequence_ids'][stiff_mask]
        s_X = sdata['X_Format_B'][stiff_mask]
        for sid, x in zip(s_sids, s_X):
            seen_sids[str(sid)] = x

    valid_indices = np.where(~l4_mask)[0]
    print(f"[*] 유효 시퀀스 수: {len(valid_indices)}개")

    # 3. 전수 추론 수행
    all_probs = []
    all_states = []
    all_state_labels = []

    print("[*] 공식 모델(sd1_v3) 전수 추론 진행 중...")
    for idx in valid_indices:
        sid = str(sequence_ids[idx])
        x_seq = seen_sids[sid]
        diag = clf.diagnose(x_seq)
        all_probs.append(diag['probability'])
        all_states.append(diag['state_3state'])
        all_state_labels.append(diag['state_label'])

    all_probs = np.array(all_probs)
    all_states = np.array(all_states)
    valid_labels = orig_labels[valid_indices]

    # 4. 카테고리별 후보군 필터링 (서로 다른 세션으로 각 2개씩 선정)
    np.random.seed(42)

    def get_base_session(sid):
        parts = str(sid).split('_')
        return '_'.join([p for p in parts if not p.startswith('view')])

    # 카테고리 1: 완전 정상 데이터 (원천 폴더명에 [정상] 및 Key209 포함 시퀀스 중 무작위 2선)
    cand_cat1 = [i for i, idx in enumerate(valid_indices) if "정상" in str(sequence_ids[idx]) and "Key209" in str(sequence_ids[idx])]
    if len(cand_cat1) < 2:
        cand_cat1 = list(np.where(valid_labels == 0)[0])
    np.random.shuffle(cand_cat1)
    sel_cat1 = []
    seen_sess1 = set()
    for c in cand_cat1:
        s = get_base_session(sequence_ids[valid_indices[c]])
        if s not in seen_sess1:
            seen_sess1.add(s)
            sel_cat1.append(c)
            if len(sel_cat1) == 2:
                break

    # 카테고리 2: 랜덤 오류 데이터 (AI-Hub 라벨 == 1 중 서로 다른 세션 2선)
    cand_cat2 = list(np.where(valid_labels == 1)[0])
    np.random.shuffle(cand_cat2)
    sel_cat2 = []
    seen_sess2 = set()
    for c in cand_cat2:
        s = get_base_session(sequence_ids[valid_indices[c]])
        if s not in seen_sess2:
            seen_sess2.add(s)
            sel_cat2.append(c)
            if len(sel_cat2) == 2:
                break

    # 카테고리 3: 라벨 정상 & 모델 예측 정상 (y == 0 and prob < tau_lower, 서로 다른 세션)
    cand_cat3 = np.where((valid_labels == 0) & (all_probs < clf.tau_lower))[0]
    sort_cat3 = cand_cat3[np.argsort(all_probs[cand_cat3])]
    sel_cat3 = []
    seen_sess3 = set()
    for c in sort_cat3:
        s = get_base_session(sequence_ids[valid_indices[c]])
        if s not in seen_sess3:
            seen_sess3.add(s)
            sel_cat3.append(c)
            if len(sel_cat3) == 2:
                break

    # 카테고리 4: 라벨 오류 & 모델 예측 오류 (y == 1 and prob >= tau_upper, 서로 다른 세션)
    cand_cat4 = np.where((valid_labels == 1) & (all_probs >= clf.tau_upper))[0]
    sort_cat4 = cand_cat4[np.argsort(-all_probs[cand_cat4])]
    sel_cat4 = []
    seen_sess4 = set()
    for c in sort_cat4:
        s = get_base_session(sequence_ids[valid_indices[c]])
        if s not in seen_sess4:
            seen_sess4.add(s)
            sel_cat4.append(c)
            if len(sel_cat4) == 2:
                break

    # 5. D:\Data\temp 폴더 초기화 및 재구성
    os.makedirs(TEMP_ROOT, exist_ok=True)
    # 기존 바벨데드리프트 관련 폴더를 제외하고 sd1 관련 폴더 정리
    for item in os.listdir(TEMP_ROOT):
        item_path = os.path.join(TEMP_ROOT, item)
        if item.startswith("0") or item.startswith("Case_") or "스티프" in item:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    categories = [
        ("01_랜덤_정상_데이터", sel_cat1, "AI-Hub 라벨 기준 정상으로 표기된 데이터 (무작위 2선)"),
        ("02_랜덤_오류_데이터", sel_cat2, "AI-Hub 라벨 기준 결함으로 표기된 데이터 (무작위 2선)"),
        ("03_라벨정상_모델예측정상_데이터", sel_cat3, "AI-Hub 라벨 [정상] & 모델 예측 [정상 확정] (True Negative 2선)"),
        ("04_라벨오류_모델예측오류_데이터", sel_cat4, "AI-Hub 라벨 [결함] & 모델 예측 [결함 확정] (True Positive 2선)")
    ]

    summary_records = []

    for cat_name, sel_indices, cat_desc in categories:
        cat_dir = os.path.join(TEMP_ROOT, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"\n[{cat_name}] 구축 시작 ({cat_desc})")

        for c_i, v_idx in enumerate(sel_indices, start=1):
            real_idx = valid_indices[v_idx]
            sid = str(sequence_ids[real_idx])
            y_l = int(valid_labels[v_idx])
            p_val = float(all_probs[v_idx])
            s_val = int(all_states[v_idx])
            s_lbl = str(all_state_labels[v_idx])
            x_seq = seen_sids[sid]

            ok = export_single_case(cat_dir, c_i, sid, y_l, p_val, s_val, s_lbl, x_seq)
            if ok:
                summary_records.append({
                    "category": cat_name,
                    "case_idx": c_i,
                    "sid": sid,
                    "label": "정상" if y_l == 0 else "오류",
                    "prob": f"{p_val*100:.1f}%",
                    "diagnosis": s_lbl
                })

    # 6. 전체 가이드 텍스트 생성
    guide_text = f"""================================================================================
  [vidio-health-trainer] 스티프 데드리프트 sd1 육안 검증 샘플 가이드
================================================================================

본 디렉토리(D:/Data/temp)는 사용자가 스티프 데드리프트 동작의 '무릎 고정 여부'와 
'AI-Hub 라벨의 실제 상태'를 직접 동영상으로 비교·확인할 수 있도록 4개 카테고리로 구축되었습니다.

[배치된 4개 카테고리 폴더 구성]
1. 01_랜덤_정상_데이터/
   - AI-Hub 라벨이 '정상(0)'으로 되어 있는 데이터 중 무작위 2선.
   - 실제로 무릎이 어떻게 고정되어 있는지 확인하는 기준점입니다.

2. 02_랜덤_오류_데이터/
   - AI-Hub 라벨이 '결함(1)'으로 되어 있는 데이터 중 무작위 2선.
   - 라벨은 오류이지만 실제로 무릎을 구부렸는지, 아니면 바벨/상체만 흔들렸는지 육안 확인 대상입니다.

3. 03_라벨정상_모델예측정상_데이터/ (True Negative)
   - AI-Hub 라벨 [정상] & 공식 모델(sd1_v3) 예측 [정상 확정] (결함 확률 < 42.86%).
   - 인공지능과 라벨이 완벽하게 일치한 깨끗한 모범 자세입니다.

4. 04_라벨오류_모델예측오류_데이터/ (True Positive)
   - AI-Hub 라벨 [결함] & 공식 모델(sd1_v3) 예측 [결함 확정] (결함 확률 >= 57.14%).
   - 인공지능이 무릎 관절 궤적에서 확실한 굴곡/흔들림 결함을 감지한 시퀀스입니다.

[동영상 시청 안내]
- 각 케이스 폴더 내에 시점별 MP4 비디오가 렌더링되어 있습니다:
  * 비디오_시점B_view2.mp4 : 우측 전방 45도 시점 (무릎 굽힘 및 바벨 궤적 확인에 가장 적합)
  * 비디오_시점C_view3.mp4 : 완전 정면 시점 (바벨 수평 및 척추 대칭 확인에 적합)
  * 비디오_시점D_view4.mp4 : 좌측 전방 45도 시점
  * 비디오_시점A_view5.mp4 & 비디오_시점E_view1.mp4 : 후방 시점
- [상세진단]_.txt 파일을 함께 열어보시면 실측 무릎 각도(시작 각도, 최저점 각도, 변화량)가 기록되어 있습니다.
================================================================================
"""
    guide_path = os.path.join(TEMP_ROOT, "[스티프데드리프트_샘플_육안검증_가이드].txt")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_text)

    print(f"\n[+] 전체 가이드 파일 생성 완료: {guide_path}")
    print("[+] 모든 샘플 배포가 완료되었습니다.")

if __name__ == '__main__':
    main()
