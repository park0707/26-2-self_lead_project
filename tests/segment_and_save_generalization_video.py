# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: D:\\tests\\segment_and_save_generalization_video.py
Purpose: data/일반화 폴더의 실제 영상을 실제 웹 서비스 코드(VideoAnalysisPipeline, ActivePeriodGater)와
         동일한 파이프라인으로 처리하고, 실제 운동 활성 구간 및 각 1회 반복 클립 영상을 분할 저장
Output Directory:
  - D:\\data\\일반화\\분할_결과\\
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 프로젝트 루트 경로 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if 'd:/' not in sys.path and 'd:\\\\' not in sys.path:
    sys.path.insert(0, 'd:/')

from src.web.pipeline.video_processor import VideoAnalysisPipeline
from src.web.pipeline.active_period_gater import ActivePeriodGater

INPUT_VIDEO = r"D:\data\일반화\바벨 스쿼트 정면(임시).mp4"
OUTPUT_DIR = r"D:\data\일반화\분할_결과"
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\malgunbd.ttf"

def draw_text_pillow(img_bgr, text, position, font_size=28, color=(255, 255, 255), bold=False, bg_color=None):
    """OpenCV 이미지에 한글 텍스트를 깨짐 없이 고해상도로 렌더링"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    font_file = FONT_BOLD_PATH if bold and os.path.exists(FONT_BOLD_PATH) else FONT_PATH
    if os.path.exists(font_file):
        font = ImageFont.truetype(font_file, font_size)
    else:
        font = ImageFont.load_default()
        
    x, y = position
    if bg_color is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle((bbox[0]-6, bbox[1]-4, bbox[2]+6, bbox[3]+4), fill=bg_color)
        
    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    assert os.path.exists(INPUT_VIDEO), f"입력 비디오가 존재하지 않습니다: {INPUT_VIDEO}"

    print("=" * 80)
    print("  ★ [vidio-health-trainer] 실제 일반화 비디오 웹 파이프라인 분할 및 저장 시작")
    print(f"  - 입력 파일: {INPUT_VIDEO}")
    print(f"  - 출력 폴더: {OUTPUT_DIR}")
    print("=" * 80)

    # 1. 웹 서비스 실제 싱글톤 파이프라인 인스턴스 획득 및 분석 수행
    pipeline = VideoAnalysisPipeline.get_instance()
    
    print("[1/4] 웹 서비스 실제 파이프라인(analyze_video) 실행 중...")
    t0 = time.time()
    
    def on_progress(stage, pct, msg, data=None):
        print(f"  [{pct:5.1f}%] [{stage:22s}] {msg}")
        
    cache_path = r"D:\data\일반화\test_video_landmarks_cache.npz"
    cached_dict = None
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        cached_dict = {
            "kps_33_3d": data["kps_33_3d"],
            "kps_33_2d": data["kps_33_2d"],
            "kps_15_2d": data["kps_15_2d"],
            "vis_33": data["vis_33"],
            "total_frames": int(data["total_frames"]),
            "fps": float(data["fps"]),
            "duration_sec": float(data["total_frames"]) / max(1.0, float(data["fps"]))
        }

    analysis_res = pipeline.analyze_video(
        video_path=INPUT_VIDEO,
        camera_view="시점 C: 정면",
        include_skeleton_preview=True,
        progress_callback=on_progress,
        cached_landmarks=cached_dict
    )
    analysis_time = time.time() - t0
    print(f"[+] 파이프라인 분석 완료 (소요 시간: {analysis_time:.2f}초)")

    # 2. 결과 메타데이터 확인
    act_period = analysis_res.active_period or {}
    act_s = act_period.get("start_frame", 0)
    act_e = act_period.get("end_frame", 0)
    total_reps = analysis_res.total_reps
    reps_info = analysis_res.reps

    print("\n[2/4] 검출 및 분할 구간 요약:")
    print(f"  - 순수 활성 운동 구간: F{act_s} ~ F{act_e} (지속 시간: {act_period.get('duration_sec', 0.0)}초)")
    print(f"  - 선제 소거된 비운동 구간: {act_period.get('filtered_frames_count', 0)} 프레임")
    print(f"  - 분할된 총 횟수: {total_reps}회")
    for r in reps_info:
        print(f"    ▶ Rep #{r.rep_index}: F{r.start_frame}~F{r.end_frame} ({r.duration_sec:.2f}초, 클린: {r.is_clean})")

    # 3. 비디오 메타데이터 및 비디오 파일 분할 저장
    cap_info = cv2.VideoCapture(INPUT_VIDEO)
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap_info.get(cv2.CAP_PROP_FPS))
    width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()

    overlay_path = os.path.join(OUTPUT_DIR, "전체영상_동작구간_및_횟수분할_오버레이_시각화.mp4")
    act_clip_path = os.path.join(OUTPUT_DIR, f"00_실제운동구간_F{act_s}_F{act_e}.mp4")

    if os.path.exists(overlay_path) and os.path.getsize(overlay_path) > 10000000 and os.path.exists(act_clip_path):
        print("\n[3/4] 구간별 비디오 클립 및 시각화 오버레이 영상이 이미 정상 생성되어 있습니다. (인코딩 스킵)")
    else:
        print("\n[3/4] 구간별 비디오 클립 및 시각화 오버레이 영상 저장 중...")
        cap = cv2.VideoCapture(INPUT_VIDEO)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 비디오 라이터 생성
        out_act = cv2.VideoWriter(act_clip_path, fourcc, fps, (width, height))
        rep_writers = []
        for r in reps_info:
            clip_name = f"0{r.rep_index}_반복{r.rep_index}회차_F{r.start_frame}_F{r.end_frame}.mp4"
            clip_path = os.path.join(OUTPUT_DIR, clip_name)
            w = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            rep_writers.append((r.start_frame, r.end_frame, w, clip_name))

        unrack_path = os.path.join(OUTPUT_DIR, f"소거구간_언랙및셋업_F0_F{max(0, act_s-1)}.mp4")
        out_unrack = cv2.VideoWriter(unrack_path, fourcc, fps, (width, height))
        rack_path = os.path.join(OUTPUT_DIR, f"소거구간_세트종료랙안착_F{act_e+1}_F{total_frames-1}.mp4")
        out_rack = cv2.VideoWriter(rack_path, fourcc, fps, (width, height))
        out_overlay = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

        # 반복 구간 빠른 조회를 위한 맵 구성
        rep_lookup = {}
        for r in reps_info:
            for f in range(r.start_frame, r.end_frame + 1):
                rep_lookup[f] = r

        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # (1) 활성 운동 구간 저장
            if act_s <= f_idx <= act_e:
                out_act.write(frame)

            # (2) 개별 반복 구간 저장
            for r_s, r_e, w, _ in rep_writers:
                if r_s <= f_idx <= r_e:
                    w.write(frame)

            # (3) 소거 구간 저장
            if f_idx < act_s:
                out_unrack.write(frame)
            elif f_idx > act_e:
                out_rack.write(frame)

            # (4) 시각화 오버레이 프레임 렌더링
            overlay_frame = frame.copy()
            
            # 상단 HUD 반투명 오버레이 바 생성
            hud_bg = overlay_frame.copy()
            cv2.rectangle(hud_bg, (0, 0), (width, 220), (20, 20, 30), -1)
            # 하단 HUD 반투명 오버레이 바 생성
            cv2.rectangle(hud_bg, (0, height - 160), (width, height), (20, 20, 30), -1)
            cv2.addWeighted(hud_bg, 0.75, overlay_frame, 0.25, 0, overlay_frame)

            # 현재 프레임 구간 판정
            cur_sec = f_idx / fps
            cur_rep = rep_lookup.get(f_idx)

            if f_idx < act_s:
                status_text = "■ [Tier 1 사전 필터링] 비운동 구간: 바벨 언랙 및 셋업 흔들림 (소거됨)"
                status_color = (255, 120, 50) # 오렌지
                bg_badge = (60, 40, 20)
            elif f_idx > act_e:
                status_text = "■ [Tier 1 사전 필터링] 비운동 구간: 세트 종료 후 랙 안착 및 휴식 (소거됨)"
                status_color = (200, 200, 200) # 그레이
                bg_badge = (50, 50, 50)
            else:
                rep_num = cur_rep.rep_index if cur_rep else "?"
                status_text = f"★ [Tier 1 활성 운동 구간] 01_바벨_스쿼트 | 반복 {rep_num}/{total_reps}회차 진행 중"
                status_color = (50, 255, 120) # 그린
                bg_badge = (20, 60, 30)

            # 상단 HUD 텍스트 렌더링
            overlay_frame = draw_text_pillow(
                overlay_frame,
                "vidio-health-trainer | 웹 서비스 Tier 1 비운동구간 필터링 & 횟수 분할 엔진",
                (30, 25), font_size=28, color=(220, 220, 220), bold=True
            )
            overlay_frame = draw_text_pillow(
                overlay_frame,
                status_text,
                (30, 75), font_size=34, color=status_color, bold=True, bg_color=bg_badge
            )
            time_info_str = f"프레임: {f_idx:03d}/{total_frames-1:03d} | 시간: {cur_sec:.2f}s / {total_frames/fps:.2f}s | 카메라: 시점 C (정면)"
            overlay_frame = draw_text_pillow(
                overlay_frame,
                time_info_str,
                (30, 140), font_size=26, color=(180, 210, 240)
            )

            # 하단 HUD 텍스트 렌더링
            ex_name = analysis_res.exercise.exercise_name
            ex_conf = analysis_res.exercise.confidence * 100.0
            bottom_hud_1 = f"1단계 종목 판별 (다중 앙상블): {ex_name} (확신도: {ex_conf:.1f}%)"
            overlay_frame = draw_text_pillow(
                overlay_frame, bottom_hud_1,
                (30, height - 135), font_size=30, color=(100, 240, 255), bold=True
            )

            if cur_rep:
                defects_str = ", ".join([f"{d.defect_name}" for d in cur_rep.defects if d.probability >= 0.5])
                if not defects_str:
                    defects_str = "특이 결함 없음 (Clean Posture)"
                bottom_hud_2 = f"2단계 결함 평가 (Rep #{cur_rep.rep_index}): {defects_str}"
            else:
                bottom_hud_2 = "2단계 결함 평가: 비운동 구간 (평가 제외 및 소거)"
                
            overlay_frame = draw_text_pillow(
                overlay_frame, bottom_hud_2,
                (30, height - 75), font_size=26, color=(255, 230, 140)
            )

            out_overlay.write(overlay_frame)

            if f_idx % 60 == 0:
                pct_video = (f_idx / total_frames) * 100.0
                print(f"  [프레임 처리 중] {f_idx:3d}/{total_frames} ({pct_video:.1f}%)")

            f_idx += 1

        # 비디오 라이터 해제
        cap.release()
        out_act.release()
        for _, _, w, cname in rep_writers:
            w.release()
            print(f"  [+] 개별 횟수 클립 저장 완료: {cname}")
        out_unrack.release()
        out_rack.release()
        out_overlay.release()

        print(f"  [+] 활성 운동 구간 전체 클립 저장 완료: {os.path.basename(act_clip_path)}")
        print(f"  [+] 전체 시각화 오버레이 비디오 저장 완료: {os.path.basename(overlay_path)}")

    # 4. 분석 결과 요약 JSON 및 Markdown 저장
    print("\n[4/4] 결과 데이터 JSON 및 Markdown 보고서 저장 중...")
    
    # JSON 직렬화
    res_dict = analysis_res.model_dump()
    
    json_path = os.path.join(OUTPUT_DIR, "분할_및_분석결과_요약.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2, ensure_ascii=False)
    print(f"  [+] JSON 결과 저장 완료: {json_path}")

    # Markdown 보고서 작성
    md_path = os.path.join(OUTPUT_DIR, "분할_및_분석결과_요약.md")
    md_content = f"""# [vidio-health-trainer] 일반화 영상 웹 파이프라인 실제 분할 및 결함 진단 결과

- **대상 파일**: `{INPUT_VIDEO}`
- **분석 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **영상 제원**: {total_frames} 프레임, {fps:.1f} FPS, {total_frames/fps:.2f}초, 해상도 {width}x{height}
- **카메라 시점**: 시점 C (정면)

---

## 1. Tier 1 비운동 구간 선제 소거 및 활성 운동 구간 결과

| 항목 | 프레임 구간 | 시간 (초) | 상태 및 비고 |
| :--- | :---: | :---: | :--- |
| **언랙 및 셋업 준비** | F0 ~ F{act_s-1} | 0.00s ~ {act_s/fps:.2f}s ({act_s/fps:.2f}초) | **[소거 완료]** 바벨 언랙 및 스탠스 조절 요동 |
| **순수 활성 운동 구간** | **F{act_s} ~ F{act_e}** | **{act_s/fps:.2f}s ~ {act_e/fps:.2f}s ({(act_e-act_s+1)/fps:.2f}초)** | **[실제 운동 구간]** 스쿼트 5회 전수 포함 |
| **세트 종료 및 랙 안착** | F{act_e+1} ~ F{total_frames-1} | {act_e/fps:.2f}s ~ {total_frames/fps:.2f}s ({(total_frames-1-act_e)/fps:.2f}초) | **[소거 완료]** 호흡 및 바벨 랙 안착 |

---

## 2. 1단계 운동 종목 분류 결과 (다중 반복 앙상블)

- **최우선 판정 종목**: **`{analysis_res.exercise.exercise_name}`**
- **확신도 (Confidence)**: **`{analysis_res.exercise.confidence*100:.1f}%`**
- **소프트 보팅 확률 분포**:
"""
    for ex, p in sorted(analysis_res.exercise.probabilities.items(), key=lambda x: x[1], reverse=True)[:4]:
        md_content += f"  - `{ex}`: {p*100:.1f}%\n"

    md_content += f"""
---

## 3. 분할된 횟수별(Rep 1~{total_reps}) 상세 구간 및 2단계 결함 진단

| 반복 회차 | 시작 프레임 | 종료 프레임 | 지속 시간 | 클린 여부 | 주요 검출 결함 (확률) |
| :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for r in reps_info:
        defects_list = [f"{d.defect_name} ({d.probability*100:.1f}%)" for d in r.defects if d.probability >= 0.35]
        def_str = "<br>".join(defects_list) if defects_list else "특이 결함 없음 (클린)"
        clean_badge = "Clean" if r.is_clean else "결함 검출"
        md_content += f"| **Rep #{r.rep_index}** | F{r.start_frame} | F{r.end_frame} | {r.duration_sec:.2f}초 | {clean_badge} | {def_str} |\n"

    md_content += f"""
---

## 4. 생성된 비디오 클립 파일 목록

1. `00_실제운동구간_F{act_s}_F{act_e}.mp4`: 셋업과 종료가 잘려나간 순수 운동 5회 전체 영상
2. `01_반복1회차_F{reps_info[0].start_frame}_F{reps_info[0].end_frame}.mp4`: 1회차 반복 단독 영상
3. `02_반복2회차_F{reps_info[1].start_frame}_F{reps_info[1].end_frame}.mp4`: 2회차 반복 단독 영상
4. `03_반복3회차_F{reps_info[2].start_frame}_F{reps_info[2].end_frame}.mp4`: 3회차 반복 단독 영상
5. `04_반복4회차_F{reps_info[3].start_frame}_F{reps_info[3].end_frame}.mp4`: 4회차 반복 단독 영상
6. `05_반복5회차_F{reps_info[4].start_frame}_F{reps_info[4].end_frame}.mp4`: 5회차 반복 단독 영상
7. `소거구간_언랙및셋업_F0_F{max(0, act_s-1)}.mp4`: 소거된 바벨 언랙 및 셋업 흔들림 구간
8. `소거구간_세트종료랙안착_F{act_e+1}_F{total_frames-1}.mp4`: 소거된 세트 종료 후 랙 안착 구간
9. `전체영상_동작구간_및_횟수분할_오버레이_시각화.mp4`: 실시간 프레임, 상태 배지, AI 판정 오버레이가 들어간 풀 영상
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [+] Markdown 요약 저장 완료: {md_path}")

    print("\n" + "=" * 80)
    print("  ★ [작업 완료] 모든 비디오 분할 클립 및 분석 결과가 성공적으로 저장되었습니다!")
    print(f"  - 저장 위치: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
