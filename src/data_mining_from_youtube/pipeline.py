# -*- coding: utf-8 -*-
"""
유튜브 피트니스 영상 검색->다운로드->품질검사->3D추출->반복분할->비디오삭제 마스터 파이프라인
"""
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional

from .search_queries import EXERCISE_QUERIES, get_search_query
from .downloader import YouTubeVideoDownloader
from .pose_extractor import YouTubePoseExtractor
from .quality_filter import YouTubeQualityFilter
from src.core.skeleton2d import Joint15
from src.preprocessing.rep_segmenter import RepetitionSegmenter

class YouTubeMiningPipeline:
    """무인 자동 유튜브 피트니스 데이터 마이닝 파이프라인"""

    def __init__(
        self,
        output_dir: str = "d:/Data/피처_데이터/YouTube_Mined_3D",
        temp_dir: str = "d:/Data/temp_youtube_videos"
    ):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.downloader = YouTubeVideoDownloader(temp_dir=self.temp_dir)
        self.extractor = YouTubePoseExtractor()
        self.filter = YouTubeQualityFilter()

    def mine_exercise(
        self,
        exercise_key: str,
        target_video_count: int = 2,
        max_candidates: int = 5
    ) -> List[Dict[str, Any]]:
        """특정 운동 종목에 대해 유튜브 검색부터 1회 반복 단위 3D 데이터셋 저장까지 수행"""
        ex_info = EXERCISE_QUERIES.get(exercise_key)
        if not ex_info:
            raise ValueError(f"Unknown exercise key: {exercise_key}")

        ex_idx = ex_info['idx']
        ex_name = ex_info['name_ko']
        print(f"[*] Starting mining for [{exercise_key}]: {ex_name}")

        query = get_search_query(exercise_key)
        print(f"[*] Search query: {query}")

        candidates = self.downloader.search_candidates(
            query=query,
            max_results=max_candidates,
            min_duration=10,
            max_duration=180
        )
        print(f"[*] Found {len(candidates)} candidate video(s).")

        saved_records = []
        success_count = 0

        for cand in candidates:
            if success_count >= target_video_count:
                break

            vid = cand['id']
            title = cand['title']
            print(f"\n--- Processing Video: {vid} | {title} ---")

            # 1. 720p 비디오 다운로드
            v_path = self.downloader.download_video(vid, max_height=720)
            if not v_path:
                print("[-] Download failed. Skipping.")
                continue

            try:
                # 2. MediaPipe 3D 관절 추출
                print(f"[*] Extracting 3D pose landmarks (MediaPipe Heavy)...")
                kps_3d, visibilities, fps = self.extractor.extract_from_video(v_path, apply_smoothing=True)

                if kps_3d.shape[0] == 0:
                    print("[-] Failed to extract any pose. Skipping.")
                    continue

                # 3. 4단계 품질 게이팅
                is_valid, reason, q_meta = self.filter.evaluate(kps_3d, visibilities)
                if not is_valid:
                    print(f"[-] Quality Gate Rejected: {reason}. Skipping.")
                    continue
                print(f"[+] Quality Gate Passed! Viewpoint: {q_meta.get('viewpoint')}, ROM Y: {q_meta.get('max_joint_rom_y'):.3f}m")

                # 4. 1회 반복 단위 분할 (Repetition Segmentation)
                # 3D world landmarks는 Pelvis가 (0,0,0)에 고정되어 있으므로,
                # 지면(발목) 대비 골반 상대 높이 궤적을 복원하여 분할기에 제공
                kps_seg = np.copy(kps_3d)
                ankle_y = (kps_3d[:, Joint15.LEFT_ANKLE, 1] + kps_3d[:, Joint15.RIGHT_ANKLE, 1]) / 2.0
                kps_seg[:, Joint15.PELVIS, 1] = - (ankle_y - kps_3d[:, Joint15.PELVIS, 1])
                segments = RepetitionSegmenter.segment_repetitions(kps_seg)

                if not segments or len(segments) == 0:
                    # 반복 분할 미감지 시 전체 시퀀스를 1개로 활용
                    segments = [(0, kps_3d.shape[0] - 1, kps_3d.shape[0] // 2)]

                print(f"[+] Detected {len(segments)} valid repetition(s).")

                # 5. 각 반복별 NPZ 및 JSON 메타데이터 저장
                ex_out_dir = os.path.join(self.output_dir, exercise_key)
                os.makedirs(ex_out_dir, exist_ok=True)

                for rep_idx, (start_f, end_f, peak_f) in enumerate(segments):
                    rep_kps = kps_3d[start_f:end_f + 1]
                    rep_vis = visibilities[start_f:end_f + 1]

                    if rep_kps.shape[0] < 15: # 0.5초 미만 너무 짧은 구간은 스킵
                        continue

                    file_stem = f"yt_{vid}_rep{rep_idx:02d}"
                    npz_path = os.path.join(ex_out_dir, f"{file_stem}.npz")
                    json_path = os.path.join(ex_out_dir, f"{file_stem}.json")

                    np.savez_compressed(
                        npz_path,
                        kps_3d=rep_kps,
                        visibilities=rep_vis,
                        exercise_idx=ex_idx,
                        exercise_name=exercise_key
                    )

                    meta_dict = {
                        'video_id': vid,
                        'title': title,
                        'uploader': cand.get('uploader'),
                        'duration': cand.get('duration'),
                        'exercise_idx': ex_idx,
                        'exercise_name': exercise_key,
                        'rep_index': rep_idx,
                        'start_frame': int(start_f),
                        'end_frame': int(end_f),
                        'peak_frame': int(peak_f),
                        'frames': int(rep_kps.shape[0]),
                        'fps': float(fps),
                        'quality': q_meta
                    }

                    with open(json_path, 'w', encoding='utf-8') as jf:
                        json.dump(meta_dict, jf, indent=2, ensure_ascii=False)

                    saved_records.append(meta_dict)

                success_count += 1
                print(f"[+] Successfully saved {len(segments)} reps from video {vid}.")

            finally:
                # 6. 임시 비디오 파일 즉시 영구 삭제 (스토리지 절약)
                self.downloader.cleanup_video(v_path)
                print(f"[*] Cleaned up temp video: {os.path.basename(v_path)}")

        return saved_records
