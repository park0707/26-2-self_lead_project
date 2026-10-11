# -*- coding: utf-8 -*-
"""
yt-dlp 기반 유튜브 메타데이터 무인 검색 및 720p 임시 MP4 비디오 다운로더
"""
import os
import glob
from typing import List, Dict, Any, Optional
import yt_dlp

class YouTubeVideoDownloader:
    """yt-dlp를 활용한 고효율 검색 및 표적 스트림 다운로더"""

    def __init__(self, temp_dir: str = "d:/Data/temp_youtube_videos"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def search_candidates(
        self,
        query: str,
        max_results: int = 5,
        min_duration: int = 10,
        max_duration: int = 180
    ) -> List[Dict[str, Any]]:
        """검색어 기반 메타데이터만 신속 획득 (비디오 다운로드 X)"""
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'skip_download': True,
            'no_warnings': True
        }

        search_target = f"ytsearch{max_results * 5}:{query}"
        candidates = []

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(search_target, download=False)
                if 'entries' in info:
                    for entry in info['entries']:
                        if not entry:
                            continue
                        dur = entry.get('duration') or 0
                        title = entry.get('title', '')
                        vid = entry.get('id', '')
                        url = entry.get('url') or f"https://www.youtube.com/watch?v={vid}"

                        # 길이 필터 (10초 ~ 180초)
                        if dur < min_duration or dur > max_duration:
                            continue

                        # 제목 블랙리스트
                        t_lower = title.lower()
                        if any(bad in t_lower for bad in ['fail', 'meme', 'compilation', 'shorts', 'podcast']):
                            continue

                        candidates.append({
                            'id': vid,
                            'title': title,
                            'url': url,
                            'duration': dur,
                            'uploader': entry.get('uploader', '')
                        })
                        if len(candidates) >= max_results:
                            break
            except Exception as e:
                print(f"[-] Search error for query '{query}': {e}")

        return candidates

    def download_video(self, video_url_or_id: str, max_height: int = 720) -> Optional[str]:
        """지정된 비디오를 720p 이하 MP4 포맷으로 임시 디렉토리에 다운로드"""
        if not video_url_or_id.startswith('http'):
            video_url = f"https://www.youtube.com/watch?v={video_url_or_id}"
            vid = video_url_or_id
        else:
            video_url = video_url_or_id
            vid = video_url_or_id.split('v=')[-1].split('&')[0]

        out_template = os.path.join(self.temp_dir, f"{vid}.%(ext)s")

        ydl_opts = {
            # 오디오 병합(ffmpeg 필요) 없이 순수 비디오 스트림 다운로드 (OpenCV 분석에 오디오 불필요)
            'format': f'bestvideo[height<={max_height}][ext=mp4]/best[height<={max_height}][ext=mp4]/bestvideo[height<={max_height}]/best',
            'outtmpl': out_template,
            'quiet': True,
            'no_warnings': True,
            'overwrites': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
                # 생성된 파일 경로 찾기
                matches = glob.glob(os.path.join(self.temp_dir, f"{vid}.*"))
                for m in matches:
                    if m.endswith(('.mp4', '.mkv', '.webm')):
                        return os.path.abspath(m)
            except Exception as e:
                print(f"[-] Download error for '{video_url}': {e}")
                return None
        return None

    def cleanup_video(self, file_path: str):
        """추출 완료된 동영상 파일 즉시 영구 삭제"""
        if file_path and os.path.exists(file_path):
            import gc
            import time
            gc.collect()
            for attempt in range(5):
                try:
                    os.remove(file_path)
                    return
                except Exception as e:
                    time.sleep(0.2)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[-] Failed to delete temp video {file_path}: {e}")
