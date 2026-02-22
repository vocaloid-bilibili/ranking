# video/clip.py
"""视频片段生成"""

from pathlib import Path
from typing import List, Optional
import subprocess

import pandas as pd

from common.logger import logger
from bilibili.client import BilibiliClient
from video.climax import find_climax_segment
from video.overlay import build_overlay_cmd


class ClipGenerator:
    """视频片段生成器"""

    def __init__(
        self,
        *,
        api_client: BilibiliClient,
        output_dir: Path,
        font_file: str,
        ffmpeg_bin: str,
    ):
        self.api_client = api_client
        self.output_dir = output_dir
        self.font_file = font_file
        self.ffmpeg_bin = ffmpeg_bin

    def generate(
        self,
        row: pd.Series,
        clip_index: int,
        clip_duration: float,
        issue_date: str,
    ) -> Optional[Path]:
        """生成单个视频片段"""
        bvid = str(row.get("bvid", "")).strip()
        if not bvid:
            return None

        logger.info(f"处理 #{clip_index} | {row.get('title', '')}")

        segment = self._ensure_segment(bvid, clip_duration)
        if not segment:
            return None

        overlay_args, output_path = build_overlay_cmd(
            segment_path=segment,
            row=row,
            clip_index=clip_index,
            issue_date=issue_date,
            output_dir=self.output_dir,
            font_file=self.font_file,
        )

        cmd = [self.ffmpeg_bin] + overlay_args
        self._add_encode_args(cmd)
        cmd += ["-movflags", "+faststart", str(output_path), "-loglevel", "error"]

        try:
            subprocess.run(cmd, check=True)
            return output_path
        except Exception:
            return None

    def _ensure_segment(self, bvid: str, duration: float) -> Optional[Path]:
        """确保视频片段存在"""
        video = self.api_client.download_video(bvid)
        if not video:
            return None

        cached = video.parent / f"{bvid}_{int(duration)}s.mp4"
        if cached.exists():
            return cached

        audio = self.api_client.extract_audio(bvid, video)
        if not audio:
            return None

        try:
            start, _ = find_climax_segment(str(audio), clip_duration=duration)
        except:
            start = 0.0

        fade_out = max(duration - 1.0, 0.0)
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video),
            "-vf",
            f"fade=t=in:st=0:d=1,fade=t=out:st={fade_out:.3f}:d=1",
            "-af",
            f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out:.3f}:d=1",
        ]
        self._add_encode_args(cmd)
        cmd += [
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(cached),
            "-loglevel",
            "error",
        ]

        try:
            subprocess.run(cmd, check=True)
            return cached
        except:
            return None

    def _add_encode_args(self, cmd: List[str]):
        cmd += [
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
