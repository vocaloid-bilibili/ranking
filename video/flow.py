# video/daily_video_flow.py
"""日刊视频生成主流程"""

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

from common.logger import logger
from common.models import ScraperConfig
from bilibili.client import BilibiliClient
from video.config import VideoConfig, load_video_config
from video.climax import find_climax_segment
from video.issue import Issue
from video.cover import Cover
from video.achievement import Achievement
from video.clip import ClipFlow


class DailyVideoFlow:
    """日刊视频生成主流程"""

    def __init__(self, cfg: VideoConfig = None) -> None:
        self.cfg = cfg or load_video_config()

        # 确保目录存在
        self.cfg.paths.ensure_dirs()

        # API 客户端
        self.api_client = BilibiliClient(
            config=ScraperConfig(),
            videos_root=self.cfg.paths.videos_cache,
            ffmpeg_bin=self.cfg.ffmpeg.bin,
        )

        # 期刊管理器
        self.issue_mgr = Issue(
            ranking_main_dir=self.cfg.paths.daily_ranking_main,
            ranking_new_dir=self.cfg.paths.daily_ranking_new,
            first_issue_date=self.cfg.video.first_issue_date,
        )

        # 封面生成器
        self.cover_mgr = Cover(
            videos_cache=self.cfg.paths.videos_cache,
            font_regular=self.cfg.fonts.regular,
            font_bold=self.cfg.fonts.bold,
            card_width=self.cfg.ui.card_width,
            card_height=self.cfg.ui.card_height,
            card_radius=self.cfg.ui.card_radius,
            weekday_colors=self.cfg.weekday_colors,
            ffmpeg_bin=self.cfg.ffmpeg.bin,
        )

        # 成就片段生成器
        self.achieve_clipper = Achievement(
            milestone_dir=self.cfg.paths.milestone,
            config_dir=self.cfg.project_root / "config",
            image_factory=self.cover_mgr,
        )

        # 视频片段生成器
        self.clip_flow = ClipFlow(
            api_client=self.api_client,
            daily_video_dir=self.cfg.paths.daily_video_output,
            icon_dir=self.cfg.paths.icon_dir,
            font_regular=self.cfg.fonts.regular,
            font_bold=self.cfg.fonts.bold,
            ffmpeg_bin=self.cfg.ffmpeg.bin,
        )

        self.daily_video_dir = self.cfg.paths.daily_video_output
        self.clip_duration = self.cfg.video.clip_duration
        self.ffmpeg_bin = self.cfg.ffmpeg.bin

        c = self.cfg.ui.scroll_bg_color
        self.bg_color = tuple(c) if len(c) == 4 else (c[0], c[1], c[2], 255)
        self.ui = self.cfg.ui

    async def close(self):
        """关闭资源"""
        await self.api_client.close_session()

    def run(self) -> None:
        """执行视频生成"""
        self.daily_video_dir.mkdir(parents=True, exist_ok=True)

        combined_rows, issue_date, issue_idx, excel_date = (
            self.issue_mgr.prepare_video_data(self.cfg.video.top_n)
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_clips = executor.submit(
                self._generate_clips, combined_rows, issue_date
            )
            future_achieve = executor.submit(
                self._generate_achievement_video, excel_date, issue_date, issue_idx
            )
            self._generate_covers(combined_rows, issue_date, issue_idx)

            cover_vertical_path = (
                self.daily_video_dir / f"{issue_idx}_{issue_date}_cover_3-4.jpg"
            )
            cover_intro_path = (
                self.daily_video_dir / f"tmp_cover_intro_{issue_date}.mp4"
            )
            if cover_vertical_path.exists():
                self._create_cover_intro_clip(cover_vertical_path, cover_intro_path)

            index_to_path = future_clips.result()
            achieve_vid = future_achieve.result()

        if not index_to_path:
            logger.error("没有生成视频片段")
            return

        all_clips = [index_to_path[i] for i in sorted(index_to_path.keys())]

        if cover_intro_path.exists():
            all_clips.insert(0, cover_intro_path)

        if achieve_vid:
            all_clips.append(achieve_vid)

        final_path = self.daily_video_dir / f"{issue_idx}_{issue_date}.mp4"
        self._concat_clips(all_clips, final_path)
        logger.info(f"完成: {final_path}")

        self._cleanup_temp_files(all_clips)

    def _create_cover_intro_clip(self, image_path: Path, output_path: Path) -> None:
        """生成封面片头视频"""
        filter_complex = (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=40:5[bg];"
            "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
            "setsar=1,fps=60"
        )

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v",
            "libx264",
            "-t",
            "0.5",
            "-filter_complex",
            filter_complex,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
            "-loglevel",
            "error",
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            logger.error(f"生成封面片头视频出错: {e}")

    def _cleanup_temp_files(self, clip_paths: List[Path]) -> None:
        """清理临时文件"""
        for p in clip_paths:
            if p.exists():
                p.unlink()
        temp_text_root = self.daily_video_dir / "temp_texts"
        if temp_text_root.exists():
            shutil.rmtree(temp_text_root, ignore_errors=True)

    def _generate_clips(self, rows, issue_date) -> Dict[int, Path]:
        """生成视频片段"""
        tasks = [(i + 1, r.to_dict()) for i, r in enumerate(rows)]
        res = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(self._worker, t, issue_date): t for t in tasks}
            for f in as_completed(futures):
                idx, path = f.result()
                if path:
                    res[idx] = path
        return res

    def _worker(self, task, issue_date):
        """处理单个视频片段"""
        idx, r_dict = task
        row = pd.Series(r_dict)
        current_duration = self.clip_duration
        rank_val = row.get("rank", 999)

        if rank_val <= 3:
            current_duration = 20.0

        path = self.clip_flow.generate_clip(row, idx, current_duration, issue_date)
        return idx, path

    def _generate_covers(self, rows, date_str, idx):
        """生成封面"""
        urls_16_9 = self.cover_mgr.select_cover_urls_grid(rows)
        self.cover_mgr.generate_grid_cover(
            urls_16_9,
            self.daily_video_dir / f"{idx}_{date_str}_cover.jpg",
            issue_date=date_str,
            issue_index=idx,
        )

        urls_3_4 = self.cover_mgr.select_cover_urls_3_4(rows)
        self.cover_mgr.generate_vertical_cover(
            urls_3_4,
            self.daily_video_dir / f"{idx}_{date_str}_cover_3-4.jpg",
            issue_date=date_str,
            issue_index=idx,
        )

    def _generate_achievement_video(self, ex_date, is_date, idx) -> Optional[Path]:
        """生成成就视频"""
        rows = self.achieve_clipper.load_rows(ex_date, is_date)

        out_path = self.daily_video_dir / f"tmp_achievement_{is_date}.mp4"
        strip_img, strip_h = self.achieve_clipper.build_strip(
            rows, width=1080, gap=self.ui.card_gap
        )

        screen_w, screen_h = 1080, 1920
        initial_list_top_y = 800

        target_bottom_y = screen_h / 2
        target_top_y = target_bottom_y - strip_h

        total_dist = initial_list_top_y - target_top_y
        if total_dist < 0:
            total_dist = 0

        scroll_duration = total_dist / self.ui.scroll_speed_pps
        total_duration = self.ui.scroll_hold_time + scroll_duration

        fps = self.cfg.video.fps
        total_frames = int(total_duration * fps)

        ed_info = self.achieve_clipper.get_ed_info(idx)
        bgm_bvid = ed_info.get("bvid")

        audio_input_args = [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
        ]
        audio_map = "1:a"

        if bgm_bvid:
            v_path = self.api_client.download_video(bgm_bvid)
            if v_path:
                a_path = self.api_client.ensure_audio(bgm_bvid, v_path)
                if a_path:
                    start, _ = find_climax_segment(
                        str(a_path), clip_duration=total_duration
                    )
                    audio_input_args = ["-ss", f"{start:.3f}", "-i", str(a_path)]
                    audio_map = "1:a"

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{screen_w}x{screen_h}",
            "-pix_fmt",
            "rgba",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        cmd += audio_input_args

        fade_out_start = max(0, total_duration - 1.0)
        af_str = f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start:.3f}:d=1"

        cmd += ["-map", "0:v", "-map", audio_map, "-af", af_str]
        self._add_x264_encode_args(cmd)
        cmd += ["-t", f"{total_duration:.3f}", str(out_path), "-loglevel", "error"]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        bg_base = Image.new("RGBA", (screen_w, screen_h), self.bg_color)

        try:
            logger.info(f"开始生成成就视频 (总时长: {total_duration:.1f}s)...")
            for frame_index in range(total_frames):
                t = frame_index / fps
                if t < self.ui.scroll_hold_time:
                    curr_strip_y = float(initial_list_top_y)
                    header_opacity = 1.0
                else:
                    scroll_t = t - self.ui.scroll_hold_time
                    curr_strip_y = initial_list_top_y - (
                        scroll_t * self.ui.scroll_speed_pps
                    )
                    fade_duration = 1.5
                    if scroll_t < fade_duration:
                        header_opacity = 1.0 - (scroll_t / fade_duration)
                    else:
                        header_opacity = 0.0

                frame = bg_base.copy()
                paste_y = int(curr_strip_y)

                if paste_y < screen_h and (paste_y + strip_h) > 0:
                    frame.paste(strip_img, (0, paste_y), strip_img)

                if header_opacity > 0:
                    header_img = self.cover_mgr.create_header(
                        screen_w,
                        screen_h,
                        header_opacity,
                        ed_info=ed_info,
                        list_top_y=paste_y,
                    )
                    frame.alpha_composite(header_img)

                process.stdin.write(frame.tobytes())

                if frame_index % 300 == 0:
                    logger.info(f"成就视频进度: {t:.1f}/{total_duration:.1f}s")

            process.stdin.close()
            process.wait()
            return out_path
        except Exception as exc:
            logger.error(f"成就视频出错: {exc}")
            try:
                process.stdin.close()
            except:
                pass
            return None

    def _concat_clips(self, clip_paths: List[Path], output_path: Path) -> None:
        """拼接视频片段"""
        cmd = [self.ffmpeg_bin, "-y"]
        for p in clip_paths:
            cmd += ["-i", str(p)]

        n = len(clip_paths)
        va = "".join(f"[{i}:v][{i}:a]" for i in range(n))
        filter_complex = f"{va}concat=n={n}:v=1:a=1[v][a]"

        cmd += ["-filter_complex", filter_complex, "-map", "[v]", "-map", "[a]"]
        self._add_x264_encode_args(cmd)
        cmd += ["-movflags", "+faststart", str(output_path), "-loglevel", "error"]
        subprocess.run(cmd, check=True)

    def _add_x264_encode_args(self, cmd: List[str]) -> None:
        """添加 x264 编码参数"""
        cmd.extend(
            [
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
        )
