# video/cover.py
"""封面生成器"""

from pathlib import Path
from typing import List, Dict, Any
import random
import subprocess
from datetime import datetime

import pandas as pd
from PIL import ImageFont

from common.logger import logger
from video.utils import ffmpeg_escape_path


class CoverGenerator:
    """封面生成器"""

    def __init__(
        self,
        *,
        font_bold: str,
        weekday_colors: Dict[int, str],
        ffmpeg_bin: str = "ffmpeg",
    ):
        self.font_bold_file = font_bold
        self.weekday_colors = weekday_colors
        self.ffmpeg_bin = ffmpeg_bin

    def _get_theme_color(self, issue_date: str) -> str:
        """根据期刊日期获取主题色"""
        try:
            date_obj = datetime.strptime(issue_date, "%Y%m%d")
            weekday = date_obj.weekday()
        except:
            weekday = 6
        return self.weekday_colors.get(weekday, "#55CCCC")

    def select_urls(self, rows: List[pd.Series], layout: str = "grid") -> List[str]:
        """
        选取封面图片 URL

        Args:
            rows: 数据行列表
            layout: "grid" (16:9) 或 "vertical" (3:4)
        """
        if not rows:
            return []

        data = self._parse_rows(rows)
        candidates = self._select_candidates(data)

        if layout == "grid":
            return self._build_grid_urls(candidates)
        else:
            return self._build_vertical_urls(candidates)

    def _parse_rows(self, rows: List[pd.Series]) -> List[Dict]:
        """解析行数据"""
        result = []
        for r in rows:
            try:
                rank = int(r.get("rank", 999))
            except:
                rank = 999
            try:
                count = int(r.get("count", 0))
            except:
                count = 0

            result.append(
                {
                    "bvid": str(r.get("bvid", "")),
                    "rank": rank,
                    "is_new": bool(r.get("is_new", False)),
                    "count": count,
                    "url": str(r.get("image_url", "")).strip(),
                }
            )
        result.sort(key=lambda x: x["rank"])
        return result

    def _select_candidates(self, data: List[Dict]) -> Dict[str, Any]:
        """选取候选封面"""
        used_bvids = set()
        result = {}

        # 总榜第一
        if data:
            rank1 = data[0]
            result["total_rank_1"] = rank1
            used_bvids.add(rank1["bvid"])

        # 排名最高的新曲
        for item in data:
            if item["is_new"] and item["bvid"] not in used_bvids:
                result["new_rank_1"] = item
                used_bvids.add(item["bvid"])
                break

        if "new_rank_1" not in result:
            for item in data:
                if item["bvid"] not in used_bvids:
                    result["new_rank_1"] = item
                    used_bvids.add(item["bvid"])
                    break

        def pick_one(pool):
            candidates = [x for x in pool if x["bvid"] not in used_bvids]
            if candidates:
                selected = random.choice(candidates)
                used_bvids.add(selected["bvid"])
                return selected
            return None

        # 其他新曲
        result["other_new"] = pick_one([d for d in data if d["is_new"]])

        # 上榜次数最多
        remaining = sorted(
            [d for d in data if d["bvid"] not in used_bvids],
            key=lambda x: x["count"],
            reverse=True,
        )
        result["high_count"] = pick_one(remaining[:3])

        # Top 2-5, Top 6-10
        result["top_2_5"] = pick_one([d for d in data if 2 <= d["rank"] <= 5])
        result["top_6_10"] = pick_one([d for d in data if 6 <= d["rank"] <= 10])

        return result

    def _build_grid_urls(self, candidates: Dict) -> List[str]:
        """构建 16:9 封面 URL 列表"""
        urls = []
        urls.append(candidates.get("total_rank_1", {}).get("url", ""))
        urls.append(candidates.get("new_rank_1", {}).get("url", ""))

        small = [
            candidates.get("other_new"),
            candidates.get("high_count"),
            candidates.get("top_2_5"),
            candidates.get("top_6_10"),
        ]
        valid = [x for x in small if x]
        random.shuffle(valid)
        urls.extend(x["url"] for x in valid)
        return urls

    def _build_vertical_urls(self, candidates: Dict) -> List[str]:
        """构建 3:4 封面 URL 列表"""
        urls = []
        urls.append(candidates.get("total_rank_1", {}).get("url", ""))
        urls.append(candidates.get("new_rank_1", {}).get("url", ""))
        urls.append(candidates.get("high_count", {}).get("url", ""))

        bottom = [
            candidates.get("other_new"),
            candidates.get("top_2_5"),
            candidates.get("top_6_10"),
        ]
        valid = [x for x in bottom if x]
        random.shuffle(valid)
        urls.extend(x["url"] for x in valid)
        return urls

    def generate_grid(
        self,
        urls: List[str],
        output_path: Path,
        issue_date: str = "",
        issue_index: int = 0,
    ):
        """生成 16:9 封面"""
        if not urls:
            logger.warning("封面生成失败：没有可用的 URL")
            return

        display_urls = urls[:6]
        while len(display_urls) < 6:
            display_urls.append(display_urls[-1] if display_urls else "")

        theme_color = self._get_theme_color(issue_date)
        font_path = ffmpeg_escape_path(self.font_bold_file)

        # 解析日期
        try:
            date_obj = datetime.strptime(issue_date, "%Y%m%d")
            month, day = date_obj.strftime("%m"), date_obj.strftime("%d")
        except:
            month, day = "01", "01"

        # 构建 FFmpeg 命令
        cmd = [self.ffmpeg_bin, "-y"]
        for url in display_urls:
            cmd += ["-i", url]

        filters = self._build_grid_filters(
            theme_color, font_path, month, day, issue_index
        )

        cmd += [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
            "-loglevel",
            "error",
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info(f"封面已保存: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"封面生成失败: {e}")

    def _build_grid_filters(
        self,
        theme_color: str,
        font_path: str,
        month: str,
        day: str,
        issue_index: int,
    ) -> List[str]:
        """构建 16:9 封面滤镜"""
        filters = []

        # 图片处理
        filters.append(
            "[0:v]scale=1000:562:force_original_aspect_ratio=increase,crop=1000:562,setsar=1,"
            "pad=1024:586:12:12:white[v0_raw]"
        )
        filters.append(
            "[1:v]scale=800:450:force_original_aspect_ratio=increase,crop=800:450,setsar=1,"
            "pad=824:474:12:12:white[v1_raw]"
        )
        for i in range(2, 6):
            filters.append(
                f"[{i}:v]scale=420:236:force_original_aspect_ratio=increase,crop=420:236,setsar=1,"
                f"pad=436:252:8:8:white[v{i}_raw]"
            )

        # 位置定义
        pos = {
            "v2": (68, 290),
            "v3": (524, 290),
            "v4": (980, 290),
            "v5": (1436, 290),
            "v0": (80, 460),
            "v1": (1016, 570),
        }

        # 背景
        filters.append(
            "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
            "gblur=sigma=30,eq=saturation=1.4:brightness=-0.05,"
            "drawbox=c=white@0.1:t=fill[bg_blur]"
        )

        # 阴影
        shadow_cmds = [
            f"drawbox=x={pos['v0'][0]+12}:y={pos['v0'][1]+12}:w=1024:h=586:c=black@0.2:t=fill",
            f"drawbox=x={pos['v1'][0]+12}:y={pos['v1'][1]+12}:w=824:h=474:c=black@0.2:t=fill",
        ]
        for k in ["v2", "v3", "v4", "v5"]:
            shadow_cmds.append(
                f"drawbox=x={pos[k][0]+10}:y={pos[k][1]+10}:w=436:h=252:c=black@0.25:t=fill"
            )
        filters.append(f"[bg_blur]{','.join(shadow_cmds)},gblur=sigma=25[bg_shadow]")

        # 叠加小图
        current = "bg_shadow"
        for i, k in enumerate(["v2", "v3", "v4", "v5"]):
            nxt = f"l_s_{i}"
            filters.append(
                f"[{current}][{k}_raw]overlay=x={pos[k][0]}:y={pos[k][1]}[{nxt}]"
            )
            current = nxt

        # 叠加大图
        filters.append(
            f"[{current}][v0_raw]overlay=x={pos['v0'][0]}:y={pos['v0'][1]}[l_main]"
        )
        filters.append(
            f"[l_main][v1_raw]overlay=x={pos['v1'][0]}:y={pos['v1'][1]}[l_final_img]"
        )

        # 顶部横幅
        filters.append(
            f"[l_final_img]drawbox=x=0:y=0:w=1920:h=260:color={theme_color}:t=fill[banner]"
        )

        # 分割线
        filters.append(
            "[banner]drawbox=x=770:y=50:w=6:h=180:color=white@0.7:t=fill[deco]"
        )

        # 日期
        filters.append(
            f"[deco]drawtext=fontfile='{font_path}':text='{month}/':"
            f"fontsize=140:fontcolor=white:x=240:y=85:"
            f"shadowx=4:shadowy=4:shadowcolor=black@0.3[t1]"
        )
        filters.append(
            f"[t1]drawtext=fontfile='{font_path}':text='{day}':"
            f"fontsize=240:fontcolor=white:x=460:y=60:"
            f"shadowx=6:shadowy=6:shadowcolor=black@0.3[t2]"
        )

        # 标题
        filters.append(
            f"[t2]drawtext=fontfile='{font_path}':text='日刊虚拟歌手':"
            f"fontsize=80:fontcolor=white@0.95:x=800:y=50:"
            f"shadowx=3:shadowy=3:shadowcolor=black@0.3[t3]"
        )
        filters.append(
            f"[t3]drawtext=fontfile='{font_path}':text='外语排行榜':"
            f"fontsize=100:fontcolor=white:x=800:y=135:"
            f"shadowx=4:shadowy=4:shadowcolor=black@0.3[t4]"
        )

        # 期数
        issue_text = f"VOL.{issue_index}" if issue_index > 0 else ""
        filters.append(
            f"[t4]drawtext=fontfile='{font_path}':text='{issue_text}':"
            f"fontsize=110:fontcolor=white@0.25:"
            f"x=1920-tw-140:y=(260-th)/2[vout]"
        )

        return filters

    def generate_vertical(
        self,
        urls: List[str],
        output_path: Path,
        issue_date: str = "",
        issue_index: int = 0,
    ):
        """生成 3:4 竖屏封面"""
        if not urls:
            logger.warning("3:4 封面生成失败：没有可用的 URL")
            return

        valid_urls = urls[:6]
        count = len(valid_urls)

        while len(valid_urls) < 6:
            valid_urls.append(valid_urls[0] if valid_urls else "")

        border_color = self._get_theme_color(issue_date)
        font_path = ffmpeg_escape_path(self.font_bold_file)

        # 解析日期
        try:
            date_obj = datetime.strptime(issue_date, "%Y%m%d")
            month, day = date_obj.strftime("%m"), date_obj.strftime("%d")
        except:
            month = issue_date[-4:-2] if len(issue_date) >= 6 else "01"
            day = issue_date[-2:] if len(issue_date) >= 2 else "01"

        cmd = [self.ffmpeg_bin, "-y"]
        for u in valid_urls:
            cmd += ["-i", u]

        filters = self._build_vertical_filters(
            border_color, font_path, month, day, count
        )

        cmd += [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
            "-loglevel",
            "error",
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info(f"3:4 封面已保存: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"3:4 封面生成失败: {e}")

    def _build_vertical_filters(
        self,
        border_color: str,
        font_path: str,
        month: str,
        day: str,
        count: int,
    ) -> List[str]:
        """构建 3:4 封面滤镜"""
        W, H = 1920, 2560
        filters = []

        # 背景
        filters.append(
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},setsar=1,"
            f"boxblur=40:5,eq=brightness=-0.1:saturation=1.3[bg]"
        )

        # 图片尺寸
        hero_w, hero_h = 1600, int(1600 * 9 / 16)
        hero_pad = 20
        small_w, small_h = 1000, int(1000 * 9 / 16)
        small_pad = 15

        # 处理图片
        processed = []
        for i in range(count):
            is_hero = i == 0
            tw = hero_w if is_hero else small_w
            th = hero_h if is_hero else small_h
            pad = hero_pad if is_hero else small_pad
            pw, ph = tw + 2 * pad, th + 2 * pad
            lbl = f"img{i}"
            processed.append(lbl)
            filters.append(
                f"[{i}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                f"pad={pw}:{ph}:{pad}:{pad}:white[{lbl}]"
            )

        # 小图位置
        hero_x, hero_y = "(W-w)/2", "(H-h)/2 + 250"
        top_row_y = 600
        btm_row_y = "H-h-100"

        small_pos = [
            ("-100", top_row_y),
            ("W-w+100", top_row_y),
            ("-150", btm_row_y),
            ("(W-w)/2", btm_row_y),
            ("W-w+150", btm_row_y),
        ]

        # 叠加小图
        current = "bg"
        for i in range(1, count):
            lbl = processed[i]
            pos_idx = min(i - 1, len(small_pos) - 1)
            px, py = small_pos[pos_idx]
            nxt = f"tmp_bg_{i}"
            filters.append(f"[{current}][{lbl}]overlay=x={px}:y={py}[{nxt}]")
            current = nxt

        # 主图阴影和叠加
        if processed:
            hero_lbl = processed[0]
            filters.append(
                f"[{hero_lbl}]split[h_src][h_sh_raw];"
                f"[h_sh_raw]drawbox=c=black:t=fill,format=rgba,"
                f"gblur=sigma=40,colorchannelmixer=aa=0.45[hero_shadow]"
            )
            filters.append(
                f"[{current}][hero_shadow]overlay=x={hero_x}+30:y={hero_y}+40[bg_w_shadow]"
            )
            filters.append(
                f"[bg_w_shadow][h_src]overlay=x={hero_x}:y={hero_y}[combined_img]"
            )
        else:
            filters.append(f"[{current}]copy[combined_img]")

        # 标题文字
        text1, text2 = "日刊虚拟歌手", "外语排行榜"
        fill_color = "white@0.95"
        border_w = 22
        font_size_1, font_size_2 = 260, 220
        title_y = 220

        # 计算右对齐锚点
        try:
            font_obj = ImageFont.truetype(self.font_bold_file, font_size_1)
            w1 = font_obj.getlength(text1)
        except:
            w1 = 800
        anchor_x = (W / 2) + (w1 / 2)

        filters.append(
            f"[combined_img]drawtext=fontfile='{font_path}':text='{text1}':"
            f"fontsize={font_size_1}:fontcolor={fill_color}:"
            f"borderw={border_w}:bordercolor={border_color}:"
            f"x={anchor_x}-tw:y={title_y}:"
            f"shadowx=8:shadowy=8:shadowcolor=black@0.4[txt1]"
        )
        filters.append(
            f"[txt1]drawtext=fontfile='{font_path}':text='{text2}':"
            f"fontsize={font_size_2}:fontcolor={fill_color}:"
            f"borderw={border_w}:bordercolor={border_color}:"
            f"x={anchor_x}-tw:y={title_y + font_size_1 + 40}:"
            f"shadowx=5:shadowy=5:shadowcolor=black@0.4[txt2]"
        )

        # 日期
        month_size = 300
        day_size = month_size * 2
        date_y = 1700

        filters.append(
            f"[txt2]drawtext=fontfile='{font_path}':text='{month}':"
            f"fontsize={month_size}:fontcolor={fill_color}:"
            f"borderw=24:bordercolor={border_color}:"
            f"x=(w/2)-500:y={date_y}:"
            f"shadowx=8:shadowy=8:shadowcolor=black@0.6[txt3]"
        )
        filters.append(
            f"[txt3]drawtext=fontfile='{font_path}':text='/':"
            f"fontsize={month_size}:fontcolor={fill_color}:"
            f"borderw=24:bordercolor={border_color}:"
            f"x=(w/2)-80:y={date_y}:"
            f"shadowx=6:shadowy=6:shadowcolor=black@0.5[txt4]"
        )

        day_offset = (day_size - month_size) / 2
        filters.append(
            f"[txt4]drawtext=fontfile='{font_path}':text='{day}':"
            f"fontsize={day_size}:fontcolor={fill_color}:"
            f"borderw=32:bordercolor={border_color}:"
            f"x=(w/2)+80:y={date_y - day_offset}:"
            f"shadowx=10:shadowy=10:shadowcolor=black@0.6[vout]"
        )

        return filters
