# video/overlay.py
"""视频叠加层生成"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from video.utils import (
    ffmpeg_escape,
    ffmpeg_escape_path,
    write_text_to_file,
    format_number,
    split_text_by_pixel_width,
)
from video.icons import get_icon_renderer


# 布局常量
TITLE_START_Y = 60
TITLE_FONT_SIZE = 50
TITLE_LINE_HEIGHT = 80
MAX_TEXT_WIDTH = 960
ICON_SIZE = 30


def build_overlay_cmd(
    *,
    segment_path: Path,
    row: pd.Series,
    clip_index: int,
    issue_date: str,
    output_dir: Path,
    font_file: str,
) -> Tuple[List[str], Path]:
    """
    构建视频叠加层 FFmpeg 命令

    Returns:
        (cmd_args, output_path)
    """
    # 解析数据
    bvid = str(row.get("bvid", "")).strip()
    title = str(row.get("title", "")).strip()
    author = str(row.get("author", "")).strip()
    pubdate = str(row.get("pubdate", "")).strip()
    point = row.get("point", "")
    view = row.get("view", "")
    favorite = row.get("favorite", "")
    coin = row.get("coin", "")
    like = row.get("like", "")
    danmaku = row.get("danmaku", "")
    reply = row.get("reply", "")
    share = row.get("share", "")
    rank = row.get("rank", None)
    count = row.get("count", 0)
    is_new = bool(row.get("is_new", False))
    rank_before = row.get("rank_before", None)

    temp_dir = output_dir / "temp_texts" / f"{issue_date}_{bvid}"
    fontfile = ffmpeg_escape_path(font_file)

    # 获取图标
    icon_renderer = get_icon_renderer()
    icon_paths = icon_renderer.get_all(size=ICON_SIZE)

    # 处理标题
    title_lines = split_text_by_pixel_width(
        title, font_file, TITLE_FONT_SIZE, MAX_TEXT_WIDTH, max_lines=3
    ) or [""]
    shift_y = (len(title_lines) - 1) * TITLE_LINE_HEIGHT

    # 构建命令
    cmd = ["-y", "-i", str(segment_path)]

    # 添加图标输入
    stats_order = ["播放", "收藏", "硬币", "点赞", "弹幕", "评论", "分享"]
    icon_indices = {}
    for idx, label in enumerate(stats_order):
        cmd += ["-loop", "1", "-i", str(icon_paths[label])]
        icon_indices[label] = idx + 1

    # 构建滤镜
    base_filters = _build_base_filters()
    icon_filters = _build_icon_filters(stats_order, icon_indices)

    text_filters = _build_text_filters(
        fontfile=fontfile,
        temp_dir=temp_dir,
        title_lines=title_lines,
        shift_y=shift_y,
        rank=rank,
        is_new=is_new,
        rank_before=rank_before,
        count=count,
        bvid=bvid,
        pubdate=pubdate,
        author=author,
        point=point,
    )

    stat_filters = _build_stat_filters(
        fontfile=fontfile,
        shift_y=shift_y,
        stats={
            "播放": view,
            "收藏": favorite,
            "硬币": coin,
            "点赞": like,
            "弹幕": danmaku,
            "评论": reply,
            "分享": share,
        },
    )

    overlay_filters = _build_icon_overlay_filters(stats_order, shift_y)
    watermark_filters = _build_watermark_filters(fontfile)

    # 组装滤镜链
    all_filters = base_filters + icon_filters

    # 文字叠加
    current = "vbase"
    all_draw_ops = text_filters + stat_filters + watermark_filters
    for idx, expr in enumerate(all_draw_ops):
        nxt = f"vtxt{idx}"
        all_filters.append(f"[{current}]{expr}[{nxt}]")
        current = nxt

    # 图标叠加
    for idx, (label, shadow_lbl, main_lbl, x, y) in enumerate(overlay_filters):
        step1 = f"vicon{idx}_sh"
        step2 = f"vicon{idx}"
        all_filters.append(f"[{current}][{shadow_lbl}]overlay={x}+2:{y}+2[{step1}]")
        all_filters.append(f"[{step1}][{main_lbl}]overlay={x}:{y}[{step2}]")
        current = step2

    # 最终输出标签
    all_filters.append(f"[{current}]copy[vfinal]")

    filter_complex = ";".join(all_filters)

    output_path = output_dir / f"tmp_{issue_date}_{clip_index:02d}_{bvid}.mp4"

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[vfinal]",
        "-map",
        "[aout]",
        "-shortest",
    ]

    return cmd, output_path


def _build_base_filters() -> List[str]:
    """基础视频处理滤镜"""
    return [
        "[0:v]settb=AVTB,setpts=PTS-STARTPTS,setsar=1,fps=60[v0]",
        "[v0]scale=trunc(1920*a/2)*2:1920,setsar=1,"
        "crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,boxblur=20:8[bg]",
        "[v0]scale=1080:-1,setsar=1[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[vbase]",
        "[0:a]asetpts=PTS-STARTPTS[aout]",
    ]


def _build_icon_filters(
    stats_order: List[str], icon_indices: Dict[str, int]
) -> List[str]:
    """图标处理滤镜"""
    filters = []
    gray_val = 220

    for idx, label in enumerate(stats_order):
        input_idx = icon_indices[label]
        raw = f"icon{idx}_raw"
        shadow = f"icon{idx}_shadow"
        main = f"icon{idx}_main"

        filters.append(
            f"[{input_idx}:v]"
            f"scale={ICON_SIZE}:{ICON_SIZE}:flags=lanczos,"
            f"format=rgba,"
            f"split[{raw}_1][{raw}_2];"
            f"[{raw}_1]geq=r=0:g=0:b=0:a='alpha(X,Y)*0.6',gblur=sigma=1[{shadow}];"
            f"[{raw}_2]geq=r={gray_val}:g={gray_val}:b={gray_val}:a='alpha(X,Y)'[{main}]"
        )

    return filters


def _build_text_filters(
    *,
    fontfile: str,
    temp_dir: Path,
    title_lines: List[str],
    shift_y: int,
    rank,
    is_new: bool,
    rank_before,
    count,
    bvid: str,
    pubdate: str,
    author: str,
    point,
) -> List[str]:
    """文字叠加滤镜"""
    filters = []

    # 标题
    curr_y = TITLE_START_Y
    for i, line in enumerate(title_lines):
        txt_path = write_text_to_file(line, temp_dir, f"title_{i}.txt")
        filters.append(
            f"drawtext=fontfile='{fontfile}':textfile='{txt_path}':"
            f"x=60:y={curr_y}:fontsize={TITLE_FONT_SIZE}:fontcolor=white:"
            f"box=1:boxcolor=black@0.45:boxborderw=14:"
            f"shadowx=2:shadowy=2:shadowcolor=black@0.55"
        )
        curr_y += TITLE_LINE_HEIGHT

    # 排名
    rank_y = 140 + shift_y
    rank_before_str = str(rank_before).strip() if rank_before is not None else "-"

    if is_new and rank and rank > 10:
        rank_text = ffmpeg_escape("NEW!!")
        rank_color = "#FF3333"
        show_change = False
    else:
        rank_text = ffmpeg_escape(f"# {rank}")
        rank_color = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}.get(rank, "#00E5FF")
        show_change = True

    filters.append(
        f"drawtext=fontfile='{fontfile}':text='{rank_text}':"
        f"x=60:y={rank_y}:fontsize=120:fontcolor={rank_color}:"
        f"shadowx=3:shadowy=3:shadowcolor=black@0.9"
    )

    # 排名变化
    if show_change:
        arrow_y = 150 + shift_y
        rank_val = int(rank) if rank and str(rank).isdigit() else 0
        arrow_x = 310 if rank_val >= 10 else 240

        if is_new or rank_before_str == "-":
            arrow, arrow_color, prev_text = "▲", "#FF3333", "NEW"
        else:
            rb = int(rank_before_str)
            prev_text = str(rb)
            if rb > rank:
                arrow, arrow_color = "▲", "#FF4444"
            elif rb < rank:
                arrow, arrow_color = "▼", "#4488FF"
            else:
                arrow, arrow_color = "■", "#888888"

        filters.append(
            f"drawtext=fontfile='{fontfile}':text='{ffmpeg_escape(arrow)}':"
            f"x={arrow_x}:y={arrow_y}:fontsize=90:fontcolor={arrow_color}:"
            f"shadowx=2:shadowy=2:shadowcolor=black@0.8"
        )

        prev_x_off = 0 if prev_text == "NEW" else (32 if int(prev_text) < 10 else 22)
        prev_y_off = 32 if arrow == "▲" else (12 if arrow == "▼" else 22)
        filters.append(
            f"drawtext=fontfile='{fontfile}':text='{ffmpeg_escape(prev_text)}':"
            f"x={arrow_x + prev_x_off}:y={arrow_y + prev_y_off}:"
            f"fontsize=40:fontcolor=white:"
            f"shadowx=2:shadowy=2:shadowcolor=black@0.9"
        )

    # 基本信息
    info_y = 280 + shift_y
    filters.append(
        f"drawtext=fontfile='{fontfile}':text='{ffmpeg_escape(bvid)}':"
        f"x=60:y={info_y}:fontsize=36:fontcolor=white:"
        f"shadowx=2:shadowy=2:shadowcolor=black@0.9"
    )
    filters.append(
        f"drawtext=fontfile='{fontfile}':text='{ffmpeg_escape(pubdate)}':"
        f"x=60:y={info_y + 40}:fontsize=36:fontcolor=white:"
        f"shadowx=2:shadowy=2:shadowcolor=black@0.9"
    )

    author_path = write_text_to_file(f"作者：{author}", temp_dir, "author.txt")
    filters.append(
        f"drawtext=fontfile='{fontfile}':textfile='{author_path}':"
        f"x=60:y={info_y + 80}:fontsize=36:fontcolor=white:"
        f"shadowx=2:shadowy=2:shadowcolor=black@0.9"
    )
    filters.append(
        f"drawtext=fontfile='{fontfile}':text='上榜次数：{ffmpeg_escape(count)}':"
        f"x=60:y={info_y + 120}:fontsize=36:fontcolor=white:"
        f"shadowx=2:shadowy=2:shadowcolor=black@0.9"
    )

    # 分数
    point_y = 140 + shift_y
    point_text = ffmpeg_escape(format_number(point))
    filters.append(
        f"drawtext=fontfile='{fontfile}':text='{point_text}':"
        f"x=w-tw-120:y={point_y}:fontsize=100:fontcolor=#FFD700:"
        f"shadowx=2:shadowy=2:shadowcolor=black@0.6"
    )
    filters.append(
        f"drawtext=fontfile='{fontfile}':text='pts':"
        f"x=w-tw-60:y={point_y}+(100-30)/2:fontsize=30:fontcolor=white"
    )

    return filters


def _build_stat_filters(
    *,
    fontfile: str,
    shift_y: int,
    stats: Dict[str, any],
) -> List[str]:
    """统计数据文字滤镜"""
    filters = []
    point_y = 140 + shift_y
    base_y = point_y + 100
    line_h = 38

    # 布局定义
    layout = [
        ("播放", "W-315", "w-tw-145", base_y),  # 居中
        ("收藏", "W-410", "w-tw-240", base_y + line_h),
        ("硬币", "W-410", "w-tw-240", base_y + 2 * line_h),
        ("点赞", "W-410", "w-tw-240", base_y + 3 * line_h),
        ("弹幕", "W-220", "w-tw-50", base_y + line_h),
        ("评论", "W-220", "w-tw-50", base_y + 2 * line_h),
        ("分享", "W-220", "w-tw-50", base_y + 3 * line_h),
    ]

    for label, _, value_x, y in layout:
        value = ffmpeg_escape(format_number(stats.get(label, "")))
        filters.append(
            f"drawtext=fontfile='{fontfile}':text='+{value}':"
            f"x={value_x}:y={y + 2}:fontsize=30:fontcolor=#FFD700:"
            f"shadowx=1:shadowy=1:shadowcolor=black@0.5"
        )

    return filters


def _build_icon_overlay_filters(
    stats_order: List[str],
    shift_y: int,
) -> List[Tuple[str, str, str, str, int]]:
    """图标叠加位置信息"""
    point_y = 140 + shift_y
    base_y = point_y + 100
    line_h = 38

    layout = [
        ("播放", "W-315", base_y),
        ("收藏", "W-410", base_y + line_h),
        ("硬币", "W-410", base_y + 2 * line_h),
        ("点赞", "W-410", base_y + 3 * line_h),
        ("弹幕", "W-220", base_y + line_h),
        ("评论", "W-220", base_y + 2 * line_h),
        ("分享", "W-220", base_y + 3 * line_h),
    ]

    result = []
    for idx, (label, x, y) in enumerate(layout):
        shadow_lbl = f"icon{stats_order.index(label)}_shadow"
        main_lbl = f"icon{stats_order.index(label)}_main"
        result.append((label, shadow_lbl, main_lbl, x, y))

    return result


def _build_watermark_filters(fontfile: str) -> List[str]:
    """水印滤镜"""
    wm1 = ffmpeg_escape("术力口数据姬")
    wm2 = ffmpeg_escape("vocabili.top")

    return [
        f"drawtext=fontfile='{fontfile}':text='{wm1}':"
        f"x=60:y=h-600:fontsize=36:fontcolor=white@0.65:"
        f"shadowx=1:shadowy=1:shadowcolor=black@0.65",
        f"drawtext=fontfile='{fontfile}':text='{wm2}':"
        f"x=60:y=h-560:fontsize=32:fontcolor=white@0.35:"
        f"shadowx=1:shadowy=1:shadowcolor=black@0.35",
    ]
