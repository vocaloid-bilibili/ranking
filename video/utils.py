# video/utils.py
"""视频模块共用工具函数"""

import math
from pathlib import Path
from typing import List
from PIL import ImageFont


def ffmpeg_escape(text: str) -> str:
    """FFmpeg 文本转义"""
    s = str(text)
    return (
        s.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("%", r"\%")
    )


def ffmpeg_escape_path(path: str) -> str:
    """FFmpeg 路径转义"""
    p = str(path).replace("\\", "/")
    if ":" in p:
        drive, rest = p.split(":", 1)
        p = f"{drive}\\:{rest}"
    return p


def write_text_to_file(content: str, folder: Path, filename: str) -> str:
    """写入文本文件并返回转义后的路径"""
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / filename
    file_path.write_text(content, encoding="utf-8")
    return ffmpeg_escape_path(str(file_path))


def format_number(x) -> str:
    """格式化数字为千分位"""
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "-"
        return f"{int(x):,}"
    except Exception:
        if str(x).strip() == "":
            return "-"
        return str(x)


def split_text_by_pixel_width(
    text: str,
    font_path: str,
    font_size: int,
    max_width: int,
    max_lines: int = 3,
) -> List[str]:
    """按像素宽度分割文本"""
    text = (text or "").strip()
    if not text:
        return []

    font = ImageFont.truetype(font_path, font_size)
    lines = []
    current_line = ""

    for char in text:
        test_line = current_line + char
        width = font.getlength(test_line)

        if width <= max_width:
            current_line = test_line
        else:
            if len(lines) >= max_lines - 1:
                while current_line and font.getlength(current_line + "...") > max_width:
                    current_line = current_line[:-1]
                lines.append(current_line + "...")
                return lines

            lines.append(current_line)
            current_line = char

    if current_line:
        lines.append(current_line)

    return lines


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """文字换行"""
    lines = []
    curr = ""
    if not text:
        return lines
    for char in text:
        if font.getbbox(curr + char)[2] <= max_width:
            curr += char
        else:
            lines.append(curr)
            curr = char
    if curr:
        lines.append(curr)
    return lines
