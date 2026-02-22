# common/formatters.py
"""文本格式化与清洗工具"""

import re
from pathlib import Path
from functools import lru_cache
import pandas as pd
import yaml

_EXCEL_ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HTML_TAGS = re.compile(r"<.*?>")
_INVISIBLE_CHARS = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F"
    r"\u200B-\u200F\u2028-\u202F\u205F-\u206F\uFEFF\uFFFE-\uFFFF]"
)


@lru_cache(maxsize=1)
def _load_tid_map() -> dict:
    """加载 tid 映射表"""
    config_path = Path(__file__).parent.parent / "config" / "tid.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return {int(k): str(v) for k, v in data.get("tname", {}).items()}
    except Exception:
        return {}


def format_tid(tid) -> str:
    """将 tid 转换为 tname"""
    if pd.isna(tid):
        return ""
    tid_map = _load_tid_map()
    return tid_map.get(int(tid), str(int(tid)))


# 列格式化器映射
COLUMN_FORMATTERS = {
    "tid": format_tid,
}


def clean_excel_chars(text: str) -> str:
    """移除Excel不支持的控制字符"""
    if not isinstance(text, str):
        return text
    return _EXCEL_ILLEGAL_CHARS.sub("", text)


def clean_text(text: str, html: bool = True, invisible: bool = True) -> str:
    """综合清洗文本"""
    if not isinstance(text, str):
        return text
    if html:
        text = _HTML_TAGS.sub("", text)
    if invisible:
        text = _INVISIBLE_CHARS.sub("", text)
    return text


def format_duration(seconds: int) -> str:
    """将秒数格式化为 "M分S秒" 格式"""
    seconds = max(0, seconds - 1)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}分{secs}秒" if minutes > 0 else f"{secs}秒"


def format_aid(aid) -> str:
    """格式化aid（防止科学计数法，保持整数字符串）"""
    import pandas as pd

    if pd.isna(aid) or str(aid).strip() == "":
        return ""
    try:
        return f"{float(aid):.0f}"
    except (ValueError, TypeError):
        return str(aid)
