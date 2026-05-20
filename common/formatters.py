# common/formatters.py
"""文本格式化与清洗工具"""

import re
from functools import lru_cache
from pathlib import Path

import yaml

_HTML_TAGS = re.compile(r"<.*?>")
_INVISIBLE_CHARS = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F"
    r"\u200B-\u200F\u2028-\u202F\u205F-\u206F\uFEFF\uFFFE-\uFFFF]"
)


def clean_text(text: str, html: bool = True, invisible: bool = True) -> str:
    """综合清洗文本"""
    if not isinstance(text, str):
        return text
    if html:
        text = _HTML_TAGS.sub("", text)
    if invisible:
        text = _INVISIBLE_CHARS.sub("", text)
    return text


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


def tid_to_tname(tid) -> str:
    """将分区 ID 转换为分区名称"""
    try:
        tid_int = int(float(tid))
    except (ValueError, TypeError):
        return str(tid)
    tid_map = _load_tid_map()
    return tid_map.get(tid_int, str(tid_int))
