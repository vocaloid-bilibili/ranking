# common/io.py
"""文件读写工具 — JSONL"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from common.logger import logger


def save_jsonl(
    records: List[Dict[str, Any]],
    path: Union[str, Path],
    usecols: Optional[List[str]] = None,
):
    """
    保存记录到 JSONL 文件

    Args:
        records: 记录列表
        path: 保存路径
        usecols: 需要保存的字段，None = 全部
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            if usecols:
                row = {k: record.get(k, "") for k in usecols}
            else:
                row = record
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"已保存 {len(records)} 条记录到 {path}")


def load_jsonl(
    path: Union[str, Path],
    usecols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    读取 JSONL 文件

    Args:
        path: 文件路径
        usecols: 需要读取的字段，None = 全部
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"文件不存在: {path}")
        return []

    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if usecols:
                    row = {k: row[k] for k in usecols if k in row}
                records.append(row)
    except Exception as e:
        logger.error(f"读取失败: {path}, 错误: {e}")
        return []

    return records


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_file(directory: Path, pattern: str = "*.jsonl") -> Optional[Path]:
    """获取目录中最新的文件"""
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)
