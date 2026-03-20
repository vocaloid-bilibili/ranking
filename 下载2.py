# 下载2.py
"""下载日报所需的 5 个文件"""

from pathlib import Path
from datetime import datetime, timedelta

import yaml

from services.sftp import SFTPClient
from common.logger import logger

REMOTE_BASE = "/home/vocabili"
LOCAL_BASE = Path("data")


def build_mappings(date: str, date_prev: str) -> list[dict]:
    """构建远程→本地映射，data/ 以下结构与服务器一致"""
    files = [
        f"data/snapshot/main/{date}.xlsx",
        f"data/daily/ranking/main/{date}与{date_prev}.xlsx",
        f"data/daily/ranking/new/新曲榜{date}与{date_prev}.xlsx",
        f"data/daily/diff/new/新曲{date}与新曲{date_prev}.xlsx",
        "data/collected.xlsx",
    ]
    mappings = []
    for rel in files:
        remote = f"{REMOTE_BASE}/{rel}"
        local = Path(rel)
        local.parent.mkdir(parents=True, exist_ok=True)
        mappings.append({"remote": remote, "local": str(local)})
    return mappings


def main():
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    date = today.strftime("%Y%m%d")
    date_prev = yesterday.strftime("%Y%m%d")

    # 读 SFTP 连接信息
    config_path = Path("config/sftp.yaml")
    try:
        server = yaml.safe_load(config_path.read_text(encoding="utf-8"))["server"]
    except (FileNotFoundError, KeyError):
        logger.error(f"找不到配置文件或 server 配置: {config_path}")
        return

    mappings = build_mappings(date, date_prev)

    logger.info(f"日期: {date}, 前一天: {date_prev}")
    for m in mappings:
        logger.info(f"  {m['remote']}  ->  {m['local']}")

    with SFTPClient(
        host=server["host"],
        port=server["port"],
        username=server["username"],
        password=server["password"],
    ) as client:
        result = client.download_mapped(mappings)
        logger.info(f"下载完成: {result}")


if __name__ == "__main__":
    main()
