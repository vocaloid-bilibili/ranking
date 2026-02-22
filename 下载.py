# 下载.py
"""从服务器下载数据"""

from pathlib import Path
from datetime import datetime, timedelta
import yaml

from services.sftp import SFTPClient
from common.config import get_paths
from common.logger import logger


def get_date_params(days_back: int = 0) -> dict:
    target = datetime.now() - timedelta(days=days_back)
    return {
        "date": target.strftime("%Y%m%d"),
        "date_hyphen": target.strftime("%Y-%m-%d"),
    }


def resolve_local_path(item: dict, paths, **kwargs) -> Path:
    local_key = item.get("local_key")
    filename = item.get("filename", "")

    if local_key and hasattr(paths, local_key):
        base = getattr(paths, local_key)
        if filename:
            return base / filename.format(**kwargs)
        return base

    return Path(item.get("local", "").format(**kwargs))


def main():
    config_path = Path("config/sftp.yaml")
    paths = get_paths()

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error(f"找不到配置文件: {config_path}")
        return

    server = config.get("server", {})
    download_list = config.get("download", [])

    if not download_list:
        logger.warning("未找到 download 配置")
        return

    date_params = get_date_params(days_back=0)

    mappings = []
    for item in download_list:
        remote = item["remote"].format(**date_params)
        local = resolve_local_path(item, paths, **date_params)
        local.parent.mkdir(parents=True, exist_ok=True)
        mappings.append({"remote": remote, "local": str(local)})
        logger.info(f"映射: {remote} -> {local}")

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
