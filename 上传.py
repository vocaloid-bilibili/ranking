from pathlib import Path
import yaml

from services.sftp import SFTPClient
from common.config import get_paths
from common.logger import logger


def main():
    config_path = Path("config/sftp.yaml")
    paths = get_paths()

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error(f"找不到配置文件: {config_path}")
        return

    server = config.get("server", {})
    upload_config = config.get("upload", {})
    remote_base = upload_config.get("remote_base", "/home/vocabili")
    files = upload_config.get("files", [])

    if not files:
        logger.warning("未找到上传文件列表")
        return

    mappings = []
    for item in files:
        key = item.get("key")
        remote_path = f"{remote_base}/{item.get('remote')}"

        if key and hasattr(paths, key):
            local = getattr(paths, key)
        else:
            local = Path(item.get("local", ""))

        if not local.exists():
            logger.warning(f"文件不存在，跳过: {local}")
            continue

        mappings.append({"local": str(local), "remote": remote_path})

    with SFTPClient(
        host=server["host"],
        port=server["port"],
        username=server["username"],
        password=server["password"],
    ) as client:
        result = client.upload_mapped(mappings)
        logger.info(f"上传完成: {result}")


if __name__ == "__main__":
    main()
    input("按回车退出...")
