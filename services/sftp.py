# services/sftp.py
"""SFTP 远程文件传输服务"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

import paramiko

from common.logger import logger


@dataclass
class TransferResult:
    """传输结果"""

    success: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)

    @property
    def all_success(self) -> bool:
        return len(self.failed) == 0

    def __str__(self) -> str:
        return f"成功: {len(self.success)}, 失败: {len(self.failed)}"


@dataclass
class CommandResult:
    """命令执行结果"""

    exit_status: int
    output: str
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.exit_status == 0


class SFTPClient:
    """SFTP 客户端"""

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        timeout: int = 10,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout

        self._ssh: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None

    def connect(self) -> bool:
        """建立连接"""
        try:
            self._ssh = paramiko.SSHClient()
            self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self._ssh.connect(
                self.host,
                self.port,
                self.username,
                self.password,
                timeout=self.timeout,
            )
            self._sftp = self._ssh.open_sftp()
            logger.info(f"已连接到 {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self._sftp:
            self._sftp.close()
            self._sftp = None
        if self._ssh:
            self._ssh.close()
            self._ssh = None
        logger.info("连接已关闭")

    def __enter__(self) -> "SFTPClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def upload_file(self, local_path: Union[str, Path], remote_path: str) -> bool:
        """上传单个文件"""
        if not self._sftp:
            raise RuntimeError("未连接到服务器")

        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"本地文件不存在: {local_path}")
            return False

        try:
            self._sftp.put(str(local_path), remote_path)
            logger.info(f"上传成功: {local_path} -> {remote_path}")
            return True
        except Exception as e:
            logger.error(f"上传失败 {local_path}: {e}")
            return False

    def upload_batch(
        self,
        local_files: List[Union[str, Path]],
        remote_dir: str,
    ) -> TransferResult:
        """批量上传文件到目录"""
        result = TransferResult()

        for local_file in local_files:
            local_path = Path(local_file)
            remote_path = f"{remote_dir}/{local_path.name}"

            if self.upload_file(local_path, remote_path):
                result.success.append(str(local_path))
            else:
                result.failed.append(str(local_path))

        return result

    def upload_mapped(self, mappings: List[Dict[str, str]]) -> TransferResult:
        """按映射上传文件"""
        result = TransferResult()

        for item in mappings:
            local = item["local"]
            remote = item["remote"]

            if self.upload_file(local, remote):
                result.success.append(local)
            else:
                result.failed.append(local)

        return result

    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        """下载单个文件"""
        if not self._sftp:
            raise RuntimeError("未连接到服务器")

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._sftp.stat(remote_path)
        except FileNotFoundError:
            logger.warning(f"远程文件不存在: {remote_path}")
            return False

        try:
            self._sftp.get(remote_path, str(local_path))
            logger.info(f"下载成功: {remote_path} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"下载失败 {remote_path}: {e}")
            return False

    def download_mapped(
        self,
        mappings: List[Dict[str, str]],
        **format_args,
    ) -> TransferResult:
        """
        按映射下载文件

        Args:
            mappings: [{"remote": "...", "local": "..."}]
            **format_args: 用于格式化路径的参数
        """
        result = TransferResult()

        for item in mappings:
            remote = item["remote"].format(**format_args)
            local = item["local"].format(**format_args)

            if self.download_file(remote, local):
                result.success.append(local)
            else:
                result.failed.append(remote)

        return result

    def execute(self, command: str) -> CommandResult:
        """执行远程命令"""
        if not self._ssh:
            raise RuntimeError("未连接到服务器")

        _, stdout, stderr = self._ssh.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode("utf-8", errors="ignore")
        error = (
            stderr.read().decode("utf-8", errors="ignore") if exit_status != 0 else None
        )

        return CommandResult(exit_status=exit_status, output=output, error=error)
