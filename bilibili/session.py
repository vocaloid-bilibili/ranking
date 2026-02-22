# bilibili/session.py
"""B站Session管理"""

import time
import asyncio
import aiohttp
from typing import Optional
from bilibili_api import request_settings

from common.logger import logger
from common.proxy import Proxy


async def reset_bilibili_api_state():
    """重置 bilibili-api 库内部状态"""
    try:
        from bilibili_api.utils import network

        for attr in ["__session", "_session", "session", "__client", "_client"]:
            if hasattr(network, attr):
                client = getattr(network, attr)
                if client is not None:
                    try:
                        if hasattr(client, "aclose"):
                            await client.aclose()
                        elif hasattr(client, "close"):
                            client.close()
                    except:
                        pass
                    setattr(network, attr, None)
        for attr in [
            "wbi_mixin_key",
            "last_refresh_time",
            "_wbi_mixin_key",
            "_last_refresh_time",
        ]:
            if hasattr(network, attr):
                setattr(network, attr, None if "key" in attr else 0)
    except Exception as e:
        logger.warning(f"重置 bilibili-api 状态时出错: {e}")


class SessionManager:
    """管理aiohttp Session和bilibili-api状态"""

    def __init__(
        self,
        proxy: Optional[Proxy] = None,
        restart_interval: int = 90,
        restart_cooldown: int = 15,
    ):
        self.proxy = proxy
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_restart_time: float = 0
        self._restart_interval = restart_interval
        self._restart_cooldown = restart_cooldown
        self._consecutive_zeros: int = 0

        if self.proxy:
            request_settings.set_proxy(self.proxy.proxy_server)

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def check_and_restart(self, force: bool = False):
        now = time.time()
        if self._last_restart_time == 0:
            self._last_restart_time = now
            return

        elapsed = now - self._last_restart_time
        should_restart = force or (elapsed >= self._restart_interval)

        if should_restart:
            reason = "强制" if force else f"运行{int(elapsed)}秒"
            logger.info(f"【{reason}重启】开始重置状态...")

            await self.close_session()
            await reset_bilibili_api_state()

            if self.proxy:
                request_settings.set_proxy(self.proxy.proxy_server)

            logger.info(f"冷却等待 {self._restart_cooldown} 秒...")
            await asyncio.sleep(self._restart_cooldown)

            self.session = aiohttp.ClientSession()
            self._last_restart_time = time.time()
            self._consecutive_zeros = 0
            logger.info("重启完成")

    def record_zero_batch(self) -> bool:
        self._consecutive_zeros += 1
        if self._consecutive_zeros >= 2:
            logger.warning(f"连续 {self._consecutive_zeros} 批全0，触发强制重启...")
            return True
        return False

    def reset_zero_count(self):
        self._consecutive_zeros = 0
