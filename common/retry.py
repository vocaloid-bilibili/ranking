# common/retry.py
import asyncio
import time
from typing import TypeVar, Callable, Awaitable
from common.logger import logger

T = TypeVar("T")


class RetryHandler:
    def __init__(self, max_retries: int = 10, sleep_time: float = 0.5):
        self.max_retries = max_retries
        self.sleep_time = sleep_time

    def retry(self, func: Callable[..., T], *args, max_retries=None, **kwargs) -> T:
        retries = max_retries or self.max_retries
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"第 {attempt + 1}/{retries} 次尝试失败: {e}")
                time.sleep(self.sleep_time)
        raise Exception("超过最大重试次数")

    async def retry_async(
        self, func: Callable[..., Awaitable[T]], *args, max_retries=None, **kwargs
    ) -> T:
        retries = max_retries or self.max_retries
        for attempt in range(retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"第 {attempt + 1}/{retries} 次尝试失败: {e}")
                await asyncio.sleep(self.sleep_time)
        raise Exception("超过最大重试次数")
