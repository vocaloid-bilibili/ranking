# 合并.py
"""合并数据"""

import asyncio

from ranking.processor import RankingProcessor


async def main():
    processor = RankingProcessor(period="daily_combination")
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())
