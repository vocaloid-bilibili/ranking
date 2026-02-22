# 历史回顾.py
import asyncio
from ranking.processor import RankingProcessor
from common.dates import get_history_dates


async def main():
    dates = get_history_dates()
    processor = RankingProcessor(period="history")
    await processor.run(dates=dates)


if __name__ == "__main__":
    asyncio.run(main())
