# 翻唱周刊.py
import asyncio

from common.dates import get_cover_weekly_dates
from ranking.processor import RankingProcessor


async def main():
    dates = get_cover_weekly_dates()
    processor = RankingProcessor(period="cover_weekly")
    await processor.run(dates=vars(dates))


if __name__ == "__main__":
    asyncio.run(main())
