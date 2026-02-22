# 周刊.py
import asyncio
from ranking.processor import RankingProcessor
from common.dates import get_weekly_dates


async def main():
    dates = get_weekly_dates()
    processor = RankingProcessor(period="weekly")
    await processor.run(dates=vars(dates))


if __name__ == "__main__":
    asyncio.run(main())
