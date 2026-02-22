# 月刊.py
import asyncio
from ranking.processor import RankingProcessor
from common.dates import get_monthly_dates


async def main():
    dates = get_monthly_dates()
    processor = RankingProcessor(period="monthly")
    await processor.run(dates=vars(dates))


if __name__ == "__main__":
    asyncio.run(main())
