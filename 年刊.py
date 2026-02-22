# 年刊.py
import asyncio
from ranking.processor import RankingProcessor


async def main():
    """生成年刊"""
    dates = {
        "old_date": "20250101",
        "new_date": "20260101",
        "target_date": "2025",
    }
    processor = RankingProcessor(period="annual")
    await processor.run(dates=dates)


if __name__ == "__main__":
    asyncio.run(main())
