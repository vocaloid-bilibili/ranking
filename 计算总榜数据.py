# 计算总榜数据.py
import asyncio
from ranking.processor import RankingProcessor


async def main():
    processor = RankingProcessor(period="special")
    await processor.run(song_data="20260222")


if __name__ == "__main__":
    asyncio.run(main())
