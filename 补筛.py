# 补筛.py
import asyncio

from common.config import get_paths
from common.models import ScraperConfig
from bilibili.client import BilibiliClient
from bilibili.scraper import BilibiliScraper


async def main():
    paths = get_paths()
    output_dir = paths.export / "hot_rank"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ScraperConfig(
        OUTPUT_DIR=output_dir,
        COLLECTED_FILE=paths.collected,
    )

    client = BilibiliClient(config=config)
    scraper = BilibiliScraper(
        client=client,
        mode="hot_rank",
        days=15,
        config=config,
    )

    try:
        await scraper.process_hot_rank_videos()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
