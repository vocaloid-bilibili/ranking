# 抓取数据.py
import asyncio

from bilibili.client import BilibiliClient
from bilibili.scraper import BilibiliScraper
from common.config import get_paths
from common.models import ScraperConfig


async def main():
    paths = get_paths()

    config = ScraperConfig(OUTPUT_DIR=paths.snapshot_main)
    client = BilibiliClient(config=config)
    scraper = BilibiliScraper(
        client=client,
        mode="old",
        config=config,
        input_file=paths.collected,
    )

    try:
        videos = await scraper.process_old_songs()
        usecols = paths.load_usecols("stat")
        await scraper.save_to_excel(videos, usecols=usecols)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
