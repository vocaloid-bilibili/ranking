# scripts/crawl_snapshot.py
import asyncio

import _bootstrap  # noqa: F401

from bilibili.client import BilibiliClient
from bilibili.scraper import BilibiliScraper
from common.config import get_paths
from common.io import save_jsonl
from common.models import ScraperConfig
from crawl.pipeline import CrawlPipeline


async def main():
    paths = get_paths()

    config = ScraperConfig(OUTPUT_DIR=paths.snapshot_main)
    client = BilibiliClient(config=config)
    scraper = BilibiliScraper(client=client, config=config)
    pipeline = CrawlPipeline(
        scraper=scraper,
        mode="old",
        config=config,
        input_file=paths.collected,
    )

    try:
        videos = await pipeline.process_old_songs()
        if videos:
            usecols = paths.load_usecols("snapshot_main")
            save_jsonl(videos, pipeline.filename, usecols=usecols)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
