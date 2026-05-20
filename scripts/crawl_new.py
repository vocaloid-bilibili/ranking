# scripts/crawl_new.py
import asyncio

import _bootstrap  # noqa: F401

from bilibili.client import BilibiliClient
from bilibili.scraper import BilibiliScraper
from common.config import get_paths
from common.io import save_jsonl
from common.models import ScraperConfig, SearchOptions
from crawl.pipeline import CrawlPipeline


async def main():
    paths = get_paths()
    keywords = paths.load_keywords()

    config = ScraperConfig(KEYWORDS=keywords, OUTPUT_DIR=paths.snapshot_new)
    search_options = [
        SearchOptions(video_zone_type=0),
        SearchOptions(video_zone_type=3),
        SearchOptions(video_zone_type=30),
        SearchOptions(newlist_rids=[30]),
    ]

    client = BilibiliClient(config=config)
    scraper = BilibiliScraper(
        client=client,
        config=config,
        search_options=search_options,
    )
    pipeline = CrawlPipeline(
        scraper=scraper,
        mode="new",
        days=2,
        config=config,
    )

    try:
        videos = await pipeline.process_new_songs()
        if videos:
            usecols = paths.load_usecols("snapshot_new")
            save_jsonl(videos, pipeline.filename, usecols=usecols)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
