# scripts/crawl_special.py
import asyncio

import _bootstrap  # noqa: F401
import yaml

from bilibili.client import BilibiliClient
from bilibili.scraper import BilibiliScraper
from common.config import get_paths
from common.io import save_jsonl
from common.models import ScraperConfig, SearchOptions
from crawl.pipeline import CrawlPipeline


async def main():
    paths = get_paths()
    cfg = yaml.safe_load(paths.special_config.read_text(encoding="utf-8"))

    config = ScraperConfig(
        KEYWORDS=cfg["keywords"],
        OUTPUT_DIR=paths.special_data,
        NAME=cfg["name"],
    )

    search_options = [
        SearchOptions(
            time_start=cfg.get("time_start", "2025-09-27"),
            time_end=cfg.get("time_end", "2025-11-27"),
            video_zone_type=cfg.get("video_zone_type", 0),
        )
    ]

    client = BilibiliClient(config=config)
    scraper = BilibiliScraper(
        client=client,
        config=config,
        search_options=search_options,
    )
    pipeline = CrawlPipeline(
        scraper=scraper,
        mode="special",
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
