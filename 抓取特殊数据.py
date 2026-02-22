# 抓取特殊数据.py
import asyncio
import yaml

from common.config import get_paths
from common.models import ScraperConfig, SearchOptions
from bilibili.client import BilibiliClient
from bilibili.scraper import BilibiliScraper


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
        mode="special",
        config=config,
        search_options=search_options,
    )

    try:
        videos = await scraper.process_new_songs()
        await scraper.save_to_excel(videos)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
