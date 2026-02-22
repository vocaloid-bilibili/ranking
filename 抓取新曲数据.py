# 抓取新曲数据.py
import asyncio

from common.config import get_paths
from common.models import ScraperConfig, SearchOptions
from bilibili.client import BilibiliClient
from bilibili.scraper import BilibiliScraper


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
        mode="new",
        days=2,
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
