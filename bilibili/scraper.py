# bilibili/scraper.py
"""B站 API 数据获取与转换"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from bilibili.client import BilibiliClient
from common.formatters import clean_text, tid_to_tname
from common.logger import logger
from common.models import ScraperConfig, SearchOptions, SearchRestrictions


def transform_api_response(api_data: Dict[str, Any]) -> Dict[str, Any]:
    """将B站API响应转换为标准格式"""
    return {
        "title": clean_text(api_data.get("title", "")),
        "bvid": api_data.get("bvid", ""),
        "aid": str(api_data.get("id", "")),
        "uploader": api_data.get("upper", {}).get("name", "-"),
        "copyright": api_data.get("copyright", 1),
        "pubdate": datetime.fromtimestamp(api_data.get("pubtime", 0)).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "duration": api_data.get("duration", 0),
        "page": api_data.get("page", 1),
        "thumbnail": api_data.get("cover", ""),
        "view": api_data.get("cnt_info", {}).get("play", 0),
        "favorite": api_data.get("cnt_info", {}).get("collect", 0),
        "coin": api_data.get("cnt_info", {}).get("coin", 0),
        "like": api_data.get("cnt_info", {}).get("thumb_up", 0),
        "danmaku": api_data.get("cnt_info", {}).get("danmaku", 0),
        "reply": api_data.get("cnt_info", {}).get("reply", 0),
        "share": api_data.get("cnt_info", {}).get("share", 0),
        "tid": tid_to_tname(api_data.get("tid", 0)),
        "intro": api_data.get("intro", ""),
    }


class BilibiliScraper:
    """B站 API 数据获取器 — 只负责与 B站 API 交互"""

    def __init__(
        self,
        client: BilibiliClient,
        config: ScraperConfig = ScraperConfig(),
        search_options: Optional[List[SearchOptions]] = None,
        search_restrictions: Optional[SearchRestrictions] = None,
    ):
        self.client = client
        self.config = config
        self.search_options = search_options or [SearchOptions()]
        self.search_restrictions = search_restrictions

    async def fetch_video_details(self, aids: List[str]) -> List[Dict[str, Any]]:
        """批量获取视频详情，返回标准格式"""
        int_aids = [int(aid) for aid in aids if aid and aid.isdigit()]
        if not int_aids:
            return []
        stats = await self.client.get_batch_details(int_aids)
        return [transform_api_response(info) for info in stats.values()]

    async def search_aids(
        self,
        start_time: datetime,
        today: datetime,
        mode: str,
    ) -> List[str]:
        """通过搜索和 newlist 获取 aid 列表"""
        aids: Set[str] = set()
        for option in self.search_options:
            if option.video_zone_type is None:
                continue
            if mode == "new":
                option.time_start = start_time.strftime("%Y-%m-%d")
                option.time_end = today.strftime("%Y-%m-%d")

            logger.info(
                f"搜索：分区={option.video_zone_type}, 时间={option.time_start}~{option.time_end}"
            )
            found = await self.client.search_aids(
                self.config.KEYWORDS, option, self.search_restrictions
            )
            aids.update(found)
            await asyncio.sleep(self.config.SLEEP_TIME)

        if mode == "new":
            all_rids = {rid for opt in self.search_options for rid in opt.newlist_rids}
            for rid in all_rids:
                found = await self.client.get_newlist_aids(rid, 50, start_time)
                aids.update(found)
                await asyncio.sleep(self.config.SLEEP_TIME)

        return list(aids)

    async def fetch_hot_rank_page(
        self, cate_id: int, time_from: str, time_to: str
    ) -> List[Dict[str, Any]]:
        """获取热门排行一页数据"""
        return await self.client.get_newlist_rank_videos(cate_id, time_from, time_to)

    def filter_hot_rank_videos(
        self,
        raw_videos: List[Dict[str, Any]],
        existing_bvids: Set[str],
    ) -> tuple[List[Dict[str, Any]], bool]:
        """过滤热榜视频，返回 (filtered, should_stop)"""
        filtered = []
        low_view_count = 0

        for video in raw_videos:
            view = int(video.get("play", 0))
            bvid = video.get("bvid")

            if 0 < view < self.config.MIN_TOTAL_VIEW:
                low_view_count += 1
                continue

            if not bvid or bvid in existing_bvids:
                continue

            duration = int(video.get("duration", 0))
            if duration <= self.config.MIN_VIDEO_DURATION:
                continue

            filtered.append(
                {
                    "title": clean_text(video.get("title", "")),
                    "bvid": bvid,
                    "aid": video.get("id"),
                    "view": view,
                    "pubdate": video.get("pubdate"),
                    "uploader": video.get("author", ""),
                    "thumbnail": video.get("pic", ""),
                }
            )

        should_stop = low_view_count >= self.config.LOW_VIEW_STOP_COUNT
        return filtered, should_stop
