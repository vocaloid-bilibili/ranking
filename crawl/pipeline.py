# crawl/pipeline.py
"""采集管线 — 协调 API 获取、streak 管理、文件持久化"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

from bilibili.scraper import BilibiliScraper
from common.config import Paths, get_paths
from common.dates import get_today
from common.io import get_latest_file, load_jsonl, save_jsonl
from common.logger import logger
from common.models import ScraperConfig
from crawl.streak import Streak


def _apply_filters(items: List[Any], filters: List) -> List[Any]:
    for f in filters:
        items = [item for item in items if f(item)]
    return items


def _merge_streak(
    api_videos: List[Dict],
    local_index: Optional[Dict[str, Dict]],
) -> List[Dict[str, Any]]:
    """将 API 数据与本地 streak 合并"""
    records = []
    for api_info in api_videos:
        aid_str = api_info.get("aid", "")
        try:
            local = local_index.get(aid_str, {}) if local_index else {}
            record = {**api_info, "streak": local.get("streak", 0)}
            records.append(record)
        except Exception as e:
            logger.error(f"构建记录出错 (aid: {aid_str}): {e}")
    return records


class CrawlPipeline:
    """采集管线"""

    def __init__(
        self,
        scraper: BilibiliScraper,
        mode: Literal["new", "old", "special", "hot_rank"],
        input_file: Union[str, Path, None] = None,
        days: int = 2,
        config: ScraperConfig = ScraperConfig(),
        paths: Optional[Paths] = None,
    ):
        self.scraper = scraper
        self.mode = mode
        self.config = config
        self.days = days
        self._paths = paths or get_paths()
        self.today = get_today()

        if self.config.OUTPUT_DIR is None:
            self.config.OUTPUT_DIR = self._paths.snapshot_main
        if self.config.COLLECTED_FILE is None:
            self.config.COLLECTED_FILE = self._paths.collected

        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        self.songs: List[Dict[str, Any]] = []
        self.existing_bvids: Set[str] = set()

        self._init_mode(input_file)
        self.streak_manager = Streak(
            base_threshold=config.BASE_THRESHOLD,
            streak_threshold=config.STREAK_THRESHOLD,
            min_total_view=config.MIN_TOTAL_VIEW,
        )

    def _init_mode(self, input_file):
        if self.mode == "new":
            self.filename = (
                self.config.OUTPUT_DIR / f"新曲{self.today.strftime('%Y%m%d')}.jsonl"
            )
            self.start_time = self.today - timedelta(days=self.days)
        elif self.mode == "old":
            self.filename = (
                self.config.OUTPUT_DIR / f"{self.today.strftime('%Y%m%d')}.jsonl"
            )
            self.songs = load_jsonl(input_file)
            for row in self.songs:
                row.setdefault("streak", 0)
                row["aid"] = str(row.get("aid", "")).replace(".0", "")
        elif self.mode == "special":
            self.filename = self.config.OUTPUT_DIR / f"{self.config.NAME}.jsonl"
        elif self.mode == "hot_rank":
            self.start_date = self.today
            self.end_date = self.start_date - timedelta(days=self.days)
            self.filename = (
                self.config.OUTPUT_DIR
                / f"{self.config.HOT_RANK_CATE_ID}-hot_rank_{self.end_date.strftime('%Y%m%d')}_to_{self.start_date.strftime('%Y%m%d')}.jsonl"
            )
            self.existing_bvids = self._load_existing_bvids(self._paths.collected)

    def _load_existing_bvids(self, path: Union[str, Path]) -> Set[str]:
        try:
            records = load_jsonl(path, usecols=["bvid"])
            return {r["bvid"] for r in records if r.get("bvid")}
        except (FileNotFoundError, ValueError, KeyError):
            return set()

    def is_census_day(self) -> bool:
        return (self.today.weekday() == 5) or (self.today.day == 1)

    # ==================== 处理入口 ====================

    async def process_new_songs(self) -> List[Dict[str, Any]]:
        logger.info("开始处理新曲数据")
        api_filters = [
            lambda v: v.get("title", "") != "已失效视频",
            lambda v: v.get("duration", 0) > self.config.MIN_VIDEO_DURATION,
        ]
        final_filters = self._build_time_filters()
        return await self._process_pipeline(api_filters, final_filters)

    async def process_old_songs(self) -> List[Dict[str, Any]]:
        logger.info("开始处理旧曲数据")
        census_mode = self.is_census_day()
        songs_to_process = self.streak_manager.get_songs_to_update(
            self.songs, census_mode
        )
        logger.info(
            f"{'普查' if census_mode else '常规'}模式：处理 {len(songs_to_process)} 个视频"
        )

        if not songs_to_process:
            return []

        aid_source = [str(r.get("aid", "")) for r in songs_to_process]

        api_filters = []
        all_videos = await self._process_pipeline(
            api_filters, [], aid_source=aid_source
        )

        if not all_videos:
            return []

        failed_bvids = {v["bvid"] for v in all_videos if v.get("title") == "已失效视频"}
        valid_videos = [v for v in all_videos if v["bvid"] not in failed_bvids]

        if failed_bvids:
            logger.info(f"发现 {len(failed_bvids)} 个已失效视频")

        old_views = self._load_old_views()

        self.songs = self.streak_manager.update_songs(
            self.songs, valid_videos, old_views, census_mode, failed_bvids
        )
        usecols = self._paths.load_usecols("collected")
        save_jsonl(self.songs, self._paths.collected, usecols=usecols)

        return valid_videos

    def _load_old_views(self) -> Dict[str, int]:
        latest = get_latest_file(self._paths.snapshot_main, "*.jsonl")
        if not latest:
            logger.warning("未找到上期快照，old_views 为空")
            return {}
        records = load_jsonl(latest, usecols=["bvid", "view"])
        return {r["bvid"]: int(r.get("view", 0)) for r in records if r.get("bvid")}

    async def process_hot_rank_videos(self) -> None:
        logger.info(f"时间范围：{self.end_date:%Y-%m-%d} 至 {self.start_date:%Y-%m-%d}")
        all_videos: List[Dict[str, Any]] = []
        current_date = self.start_date
        should_stop_all = False

        while current_date >= self.end_date and not should_stop_all:
            next_date = max(current_date - timedelta(days=90), self.end_date)
            time_from = next_date.strftime("%Y%m%d")
            time_to = current_date.strftime("%Y%m%d")

            raw_videos = await self.scraper.fetch_hot_rank_page(
                self.config.HOT_RANK_CATE_ID, time_from, time_to
            )

            if raw_videos:
                filtered, should_stop = self.scraper.filter_hot_rank_videos(
                    raw_videos, self.existing_bvids
                )
                all_videos.extend(filtered)
                if should_stop:
                    logger.info("达到低播放量停止条件，结束采集")
                    should_stop_all = True

            current_date = next_date - timedelta(days=1)
            await asyncio.sleep(2)

        if all_videos:
            all_videos.sort(key=lambda x: x.get("view", 0), reverse=True)
            logger.info(f"采集到 {len(all_videos)} 个新视频")
            save_jsonl(all_videos, self.filename)
        else:
            logger.info("未采集到新视频")

    # ==================== 内部方法 ====================

    async def _process_pipeline(
        self, api_filters, final_filters, aid_source=None
    ) -> List[Dict[str, Any]]:
        if aid_source:
            aids = aid_source
        else:
            aids = await self.scraper.search_aids(
                self.start_time, self.today, self.mode
            )
        logger.info(f"共获取 {len(aids)} 个 aid")
        if not aids:
            return []

        raw_data = await self.scraper.fetch_video_details(aids)
        filtered_data = _apply_filters(raw_data, api_filters)
        local_index = self._build_local_data_index()
        videos = _merge_streak(filtered_data, local_index)
        return _apply_filters(videos, final_filters)

    def _build_time_filters(self):
        filters = []
        if self.mode == "new":
            filters.append(
                lambda v: (
                    datetime.strptime(v["pubdate"], "%Y-%m-%d %H:%M:%S")
                    > self.start_time
                )
            )
        elif self.mode == "special":
            option = (
                self.scraper.search_options[0] if self.scraper.search_options else None
            )
            if option and option.time_start and option.time_end:
                start = f"{option.time_start} 00:00:00"
                end_date = datetime.strptime(option.time_end, "%Y-%m-%d") + timedelta(
                    days=1
                )
                end = end_date.strftime("%Y-%m-%d %H:%M:%S")
                filters.append(lambda v, s=start, e=end: s <= v["pubdate"] < e)
        return filters

    def _build_local_data_index(self) -> Optional[Dict[str, Dict]]:
        if not self.songs:
            return None
        return {str(r.get("aid", "")): r for r in self.songs}
