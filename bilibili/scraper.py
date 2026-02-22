# bilibili/scraper.py
"""B站视频数据采集工作流"""

import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set, Union, Literal
from dataclasses import asdict

from common.config import get_paths, Paths
from common.logger import logger
from common.io import save_excel
from common.formatters import clean_text
from common.models import VideoInfo, SearchOptions, SearchRestrictions, ScraperConfig
from common.merge import RecordMerger
from ranking.streak import StreakManager
from bilibili.client import BilibiliClient


def transform_api_response(api_data: Dict[str, Any]) -> Dict[str, Any]:
    """将B站API响应转换为标准格式"""
    return {
        "bvid": api_data.get("bvid", ""),
        "aid": str(api_data.get("id", "")),
        "title": clean_text(api_data.get("title", "")),
        "uploader": api_data.get("upper", {}).get("name", ""),
        "copyright": api_data.get("copyright", 1),
        "pubdate": datetime.fromtimestamp(api_data.get("pubtime", 0)).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "duration": api_data.get("duration", 0),
        "page": api_data.get("page", 1),
        "tid": api_data.get("tid", 0),
        "view": api_data.get("cnt_info", {}).get("play", 0),
        "favorite": api_data.get("cnt_info", {}).get("collect", 0),
        "coin": api_data.get("cnt_info", {}).get("coin", 0),
        "like": api_data.get("cnt_info", {}).get("thumb_up", 0),
        "danmaku": api_data.get("cnt_info", {}).get("danmaku", 0),
        "reply": api_data.get("cnt_info", {}).get("reply", 0),
        "share": api_data.get("cnt_info", {}).get("share", 0),
        "image_url": api_data.get("cover", ""),
        "intro": api_data.get("intro", ""),
    }


def apply_filters(items: List[Any], filters: List) -> List[Any]:
    for f in filters:
        items = [item for item in items if f(item)]
    return items


def create_video_info_list(
    api_videos: List[Dict], local_index: Optional[Dict], merge_strategy
) -> List[VideoInfo]:
    videos = []
    for api_info in api_videos:
        aid_str = api_info.get("aid", "")
        try:
            local_info = local_index.get(aid_str, {}) if local_index else {}
            payload = merge_strategy(api_info, local_info)
            videos.append(VideoInfo(**payload))
        except Exception as e:
            logger.error(f"构建 VideoInfo 出错 (aid: {aid_str}): {e}")
    return videos


class BilibiliScraper:
    """B站视频数据工作流处理器"""

    today: datetime = (
        datetime.now() + timedelta(days=1)
        if datetime.now().hour >= 23
        else datetime.now()
    ).replace(hour=0, minute=0, second=0, microsecond=0)

    def __init__(
        self,
        client: BilibiliClient,
        mode: Literal["new", "old", "special", "hot_rank"],
        input_file: Union[str, Path, None] = None,
        days: int = 2,
        config: ScraperConfig = ScraperConfig(),
        search_options: List[SearchOptions] = None,
        search_restrictions: SearchRestrictions = None,
        paths: Optional[Paths] = None,
    ):
        self.client = client
        self.mode = mode
        self.config = config
        self.search_options = search_options or [SearchOptions()]
        self.search_restrictions = search_restrictions
        self.days = days
        self._paths = paths or get_paths()

        if self.config.OUTPUT_DIR is None:
            self.config.OUTPUT_DIR = self._paths.snapshot_main
        if self.config.COLLECTED_FILE is None:
            self.config.COLLECTED_FILE = self._paths.collected

        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        self.songs = pd.DataFrame()
        self.existing_bvids: Set[str] = set()

        self._init_mode(input_file)
        self.streak_manager = StreakManager(
            base_threshold=config.BASE_THRESHOLD,
            streak_threshold=config.STREAK_THRESHOLD,
            min_total_view=config.MIN_TOTAL_VIEW,
            update_cols=config.UPDATE_COLS,
        )

    def _init_mode(self, input_file):
        if self.mode == "new":
            self.filename = (
                self.config.OUTPUT_DIR / f"新曲{self.today.strftime('%Y%m%d')}.xlsx"
            )
            self.start_time = self.today - timedelta(days=self.days)
        elif self.mode == "old":
            self.filename = (
                self.config.OUTPUT_DIR / f"{self.today.strftime('%Y%m%d')}.xlsx"
            )
            self.songs = pd.read_excel(input_file)
            if "streak" not in self.songs.columns:
                self.songs["streak"] = 0
            if "aid" in self.songs.columns:
                self.songs["aid"] = (
                    self.songs["aid"].astype(str).str.replace(r"\.0$", "", regex=True)
                )
            else:
                self.songs["aid"] = ""
        elif self.mode == "special":
            self.filename = self.config.OUTPUT_DIR / f"{self.config.NAME}.xlsx"
        elif self.mode == "hot_rank":
            self.start_date = self.today
            self.end_date = self.start_date - timedelta(days=self.days)
            self.filename = (
                self.config.OUTPUT_DIR
                / f"{self.config.HOT_RANK_CATE_ID}-hot_rank_{self.end_date.strftime('%Y%m%d')}_to_{self.start_date.strftime('%Y%m%d')}.xlsx"
            )
            self.existing_bvids = self._load_existing_bvids(self._paths.collected)

    def _load_existing_bvids(self, path: Union[str, Path]) -> Set[str]:
        try:
            df = pd.read_excel(path, usecols=["bvid"])
            return set(df["bvid"].dropna().astype(str))
        except:
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
        merge_strategy = RecordMerger.new_song()
        videos = await self._process_pipeline(
            api_filters, merge_strategy, final_filters
        )
        return [asdict(v) for v in videos]

    async def process_old_songs(self) -> List[Dict[str, Any]]:
        logger.info("开始处理旧曲数据")
        census_mode = self.is_census_day()
        songs_to_process = (
            self.songs
            if census_mode
            else self.songs.loc[self.songs["streak"] < self.config.STREAK_THRESHOLD]
        )
        logger.info(
            f"{'普查' if census_mode else '常规'}模式：处理 {len(songs_to_process)} 个视频"
        )

        if songs_to_process.empty:
            return []

        api_filters = [lambda v: v.get("title", "") != "已失效视频"]
        merge_strategy = RecordMerger.old_song(self.config.LOCAL_METADATA_FIELDS)
        videos = await self._process_pipeline(
            api_filters, merge_strategy, [], aid_source=songs_to_process["aid"].tolist()
        )

        if videos:
            self.songs = self.streak_manager.update_songs(
                self.songs, videos, census_mode
            )
            usecols = self._paths.load_usecols("record")
            save_excel(self.songs, self._paths.collected, usecols=usecols)

        return [asdict(v) for v in videos]

    async def process_hot_rank_videos(self) -> None:
        logger.info(f"时间范围：{self.end_date:%Y-%m-%d} 至 {self.start_date:%Y-%m-%d}")
        all_videos = []
        current_date = self.start_date

        while current_date >= self.end_date:
            next_date = max(current_date - timedelta(days=90), self.end_date)
            time_from = next_date.strftime("%Y%m%d")
            time_to = current_date.strftime("%Y%m%d")

            raw_videos = await self.client.get_newlist_rank_videos(
                self.config.HOT_RANK_CATE_ID, time_from, time_to
            )
            if raw_videos:
                filtered = self._filter_hot_rank_videos(raw_videos)
                all_videos.extend(filtered)

            current_date = next_date - timedelta(days=1)
            await asyncio.sleep(2)

        if all_videos:
            all_videos.sort(key=lambda x: x.get("view", 0), reverse=True)
            logger.info(f"采集到 {len(all_videos)} 个新视频")
            await self.save_to_excel(all_videos)
        else:
            logger.info("未采集到新视频")

    # ==================== 内部方法 ====================

    async def _process_pipeline(
        self, api_filters, merge_strategy, final_filters, aid_source=None
    ):
        aids = aid_source if aid_source else await self._get_all_aids()
        logger.info(f"共获取 {len(aids)} 个 aid")
        if not aids:
            return []

        raw_data = await self._fetch_raw_video_data(aids)
        filtered_data = apply_filters(raw_data, api_filters)
        local_index = self._build_local_data_index()
        videos = create_video_info_list(filtered_data, local_index, merge_strategy)
        return apply_filters(videos, final_filters)

    def _build_time_filters(self):
        filters = []
        if self.mode == "new":
            filters.append(
                lambda v: datetime.strptime(v.pubdate, "%Y-%m-%d %H:%M:%S")
                > self.start_time
            )
        elif self.mode == "special":
            option = self.search_options[0] if self.search_options else None
            if option and option.time_start and option.time_end:
                start = f"{option.time_start} 00:00:00"
                end_date = datetime.strptime(option.time_end, "%Y-%m-%d") + timedelta(
                    days=1
                )
                end = end_date.strftime("%Y-%m-%d %H:%M:%S")
                filters.append(lambda v, s=start, e=end: s <= v.pubdate < e)
        return filters

    def _build_local_data_index(self):
        if self.songs.empty or "aid" not in self.songs.columns:
            return None
        return self.songs.set_index("aid").to_dict("index")

    async def _get_all_aids(self) -> List[str]:
        aids: Set[str] = set()
        for option in self.search_options:
            if option.video_zone_type is None:
                continue
            if self.mode == "new":
                option.time_start = self.start_time.strftime("%Y-%m-%d")
                option.time_end = self.today.strftime("%Y-%m-%d")

            logger.info(
                f"搜索：分区={option.video_zone_type}, 时间={option.time_start}~{option.time_end}"
            )
            found = await self.client.search_aids(
                self.config.KEYWORDS, option, self.search_restrictions
            )
            aids.update(found)
            await asyncio.sleep(self.config.SLEEP_TIME)

        if self.mode == "new":
            all_rids = {rid for opt in self.search_options for rid in opt.newlist_rids}
            for rid in all_rids:
                found = await self.client.get_newlist_aids(rid, 50, self.start_time)
                aids.update(found)
                await asyncio.sleep(self.config.SLEEP_TIME)

        return list(aids)

    async def _fetch_raw_video_data(self, aids: List[str]) -> List[Dict[str, Any]]:
        int_aids = [int(aid) for aid in aids if aid and aid.isdigit()]
        if not int_aids:
            return []
        stats = await self.client.get_batch_details(int_aids)
        return [transform_api_response(info) for info in stats.values()]

    def _filter_hot_rank_videos(self, raw_videos):
        filtered = []
        for video in raw_videos:
            view = int(video.get("play", 0))
            if 0 < view < self.config.MIN_TOTAL_VIEW:
                break
            bvid = video.get("bvid")
            if not bvid or bvid in self.existing_bvids:
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
                    "author": video.get("author", ""),
                    "image_url": video.get("pic", ""),
                }
            )
        return filtered

    async def save_to_excel(self, videos, usecols=None):
        if not videos:
            logger.info("没有数据需要保存")
            return
        df = pd.DataFrame(videos).sort_values(by="view", ascending=False)
        save_excel(df, self.filename, usecols=usecols)
