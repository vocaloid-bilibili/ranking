# src/bilibili_scraper.py
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import asdict
from typing import List, Optional, Dict, Literal, Any, Set, Union, Callable, Coroutine
from pathlib import Path
import json

from utils.logger import logger
from utils.io_utils import save_to_excel
from utils.formatters import clean_tags, convert_duration
from utils.calculator import calculate_threshold, calculate_failed_mask
from utils.dataclass import VideoInfo, SearchOptions, SearchRestrictions, Config
from src.bilibili_api_client import BilibiliApiClient

class BilibiliScraper:
    """
    B站视频数据工作流处理器。
    负责根据模式编排数据获取、处理和更新的流程。
    """
    today: datetime = (datetime.now() + timedelta(days=1) if datetime.now().hour >= 23 else datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)

    def __init__(self, 
                 api_client: BilibiliApiClient,
                 mode: Literal["new", "old", "special", "hot_rank"], 
                 input_file: Union[str, Path, None] = None, 
                 days: int = 2,
                 config: Config = Config(), 
                 search_options: list[SearchOptions] = [SearchOptions()],
                 search_restrictions: SearchRestrictions | None = None,
                ):
        self.api_client = api_client
        self.mode = mode
        self.config = config
        self.search_options = search_options
        self.search_restrictions = search_restrictions
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        self.songs = pd.DataFrame()
        self.existing_bvids: Set[str] = set()

        if self.mode == "new":
            self.filename = self.config.OUTPUT_DIR / f"新曲{self.today.strftime('%Y%m%d')}.xlsx"
            self.start_time = self.today - timedelta(days=days)
        elif self.mode == "old":
            self.filename = self.config.OUTPUT_DIR / f"{self.today.strftime('%Y%m%d')}.xlsx"
            self.songs = pd.read_excel(input_file)
            if 'streak' not in self.songs.columns:
                self.songs['streak'] = 0
            if 'aid' in self.songs.columns:
                self.songs['aid'] = self.songs['aid'].astype(str).str.replace(r'\.0$', '', regex=True)
            else:
                self.songs['aid'] = ''
        elif self.mode == "special":
            self.filename = self.config.OUTPUT_DIR / f"{self.config.NAME}.xlsx"

        elif self.mode == "hot_rank":
            self.start_date = self.today
            self.end_date = self.start_date - timedelta(days=days)
            self.filename = self.config.OUTPUT_DIR / f"{self.config.HOT_RANK_CATE_ID}-hot_rank_{self.end_date.strftime('%Y%m%d')}_to_{self.start_date.strftime('%Y%m%d')}.xlsx"
            self.existing_bvids = self._load_existing_bvids("收录曲目.xlsx")

    def _load_existing_bvids(self, file_path: Union[str, Path]) -> Set[str]:
        try:
            existing_df = pd.read_excel(file_path, usecols=['bvid'])
            bvids = set(existing_df['bvid'].dropna().astype(str))
            logger.info(f"从 {file_path} 加载了 {len(bvids)} 个已收录的 bvid。")
            return bvids
        except FileNotFoundError:
            logger.warning(f"{file_path} 不存在，未加载任何已收录的 bvid。")
            return set()
        except Exception as e:
            logger.error(f"加载已收录 bvid 时出错: {e}")
            return set()
        
    async def process_new_songs(self) -> List[Dict[str, Any]]:
        """处理新曲数据的主入口。"""
        logger.info("开始处理新曲数据")
        api_filters = [
            lambda v: v.get('title', '') != "已失效视频",
            lambda v: v.get('duration', 0) > self.config.MIN_VIDEO_DURATION
        ]
        final_filters = []
        if self.mode == "new":
            final_filters.append(lambda v: datetime.strptime(v.pubdate, '%Y-%m-%d %H:%M:%S') > self.start_time)
        elif self.mode == "special":
            option = self.search_options[0] if self.search_options else None
            if option and option.time_start and option.time_end:
                start = f"{option.time_start} 00:00:00"
                end_date = datetime.strptime(option.time_end, "%Y-%m-%d") + timedelta(days=1)
                end = end_date.strftime("%Y-%m-%d %H:%M:%S")
                final_filters.append(lambda v: start <= v.pubdate < end)
        video_objects = await self._process_song_data_pipeline(
            aid_source_coro=self._get_all_aids(),
            api_filters=api_filters,
            merging_strategy=self._create_new_song_merge_payload,
            final_filters=final_filters
        )
        return [asdict(v) for v in video_objects]
    
    async def process_old_songs(self) -> List[Dict[str, Any]]:
        """处理旧曲数据的主入口。"""
        logger.info("开始处理旧曲数据")
        census_mode = self.is_census_day()
        
        songs_to_process_df = self.songs if census_mode else self.songs.loc[self.songs['streak'] < self.config.STREAK_THRESHOLD]
        logger.info(f"{'普查' if census_mode else '常规'}模式：准备处理 {len(songs_to_process_df)} 个视频")
        if songs_to_process_df.empty: return []

        async def get_old_song_aids():
            return songs_to_process_df['aid'].tolist()

        api_filters = [lambda v: v.get('title', '') != "已失效视频"]

        video_objects = await self._process_song_data_pipeline(
            aid_source_coro=get_old_song_aids(),
            api_filters=api_filters,
            merging_strategy=self._create_old_song_merge_payload,
            final_filters=[] 
        )
        
        if video_objects:
            self.update_recorded_songs(video_objects, census_mode)
            
        return [asdict(v) for v in video_objects]

    async def _process_song_data_pipeline(
        self,
        aid_source_coro: Coroutine[Any, Any, List[str]],
        api_filters: List[Callable],
        merging_strategy: Callable,
        final_filters: List[Callable]
    ) -> List[VideoInfo]:
        """
        处理歌曲数据的通用流水线。
        """
        # 获取 AIDs
        aids = await aid_source_coro
        logger.info(f"共获取 {len(aids)} 个 aid")
        if not aids: return []

        # 获取原始 API 数据
        raw_api_data = await self._fetch_raw_video_data(aids)

        # 对原始 API 数据进行初步过滤
        filtered_api_data = self._apply_filters(raw_api_data, api_filters)

        # 合并数据并创建 VideoInfo 对象列表
        videos = self._create_video_info_list(filtered_api_data, merging_strategy)

        # 对创建好的 VideoInfo 对象进行最终过滤
        final_videos = self._apply_filters(videos, final_filters)

        return final_videos

    def _create_new_song_merge_payload(self, api_info: Dict, local_info: Dict) -> Dict:
        """为新曲模式创建合并后的数据字典。"""
        return {
            **api_info,
            'duration': convert_duration(api_info.get('duration', 0)),
            'name': api_info.get('title', ''), 
            'author': api_info.get('uploader', ''),
            'synthesizer': "", 'vocal': "", 'type': ""
        }

    def _create_old_song_merge_payload(self, api_info: Dict, local_info: Dict) -> Dict:
        """为旧曲模式创建合并后的数据字典。"""
        synced_local_data = {k: local_info.get(k, '') for k in self.config.LOCAL_METADATA_FIELDS}
        if not local_info:
             synced_local_data['name'] = api_info.get('title', '')
             synced_local_data['author'] = api_info.get('uploader', '')

        if synced_local_data.get('copyright') not in [100, 101]:
            synced_local_data['copyright'] = api_info.get('copyright')

        return { **api_info, **synced_local_data, 'duration': convert_duration(api_info.get('duration', 0)) }

    def _apply_filters(self, items: List[Any], filters: List[Callable[[Any], bool]]) -> List[Any]:
        """对列表应用一系列过滤函数。"""
        for f in filters:
            items = [item for item in items if f(item)]
        return items

    def _create_video_info_list(self, api_videos_data: List[Dict], merging_strategy: Callable) -> List[VideoInfo]:
        """根据API数据和传入的合并策略，创建VideoInfo对象列表。"""
        videos: List[VideoInfo] = []
        songs_by_aid = self.songs.set_index('aid') if not self.songs.empty and 'aid' in self.songs.columns else None

        for api_info in api_videos_data:
            aid_str = api_info['aid']
            try:
                local_info = {}
                if songs_by_aid is not None and aid_str in songs_by_aid.index:
                    local_info = songs_by_aid.loc[aid_str].to_dict()

                final_payload = merging_strategy(api_info, local_info)
                videos.append(VideoInfo(**final_payload))
            except Exception as e:
                logger.error(f"构建 VideoInfo 对象时出错 (aid: {aid_str}): {e}")
        return videos

    async def _fetch_raw_video_data(self, aids: List[str]) -> List[Dict[str, Any]]:
        """获取原始数据"""
        int_aids = [int(aid) for aid in aids if aid and aid.isdigit()]
        if not int_aids: return []
        
        stats = await self.api_client.get_batch_details_by_aid(int_aids)
        
        videos_data = []
        for aid, info in stats.items():
            videos_data.append({
                'bvid': info.get('bvid', ''), 'aid': str(aid), 'title': clean_tags(info.get('title', '')),
                'uploader': info.get('upper', {}).get('name', ''), 'copyright': info.get('copyright', 1),
                'pubdate': datetime.fromtimestamp(info.get('pubtime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'duration': info.get('duration', 0), # 保持为整数
                'page': info.get('page', 1), 'view': info.get('cnt_info', {}).get('play', 0),
                'favorite': info.get('cnt_info', {}).get('collect', 0), 'coin': info.get('cnt_info', {}).get('coin', 0),
                'like': info.get('cnt_info', {}).get('thumb_up', 0), 'danmaku': info.get('cnt_info', {}).get('danmaku', 0),
                'reply': info.get('cnt_info', {}).get('reply', 0), 'share': info.get('cnt_info', {}).get('share', 0),
                'image_url': info.get('cover', ''), 'intro': info.get('intro', '')
            })
        return videos_data

    def is_census_day(self) -> bool:
        return (self.today.weekday() == 5) or (self.today.day == 1)

    async def _get_all_aids(self) -> List[str]:
        aids: Set[str] = set()
        search_tasks = [opt for opt in self.search_options if opt.video_zone_type is not None]
        if search_tasks:
            for option in search_tasks:
                if self.mode == "new":
                    option.time_start = self.start_time.strftime('%Y-%m-%d')
                    option.time_end = self.today.strftime('%Y-%m-%d')
                found_aids = await self.api_client.get_aids_from_search(self.config.KEYWORDS, option, self.search_restrictions)
                aids.update(found_aids)
                await asyncio.sleep(self.config.SLEEP_TIME)
        
        if self.mode == "new":
            all_newlist_rids = {rid for opt in self.search_options for rid in opt.newlist_rids}
            if all_newlist_rids:
                for rid in all_newlist_rids:
                    found_aids = await self.api_client.get_aids_from_newlist(rid=rid, ps=50, start_time=self.start_time)
                    aids.update(found_aids)
                    await asyncio.sleep(self.config.SLEEP_TIME)

        return list(aids)

    def _process_streaks(self, old_views: pd.Series, updated_ids: pd.Index, census_mode: bool):
        self.songs = self.songs.set_index('bvid')
        for bvid in updated_ids:
            if bvid not in self.songs.index: continue
            new_view = self.songs.at[bvid, 'view']
            old_view = old_views.get(bvid, new_view)
            actual_incr = new_view - old_view
            current_streak = self.songs.at[bvid, 'streak']
            threshold = calculate_threshold(current_streak, census_mode, self.config.BASE_THRESHOLD, self.config.STREAK_THRESHOLD)
            condition = (new_view < self.config.MIN_TOTAL_VIEW) and (actual_incr < threshold)
            self.songs.at[bvid, 'streak'] = current_streak + 1 if condition else 0
        
        if not census_mode:
            unprocessed = ~self.songs.index.isin(updated_ids) & ~self.songs['is_failed']
            self.songs.loc[unprocessed, 'streak'] += 1
        
        self.songs.loc[self.songs['is_failed'], 'streak'] = 0
        self.songs.reset_index(inplace=True)

    def update_recorded_songs(self, videos: List[VideoInfo], census_mode: bool):
        if not videos: return
        update_df = pd.DataFrame([asdict(v) for v in videos])
        update_df = update_df[self.config.UPDATE_COLS]

        old_views = self.songs.set_index('bvid')['view']
        self.songs['is_failed'] = calculate_failed_mask(self.songs, update_df, census_mode, self.config.STREAK_THRESHOLD)
        
        self.songs = self.songs.set_index('bvid')
        update_df = update_df.set_index('bvid')
        
        self.songs.update(update_df)
        self.songs.reset_index(inplace=True)
        self._process_streaks(old_views, update_df.index, census_mode)
        
        self.songs = self.songs.sort_values(['is_failed', 'view'], ascending=[False, False]).drop('is_failed', axis=1)
        
        usecols = json.load(Path('config/usecols.json').open(encoding='utf-8'))["columns"]["record"]
        save_to_excel(self.songs, "收录曲目.xlsx", usecols=usecols)
    
    async def process_hot_rank_videos(self) -> None:
        """
        处理热门榜视频数据的主入口。
        """
        logger.info(f"时间范围：{self.end_date.strftime('%Y-%m-%d')} 至 {self.start_date.strftime('%Y-%m-%d')}")
        
        all_videos_data = []
        current_date = self.start_date

        while current_date >= self.end_date:
            next_date = max(current_date - timedelta(days=90), self.end_date)
            time_from = next_date.strftime('%Y%m%d')
            time_to = current_date.strftime('%Y%m%d')

            raw_videos = await self.api_client.get_videos_from_newlist_rank(
                self.config.HOT_RANK_CATE_ID, time_from, time_to
            )
            
            if not raw_videos:
                current_date = next_date - timedelta(days=1)
                await asyncio.sleep(1)
                continue

            stop_processing_this_chunk = False
            for video in raw_videos:
                if stop_processing_this_chunk:
                    break
                
                view_count = int(video.get('play'))
                if 0 < view_count < self.config.MIN_TOTAL_VIEW:
                    logger.info(f"播放量{view_count}低于{self.config.MIN_TOTAL_VIEW}，结束当前时段")
                    stop_processing_this_chunk = True
                    continue

                bvid = video.get('bvid')
                if not bvid or bvid in self.existing_bvids:
                    continue
                
                duration = int(video.get('duration'))
                if duration <= self.config.MIN_VIDEO_DURATION:
                    continue

                all_videos_data.append({
                    'title': clean_tags(video.get('title', '')),
                    'bvid': bvid,
                    'aid': video.get('id'),
                    'view': view_count,
                    'pubdate': video.get('pubdate'),
                    'author': video.get('author', ''),
                    'image_url': video.get('pic', '')
                })

            current_date = next_date - timedelta(days=1)
            await asyncio.sleep(2)

        if not all_videos_data:
            logger.info("未采集到新视频。") 
            return
            
        videos = sorted(all_videos_data, key=lambda x: x.get('view', 0), reverse=True)
        logger.info(f"采集到 {len(videos)} 个新视频。")
        await self.save_to_excel(videos)

    async def save_to_excel(self, videos: List[Dict[str, Any]], usecols: Optional[List[str]] = None) -> None:
        if not videos:
            logger.info("没有数据需要保存。")
            return
        df = pd.DataFrame(videos).sort_values(by='view', ascending=False)
        save_to_excel(df, self.filename, usecols=usecols)
