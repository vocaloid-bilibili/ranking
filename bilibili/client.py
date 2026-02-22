# bilibili/client.py
"""B站API客户端"""

import asyncio
import subprocess
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set

import aiohttp
from bilibili_api import search

from common.logger import logger
from common.proxy import Proxy
from common.retry import RetryHandler
from common.models import ScraperConfig, SearchOptions, SearchRestrictions
from bilibili.session import SessionManager


class BilibiliClient:
    """B站统一客户端"""

    def __init__(
        self,
        config: ScraperConfig,
        proxy: Optional[Proxy] = None,
        videos_root: Optional[Path] = None,
        ffmpeg_bin: str = "ffmpeg",
    ):
        self.config = config
        self.proxy = proxy
        self.ffmpeg_bin = ffmpeg_bin
        self.videos_root = videos_root

        if self.videos_root:
            self.videos_root.mkdir(parents=True, exist_ok=True)

        self.session_mgr = SessionManager(proxy=proxy)
        self.sem = asyncio.Semaphore(config.SEMAPHORE_LIMIT)
        self.retry = RetryHandler(config.MAX_RETRIES, config.SLEEP_TIME)

    async def close(self):
        await self.session_mgr.close_session()

    # ==================== HTTP ====================

    async def _fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        session = await self.session_mgr.get_session()
        headers = {"User-Agent": random.choice(self.config.HEADERS)}
        proxy_url = self.proxy.proxy_server if self.proxy else None
        async with session.get(
            url,
            headers=headers,
            proxy=proxy_url,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if response.status == 200:
                return await response.json()
            raise Exception(f"HTTP {response.status}")

    # ==================== 搜索 ====================

    def _search_by_type(self, keyword: str, page: int, options: SearchOptions):
        return search.search_by_type(
            keyword,
            search_type=search.SearchObjectType.VIDEO,
            order_type=search.OrderVideo.PUBDATE,
            video_zone_type=options.video_zone_type,
            time_start=options.time_start,
            time_end=options.time_end,
            page=page,
            page_size=options.page_size or 50,
        )

    async def search_aids(
        self,
        keywords: List[str],
        options: SearchOptions,
        restrictions: Optional[SearchRestrictions],
    ) -> List[str]:
        start = self._parse_date(options.time_start) or (
            datetime.now() - timedelta(days=2)
        )
        end = self._parse_date(options.time_end) or datetime.now()
        all_aids: Set[str] = set()

        logger.info(
            f"[分区 {options.video_zone_type}] 搜索: {start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
        )
        aids, hit_kw = await self._search_in_range(
            keywords, options, restrictions, all_aids
        )
        all_aids.update(aids)
        logger.info(f"第一轮: {len(all_aids)}个, {len(hit_kw)}个达上限")

        if hit_kw and (end - start).days > 0:
            logger.info(f"按天拆分 {len(hit_kw)} 个关键词...")
            cur = start
            while cur <= end:
                await self.session_mgr.check_and_restart()
                day_opt = self._make_options_for_range(options, cur, cur)
                day_aids, _ = await self._search_in_range(
                    list(hit_kw), day_opt, restrictions, all_aids
                )
                new_count = len(day_aids - all_aids)
                all_aids.update(day_aids)
                logger.info(f"  {cur:%m-%d}: +{new_count}个")
                cur += timedelta(days=1)

        logger.info(f"===== 完成: {len(all_aids)}个 =====")
        return list(all_aids)

    async def _search_in_range(
        self,
        keywords: List[str],
        options: SearchOptions,
        restrictions: Optional[SearchRestrictions],
        collected_aids: Set[str],
    ) -> tuple[Set[str], Set[str]]:
        MAX_PAGE = 20
        aids: Set[str] = set()
        hit_limit_keywords: Set[str] = set()
        batch_size = 3
        keyword_pages = {kw: 1 for kw in keywords}
        active_keywords = keywords[:]

        while active_keywords:
            await self.session_mgr.check_and_restart()
            current_batch = active_keywords[:batch_size]
            logger.info(f"[分区 {options.video_zone_type}] 批次: {current_batch}")

            async def sem_fetch(keyword: str) -> Dict[str, Any]:
                async with self.sem:
                    page = keyword_pages[keyword]
                    result = await self.retry.retry_async(
                        self._search_by_type, keyword, page, options
                    )
                    return self._parse_search_result(
                        result,
                        keyword,
                        page,
                        options.page_size or 50,
                        MAX_PAGE,
                        restrictions,
                        collected_aids,
                        aids,
                    )

            results = await asyncio.gather(*[sem_fetch(kw) for kw in current_batch])

            batch_zero_count = 0
            for r in results:
                kw = r["keyword"]
                total = r["total"]
                found = len(r["aids"])
                aids.update(r["aids"])

                if total == 0:
                    logger.warning(f"    '{kw}' 第{keyword_pages[kw]}页: 0条")
                    batch_zero_count += 1
                else:
                    logger.info(
                        f"    '{kw}' 第{keyword_pages[kw]}页: {total}条, 新增{found}"
                    )

                if r["hit_limit"]:
                    hit_limit_keywords.add(kw)
                    logger.warning(f"    '{kw}' 达到{MAX_PAGE}页上限")

                if r["end"]:
                    if kw in active_keywords:
                        active_keywords.remove(kw)
                else:
                    keyword_pages[kw] += 1

            if batch_zero_count == len(current_batch):
                if self.session_mgr.record_zero_batch():
                    await self.session_mgr.check_and_restart(force=True)
            else:
                self.session_mgr.reset_zero_count()

            remaining = [k for k in active_keywords if k not in current_batch]
            active_keywords = remaining + [
                k for k in current_batch if k in active_keywords
            ]
            await asyncio.sleep(1.5)

        return aids, hit_limit_keywords

    def _parse_search_result(
        self,
        result,
        keyword,
        page,
        page_size,
        max_page,
        restrictions,
        collected_aids,
        current_aids,
    ) -> Dict[str, Any]:
        if not result or "result" not in result:
            return {
                "end": True,
                "keyword": keyword,
                "aids": [],
                "total": 0,
                "hit_limit": False,
            }

        videos = result.get("result", [])
        total = len(videos)
        end = not videos or total < page_size
        hit_limit = page >= max_page and not end
        if hit_limit:
            end = True

        temp_aids = []
        for item in videos:
            if restrictions:
                if (
                    restrictions.min_favorite
                    and item["favorites"] < restrictions.min_favorite
                ):
                    return {
                        "end": True,
                        "keyword": keyword,
                        "aids": temp_aids,
                        "total": total,
                        "hit_limit": False,
                    }
                if restrictions.min_view and item["play"] < restrictions.min_view:
                    return {
                        "end": True,
                        "keyword": keyword,
                        "aids": temp_aids,
                        "total": total,
                        "hit_limit": False,
                    }
            aid = str(item["aid"])
            if aid not in collected_aids and aid not in current_aids:
                temp_aids.append(aid)

        return {
            "end": end,
            "keyword": keyword,
            "aids": temp_aids,
            "total": total,
            "hit_limit": hit_limit,
        }

    # ==================== 其他API ====================

    async def get_newlist_aids(
        self, rid: int, ps: int, start_time: datetime
    ) -> List[str]:
        aids: Set[str] = set()
        page = 1
        try:
            while True:
                await self.session_mgr.check_and_restart()
                url = f"https://api.bilibili.com/x/web-interface/newlist?rid={rid}&ps={ps}&pn={page}"
                data = await self.retry.retry_async(self._fetch_json, url)
                if data and data.get("data"):
                    videos = data["data"]["archives"]
                    recent = [
                        v
                        for v in videos
                        if datetime.fromtimestamp(v["pubdate"]) > start_time
                    ]
                    logger.info(
                        f"newlist {rid} 第{page}页: 返回{len(videos)}, 符合{len(recent)}"
                    )
                    if not recent:
                        break
                    aids.update(str(v["aid"]) for v in recent)
                    page += 1
                    await asyncio.sleep(self.config.SLEEP_TIME)
                else:
                    break
            return list(aids)
        except Exception as e:
            logger.error(f"newlist出错: {e}")
            return list(aids)

    async def get_batch_details(self, aids: List[int]) -> Dict[int, Dict[str, Any]]:
        BATCH = 50
        all_stats = {}
        for i in range(0, len(aids), BATCH):
            await self.session_mgr.check_and_restart()
            batch = aids[i : i + BATCH]
            url = f"https://api.bilibili.com/medialist/gateway/base/resource/infos?resources={','.join(f'{a}:2' for a in batch)}"
            logger.info(f"medialist 批次 {i // BATCH + 1}...")
            try:
                data = await self.retry.retry_async(self._fetch_json, url)
                if data and data.get("code") == 0 and data.get("data"):
                    for item in data["data"]:
                        all_stats[item["id"]] = item
                await asyncio.sleep(self.config.SLEEP_TIME)
            except Exception as e:
                logger.error(f"批次异常: {e}")
        return all_stats

    async def get_newlist_rank_videos(
        self, cate_id: int, time_from: str, time_to: str
    ) -> List[Dict[str, Any]]:
        all_videos = []
        page = 1
        while True:
            await self.session_mgr.check_and_restart()
            url = (
                f"https://api.bilibili.com/x/web-interface/newlist_rank?"
                f"main_ver=v3&search_type=video&view_type=hot_rank&copy_right=-1&order=click"
                f"&cate_id={cate_id}&page={page}&pagesize=50&time_from={time_from}&time_to={time_to}"
            )
            logger.info(f"{time_from}~{time_to}, 第{page}页...")
            try:
                data = await self.retry.retry_async(self._fetch_json, url)
            except Exception as e:
                logger.error(f"hot_rank 请求失败: {e}")
                break
            if not data or data.get("code") != 0:
                break
            videos = data.get("data", {}).get("result")
            if not videos:
                break
            all_videos.extend(videos)
            page += 1
            await asyncio.sleep(self.config.SLEEP_TIME)
        return all_videos

    # ==================== 下载 ====================

    def download_video(self, bvid: str) -> Optional[Path]:
        if not self.videos_root:
            return None
        import yt_dlp

        bvid_dir = self.videos_root / bvid
        bvid_dir.mkdir(exist_ok=True)
        cached = bvid_dir / f"{bvid}.mp4"
        if cached.exists():
            return cached
        try:
            with yt_dlp.YoutubeDL(
                {
                    "format": "bv*+ba/best",
                    "outtmpl": str(bvid_dir / f"{bvid}.%(ext)s"),
                    "quiet": True,
                }
            ) as ydl:
                ydl.extract_info(
                    f"https://www.bilibili.com/video/{bvid}", download=True
                )
            for p in bvid_dir.glob(f"{bvid}.*"):
                if p.suffix.lower() == ".mp4" and p != cached:
                    p.rename(cached)
            return cached if cached.exists() else None
        except Exception as e:
            logger.error(f"[{bvid}] 下载失败: {e}")
            return None

    def extract_audio(self, bvid: str, video_path: Path) -> Optional[Path]:
        audio = video_path.parent / f"{bvid}.wav"
        if audio.exists():
            return audio
        try:
            subprocess.run(
                [
                    self.ffmpeg_bin,
                    "-y",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "22050",
                    str(audio),
                    "-loglevel",
                    "error",
                ],
                check=True,
            )
            return audio
        except:
            return None

    # ==================== 工具 ====================

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except:
                continue
        return None

    def _make_options_for_range(
        self, base: SearchOptions, start: datetime, end: datetime
    ) -> SearchOptions:
        return SearchOptions(
            video_zone_type=base.video_zone_type,
            time_start=start.strftime("%Y-%m-%d"),
            time_end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            page_size=base.page_size,
        )
