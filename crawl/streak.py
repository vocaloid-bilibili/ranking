# crawl/streak.py
"""Streak管理模块 — 连续低增长计数"""

from typing import Any, Dict, List, Optional, Set


class Streak:
    """
    streak = 连续多少次增量不达标
    达到阈值后常规模式跳过，只在普查日查询
    """

    def __init__(
        self,
        base_threshold: int,
        streak_threshold: int,
        min_total_view: int,
    ):
        self.base_threshold = base_threshold
        self.streak_threshold = streak_threshold
        self.min_total_view = min_total_view

    # ==================== 公开方法 ====================

    def get_songs_to_update(
        self, songs: List[Dict[str, Any]], census_mode: bool
    ) -> List[Dict[str, Any]]:
        """筛选本次需要请求 API 的视频"""
        if census_mode:
            return songs
        return [r for r in songs if r.get("streak", 0) < self.streak_threshold]

    def update_songs(
        self,
        songs: List[Dict[str, Any]],
        videos: List[Dict[str, Any]],
        old_views: Dict[str, int],
        census_mode: bool,
        failed_bvids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        用 API 返回数据更新 songs，重新计算 streak

        两种情况：
        1. 确认失效（title="已失效视频"）→ streak = 0，排最前
        2. 其他所有 → 更新字段 + 按增量判断 streak，按 view 降序
        """
        if not videos:
            return songs

        failed_bvids = failed_bvids or set()
        video_map: Dict[str, Dict[str, Any]] = {
            v["bvid"]: v for v in videos if v.get("bvid")
        }

        failed: List[Dict[str, Any]] = []
        active: List[Dict[str, Any]] = []

        for row in songs:
            row = dict(row)
            bvid = row.get("bvid", "")

            if bvid in failed_bvids:
                row["streak"] = 0
                failed.append(row)

            elif bvid in video_map:
                row.update(video_map[bvid])
                row["streak"] = self._next_streak(
                    old_view=old_views.get(bvid, 0),
                    new_view=video_map[bvid].get("view", 0),
                    current_streak=row.get("streak", 0),
                    census_mode=census_mode,
                )
                active.append(row)

            else:
                row["streak"] = row.get("streak", 0) + 1
                active.append(row)

        failed.sort(key=lambda r: -old_views.get(r.get("bvid", ""), 0))
        active.sort(key=lambda r: -old_views.get(r.get("bvid", ""), 0))

        return failed + active

    # ==================== 内部方法 ====================

    def _next_streak(
        self,
        old_view: int,
        new_view: int,
        current_streak: int,
        census_mode: bool,
    ) -> int:
        incr = new_view - old_view
        threshold = self._dynamic_threshold(current_streak, census_mode)
        low_growth = new_view < self.min_total_view and incr < threshold
        return current_streak + 1 if low_growth else 0

    def _dynamic_threshold(self, streak: int, census_mode: bool) -> int:
        if not census_mode:
            return self.base_threshold
        gap = min(7, max(0, streak - self.streak_threshold))
        return self.base_threshold * (gap + 1)
