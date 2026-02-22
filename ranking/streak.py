# ranking/streak.py
"""Streak管理模块"""

import pandas as pd
from typing import List
from dataclasses import asdict

from common.models import VideoInfo


class StreakManager:
    """连续低增长计数管理"""

    def __init__(
        self,
        base_threshold: int,
        streak_threshold: int,
        min_total_view: int,
        update_cols: List[str],
    ):
        self.base_threshold = base_threshold
        self.streak_threshold = streak_threshold
        self.min_total_view = min_total_view
        self.update_cols = update_cols

    def _calculate_threshold(self, streak: int, census_mode: bool) -> int:
        if not census_mode:
            return self.base_threshold
        gap = min(7, max(0, streak - self.streak_threshold))
        return self.base_threshold * (gap + 1)

    def _is_failed(
        self, df: pd.DataFrame, update_bvids: set, census_mode: bool
    ) -> pd.Series:
        if census_mode:
            return ~df["bvid"].isin(update_bvids)
        return (df["streak"] < self.streak_threshold) & ~df["bvid"].isin(update_bvids)

    def update_songs(
        self, songs_df: pd.DataFrame, videos: List[VideoInfo], census_mode: bool
    ) -> pd.DataFrame:
        if not videos:
            return songs_df

        update_df = pd.DataFrame([asdict(v) for v in videos])[self.update_cols]
        update_bvids = set(update_df["bvid"])

        songs_df = songs_df.copy()
        old_views = songs_df.set_index("bvid")["view"].to_dict()

        # 标记失效
        songs_df["is_failed"] = self._is_failed(songs_df, update_bvids, census_mode)

        # 更新数据
        songs_df = songs_df.set_index("bvid")
        update_df = update_df.set_index("bvid")
        songs_df.update(update_df)
        songs_df = songs_df.reset_index()

        # 计算 streak
        is_updated = songs_df["bvid"].isin(update_bvids)

        # 未更新的视频（非普查模式）
        if not census_mode:
            mask_unprocessed = ~is_updated & ~songs_df["is_failed"]
            songs_df.loc[mask_unprocessed, "streak"] += 1

        # 已更新的视频
        for bvid in update_bvids:
            if bvid not in songs_df["bvid"].values:
                continue

            idx = songs_df[songs_df["bvid"] == bvid].index[0]
            new_view = songs_df.at[idx, "view"]
            old_view = old_views.get(bvid, new_view)
            incr = new_view - old_view
            streak = songs_df.at[idx, "streak"]
            threshold = self._calculate_threshold(streak, census_mode)

            should_increment = (new_view < self.min_total_view) and (incr < threshold)
            songs_df.at[idx, "streak"] = streak + 1 if should_increment else 0

        # 失效视频streak归0
        songs_df.loc[songs_df["is_failed"], "streak"] = 0

        return songs_df.sort_values(
            ["is_failed", "view"], ascending=[False, False]
        ).drop("is_failed", axis=1)

    def get_songs_to_update(self, df: pd.DataFrame, census_mode: bool) -> pd.DataFrame:
        if census_mode:
            return df
        return df.loc[df["streak"] < self.streak_threshold]
