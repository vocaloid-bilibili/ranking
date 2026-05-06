# ranking/processor.py
"""榜单生成处理器"""

import asyncio
import inspect
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
from common.config import ColumnConfig, get_app_config, get_paths
from common.data import DataLoader
from common.dates import (
    get_cover_weekly_dates,
    get_daily_dates,
    get_daily_new_song_dates,
    get_history_dates,
    get_monthly_dates,
    get_weekly_dates,
)
from common.io import load_excel
from common.merge import DataFrameMerger
from ranking.rank_ops import (
    _build_bvid_name_bridge,
    calculate_ranks,
    keep_highest_score,
    update_board_count,
    update_rank_change,
)
from ranking.record import process_records


class RankingProcessor:
    """负责生成和处理不同类型的排行榜数据"""

    def __init__(self, period: str):
        self.period = period
        self.config = get_app_config()
        self.paths = get_paths()
        self.column_config = ColumnConfig()
        self.data_loader = DataLoader(period)

        self._dispatch_map = {
            "weekly": self.run_periodic_ranking,
            "monthly": self.run_periodic_ranking,
            "annual": self.run_periodic_ranking,
            "daily": self.run_daily_diff_async,
            "daily_combination": self.run_combination,
            "daily_new_song": self.run_daily_new_song,
            "special": self.run_special,
            "history": self.run_history,
            "cover_weekly": self.run_periodic_ranking,
        }

        self._date_methods = {
            "weekly": get_weekly_dates,
            "monthly": get_monthly_dates,
            "daily": get_daily_dates,
            "daily_combination": get_daily_new_song_dates,
            "daily_new_song": get_daily_new_song_dates,
            "history": get_history_dates,
            "cover_weekly": get_cover_weekly_dates,
        }

    def _get_period_config(self, key: str, default=None):
        """获取当前周期的配置项"""
        return self.config.get_period_option(self.period, key, default)

    def _get_path(self, key: str, path_type: str = "input_paths", **kwargs) -> Path:
        """获取路径"""
        return self.config.get_period_path(self.period, key, path_type, **kwargs)

    def get_dates(self) -> Dict[str, str]:
        method = self._date_methods.get(self.period)
        if not method:
            return {}
        result = method()
        if hasattr(result, "__dataclass_fields__"):
            return {k: getattr(result, k) for k in result.__dataclass_fields__}
        return result

    def _unify_metadata(
        self, df: pd.DataFrame, collected: pd.DataFrame
    ) -> pd.DataFrame:
        meta_cols = self.column_config.get_columns("meta")
        df = df.copy()
        source = collected[
            ["bvid"] + [c for c in meta_cols if c in collected.columns]
        ].drop_duplicates("bvid")

        cols_to_drop = [c for c in meta_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        df = pd.merge(df, source, on="bvid", how="left")
        return df

    def _apply_data_filter(
        self,
        old_data: pd.DataFrame,
        new_data: pd.DataFrame,
        filter_config: dict,
        collected_data: pd.DataFrame = None,
    ) -> tuple:
        """
        根据配置中的 filter 字段过滤数据。
        先用 collected 补全元数据，再按条件筛选。
        """
        if collected_data is not None and not collected_data.empty:
            for field_name in filter_config:
                if field_name not in collected_data.columns:
                    continue
                field_map = (
                    collected_data.dropna(subset=[field_name])
                    .drop_duplicates(subset=["bvid"])
                    .set_index("bvid")[field_name]
                    .to_dict()
                )
                for df in [old_data, new_data]:
                    if field_name in df.columns and "bvid" in df.columns:
                        mapped = df["bvid"].map(field_map)
                        df[field_name] = mapped.fillna(df[field_name])

        for field_name, value in filter_config.items():
            if field_name in new_data.columns:
                new_data = new_data[new_data[field_name] == value].copy()
            if field_name in old_data.columns:
                old_data = old_data[old_data[field_name] == value].copy()

        return old_data, new_data

    async def run(self, **kwargs):
        handler = self._dispatch_map.get(self.period)
        if not handler:
            raise ValueError(f"未知的任务类型: {self.period}")

        self._validate_kwargs(self.period, kwargs)

        if inspect.iscoroutinefunction(handler):
            await handler(**kwargs)
        else:
            handler(**kwargs)

    def _validate_kwargs(self, period: str, kwargs: dict):
        required = {
            "weekly": "dates",
            "monthly": "dates",
            "annual": "dates",
            "history": "dates",
            "special": "song_data",
            "cover_weekly": "dates",
        }
        if period in required and required[period] not in kwargs:
            raise ValueError(f"'{period}' 模式需要 '{required[period]}' 参数")

    # ==================== 期刊 ====================

    def run_periodic_ranking(self, dates: dict):
        old_data = self.data_loader.load_merged_data(date=dates["old_date"])
        new_data = self.data_loader.load_toll_data(date=dates["new_date"])

        collected_data = None
        if self.paths.collected.exists():
            collected_data = load_excel(self.paths.collected)

        data_filter = self._get_period_config("filter")
        if data_filter:
            old_data, new_data = self._apply_data_filter(
                old_data, new_data, data_filter, collected_data
            )
        df = process_records(
            new_data=new_data,
            old_data=old_data,
            use_old_data=True,
            old_time_toll=dates["old_date"],
            collected_data=collected_data,
            ranking_type=self._get_period_config("ranking_type"),
        )

        toll_ranking = keep_highest_score(df)
        toll_ranking = calculate_ranks(toll_ranking)
        toll_ranking = self._apply_update_options(toll_ranking, dates)

        toll_ranking_path = self._get_path(
            "toll_ranking", "output_paths", target_date=dates["target_date"]
        )
        self.data_loader.save(toll_ranking, toll_ranking_path, "final_ranking")

        if self._get_period_config("has_new_ranking"):
            self._generate_new_ranking(toll_ranking, dates)

    def _apply_update_options(self, df: pd.DataFrame, dates: dict) -> pd.DataFrame:
        update_opts = self._get_period_config("update_options", {})
        if not update_opts or not any(update_opts.values()):
            return df

        prev_path = self._get_path(
            "toll_ranking", "output_paths", target_date=dates["previous_date"]
        )

        if update_opts.get("count"):
            df = update_board_count(df, prev_path)
        if update_opts.get("rank_and_rate"):
            df = update_rank_change(df, prev_path)

        return df

    def _generate_new_ranking(self, toll_ranking: pd.DataFrame, dates: dict):
        start_date = datetime.strptime(dates["old_date"], "%Y%m%d")
        end_date = datetime.strptime(dates["new_date"], "%Y%m%d")

        on_board_names = self._get_on_board_names(toll_ranking, self.period)

        if self.period == "weekly":
            start_date = start_date - timedelta(days=7)

        mask = (
            (pd.to_datetime(toll_ranking["pubdate"]) >= start_date)
            & (pd.to_datetime(toll_ranking["pubdate"]) < end_date)
            & (~toll_ranking["name"].isin(on_board_names))
        )
        new_ranking = toll_ranking[mask].copy()

        if not new_ranking.empty:
            new_ranking = calculate_ranks(new_ranking)
            output_path = self._get_path(
                "new_ranking", "output_paths", target_date=dates["target_date"]
            )
            self.data_loader.save(new_ranking, output_path, "final_ranking")

    def _get_on_board_names(self, df: pd.DataFrame, period: str) -> set:
        if period == "weekly" and "count" in df.columns:
            return set(df[df["count"] > 0]["name"])
        if period == "monthly" and "rank" in df.columns:
            return set(df[df["rank"] <= 20]["name"])
        return set()

    # ==================== 日刊合并 ====================

    def run_combination(self):
        dates = self.get_dates()

        main_path = self._get_path("main_diff", "input_paths", **dates)
        new_song_path = self._get_path("new_song_diff", "input_paths", **dates)

        df_main = load_excel(main_path)
        df_new_song = load_excel(new_song_path)
        existing_collected_df = load_excel(self.paths.collected)

        # Step 1: new diff 先做同名冲突检测（在写入collected之前）
        df_new_song = DataFrameMerger.resolve_name_conflicts(
            df_new_song, existing_collected_df
        )

        # Step 2: new diff 覆盖/追加到 collected
        updated_collected = self._update_collected_songs(
            df_new_song, df_main, existing_collected_df
        )

        # Step 3: 合并 diff 用于得分
        raw_combined_df = DataFrameMerger.combine_outer(df_new_song, df_main, on="bvid")

        # Step 4: 用 updated collected 统一所有元数据
        raw_combined_df = self._unify_metadata(raw_combined_df, updated_collected)

        # Step 5: 生成排名
        self._process_and_save_combined_ranking(raw_combined_df, dates)

        # Step 6: 更新快照
        self._update_master_data_for_next_day(dates, updated_collected)

    def _update_collected_songs(
        self,
        df_new_song: pd.DataFrame,
        df_main: pd.DataFrame,
        existing_df: pd.DataFrame,
    ) -> pd.DataFrame:
        metadata_cols = self.column_config.get_columns("metadata_update_cols")
        existing_df = existing_df.copy()

        if not df_new_song.empty:
            new_updates = df_new_song[
                ["bvid"] + [c for c in metadata_cols if c in df_new_song.columns]
            ].drop_duplicates(subset=["bvid"], keep="last")

            existing_df = existing_df.set_index("bvid")
            existing_df.update(new_updates.set_index("bvid"))
            existing_df = existing_df.reset_index()

        new_bvids = df_new_song[~df_new_song["bvid"].isin(existing_df["bvid"])][
            "bvid"
        ].unique()

        if len(new_bvids) > 0:
            record_cols = self.column_config.get_columns("record")
            new_songs = df_new_song[df_new_song["bvid"].isin(new_bvids)].copy()
            new_songs = new_songs.drop_duplicates(subset=["bvid"], keep="last")
            new_songs["streak"] = 0
            new_songs = new_songs[[c for c in record_cols if c in new_songs.columns]]
            existing_df = pd.concat([existing_df, new_songs], ignore_index=True)

        output_path = self._get_path("collected_songs", "output_paths")
        self.data_loader.save(existing_df, output_path, "record")
        return existing_df

    def _process_and_save_combined_ranking(self, df: pd.DataFrame, dates: dict):
        merged_df = keep_highest_score(df)
        ranked_df = calculate_ranks(merged_df)

        old_path = self._get_path("previous_combined", "input_paths", **dates)
        processed_df = update_board_count(ranked_df, old_path)
        processed_df = update_rank_change(processed_df, old_path)

        output_path = self._get_path("combined_ranking", "output_paths", **dates)
        self.data_loader.save(processed_df, output_path, "final_ranking")

    def _update_master_data_for_next_day(self, dates: dict, df_collected: pd.DataFrame):
        main_path = self._get_path("main_data", "input_paths", **dates)
        new_song_path = self._get_path("new_song_data", "input_paths", **dates)

        df_main = load_excel(main_path)
        df_new_song = load_excel(new_song_path)

        promotable = pd.merge(
            df_new_song, df_collected[["bvid"]], on="bvid", how="inner"
        )
        stat_cols = self.column_config.get_columns("stat")
        df_promoted = promotable[
            [c for c in stat_cols if c in promotable.columns]
        ].copy()

        updated_main = pd.concat([df_main, df_promoted], ignore_index=True)
        updated_main = updated_main.drop_duplicates(
            subset=["bvid"], keep="last"
        ).reset_index(drop=True)

        update_cols = self.column_config.get_columns("metadata_update_cols")
        update_source = df_collected[
            ["bvid"] + [c for c in update_cols if c in df_collected.columns]
        ].copy()

        cols_to_drop = [c for c in update_cols if c in updated_main.columns]
        base_df = updated_main.drop(columns=cols_to_drop)
        final_df = pd.merge(base_df, update_source, on="bvid", how="left")

        output_path = self._get_path("main_data", "output_paths", **dates)
        self.data_loader.save(final_df, output_path, "stat")

    # ==================== 日刊新曲 ====================

    def run_daily_new_song(self):
        dates = self.get_dates()

        diff_path = self._get_path("diff_file", "input_paths", **dates)
        prev_rank_path = self._get_path("previous_ranking", "input_paths", **dates)

        new_ranking = load_excel(diff_path)
        prev_cols = ["name", "rank"]
        prev_full = load_excel(prev_rank_path)
        if "bvid" in prev_full.columns:
            prev_cols.append("bvid")
        prev_ranking = prev_full[prev_cols]

        new_ranking = keep_highest_score(new_ranking)
        new_ranking = self._filter_rising_songs(new_ranking, prev_ranking)

        if not new_ranking.empty:
            new_ranking = calculate_ranks(new_ranking)

        output_path = self._get_path("ranking", "output_paths", **dates)
        self.data_loader.save(new_ranking, output_path, "new_ranking")

    def _filter_rising_songs(
        self, df: pd.DataFrame, prev_df: pd.DataFrame
    ) -> pd.DataFrame:
        df = df.sort_values(by="point", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        merged = df.merge(
            prev_df[["name", "rank"]], on="name", how="left", suffixes=("", "_prev")
        )
        merged["rank_prev"] = merged["rank_prev"].fillna(1000)

        if "bvid" in df.columns and "bvid" in prev_df.columns:
            bridge = _build_bvid_name_bridge(df, prev_df)
            if bridge:
                prev_rank_dict = prev_df.set_index("name")["rank"].to_dict()
                for idx, row in merged.iterrows():
                    if row["rank_prev"] == 1000:
                        bridged = bridge.get(row["name"])
                        if bridged and bridged in prev_rank_dict:
                            merged.at[idx, "rank_prev"] = prev_rank_dict[bridged]

        rising = merged[merged["rank"] < merged["rank_prev"]].copy()
        return rising.drop(columns=["rank_prev"], errors="ignore")

    # ==================== 日刊差异 ====================

    async def run_daily_diff_async(self, **kwargs):
        results = await asyncio.gather(
            self._process_diff_task_async("main"),
            self._process_diff_task_async("new_song"),
        )
        return results

    async def _process_diff_task_async(self, task_type: str) -> pd.DataFrame:
        dates = get_daily_dates()
        paths_info, collected_path, threshold = self._get_diff_task_config(
            task_type, dates
        )

        collected_data = None
        if collected_path is not None:
            collected_data = await asyncio.to_thread(load_excel, collected_path)

        old_data, new_data = await asyncio.gather(
            asyncio.to_thread(load_excel, paths_info["old"]),
            asyncio.to_thread(load_excel, paths_info["new"]),
        )
        df = process_records(
            new_data=new_data,
            old_data=old_data,
            use_old_data=True,
            collected_data=collected_data,
            ranking_type="daily",
            old_time_toll=dates["old_date"],
            allow_missing_old=(task_type == "new_song"),
        )

        if threshold:
            df = df[df["point"] >= threshold]
        df = df.sort_values("point", ascending=False)

        usecols_key = "new" if task_type == "new_song" else "old"

        await asyncio.to_thread(
            self.data_loader.save, df, paths_info["output"], usecols_key
        )
        return df

    def _get_diff_task_config(self, task_type: str, dates: dict) -> tuple:
        if task_type == "main":
            return (
                {
                    "old": self._get_path(
                        "main_data", "input_paths", date=dates["old_date"]
                    ),
                    "new": self._get_path(
                        "main_data", "input_paths", date=dates["new_date"]
                    ),
                    "output": self._get_path("main_diff", "output_paths", **dates),
                },
                self._get_path("collected_songs", "input_paths"),
                None,
            )
        else:
            return (
                {
                    "old": self._get_path(
                        "new_song_data", "input_paths", date=dates["old_date"]
                    ),
                    "new": self._get_path(
                        "new_song_data", "input_paths", date=dates["new_date"]
                    ),
                    "output": self._get_path("new_song_diff", "output_paths", **dates),
                },
                self._get_path("collected_songs", "input_paths"),
                self._get_period_config("threshold"),
            )

    # ==================== 特刊/历史 ====================

    def run_special(self, song_data: str):
        input_path = self._get_path("input_path", "paths", song_data=song_data)
        output_path = self._get_path("output_path", "paths", song_data=song_data)

        df = load_excel(input_path)
        processing_opts = self._get_period_config("processing_options", {})

        collected_data = None
        if "collected_data" in processing_opts:
            collected_data = load_excel(processing_opts["collected_data"])

        df = process_records(
            new_data=df,
            ranking_type=self._get_period_config("ranking_type", "special"),
            use_old_data=processing_opts.get("use_old_data"),
            collected_data=collected_data,
        )

        df = keep_highest_score(df)
        df = calculate_ranks(df)
        self.data_loader.save(df, output_path)

    def run_history(self, dates: dict):
        input_path = self._get_path("weekly_ranking", "input_paths", **dates)
        df = load_excel(input_path)

        history_cols = self.column_config.get_columns("history")
        df = df[df["rank"] <= 5][history_cols].copy()

        output_path = self._get_path("history_file", "output_paths", **dates)
        self.data_loader.save(df, output_path, "history")
