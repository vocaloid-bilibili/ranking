# services/json_export.py
"""JSON数据导出服务 - 周刊/月刊/特殊榜单"""

import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import yaml

from common.logger import logger
from common.config import get_paths, get_app_config, Paths


# ==================== JSON编码器 ====================


class NpEncoder(json.JSONEncoder):
    """支持 NumPy 和 Pandas 类型的 JSON 编码器"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
            return str(obj)
        return super().default(obj)


# ==================== 基础导出器 ====================


class BaseExporter(ABC):
    """导出器基类"""

    def __init__(self, paths: Paths = None):
        self.paths = paths or get_paths()
        self.config = get_app_config()
        self.json_output_dir = self.paths.export_json
        self.json_output_dir.mkdir(parents=True, exist_ok=True)

    def read_excel_safe(self, path: Path, sheet_name=0) -> Optional[pd.DataFrame]:
        """安全读取 Excel"""
        if not path.exists():
            logger.warning(f"文件不存在: {path}")
            return None
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except Exception as e:
            logger.error(f"读取Excel失败 [{path}]: {e}")
            return None

    def load_exclude_list(self) -> List[str]:
        """加载排除名单"""
        if not self.paths.exclude_singers.exists():
            return []
        try:
            with open(self.paths.exclude_singers, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"读取排除配置失败: {e}")
            return []

    def get_honor_map(self) -> Dict[str, List[str]]:
        """获取成就映射"""
        honor_map: Dict[str, List[str]] = {}

        # 成就主文件在 achievement 目录下
        master_file = self.paths.achievement / "成就.xlsx"
        if not master_file.exists():
            return honor_map

        try:
            xls = pd.ExcelFile(master_file)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if "name" in df.columns:
                    for name in df["name"].dropna().astype(str):
                        name = name.strip()
                        if name not in honor_map:
                            honor_map[name] = []
                        if sheet_name not in honor_map[name]:
                            honor_map[name].append(sheet_name)
            return honor_map
        except Exception as e:
            logger.error(f"读取成就文件失败: {e}")
            return {}

    def calculate_role_stats(
        self,
        df: pd.DataFrame,
        role_col: str,
        point_col: str,
        exclude_list: List[str],
        limit: int = 10,
        name_col: str = "name",
    ) -> List[Dict]:
        """计算角色统计"""
        stats_map: Dict[str, int] = {}
        firstname_map: Dict[str, tuple] = {}

        if role_col not in df.columns or point_col not in df.columns:
            return []

        for _, row in df.iterrows():
            role_str = str(row[role_col])
            points = row[point_col] if pd.notna(row[point_col]) else 0
            song_name = str(row.get(name_col, "")) if name_col in df.columns else ""

            for name in role_str.split("、"):
                name = name.strip()
                if name and name not in exclude_list:
                    stats_map[name] = stats_map.get(name, 0) + points
                    if name not in firstname_map or points > firstname_map[name][0]:
                        firstname_map[name] = (points, song_name)

        result = [
            {"name": k, "score": v, "firstname": firstname_map.get(k, (0, ""))[1]}
            for k, v in stats_map.items()
        ]
        result.sort(key=lambda x: x["score"], reverse=True)

        return [{**item, "rank": idx + 1} for idx, item in enumerate(result[:limit])]

    def build_image_map(
        self, *dataframes: pd.DataFrame
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        """构建图片映射 (bvid -> image_url, name -> image_url)"""
        bvid_map: Dict[str, str] = {}
        name_map: Dict[str, str] = {}

        for df in dataframes:
            if df is None:
                continue
            if "bvid" in df.columns and "image_url" in df.columns:
                for _, row in df.iterrows():
                    bvid = row.get("bvid")
                    img = row.get("image_url")
                    if bvid and img and pd.notna(img) and bvid not in bvid_map:
                        bvid_map[bvid] = img

        df_catalog = self.read_excel_safe(self.paths.collected)
        if df_catalog is not None:
            if "name" in df_catalog.columns and "image_url" in df_catalog.columns:
                for _, row in df_catalog.iterrows():
                    name = str(row.get("name", "")).strip()
                    img = row.get("image_url")
                    if name and img and pd.notna(img) and name not in name_map:
                        name_map[name] = img

            if "bvid" in df_catalog.columns and "image_url" in df_catalog.columns:
                for _, row in df_catalog.iterrows():
                    bvid = row.get("bvid")
                    img = row.get("image_url")
                    if bvid and img and pd.notna(img) and bvid not in bvid_map:
                        bvid_map[bvid] = img

        return bvid_map, name_map

    def fill_images(
        self, records: List[Dict], bvid_map: Dict[str, str], name_map: Dict[str, str]
    ):
        """填充图片 URL"""
        for item in records:
            if item.get("image_url"):
                continue

            bvid = item.get("bvid")
            if bvid and bvid in bvid_map:
                item["image_url"] = bvid_map[bvid]
                continue

            name = str(item.get("name", "")).strip()
            if name and name in name_map:
                item["image_url"] = name_map[name]

    def save_json(self, data: Dict, filename: str):
        """保存 JSON"""
        output_path = self.json_output_dir / filename
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            logger.info(f"已保存: {output_path}")
        except Exception as e:
            logger.error(f"写入JSON失败: {e}")

    @abstractmethod
    def process(self) -> Dict[str, Any]:
        pass


# ==================== 周刊导出器 ====================


class WeeklyExporter(BaseExporter):
    """周刊JSON导出"""

    def __init__(self, target_date: datetime.date = None, paths: Paths = None):
        super().__init__(paths)
        self.target_date = target_date or self._get_last_saturday()
        self.date_hyphen = self.target_date.strftime("%Y-%m-%d")
        self.date_compact = self.target_date.strftime("%Y%m%d")
        logger.info(f"周刊导出目标日期: {self.date_hyphen}")

    def _get_last_saturday(self) -> datetime.date:
        today = datetime.date.today()
        days_to_subtract = (today.weekday() - 5) % 7
        return today - datetime.timedelta(days=days_to_subtract)

    def _get_issue_index(self) -> int:
        start_date = datetime.datetime.strptime(
            self.config.weekly.start_date, "%Y-%m-%d"
        ).date()
        days_diff = (self.target_date - start_date).days
        return (days_diff // 7) + self.config.weekly.start_index

    def _get_last_week_data(self, data_type: str) -> Optional[pd.DataFrame]:
        last_week = self.target_date - datetime.timedelta(days=7)
        last_week_str = last_week.strftime("%Y-%m-%d")

        if data_type == "total":
            path = self.paths.weekly_main / f"{last_week_str}.xlsx"
        else:
            path = self.paths.weekly_new / f"新曲{last_week_str}.xlsx"

        return self.read_excel_safe(path)

    def _get_last_week_op(self) -> Dict:
        df = self._get_last_week_data("total")
        if df is None or df.empty:
            return {}

        row = df[df["rank"] == 1]
        if row.empty:
            return {}

        rec = row.iloc[0]
        return {
            "title": rec.get("title", ""),
            "bvid": rec.get("bvid", ""),
            "author": rec.get("author", ""),
            "pubdate": rec.get("pubdate", ""),
            "image_url": rec.get("image_url", ""),
        }

    def calculate_stats(self, df_total: pd.DataFrame, df_new: pd.DataFrame) -> Dict:
        stats = {}
        df_top100 = df_total.head(100)

        if "point" in df_total.columns:
            stats["count_over_500k"] = len(df_total[df_total["point"] >= 500000])
            stats["count_over_100k"] = len(df_total[df_total["point"] >= 100000])
            stats["count_over_50k"] = len(df_total[df_total["point"] >= 50000])
        else:
            stats["count_over_500k"] = 0
            stats["count_over_100k"] = 0
            stats["count_over_50k"] = 0

        metrics = ["view", "favorite", "coin", "like", "danmaku", "reply", "share"]
        for metric in metrics:
            if metric in df_top100.columns:
                stats[f"total_{metric}"] = int(
                    pd.to_numeric(df_top100[metric], errors="coerce").sum()
                )
            else:
                stats[f"total_{metric}"] = 0

        if len(df_total) >= 20 and "point" in df_total.columns:
            stats["cutoff_main"] = df_total.iloc[19]["point"]
        else:
            stats["cutoff_main"] = 0

        if len(df_total) >= 100 and "point" in df_total.columns:
            stats["cutoff_sub"] = df_total.iloc[99]["point"]
        else:
            stats["cutoff_sub"] = (
                df_total.iloc[-1]["point"] if not df_total.empty else 0
            )

        if len(df_new) >= 10 and "point" in df_new.columns:
            stats["cutoff_new"] = df_new.iloc[9]["point"]
        else:
            stats["cutoff_new"] = df_new.iloc[-1]["point"] if not df_new.empty else 0

        two_weeks_ago = pd.Timestamp(self.target_date) - pd.Timedelta(days=14)
        if "pubdate" in df_total.columns:
            pubdate = pd.to_datetime(df_total["pubdate"], errors="coerce")
            stats["count_new_main"] = int((pubdate.iloc[:20] >= two_weeks_ago).sum())
            stats["count_new_total"] = int((pubdate.iloc[:100] >= two_weeks_ago).sum())
        else:
            stats["count_new_main"] = 0
            stats["count_new_total"] = 0

        return stats

    def _calculate_daily_trends(
        self, target_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        trends = {name: {str(k): "-" for k in range(1, 8)} for name in target_names}

        for day_idx in range(1, 8):
            offset = 7 - day_idx
            curr_date = self.target_date - datetime.timedelta(days=offset)
            prev_date = curr_date - datetime.timedelta(days=1)

            filename = (
                f"{curr_date.strftime('%Y%m%d')}与{prev_date.strftime('%Y%m%d')}.xlsx"
            )
            file_path = self.paths.daily_ranking_main / filename

            df_daily = self.read_excel_safe(file_path)
            if (
                df_daily is not None
                and "name" in df_daily.columns
                and "rank" in df_daily.columns
            ):
                daily_map = df_daily.groupby("name")["rank"].min().to_dict()
                for name in target_names:
                    if name in daily_map:
                        trends[name][str(day_idx)] = int(daily_map[name])

        return trends

    def _get_last_week_role_ranks(
        self, role_col: str, exclude_list: List[str]
    ) -> Dict[str, int]:
        df = self._get_last_week_data("total")
        if df is None:
            return {}

        stats_map: Dict[str, int] = {}
        if role_col in df.columns and "point" in df.columns:
            for _, row in df.iterrows():
                points = row["point"] if pd.notna(row["point"]) else 0
                for name in str(row[role_col]).split("、"):
                    name = name.strip()
                    if name and name not in exclude_list:
                        stats_map[name] = stats_map.get(name, 0) + points

        sorted_list = sorted(stats_map.items(), key=lambda x: x[1], reverse=True)
        return {name: idx + 1 for idx, (name, _) in enumerate(sorted_list)}

    def process(self) -> Dict[str, Any]:
        issue_index = self._get_issue_index()
        start_date = self.target_date - datetime.timedelta(days=7)

        fmt_start = f"{start_date.year}年{start_date.month}月{start_date.day}日 00:00"
        fmt_end = f"{self.target_date.year}年{self.target_date.month}月{self.target_date.day}日 00:00"

        output = {
            "date": self.date_compact,
            "index": issue_index,
            "period": f"{fmt_start} ~ {fmt_end}",
            "op": self._get_last_week_op(),
            "stat": {},
        }

        path_total = self.paths.weekly_main / f"{self.date_hyphen}.xlsx"
        path_new = self.paths.weekly_new / f"新曲{self.date_hyphen}.xlsx"

        df_total = self.read_excel_safe(path_total)
        df_new = self.read_excel_safe(path_new)

        if df_total is None or df_new is None:
            logger.error("核心榜单文件缺失")
            return output

        logger.info(f"处理第 {issue_index} 期周刊")

        current_stats = self.calculate_stats(df_total, df_new)

        df_last_total = self._get_last_week_data("total")
        df_last_new = self._get_last_week_data("new")

        if df_last_total is not None and df_last_new is not None:
            original_date = self.target_date
            self.target_date = self.target_date - datetime.timedelta(days=7)
            last_stats = self.calculate_stats(df_last_total, df_last_new)
            self.target_date = original_date

            output["stat"] = {
                key: {"value": value, "diff": value - last_stats.get(key, value)}
                for key, value in current_stats.items()
            }
        else:
            output["stat"] = {
                key: {"value": value, "diff": "-"}
                for key, value in current_stats.items()
            }

        top_20_total = df_total.head(20).to_dict(orient="records")
        sub_rank_total = df_total.iloc[20:100].to_dict(orient="records")
        top_10_new = df_new.head(10).to_dict(orient="records")

        if "bvid" in df_total.columns and "rank" in df_total.columns:
            total_rank_map = dict(zip(df_total["bvid"], df_total["rank"]))
            for song in top_10_new:
                song["main_rank"] = total_rank_map.get(song.get("bvid"), "-")

        honor_map = self.get_honor_map()
        all_songs = top_20_total + sub_rank_total + top_10_new
        for song in all_songs:
            song["honor"] = honor_map.get(str(song.get("name", "")).strip(), [])

        target_names = list(
            dict.fromkeys([x["name"] for x in all_songs if x.get("name")])
        )
        trends = self._calculate_daily_trends(target_names)
        for song in all_songs:
            song["daily_trends"] = trends.get(
                song.get("name"), {str(k): "-" for k in range(1, 8)}
            )

        output["total_rank_top20"] = top_20_total
        output["total_rank_sub"] = sub_rank_total
        output["new_rank_top10"] = top_10_new

        exclude_list = self.load_exclude_list()
        output["vocal_stats"] = self.calculate_role_stats(
            df_total, "vocal", "point", exclude_list, 10
        )
        output["producer_stats"] = self.calculate_role_stats(
            df_total, "author", "point", [], 10
        )

        last_vocal_ranks = self._get_last_week_role_ranks("vocal", exclude_list)
        last_producer_ranks = self._get_last_week_role_ranks("author", [])

        for item in output["vocal_stats"]:
            item["last_rank"] = last_vocal_ranks.get(item["name"], "-")
        for item in output["producer_stats"]:
            item["last_rank"] = last_producer_ranks.get(item["name"], "-")

        bvid_map, name_map = self.build_image_map(df_total, df_new)

        extras = {
            "million_record": self.paths.milestone / f"百万记录{self.date_hyphen}.xlsx",
            "history_record": self.paths.history / f"历史{self.date_hyphen}.xlsx",
            "achievement_record": self.paths.achievement
            / f"成就{self.date_hyphen}.xlsx",
        }

        for key, path in extras.items():
            df_ex = self.read_excel_safe(path)
            if df_ex is not None:
                df_ex = df_ex.where(pd.notnull(df_ex), None)
                records = df_ex.to_dict(orient="records")
                self.fill_images(records, bvid_map, name_map)
                output[key] = records
            else:
                output[key] = []

        return output

    def run(self):
        data = self.process()
        self.save_json(data, f"{self.date_hyphen}.json")


# ==================== 月刊导出器 ====================


class MonthlyExporter(BaseExporter):
    """月刊JSON导出"""

    def __init__(self, target_month: str = None, paths: Paths = None):
        super().__init__(paths)
        self.target_month = target_month or self._get_last_month()

        year, month = map(int, self.target_month.split("-"))
        self.target_year = year
        self.target_month_num = month

        self.month_start = datetime.date(year, month, 1)
        if month == 12:
            self.month_end = datetime.date(year + 1, 1, 1)
        else:
            self.month_end = datetime.date(year, month + 1, 1)

        logger.info(f"月刊导出目标月份: {self.target_month}")

    def _get_last_month(self) -> str:
        today = datetime.date.today()
        first = today.replace(day=1)
        last_month = first - datetime.timedelta(days=1)
        return last_month.strftime("%Y-%m")

    def _get_issue_index(self) -> int:
        start_year = self.config.monthly.start_year
        start_month = self.config.monthly.start_month
        months_diff = (self.target_year - start_year) * 12 + (
            self.target_month_num - start_month
        )
        return months_diff + self.config.monthly.start_index

    def _get_last_month_str(self) -> str:
        if self.target_month_num == 1:
            return f"{self.target_year - 1}-12"
        else:
            return f"{self.target_year}-{str(self.target_month_num - 1).zfill(2)}"

    def _get_last_month_data(self, data_type: str) -> Optional[pd.DataFrame]:
        last_month_str = self._get_last_month_str()
        if data_type == "total":
            path = self.paths.monthly_main / f"{last_month_str}.xlsx"
        else:
            path = self.paths.monthly_new / f"新曲{last_month_str}.xlsx"
        return self.read_excel_safe(path)

    def _get_last_month_op(self) -> Dict:
        df = self._get_last_month_data("total")
        if df is None or df.empty:
            return {}

        row = df[df["rank"] == 1]
        if row.empty:
            return {}

        rec = row.iloc[0]
        return {
            "title": rec.get("title", ""),
            "bvid": rec.get("bvid", ""),
            "author": rec.get("author", ""),
            "pubdate": rec.get("pubdate", ""),
            "image_url": rec.get("image_url", ""),
        }

    def calculate_stats(self, df_total: pd.DataFrame, df_new: pd.DataFrame) -> Dict:
        stats = {}
        df_top200 = df_total.head(200)

        if "point" in df_total.columns:
            stats["count_over_1m"] = len(df_total[df_total["point"] >= 1000000])
            stats["count_over_500k"] = len(df_total[df_total["point"] >= 500000])
            stats["count_over_100k"] = len(df_total[df_total["point"] >= 100000])
        else:
            stats["count_over_1m"] = 0
            stats["count_over_500k"] = 0
            stats["count_over_100k"] = 0

        metrics = ["view", "favorite", "coin", "like", "danmaku", "reply", "share"]
        for metric in metrics:
            if metric in df_top200.columns:
                stats[f"total_{metric}"] = int(
                    pd.to_numeric(df_top200[metric], errors="coerce").sum()
                )
            else:
                stats[f"total_{metric}"] = 0

        if len(df_total) >= 20:
            stats["cutoff_main"] = df_total.iloc[19].get("point", 0)
        else:
            stats["cutoff_main"] = 0

        if len(df_total) >= 200:
            stats["cutoff_sub"] = df_total.iloc[199].get("point", 0)
        else:
            stats["cutoff_sub"] = (
                df_total.iloc[-1].get("point", 0) if not df_total.empty else 0
            )

        if len(df_new) >= 20:
            stats["cutoff_new"] = df_new.iloc[19].get("point", 0)
        else:
            stats["cutoff_new"] = (
                df_new.iloc[-1].get("point", 0) if not df_new.empty else 0
            )

        if "pubdate" in df_total.columns:
            pubdate = pd.to_datetime(df_total["pubdate"], errors="coerce")
            month_start = pd.Timestamp(self.month_start)
            month_end = pd.Timestamp(self.month_end)

            main_mask = (pubdate.iloc[:20] >= month_start) & (
                pubdate.iloc[:20] < month_end
            )
            stats["count_new_main"] = int(main_mask.sum())

            top200_mask = (pubdate.iloc[:200] >= month_start) & (
                pubdate.iloc[:200] < month_end
            )
            stats["count_new_total"] = int(top200_mask.sum())
        else:
            stats["count_new_main"] = 0
            stats["count_new_total"] = 0

        return stats

    def _get_recent_saturdays(self, count: int = 5) -> List[datetime.date]:
        saturdays = []
        current = self.month_end - datetime.timedelta(days=1)

        while len(saturdays) < count:
            days_since = (current.weekday() - 5) % 7
            saturday = current - datetime.timedelta(days=days_since)
            if saturday not in saturdays:
                saturdays.append(saturday)
            current = saturday - datetime.timedelta(days=1)

        saturdays.sort()
        return saturdays

    def _calculate_weekly_trends(
        self, target_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        trends = {name: {str(k): "-" for k in range(1, 6)} for name in target_names}
        saturdays = self._get_recent_saturdays(5)

        for week_idx, saturday in enumerate(saturdays, start=1):
            path = self.paths.weekly_main / f"{saturday.strftime('%Y-%m-%d')}.xlsx"
            df = self.read_excel_safe(path)

            if df is not None and "name" in df.columns and "rank" in df.columns:
                weekly_map = df.groupby("name")["rank"].min().to_dict()
                for name in target_names:
                    if name in weekly_map:
                        trends[name][str(week_idx)] = int(weekly_map[name])

        return trends

    def _get_history_record(self, name_map: Dict[str, str]) -> List[Dict]:
        history_year = self.target_year - 1
        history_month_str = f"{history_year}-{str(self.target_month_num).zfill(2)}"

        path = self.paths.monthly_main / f"{history_month_str}.xlsx"
        df = self.read_excel_safe(path)

        if df is None:
            logger.warning(f"历史文件不存在: {path}")
            return []

        top5 = df.head(5).to_dict(orient="records")

        for item in top5:
            if not item.get("image_url") or pd.isna(item.get("image_url")):
                name = str(item.get("name", "")).strip()
                if name in name_map:
                    item["image_url"] = name_map[name]

        return top5

    def _get_last_month_role_ranks(
        self, role_col: str, exclude_list: List[str]
    ) -> Dict[str, int]:
        df = self._get_last_month_data("total")
        if df is None:
            return {}

        stats_map: Dict[str, int] = {}
        if role_col in df.columns and "point" in df.columns:
            for _, row in df.iterrows():
                points = row["point"] if pd.notna(row["point"]) else 0
                for name in str(row[role_col]).split("、"):
                    name = name.strip()
                    if name and name not in exclude_list:
                        stats_map[name] = stats_map.get(name, 0) + points

        sorted_list = sorted(stats_map.items(), key=lambda x: x[1], reverse=True)
        return {name: idx + 1 for idx, (name, _) in enumerate(sorted_list)}

    def process(self) -> Dict[str, Any]:
        issue_index = self._get_issue_index()

        fmt_start = f"{self.month_start.year}年{self.month_start.month}月{self.month_start.day}日 00:00"
        fmt_end = f"{self.month_end.year}年{self.month_end.month}月{self.month_end.day}日 00:00"

        output = {
            "date": self.target_month.replace("-", ""),
            "index": issue_index,
            "period": f"{fmt_start} ~ {fmt_end}",
            "op": self._get_last_month_op(),
            "stat": {},
        }

        path_total = self.paths.monthly_main / f"{self.target_month}.xlsx"
        path_new = self.paths.monthly_new / f"新曲{self.target_month}.xlsx"

        df_total = self.read_excel_safe(path_total)
        df_new = self.read_excel_safe(path_new)

        if df_total is None or df_new is None:
            logger.error("核心榜单文件缺失")
            return output

        logger.info(f"处理第 {issue_index} 期月刊")

        current_stats = self.calculate_stats(df_total, df_new)

        df_last_total = self._get_last_month_data("total")
        df_last_new = self._get_last_month_data("new")

        if df_last_total is not None and df_last_new is not None:
            original = (
                self.target_month,
                self.target_year,
                self.target_month_num,
                self.month_start,
                self.month_end,
            )

            last_str = self._get_last_month_str()
            y, m = map(int, last_str.split("-"))
            self.target_month = last_str
            self.target_year = y
            self.target_month_num = m
            self.month_start = datetime.date(y, m, 1)
            self.month_end = (
                datetime.date(y + 1, 1, 1) if m == 12 else datetime.date(y, m + 1, 1)
            )

            last_stats = self.calculate_stats(df_last_total, df_last_new)

            (
                self.target_month,
                self.target_year,
                self.target_month_num,
                self.month_start,
                self.month_end,
            ) = original

            output["stat"] = {
                key: {"value": value, "diff": value - last_stats.get(key, value)}
                for key, value in current_stats.items()
            }
        else:
            output["stat"] = {
                key: {"value": value, "diff": "-"}
                for key, value in current_stats.items()
            }

        top_20_total = df_total.head(20).to_dict(orient="records")
        sub_rank_total = df_total.iloc[20:200].to_dict(orient="records")
        top_20_new = df_new.head(20).to_dict(orient="records")

        if "name" in df_total.columns and "rank" in df_total.columns:
            total_rank_map = dict(zip(df_total["name"], df_total["rank"]))
            for song in top_20_new:
                song["main_rank"] = total_rank_map.get(song.get("name"), "-")

        honor_map = self.get_honor_map()
        all_songs = top_20_total + sub_rank_total + top_20_new
        for song in all_songs:
            song["honor"] = honor_map.get(str(song.get("name", "")).strip(), [])

        target_names = list(
            dict.fromkeys([x["name"] for x in all_songs if x.get("name")])
        )
        trends = self._calculate_weekly_trends(target_names)
        for song in all_songs:
            song["weekly_trends"] = trends.get(
                song.get("name"), {str(k): "-" for k in range(1, 6)}
            )

        output["total_rank_top20"] = top_20_total
        output["total_rank_sub"] = sub_rank_total
        output["new_rank_top20"] = top_20_new

        exclude_list = self.load_exclude_list()
        output["vocal_stats"] = self.calculate_role_stats(
            df_total, "vocal", "point", exclude_list, 10
        )
        output["producer_stats"] = self.calculate_role_stats(
            df_total, "author", "point", [], 10
        )

        last_vocal_ranks = self._get_last_month_role_ranks("vocal", exclude_list)
        last_producer_ranks = self._get_last_month_role_ranks("author", [])

        for item in output["vocal_stats"]:
            item["last_rank"] = last_vocal_ranks.get(item["name"], "-")
        for item in output["producer_stats"]:
            item["last_rank"] = last_producer_ranks.get(item["name"], "-")

        bvid_map, name_map = self.build_image_map(df_total, df_new)

        million_path = self.paths.milestone / f"百万记录{self.target_month}.xlsx"
        df_million = self.read_excel_safe(million_path)
        if df_million is not None:
            records = df_million.where(pd.notnull(df_million), None).to_dict(
                orient="records"
            )
            self.fill_images(records, bvid_map, name_map)
            output["million_record"] = records
        else:
            output["million_record"] = []

        output["history_record"] = self._get_history_record(name_map)

        return output

    def run(self):
        data = self.process()
        self.save_json(data, f"{self.target_month}.json")


# ==================== 特殊榜单导出器 ====================


class SpecialExporter(BaseExporter):
    """特殊榜单JSON导出"""

    def __init__(self, name: str, ranking_file: Path = None, paths: Paths = None):
        super().__init__(paths)
        self.name = name
        self.ranking_file = ranking_file or (
            self.paths.special_ranking / f"{name}.xlsx"
        )

    def process(self) -> Dict[str, Any]:
        df = self.read_excel_safe(self.ranking_file)

        if df is None:
            logger.error(f"特殊榜单文件不存在: {self.ranking_file}")
            return {"name": self.name, "songs": []}

        logger.info(f"处理特殊榜单: {self.name}")

        songs = df.to_dict(orient="records")

        honor_map = self.get_honor_map()
        for song in songs:
            song["honor"] = honor_map.get(str(song.get("name", "")).strip(), [])

        bvid_map, name_map = self.build_image_map(df)
        self.fill_images(songs, bvid_map, name_map)

        return {"name": self.name, "count": len(songs), "songs": songs}

    def run(self):
        data = self.process()
        self.save_json(data, f"special_{self.name}.json")


# ==================== 便捷函数 ====================


def export_weekly(target_date: datetime.date = None):
    exporter = WeeklyExporter(target_date=target_date)
    exporter.run()


def export_monthly(target_month: str = None):
    exporter = MonthlyExporter(target_month=target_month)
    exporter.run()


def export_special(name: str, ranking_file: Path = None):
    exporter = SpecialExporter(name=name, ranking_file=ranking_file)
    exporter.run()
