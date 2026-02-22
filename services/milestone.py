# services/milestone.py
"""播放量里程碑检测服务"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional

from common.logger import logger
from common.io import save_excel
from common.config import get_paths


@dataclass
class MilestoneRecord:
    """里程碑记录"""

    title: str
    bvid: str
    name: str
    author: str
    pubdate: datetime
    image_url: str
    milestone: int


@dataclass
class MilestoneResult:
    """检测结果"""

    million_records: List[MilestoneRecord] = field(default_factory=list)
    ten_thousand_records: List[MilestoneRecord] = field(default_factory=list)

    def print_summary(self):
        if self.million_records:
            logger.info("--- 百万达成 ---")
            for r in sorted(self.million_records, key=lambda x: -x.milestone):
                print(f"{r.milestone * 100}万：{r.name}   {r.bvid}")

        if self.ten_thousand_records:
            logger.info("--- 十万达成 ---")
            for r in sorted(self.ten_thousand_records, key=lambda x: -x.milestone):
                print(f"{r.milestone * 10}万：{r.name}   {r.bvid}")


class MilestoneDetector:
    """播放量里程碑检测器"""

    MERGE_COLS = ["bvid", "view", "title", "name", "author", "pubdate", "image_url"]

    def __init__(self, data_dir: Path = None):
        paths = get_paths()
        self.data_dir = data_dir or paths.snapshot_main

    def detect(self, date1: str, date2: str) -> Optional[MilestoneResult]:
        file1 = self.data_dir / f"{date1}.xlsx"
        file2 = self.data_dir / f"{date2}.xlsx"

        try:
            df1 = pd.read_excel(file1)
            df2 = pd.read_excel(file2)
        except FileNotFoundError as e:
            logger.error(f"找不到文件: {e.filename}")
            return None

        df = pd.merge(
            df1[["bvid", "view"]],
            df2[self.MERGE_COLS],
            on="bvid",
            how="right",
            suffixes=("_old", "_new"),
        )
        df["view_old"] = df["view_old"].fillna(0)
        df["pubdate"] = pd.to_datetime(df["pubdate"], format="%Y-%m-%d %H:%M:%S")

        date1_dt = datetime.strptime(date1, "%Y%m%d")
        df["is_new"] = df["pubdate"] > date1_dt

        return self._process_milestones(df)

    def _process_milestones(self, df: pd.DataFrame) -> MilestoneResult:
        result = MilestoneResult()

        for _, row in df.iterrows():
            if not row["is_new"] and row["view_old"] == 0:
                continue

            old_million = 0 if row["is_new"] else int(row["view_old"] // 1_000_000)
            new_million = int(row["view_new"] // 1_000_000)
            old_10w = 0 if row["is_new"] else int(row["view_old"] // 100_000)
            new_10w = int(row["view_new"] // 100_000)

            base_info = {
                "title": row["title"],
                "bvid": row["bvid"],
                "name": row["name"],
                "author": row["author"],
                "pubdate": row["pubdate"],
                "image_url": row["image_url"],
            }

            if new_million > old_million:
                for m in range(old_million + 1, new_million + 1):
                    result.million_records.append(
                        MilestoneRecord(**base_info, milestone=m)
                    )

            if new_10w > old_10w:
                for m in range(old_10w + 1, new_10w + 1):
                    if m <= 9 or m % 10 == 0:
                        result.ten_thousand_records.append(
                            MilestoneRecord(**base_info, milestone=m)
                        )

        return result


def run_milestone_check(
    data_dir: Path = None,
    output_dir: Path = None,
    save_weekly_only: bool = True,
):
    """
    执行里程碑检测
    - 每日：保存十万记录到 milestone/100k/
    - 每周六：保存百万记录到 milestone/
    """
    paths = get_paths()

    data_dir = data_dir or paths.snapshot_main
    output_dir = output_dir or paths.milestone
    output_dir_100k = output_dir / "100k"

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    date2 = today.strftime("%Y%m%d")
    date1 = (today - timedelta(days=1)).strftime("%Y%m%d")
    is_saturday = today.weekday() == 5

    detector = MilestoneDetector(data_dir=data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_100k.mkdir(parents=True, exist_ok=True)

    # 日对比
    logger.info("--- 正在执行日对比 ---")
    result_daily = detector.detect(date1, date2)

    if result_daily:
        result_daily.print_summary()

        # 每日保存十万记录
        if result_daily.ten_thousand_records:
            df_10w = pd.DataFrame(
                [
                    {
                        "title": r.title,
                        "bvid": r.bvid,
                        "name": r.name,
                        "author": r.author,
                        "pubdate": r.pubdate,
                        "image_url": r.image_url,
                        "10w_crossed": r.milestone,
                    }
                    for r in result_daily.ten_thousand_records
                ]
            )
            df_10w = df_10w.sort_values("10w_crossed", ascending=False)

            output_file_10w = output_dir_100k / f"十万记录{date2}与{date1}.xlsx"
            save_excel(df_10w, output_file_10w)
            logger.info(f"已保存十万记录: {output_file_10w}")

    # 周对比（仅周六）
    if is_saturday:
        date1_weekly = (today - timedelta(days=7)).strftime("%Y%m%d")
        logger.info("--- 正在执行周对比 ---")
        result_weekly = detector.detect(date1_weekly, date2)

        if result_weekly:
            result_weekly.print_summary()

            if result_weekly.million_records:
                df_million = pd.DataFrame(
                    [
                        {
                            "title": r.title,
                            "bvid": r.bvid,
                            "name": r.name,
                            "author": r.author,
                            "pubdate": r.pubdate,
                            "image_url": r.image_url,
                            "million_crossed": r.milestone,
                        }
                        for r in result_weekly.million_records
                    ]
                )
                df_million = df_million.sort_values("million_crossed", ascending=False)

                output_file = output_dir / f"百万记录{today:%Y-%m-%d}.xlsx"
                save_excel(df_million, output_file)
                logger.info(f"已保存百万记录: {output_file}")
