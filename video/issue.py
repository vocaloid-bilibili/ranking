# video/issue.py
"""期刊数据管理"""

from pathlib import Path
import re
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import pandas as pd

from common.logger import logger


class Issue:
    """期刊信息管理"""

    def __init__(
        self, ranking_main_dir: Path, ranking_new_dir: Path, first_issue_date: str
    ):
        self.ranking_main_dir = ranking_main_dir
        self.ranking_new_dir = ranking_new_dir
        self.first_issue_date = first_issue_date

    def get_latest_ranking_excel(self) -> Optional[Path]:
        """获取最新的日刊排行榜文件"""
        files = list(self.ranking_main_dir.glob("*.xlsx"))
        if not files:
            logger.warning(f"未找到排行榜文件: {self.ranking_main_dir}")
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    def get_newsong_excel(self, main_excel_path: Path) -> Optional[Path]:
        """根据主榜文件找到对应的新曲榜文件"""
        # 从文件名提取日期：20250601与20250531.xlsx -> 20250601
        m = re.search(r"(20\d{6})", main_excel_path.stem)
        if not m:
            return None

        date_str = m.group(1)
        candidates = [
            p for p in self.ranking_new_dir.glob("*.xlsx") if date_str in p.stem
        ]
        return candidates[0] if candidates else None

    def infer_issue_info(self, excel_path: Path) -> Tuple[str, int, str]:
        """
        推断期刊信息

        Returns:
            (issue_date_str, issue_index, excel_date_str)
        """
        stem = excel_path.stem
        m = re.search(r"(20\d{6})", stem)
        excel_date_str = m.group(1) if m else self.first_issue_date

        excel_dt = datetime.strptime(excel_date_str, "%Y%m%d")
        first_dt = datetime.strptime(self.first_issue_date, "%Y%m%d")

        # 期刊日期是 Excel 日期的前一天
        issue_video_dt = excel_dt - timedelta(days=1)
        issue_date_str = issue_video_dt.strftime("%Y%m%d")

        # 计算期数
        diff = (issue_video_dt - first_dt).days
        issue_index = max(1, diff + 1)

        logger.info(f"期刊日期: {issue_date_str}, 期数: {issue_index}")
        return issue_date_str, issue_index, excel_date_str

    def prepare_video_data(self, top_n: int) -> Tuple[List[pd.Series], str, int, str]:
        """
        准备视频制作数据

        Args:
            top_n: 取前 N 名

        Returns:
            (combined_rows, issue_date, issue_index, excel_date)
        """
        excel_path = self.get_latest_ranking_excel()
        if excel_path is None:
            raise FileNotFoundError("未找到排行榜文件")

        issue_date, idx, ex_date = self.infer_issue_info(excel_path)

        # 读取主榜
        df_total = pd.read_excel(excel_path, dtype={"bvid": str})
        df_top = (
            df_total.sort_values("rank")
            .head(top_n)
            .sort_values("rank", ascending=False)
        )
        top_bvids = set(
            str(r["bvid"]).strip()
            for _, r in df_top.iterrows()
            if pd.notna(r.get("bvid"))
        )

        # 读取新曲榜
        newsong_path = self.get_newsong_excel(excel_path)
        new_rows = []

        if newsong_path and newsong_path.exists():
            df_new = pd.read_excel(newsong_path, dtype={"bvid": str})
            if "rank" in df_new.columns:
                df_new = df_new.sort_values("rank")

            # 取不在主榜中的新曲（最多2首）
            for _, row in df_new.iterrows():
                bvid = str(row.get("bvid", "")).strip()
                if bvid and bvid not in top_bvids:
                    new_rows.append(row)
                    if len(new_rows) >= 2:
                        break

        # 组合数据
        combined = []

        # 先加入新曲（倒序，因为视频是从后往前播）
        for r in reversed(new_rows):
            s = r.copy()
            s["is_new"] = True
            s["rank"] = 999
            combined.append(s)

        # 再加入主榜
        for _, r in df_top.iterrows():
            s = r.copy()
            rank_before = str(r.get("rank_before", "-")).strip()
            s["is_new"] = rank_before == "-"
            combined.append(s)

        return combined, issue_date, idx, ex_date
