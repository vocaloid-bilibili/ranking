# common/data.py
"""数据加载与保存"""

import pandas as pd
from pathlib import Path
from typing import Optional
from common.config import get_app_config, get_paths, ColumnConfig
from common.io import save_excel, load_excel


class DataLoader:
    """数据加载器"""

    def __init__(self, period: str = "daily"):
        self.period = period
        self.config = get_app_config()
        self.paths = get_paths()
        self.column_config = ColumnConfig()

    def get_path(self, key: str, path_type: str = "input_paths", **kwargs) -> Path:
        """获取路径"""
        return self.config.get_period_path(self.period, key, path_type, **kwargs)

    def get_data_source(self, key: str, **kwargs) -> Path:
        """获取数据源路径"""
        sources = self.config.get("data_sources", default={})
        template = sources.get(key, "")
        return Path(template.format(**kwargs))

    def load_toll_data(self, date: str) -> pd.DataFrame:
        path = self.get_data_source("toll_data", date=date)
        cols = self.column_config.get_columns("stat")
        return load_excel(path, usecols=cols)

    def load_new_data(self, date: str) -> pd.DataFrame:
        path = self.get_data_source("new_data", date=date)
        cols = self.column_config.get_columns("new_stat")
        return load_excel(path, usecols=cols)

    def load_merged_data(self, date: str) -> pd.DataFrame:
        toll_data = self.load_toll_data(date)
        new_data = self.load_new_data(date)

        if new_data.empty:
            return toll_data

        return pd.concat([toll_data, new_data]).drop_duplicates(
            subset=["bvid"], keep="first"
        )

    def save(self, df: pd.DataFrame, path: Path, columns_key: Optional[str] = None):
        cols = self.column_config.get_columns(columns_key) if columns_key else None
        save_excel(df, path, usecols=cols)
