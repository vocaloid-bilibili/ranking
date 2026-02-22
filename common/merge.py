# common/merge.py
"""数据合并策略"""

from typing import Dict, List, Set, Callable
import pandas as pd

from common.formatters import format_duration
from common.logger import logger


class RecordMerger:
    """单条记录合并策略"""

    @staticmethod
    def new_song() -> Callable[[Dict, Dict], Dict]:
        def merge(api: Dict, local: Dict) -> Dict:
            return {
                **api,
                "duration": format_duration(api.get("duration", 0)),
                "name": api.get("title", ""),
                "author": api.get("uploader", ""),
                "synthesizer": "",
                "vocal": "",
                "type": "",
            }

        return merge

    @staticmethod
    def old_song(local_fields: List[str]) -> Callable[[Dict, Dict], Dict]:
        def merge(api: Dict, local: Dict) -> Dict:
            synced = {k: local.get(k, "") for k in local_fields}
            if not local:
                synced["name"] = api.get("title", "")
                synced["author"] = api.get("uploader", "")
            if synced.get("copyright") not in [100, 101]:
                synced["copyright"] = api.get("copyright")
            return {
                **api,
                **synced,
                "duration": format_duration(api.get("duration", 0)),
            }

        return merge


class DataFrameMerger:
    """DataFrame合并操作"""

    @staticmethod
    def resolve_name_conflicts(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or ref_df.empty:
            return df

        author_map = (
            ref_df.drop_duplicates("name").set_index("name")["author"].to_dict()
        )

        def normalize(s) -> Set[str]:
            return {p.strip() for p in str(s or "").split("、") if p.strip()}

        def resolve(row):
            name, author = row["name"], row["author"]
            if name in author_map and normalize(author) != normalize(author_map[name]):
                logger.info(f"同名冲突: '{name}' -> '{name}({author})'")
                return f"{name}({author})"
            return name

        df = df.copy()
        df["name"] = df.apply(resolve, axis=1)
        return df

    @staticmethod
    def combine_outer(
        df1: pd.DataFrame, df2: pd.DataFrame, on: str = "bvid"
    ) -> pd.DataFrame:
        merged = pd.merge(df1, df2, on=on, how="outer", suffixes=("", "_sec"))
        for col in [c for c in merged.columns if c.endswith("_sec")]:
            base = col[:-4]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[col])
            else:
                merged[base] = merged[col]
            merged.drop(columns=[col], inplace=True)
        return merged
