# ranking/rank_ops.py
"""DataFrame级别的排名操作"""

import pandas as pd
from pathlib import Path


STAT_COLS = ["view", "favorite", "coin", "like", "danmaku", "reply", "share"]
RATE_COLS = ["viewR", "favoriteR", "coinR", "likeR", "danmakuR", "replyR", "shareR"]
FIX_COLS = ["fixA", "fixB", "fixC", "fixD"]


def format_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """格式化评分系数列为2位小数字符串"""
    df = df.copy()
    for col in RATE_COLS + FIX_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return df


def calculate_ranks(df: pd.DataFrame, point_col: str = "point") -> pd.DataFrame:
    """计算各项指标排名"""
    df = df.sort_values(point_col, ascending=False).copy()

    for col in STAT_COLS:
        if col in df.columns:
            df[f"{col}_rank"] = df[col].rank(ascending=False, method="min")

    df["rank"] = df[point_col].rank(ascending=False, method="min")
    return format_rate_columns(df)


def _build_bvid_name_bridge(
    df: pd.DataFrame,
    prev_df: pd.DataFrame,
) -> dict:
    if "bvid" not in df.columns or "bvid" not in prev_df.columns:
        return {}

    prev_names = set(prev_df["name"].unique()) if "name" in prev_df.columns else set()
    current_names = set(df["name"].unique()) if "name" in df.columns else set()

    missing_names = current_names - prev_names
    if not missing_names:
        return {}

    prev_bvid_to_name = prev_df.set_index("bvid")["name"].to_dict()

    bridge = {}
    for name in missing_names:
        bvids = df.loc[df["name"] == name, "bvid"].tolist()

        candidates = set()
        for bvid in bvids:
            old_name = prev_bvid_to_name.get(bvid)
            if old_name and old_name in prev_names:
                candidates.add(old_name)

        if len(candidates) == 1:
            bridge[name] = candidates.pop()
        elif len(candidates) > 1:
            best_name = None
            best_rank = float("inf")
            for cn in candidates:
                rank_rows = prev_df.loc[prev_df["name"] == cn, "rank"]
                if not rank_rows.empty:
                    r = rank_rows.iloc[0]
                    if r < best_rank:
                        best_rank = r
                        best_name = cn
            if best_name:
                bridge[name] = best_name

    return bridge


def update_rank_change(df: pd.DataFrame, prev_path: Path) -> pd.DataFrame:
    """更新排名变化和增长率"""
    df = df.copy()

    if not prev_path.exists():
        df["rank_before"] = "-"
        df["point_before"] = "-"
        df["rate"] = "NEW"
        return df

    df_prev = pd.read_excel(prev_path)
    prev_dict = df_prev.set_index("name")[["rank", "point"]].to_dict(orient="index")

    bridge = _build_bvid_name_bridge(df, df_prev)

    def _lookup(name, field):
        if name in prev_dict:
            return prev_dict[name].get(field, "-")
        bridged = bridge.get(name)
        if bridged and bridged in prev_dict:
            return prev_dict[bridged].get(field, "-")
        return "-"

    df["rank_before"] = df["name"].map(lambda x: _lookup(x, "rank"))
    df["point_before"] = df["name"].map(lambda x: _lookup(x, "point"))

    def calc_rate(row):
        if row["point_before"] == "-":
            return "NEW"
        if row["point_before"] == 0:
            return "inf"
        return f"{(row['point'] - row['point_before']) / row['point_before']:.2%}"

    df["rate"] = df.apply(calc_rate, axis=1)
    return df.sort_values("point", ascending=False)


def update_board_count(
    df: pd.DataFrame, prev_path: Path, top_n: int = 20
) -> pd.DataFrame:
    """更新在榜次数"""
    df = df.copy()

    if not prev_path.exists():
        df["count"] = (df["rank"] <= top_n).astype(int)
        return df

    df_prev = pd.read_excel(prev_path)
    prev_count = df_prev.set_index("name")["count"].to_dict()

    bridge = _build_bvid_name_bridge(df, df_prev)

    def _lookup_count(name):
        if name in prev_count:
            return prev_count[name]
        bridged = bridge.get(name)
        if bridged and bridged in prev_count:
            return prev_count[bridged]
        return 0

    df["count"] = df["name"].map(_lookup_count) + (df["rank"] <= top_n).astype(int)
    return df


def keep_highest_score(
    df: pd.DataFrame, by: str = "name", score: str = "point"
) -> pd.DataFrame:
    """同名去重，保留最高分"""
    if df.empty:
        return df
    return df.loc[df.groupby(by)[score].idxmax()].reset_index(drop=True)
