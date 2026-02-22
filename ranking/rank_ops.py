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

    df["rank_before"] = df["name"].map(lambda x: prev_dict.get(x, {}).get("rank", "-"))
    df["point_before"] = df["name"].map(
        lambda x: prev_dict.get(x, {}).get("point", "-")
    )

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

    df["count"] = df["name"].map(lambda x: prev_count.get(x, 0)) + (
        df["rank"] <= top_n
    ).astype(int)
    return df


def keep_highest_score(
    df: pd.DataFrame, by: str = "name", score: str = "point"
) -> pd.DataFrame:
    """同名去重，保留最高分"""
    if df.empty:
        return df
    return df.loc[df.groupby(by)[score].idxmax()].reset_index(drop=True)
