# common/dates.py
"""日期工具"""

from datetime import datetime, timedelta

NIGHT_HOUR = 23


def get_today() -> datetime:
    """
    获取业务日期（datetime）

    23:00 以后视为下一天。
    返回当天 00:00:00。
    """
    now = datetime.now()
    if now.hour >= NIGHT_HOUR:
        now += timedelta(days=1)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def get_snapshot_date() -> str:
    """获取快照日期字符串（YYYYMMDD）"""
    return get_today().strftime("%Y%m%d")
