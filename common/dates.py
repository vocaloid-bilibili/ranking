# common/dates.py
"""日期计算工具"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict
from dateutil.relativedelta import relativedelta


@dataclass
class DateRange:
    """日期范围"""

    new_date: str  # 新数据日期 YYYYMMDD
    old_date: str  # 旧数据日期 YYYYMMDD
    target_date: str  # 目标日期（用于命名）
    previous_date: str = ""  # 上期日期


def get_weekly_dates() -> DateRange:
    """计算周刊日期"""
    today = datetime.now()
    new_day = today - timedelta(days=(today.weekday() - 5 + 7) % 7)
    old_day = new_day - timedelta(days=7)
    return DateRange(
        new_date=new_day.strftime("%Y%m%d"),
        old_date=old_day.strftime("%Y%m%d"),
        target_date=new_day.strftime("%Y-%m-%d"),
        previous_date=old_day.strftime("%Y-%m-%d"),
    )


def get_monthly_dates() -> DateRange:
    """计算月刊日期"""
    new_day = datetime.now().replace(day=1)
    new_month = new_day - timedelta(days=1)
    old_day = new_month.replace(day=1)
    old_month = old_day - relativedelta(months=1)
    return DateRange(
        new_date=new_day.strftime("%Y%m%d"),
        old_date=old_day.strftime("%Y%m%d"),
        target_date=new_month.strftime("%Y-%m"),
        previous_date=old_month.strftime("%Y-%m"),
    )


def get_daily_dates() -> Dict[str, str]:
    """计算日刊日期"""
    now_day = (datetime.now() - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return {
        "new_date": (now_day + timedelta(days=1)).strftime("%Y%m%d"),
        "old_date": now_day.strftime("%Y%m%d"),
    }


def get_daily_new_song_dates() -> Dict[str, str]:
    """计算每日新曲榜日期"""
    now_day = (datetime.now() - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return {
        "new_date": (now_day + timedelta(days=1)).strftime("%Y%m%d"),
        "now_date": now_day.strftime("%Y%m%d"),
        "old_date": (now_day - timedelta(days=1)).strftime("%Y%m%d"),
    }


def get_history_dates() -> Dict[str, str]:
    """计算历史回顾日期"""
    today = datetime.now()
    now_day = today - timedelta(days=(today.weekday() - 5 + 7) % 7)
    history_day = now_day - timedelta(weeks=52)
    return {
        "old_date": history_day.strftime("%Y-%m-%d"),
        "target_date": now_day.strftime("%Y-%m-%d"),
    }
