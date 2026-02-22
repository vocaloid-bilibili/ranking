# common/models.py
"""数据模型定义"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from enum import Enum, auto


class RankingType(Enum):
    """榜单类型"""

    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    ANNUAL = auto()
    SPECIAL = auto()


@dataclass
class VideoInfo:
    """视频信息"""

    title: str
    bvid: str
    aid: str
    name: str
    author: str
    uploader: str = ""
    copyright: int = 0
    synthesizer: str = ""
    vocal: str = ""
    type: str = ""
    pubdate: str = ""
    duration: str = ""
    page: int = 0
    view: int = 0
    favorite: int = 0
    coin: int = 0
    like: int = 0
    danmaku: int = 0
    reply: int = 0
    share: int = 0
    image_url: str = ""
    intro: str = ""
    streak: int = 0


@dataclass
class VideoStats:
    """视频统计数据（用于计算）"""

    view: int = 0
    favorite: int = 0
    coin: int = 0
    like: int = 0
    danmaku: int = 0
    reply: int = 0
    share: int = 0
    copyright: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "VideoStats":
        return cls(
            view=int(d.get("view", 0)),
            favorite=int(d.get("favorite", 0)),
            coin=int(d.get("coin", 0)),
            like=int(d.get("like", 0)),
            danmaku=int(d.get("danmaku", 0)),
            reply=int(d.get("reply", 0)),
            share=int(d.get("share", 0)),
            copyright=int(d.get("copyright", 1)),
        )


@dataclass
class ScoreResult:
    """评分计算结果"""

    # 统计增量
    view: int = 0
    favorite: int = 0
    coin: int = 0
    like: int = 0
    danmaku: int = 0
    reply: int = 0
    share: int = 0
    # 评分系数
    view_rate: float = 0.0
    favorite_rate: float = 0.0
    coin_rate: float = 0.0
    like_rate: float = 0.0
    danmaku_rate: float = 0.0
    reply_rate: float = 0.0
    share_rate: float = 0.0
    # 修正系数
    fix_a: float = 0.0
    fix_b: float = 0.0
    fix_c: float = 0.0
    fix_d: float = 0.0
    # 总分
    point: int = 0


@dataclass
class SearchOptions:
    """搜索参数"""

    video_zone_type: Optional[int] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    page_size: int = 50
    newlist_rids: List[int] = field(default_factory=list)


@dataclass
class SearchRestrictions:
    """搜索过滤条件"""

    min_favorite: Optional[int] = None
    min_view: Optional[int] = None


@dataclass
class ScraperConfig:
    """爬虫配置"""

    KEYWORDS: List[str] = field(default_factory=list)
    HEADERS: List[str] = field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        ]
    )
    MAX_RETRIES: int = 5
    SEMAPHORE_LIMIT: int = 5
    MIN_VIDEO_DURATION: int = 20
    SLEEP_TIME: float = 0.2
    OUTPUT_DIR: Optional[Path] = None
    COLLECTED_FILE: Optional[Path] = None
    NAME: Optional[str] = None
    STREAK_THRESHOLD: int = 7
    MIN_TOTAL_VIEW: int = 10000
    BASE_THRESHOLD: int = 100
    HOT_RANK_CATE_ID: int = 30
    LOCAL_METADATA_FIELDS: List[str] = field(
        default_factory=lambda: [
            "bvid",
            "name",
            "author",
            "copyright",
            "synthesizer",
            "vocal",
            "type",
        ]
    )
    UPDATE_COLS: List[str] = field(
        default_factory=lambda: [
            "bvid",
            "aid",
            "title",
            "view",
            "uploader",
            "copyright",
            "image_url",
        ]
    )


@dataclass
class VideoInvalidException(Exception):
    """视频失效异常"""

    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
