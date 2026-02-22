# common/config.py
"""统一配置管理"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from functools import cached_property


# ==================== 配置加载器 ====================


class ConfigLoader:
    """配置加载器（单例）"""

    _instance: Optional["ConfigLoader"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, path: str = "config/app.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(path)
        return cls._instance

    def _load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def get(self, *keys, default=None):
        """获取嵌套配置"""
        result = self._config
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
            else:
                return default
            if result is None:
                return default
        return result

    @property
    def raw(self) -> Dict[str, Any]:
        return self._config

    @classmethod
    def reload(cls, path: str = "config/app.yaml"):
        cls._instance = None
        return cls(path)


def get_config() -> ConfigLoader:
    return ConfigLoader()


# ==================== 路径配置 ====================


@dataclass
class Paths:
    """路径配置"""

    collected: Path = field(default_factory=lambda: Path("data/collected.xlsx"))
    snapshot_main: Path = field(default_factory=lambda: Path("data/snapshot/main"))
    snapshot_new: Path = field(default_factory=lambda: Path("data/snapshot/new"))
    daily_diff_old: Path = field(default_factory=lambda: Path("data/daily/diff/old"))
    daily_diff_new: Path = field(default_factory=lambda: Path("data/daily/diff/new"))
    daily_ranking_main: Path = field(
        default_factory=lambda: Path("data/daily/ranking/main")
    )
    daily_ranking_new: Path = field(
        default_factory=lambda: Path("data/daily/ranking/new")
    )
    weekly_main: Path = field(default_factory=lambda: Path("data/weekly/main"))
    weekly_new: Path = field(default_factory=lambda: Path("data/weekly/new"))
    monthly_main: Path = field(default_factory=lambda: Path("data/monthly/main"))
    monthly_new: Path = field(default_factory=lambda: Path("data/monthly/new"))
    annual: Path = field(default_factory=lambda: Path("data/annual"))
    special_data: Path = field(default_factory=lambda: Path("data/special/data"))
    special_ranking: Path = field(default_factory=lambda: Path("data/special/ranking"))
    achievement: Path = field(default_factory=lambda: Path("data/features/achievement"))
    milestone: Path = field(default_factory=lambda: Path("data/features/milestone"))
    history: Path = field(default_factory=lambda: Path("data/features/history"))
    downloads_video: Path = field(default_factory=lambda: Path("downloads/videos"))
    downloads_image: Path = field(default_factory=lambda: Path("downloads/image_url"))
    export: Path = field(default_factory=lambda: Path("export"))
    keywords: Path = field(default_factory=lambda: Path("config/keywords.json"))
    usecols: Path = field(default_factory=lambda: Path("config/usecols.json"))
    ai_config: Path = field(default_factory=lambda: Path("config/ai.yaml"))
    prompt_template: Path = field(
        default_factory=lambda: Path("config/prompt_template.txt")
    )
    exclude_singers: Path = field(
        default_factory=lambda: Path("config/exclude_singers.yaml")
    )
    special_config: Path = field(default_factory=lambda: Path("config/special.yaml"))

    @classmethod
    def load(cls) -> "Paths":
        cfg = get_config()
        paths = cfg.get("paths", default={})
        configs = cfg.get("configs", default={})

        def get_path(*keys, default: str) -> Path:
            result = paths
            for key in keys:
                if isinstance(result, dict):
                    result = result.get(key)
                else:
                    return Path(default)
            return Path(result) if result else Path(default)

        return cls(
            collected=Path(paths.get("collected", "data/collected.xlsx")),
            snapshot_main=get_path("snapshot", "main", default="data/snapshot/main"),
            snapshot_new=get_path("snapshot", "new", default="data/snapshot/new"),
            daily_diff_old=get_path("daily", "diff_old", default="data/daily/diff/old"),
            daily_diff_new=get_path("daily", "diff_new", default="data/daily/diff/new"),
            daily_ranking_main=get_path(
                "daily", "ranking_main", default="data/daily/ranking/main"
            ),
            daily_ranking_new=get_path(
                "daily", "ranking_new", default="data/daily/ranking/new"
            ),
            weekly_main=get_path("weekly", "main", default="data/weekly/main"),
            weekly_new=get_path("weekly", "new", default="data/weekly/new"),
            monthly_main=get_path("monthly", "main", default="data/monthly/main"),
            monthly_new=get_path("monthly", "new", default="data/monthly/new"),
            annual=Path(paths.get("annual", "data/annual")),
            special_data=get_path("special", "data", default="data/special/data"),
            special_ranking=get_path(
                "special", "ranking", default="data/special/ranking"
            ),
            achievement=get_path(
                "features", "achievement", default="data/features/achievement"
            ),
            milestone=get_path(
                "features", "milestone", default="data/features/milestone"
            ),
            history=get_path("features", "history", default="data/features/history"),
            downloads_video=get_path("downloads", "video", default="downloads/videos"),
            downloads_image=get_path(
                "downloads", "image", default="downloads/image_url"
            ),
            export=Path(cfg.get("dirs", "export", default="export")),
            keywords=Path(configs.get("keywords", "config/keywords.json")),
            usecols=Path(configs.get("usecols", "config/usecols.json")),
            ai_config=Path(configs.get("ai", "config/ai.yaml")),
            prompt_template=Path(
                configs.get("prompt_template", "config/prompt_template.txt")
            ),
            exclude_singers=Path(
                configs.get("exclude_singers", "config/exclude_singers.yaml")
            ),
            special_config=Path(configs.get("special", "config/special.yaml")),
        )

    def ensure_dirs(self):
        for name in self.__dataclass_fields__:
            path = getattr(self, name)
            if isinstance(path, Path) and not path.suffix:
                path.mkdir(parents=True, exist_ok=True)

    def load_keywords(self) -> List[str]:
        return json.loads(self.keywords.read_text(encoding="utf-8"))

    def load_usecols(self, key: str) -> List[str]:
        data = json.loads(self.usecols.read_text(encoding="utf-8"))
        return data.get("columns", {}).get(key, [])


@dataclass
class Templates:
    """文件名模板"""

    snapshot_main: str = "{date}.xlsx"
    snapshot_new: str = "新曲{date}.xlsx"
    diff: str = "{new_date}与{old_date}.xlsx"
    diff_new: str = "新曲{new_date}与新曲{old_date}.xlsx"
    ranking_new: str = "新曲榜{new_date}与{old_date}.xlsx"
    combined: str = "{new_date}与{old_date}.xlsx"

    @classmethod
    def load(cls) -> "Templates":
        cfg = get_config()
        tpl = cfg.get("templates", default={})
        return cls(
            snapshot_main=tpl.get("snapshot_main", "{date}.xlsx"),
            snapshot_new=tpl.get("snapshot_new", "新曲{date}.xlsx"),
            diff=tpl.get("diff", "{new_date}与{old_date}.xlsx"),
            diff_new=tpl.get("diff_new", "新曲{new_date}与新曲{old_date}.xlsx"),
            ranking_new=tpl.get("ranking_new", "新曲榜{new_date}与{old_date}.xlsx"),
            combined=tpl.get("combined", "{new_date}与{old_date}.xlsx"),
        )


# ==================== 服务配置 ====================


@dataclass
class DailyConfig:
    threshold: int = 2000

    @classmethod
    def load(cls) -> "DailyConfig":
        cfg = get_config()
        daily = cfg.get("daily", default={})
        return cls(threshold=daily.get("threshold", 2000))


@dataclass
class WeeklyConfig:
    start_date: str = "2024-09-07"
    start_index: int = 1
    new_song_days: int = 7

    @classmethod
    def load(cls) -> "WeeklyConfig":
        cfg = get_config()
        weekly = cfg.get("weekly", default={})
        return cls(
            start_date=weekly.get("start_date", "2024-09-07"),
            start_index=weekly.get("start_index", 1),
            new_song_days=weekly.get("new_song_days", 7),
        )


@dataclass
class MonthlyConfig:
    start_year: int = 2024
    start_month: int = 9
    start_index: int = 1

    @classmethod
    def load(cls) -> "MonthlyConfig":
        cfg = get_config()
        monthly = cfg.get("monthly", default={})
        return cls(
            start_year=monthly.get("start_year", 2024),
            start_month=monthly.get("start_month", 9),
            start_index=monthly.get("start_index", 1),
        )


@dataclass
class AchievementConfig:
    start_file: str = "2024-09-07.xlsx"
    start_index: int = 1

    @classmethod
    def load(cls) -> "AchievementConfig":
        cfg = get_config()
        achi = cfg.get("achievement", default={})
        return cls(
            start_file=achi.get("start_file", "2024-09-07.xlsx"),
            start_index=achi.get("start_index", 1),
        )


@dataclass
class DuplicateConfig:
    hash_threshold: int = 15
    date_threshold_days: int = 3

    @classmethod
    def load(cls) -> "DuplicateConfig":
        cfg = get_config()
        dup = cfg.get("duplicate", default={})
        return cls(
            hash_threshold=dup.get("hash_threshold", 15),
            date_threshold_days=dup.get("date_threshold_days", 3),
        )


# ==================== 统一应用配置 ====================


class AppConfig:
    """应用程序统一配置"""

    def __init__(self):
        self._loader = get_config()

    @cached_property
    def paths(self) -> Paths:
        return Paths.load()

    @cached_property
    def templates(self) -> Templates:
        return Templates.load()

    @cached_property
    def daily(self) -> DailyConfig:
        return DailyConfig.load()

    @cached_property
    def weekly(self) -> WeeklyConfig:
        return WeeklyConfig.load()

    @cached_property
    def monthly(self) -> MonthlyConfig:
        return MonthlyConfig.load()

    @cached_property
    def achievement(self) -> AchievementConfig:
        return AchievementConfig.load()

    @cached_property
    def duplicate(self) -> DuplicateConfig:
        return DuplicateConfig.load()

    def get(self, *keys, default=None):
        return self._loader.get(*keys, default=default)

    def get_file_path(self, base_path: Path, template: str, **kwargs) -> Path:
        filename = template.format(**kwargs)
        return base_path / filename

    def get_period_path(
        self, period: str, key: str, path_type: str = "input_paths", **kwargs
    ) -> Path:
        """获取周期配置中的路径"""
        period_cfg = self._loader.get(period, default={})
        paths_dict = period_cfg.get(path_type, {})
        template = paths_dict.get(key, "")
        if not template:
            raise KeyError(f"路径 '{key}' 未在 {period}.{path_type} 中定义")
        return Path(template.format(**kwargs))

    def get_period_option(self, period: str, key: str, default=None):
        """获取周期配置中的选项"""
        period_cfg = self._loader.get(period, default={})
        return period_cfg.get(key, default)


# ==================== 全局实例 ====================

_app_config: Optional[AppConfig] = None


def get_app_config() -> AppConfig:
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
    return _app_config


def get_paths() -> Paths:
    return get_app_config().paths


def get_templates() -> Templates:
    return get_app_config().templates


# ==================== 列配置 ====================


class ColumnConfig:
    def __init__(self):
        self._paths = get_paths()
        self._data = json.loads(self._paths.usecols.read_text(encoding="utf-8"))

    def get_columns(self, key: str) -> List[str]:
        return self._data.get("columns", {}).get(key, [])
